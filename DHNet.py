import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from pnp_admm_Unet1c_V2 import pnp_ADMM_DH
from utils import *
import PIL.Image as Image
import torch
import torch.nn.functional as F
from models.Unet import Unet
import numpy as np
from utils.functions import generate_otf_torch, psnr, norm_tensor, autopad
from torch.fft import fft2, ifft2, fftshift, ifftshift
import time
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import Adam
import torch.optim as optim

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


class SoftThreshold(nn.Module):
    def __init__(self):
        super(SoftThreshold, self).__init__()

        self.soft_thr = nn.Parameter(torch.tensor([0.01]), requires_grad=True)

    def forward(self, x):
        return torch.mul(torch.sign(x), torch.nn.functional.relu(torch.abs(x) - self.soft_thr))


class CBL(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, padding=None, g=1, activation=True):
        super(CBL, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, padding), bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU() if activation is True else (
            activation if isinstance(activation, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class resblock(nn.Module):
    def __init__(self, c):
        super(resblock, self).__init__()
        self.CBL1 = CBL(c, c, k=3, s=1, activation=True)
        self.CBL2 = CBL(c, c, k=3, s=1, activation=False)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        return self.act(x + self.CBL2(self.CBL1(x)))


def initialization(y, AT, device):
    o = torch.fft.ifft2(torch.multiply(torch.fft.fft2(y), AT)).abs().to(device)
    # o.requires_grad = False
    v = torch.zeros_like(y).to(device)
    u = torch.zeros_like(y).to(device)
    return o, v, u


class denoiser(nn.Module):
    def __init__(self, c):
        super(denoiser, self).__init__()
        self.resblock1 = resblock(c)
        self.resblock2 = resblock(c)
        self.soft_thr = SoftThreshold()

    def forward(self, xin):
        x = self.resblock1(xin)
        x_thr = self.soft_thr(x)
        x_out = self.resblock2(x_thr)
        x_forward_backward = self.resblock2(x)
        stage_symloss = x_forward_backward - xin
        return x_out, stage_symloss


class DHNet(nn.Module):
    def __init__(self, wavelength, nx, ny, deltax, deltay, distance, n_c=1, device=None):
        super(DHNet, self).__init__()
        self.A = generate_otf_torch(wavelength, nx, ny, deltax, deltay, distance)
        self.AT = generate_otf_torch(wavelength, nx, ny, deltax, deltay, -distance)
        self.denoiser = denoiser(n_c)

        if device:
            self.device = device
            self.A = self.A.to(device)
            self.AT = self.AT.to(device)
            self.denoiser = self.denoiser.to(device)
        else:
            self.device = 'cpu'
        # self.rho = torch.nn.Parameter(torch.tensor([1e-5], requires_grad=False))
        self.rho = torch.tensor([1e-3], requires_grad=False)

    def inverse_step(self, v, u, y):
        '''
        inverse step (proximal operator for imaging forward model) for o-update
            o^{k+1} = argmin ||Ao-y||^2+(rho/2)||o-o_tilde||^2
        :param v:
        :param u:
        :param y: observation
        :param rho:
        :param A: forward operation matrix
        :return: update for o
        '''
        o_tilde = v - u
        # o_tilde  = norm_tensor(o_tilde)

        # numerator
        temp = torch.multiply(torch.fft.fft2(y), self.AT)
        temp = torch.fft.ifft2(temp).abs()
        temp = norm_tensor(temp)
        temp = temp + self.rho * o_tilde
        n = torch.fft.fft2(temp.to(torch.complex64))

        # denominator
        ATA = torch.multiply(self.AT, self.A)
        ones_array = torch.ones_like(ATA)
        # |A|^2 OTF的intensity/magnitude 为 unit 1
        d = ones_array * self.rho + ATA
        # d = ones_array * rho + AT_square
        small_value = 1e-8
        small_value = torch.full(d.size(), small_value)
        small_value = small_value.to(torch.complex64).to(self.device)
        d = torch.where(d == 0, small_value, d)
        d = d.to(torch.complex64)
        o_next = torch.fft.ifft2(n / d).abs()
        # o_next = norm_tensor(o_next)
        return o_next

    def denoise_step(self, o, u):
        '''
        denoise step using pretrained denoiser
        :param o:
        :param u:
        :param rho:
        :return:
        '''
        v_tilde = o + u
        # normalize to [0,1]
        v_min = torch.min(v_tilde)
        v_max = torch.max(v_tilde)
        v_tilde = norm_tensor(v_tilde)
        # change channel to 3
        v_next, stage_loss = self.denoiser(v_tilde.to(torch.float32).to(self.device))
        v_next = v_min + (v_max - v_min) * norm_tensor(v_next)
        return v_next, stage_loss

    def forward(self, o, v, u, y):
        o = self.inverse_step(v, u, y)
        v, stage_loss = self.denoise_step(o, u)
        stage_loss = torch.sqrt(torch.sum(torch.pow(stage_loss, 2))) / stage_loss.numel()
        u = u + (o - v)
        return o, v, u, stage_loss


def forward_propagation(x, A):
    out = torch.fft.ifft2(torch.multiply(torch.fft.fft2(x), A))
    return out


if __name__ == "__main__":
    y = torch.randn([1, 1, 512, 512])
    # ---- define propagation kernel -----
    w = 632e-9
    deltax = 3.45e-6
    deltay = 3.45e-6
    distance = 0.02
    nx = 512
    ny = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = DHNet(w, nx, ny, deltax, deltay, distance, n_c=1, device=device)
    optimizer = Adam(net.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=3, factor=0.5, threshold=0.001,
                                                     verbose=True)
    o, v, u = initialization(y, net.AT, device)
    o, v, u, stage_loss = net(o, v, u, y)
    pred = forward_propagation(v, net.A)
    mse_loss = nn.MSELoss()(pred.abs(), y)
    total_loss = mse_loss + stage_loss
    total_loss.backward()

    # for name, param in net.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)
