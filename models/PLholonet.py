"""
settting:
- Python 3.8.13
- torch 1.11.0
- cuda 11.3
- cudnn 8.2

major difference:
the update sequence becomes
initialization ==> (z-update ==> phi-update ==> x-update) * 5 => output x
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.utilis import batch_FT2d, batch_iFT2d, autopad
from utils.dataset import create_dataloader_Poisson
from torch.fft import fft2, ifft2, fftshift, ifftshift
from utils.utilis import PCC, PSNR, accuracy, random_init, tensor2value, plotcube, plot_img, inner_check_vis
import os


class PLholonet_block(nn.Module):
    def __init__(self, d, alpha=10, verboseFlag=False):
        super(PLholonet_block, self).__init__()
        self.rho1 = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
        # torch.nn.init.normal_(self.rho1)
        self.rho2 = torch.nn.Parameter(torch.randn([1]), requires_grad=True)
        torch.nn.init.normal_(self.rho2)
        # self.lamda = torch.nn.Parameter(torch.randn([1]),requires_grad=True).to(device)
        # torch.nn.init.normal_(self.lamda)
        # self.denoiser = ResUNet().to(device)
        self.denoiser = denoiser(d)
        self.flag = verboseFlag
        self.alpha = alpha

    def batch_forward_proj(self, field_batch, otf3d_batch, intensity=True, scale=True):
        '''

        :param field_batch:  3d field for batch [B, C, H, W]
        :param otf3d_batch: OTF3d for batch [B, C, H, W] or [C, H, W]
        :return: holo_batch [B,1, H, W]
        '''
        [B, C, H, W] = field_batch.shape
        if len(otf3d_batch.shape) == 3:
            otf3d_batch = otf3d_batch.unsqueeze(0)
        assert otf3d_batch.shape[1] == field_batch.shape[1], "The depth slice does not match between field and OTF"
        Fholo3d = torch.mul(fft2(field_batch), otf3d_batch)
        Fholo = torch.sum(Fholo3d, dim=1, keepdim=True)
        holo = ifft2(Fholo)
        if intensity:
            holo = torch.abs(holo)
            if scale:
                # max_range = C
                # holo = holo/max_range
                # assert holo.max()[0]< 1.0,"The inner hologram is larger than 1"
                mintmp = holo.view([B, 1, H * W]).min(2, keepdim=True)[0].unsqueeze(-1)
                maxtmp = holo.view([B, 1, H * W]).max(2, keepdim=True)[0].unsqueeze(-1)
                holo = (holo - mintmp) / (maxtmp - mintmp)
                return holo
            else:
                return holo
        return holo

    def batch_back_proj(self, holo_batch, otf3d_batch, real_constraint=True):
        """

        :param holo_batch: holo_batch [B,1, H, W] or [B,H,W]
        :param otf3d_batch: OTF3d for batch [B, C, H, W] or [C, H, W]
        :return:
        """
        if len(holo_batch.shape) == 3:
            holo_batch = holo_batch.unsqueeze(1)  # [B,1,H,W]
        if len(otf3d_batch.shape) == 3:
            otf3d_batch = otf3d_batch.unsqueeze(0)
        holo_batch = holo_batch.to(torch.complex64)
        conj_otf3d = torch.conj(otf3d_batch)
        volumne_slice = otf3d_batch.shape[1]
        holo_expand = holo_batch.tile([1, volumne_slice, 1, 1])
        holo_expand = fft2(holo_expand)  # 不需要shift 因为otf3d在构造的时候已经考虑了
        field_ft = torch.multiply(holo_expand, conj_otf3d)
        field3d = ifft2(field_ft)  # 不需要shift
        if real_constraint:
            return torch.real(field3d)
        return field3d

    def X_update(self, phi, z, u1, u2, otf3d):
        "proximal operator for forward propagation "
        x1 = phi + u1
        x2 = z + u2
        # numerator n = F(ifft(rho1 * otf * fft(x1)) + rho2*x2)
        temp = self.batch_back_proj(x1, otf3d)
        n = self.rho1 * temp + self.rho2 * x2
        # n = batch_FT2d(n.to(torch.complex64))
        n = fft2(n.to(torch.complex64))
        # denominator d = (|OTF|^2 + 1)

        # denominator d = (rho1*|OTF|^2 + rho2)
        #  in fact |OTF|^2==1
        otf_square = torch.abs(otf3d) ** 2
        ones_array = torch.ones_like(otf_square)
        d = ones_array * self.rho2 + otf_square * self.rho1
        d = d.to(torch.complex64)

        # final fraction
        # x_next = batch_iFT2d(n / d)
        x_next = ifft2(n / d)

        return x_next.real

    def Phi_update(self, x, z, u1, u2, otf3d, y):
        """
        proximal operator for Poisson signal
        :param x: [B,D,H,W]
        :param z:
        :param u1: [B,1,H,W]
        :param u2: []
        :param otf3d:[B,D,H,W]
        :param K1: [B,1,H,W]
        :return: phi_next [B,1,H,W]
        """
        phi_tilde = self.batch_forward_proj(x, otf3d) - u1
        temp = self.rho1 * phi_tilde - self.alpha
        phi_next = 1 / (2 * self.rho1) * (temp + torch.sqrt((temp ** 2) + 4 * y * self.rho1))
        return phi_next

    def Z_update(self, x, phi, u1, u2, otf3d):
        [B, C, W, H] = x.shape
        z_tilde = x - u2
        # z_tilde = z_tilde.view([B*C,1,W,H])
        z_next, stage_symloss = self.denoiser(z_tilde)
        return z_next, stage_symloss

    def forward(self, x, phi, z, u1, u2, otf3d, y):
        z, stage_symloss = self.Z_update(x, phi, u1, u2, otf3d)
        phi = self.Phi_update(x, z, u1, u2, otf3d, y)
        x = self.X_update(phi, z, u1, u2, otf3d)
        # Lagrangian updates
        # batch_size = x.shape[0]
        # otf3d_tensor = otf3d.tile([batch_size,1,1,1])
        u1 = u1 + phi - self.batch_forward_proj(x, otf3d)
        u2 = u2 + z - x
        # print(stage_symloss.shape)
        return x, phi, z, u1, u2, stage_symloss


class PLholonet(nn.Module):
    def __init__(self, n, d, alpha, sysloss_param=2e-3, verboseFlag=False):
        super(PLholonet, self).__init__()
        self.n = n
        self.blocks = nn.ModuleList([])
        for i in range(n):
            self.blocks.append(PLholonet_block(d, alpha=alpha))
        self.Batchlayer = torch.nn.BatchNorm2d(d)
        self.Activation = torch.nn.Sigmoid()
        self.sysloss_param = sysloss_param
        self.flag = verboseFlag
        if self.flag:
            self.iter = 0
            self.dir = './vis_for_check/'
            if not os.path.isdir(self.dir):
                os.makedirs(self.dir)

    def forward(self, y, otf3d):
        """
        :param y: observation (hologram)
        :return:
        """
        # initialization
        device = y.device

        x = self.blocks[0].batch_back_proj(y, otf3d)

        phi = Variable(y.data.clone()).to(device)
        z = Variable(x.data.clone()).to(device)
        u1 = torch.zeros(y.size()).to(device)
        u2 = torch.zeros(x.size()).to(device)
        stage_symlosses = torch.tensor([0.0]).to(device)

        # building the blocks
        for i in range(self.n):
            x, phi, z, u1, u2, stage_symloss = self.blocks[i](x, phi, z, u1, u2, otf3d, y)  # x,phi,z,u1,u2,otf3d, y
            stage_symlosses += torch.sqrt(torch.sum(torch.pow(stage_symloss, 2))) / stage_symloss.numel()
            # stage_symlosses.append(torch.sqrt(torch.sum(torch.pow(stage_symloss,2)))/stage_symloss.numel())
            # print('stage',i,'\n symloss',torch.sqrt(torch.sum(torch.pow(stage_symloss,2)))/stage_symloss.numel())
        # _,_,final_z,_,_,_ = self.blocks[i](x,phi,z,u1,u2,otf3d,y) # take the z value (Denoised)
        # x = self.Batchlayer(final_z)
        x = self.Activation(x)
        # self.iter += 1
        return x, self.sysloss_param * stage_symlosses / self.n


class resblock(nn.Module):
    def __init__(self, c):
        super(resblock, self).__init__()
        self.CBL1 = CBL(c, c, k=3, s=1, activation=True)
        self.CBL2 = CBL(c, c, k=3, s=1, activation=False)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        return self.act(x + self.CBL2(self.CBL1(x)))


class CBL(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, padding=None, g=1, activation=True):
        super(CBL, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, padding), bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU() if activation is True else (
            activation if isinstance(activation, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SoftThreshold(nn.Module):
    def __init__(self):
        super(SoftThreshold, self).__init__()

        self.soft_thr = nn.Parameter(torch.tensor([0.01]), requires_grad=True)

    def forward(self, x):
        return torch.mul(torch.sign(x), torch.nn.functional.relu(torch.abs(x) - self.soft_thr))


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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # K1 = torch.randn([4,1,256,256])
    # otf3d = obj3d(wave_length = 633*nm, img_rows = 256, img_cols=256, slice=5,size = 10*mm, depth = 2*cm).get_otf3d()
    # net = PLholonet(n=2,d=5).to(device)
    # # x,phi,z,u1,u2 = net(K1,otf3d)
    # x, stage_symlosses = net(K1,otf3d)
    ALPHA = 30  # photon level
    path = "/Users/zhangyunping/PycharmProjects/PLParticleTracking/data/LLParticle/Nxy256_Nz7_ppv1.1e-04_dz6.9mm_pps13.8um_lambda660nm"
    dataloader, dataset = create_dataloader_Poisson(path, batch_size=2, alpha=ALPHA, is_training=True)
    model = PLholonet(n=5, d=7, alpha=ALPHA, verboseFlag=True)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        model = model.module.to("cuda")
        model.device = torch.device('cuda')
    else:
        model = torch.nn.DataParallel(model)
        model.device = torch.device('cpu')

    for batch_i, (y, label, otf3d) in enumerate(dataloader):
        y = y.to(device=model.device)
        otf3d = otf3d.to(torch.complex64).to(device=model.device)
        label = label.to(torch.float32).to(device=model.device)
        x, stage_symlosses = model(y, otf3d)
        break
