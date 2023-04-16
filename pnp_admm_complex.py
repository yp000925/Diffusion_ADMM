"""
    Plug and Play ADMM with complex channel for Compressive Holography

"""
import logging
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import cv2
import glob
import scipy.io as sio
import scipy.misc
from utils.load_model import load_model
from utils.functions import psnr, generate_otf_torch, rgb_to_gray, gray_to_rgb
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import mse_loss
import time
from utils.helper import Early_stop
from utils.layer import psnr_metric, l2_loss, norm_tensor


timestr = time.strftime("Unet%Y-%m-%d-%H_%M_%S",time.localtime())
out_dir = 'output/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
out_dir = out_dir+timestr
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
writer = SummaryWriter(out_dir + '/runs/')




class pnp_ADMM_DH_C():
    def __init__(self, wavelength, nx, ny, deltax, deltay, distance, model, n_c=1, device=None, visual_check=False):
        self.A = generate_otf_torch(wavelength, nx, ny, deltax, deltay, distance)
        self.AT = generate_otf_torch(wavelength, nx, ny, deltax, deltay, -distance)
        self.denoiser = model
        self.n_c = n_c
        self.visual_check = visual_check
        if device:
            self.device = device
            self.A = self.A.to(device)
            self.AT = self.AT.to(device)
            self.denoiser = self.denoiser.to(device)
        else:
            self.device = 'cpu'
        self.rho = 1
        self.gamma = 1
        self.error = 0

    def inverse_step(self,v, I, rho):
        # numerator


        # denominator
        AT_square = torch.abs(self.AT) ** 2
        ones_array = torch.ones_like(AT_square)
        # |A|^2 OTF的intensity/magnitude 为 unit 1
        d = ones_array * rho + torch.ones_like(AT_square)
        # d = ones_array * rho + AT_square
        d = d.to(torch.complex64)



class InverseLayer(nn.Module):
    def __init__(self, A, AT, rho):
        super(InverseLayer,self).__init__()
        self.A =  A
        self.AT = AT
        self.rho = rho


