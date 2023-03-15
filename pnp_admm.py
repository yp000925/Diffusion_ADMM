"""
    Plug and Play ADMM for Compressive Holography

"""
import logging
import os
import numpy as np
from PIL import Image
import torch
import cv2
import glob
import scipy.io as sio
import scipy.misc
from utils.load_model import load_model
from utils.functions import psnr, generate_otf_torch,rgb_to_gray,gray_to_rgb
from tqdm import tqdm
import matplotlib.pyplot as plt

out_dir = 'output/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

class pnp_ADMM_DH():
    def __init__(self, wavelength, nx, ny, deltax, deltay, distance, denoiser, n_c = 1,device=None,visual_check=False):
        self.A = generate_otf_torch(wavelength, nx, ny, deltax, deltay, distance)
        self.AT = generate_otf_torch(wavelength, nx, ny, deltax, deltay, -distance)
        self.denoiser = denoiser
        self.n_c = n_c
        self.visual_check= visual_check
        if device:
            self.device = device
            self.A = self.A.to(device)
            self.AT = self.AT.to(device)
            self.denoiser = self.denoiser.to(device)
        else:
            self.device = 'cpu'


    def inverse_step(self, v, u, y, rho):
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

        # numerator
        temp = torch.multiply(torch.fft.fft2(y), self.AT)
        n = temp + rho * torch.fft.fft2(o_tilde)

        # denominator
        AT_square = torch.abs(self.AT) ** 2
        ones_array = torch.ones_like(AT_square)
        d = ones_array * rho + AT_square
        d = d.to(torch.complex64)
        o_next = torch.fft.ifft2(n / d)
        return o_next.abs()

    def denoise_step(self,o, u, denoiser):
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
        v_tilde = (v_tilde-v_min)/(v_max-v_min)
        # change channel to 3
        v_tilde = gray_to_rgb(v_tilde).to(self.device)
        v_next = denoiser(v_tilde.unsqueeze(0)).to('cpu').squeeze(0)
        # change back to 1
        v_next = rgb_to_gray(v_next).to(self.device)

        return v_next.squeeze(0)

    def pnp_Admm_DH(self, y, opts):
        maxitr = opts['maxitr']
        verbose = opts['verbose']
        rho = opts['rho'].to(self.device)
        gt = opts['gt'].to(self.device)
        y = y.to(self.device)

        """Initialization using Weiner Deconvolution method"""
        o = torch.fft.ifft2(torch.multiply(torch.fft.fft2(y), self.AT)).abs().to(self.device)
        v = torch.fft.ifft2(torch.multiply(torch.fft.fft2(y), self.AT)).abs().to(self.device)
        u = torch.zeros_like(y).to(self.device)
        o_min = torch.min(o)
        o_max = torch.max(o)
        o = (o-o_min)/(o_max-o_min)

        """Start ADMM-PnP"""
        pbar = tqdm(range(maxitr))
        if verbose:
            logger = logging.getLogger(__name__)
            logging.basicConfig(format="%(message)s",level=logging.INFO)
            formater = logging.Formatter("%(message)s")
            # define the filehaddler for writing log in file
            log_file = 'admm_dh.log'
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formater)
            # define the StreamHandler for writing on the screen
            sh = logging.StreamHandler()
            sh.setLevel(logging.ERROR)
            sh.setFormatter(formater)
            # add both handlers
            logger.addHandler(fh)
            logger.addHandler(sh)
            logger.info('\n Training========================================')
            logger.info(('\n'+'%10s'*2)%('Iteration  ','PSNR'))
        for i in pbar:
            if self.visual_check:
                file_name = out_dir+('v_{:d}.png').format(i)
                plt.imsave(file_name,v.cpu().numpy(),cmap='gray')
                file_name = out_dir+('u_{:d}.png').format(i)
                plt.imsave(file_name,u.cpu().numpy(),cmap='gray')
                file_name = out_dir+('o_{:d}.png').format(i)
                plt.imsave(file_name,o.cpu().numpy(),cmap='gray')
            o = self.inverse_step(v,u,y,rho)
            v = self.denoise_step(o,u,self.denoiser)
            u += (o - v)

            """ Monitoring. """
            if verbose:
                info = ("{}, \t {}").format(i+1, psnr(v,gt))
                pbar.set_description(info)
                # logger.info(info)

        return v

