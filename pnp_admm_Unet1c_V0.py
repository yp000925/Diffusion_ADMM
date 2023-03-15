"""
    Plug and Play ADMM for Compressive Holography
    With visualization
    change to 1c from pnp_admm2.py

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
from utils.functions import psnr, generate_otf_torch, rgb_to_gray, gray_to_rgb
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import mse_loss

out_dir = 'output/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
writer = SummaryWriter(out_dir + '/runs/')


class pnp_ADMM_DH():
    def __init__(self, wavelength, nx, ny, deltax, deltay, distance, denoiser, n_c=1, device=None, visual_check=False):
        self.A = generate_otf_torch(wavelength, nx, ny, deltax, deltay, distance)
        self.AT = generate_otf_torch(wavelength, nx, ny, deltax, deltay, -distance)
        self.denoiser = denoiser
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

    def rho_update(self):
        self.rho = self.gamma * self.rho
        return

    def inverse_step(self, v, u, y, rho, o_old):
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
        # |A|^2 OTF的intensity/magnitude 为 unit 1
        d = ones_array * rho + torch.ones_like(AT_square)
        # d = ones_array * rho + AT_square
        d = d.to(torch.complex64)
        o_next = torch.fft.ifft2(n / d).abs()
        o_next = (o_next-torch.min(o_next))/(torch.max(o_next)-torch.min(o_next))
        return o_next, mse_loss(o_old, o_next)

    def denoise_step(self, o, u, denoiser, v_old):
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
        v_tilde = (v_tilde - v_min) / (v_max - v_min)
        # change channel to 3
        v_next = denoiser(v_tilde.unsqueeze(0).unsqueeze(0)).to(torch.float32).to(self.device)
        v_next = v_next[0,0,:,:]
        # v_tilde = gray_to_rgb(v_tilde).to(self.device)
        # v_next = denoiser(v_tilde.unsqueeze(0)).to('cpu').squeeze(0)
        # # change back to 1
        # v_next = rgb_to_gray(v_next).to(self.device)
        return v_next, mse_loss(v_old, v_next)

    def pnp_Admm_DH(self, y, opts):
        maxitr = opts['maxitr']
        verbose = opts['verbose']
        # rho = opts['rho'].to(self.device)
        gt = opts['gt'].to(self.device)
        y = y.to(self.device)
        self.rho = opts['rho'].to(self.device)
        self.gamma = opts['gamma']
        self.tol = opts['tol']
        self.eta = opts['eta']
        mtr_old = 10000  # set a sufficient large value

        """Initialization using Weiner Deconvolution method"""
        o = torch.fft.ifft2(torch.multiply(torch.fft.fft2(y), self.AT)).abs().to(self.device)
        o_min = torch.min(o)
        o_max = torch.max(o)
        o = (o - o_min) / (o_max - o_min)
        v = o.clone().to(self.device)
        u = torch.zeros_like(y).to(self.device)


        """Start ADMM-PnP"""
        pbar = tqdm(range(maxitr))
        if verbose:
            logger = logging.getLogger(__name__)
            logging.basicConfig(format="%(message)s", level=logging.INFO)
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
            logger.info(('\n' + '%10s' * 2) % ('Iteration  ', 'PSNR'))
        if self.visual_check:
            fig,ax = plt.subplots(1,3)
        for i in pbar:
            if self.visual_check:
                file_name = out_dir + ('v_{:d}.png').format(i)
                # plt.imsave(file_name, v.cpu().numpy(), cmap='gray')
                ax[0].imshow(v.cpu().numpy(),cmap = 'gray')
                ax[0].set_title(('v_{:d} update').format(i))
                file_name = out_dir + ('u_{:d}.png').format(i)
                # plt.imsave(file_name, u.cpu().numpy(), cmap='gray')
                ax[1].imshow(u.cpu().numpy(),cmap = 'gray')
                ax[1].set_title(('u_{:d} update').format(i))
                file_name = out_dir + ('o_{:d}.png').format(i)
                # plt.imsave(file_name, o.cpu().numpy(), cmap='gray')
                ax[2].imshow(o.cpu().numpy(),cmap = 'gray')
                ax[2].set_title(('o_{:d} update').format(i))
                fig.show()
            o, e1 = self.inverse_step(v, u, y, self.rho, o)
            v, e2 = self.denoise_step(o, u, self.denoiser, v)
            u += (o - v)
            e3 = torch.sqrt(torch.sum(torch.square(o - v))) / u.numel()
            mtr_current = e1 + e2 + e3
            if mtr_current <= self.tol:
                print('Loop ended with matric value {:.4f}'.format(mtr_current))
                break
            else:
                if mtr_current >= self.eta * mtr_old:
                    self.rho_update()
            mtr_old = mtr_current

            """ Monitoring. """
            if verbose:
                info = ("{}, \t {}").format(i + 1, psnr(v, gt))
                pbar.set_description(info)
                writer.add_scalar('o_update', e1,i)
                writer.add_scalar('v_update', e2,i)
                writer.add_scalar('u_update', e3,i)
                writer.add_scalar('delta',mtr_current,i)
                writer.add_scalar('rho', self.rho,i)
                writer.add_scalar('gamma', self.gamma,i)
                writer.add_scalar('eta', self.eta,i)
                writer.add_scalar('PSNR',psnr(v, gt),i)

        return v,o,u
