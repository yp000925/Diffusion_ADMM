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

timestr = time.strftime("Unet%Y-%m-%d-%H_%M_%S", time.localtime())
out_dir = 'output/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
out_dir = out_dir + timestr
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
writer = SummaryWriter(out_dir + '/runs/')


class pnp_ADMM_DH_C():
    def __init__(self, wavelength, nx, ny, deltax, deltay, distance, model, n_c=1, device=None, visual_check=None):
        self.A = generate_otf_torch(wavelength, nx, ny, deltax, deltay, distance)
        self.AT = generate_otf_torch(wavelength, nx, ny, deltax, deltay, -distance)
        self.denoiser = model
        self.n_c = n_c

        if visual_check:
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
        # ATA = torch.multiply(self.AT,self.A)
        ATA = torch.abs(self.AT) ** 2
        ones_array = torch.ones_like(ATA).to(torch.complex64)
        # |A|^2 OTF的intensity/magnitude 为 unit 1
        # d = ones_array * rho + torch.ones_like(AT_square)
        d = ones_array * rho + ATA
        o_next = torch.fft.ifft2(n / d)
        return o_next, mse_loss(o_old.real, o_next.real)

    def denoise_step(self, o, u, denoiser, v_old):
        '''
        denoise step using pretrained denoiser
        :param o:
        :param u:
        :param rho:
        :return:
        '''
        v_tilde = o + u
        # real = norm_tensor(v_tilde.real)
        # imag = norm_tensor(v_tilde.imag)
        amp = norm_tensor(v_tilde.abs())
        angle = norm_tensor(v_tilde.angle())
        # denoising real and imag part separately
        amp_next = denoiser(amp.unsqueeze(0).unsqueeze(0)).to(torch.float32).to(self.device)
        angle_next = denoiser(angle.unsqueeze(0).unsqueeze(0)).to(torch.float32).to(self.device)

        v_next = torch.complex(amp_next, angle_next).to(self.device)
        v_next = torch.polar(amp_next, angle_next*torch.pi).to(self.device)
        v_next = v_next[0, 0, :, :]
        return v_next, mse_loss(v_old.real, v_next.real)

    def pnp_ADMM_complex(self, y, opts):
        maxitr = opts['maxitr']
        verbose = opts['verbose']
        # rho = opts['rho'].to(self.device)
        gt = opts['gt'].to(self.device)
        y = y.to(self.device)
        self.rho = opts['rho'].to(self.device)
        self.gamma = opts['gamma']
        self.tol = opts['tol']
        self.eta = opts['eta']
        mtr_old = np.inf  # set a sufficient large value

        """Initialization using Weiner Deconvolution method"""
        o = torch.fft.ifft2(torch.multiply(torch.fft.fft2(y), self.AT)).to(self.device)
        init = o.cpu()
        v = o.clone().to(self.device)
        u = torch.zeros_like(y).to(self.device).to(torch.complex64)

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
            logger.info(('\n' + '%10s' * 3) % ('Iteration  ', 'o_PSNR','v_PSNR'))

        if self.visual_check:
            fig, ax = plt.subplots(1, 3)

        for i in pbar:
            if self.visual_check and  i%self.visual_check == 0:
                fig, ax = plt.subplots(3, 3)
                ax[0, 0].imshow(y.cpu().numpy(), cmap='gray')
                ax[0, 1].imshow(gt.cpu().numpy(), cmap='gray')
                ax[0, 2].imshow(torch.abs(init).numpy(), cmap='gray')
                ax[0, 2].set_title(('BP \n PSNR{:.2f}').format(psnr( torch.abs(init), gt.cpu()).numpy()))

                ax[1, 0].imshow(v.real.cpu().numpy(), cmap='gray')
                ax[1, 0].set_title(('v_{} \n PSNR{:.2f}').format(i, psnr(v.real.cpu(), gt.cpu()).numpy()))
                ax[1, 1].imshow(o.real.cpu().numpy(), cmap='gray')
                ax[1, 1].set_title(('o_{} \n PSNR{:.2f}').format(i, psnr(o.real.cpu(), gt.cpu()).numpy()))
                ax[1, 2].imshow(u.real.cpu().numpy(), cmap='gray')
                ax[1, 2].set_title('u_{}'.format(i))

                ax[2, 0].imshow(v.imag.cpu().numpy(), cmap='gray')
                ax[2, 0].set_title(('v_imag{}').format(i))
                ax[2, 1].imshow(o.imag.cpu().numpy(), cmap='gray')
                ax[2, 1].set_title(('o_imag{}').format(i))
                ax[2, 2].imshow(u.imag.cpu().numpy(), cmap='gray')
                ax[2, 2].set_title('u_imag{}'.format(i))

                fig.show()

            o, e1 = self.inverse_step(v, u, y, self.rho, o)
            v, e2 = self.denoise_step(o, u, self.denoiser, v)
            u += (o - v)
            o_psnr = psnr(o.real.cpu(), gt.cpu()).numpy()
            v_psnr = psnr(v.real.cpu(), gt.cpu()).numpy()
            """ Monitoring. """
            if verbose:
                info = ("{}, \t {} \t {}").format(i + 1, o_psnr, v_psnr)
                pbar.set_description(info)

        return v, o, u
