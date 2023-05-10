"""
    Plug and Play ADMM for Compressive Holography
    With visualization
    The model structure is Unet with 1 channel
    the stop criterian is changed for taking PSNR as consideration
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
from utils.functions import psnr, generate_otf_torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import mse_loss
import time
from utils.helper import Early_stop
from utils.layer import psnr_metric, l2_loss, norm_tensor


model_name = 'Diffuser/'
timestr = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime())
out_dir = 'output/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
out_dir = out_dir+model_name
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
out_dir = out_dir+timestr
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
writer = SummaryWriter(out_dir)


class diffuser_ADMM_DH():
    def __init__(self, wavelength, nx, ny, deltax, deltay, distance, diffuser,diffusion_args, n_c=1, device=None, visual_check=False):
        self.A = generate_otf_torch(wavelength, nx, ny, deltax, deltay, distance)
        self.AT = generate_otf_torch(wavelength, nx, ny, deltax, deltay, -distance)
        self.diffusion_args = diffusion_args
        self.diffuser = diffuser
        self.n_c = n_c
        self.visual_check = visual_check
        if device:
            self.device = device
            self.A = self.A.to(device)
            self.AT = self.AT.to(device)
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
        temp = torch.fft.ifft2(temp).abs()
        temp = norm_tensor(temp)
        temp = temp + rho * o_tilde
        n = torch.fft.fft2(temp.to(torch.complex64))

        # denominator
        ATA = torch.multiply(self.AT,self.A)
        ones_array = torch.ones_like(ATA)
        # |A|^2 OTF的intensity/magnitude 为 unit 1
        d = ones_array * rho + ATA
        # d = ones_array * rho + AT_square
        small_value = 1e-8
        small_value =  torch.full(d.size(),small_value)
        small_value = small_value.to(torch.complex64).to(self.device)
        d = torch.where(d == 0, small_value, d)
        d = d.to(torch.complex64)
        o_next = torch.fft.ifft2(n / d).abs()
        # o_next = (o_next-torch.min(o_next))/(torch.max(o_next)-torch.min(o_next))
        return o_next, mse_loss(o_old, o_next)

    def denoise_step(self, o, u, v_old):
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
        v_tilde = v_tilde.unsqueeze(0).unsqueeze(0)
        v_tilde = v_tilde.repeat([1,3,1,1]).to(self.device)

        v_next = self.diffuser.denoise(v_tilde,self.diffusion_args)[-1]

        v_next = torch.mean(v_next,1, keepdim=True).to(self.device)
        v_next = v_next[0,0,:,:]

        #back to the previous distribution range
        v_next = v_min+v_next*(v_max-v_min)

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
        mtr_old = np.inf  # set a sufficient large value
        # early_stop_checker = Early_stop(patience=opts['patient'],tol=opts['psnr_tol'])
        """Initialization using Weiner Deconvolution method"""
        o = torch.fft.ifft2(torch.multiply(torch.fft.fft2(y), self.AT)).abs().to(self.device)

        o = norm_tensor(o)
        init = o.cpu()
        v = torch.zeros_like(y).to(self.device)
        u = torch.zeros_like(y).to(self.device)
        v, e2 = self.denoise_step(o, u, v)
        # u += self.mu*(o - v)
        u += (o - v)

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

        for i in pbar:
            if self.visual_check and i % self.visual_check == 0:
                fig, ax = plt.subplots(2, 3)
                ax[0,0].imshow(y.cpu().numpy(), cmap='gray')
                ax[0,1].imshow(gt.cpu().numpy(), cmap='gray')
                ax[0,2].imshow(init.numpy(), cmap='gray')
                ax[0,2].set_title(('BP \n PSNR{:.2f}').format(psnr(init, gt.cpu()).numpy()))
                ax[1,0].imshow(v.cpu().numpy(), cmap='gray')
                ax[1,0].set_title(('v_{} \n PSNR{:.2f}').format(i,psnr(v.cpu(), gt.cpu()).numpy()))
                ax[1,1].imshow(o.cpu().numpy(), cmap='gray')
                ax[1,1].set_title(('o_{} \n PSNR{:.2f}').format(i,psnr(o.cpu(), gt.cpu()).numpy()))
                ax[1,2].imshow(u.cpu().numpy(), cmap='gray')
                ax[1,2].set_title('u_{}'.format(i))
                fig.show()
            v, e2 = self.denoise_step(o, u, v)
            o, e1 = self.inverse_step(v, u, y, self.rho, o)
            u += (o - v)

            e3 = torch.sqrt(torch.sum(torch.square(o - v))) / u.numel()
            mtr_current = e1 + e2 + e3
            o_psnr = psnr(o,gt)
            v_psnr = psnr(v,gt)


            if mtr_current <= self.tol:
                print('Loop ended with matric value {:.4f}'.format(mtr_current))
                break
            else:
                if mtr_current >= self.eta * mtr_old:
                    self.rho_update()
            mtr_old = mtr_current

            """ Monitoring. """
            if verbose:
                info = ("{}, \t {} \t {}").format(i + 1, o_psnr, v_psnr)
                pbar.set_description(info)
                writer.add_scalar('o_update', e1,i)
                writer.add_scalar('v_update', e2,i)
                writer.add_scalar('u_update', e3,i)
                writer.add_scalar('delta',mtr_current,i)
                writer.add_scalar('rho', self.rho,i)
                writer.add_scalar('gamma', self.gamma,i)
                writer.add_scalar('eta', self.eta,i)
                writer.add_scalar('v_PSNR',psnr(v, gt),i)
                writer.add_scalar('o_PSNR',psnr(o, gt),i)
        return v,o,u
