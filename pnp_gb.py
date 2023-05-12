"""
    Plug and Play Gradient based method for Holography reconstruction

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
from utils.functions import psnr, generate_otf_torch, rgb_to_gray, gray_to_rgb,zero_padding_torch,crop_img_torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import mse_loss
from utils.layer import psnr_metric, l2_loss, norm_tensor
import time

model_name = "Unet1c_GB/"
timestr = time.strftime("%Y-%m-%d-%H_%M_%S/", time.localtime())
out_dir = 'output/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
out_dir = out_dir + model_name
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
writer = SummaryWriter(out_dir + timestr)


class GB_PNP_DH():
    def __init__(self, wavelength, nx, ny, deltax, deltay, distance, denoiser, n_c=1, device=None, visual_check=False,
                 pad_size=[768,768]):
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
        # self.gamma = 1
        self.error = 0
        self.pad_size = pad_size
        # self.mu = 1

    # def forward_prop(self, f_in):
    #     fs_out = torch.multiply(torch.fft.fft2(f_in), self.A)
    #     f_out = torch.fft.ifft2(fs_out)
    #     return f_out
    #
    # def backward_prop(self, f_in):
    #     fs_out = torch.multiply(torch.fft.fft2(f_in), self.AT)
    #     f_out = torch.fft.ifft2(fs_out)
    #     return f_out

    def forward_prop(self, f_in, crop_size):
        fs_out = torch.multiply(torch.fft.fft2(f_in), self.A)
        f_out = torch.fft.ifft2(fs_out)
        f_out = crop_img_torch(f_out, crop_size=crop_size)
        return f_out

    def backward_prop(self, f_in, pad_size):
        f_in = zero_padding_torch(f_in,pad_size=pad_size)
        fs_out = torch.multiply(torch.fft.fft2(f_in), self.AT)
        f_out = torch.fft.ifft2(fs_out)
        return f_out

    def cal_gradient(self, x, y):
        '''

        :param x: the complex-valued transmittance of the sample
        :param y: Intensity image (absolute value)
        :return: Wirtinger gradient
        '''
        temp = self.forward_prop(x,crop_size=y.shape)
        # temp = (torch.abs(temp) - y)
        temp = (torch.abs(temp)-y)*torch.exp(1j*torch.angle(temp))
        gradient = 0.5 * self.backward_prop(temp,pad_size=self.pad_size)
        return gradient

    def o_update(self, u, y, o_old):
        '''
        gradient projection updation
        :param u:
        :param y:
        :param o_old:
        :return:
        '''
        df = self.cal_gradient(u, y)
        o_next = u - self.rho * df
        return o_next.abs(), mse_loss(o_old.abs(), o_next.abs())

    def v_update(self, o, denoiser, v_old):
        '''
        Denoiser update
        :param o:
        :param denoiser:
        :param v_old:
        :return:
        '''
        v_min = torch.min(o)
        v_max = torch.max(o)
        o = norm_tensor(o)
        v_next = denoiser(o.unsqueeze(0).unsqueeze(0)).to(self.device)
        v_next = v_next[0, 0, :, :]
        # v_next = norm_tensor(v_next)
        v_next = v_min + (v_max - v_min) * norm_tensor(v_next)
        return v_next, mse_loss(v_old, v_next)

    def GB_pnp_dh(self, y, opts):
        maxitr = opts['maxitr']
        verbose = opts['verbose']
        gt = opts['gt'].to(self.device)
        y = y.to(self.device)
        self.rho = opts['rho'].to(self.device)
        self.tol = opts['tol']
        self.eta = opts['eta']
        # self.beta = opts['beta']
        mtr_old = np.inf

        """Initialization using Weiner Deconvolution method"""
        o = self.backward_prop(y).to(self.device)
        init = o.abs().cpu()
        v = torch.zeros_like(y).to(self.device)
        u = o.clone()

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
            logger.info(('\n' + '%13s' * 5) % ('Iteration  ', 'o_PSNR', 'v_PSNR', 'o_residue', 'v_residue'))
        for i in pbar:
            o_next, e1 = self.o_update(u, y, o)
            v_next, e2 = self.v_update(o_next, self.denoiser, v)
            u_next = v_next + (i + 1) / (i + 4) * (v_next - v)
            # u_next = v_next
            mtr_current = e1 + e2
            # if mtr_current >= self.eta * mtr_old:
            #     self.rho_update()
            # mtr_old = mtr_current
            v = v_next
            o = o_next
            u = u_next
            if self.visual_check and (i + 1) % self.visual_check == 0:
                fig, ax = plt.subplots(2, 3)
                ax[0, 0].imshow(y.cpu().numpy(), cmap='gray')
                ax[0, 1].imshow(gt.cpu().numpy(), cmap='gray')
                ax[0, 2].imshow(init.numpy(), cmap='gray')
                ax[0, 2].set_title(('BP \n PSNR{:.2f}').format(psnr(init, gt.cpu()).numpy()))
                ax[1, 0].imshow(o.cpu().numpy(), cmap='gray')
                ax[1, 0].set_title(('o_{} \n PSNR{:.2f}').format(i, psnr(o.cpu(), gt.cpu()).numpy()))
                ax[1, 1].imshow(v.cpu().numpy(), cmap='gray')
                ax[1, 1].set_title(('v_{} \n PSNR{:.2f}').format(i, psnr(v.cpu(), gt.cpu()).numpy()))
                ax[1, 2].imshow(u.cpu().numpy(), cmap='gray')
                ax[1, 2].set_title('u_{}'.format(i))
                fig.show()
            """ Monitoring. """
            if verbose and i != 0:
                o_psnr = psnr(o, gt)
                v_psnr = psnr(v, gt)
                info = ("{}, \t {:.2f} \t {:.2f}\t {:4f} \t {:4f}").format(i + 1, o_psnr, v_psnr, e1, e2)
                pbar.set_description(info)
                writer.add_scalar('update/o_update', e1, i)
                writer.add_scalar('update/v_update', e2, i)
                writer.add_scalar('update/delta', mtr_current, i)
                writer.add_scalar('params/rho', self.rho, i)
                # writer.add_scalar('params/gamma', self.gamma, i)
                writer.add_scalar('params/eta', self.eta, i)
                writer.add_scalar('metric/v_PSNR', psnr(v, gt), i)
                writer.add_scalar('metric/o_PSNR', psnr(o, gt), i)

        return v, o, u
