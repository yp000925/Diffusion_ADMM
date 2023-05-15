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
from torchvision.transforms import Resize
import cv2
import glob
import scipy.io as sio
import scipy.misc
from utils.load_model import load_model
from utils.functions import psnr, generate_otf_torch, rgb_to_gray, gray_to_rgb, zero_padding_torch, crop_img_torch

from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import mse_loss
import time
from utils.helper import Early_stop
from utils.layer import psnr_metric, l2_loss, norm_tensor

model_name = 'Diffuser_GB/'
timestr = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
out_dir = 'output/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
out_dir = out_dir + model_name
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
out_dir = out_dir + timestr
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
writer = SummaryWriter(out_dir)


class diffuser_GB_DH():
    def __init__(self, wavelength, nx, ny, deltax, deltay, distance, diffuser, diffusion_args, n_c=1, device=None,
                 visual_check=False, pad_size=None):
        self.A = generate_otf_torch(wavelength, nx, ny, deltax, deltay, distance,pad_size = pad_size)
        self.AT = generate_otf_torch(wavelength, nx, ny, deltax, deltay, -distance,pad_size = pad_size)
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
        self.pad_size = pad_size
        self.crop_size = [nx, ny]
        self.nettrs = Resize(size=256)
        self.backnettrs = Resize(size=self.crop_size[0])

    def forward_prop(self, f_in):
        fs_out = torch.multiply(torch.fft.fft2(f_in), self.A)
        f_out = torch.fft.ifft2(fs_out)
        return f_out

    def backward_prop(self, f_in):
        fs_out = torch.multiply(torch.fft.fft2(f_in), self.AT)
        f_out = torch.fft.ifft2(fs_out)
        return f_out

    def forward_op(self, x_in, crop_size):
        x_out = self.forward_prop(x_in)
        x_out = crop_img_torch(x_out, crop_size=crop_size)
        return x_out

    def backward_op(self, x_in, pad_size):
        x_out = zero_padding_torch(x_in, pad_size=pad_size)
        x_out = self.backward_prop(x_out)
        return x_out

    def cal_gradient(self, x, y):
        '''

        :param x: the complex-valued transmittance of the sample
        :param y: Intensity image (absolute value)
        :return: Wirtinger gradient
        '''
        temp = self.forward_op(x, crop_size=y.shape)
        # temp = (torch.abs(temp) - y)
        temp = (torch.abs(temp) - y) * torch.exp(1j * torch.angle(temp))
        gradient = 0.5 * self.backward_op(temp, pad_size=self.pad_size)
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

    def v_update(self, o, v_old):
        '''
        denoise step using pretrained denoiser
        :param o:
        :param u:
        :param rho:
        :return:
        '''
        o = crop_img_torch(o, crop_size=self.crop_size)
        # normalize to [0,1]
        v_min = torch.min(o)
        v_max = torch.max(o)
        o = norm_tensor(o)

        o = o.unsqueeze(0).unsqueeze(0)
        o = o.repeat([1, 3, 1, 1]).to(self.device)
        o = self.nettrs(o)
        v_next = self.diffuser.denoise(o, self.diffusion_args)[-1]

        v_next = torch.mean(v_next, 1, keepdim=True).to(self.device)
        v_next = self.backnettrs(v_next)
        v_next = v_next[0, 0, :, :]


        # back to the previous distribution range
        v_next = v_min + (v_max - v_min) * norm_tensor(v_next)

        v_next = zero_padding_torch(v_next, pad_size=self.pad_size)
        return v_next, mse_loss(v_old, v_next)

    def GB_pnp_dh(self, y, opts, save = False):
        maxitr = opts['maxitr']
        verbose = opts['verbose']
        # rho = opts['rho'].to(self.device)
        gt = opts['gt'].to(self.device)
        y = y.to(self.device)
        self.rho = opts['rho'].to(self.device)
        self.tol = opts['tol']
        self.eta = opts['eta']
        mtr_old = np.inf  # set a sufficient large value
        # early_stop_checker = Early_stop(patience=opts['patient'],tol=opts['psnr_tol'])
        """Initialization using Weiner Deconvolution method"""
        o = self.backward_op(y, pad_size=self.pad_size).to(self.device)
        init = o.abs()
        v = torch.zeros_like(init).to(self.device)
        u = o.clone()

        """Start GB-PnP"""
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
            v_next, e2 = self.v_update(o_next, v)
            u_next = v_next + (i + 1) / (i + 4) * (v_next - v)

            v = v_next
            o = o_next
            u = u_next
            mtr_current = e1 + e2
            bp_psnr = psnr(crop_img_torch(init,self.crop_size), crop_img_torch(gt,self.crop_size)).cpu().numpy()
            o_psnr = psnr(crop_img_torch(o,self.crop_size), crop_img_torch(gt,self.crop_size)).cpu().numpy()
            v_psnr = psnr(crop_img_torch(v,self.crop_size), crop_img_torch(gt,self.crop_size)).cpu().numpy()

            if self.visual_check and (i + 1) % self.visual_check == 0:
                fig, ax = plt.subplots(2, 3)
                ax[0, 0].imshow(y.cpu().numpy(), cmap='gray')
                ax[0, 1].imshow(crop_img_torch(gt,self.crop_size).cpu().numpy(), cmap='gray')
                ax[0, 1].set_title('GT')
                ax[0, 2].imshow(crop_img_torch(init,self.crop_size).cpu().numpy(), cmap='gray')
                ax[0, 2].set_title(('BP \n PSNR{:.2f}').format(i,bp_psnr))
                ax[1, 0].imshow(crop_img_torch(o,self.crop_size).cpu().numpy(), cmap='gray')
                ax[1, 0].set_title(('o_{} \n PSNR{:.2f}').format(i,o_psnr))
                ax[1, 1].imshow(crop_img_torch(v,self.crop_size).cpu().numpy(), cmap='gray')
                ax[1, 1].set_title(('v_{} \n PSNR{:.2f}').format(i,v_psnr))
                ax[1, 2].imshow(crop_img_torch(u,self.crop_size).cpu().numpy(), cmap='gray')
                ax[1, 2].set_title('u_{}'.format(i))
                if save:
                    fig.savefig(out_dir+('/output_{}').format(i), cmap='gray')
                fig.show()

            """ Monitoring. """
            if verbose and i != 0:
                # o_psnr = psnr(o, gt)
                # v_psnr = psnr(v, gt)
                info = ("{}, \t {:.2f} \t {:.2f}\t {:4f} \t {:4f}").format(i + 1, o_psnr, v_psnr, e1, e2)
                pbar.set_description(info)
                writer.add_scalar('update/o_update', e1, i)
                writer.add_scalar('update/v_update', e2, i)
                writer.add_scalar('update/delta', mtr_current, i)
                writer.add_scalar('params/rho', self.rho, i)
                # writer.add_scalar('params/gamma', self.gamma, i)
                writer.add_scalar('params/eta', self.eta, i)
                writer.add_scalar('metric/v_PSNR', v_psnr, i)
                writer.add_scalar('metric/o_PSNR', o_psnr, i)
        return v, o, u
