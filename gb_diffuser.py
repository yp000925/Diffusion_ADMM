"""
    Plug and Play ADMM for Compressive Holography
    With visualization
    The model structure is Unet with 1 channel
    the stop criterian is changed for taking PSNR as consideration
"""
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
from models.diffuser import diffusion_default
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import mse_loss
import time
from utils.diffuserHelper import Denoising,create_model,efficient_generalized_steps_with_physics
import argparse
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
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
def data_defaults():
    '''
    defaults for DH dataset
    '''
    res = dict(
        image_size=256,
        channels=3,
        logit_transform=False,
        uniform_dequantization=False,
        gaussian_dequantization=False,
        random_flip=False,
        num_workers=1,
        dh_dataset=True,
        rescaled = False
    )

    return dict2namespace(res)
def model_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        type="openai",
        image_size=256,
        num_channels=256,
        num_res_blocks=2,
        in_channels = 3,
        out_channels =3,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=64,
        attention_resolutions= "32,16,8",
        channel_mult="",
        dropout=0.0,
        resamp_with_conv = True,
        learn_sigma = True,
        class_cond=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        var_type='fixedsmall',
        use_fp16=False,
        use_new_attention_order=False,
        task = 'denoise'
    )
    return dict2namespace(res)

class diffuser_GB_DH():
    def __init__(self, wavelength, nx, ny, deltax, deltay, distance, model_pth, diffusion_args, device=None,
                 visual_check=False, pad_size=None):
        self.A = generate_otf_torch(wavelength, nx, ny, deltax, deltay, distance,pad_size = pad_size)
        self.AT = generate_otf_torch(wavelength, nx, ny, deltax, deltay, -distance,pad_size = pad_size)
        self.diffusion_args = diffusion_args
        if device:
            self.device = device
            self.A = self.A.to(device)
            self.AT = self.AT.to(device)
        else:
            self.device = 'cpu'
        # self.n_c = n_c
        self.visual_check = visual_check

        self.rho = 1
        self.gamma = 1
        self.error = 0
        self.pad_size = pad_size
        self.crop_size = [nx, ny]
        self.nettrs = Resize(size=256)
        self.backnettrs = Resize(size=self.crop_size[0])

        if model_pth == None:
            ckpt = "logs/imagenet/256x256_diffusion_uncond.pt"
        else:
            ckpt = model_pth

        self.model  = create_model(**vars(model_defaults()))
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.model = torch.nn.DataParallel(self.model)
        self.data_args = data_defaults()
        self.H_funcs = Denoising(self.data_args.channels, self.data_args.image_size, self.device)

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
        temp = (torch.abs(temp) - y)
        # temp = (torch.abs(temp) - y) * torch.exp(1j * torch.angle(temp))
        gradient =self.backward_op(temp, pad_size=self.pad_size)
        return gradient


    def reconstruction(self, holo, last=False):
        y_0 = self.backward_op(holo, pad_size=self.pad_size).abs().to(self.device)
        y_0 = self.H_funcs.H(y_0)
        y_0 = y_0 + self.diffusion_args.sigma_0 * torch.randn_like(y_0)
        x = torch.randn(
            y_0.shape[0],
            self.data_args.channels,
            self.data_args.image_size,
            self.data_args.image_size,
            device=self.device,
        )

        sigma_0 = self.diffusion_args.sigma_0
        betas = get_beta_schedule(beta_schedule=self.diffusion_args.beta_schedule,
                                  beta_start=self.diffusion_args.beta_start,
                                  beta_end=self.diffusion_args.beta_end,
                                  num_diffusion_timesteps=self.diffusion_args.num_diffusion_timesteps,
                                  )
        betas = torch.from_numpy(betas).float().to(self.device)
        timesteps = self.diffusion_args.timesteps

        skip = betas.shape[0] // timesteps
        seq = range(0, betas.shape[0], skip)
        with torch.no_grad():
            x = efficient_generalized_steps_with_physics(x, seq, self.model, betas, self.H_funcs, y_0, holo, sigma_0, \
                                        etaB=self.diffusion_args.etaB, etaA=self.diffusion_args.eta, etaC=self.diffusion_args.eta, cls_fn=None,
                                                     classes=None,gamma=self.gamma, gradient_cal=self.cal_gradient)
            if last:
                x = x[0][-1]
        return x


def get_beta_schedule(beta_schedule,beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                    )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

if __name__ == "__main__":
    import  PIL.Image as Image
    import torch
    import torch.nn.functional as F
    from models.Unet import Unet
    import numpy as np
    from utils.functions import generate_otf_torch, psnr,norm_tensor, crop_img_torch,zero_padding_torch,prepross_bg
    from torch.fft import fft2, ifft2, fftshift, ifftshift
    import time
    import matplotlib.pyplot as plt
    from models.diffuser import diffusion, diffusion_default

    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    """ Load pre-trained Unet pth"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    denoiser = diffusion(model_pth="256x256_diffusion_uncond.pt", model_type='default', device=device,
                         silence_diffuser=True)
    diffusion_args = diffusion_default()
    diffusion_args.sigma_0 = 0.2
    diffusion_args.timesteps = 50
    diffusion_args.num_diffusion_timesteps = 1000


    """ Load the GT intensity map and get the diffraction pattern"""
    # img = Image.open('test_image.png').resize([256, 256]).convert('L')
    # img = Image.open('test_image2.jpg').resize([512, 512]).convert('L')
    # img = Image.open('cameraman.bmp').resize([512, 512]).convert('L')
    # img = Image.open('USAF1951.jpg').resize([256, 256]).convert('L')
    # gt_intensity = torch.from_numpy(np.array(img))
    # gt_intensity = gt_intensity / torch.max(gt_intensity)
    #

    """ Load the GT intensity map and get the diffraction pattern"""

    # img = Image.open('ExpSample/DCOD/USAF/hologram.tif').resize([256, 256])
    # img = np.array(img)
    # img = img/img.max()
    img = Image.open('ExpSample/DCOD/USAF/hologram.tif')
    img = np.array(img)
    img = img/img.max()
    holo = torch.from_numpy(np.array(img)).to(torch.float32)
    holo = crop_img_torch(holo,crop_size=[360,360])

    # ---- define propagation kernel -----
    w = 532.3e-9
    # deltax = 2*1.12e-6
    # deltay = 2*1.12e-6
    deltax = 1.12e-6
    deltay = 1.12e-6
    distance = 1065e-6
    nx = 360
    ny = 360

    nx_extend = 512
    ny_extend = 512
    pad_size = [nx_extend,ny_extend]

    # holo = holo.unsqueeze(0).unsqueeze(0)
    # holo = holo.to(device)
    # ---- set solver -----
    # solver = GB_PNP_DH(w, nx, ny, deltax, deltay, distance, model, device=device, visual_check=50)
    solver = diffuser_GB_DH(w, nx, ny, deltax, deltay, distance,model_pth="256x256_diffusion_uncond.pt",diffusion_args=diffusion_args,
                            device=device, visual_check=10, pad_size=pad_size)
    # gt_intensity = zero_padding_torch(gt_intensity,pad_size)
    A = solver.A
    AT = solver.AT
    opts = dict(rho=torch.tensor([1.5]), maxitr=15, verbose=True, gt=torch.ones_like(holo), eta=0.9,
                tol=0.0000001,psnr_tol=0)

    # ---- forward and backward propagation -----
    # holo = solver.forward_op(gt_intensity.to(device),crop_size=[nx,ny])
    # holo = torch.abs(holo)


    rec = solver.backward_op(holo.to(device), pad_size=pad_size)
    rec = torch.abs(rec)
    rec = norm_tensor(rec)
    # plt.imshow(rec.cpu().numpy(),cmap='gray')
    # plt.show()


    # ---- reconstruction using ADMMPnP-----
    with torch.no_grad():
        out = solver.reconstruction(holo)
        fig, ax = plt.subplots(2, 3)