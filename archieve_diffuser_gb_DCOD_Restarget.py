'''
use test image -> ImageNet data
'''
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from pnp_gb_diffuser import diffuser_GB_DH
from utils import *
import PIL.Image as Image
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
diffusion_args.num_diffusion_timesteps = 2000



""" Load the GT intensity map and get the diffraction pattern"""

img = Image.open('ExpSample/DCOD/USAF/hologram.tif').resize([256, 256])
img = np.array(img)
img = img/img.max()
holo = torch.from_numpy(np.array(img)).to(torch.float32)

# ---- define propagation kernel -----
w = 532.3e-9
deltax = 2*1.12e-6
deltay = 2*1.12e-6
distance = 1065e-6
nx = 256
ny = 256

nx_extend = 512
ny_extend = 512
pad_size = [nx_extend,ny_extend]

# holo = holo.unsqueeze(0).unsqueeze(0)
# holo = holo.to(device)
# ---- set solver -----
# solver = GB_PNP_DH(w, nx, ny, deltax, deltay, distance, model, device=device, visual_check=50)
solver = diffuser_GB_DH(w, nx, ny, deltax, deltay, distance, denoiser,diffusion_args, device=device, visual_check=1, pad_size=pad_size)
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
    out = solver.GB_pnp_dh(holo, opts,save =True)
    fig, ax = plt.subplots(2, 3)
    # ax[0,0].imshow(holo.cpu().numpy(), cmap='gray')
    # ax[0,1].imshow(gt_intensity.cpu().numpy(), cmap='gray')
    # ax[0,2].imshow(rec.cpu().numpy(), cmap='gray')
    # ax[0,2].set_title(('BP \n PSNR{:.2f}').format(psnr(rec.cpu(), gt_intensity.cpu()).numpy()))
    # ax[1,1].imshow(out[0].cpu().numpy(), cmap='gray')
    # ax[1,1].set_title(('v_out \n PSNR{:.2f}').format(psnr(out[0].cpu(), gt_intensity.cpu()).numpy()))
    # ax[1,0].imshow(out[1].cpu().numpy(), cmap='gray')
    # ax[1,0].set_title(('o_out \n PSNR{:.2f}').format(psnr(out[1].cpu(), gt_intensity.cpu()).numpy()))
    # ax[1,2].imshow(out[2].cpu().numpy(), cmap='gray')
    # ax[1,2].set_title('u_out')
    # fig.show()

# ---- find best rho-----
# rhos = np.linspace(1e-3, 1e-2, 100)
# idx = 0
# vis_dir = "vis_dir/"
# if not os.path.exists(vis_dir):
#     os.mkdir(vis_dir)
# with torch.no_grad():
#     for idx, rho in enumerate(rhos):
#         idx += 1
#         opts = dict(rho=torch.tensor(rho), maxitr=150, verbose=True, gt=torch.tensor(gt_intensity), eta=0.9,
#                     tol=0.0000001, gamma=1, psnr_tol=0, patient=8)
#         out = solver.pnp_Admm_DH(holo, opts)
#         fig, ax = plt.subplots(1, 4)
#         ax[0].imshow(holo.cpu().numpy(), cmap='gray')
#         ax[1].imshow(gt_intensity.cpu().numpy(), cmap='gray')
#         ax[2].imshow(rec.cpu().numpy(), cmap='gray')
#         ax[2].set_title(('Processed \n PSNR{:.2f}').format(psnr(rec.cpu(), gt_intensity.cpu()).numpy()))
#         ax[3].imshow(out.cpu().numpy(), cmap='gray')
#         ax[3].set_title(('Processed \n PSNR{:.2f}').format(psnr(out.cpu(), gt_intensity.cpu()).numpy()))
#         file_name = vis_dir + ('{:.6f}.png').format(rho)
#         fig.savefig(file_name, cmap='gray')
#         print(idx,rho)