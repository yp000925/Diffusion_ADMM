'''
use test image -> ImageNet data
'''
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pnp_admm_Unet1c_V2 import pnp_ADMM_DH
from utils import *
import PIL.Image as Image
import torch
import torch.nn.functional as F
from models.Unet import Unet
import numpy as np
from utils.functions import generate_otf_torch, psnr,norm_tensor
from torch.fft import fft2, ifft2, fftshift, ifftshift
import time
import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

# out_dir = 'output/'
#
# if not os.path.exists(out_dir):
#     os.mkdir(out_dir)
""" Load pre-trained Unet pth"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Unet(1, 1, chans=64)
model = torch.nn.DataParallel(model).to(device)
ckpt_pth = "/home/zhangyp/project/Diffusion_ADMM/pre_train_exp/Unet/final.pt"
loader = torch.load(ckpt_pth, map_location=device)
model.load_state_dict(loader['model_state_dict'])
print('#Parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

""" Load the GT intensity map and get the diffraction pattern"""
# img = Image.open('test_image.png').resize([512, 512]).convert('L')
img = Image.open('ExpSample/celeA/testsample.jpeg').resize([512, 512]).convert('L')
# img = Image.open('test_image2.jpg').resize([512, 512]).convert('L')
# img = Image.open('USAF1951.jpg').resize([512, 512]).convert('L')
gt_intensity = torch.from_numpy(np.array(img))
gt_intensity = gt_intensity / torch.max(gt_intensity)

# ---- define propagation kernel -----
w = 632e-9
deltax = 3.45e-6
deltay = 3.45e-6
distance = 0.02
nx = 512
ny = 512
# ---- forward and backward propagation -----
A = generate_otf_torch(w, nx, ny, deltax, deltay, distance)
holo = ifft2(torch.multiply(A, fft2(gt_intensity)))  # 此处应该是gt_intensity才对
holo = torch.abs(holo)
# holo = norm_tensor(holo)
holo = holo / torch.max(holo)
# Image.fromarray(holo.numpy()*255).show(title='hologram(diffraction pattern)')
AT = generate_otf_torch(w, nx, ny, deltax, deltay, -distance)
rec = ifft2(torch.multiply(AT, fft2(holo)))
rec = torch.abs(rec)
rec = norm_tensor(rec)
# rec = rec / torch.max(rec)
# Image.fromarray(rec.numpy()*255).show(title='BP')

# ---- set solver -----
solver = pnp_ADMM_DH(w, nx, ny, deltax, deltay, distance, model, device=device, visual_check=10)
A = solver.A
AT = solver.AT
opts = dict(rho=torch.tensor([0.05]), maxitr=100, verbose=True, gt=torch.tensor(gt_intensity), eta=0.9,
            tol=0.0000001, gamma=1, psnr_tol=0, patient=100, mu=0.01)

# ---- reconstruction using ADMMPnP-----
with torch.no_grad():
    out = solver.pnp_Admm_DH(holo, opts)
    fig, ax = plt.subplots(2, 3)
    ax[0,0].imshow(holo.cpu().numpy(), cmap='gray')
    ax[0,1].imshow(gt_intensity.cpu().numpy(), cmap='gray')
    ax[0,2].imshow(rec.cpu().numpy(), cmap='gray')
    ax[0,2].set_title(('BP \n PSNR{:.2f}').format(psnr(rec.cpu(), gt_intensity.cpu()).numpy()))
    ax[1,0].imshow(out[0].cpu().numpy(), cmap='gray')
    ax[1,0].set_title(('v_out \n PSNR{:.2f}').format(psnr(out[0].cpu(), gt_intensity.cpu()).numpy()))
    ax[1,1].imshow(out[1].cpu().numpy(), cmap='gray')
    ax[1,1].set_title(('o_out \n PSNR{:.2f}').format(psnr(out[1].cpu(), gt_intensity.cpu()).numpy()))
    ax[1,2].imshow(out[2].cpu().numpy(), cmap='gray')
    ax[1,2].set_title('u_out')
    fig.show()

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