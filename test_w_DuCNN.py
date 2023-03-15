"""
Substitude the model with different structure
use the pre-trained model -> DnCNN realSNDnCNN learn the residual
"""
from pnp_admm_res import pnp_ADMM_DH
from utils import *
import  PIL.Image as Image
import torch
import torch.nn.functional as F
from models.Unet import Unet
import numpy as np
from utils.functions import generate_otf_torch,psnr
from torch.fft import  fft2,ifft2,fftshift,ifftshift
import os
from utils.load_model import load_model
import matplotlib.pyplot as plt


SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

out_dir = 'output/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sigma = 15
model = load_model(model_type='DnCNN', sigma=sigma, device=device)
print('#Parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
denoiser = {}
denoiser['model'] = model
denoiser['sigma'] = sigma
""" Load the GT intensity map and get the diffraction pattern"""
img = Image.open('USAF1951.jpg').resize([512,512]).convert('L')
# img = Image.open('test_image2.jpg').resize([512, 512]).convert('L')
gt_intensity = torch.from_numpy(np.array(img))
gt_intensity = (gt_intensity-torch.min(gt_intensity))/(torch.max(gt_intensity)-torch.min(gt_intensity))

# ---- define propagation kernel -----
w = 632e-9
deltax=3.45e-6
deltay=3.45e-6
distance = 0.02
nx = 512
ny= 512
# ---- forward and backward propagation -----
A = generate_otf_torch(w,nx,ny,deltax,deltay,distance)
holo = ifft2(torch.multiply(A,fft2(gt_intensity)))
holo = torch.abs(holo)
holo = (holo-torch.min(holo))/(torch.max(holo)-torch.min(holo))
# Image.fromarray(holo.numpy()*255).show(title='hologram(diffraction pattern)')
AT = generate_otf_torch(w,nx,ny,deltax,deltay,-distance)
rec = ifft2(torch.multiply(AT,fft2(holo)))
rec = torch.abs(rec)
rec = (rec-torch.min(rec))/(torch.max(rec)-torch.min(rec))
# Image.fromarray(rec.numpy()*255).show(title='BP')
fig,ax = plt.subplots(1,2)
ax[0].imshow(holo.numpy(),cmap='gray')
ax[0].set_title('hologram(diffraction pattern)')
ax[1].imshow(rec.numpy(),cmap='gray')
ax[1].set_title('BP with PSNR{:.4f}'.format(psnr(rec, gt_intensity).numpy()))
fig.show()
# ---- set solver -----
solver = pnp_ADMM_DH(w,nx,ny,deltax,deltay,distance,denoiser,device=device,visual_check=False)
A = solver.A
AT = solver.AT
opts = dict(rho=torch.tensor([1e-1]), maxitr=300, verbose=True, gt = torch.tensor(gt_intensity),eta=0.3,
            tol=0.0000001,gamma=1)

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