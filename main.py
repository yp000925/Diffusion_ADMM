from pnp_admm import pnp_ADMM_DH
from utils import *
import  PIL.Image as Image
import torch
import torch.nn.functional as F
from models.Unet import Unet
import numpy as np
from utils.functions import generate_otf_torch,psnr,norm_tensor
from torch.fft import  fft2,ifft2,fftshift,ifftshift
import os
import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

out_dir = 'output/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
""" Load pre-trained Unet pth"""
# import gdown
# url = 'https://drive.google.com/file/d/1FFuauq-PUjY_kG3iiiHfDpHcG4Srl8mQ/view?usp=sharing'
# output = "denoiser.pth"
# gdown.download(url, output, quiet=False,fuzzy=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Unet(3, 3, chans=64).to(device)
model.load_state_dict(torch.load('denoiser.pth', map_location=device))
print('#Parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

""" Load the GT intensity map and get the diffraction pattern"""
img = Image.open('USAF1951.jpg').resize([512,512]).convert('L')
gt_intensity = torch.from_numpy(np.array(img))
gt_intensity = gt_intensity/torch.max(gt_intensity)

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
holo = holo/torch.max(holo)
# Image.fromarray(holo.numpy()*255).show(title='hologram(diffraction pattern)')
AT = generate_otf_torch(w,nx,ny,deltax,deltay,-distance)
rec = ifft2(torch.multiply(AT,fft2(holo)))
rec = torch.abs(rec)
rec = rec/torch.max(rec)
# Image.fromarray(rec.numpy()*255).show(title='BP')

# ---- set solver -----
solver = pnp_ADMM_DH(w,nx,ny,deltax,deltay,distance,model,device=device,visual_check=False)
A = solver.A
AT = solver.AT
opts = dict(rho=torch.tensor([1e-4]), maxitr=100, verbose=True, gt = torch.tensor(gt_intensity))

# ---- reconstruction using ADMMPnP-----
with torch.no_grad():
    out = solver.pnp_Admm_DH(holo, opts)
    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(holo.cpu().numpy(), cmap='gray')
    ax[1].imshow(gt_intensity.cpu().numpy(), cmap='gray')
    ax[2].imshow(rec.cpu().numpy(), cmap='gray')
    ax[2].set_title(('Processed \n PSNR{:.2f}').format(psnr(rec.cpu(), gt_intensity.cpu()).numpy()))
    ax[3].imshow(out.cpu().numpy(), cmap='gray')
    ax[3].set_title(('Processed \n PSNR{:.2f}').format(psnr(out.cpu(), gt_intensity.cpu()).numpy()))
    plt.show()