'''
use test image -> ImageNet data
'''

from pnp_admm2 import pnp_ADMM_DH
from utils import *
import  PIL.Image as Image
import torch
import torch.nn.functional as F
from models.Unet import Unet
import numpy as np
from utils.functions import generate_otf_torch
from torch.fft import  fft2,ifft2,fftshift,ifftshift
import os

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

out_dir = 'output/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
""" Load pre-trained Unet pth"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Unet(3, 3, chans=64).to(device)
model.load_state_dict(torch.load('denoiser.pth', map_location=device))
print('#Parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

""" Load the GT intensity map and get the diffraction pattern"""
img = Image.open('test_image.png').resize([512,512]).convert('L')
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
holo = ifft2(torch.multiply(A,fft2(gt_intensity))) #此处应该是gt_intensity才对
holo = torch.abs(holo)
holo = holo/torch.max(holo)
# Image.fromarray(holo.numpy()*255).show(title='hologram(diffraction pattern)')
AT = generate_otf_torch(w,nx,ny,deltax,deltay,-distance)
rec = ifft2(torch.multiply(AT,fft2(holo)))
rec = torch.abs(rec)
rec = rec/torch.max(rec)
# Image.fromarray(rec.numpy()*255).show(title='BP')

# ---- set solver -----
solver = pnp_ADMM_DH(w,nx,ny,deltax,deltay,distance,model,device=device,visual_check=True)
A = solver.A
AT = solver.AT
opts = dict(rho=torch.tensor([1e-2]), maxitr=150, verbose=True, gt = torch.tensor(gt_intensity),eta=0.5,
            tol=0.000000001,gamma=1)

# ---- reconstruction using ADMMPnP-----
with torch.no_grad():
    out = solver.pnp_Admm_DH(holo, opts)