"""
Check denoiser for Unet with channel 1
"""

from scipy.io import loadmat
import PIL.Image as Image
import torch
from models.Unet import Unet
import numpy as np
import os
from utils.functions import psnr, rgb_to_gray, gray_to_rgb
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, ifftshift
from utils.load_model import load_model


def psf2otf(kernel, size):
    psf = np.zeros(size, dtype=np.float32)
    centre = np.shape(kernel)[0] // 2 + 1

    psf[:centre, :centre] = kernel[(centre - 1):, (centre - 1):]
    psf[:centre, -(centre - 1):] = kernel[(centre - 1):, :(centre - 1)]
    psf[-(centre - 1):, :centre] = kernel[:(centre - 1), (centre - 1):]
    psf[-(centre - 1):, -(centre - 1):] = kernel[:(centre - 1), :(centre - 1)]

    otf = fft2(psf, size)
    return psf, otf


SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Unet(1, 1, chans=64)
model = torch.nn.DataParallel(model).to(device)
# ckpt_pth = "/home/zhangyp/project/Diffusion_ADMM/pre_train_exp/Unet-2023-03-08-15-28/best.pt"
ckpt_pth = "best.pt"
# ckpt_pth = "pre_train_exp/Unet/final.pt"
loader=torch.load(ckpt_pth, map_location=device)
model.load_state_dict(loader['model_state_dict'])
print('#Parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

img = Image.open('test_image.png').resize([512, 512]).convert('L')
# img = Image.open('USAF1951.jpg').resize([512,512]).convert('L')
# img = Image.open('test_image2.jpg').resize([512,512]).convert('L')
gt_intensity = torch.from_numpy(np.array(img))
gt_intensity = gt_intensity / torch.max(gt_intensity)

# # Gaussian and blurred image
# """
# Choose kernel from list of kernels
# """
# K_IDX = 11
# struct = loadmat('kernels_12.mat')
# kernel_list = struct['kernels'][0]
# kernel = kernel_list[K_IDX]
# kernel = kernel / np.sum(kernel.ravel())
#
# """
# Prepare the A, At operator, blurred poisson corrupted image
# """
# img = np.array(gt_intensity)
# k_pad, k_fft = psf2otf(kernel, [img.shape[0], img.shape[1]])
# noisy = np.real(ifft2(fft2(img)*k_fft))
# sigma = 0.1
# noisy = noisy+np.random.randn(noisy.shape[0],noisy.shape[1])*sigma
# noisy = (noisy - np.min(noisy)) / (np.max(noisy) - np.min(noisy))
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(noisy / np.max(noisy), cmap='gray')
# ax[0].set_title(('Blurred and Guassian PSNR {:.2f}').format(psnr(torch.tensor(noisy), gt_intensity).numpy()))
#
# noisy = torch.tensor(noisy).unsqueeze(0).unsqueeze(0).to(device)
# with torch.no_grad():
#     model.eval()
#     out = model(noisy.to(torch.float32))
# # print("Processed PSNR is ",psnr(out,gt_intensity))
# # Image.fromarray(np.array(out[0,:,:])*255).show()
# # plt.imshow(np.array(out[0,:,:]),cmap='gray')
# ax[1].imshow(np.array(out[0, 0, :, :]), cmap='gray')
# ax[1].set_title(('Processed PSNR {:.2f}').format(psnr(out[0, 0, :, :], gt_intensity).numpy()))
# fig.show()


'''Check for propagation kernel denoise 
'''
from utils.functions import generate_otf_torch
from torch.fft import  fft2,ifft2,fftshift,ifftshift

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
AT = generate_otf_torch(w,nx,ny,deltax,deltay,-distance)
rec = ifft2(torch.multiply(AT,fft2(holo)))
rec = torch.abs(rec)
rec = rec/torch.max(rec)
fig,ax = plt.subplots(1,4)
ax[0].imshow(gt_intensity,cmap='gray')
ax[0].set_title('GT intensity')
ax[1].imshow(holo.numpy(),cmap='gray')
ax[1].set_title('Hologram')
ax[2].imshow(np.array(rec),cmap='gray')
ax[2].set_title(('BP \nPSNR{:.2f}').format(psnr(rec,gt_intensity).numpy()))
noisy3 = rec.unsqueeze(0).unsqueeze(0).to(device)
with torch.no_grad():
    model.eval()
    out = model(noisy3.to(torch.float32))

ax[3].imshow(np.array(out[0, 0, :, :]),cmap='gray')
ax[3].set_title(('Processed \n PSNR{:.2f}').format(psnr(out,gt_intensity).numpy()))
fig.show()

