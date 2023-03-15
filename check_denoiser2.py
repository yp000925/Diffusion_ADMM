"""
Check denoiser for 1 channel
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
# model = load_model(model_type='DnCNN', sigma=15, device=device)
model = load_model(model_type='RealSN_DnCNN', sigma=15, device=device)
print('#Parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

img = Image.open('test_image.png').resize([512, 512]).convert('L')
# img = Image.open('USAF1951.jpg').resize([512,512]).convert('L')
gt_intensity = torch.from_numpy(np.array(img))
gt_intensity = gt_intensity / torch.max(gt_intensity)

# Gaussian and blurred image
"""
Choose kernel from list of kernels
"""
K_IDX = 11
struct = loadmat('kernels_12.mat')
kernel_list = struct['kernels'][0]
kernel = kernel_list[K_IDX]
kernel = kernel / np.sum(kernel.ravel())

"""
Prepare the A, At operator, blurred poisson corrupted image
"""
img = np.array(gt_intensity)
k_pad, k_fft = psf2otf(kernel, [img.shape[0], img.shape[1]])
noisy = np.real(ifft2(fft2(img)*k_fft))
sigma = 0.1
noisy = noisy+np.random.randn(noisy.shape[0],noisy.shape[1])*sigma
noisy = (noisy - np.min(noisy)) / (np.max(noisy) - np.min(noisy))
fig, ax = plt.subplots(1, 2)
ax[0].imshow(noisy / np.max(noisy), cmap='gray')
ax[0].set_title(('Blurred and Guassian PSNR {:.2f}').format(psnr(torch.tensor(noisy), gt_intensity).numpy()))

noisy = torch.tensor(noisy).unsqueeze(0).unsqueeze(0).to(device)
with torch.no_grad():
    model.eval()
    residual = model(noisy.to(torch.float32))
    out = noisy-residual
# print("Processed PSNR is ",psnr(out,gt_intensity))
# Image.fromarray(np.array(out[0,:,:])*255).show()
# plt.imshow(np.array(out[0,:,:]),cmap='gray')
ax[1].imshow(np.array(out[0, 0, :, :]), cmap='gray')
ax[1].set_title(('Processed PSNR {:.2f}').format(psnr(out[0, 0, :, :], gt_intensity).numpy()))
fig.show()
