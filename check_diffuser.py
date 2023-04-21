"""
Check denoiser for diffusion model , channel 3
"""

from scipy.io import loadmat
import PIL.Image as Image
import torch
from models.diffuser import diffusion,diffusion_default
import torchvision.utils as tvu
import numpy as np
import os
from utils.functions import psnr, rgb_to_gray, gray_to_rgb
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, ifftshift


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
denoiser = diffusion(model_pth="/Users/zhangyunping/PycharmProjects/ddrm-master/exp/logs/imagenet/256x256_diffusion_uncond.pt",model_type='default')



# ---- open image for test -----
img_path ="/Users/zhangyunping/PycharmProjects/Diffusion_ADMM/Test4DiffusionModel/clean_Lena.png"
img = Image.open(img_path).resize([512,512]).convert('L')
# img = Image.open('USAF1951.jpg').resize([512,512]).convert('L')
# img = Image.open('test_image2.jpg').resize([512,512]).convert('L')
gt_intensity = torch.from_numpy(np.array(img))
gt_intensity = gt_intensity / torch.max(gt_intensity)
prefix = img_path.split('/')[-1]
prefix = prefix.split('.')[0]

'''Check for propagation kernel denoise 
'''
from utils.functions import generate_otf_torch
from torch.fft import  fft2,ifft2,fftshift,ifftshift

# ---- define propagation kernel -----
# w = 632e-9
# deltax=3.45e-6
# deltay=3.45e-6
# distance = 0.02
# nx = 512
# ny= 512
w = 632e-9
deltax=3.45e-6*2
deltay=3.45e-6*2
distance = 0.02
nx = 256
ny= 256
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

# ---- define propagation kernel -----
diffusion_args = diffusion_default()

# DDRM denoising
output_folder = "output/diffuser"
holo_expand = gray_to_rgb(holo).unsqueeze(0)
y = denoiser.denoise(holo_expand,diffusion_args)
for j in range(len(y)):
    tvu.save_image(
        y[j], os.path.join(output_folder, prefix +f"_sigma{diffusion_args.sigma_0}"+f"_{j}.png")
    )
    mse = torch.mean((y[j].to(img.device) - img) ** 2)
    psnr = 10 * torch.log10(1 / mse)
    print(psnr)