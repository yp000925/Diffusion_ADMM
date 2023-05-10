"""
Check denoiser for diffusion model , channel 3
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from scipy.io import loadmat
import PIL.Image as Image
import torch
from models.diffuser import diffusion,diffusion_default
import torchvision.utils as tvu
import numpy as np

from utils.functions import psnr, rgb_to_gray, gray_to_rgb
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, ifftshift
import time

model_name = "diffuser/"
timestr = time.strftime("%Y-%m-%d-%H_%M_%S/", time.localtime())
out_dir = 'output/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
out_dir = out_dir + model_name+timestr
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

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
denoiser = diffusion(model_pth="256x256_diffusion_uncond.pt",model_type='default')



# ---- open image for test -----
img_path ="Test4DiffusionModel/clean_Lena.png"
# img = Image.open(img_path).resize([512,512]).convert('L')
img = Image.open('USAF1951.jpg').resize([512,512]).convert('L')
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
w = 632e-9
deltax=3.45e-6
deltay=3.45e-6
distance = 0.02
nx = 512
ny= 512
# w = 632e-9
# deltax=3.45e-6*2
# deltay=3.45e-6*2
# distance = 0.02
# nx = 256
# ny= 256
# ---- forward and backward propagation -----
A = generate_otf_torch(w,nx,ny,deltax,deltay,distance)
holo = ifft2(torch.multiply(A,fft2(gt_intensity)))
holo = torch.abs(holo)
holo = holo/torch.max(holo)
AT = generate_otf_torch(w,nx,ny,deltax,deltay,-distance)
rec = ifft2(torch.multiply(AT,fft2(holo)))
rec = torch.abs(rec)
rec = rec/torch.max(rec)
# fig,ax = plt.subplots(1,3)
# ax[0].imshow(gt_intensity,cmap='gray')
# ax[0].set_title('GT intensity')
# ax[1].imshow(holo.numpy(),cmap='gray')
# ax[1].set_title('Hologram')
# ax[2].imshow(np.array(rec),cmap='gray')
# ax[2].set_title(('BP \nPSNR{:.2f}').format(psnr(rec,gt_intensity).numpy()))
# plt.show()



# ---- define propagation kernel -----
diffusion_args = diffusion_default()
diffusion_args.sigma_0 = 0.2
diffusion_args.timesteps=40
diffusion_args.num_diffusion_timesteps = 2000

def resize_and_process(img_np,nx=256,ny=256):
    img_np = img_np/img_np.max()*255
    img = Image.fromarray(img_np).resize([nx,ny])
    img_np = np.array(img)
    return img_np/img_np.max()

def input_modifier_t(i):
    """
    transfer from one channel to 3 channel
    :param i:
    :return:
    """
    o = i.repeat([1,3,1,1])
    return o
def output_modifier_t(i):
    """
    transfer from 3 channel to 1 channel
    :param i:
    :return:
    """
    o = torch.mean(i,1, keepdim=True)
    return o

# DDRM denoising
rec_np = resize_and_process(np.array(rec.detach()))
input = torch.tensor(rec_np).unsqueeze(0).unsqueeze(0).to(device)

gt = torch.from_numpy(np.array(img.resize([256,256])))
gt = gt/gt.max()
gt = gt.unsqueeze(0).unsqueeze(0)

print(diffusion_args)
y = denoiser.denoise(input_modifier_t(input),diffusion_args)
plot_check = [0,len(y)-1]
for j in range(len(y)):
    # tvu.save_image(
    #     y[j], os.path.join(out_dir, prefix +f"_sigma{diffusion_args.sigma_0}"+f"_{j}.png")
    # )
    psnr_val = psnr(output_modifier_t(y[j].detach().cpu()),gt).numpy()
    print(psnr_val)
    if j in plot_check:
        tvu.save_image(
            y[j], os.path.join(out_dir, prefix +f"_sigma{diffusion_args.sigma_0}"+f"_{j}.png")
        )

fig,ax = plt.subplots(1,4)
ax[0].imshow(gt_intensity,cmap='gray')
ax[0].set_title('GT intensity')
ax[1].imshow(holo.numpy(),cmap='gray')
ax[1].set_title('Hologram')
ax[2].imshow(np.array(rec),cmap='gray')
ax[2].set_title(('BP \nPSNR{:.2f}').format(psnr(rec,gt_intensity).numpy()))
ax[3].imshow(y[-1][0,0,:,:].detach().numpy(),cmap='gray')
ax[3].set_title(('Diffuser \nPSNR{:.2f}').format(psnr_val))
plt.show()
