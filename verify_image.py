import torch
import torch.nn.functional as F
from models.Unet import Unet
import numpy as np
from utils.functions import generate_otf_torch, psnr,norm_tensor,tensor2fig,forward_propagation,prepross_bg
from torch.fft import fft2, ifft2, fftshift, ifftshift
import time
import PIL.Image as Image
import matplotlib.pyplot as plt
# img = Image.open('test_image.png').resize([512, 512]).convert('L')
img = Image.open('USAF1951.jpg').resize([512, 512]).convert('L')
gt_intensity = torch.from_numpy(np.array(img))
gt_intensity = gt_intensity / torch.max(gt_intensity)

w = 632e-9
deltax = 3.45e-6
deltay = 3.45e-6
distance = 0.02
nx = 512
ny = 512
dis_range = np.arange(0,0.05,0.0001)
# for distance in dis_range:
#     A = generate_otf_torch(w, nx, ny, deltax, deltay, distance)
#     holo = ifft2(torch.multiply(A, fft2(gt_intensity)))  # 此处应该是gt_intensity才对
#     holo = torch.abs(holo)
#     # holo = norm_tensor(holo)
#     holo = holo / torch.max(holo)
#     # Image.fromarray(holo.numpy()*255).show(title='hologram(diffraction pattern)')
#     AT = generate_otf_torch(w, nx, ny, deltax, deltay, distance)
#     rec = ifft2(torch.multiply(AT, fft2(holo)))
#     rec = torch.abs(rec)
#     rec = norm_tensor(rec)
#
#     plt.imshow(rec.cpu().numpy(),cmap='gray')
#     plt.show()
#
#     AT = generate_otf_torch(w, nx, ny, deltax, deltay, -distance)
#     rec = ifft2(torch.multiply(AT, fft2(holo)))
#     rec = torch.abs(rec)
#     rec = norm_tensor(rec)
#
#     plt.imshow(rec.cpu().numpy(),cmap='gray')
#     plt.show()

for distance in dis_range:
    A = generate_otf_torch(w, nx, ny, deltax, deltay, distance)
    holo = ifft2(torch.multiply(A, fft2(gt_intensity)))  # 此处应该是gt_intensity才对
    holo = torch.abs(holo)
    # holo = norm_tensor(holo)
    holo = holo / torch.max(holo)
    # Image.fromarray(holo.numpy()*255).show(title='hologram(diffraction pattern)')
    # AT = generate_otf_torch(w, nx, ny, deltax, deltay, -distance)
    # rec = ifft2(torch.multiply(AT, fft2(holo)))
    # rec = torch.abs(rec)
    # rec = norm_tensor(rec)
    plt.imsave('holo_distance_{:.3f}.png'.format(distance),holo.cpu().numpy(),cmap='gray')