'''
Method for automatically back-propagation method in intensity object
'''

import logging
import os
import numpy as np
from PIL import Image
import torch
from utils.functions import psnr, generate_otf_torch, rgb_to_gray, gray_to_rgb
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import mse_loss
from utils.layer import psnr_metric, l2_loss, norm_tensor
from utils.functions import tensor2fig
import time
from torch.fft import fft2, ifft2, fftshift, ifftshift
from DHNet import DHNet, initialization, forward_propagation
from torch.optim import Adam
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.nn import MSELoss

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = "AutoDH/"
timestr = time.strftime("%Y-%m-%d-%H_%M_%S/", time.localtime())
out_dir = 'output/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
out_dir = out_dir + model_name
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
writer = SummaryWriter(out_dir + timestr)


""" Load the GT intensity map and get the diffraction pattern"""
bbox = [734,480,734+256,480+256]
# bbox =[500,490,500+256,490+256]
img = Image.open('ExpSample/CAO/experiment/E5/obj.bmp').convert('L').crop(bbox)
bg = Image.open('ExpSample/CAO/experiment/E5/bg.bmp').convert('L').crop(bbox)
def prepross_bg(img, bg):
    temp = img / bg
    out = (temp - np.min(temp)) / (1 - np.min(temp))
    return out
processed_img = prepross_bg(np.array(img),np.array(bg)).astype(np.float32)
# img = Image.open('test_image2.jpg').resize([512, 512]).convert('L')
# img = Image.open('USAF1951.jpg').resize([512, 512]).convert('L')
holo = torch.from_numpy(processed_img)
holo = holo / torch.max(holo)

# ---- define propagation kernel -----
w = 660e-9
deltax = 5.86e-6
deltay = 5.86e-6
distance = 7.9e-3
nx = 256
ny = 256


# """ Load the GT intensity map and get the diffraction pattern"""
# img = Image.open('sample.bmp').convert('L')
# # img = Image.open('test_image2.jpg').resize([512, 512]).convert('L')
# # img = Image.open('USAF1951.jpg').resize([512, 512]).convert('L')
# holo = torch.from_numpy(np.array(img))
# holo = holo / torch.max(holo)
#
# # ---- define propagation kernel -----
# w = 635e-9
# deltax = 1.67e-6
# deltay = 1.67e-6
# distance = 875e-6
# nx = 1000
# ny = 1000
# ---- forward and backward propagation -----
A = generate_otf_torch(w, nx, ny, deltax, deltay, distance)
# holo = ifft2(torch.multiply(A, fft2(gt_intensity)))  # 此处应该是gt_intensity才对
# holo = torch.abs(holo)
# holo = holo / torch.max(holo)

AT = generate_otf_torch(w, nx, ny, deltax, deltay, -distance)
rec = ifft2(torch.multiply(AT, fft2(holo)))
rec = torch.abs(rec)
rec = norm_tensor(rec)
rec = rec / torch.max(rec)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(holo, cmap='gray')
ax[1].imshow(rec, cmap='gray')
# ax[1].set_title(('BP PSNR{:.2f}').format(psnr(rec, gt_intensity)))
fig.show()

# ---- Define the network -----
# net = DHNet(w, nx, ny, deltax, deltay, distance, n_c=1, device=device)
maxitr = 1000
visual_check = 100
verbose = True


pbar = tqdm(range(maxitr+1))
holo = holo.unsqueeze(0).unsqueeze(0)
# initialize o
o = torch.fft.ifft2(torch.multiply(torch.fft.fft2(holo), AT)).abs().to(device)
o.requires_grad = True

# gt = torch.tensor(gt_intensity).unsqueeze(0).unsqueeze(0)


# ---- Setting the training params -----
optimizer = Adam([o], lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=3, factor=0.5, threshold=0.001,
                                                 verbose=True)

if verbose:
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    formater = logging.Formatter("%(message)s")
    logger.info('\n Training========================================')
    logger.info(('\n' + '%10s' * 2) % ('Iteration  ', 'loss'))

for i in pbar:
    optimizer.zero_grad()
    pred = forward_propagation(o, A)
    mse_loss = MSELoss()(pred.abs(), holo)
    mse_loss.backward(retain_graph=True)
    optimizer.step()
    # scheduler.step(total_loss)

    # calculate metric
    # o_psnr = psnr(o, gt.to(o.device)).cpu()

    if verbose:
        info = ("{}, \t {}  ").format(i + 1, mse_loss)
        pbar.set_description(info)
        logger.info(info)
        # writer.add_scalar('metric/o_PSNR', o_psnr, i)
        writer.add_scalar('metric/loss', mse_loss, i)

    if visual_check and i % visual_check == 0:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(tensor2fig(holo), cmap='gray')
        ax[0].set_title('Hologram')
        # ax[1].imshow(tensor2fig(gt), cmap='gray')
        # ax[1].set_title('GT_initensity')
        ax[1].imshow(tensor2fig(o), cmap='gray')
        # ax[2].set_title(('o_{} \n PSNR{:.2f}').format(i, o_psnr))
        fig.show()
fig.savefig(out_dir + timestr +'output.jpg')
