import logging
import os
import numpy as np
from PIL import Image
import torch
import cv2
import glob
import scipy.io as sio
import scipy.misc
from utils.load_model import load_model
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

model_name = "DHNet/"
timestr = time.strftime("%Y-%m-%d-%H_%M_%S/", time.localtime())
out_dir = 'output/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
out_dir = out_dir + model_name
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
writer = SummaryWriter(out_dir + timestr)

""" Load the GT intensity map and get the diffraction pattern"""
# img = Image.open('test_image.png').resize([512, 512]).convert('L')
img = Image.open('test_image2.jpg').resize([512, 512]).convert('L')
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
rec = rec / torch.max(rec)
# Image.fromarray(rec.numpy()*255).show(title='BP')
fig, ax = plt.subplots(1, 2)
ax[0].imshow(holo, cmap='gray')
ax[1].imshow(rec, cmap='gray')
ax[1].set_title(('BP PSNR{:.2f}').format(psnr(rec, gt_intensity)))
fig.show()
# ---- Define the network -----
net = DHNet(w, nx, ny, deltax, deltay, distance, n_c=1, device=device)

# ---- Setting the training params -----
optimizer = Adam(net.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=3, factor=0.5, threshold=0.001,
                                                 verbose=True)

maxitr = 1000
pbar = tqdm(range(maxitr))
holo = holo.unsqueeze(0).unsqueeze(0)
o, v, u = initialization(holo, net.AT, device)
gt = torch.tensor(gt_intensity).unsqueeze(0).unsqueeze(0)
verbose = True
visual_check = 50

if verbose:
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    formater = logging.Formatter("%(message)s")
    logger.info('\n Training========================================')
    logger.info(('\n' + '%10s' * 5) % ('Iteration  ', 'o_PSNR', 'v_PSNR', 's_loss', 'm_loss'))


with torch.autograd.set_detect_anomaly(True):
    for i in pbar:
        optimizer.zero_grad()
        o, v, u, stage_loss = net(o, v, u, holo)
        pred = forward_propagation(v, net.A)
        mse_loss = MSELoss()(pred.abs(), holo)
        total_loss = mse_loss + stage_loss
        total_loss.backward(retain_graph=True)
        optimizer.step()
        # scheduler.step(total_loss)

        # calculate metric
        o_psnr = psnr(o, gt.to(o.device)).cpu()
        v_psnr = psnr(v, gt.to(o.device)).cpu()

        if verbose:
            info = ("{}, \t {} \t {} \t {} \t {}").format(i + 1, o_psnr, v_psnr, stage_loss, mse_loss)
            pbar.set_description(info)
            logger.info(info)
            writer.add_scalar('params/rho', net.rho, i)
            writer.add_scalar('metric/v_PSNR', v_psnr, i)
            writer.add_scalar('metric/o_PSNR', o_psnr, i)
            writer.add_scalar('metric/m_loss', mse_loss, i)
            writer.add_scalar('metric/s_loss', stage_loss, i)

        if visual_check and i % visual_check == 0:
            fig, ax = plt.subplots(2, 3)
            ax[0, 0].imshow(tensor2fig(holo), cmap='gray')
            ax[0, 1].imshow(tensor2fig(gt), cmap='gray')
            # ax[0,2].imshow(init.numpy(), cmap='gray')
            # ax[0,2].set_title(('BP \n PSNR{:.2f}').format(psnr(init, gt.cpu()).numpy()))
            ax[1, 0].imshow(tensor2fig(v), cmap='gray')
            ax[1, 0].set_title(('v_{} \n PSNR{:.2f}').format(i, v_psnr))
            ax[1, 1].imshow(tensor2fig(o), cmap='gray')
            ax[1, 1].set_title(('o_{} \n PSNR{:.2f}').format(i, o_psnr))
            ax[1, 2].imshow(tensor2fig(u), cmap='gray')
            ax[1, 2].set_title('u_{}'.format(i))
            fig.show()
