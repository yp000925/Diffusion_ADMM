'''
Method for automatically back-propagation method in intensity object
'''

import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

model_name = "AutoDHc/"
timestr = time.strftime("sample_%Y-%m-%d-%H_%M_%S/", time.localtime())
out_dir = 'output/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
out_dir = out_dir + model_name
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
writer = SummaryWriter(out_dir + timestr)

""" Load the GT intensity map and get the diffraction pattern"""
img = Image.open('sample.bmp').convert('L')
# img = Image.open('test_image2.jpg').resize([512, 512]).convert('L')
# img = Image.open('USAF1951.jpg').resize([512, 512]).convert('L')
holo = torch.from_numpy(np.array(img)).to(device)
holo = holo / torch.max(holo)

# ---- define propagation kernel -----
w = 635e-9
deltax = 1.67e-6
deltay = 1.67e-6
distance = 875e-6
nx = 1000
ny = 1000
# ---- forward and backward propagation -----
A = generate_otf_torch(w, nx, ny, deltax, deltay, distance)

AT = generate_otf_torch(w, nx, ny, deltax, deltay, -distance)
# rec = ifft2(torch.multiply(AT, fft2(holo)))
# rec = torch.angle(rec)
# rec = norm_tensor(rec)
# rec = rec / torch.max(rec)*2-1
rec = ifft2(torch.multiply(AT, fft2(holo)))
rec_amp = torch.abs(rec)
rec_amp = norm_tensor(rec_amp)
rec_amp = rec_amp / torch.max(rec_amp)

rec_phase = torch.angle(rec)
rec_phase = norm_tensor(rec_phase)

fig, ax = plt.subplots(1, 3)
ax[0].imshow(holo, cmap='gray')
ax[1].imshow(rec_amp, cmap='gray')
ax[1].set_title(('BP amplitude'))
ax[2].imshow(rec_phase, cmap='gray')
ax[2].set_title(('BP phase'))
fig.show()
# ---- Define the network -----
maxitr =5000
visual_check = 500
verbose = True

pbar = tqdm(range(maxitr + 1))
holo = holo.unsqueeze(0).unsqueeze(0)
# initialize o

# o = torch.fft.ifft2(torch.multiply(torch.fft.fft2(holo), AT)).to(device)
# o_phase = torch.angle(o)
# o_amp = torch.abs(o)
o_phase = torch.nn.Parameter(-1+2*torch.rand(holo.shape,dtype=holo.dtype, requires_grad=True)).to(device)
o_amp = torch.nn.Parameter(torch.rand(holo.shape,dtype=holo.dtype, requires_grad=True)).to(device)

o_phase.requires_grad = True
o_amp.requires_grad = True


# ---- Setting the training params -----
optimizer = Adam([o_phase, o_amp], lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=5, factor=0.5, threshold=0.001,
                                                 verbose=True)

if verbose:
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    formater = logging.Formatter("%(message)s")
    logger.info('\n Training========================================')
    logger.info(('\n' + '%10s' * 2) % ('Iteration  ','loss'))

for i in pbar:
    optimizer.zero_grad()
    o = torch.exp(1j * o_phase) * o_amp
    pred = forward_propagation(o, A).abs()

    mse_loss = MSELoss()(pred, holo)
    mse_loss.backward(retain_graph=True)
    optimizer.step()
    # scheduler.step(mse_loss)

    # # calculate metric
    # o_psnr_phase = psnr(o_phase, gt_phase.to(o.device)).cpu()
    # o_psnr_amp = psnr(o_amp, gt_amp.to(o.device)).cpu()

    if verbose:
        info = ("{}, \t {}").format(i + 1, mse_loss)
        pbar.set_description(info)
        logger.info(info)
        writer.add_scalar('metric/loss', mse_loss, i)

    if visual_check and i % visual_check == 0:
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(tensor2fig(holo), cmap='gray')
        ax[0].set_title('Hologram')
        ax[1].imshow(tensor2fig(o_phase), cmap='gray')
        ax[1].set_title(('o_p{} \n').format(i))
        ax[2].imshow(tensor2fig(o_amp), cmap='gray')
        ax[2].set_title(('o_a{} \n').format(i))
        fig.show()
fig.savefig(out_dir + timestr +'output.jpg')