'''
Method for automatically back-propagation method in intensity object
'''

import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pnp_admm_Unet1c_V2 import pnp_ADMM_DH
from utils import *
from torch.optim import Adam
import PIL.Image as Image
import torch
import torch.nn.functional as F
from models.Unet import Unet
import numpy as np
from utils.functions import generate_otf_torch, psnr,norm_tensor,tensor2fig,forward_propagation
from torch.fft import fft2, ifft2, fftshift, ifftshift
import time
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import logging
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = "AutoDHc/"
timestr = time.strftime("%Y-%m-%d-%H_%M_%S/", time.localtime())
out_dir = 'output/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
out_dir = out_dir + model_name
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
writer = SummaryWriter(out_dir + timestr)

""" Load the GT intensity map and get the diffraction pattern"""
img = Image.open('test_image.png').resize([512, 512]).convert('L')
# img = Image.open('test_image2.jpg').resize([512, 512]).convert('L')
# img = Image.open('USAF1951.jpg').resize([512, 512]).convert('L')
gt_phase = torch.from_numpy(np.array(img))
gt_phase = gt_phase / torch.max(gt_phase) * 2 - 1
gt_amp = torch.ones_like(gt_phase)

# ---- define propagation kernel -----
w = 632e-9
deltax = 3.45e-6
deltay = 3.45e-6
distance = 0.02
nx = 512
ny = 512
# ---- forward and backward propagation -----
A = generate_otf_torch(w, nx, ny, deltax, deltay, distance)

gt_obj = gt_amp * torch.exp(1j * gt_phase)

holo = ifft2(torch.multiply(A, fft2(gt_obj)))  # 此处应该是gt_intensity才对
holo = torch.abs(holo)
holo = holo / torch.max(holo)

gt_phase = torch.tensor(gt_phase).unsqueeze(0).unsqueeze(0)
gt_amp = torch.tensor(gt_amp).unsqueeze(0).unsqueeze(0)

AT = generate_otf_torch(w, nx, ny, deltax, deltay, -distance)
rec = ifft2(torch.multiply(AT, fft2(holo)))
rec_amp = torch.abs(rec)
rec_amp = norm_tensor(rec_amp)
rec_amp = rec_amp / torch.max(rec_amp)
rec_phase = torch.angle(rec)
rec_phase = norm_tensor(rec_phase)
plt.imsave(out_dir + timestr +'bp_phase.png',rec_phase)
plt.imsave(out_dir + timestr +'bp_amp.png',rec_amp)

# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(holo, cmap='gray')
# ax[1].imshow(rec, cmap='gray')
# ax[1].set_title(('BP PSNR{:.2f}').format(psnr(rec, gt_phase)))
# fig.show()

# ---- Define the network -----
maxitr = 5000
visual_check = 500
verbose = True
pbar = tqdm(range(maxitr + 1))
holo = holo.unsqueeze(0).unsqueeze(0)
holo = holo.to(device)
A = A.to(device)
AT = AT.to(device)

# # initialize o using random number
# o_phase = torch.rand(holo.shape,dtype=holo.dtype).to(device)
# o_amp =torch.rand(holo.shape,dtype=holo.dtype).to(device)
# o_phase.requires_grad = True
# o_amp.requires_grad = True


# initialize o using BP
o = torch.fft.ifft2(torch.multiply(torch.fft.fft2(holo), AT)).to(device)
o_phase = torch.angle(o)
o_amp = torch.abs(o)
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
    logger.info(('\n' + '%10s' * 4) % ('Iteration  ', 'phase_PSNR', 'amp_PSNR', 'loss'))

for i in pbar:
    optimizer.zero_grad()
    o = torch.exp(1j * o_phase) * o_amp
    pred = forward_propagation(o, A).abs()
    mse_loss =  torch.mean((pred - holo)**2)
    mse_loss.backward(retain_graph=True)
    optimizer.step()
    # scheduler.step(mse_loss)

    # calculate metric
    o_psnr_phase = psnr(o_phase, gt_phase.to(o.device)).cpu()
    o_psnr_amp = psnr(o_amp, gt_amp.to(o.device)).cpu()

    if verbose:
        info = ("{}, \t {} \t {} \t {}").format(i + 1, o_psnr_phase, o_psnr_amp, mse_loss)
        pbar.set_description(info)
        logger.info(info)
        writer.add_scalar('metric/p_PSNR', o_psnr_phase, i)
        writer.add_scalar('metric/a_PSNR', o_psnr_amp, i)
        writer.add_scalar('metric/loss', mse_loss, i)

    if visual_check and i % visual_check == 0:
        fig, ax = plt.subplots(1, 5)
        ax[0].imshow(tensor2fig(holo), cmap='gray')
        ax[0].set_title('Hologram')
        ax[1].imshow(tensor2fig(gt_phase), cmap='gray')
        ax[1].set_title('GT_p')
        ax[2].imshow(tensor2fig(o_phase), cmap='gray')
        ax[2].set_title(('o_p{} \n PSNR{:.2f}').format(i, o_psnr_phase))
        ax[3].imshow(tensor2fig(o_amp), cmap='gray')
        ax[3].set_title(('o_a{} \n PSNR{:.2f}').format(i, o_psnr_amp))
        ax[4].imshow(tensor2fig(pred), cmap='gray')
        ax[4].set_title(('holo_pred{} \n').format(i))
        fig.show()
fig.savefig(out_dir + timestr +'output.jpg')
plt.imsave(out_dir + timestr +'o_phase.png',tensor2fig(o_phase))
plt.imsave(out_dir + timestr +'o_amp.png',tensor2fig(o_amp))
plt.imsave(out_dir + timestr +'pred_holo.png',tensor2fig(pred))