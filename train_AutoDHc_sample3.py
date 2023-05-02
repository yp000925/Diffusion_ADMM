'''
use DCOD cheek sample
'''
import os
from torch.optim import Adam
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pnp_admm_Unet1c_V2 import pnp_ADMM_DH
from utils import *
import PIL.Image as Image
import torch
import torch.nn.functional as F
from models.Unet import Unet
import numpy as np
from utils.functions import generate_otf_torch, psnr,norm_tensor,tensor2fig,forward_propagation,prepross_bg
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
timestr = time.strftime("sample3_%Y-%m-%d-%H_%M_%S/", time.localtime())
out_dir = 'output/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
out_dir = out_dir + model_name
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
writer = SummaryWriter(out_dir + timestr)

""" Load the GT intensity map and get the diffraction pattern"""
img = plt.imread('ExpSample/DCOD/Cheek cells/hologram.tif')
bg = plt.imread('ExpSample/DCOD/Cheek cells/background.tif')
img = prepross_bg(img, bg)

# ---- define propagation kernel -----
w = 532.3e-9
deltax = 1.12e-6
deltay = 1.12e-6
distance = 238e-6
nx =512
ny = 512

holo = torch.from_numpy(np.array(img))
holo = holo / torch.max(holo)

# ---- forward and backward propagation -----
A = generate_otf_torch(w, nx, ny, deltax, deltay, distance)
AT = generate_otf_torch(w, nx, ny, deltax, deltay, -distance)

rec = ifft2(torch.multiply(AT, fft2(holo)))
rec_amp = torch.abs(rec)
rec_amp = norm_tensor(rec_amp)
rec_amp = rec_amp / torch.max(rec_amp)
rec_phase = torch.angle(rec)
rec_phase = norm_tensor(rec_phase)
plt.imsave(out_dir + timestr +'bp_phase.png',rec_phase)
plt.imsave(out_dir + timestr +'bp_amp.png',rec_amp)
plt.imsave(out_dir + timestr +'bp_phase.png',rec_phase,cmap='gray')
plt.imsave(out_dir + timestr +'bp_amp.png',rec_amp,cmap='gray')

fig, ax = plt.subplots(1, 3)
ax[0].imshow(holo.cpu(), cmap='gray')
ax[1].imshow(rec_amp, cmap='gray')
ax[1].set_title(('BP amplitude'))
ax[2].imshow(rec_phase, cmap='gray')
ax[2].set_title(('BP phase'))
fig.show()


# ---- Define the network -----
maxitr =30000
visual_check = 1000
verbose = True

pbar = tqdm(range(maxitr + 1))
holo = holo.unsqueeze(0).unsqueeze(0)
holo = holo.to(device)
A = A.to(device)
AT = AT.to(device)
# initialize o

# o = torch.fft.ifft2(torch.multiply(torch.fft.fft2(holo), AT)).to(device)
# o_phase = torch.angle(o)
# o_amp = torch.abs(o)
# o_phase.requires_grad = True
# o_amp.requires_grad = True
# #
# init = torch.nn.Parameter(torch.rand([1,1,nx,ny],dtype=holo.dtype)).to(device)
# o = torch.fft.ifft2(torch.multiply(torch.fft.fft2(init), AT)).to(device)
# o_phase = torch.angle(o)
# o_amp = torch.abs(o)
# o_phase.requires_grad = True
# o_amp.requires_grad = True
# o_phase = torch.nn.Parameter(-1+2*torch.rand(holo.shape,dtype=holo.dtype, requires_grad=True))
# o_amp = torch.nn.Parameter(torch.rand(holo.shape,dtype=holo.dtype, requires_grad=True))

o_phase = torch.rand(holo.shape,dtype=holo.dtype).to(device)

o_amp =torch.rand(holo.shape,dtype=holo.dtype).to(device)
# o_amp = torch.ones_like(o_phase).to(device)
o_phase.requires_grad = True
o_amp.requires_grad = True
# print(o_phase.is_leaf)
# print(o_amp.is_leaf)
# ---- Setting the training params -----
optimizer = Adam([o_phase,o_amp], lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=1000, factor=0.5, threshold=0.001,
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
    pred = pred*pred
    pred = pred/torch.max(pred)
    # pred =torch.fft.ifft2(torch.multiply(torch.fft.fft2( torch.exp(1j * o_phase) * o_amp), A))
    # mse_loss = MSELoss()(pred, holo)
    mse_loss =  torch.mean((pred - holo)**2)
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
        fig, ax = plt.subplots(1, 4)
        ax[0].imshow(tensor2fig(holo), cmap='gray')
        ax[0].set_title('Hologram')
        ax[2].imshow(tensor2fig(o_phase), cmap='gray')
        ax[2].set_title(('o_p{} \n').format(i))
        ax[1].imshow(tensor2fig(o_amp), cmap='gray')
        ax[1].set_title(('o_a{} \n').format(i))
        ax[3].imshow(tensor2fig(pred), cmap='gray')
        ax[3].set_title(('holo_pred{} \n').format(i))

        fig.show()
fig.savefig(out_dir + timestr +'output.jpg')
plt.imsave(out_dir + timestr +'o_phase.png',tensor2fig(o_phase))
plt.imsave(out_dir + timestr +'o_amp.png',tensor2fig(o_amp))
plt.imsave(out_dir + timestr +'o_phase_g.png',tensor2fig(o_phase),cmap='gray')
plt.imsave(out_dir + timestr +'o_amp_g.png',tensor2fig(o_amp),cmap='gray')