import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import diffusers
from diffusers import UNet2DModel,DDPMPipeline,DDPMScheduler
import tqdm
import json
import time
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from utils.functions import psnr, generate_otf_torch, zero_padding_torch, crop_img_torch,norm_tensor, \
    display_sample,gray_to_rgb,rgb_to_gray_tensor
from torch.fft import fft2, ifft2
import  PIL.Image as Image
import matplotlib.pyplot as plt

testname  = 'myUSAF'
model_name = 'ddpm_gb_dh/'
timestr = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
out_dir = 'output/'
out_dir = out_dir + model_name+testname+'/'+timestr
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
writer = SummaryWriter(out_dir)

def diffusion_default():
    config = dict(
        _class_name = "DDPMScheduler",
        _diffusers_version= "0.16.1",
        beta_end= 0.02,
        beta_schedule= "linear",
        beta_start=0.0001,
        clip_sample= True,
        num_train_timesteps= 30,
        prediction_type="epsilon",
        trained_betas= None,
        variance_type= "fixed_small",
        sample_max_value = 1.0,
        clip_sample_range=1.0,
        dynamic_thresholding_ratio = 0.995,
        thresholding = False
    )
    return config



class diffuser_GB_DH():
    def __init__(self, model, diffuser_scheduler, prop_kernel, device):
        self.A = generate_otf_torch(**prop_kernel)
        self.AT = torch.conj(self.A)
        self.scheduler = diffuser_scheduler
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.pad_size = prop_kernel['pad_size']
        self.crop_size = [prop_kernel['nx'], prop_kernel['ny']]

    def forward_prop(self, f_in):
        fs_out = torch.multiply(torch.fft.fft2(f_in), self.A.expand(f_in.shape).to(self.device))
        f_out = torch.fft.ifft2(fs_out)
        return f_out

    def backward_prop(self, f_in):
        fs_out = torch.multiply(torch.fft.fft2(f_in), self.AT.expand(f_in.shape).to(self.device))
        f_out = torch.fft.ifft2(fs_out)
        return f_out


    def forward_op(self, x_in):
        x_out = zero_padding_torch(x_in, pad_size=self.pad_size)
        x_out = self.forward_prop(x_out)
        x_out = crop_img_torch(x_out, crop_size=self.crop_size)
        return x_out

    def backward_op(self, x_in):
        x_out = zero_padding_torch(x_in, pad_size=self.pad_size)
        x_out = self.backward_prop(x_out)
        x_out = crop_img_torch(x_out, crop_size=self.crop_size)
        return x_out

    # def cal_gradient(self, x, y):
    #     '''
    #
    #     :param x: the complex-valued transmittance of the sample
    #     :param y: Intensity image (absolute value)
    #     :return: Wirtinger gradient
    #     '''
    #     Ax = self.forward_op(x)
    #     AtAx = self.backward_op(Ax.abs())
    #     Aty= self.backward_op(y)
    #     # temp = (torch.abs(temp) - y) * torch.exp(1j * torch.angle(temp))
    #     gradient =AtAx.abs() - Aty.abs()
    #     return gradient

    def cal_gradient(self, x, y):
        '''

        :param x: the complex-valued transmittance of the sample
        :param y: Intensity image (absolute value)
        :return: Wirtinger gradient
        '''
        Ax = self.forward_op(x)
        temp = (Ax.abs()-y)* torch.exp(1j * torch.angle(Ax))
        gradient = self.backward_op(temp)
        # temp = (torch.abs(temp) - y) * torch.exp(1j * torch.angle(temp))
        return gradient.real

    def reconstruction(self, holo, gamma, visual_check=None):
        ## initialization
        sample = self.backward_op(holo).abs().to(self.device)
        sample =norm_tensor(sample)
        display_sample(sample, 0)
        # sample=  torch.randn(
        #     1, model.config.in_channels, model.config.sample_size, model.config.sample_size
        # ).to(self.device)
        # sample=  torch.randn(
        #     1, 1, model.config.sample_size, model.config.sample_size
        # ).to(self.device)
        sample=  holo.to(self.device)
        for i, t in enumerate(tqdm.tqdm(self.scheduler.timesteps)):
            # 1. predict noise residual
            sample = gray_to_rgb(sample)
            with torch.no_grad():
                residual = self.model(sample, t).sample
                # 2. compute less noisy image and set x_t -> x_t-1
                sample = scheduler.step(residual, t, sample).prev_sample

                # 3. gradient correction
                sample = rgb_to_gray_tensor(sample)
                sample = sample - gamma*self.cal_gradient(sample, holo)


            # 3. optionally look at image
            if visual_check and (i + 1) % visual_check == 0:
                display_sample(norm_tensor(sample), i + 1)
        return sample



SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# img = Image.open('ExpSample/celeA/testsample.jpeg').resize([256, 256]).convert('L')
diffusion_config = diffusion_default()


def prepross_bg(img, bg):
    temp = img / bg
    out = (temp - np.min(temp)) / (1 - np.min(temp))
    return out



if testname == 'CAO':
    # bbox = [734,480,734+256,480+256]
    bbox =[500,490,500+256,490+256]
    img = Image.open('ExpSample/CAO/experiment/E5/obj.bmp').convert('L').crop(bbox)
    bg = Image.open('ExpSample/CAO/experiment/E5/bg.bmp').convert('L').crop(bbox)
    processed_img = prepross_bg(np.array(img),np.array(bg)).astype(np.float32)
    plt.imshow(processed_img)
    plt.title("diffraction pattern")
    plt.show()
    with open('ExpSample/CAO/experiment/E5/params.json','r') as f:
        params = json.load(f)
    # ---- define propagation kernel -----
    w = params['w']
    # deltax = 2*1.12e-6
    # deltay = 2*1.12e-6
    deltax = params['deltax']
    deltay = params['deltay']
    distance = params['distance']
    nx,ny = processed_img.shape

    diffusion_config['num_train_timesteps'] = 30
    gamma = 2
    loi_x = range(nx)
    loi_y = 60

elif testname=='DCOD':
    bbox = [100,106,100+300,106+300]
    # img = Image.open('ExpSample/DCOD/USAF/hologram.tif').resize([256, 256])
    # bg = Image.open('ExpSample/DCOD/USAF/background.tif').resize([256, 256])
    img = Image.open('ExpSample/DCOD/USAF/hologram.tif').crop(bbox).resize([256, 256])
    bg = Image.open('ExpSample/DCOD/USAF/background.tif').crop(bbox).resize([256, 256])
    processed_img = prepross_bg(np.array(img),np.array(bg)).astype(np.float32)
    plt.imshow(processed_img)
    plt.title("diffraction pattern")
    plt.show()
    with open('ExpSample/DCOD/USAF/params.json','r') as f:
        params = json.load(f)

    w = params['w']
    # deltax = params['deltax']*2
    # deltay = params['deltay']*2
    deltax = params['deltax']*300/256
    deltay = params['deltay']*300/256
    distance = params['distance']
    nx,ny = processed_img.shape

    diffusion_config['num_train_timesteps'] = 30
    gamma = 2
    loi_x = range(nx)
    loi_y = 60

elif testname =='DCOD_cheek':
    bbox = [74,62,74+330,62+330]
    img = Image.open('ExpSample/DCOD/Cheek cells/hologram.tif').crop(bbox).resize([256, 256])
    bg = Image.open('ExpSample/DCOD/Cheek cells/background.tif').crop(bbox).resize([256, 256])
    processed_img = prepross_bg(np.array(img),np.array(bg)).astype(np.float32)
    plt.imshow(processed_img)
    plt.title("diffraction pattern")
    plt.show()
    with open('ExpSample/DCOD/Cheek cells/params.json','r') as f:
        params = json.load(f)

    w = params['w']
    # deltax = params['deltax']*2
    # deltay = params['deltay']*2
    deltax = params['deltax']*330/256
    deltay = params['deltay']*330/256
    distance = params['distance']
    nx,ny = processed_img.shape
    diffusion_config['num_train_timesteps'] = 30
    gamma = 2
    loi_x = 125
    loi_y = range(ny)

elif testname =='DIH':
    bbox = [50,160,50+768,160+768]
    img = Image.open('ExpSample/DIH/Image1.bmp').crop(bbox).resize([256, 256])
    # bg = Image.open('ExpSample/DIH/params.json').crop(bbox).resize([256, 256])
    # processed_img = prepross_bg(np.array(img),np.array(bg)).astype(np.float32)
    processed_img = np.array(img)/np.array(img).max().astype(np.float32)
    plt.imshow(processed_img)
    plt.title("diffraction pattern")
    plt.show()
    with open('ExpSample/DIH/params.json','r') as f:
        params = json.load(f)

    w = params['w']
    # deltax = params['deltax']*2
    # deltay = params['deltay']*2
    deltax = params['deltax']*768/256
    deltay = params['deltay']*768/256
    distance = params['distance']
    nx,ny = processed_img.shape
    diffusion_config['num_train_timesteps'] = 30
    gamma = 2
    loi_x = 125
    loi_y = range(ny)

elif testname =='myUSAF':
    bbox = [296,710,296+256,710+256]
    img = Image.open('ExpSample/my/YP0601/group2/hologram2.tif').crop(bbox)
    bg = Image.open('ExpSample/my/YP0601/group2/background.tif').crop(bbox)
    processed_img = prepross_bg(np.array(img),np.array(bg)).astype(np.float32)
    # processed_img = np.array(img)/np.array(img).max().astype(np.float32)
    plt.imshow(processed_img)
    plt.title("diffraction pattern")
    plt.show()
    with open('ExpSample/my/YP0601/group2/params.json','r') as f:
        params = json.load(f)

    w = params['w']
    # deltax = params['deltax']*2
    # deltay = params['deltay']*2
    deltax = params['deltax']
    deltay = params['deltay']
    distance = params['distance']
    nx,ny = processed_img.shape
    diffusion_config['num_train_timesteps'] = 30
    gamma = 2
    loi_x = 130
    loi_y = range(ny)

elif testname == 'myUSAF2':
    bbox = [0,150,720,150+720]
    img = Image.open('ExpSample/my/YP0601/group2/hologram.tif').crop(bbox).resize([256, 256])
    bg = Image.open('ExpSample/my/YP0601/group2/background.tif').crop(bbox).resize([256, 256])
    processed_img = prepross_bg(np.array(img),np.array(bg)).astype(np.float32)
    # processed_img = np.array(img)/np.array(img).max().astype(np.float32)
    plt.imshow(processed_img)
    plt.title("diffraction pattern")
    plt.show()
    with open('ExpSample/my/YP0601/group2/params.json','r') as f:
        params = json.load(f)

    w = params['w']
    # deltax = params['deltax']*2
    # deltay = params['deltay']*2
    deltax = params['deltax']*720/256
    deltay = params['deltay']*720/256
    distance = params['distance']
    nx,ny = processed_img.shape
    diffusion_config['num_train_timesteps'] = 30
    gamma = 2

    loi_x = 150
    loi_y = range(ny)

prop_kernel = dict(
    wavelength = w,
    deltax = deltax,
    deltay = deltay,
    distance =distance,
    nx = nx,
    ny = ny,
    pad_size = [256,256]
)
nx_extend = 256
ny_extend = 256
pad_size = [nx_extend,ny_extend]


# ---- forward and backward propagation -----
A = generate_otf_torch(**prop_kernel)
holo = torch.from_numpy(processed_img)
# holo = norm_tensor(holo)
holo = holo / torch.max(holo)

AT = torch.conj(A)
rec = ifft2(torch.multiply(AT, fft2(holo)))
rec = torch.abs(rec)
rec = norm_tensor(rec)

# ---- loading the diffusion model -----
repo_id = "google/ddpm-celebahq-256"
# model = UNet2DModel.from_pretrained(repo_id)
model = UNet2DModel.from_pretrained("diffuser_celeB_dh")
# scheduler = DDPMScheduler.from_config(repo_id)
scheduler = DDPMScheduler.from_config(diffusion_config)

# ---- define the diffuer gradient-based reconstruction model -----
diffuser = diffuser_GB_DH(model,scheduler,prop_kernel,device)
# holo = holo.expand(1,3,nx,ny).to(device)
holo = holo.expand(1,1,nx,ny).to(device)
#
# test = gt_intensity + 0.2*torch.randn(gt_intensity.shape)
# test = test.expand(1,3,nx,ny).to(device)
# display_sample(test,0)

out = diffuser.reconstruction(holo, gamma=gamma,visual_check=20)
out = norm_tensor(out)


fig, ax = plt.subplots(2, 2)
ax[0,0].imshow(holo[0,0,:,:].cpu().numpy(), cmap='gray')
ax[1,0].imshow(rec.cpu().numpy(), cmap='gray')
ax[1,0].set_title('BP')
ax[1,1].imshow(out[0,0,:,:].cpu().numpy(), cmap='gray')
ax[1,1].set_title('out')
fig.show()
fig, ax = plt.subplots(2, 2)
ax[0,0].imshow(holo[0,0,:,:].cpu().numpy())
ax[1,0].imshow(rec.cpu().numpy())
ax[1,0].set_title('BP')
ax[1,1].imshow(out[0,0,:,:].cpu().numpy())
ax[1,1].set_title('out')
y1 = rec[loi_x,loi_y]
y1 = (y1-y1.min())/(y1.max()-y1.min())
y2 = out[0,0,loi_x,loi_y].cpu()
y2 = (y2-y2.min())/(y2.max()-y2.min())
x = range(len(y1))
ax[0,1].plot(x,y1,label='BP')
ax[0,1].plot(x,y2,label='out')
ax[0,1].legend()
plt.savefig(out_dir+'/out_compare.png')
fig.show()
plt.imsave(out_dir+'/out.png',out[0,0,:,:].cpu().numpy())
plt.imsave(out_dir+'/out_gray.png',out[0,0,:,:].cpu().numpy(),cmap='gray')
# fig, ax = plt.subplots(2, 2)
# ax[0,0].imshow(holo[0,0,:,:].cpu().numpy())
# ax[1,0].imshow(rec.cpu().numpy())
# ax[1,0].set_title(('BP \n PSNR{:.2f}').format(psnr(rec.cpu(), gt_intensity.cpu()).numpy()))
# ax[1,1].imshow(out[0,0,:,:].cpu().numpy())
# ax[1,1].set_title(('out\n PSNR{:.2f}').format(psnr(out[0,0,:,:].cpu(), gt_intensity.cpu()).numpy()))
# fig.show()
