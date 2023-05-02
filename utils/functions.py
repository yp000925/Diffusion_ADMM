import numpy as np
import matplotlib.pyplot as plt
import os
from math import pi, sqrt
from torch.fft import fft2,ifft2,fftshift,ifftshift
import torch
import torchvision
import PIL.Image as Image

def generate_otf_torch(wavelength, nx, ny, deltax,deltay, distance):
    """
    Generate the otf from [0,pi] not [-pi/2,pi/2] using torch
    :param wavelength:
    :param nx:
    :param ny:
    :param deltax:
    :param deltay:
    :param distance:
    :return:
    """
    r1 = torch.linspace(-nx/2,nx/2-1,nx)
    c1 = torch.linspace(-ny/2,ny/2-1,ny)
    deltaFx = 1/(nx*deltax)*r1
    deltaFy = 1/(nx*deltay)*c1
    mesh_qx, mesh_qy = torch.meshgrid(deltaFx,deltaFy)
    k = 2*torch.pi/wavelength
    otf = np.exp(1j*k*distance*torch.sqrt(1-wavelength**2*(mesh_qx**2
                                                           +mesh_qy**2)))
    otf = torch.fft.ifftshift(otf)
    return otf

def generate_otf(wavelength, nx, ny, deltax,deltay, distance):
    r1 = np.linspace(-nx/2,nx/2-1,nx)
    c1 = np.linspace(-ny/2,ny/2-1,ny)
    deltaFx = 1/(nx*deltax)*r1
    deltaFy = 1/(nx*deltay)*c1
    meshgrid = np.meshgrid(deltaFx,deltaFy)
    k = 2*np.pi/wavelength
    otf = np.exp(1j*k*distance*np.sqrt(1-np.power(wavelength*meshgrid[0],2)
                                       -np.power(wavelength*meshgrid[1],2)))
    otf = np.fft.fftshift(otf)
    return otf

def batch_FT2d(a_tensor):# by default FFTs the last two dimensions
    assert len(a_tensor.shape)==4, "expected dimension is 4 with batch size at first"
    return ifftshift(fft2(fftshift(a_tensor,dim = [2,3])),dim=[2,3])
def batch_iFT2d(a_tensor):
    assert len(a_tensor.shape)==4, "expected dimension is 4 with batch size at first"
    return ifftshift(ifft2(fftshift(a_tensor,dim=[2,3])),dim= [2,3])


def center_crop(H, size):
    batch, channel, Nh, Nw = H.size()

    return H[:, :, (Nh - size)//2 : (Nh+size)//2, (Nw - size)//2 : (Nw+size)//2]

def random_crop(H, size):

    batch, channel, Nh, Nw = H.size()

    x_off = int(np.floor(np.random.rand() * (Nh-size)) + size/2)
    y_off = int(np.floor(np.random.rand() * (Nw-size)) + size/2)

    return H[:, :, (x_off - size//2) : (x_off+size//2), (y_off - size//2) : (y_off+size//2)]

def center_crop_numpy(H, size):
    Nh = H.shape[0]
    Nw = H.shape[1]

    return H[(Nh - size)//2 : (Nh+size)//2, (Nw - size)//2 : (Nw+size)//2]

def amp_pha_generate(real, imag):
    field = real + 1j*imag
    amplitude = np.abs(field)
    phase = np.angle(field)

    return amplitude, phase

def make_path(path):
    import os
    if not os.path.isdir(path):
        os.mkdir(path)

def tensor2fig(tensor):
    return tensor[0,0,:,:].cpu().detach().numpy()
def save_fig_(save_path, result_data, args):

    holo, fake_holo, real_amplitude, fake_amplitude, real_phase, fake_phase, real_distance, fake_distance = result_data

    fig2 = plt.figure(2, figsize=[12, 8])

    plt.subplot(2, 3, 1)
    plt.title('input holography')
    plt.imshow(holo, cmap='gray', vmax=0.5, vmin=0)
    plt.axis('off')
    plt.subplot(2, 3, 2)
    plt.title('ground truth %dmm'%real_distance)
    plt.imshow(real_amplitude, cmap='gray', vmax=1, vmin=0)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(2, 3, 3)
    plt.title('output ' + str(np.round(fake_distance, 2)) + 'mm')
    plt.imshow(fake_amplitude, cmap='gray', vmax=1, vmin=0)
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 3, 4)
    plt.title('generated_holography')
    plt.imshow(fake_holo, cmap='gray', vmax=0.5, vmin=0)
    plt.axis('off')
    plt.subplot(2, 3, 5)
    plt.title('ground truth phase')
    plt.imshow(real_phase, cmap='hot', vmax=2.5, vmin=-0.1)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(2, 3, 6)
    plt.title('output phase')
    plt.imshow(fake_phase, cmap='hot', vmax=2.5, vmin=-0.1)
    plt.axis('off')
    plt.colorbar()

    # fig_save_name = os.path.join(p, 'test' + str(b + 1) + '.png')
    fig2.savefig(save_path)
    plt.close(fig2)

# ---- calculating PSNR (dB) of x -----


# def psnr(x,im_orig):
#     xout = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
#     norm1 = torch.sum(im_orig ** 2)
#     norm2 = torch.sum((im_orig - xout) ** 2)
#     return 10 * torch.log10(norm1 / norm2)

def norm_tensor(x):
    return (x-torch.min(x))/(torch.max(x)-torch.min(x))

# def psnr(x,im_orig):
#     xout = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
#     norm1 = torch.sum(im_orig ** 2)
#     norm2 = torch.sum((im_orig - xout) ** 2)
#     return 10 * torch.log10(norm1 / norm2)

def psnr(x,im_orig):
    x = norm_tensor(x)
    # im_orig = norm_tensor(im_orig)
    mse = torch.mean(torch.square(im_orig - x))
    psnr = torch.tensor(10.0)* torch.log10(1/ mse)
    return psnr

# def psnr(x,im_orig):
#     xout = (x - np.min(x)) / (np.max(x) - np.min(x))
#     norm1 = np.sum((np.absolute(im_orig)) ** 2)
#     norm2 = np.sum((np.absolute(x - im_orig)) ** 2)
#     psnr = 10 * np.log10( norm1 / norm2 )
#     return psnr

def gray_to_rgb(img):
    if not torch.is_tensor(img):
        img = torch.tensor(img)
    if len(img.shape) ==4:
        # means [B,1,H,W]
        expanded_arr = img.repeat(1,3,1,1)
    else:
        if len(img.shape) == 2:
            img=img.unsqueeze(0)
        if len(img.shape) == 3:
            if img.shape[-1] == 1:
                img = img.transpose([1,2,0])
            else:
                if img.shape[0] != 1:
                    print("The input shape should be [1,H,W] while received",img.shape)
                    raise ValueError
        expanded_arr = img.repeat(3,1,1)
    return expanded_arr

def rgb_to_gray(arr_in):
    if torch.is_tensor(arr_in):
        arr_in = arr_in.numpy()
    if len(arr_in.shape) ==4:
        # means [B,3,H,W]
        arr_in = arr_in.squeeze(0)
    if arr_in.shape[0]==3:
        arr_in = arr_in.transpose([1,2,0])
    img_arr = np.array(arr_in)/np.max(np.array(arr_in))
    img = Image.fromarray((img_arr*255).astype(np.uint8)).convert('L') # Image.fromarray 三通道的时候必须为unit8 类型
    arr_out=torchvision.transforms.ToTensor()(img)
    return arr_out

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def forward_propagation(x, A):
    out = torch.fft.ifft2(torch.multiply(torch.fft.fft2(x), A))
    return out


def prepross_bg(img,bg):
    temp = img/bg
    out = (temp-np.min(temp))/(1-np.min(temp))
    return out

if __name__=="__main__":
    arr = torch.rand([1,5,5])
    arr_exp = gray_to_rgb(arr)
    arr_shrink = rgb_to_gray(arr_exp)
