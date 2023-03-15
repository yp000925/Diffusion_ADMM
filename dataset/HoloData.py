from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from utils.functions import generate_otf_torch
from torch.fft import fft2, ifft2
import os
from pathlib import Path
import glob
import PIL.Image as Image
from utils.layer import norm_tensor
import matplotlib.pyplot as plt


class HoloData(Dataset):
    def __init__(self, path, kernel_params):
        w = kernel_params['w']
        deltax = kernel_params['deltax']
        deltay = kernel_params['deltay']
        distance = kernel_params['distance']
        self.nx = kernel_params['nx']
        self.ny = kernel_params['ny']

        self.A = generate_otf_torch(w, self.nx, self.ny, deltax, deltay, distance)
        self.AT = generate_otf_torch(w, self.nx, self.ny, deltax, deltay, -distance)
        try:
            f = []
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)
                if p.is_dir():
                    f += glob.glob(str(p / '**' / "*.*"), recursive=True)
                else:
                    raise Exception(f'{p} does not exist')
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in ['png', 'jpg']])
            assert self.img_files, f'{path}{p}No mat found'
        except Exception as e:
            raise Exception(f'Error loading data from {path}:{e}\n')

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        path = self.img_files[idx]
        try:
            """ Load the GT intensity map and get the diffraction pattern"""
            img = Image.open(path).resize([self.nx, self.ny]).convert('L')
            gt_intensity = torch.from_numpy(np.array(img))
            gt_intensity = norm_tensor(gt_intensity)
        except:
            print("Cannot load the intensity map")
            raise ValueError
        holo = ifft2(torch.multiply(self.A, fft2(gt_intensity)))
        holo = torch.abs(holo)
        holo = norm_tensor(holo)
        rec = ifft2(torch.multiply(self.AT, fft2(holo)))
        rec = torch.abs(rec)
        rec = norm_tensor(rec)
        return rec.unsqueeze(0), gt_intensity.unsqueeze(0)


def create_dataloader(path, batch_size, kernel_params):
    dataset = HoloData(path=path, kernel_params=kernel_params)
    batch_size = min(batch_size, len(dataset))
    dataloader = DataLoader(dataset, batch_size, shuffle=False)
    return dataloader, dataset


if __name__ == "__main__":
    path = '/Users/zhangyunping/PycharmProjects/Diffusion_ADMM/imgdata'
    kernel_params = {}
    kernel_params['w'] = 632e-9
    kernel_params['deltax'] = 3.45e-6
    kernel_params['deltay'] = 3.45e-6
    kernel_params['distance'] = 0.02
    kernel_params['nx'] = 512
    kernel_params['ny'] = 512
    dataloader, dataset = create_dataloader(path, 2, kernel_params)
    for batch_i, (x, y) in enumerate(dataloader):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(x[0, 0, :, :].numpy(), cmap='gray')
        ax[0].set_title('input')
        ax[1].imshow(y[0, 0, :, :].numpy(), cmap='gray')
        ax[1].set_title('gt_intensity')
        fig.show()
        break
