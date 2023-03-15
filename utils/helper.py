import torch
import torch.nn as nn
from torch.utils.data import Dataset
import PIL.Image as Image
import torchvision.transforms as pth_transforms
import matplotlib.pyplot as plt
import numpy as np


class Early_stop:
    def __init__(self, patience,  tol):
        self.patience = patience
        self.tol = tol
        self.counter = 0
        self.early_stop = False
        self.psnr_last = np.inf

    def __call__(self, psnr):
        update = psnr - self.psnr_last
        if update >= self.tol:
            self.counter = 0
        else:
            self.counter += 1
        self.psnr_last = psnr
        if self.counter >= self.patience:
            print("Eearly stop the training since the PSNR does not improve after {:d} iteration".format(self.patience))
            self.early_stop = True
        else:
            self.early_stop = False
        return self.early_stop
