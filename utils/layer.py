import torch
import math

def psnr_metric(y_true,y_pred):
    """

    :param y_true: normalized to 0-1
    :param y_pred:
    :return:
    """
    batch_mse = torch.mean(torch.square(y_pred - y_true),dim=[1,2,3])
    batch_psnr = torch.tensor(10.0)*torch.log10(1/batch_mse)
    avg_psnr = torch.mean(batch_psnr)
    return batch_psnr, avg_psnr

def l2_loss(y_true,y_pred):
    return torch.mean(torch.square((y_true-y_pred)))

def norm_tensor(x):
    return (x-torch.min(x))/(torch.max(x)-torch.min(x))

if __name__ =="__main__":
    y_pred = torch.randn([2,3,5,5])
    y_true = torch.ones([2,3,5,5])
    batch_psnr ,avg_psnr = psnr_metric(y_true,y_pred)