
import torch
import torch.nn as nn
import numpy as np

# ---- load the model based on the type and sigma (noise level) ----
def load_model(model_type, sigma,device):
    path = "Pretrained_models/" + model_type + "_noise" + str(sigma) + ".pth"
    if model_type == "DnCNN":
        from models.DnCNN import DnCNN
        net = DnCNN(channels=1, num_of_layers=17)
        model = nn.DataParallel(net).to(device)
    elif model_type == "SimpleCNN":
        from models.SimpleCNN import DnCNN
        model = DnCNN(1, num_of_layers = 4, lip = 0.0, no_bn = True).to(device)
    else:
        from models.realSN_models import DnCNN
        net = DnCNN(channels=1, num_of_layers=17)
        model = nn.DataParallel(net).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def  load_model_DHTrained(path,device):
    from models.DnCNN_v2 import  DnCNN
    net =  DnCNN()
    net.load_state_dict(torch.load(path, map_location=device))
    model = nn.DataParallel(net).to(device)
    model.eval()
    return model


if __name__ =="__main__":
    path = "/Users/zhangyunping/PycharmProjects/KAIR-master/denoising/dncnnDH/models/15000_G.pth"
    model = load_model_DHTrained(path,'cpu')
    # loader = torch.load(path, map_location='cpu')
    # from models.DnCNN_v2 import  DnCNN
    # net =  DnCNN()
    # net.load_state_dict(torch.load(path, map_location='cpu'))
