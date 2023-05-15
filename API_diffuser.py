import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
import torch.utils.tensorboard as tb
from utils.diffuserHelper import Denoising,Deblurring
from PIL import Image
import torchvision.utils as tvu

from utils.diffuserHelper import create_model
from utils.diffuserHelper import efficient_generalized_steps
import matplotlib.pyplot as plt
torch.set_printoptions(sci_mode=False)

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def data_transform(config, X):
    if config.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.rescaled:
        X = 2 * X - 1.0

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X

def get_beta_schedule(beta_schedule,beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                    )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def diffusion_default():
    res = dict(
        beta_schedule = "linear",
        beta_start = 0.0001,
        beta_end = 0.02,
        num_diffusion_timesteps=2000,
        sigma_0 = 0.1,
        eta = 0.85,
        etaB=1,
        timesteps=50
    )
    return dict2namespace(res)

def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    elif config.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)

def data_defaults():
    '''
    defaults for DH dataset
    '''
    res = dict(
        image_size=256,
        channels=3,
        logit_transform=False,
        uniform_dequantization=False,
        gaussian_dequantization=False,
        random_flip=False,
        num_workers=1,
        dh_dataset=True,
        rescaled = False
    )

    return dict2namespace(res)

def model_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        type="openai",
        image_size=256,
        num_channels=256,
        num_res_blocks=2,
        in_channels = 3,
        out_channels =3,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=64,
        attention_resolutions= "32,16,8",
        channel_mult="",
        dropout=0.0,
        resamp_with_conv = True,
        learn_sigma = True,
        class_cond=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        var_type='fixedsmall',
        use_fp16=False,
        use_new_attention_order=False,
        task = 'denoise'
    )
    return dict2namespace(res)

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=True,
        help="A string for documentation purpose. "
             "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--deg", type=str, required=True, help="Degradation"
    )
    parser.add_argument(
        "--sigma_0", type=float, required=True, help="Sigma_0"
    )
    parser.add_argument(
        "--eta", type=float, default=0.85, help="Eta"
    )
    parser.add_argument(
        "--etaB", type=float, default=1, help="Eta_b (before)"
    )
    parser.add_argument(
        '--subset_start', type=int, default=-1
    )
    parser.add_argument(
        '--subset_end', type=int, default=-1
    )

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
    args.image_folder = os.path.join(
        args.exp, "image_samples", args.image_folder
    )
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input(
                f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
            )
            if response.upper() == "Y":
                overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config

class diffusion():
    def __init__(self,model_pth, model_type='default', data_args=None, args=None, device=None):
        self.ckpt_pth = model_pth
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        if model_pth == None:
            ckpt = "logs/imagenet/256x256_diffusion_uncond.pt"
        else:
            ckpt = model_pth

        if model_type == "default":
            self.config_dict = model_defaults()
        else:
            self.config_dict = vars(args)

        if data_args ==None:
            self.data_args = data_defaults()
        else:
            self.data_args = data_args
        self.model  = create_model(**vars(self.config_dict))
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.model = torch.nn.DataParallel(self.model)
        if self.config_dict.task == "denoise":
            self.H_funcs = Denoising(self.data_args.channels, self.data_args.image_size, self.device)
        elif self.config_dict.task == "deblur":
            self.H_funcs = Deblurring(torch.Tensor([1/9] * 9).to(self.device), self.data_args.channels, self.data_args.image_size, self.device)
        else:
            print("Not supporting task")
            raise(ValueError)



    def denoise(self, img, diffusion_args, verbose =False):
        img = img.to(self.device)
        # check whether the channel and dimension
        img = data_transform(self.data_args, img)
        y_0 = self.H_funcs.H(img)
        y_0 = y_0 + diffusion_args.sigma_0 * torch.randn_like(y_0)
        pinv_y_0 = self.H_funcs.H_pinv(y_0).view(y_0.shape[0], self.data_args.channels, self.data_args.image_size, self.data_args.image_size)

        if verbose:
            tvu.save_image(
                inverse_data_transform(self.data_args, pinv_y_0[0]), "y0_0.png")
        ##Begin DDIM
        x = torch.randn(
            y_0.shape[0],
            self.data_args.channels,
            self.data_args.image_size,
            self.data_args.image_size,
            device=self.device,
        )
        with torch.no_grad():
            x, _ = self.sample_image(x, self.model, self.H_funcs, y_0, diffusion_args, last=False)

        x = [inverse_data_transform(self.data_args, y) for y in x]
        return x


    def sample_image(self, x, model, H_funcs, y_0, diffusion_args, last=False):
        sigma_0 = diffusion_args.sigma_0
        betas = get_beta_schedule(beta_schedule=diffusion_args.beta_schedule,
                                  beta_start=diffusion_args.beta_start,
                                  beta_end=diffusion_args.beta_end,
                                  num_diffusion_timesteps=diffusion_args.num_diffusion_timesteps,
                                  )
        betas = torch.from_numpy(betas).float().to(self.device)
        timesteps = diffusion_args.timesteps

        skip = betas.shape[0] // timesteps
        seq = range(0, betas.shape[0], skip)

        x = efficient_generalized_steps(x, seq, model, betas, H_funcs, y_0, sigma_0, \
                                        etaB=diffusion_args.etaB, etaA=diffusion_args.eta, etaC=diffusion_args.eta, cls_fn=None, classes=None)
        if last:
            x = x[0][-1]
        return x



if __name__ == "__main__":
    import time
    diffusion_args = diffusion_default()
    # output_folder = "diffusion_output"
    model_name = "diffusion_output/"
    timestr = time.strftime("%Y-%m-%d-%H_%M_%S/", time.localtime())
    output_folder = model_name+timestr
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # img_path = "exp/datasets/ood/0/orig_0.png"

    img_path ="noisy_Lena.png"
    # img_path ="diffractionpattern.png"
    prefix = img_path.split('/')[-1]
    prefix = prefix.split('.')[0]
    img = Image.open(img_path).convert('RGB')
    import torchvision.transforms as transforms
    trans = transforms.Compose([transforms.Resize(size=256),transforms.ToTensor()])
    img = trans(img).unsqueeze(0)
    stddev = 0
    mean= 0
    noise = torch.randn(img.size()) * stddev + mean
    img_with_noise = torch.clamp(img + noise, 0, 1)
    # img = torch.randn([1,3,256,256])
    denoiser = diffusion(model_pth="256x256_diffusion_uncond.pt",model_type='default')

    s = time.time()
    y = denoiser.denoise(img_with_noise,diffusion_args,verbose=True)
    e = time.time()
    print(f"Total execution time: {e-s} seconds")
    tvu.save_image(
        img_with_noise, os.path.join(output_folder, prefix +f"_std{stddev}.png")
    )
    for j in range(len(y)):
        tvu.save_image(
            y[j], os.path.join(output_folder, prefix +f"_sigma{diffusion_args.sigma_0}"+f"_{j}.png")
        )
        mse = torch.mean(( y[j].to(img.device) - img) ** 2)
        psnr = 10 * torch.log10(1 / mse)
        print(psnr)
