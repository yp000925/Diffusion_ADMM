import torch
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
from utils.layer import psnr_metric, l2_loss
from models.Unet import Unet
import torch.optim as optim
from dataset.HoloData import create_dataloader
import matplotlib.pyplot as plt
import time
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(message)s", level=logging.INFO)
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

bz = 10
lr_init = 0.0001
train_path = './imgdata/train'
eval_path = './imgdata/eval'
train_path = '/mnt/disk/zhangyp/datasets/BSR500/train'
eval_path = '/mnt/disk/zhangyp/datasets/BSR500/val'
epochs = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
directory = './pre_train_exp'
visualization = True



if not os.path.exists(directory):
    os.mkdir(directory)


def train_loop(model, dataloader, criterion, optimizer, device, epoch_idx):
    model.train()
    pbar = tqdm(enumerate(dataloader),total=len(dataloader))
    nb = len(dataloader)
    epoch_loss = 0
    avg_psnr = 0
    logger.info("\n Train ===============================================")
    logger.info(('\n' + '%10s' * 4) % ('Epoch', 'memory', 'l2-loss', 'PSNR'))
    optimizer.zero_grad()

    for batch_i, (x, y) in pbar:
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        loss = criterion(y_pred=output, y_true=y)
        loss.backward()
        optimizer.step()
        _batch_psnr, _avg_psnr = psnr_metric(y_true=y, y_pred=output)

        if device == "cpu":
            epoch_loss += loss.detach().numpy()
            avg_psnr += _avg_psnr.detach().numpy()
        else:
            epoch_loss += loss.cpu().detach().numpy()
            avg_psnr += _avg_psnr.cpu().detach().numpy()
        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
        info = ('%10s' * 2 + '%10.4g' * 2) % (
            '%g' % (epoch_idx), mem, epoch_loss / (batch_i + 1), avg_psnr / (batch_i + 1)
        )
        pbar.set_description(info)
        if epoch_loss <= 0.001:
            print("small")
            raise ValueError

    return epoch_loss, avg_psnr/ (batch_i + 1)


def eval_epoch(model, dataloader, device, criterion):
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    epoch_loss = 0
    avg_psnr = 0

    logger.info("\n Evaluation ===============================================")
    logger.info(('\n' + '%10s' * 2) % ('l2-loss', 'PSNR'))
    model.eval()
    with torch.no_grad():
        for batch_i, (x, y) in pbar:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = criterion(y_pred=output, y_true=y)
            _batch_psnr, _avg_psnr = psnr_metric(y_true=y, y_pred=output)

            if device == "cpu":
                epoch_loss += loss.numpy()
                avg_psnr += _avg_psnr.numpy()
            else:
                epoch_loss += loss.cpu().numpy()
                avg_psnr += _avg_psnr.cpu().numpy()

            info = ('%10.4g' * 2) % (loss / (batch_i + 1), avg_psnr / (batch_i + 1))
            pbar.set_description(info)
    return epoch_loss, avg_psnr/ (batch_i + 1)

def visual_after_epoch(model,dataloader,criterion, path):
    model.eval()
    with torch.no_grad():
        fig,ax = plt.subplots(3,min(4, dataloader.batch_size))

        for i, (x,y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = criterion(y_pred=output, y_true=y)
            _batch_psnr, _avg_psnr = psnr_metric(y_true=y, y_pred=output)

            for j in range(min(4, dataloader.batch_size)):
                ax[0,j].imshow(y[j,0,:,:].cpu().numpy(), cmap='gray')
                ax[0,j].set_title('GT')
                ax[1,j].imshow(x[j,0,:,:].cpu().numpy(), cmap='gray')
                b_psnr, _psnr = psnr_metric(y_true=y[j,:,:,:].unsqueeze(0),y_pred=x[j,:,:,:].unsqueeze(0))
                ax[1,j].set_title('PSNR {:.4f}'.format(_psnr.cpu().numpy()))
                ax[2,j].imshow(output[j,0,:,:].cpu().numpy(), cmap='gray')
                ax[2,j].set_title('PSNR {:.4f}'.format(_batch_psnr[j]))
            break
        fig.savefig(path)

if __name__ == "__main__":

    model_name = 'Unet'

    timestr = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime())
    save_dir = os.path.join(directory, model_name+ timestr)
    log_file = save_dir +'train.log'

    formater = logging.Formatter("%(message)s")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formater)
    # define the StreamHandler for writing on the screen
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formater)
    # add both handlers
    logger.addHandler(fh)
    # logger.addHandler(sh)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    tb_writer = SummaryWriter(save_dir)
    last_pth = os.path.join(save_dir, 'last.pt')
    best_pth = os.path.join(save_dir, 'best.pt')
    best_psnr = 0
    start_epoch = 0
    end_epoch = start_epoch + epochs

    # ---- Load dataset -----
    logger.info("\n Loading the training data...")
    kernel_params = {}
    kernel_params['w'] = 632e-9
    kernel_params['deltax'] = 3.45e-6
    kernel_params['deltay'] = 3.45e-6
    kernel_params['distance'] = 0.02
    kernel_params['nx'] = 512
    kernel_params['ny'] = 512
    logger.info("\n propagation parameters", kernel_params)

    train_loader, train_dataset = create_dataloader(train_path, bz, kernel_params)
    eval_loader, eval_dataset = create_dataloader(eval_path, bz, kernel_params)
    # ----Building model -----
    logger.info("\n Building model...")
    model = Unet(in_chans=1, out_chans=1, chans=64).to(device)
    logger.info(model)

    model = torch.nn.DataParallel(model).to(device)

    # ----Setup training schedule -----
    optimizer = optim.Adam(model.parameters(), lr=lr_init)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=3, factor=0.5, threshold=0.001, verbose=True)
    scheduler.last_epoch = start_epoch - 1

    # ----Start training -----
    logger.info('\n Start training process=======================')
    for epoch_idx in range(start_epoch, end_epoch, 1):
        train_loss, train_psnr = train_loop(model=model, dataloader=train_loader, criterion=l2_loss,
                                            optimizer=optimizer, device=device, epoch_idx=epoch_idx)
        eval_loss, eval_psnr = eval_epoch(model=model, dataloader=eval_loader, device=device, criterion=l2_loss)
        scheduler.step(train_loss)
        # ----Write log-----
        current_lr = optimizer.param_groups[0]['lr']
        tags = ['learning_rate', 'train/loss', 'train/psnr', 'val/loss', 'val/psnr']
        for x, tag in zip([np.array(current_lr), train_loss, train_psnr
                              , eval_loss, eval_psnr], tags):
            if tb_writer:
                tb_writer.add_scalar(tag, x, epoch_idx)
        # save the last
        ckpt = {
            'model_state_dict': model.state_dict(),
            'last_epoch': epoch_idx,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_name': model_name,
            'loss': train_loss,
        }
        torch.save(ckpt, last_pth)
        logger.info("Epoch {:d} saved".format(epoch_idx))

        # update the best
        if eval_psnr > best_psnr:
            best_psnr = eval_psnr
        if best_psnr == eval_psnr:
            torch.save(ckpt, best_pth)
            logger.info("Epoch {:d} is the best currently".format(epoch_idx))


        if visualization and epoch_idx % 10 == 0:
            path = save_dir + '/{:d}_Epoch'.format(epoch_idx)+'.png'
            visual_after_epoch(model,eval_loader,l2_loss, path)