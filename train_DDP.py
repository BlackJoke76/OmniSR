import os
import utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import random
import time
import numpy as np
import datetime
from losses import CharbonnierLoss
import os
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler
import torch.nn.functional as F
from utils.loader import get_training_data, get_validation_data
import time
import argparse
import options
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from tensorboardX import SummaryWriter




######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='image denoising')).parse_args()
local_rank = opt.local_rank
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')
device = torch.device("cuda", local_rank)
if opt.debug == True:
    opt.eval_now = 2

######### Logs dir ###########
dir_name = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(dir_name, 'log', opt.arch+opt.env+datetime.datetime.now().isoformat(timespec='minutes'))
logname = os.path.join(log_dir, datetime.datetime.now().isoformat(timespec='minutes')+'.txt') 

result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')
tensorlog_dir  = os.path.join(log_dir, 'tensorlog')
if dist.get_rank() == 0:
    utils.mkdir(log_dir)
    utils.mkdir(result_dir)
    utils.mkdir(model_dir)
    utils.mkdir(tensorlog_dir)
    utils.mknod(logname)
    tb_logger = SummaryWriter(log_dir=tensorlog_dir)

####### just allow one process to print info to log
if dist.get_rank() == 0:
    logging.basicConfig(filename=logname,level=logging.INFO if dist.get_rank() in [-1, 0] else logging.WARN)
    torch.distributed.barrier()
else:
    torch.distributed.barrier()
    logging.basicConfig(filename=logname,level=logging.INFO if dist.get_rank() in [-1, 0] else logging.WARN)

logging.info(opt)
logging.info(f"Now time is : {datetime.datetime.now().isoformat()}")
########### Set Seeds ###########
random.seed(1234 + dist.get_rank())
np.random.seed(1234 + dist.get_rank())
torch.manual_seed(1234 + dist.get_rank())
torch.cuda.manual_seed(1234 + dist.get_rank())
torch.cuda.manual_seed_all(1234 + dist.get_rank())

def worker_init_fn(worker_id):
    random.seed(1234 + worker_id + dist.get_rank())

g = torch.Generator()
g.manual_seed(1234 + dist.get_rank())

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True


######### Model ###########
model_restoration = utils.get_arch(opt)
model_restoration.to(device)
DINO_Net = torch.hub.load('./dinov2', 'dinov2_vitl14', source='local')
logging.info(str(model_restoration) + '\n')


######### Optimizer ###########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")

######### Resume ###########
if opt.resume:
    path_chk_rest = opt.pretrain_weights
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    if dist.get_rank() == 0:
        lr = utils.load_optim(optimizer, path_chk_rest)
        utils.load_checkpoint(model_restoration,path_chk_rest)

    # new_lr = lr
    # if opt.optimizer.lower() == 'adam':
    #     optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
    # elif opt.optimizer.lower() == 'adamw':
    #     optimizer = optim.AdamW(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
    # else:
    #     raise Exception("Error optimizer...")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=5e-5)
    logging.info("------------------------------------------------------------------------------")
    logging.info(f"==> Resuming Training with learning rate:{lr}")
    logging.info("------------------------------------------------------------------------------")
    print(optimizer.param_groups[0]['lr'])

######### DDP ###########
model_restoration = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_restoration).to(device)
model_restoration = DDP(model_restoration, device_ids=[local_rank], output_device=local_rank)

DINO_Net.to(device)
DINO_Net.eval()
DINO_Net = DDP(DINO_Net, device_ids=[local_rank], output_device=local_rank)


# ######### Scheduler ###########
if not opt.resume:
    if opt.warmup:
        logging.info("Using warmup and cosine strategy!")
        warmup_epochs = opt.warmup_epochs
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=5e-5)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
        scheduler.step()
    else:
        step = 50
        logging.info("Using StepLR,step={}!".format(step))
        scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
        scheduler.step()

######### Loss ###########
criterion_restore = CharbonnierLoss().to(device)

######### DataLoader ###########
logging.info('===> Loading datasets')
img_options_train = {'patch_size':opt.train_ps}
train_dataset = get_training_data(opt.train_dir, img_options_train, opt.debug)
train_sampler = DistributedSampler(train_dataset, shuffle=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size // dist.get_world_size(), 
        num_workers=opt.train_workers, sampler=train_sampler, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn,
        generator=g )

val_dataset = get_validation_data(opt.val_dir, opt.debug)
val_sampler = DistributedSampler(val_dataset, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size= opt.batch_size // dist.get_world_size(),
        num_workers=opt.eval_workers, sampler=val_sampler, pin_memory=False, drop_last=False, worker_init_fn=worker_init_fn,
        generator=g)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
logging.info(f"Sizeof training set: {len_trainset} sizeof validation set: {len_valset}")

######### train ###########
logging.info("===> Start Epoch {} End Epoch {}".format(start_epoch,opt.nepoch))
best_psnr = 0
best_epoch = 0
best_iter = 0
logging.info("\nEvaluation after every {} Iterations !!!\n".format(opt.eval_now))

loss_scaler = NativeScaler()
torch.cuda.empty_cache()

index = 0
DINO_patch_size = 14
img_multiple_of = 8 * opt.win_size

# the train_ps must be the multiple of win_size
UpSample = nn.UpsamplingBilinear2d(
    size=((int)(opt.train_ps * DINO_patch_size / 8), 
        (int)(opt.train_ps * DINO_patch_size / 8))
    )


for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0

    epoch_direct_loss = 0
    epoch_indirect_loss = 0
    train_id = 1
    epoch_ssim_loss = 0

    train_loader.sampler.set_epoch(epoch)
    for i, data in enumerate(train_loader, 0): 
        # zero_grad
        index += 1
        optimizer.zero_grad()
        target = data[0].to(device)
        input_ = data[1].to(device)
        point = data[2].to(device)
        normal = data[3].to(device)

        with torch.cuda.amp.autocast():
            dino_mat_features = None
            with torch.no_grad():
                input_DINO = UpSample(input_)
                dino_mat_features = DINO_Net.module.get_intermediate_layers(input_DINO, 4, True)

            restored = model_restoration(input_, dino_mat_features, point, normal)
            loss_restore = criterion_restore(restored, target)
            loss = 0.9 * loss_restore


        loss_scaler(
                loss, optimizer,parameters=model_restoration.parameters())

        loss_list = utils.distributed_concat(loss, dist.get_world_size())
        loss = 0
        for ele in loss_list:
            loss += ele.item()
        epoch_loss += loss


        if dist.get_rank() == 0:
            tb_logger.add_scalar("loss", epoch_loss, epoch+1)
    ################# Evaluation ########################
    if (epoch + 1) % opt.eval_now == 0:
        eval_shadow_rmse = 0
        eval_nonshadow_rmse = 0
        eval_rmse = 0
        with torch.no_grad():
            model_restoration.eval()
            psnr_val_rgb = []
            for _, data_val in enumerate((val_loader), 0):
                target = data_val[0].to(device)
                input_ = data_val[1].to(device)
                point = data_val[2].to(device)
                normal = data_val[3].to(device)
                filenames = data[4] 
                # Pad the input if not_multiple_of win_size * 8
                height, width = input_.shape[2], input_.shape[3]
                H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                    (width + img_multiple_of) // img_multiple_of) * img_multiple_of

                padh = H - height if height % img_multiple_of != 0 else 0
                padw = W - width if width % img_multiple_of != 0 else 0
                input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
                point = F.pad(point, (0, padw, 0, padh), 'reflect')
                normal = F.pad(normal, (0, padw, 0, padh), 'reflect')
                UpSample_val = nn.UpsamplingBilinear2d(
                    size=((int)(input_.shape[2] * (DINO_patch_size / 8)), 
                        (int)( input_.shape[3] * (DINO_patch_size / 8))))


                with torch.cuda.amp.autocast():

                    # DINO_V2
                    input_DINO = UpSample_val(input_)
                    dino_mat_features = DINO_Net.module.get_intermediate_layers(input_DINO, 4, True)
                    restored = model_restoration(input_, dino_mat_features, point, normal)


                restored = torch.clamp(restored, 0.0, 1.0)
                restored = restored[:, : ,:height, :width]
                psnr_val_rgb.append(utils.batch_PSNR(restored, target, True))


            psnr_val_rgb = sum(psnr_val_rgb) / len(val_loader)
            psnr_val_rgb_list = utils.distributed_concat(psnr_val_rgb, dist.get_world_size())
            psnr_val_rgb = 0
            for ele in psnr_val_rgb_list:
                psnr_val_rgb += ele.item()

            psnr_val_rgb = psnr_val_rgb / len(psnr_val_rgb_list)

            if dist.get_rank() == 0:
                tb_logger.add_scalar("psnr", psnr_val_rgb, epoch)

            for ele in restored:
                rgb_restored = ele.cpu().numpy().squeeze().transpose((1, 2, 0))
                utils.save_img(rgb_restored * 255.0, os.path.join(result_dir, filenames[0]))

            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                best_iter = i
                if dist.get_rank() == 0:
                    torch.save({'epoch': epoch,
                                'state_dict': model_restoration.module.state_dict(),
                                'optimizer' : optimizer.state_dict()
                                }, os.path.join(model_dir,"model_best.pth"))

            logging.info("[Ep %d it %d\t PSNR SIDD: %.4f\t ] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " \
                    % (epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr))
            logging.info("Now time is : {}".format(datetime.datetime.now().isoformat()))
            model_restoration.train()
            torch.cuda.empty_cache()
    scheduler.step()
    
    logging.info("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
    if dist.get_rank() == 0:
        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.module.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_latest.pth"))   

        if epoch%opt.checkpoint == 0:
            torch.save({'epoch': epoch, 
                        'state_dict': model_restoration.module.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 
logging.info("Now time is : {}".format(datetime.datetime.now().isoformat()))





