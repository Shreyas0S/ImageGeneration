import os, sys
import os.path as osp
import time
import random
import argparse
import numpy as np
from PIL import Image
import pprint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image,make_grid
# Safe TensorBoard import: try torch.utils, then tensorboardX, else no-op
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    try:
        from tensorboardX import SummaryWriter  # type: ignore
    except Exception:
        SummaryWriter = None  # type: ignore
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data.distributed import DistributedSampler
import multiprocessing as mp
# Optional plotting
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
import shutil

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from lib.utils import mkdir_p,get_rank,merge_args_yaml,get_time_stamp,save_args
from lib.utils import load_models_opt,load_models,save_models_opt,save_models,load_npz,params_count
from lib.perpare import prepare_dataloaders,prepare_models
from lib.modules import sample_one_batch as sample, test as test, train as train, validate as validate
from lib.datasets import get_fix_data


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Text2Img')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='../cfg/coco.yml',
                        help='optional config file')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers(default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--stamp', type=str, default='normal',
                        help='the stamp of model')
    parser.add_argument('--pretrained_model_path', type=str, default='model',
                        help='the model for training')
    parser.add_argument('--log_dir', type=str, default='new',
                        help='file path to log directory')
    parser.add_argument('--model', type=str, default='GALIP',
                        help='the model for training')
    parser.add_argument('--state_epoch', type=int, default=100,
                        help='state epoch')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--train', type=str, default='True',
                        help='if train model')
    parser.add_argument('--mixed_precision', type=str, default='False',
                        help='if use multi-gpu')
    parser.add_argument('--multi_gpus', type=str, default='False',
                        help='if use multi-gpu')
    parser.add_argument('--gpu_id', type=int, default=1,
                        help='gpu id')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--random_sample', action='store_true',default=True, 
                        help='whether to sample the dataset with random sampler')
    parser.add_argument('--early_stop_patience', type=int, default=50,
                        help='stop training if no validation improvement for this many epochs')
    parser.add_argument('--use_lora', action='store_true', default=False,
                        help='enable LoRA fine-tuning: freeze most generator weights and train only LoRA/adapter params')
    parser.add_argument('--lora_keywords', type=str, default='lora,adapter,lora_',
                        help='comma-separated keywords to match LoRA/adapter parameter names (case-insensitive)')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='max norm for gradient clipping (0 to disable)')
    args = parser.parse_args()
    return args


def main(args):
    time_stamp = get_time_stamp()
    stamp = '_'.join([str(args.model),'nf'+str(args.nf),str(args.stamp),str(args.CONFIG_NAME),str(args.imsize),time_stamp])
    args.model_save_file = osp.join(ROOT_PATH, 'saved_models', str(args.CONFIG_NAME), stamp)
    log_dir = args.log_dir
    if log_dir == 'new':
        log_dir = osp.join(ROOT_PATH, 'logs/{0}'.format(osp.join(str(args.CONFIG_NAME), 'train', stamp)))
    args.img_save_dir = osp.join(ROOT_PATH, 'imgs/{0}'.format(osp.join(str(args.CONFIG_NAME), 'train', stamp)))
    # where to save periodic samples and loss curves (inside code directory as requested)
    saved_images_root = osp.join(ROOT_PATH, 'saved_images')
    saved_images_dir = osp.join(saved_images_root, str(args.CONFIG_NAME), stamp)
    # expose saved_images_dir to modules for debug dumps
    args.saved_images_dir = saved_images_dir
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        mkdir_p(osp.join(ROOT_PATH, 'logs'))
        mkdir_p(args.model_save_file)
        mkdir_p(args.img_save_dir)
        mkdir_p(saved_images_dir)
    # prepare TensorBoard: fall back to a no-op writer if TB import failed
    class _NoOpWriter:
        def add_scalar(self, *args, **kwargs):
            pass
        def add_image(self, *args, **kwargs):
            pass
        def close(self):
            pass
    if (args.multi_gpus==True) and (get_rank() != 0):
        writer = _NoOpWriter()
    else:
        if SummaryWriter is None:
            print("Warning: TensorBoard not available. Proceeding without logging.")
            writer = _NoOpWriter()
        else:
            writer = SummaryWriter(log_dir)
    # prepare dataloader, models, data
    train_dl, valid_dl ,train_ds, valid_ds, sampler = prepare_dataloaders(args)
    CLIP4trn, CLIP4evl, image_encoder, text_encoder, netG, netD, netC = prepare_models(args)
    # LoRA fine-tuning: freeze generator except LoRA/adapter params if requested
    if getattr(args, 'use_lora', False):
        total = sum(p.numel() for p in netG.parameters())
        # freeze all generator params
        for p in netG.parameters():
            p.requires_grad = False
        # unfreeze LoRA/adapter params by name (configurable via --lora_keywords)
        raw = getattr(args, 'lora_keywords', 'lora,adapter,lora_')
        lora_keywords = [k.strip().lower() for k in raw.split(',') if k.strip()]
        if len(lora_keywords) == 0:
            lora_keywords = ['lora', 'adapter', 'lora_']
        lora_params = []
        for name, p in netG.named_parameters():
            if any(k in name.lower() for k in lora_keywords):
                p.requires_grad = True
                lora_params.append(p)
        lora_count = sum(p.numel() for p in lora_params)
        print(f"LoRA mode enabled. Generator total params: {total}, LoRA params trainable: {lora_count}")
        # fallback: if no LoRA params found, warn and leave generator trainable
        if len(lora_params) == 0:
            print("Warning: no LoRA/adapter parameters found in netG - disabling LoRA mode.")
            for p in netG.parameters():
                p.requires_grad = True
            lora_params = list(netG.parameters())
    else:
        lora_params = list(netG.parameters())
    print('**************G_paras: ',params_count(netG))
    print('**************D_paras: ',params_count(netD)+params_count(netC))
    fixed_img, fixed_sent, fixed_words, fixed_z = get_fix_data(train_dl, valid_dl, text_encoder, args)
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        fixed_grid = make_grid(fixed_img.cpu(), nrow=8, normalize=True)
        #writer.add_image('fixed images', fixed_grid, 0)
        img_name = 'gt.png'
        img_save_path = osp.join(args.img_save_dir, img_name)
        vutils.save_image(fixed_img.data, img_save_path, nrow=8, normalize=True)
    # prepare optimizer
    D_params = list(netD.parameters()) + list(netC.parameters())
    optimizerD = torch.optim.Adam(D_params, lr=args.lr_d, betas=(0.0, 0.9))
    # use only LoRA params for generator optimizer if LoRA mode enabled
    optimizerG = torch.optim.Adam(lora_params, lr=args.lr_g, betas=(0.0, 0.9))
    # LR schedulers: reduce LR if validation loss plateaus for 10 epochs
    schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizerG, mode='min', patience=10, factor=0.5, threshold=1e-4, verbose=True
    )
    schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizerD, mode='min', patience=10, factor=0.5, threshold=1e-4, verbose=True
    )
    if args.mixed_precision==True:
        scaler_D = torch.cuda.amp.GradScaler(growth_interval=args.growth_interval)
        scaler_G = torch.cuda.amp.GradScaler(growth_interval=args.growth_interval)
    else:
        scaler_D = None
        scaler_G = None
    m1 = s1 = None
    if getattr(args, 'npz_path', '') and osp.isfile(args.npz_path):
        m1, s1 = load_npz(args.npz_path)
    else:
        if (args.multi_gpus==True) and (get_rank() != 0):
            None
        else:
            print('Warning: npz_path not set or file not found. Skipping FID computation.')
    start_epoch = 1
    # load from checkpoint
    # Case 1: user provided a direct checkpoint file path (e.g., pre_coco.pth) without optimizer states
    if osp.isfile(args.pretrained_model_path) and args.pretrained_model_path.endswith('.pth'):
        try:
            netG, netD, netC = load_models(netG, netD, netC, args.pretrained_model_path)
            print(f"Loaded initial weights from file: {args.pretrained_model_path}")
        except Exception as e:
            print(f"Warning: failed to load models from file '{args.pretrained_model_path}': {e}")
    # Case 2: legacy directory + state_epoch pattern with optimizer states
    elif args.state_epoch!=1:
        start_epoch = args.state_epoch + 1
        path = osp.join(args.pretrained_model_path, 'state_epoch_%03d.pth'%(args.state_epoch))
        netG, netD, netC, optimizerG, optimizerD = load_models_opt(netG, netD, netC, optimizerG, optimizerD, path, args.multi_gpus)
    # print args
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        pprint.pprint(args)
        arg_save_path = osp.join(log_dir, 'args.yaml')
        save_args(arg_save_path, args)
        print("Start Training")
    # Start training
    test_interval,gen_interval,save_interval = args.test_interval,args.gen_interval,args.save_interval
    #torch.cuda.empty_cache()
    # start_epoch = 1
    # Track losses for plotting
    train_g_hist, train_d_hist, val_g_hist, val_d_hist = [], [], [], []
    # Track best/previous validation totals for early overfit checkpointing
    best_val_total = float('inf')
    prev_val_total = float('inf')
    no_improve_epochs = 0
    for epoch in range(start_epoch, args.max_epoch, 1):
        if (args.multi_gpus==True):
            sampler.set_epoch(epoch)
        start_t = time.time()
        # training
        args.current_epoch = epoch
        torch.cuda.empty_cache()
        avg_g_loss, avg_d_loss = train(train_dl, netG, netD, netC, text_encoder, image_encoder, optimizerG, optimizerD, scaler_G, scaler_D, args)
        print(f"Epoch {epoch}: Train G Loss={avg_g_loss:.4f}, D Loss={avg_d_loss:.4f}")
        # validation 80/20 split
        val_g_loss, val_d_loss = validate(valid_dl, netG, netD, netC, text_encoder, image_encoder, args)
        print(f"Epoch {epoch}: Val   G Loss={val_g_loss:.4f}, D Loss={val_d_loss:.4f}")
        val_total = float(val_g_loss) + float(val_d_loss)
        # record
        train_g_hist.append(float(avg_g_loss))
        train_d_hist.append(float(avg_d_loss))
        val_g_hist.append(float(val_g_loss))
        val_d_hist.append(float(val_d_loss))
        # Step LR schedulers based on validation total loss
        try:
            schedulerG.step(val_total)
            schedulerD.step(val_total)
            if (args.multi_gpus==False) or (get_rank()==0):
                print(f"Current LRs -> G: {optimizerG.param_groups[0]['lr']:.6g}, D: {optimizerD.param_groups[0]['lr']:.6g}")
        except Exception:
            pass
        # save a CSV row of losses each epoch
        try:
            mkdir_p(saved_images_dir)
            csv_path = osp.join(saved_images_dir, 'losses.csv')
            write_header = not osp.exists(csv_path)
            with open(csv_path, 'a') as f:
                if write_header:
                    f.write('epoch,train_g,train_d,val_g,val_d\n')
                f.write(f'{epoch},{avg_g_loss},{avg_d_loss},{val_g_loss},{val_d_loss}\n')
        except Exception:
            pass
        # plot and save loss curves each epoch if matplotlib is available
        if plt is not None and (args.multi_gpus==False or get_rank()==0):
            try:
                epochs = list(range(start_epoch, epoch + 1))
                plt.figure(figsize=(8,5))
                plt.plot(epochs, train_g_hist, label='G Train')
                plt.plot(epochs, val_g_hist, label='G Val')
                plt.plot(epochs, train_d_hist, label='D Train')
                plt.plot(epochs, val_d_hist, label='D Val')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training and Validation Loss')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.3)
                mkdir_p(saved_images_dir)
                plt.tight_layout()
                plt.savefig(osp.join(saved_images_dir, 'loss_curves.png'))
                plt.close()
            except Exception as _:
                pass
        torch.cuda.empty_cache()
        # save
        if epoch%save_interval==0:
            save_models_opt(netG, netD, netC, optimizerG, optimizerD, epoch, args.multi_gpus, args.model_save_file)
            torch.cuda.empty_cache()
        # additionally save every 10 epochs as requested
        # if epoch % 10 == 0:
        #     save_models_opt(netG, netD, netC, optimizerG, optimizerD, epoch, args.multi_gpus, args.model_save_file)
        # early overfit handling and best checkpointing (rank 0 only)
        # if (args.multi_gpus==False) or (get_rank()==0):
        #     # If validation increased vs previous epoch, persist the previous epoch snapshot
        #     if prev_val_total < float('inf') and (val_total > prev_val_total + 1e-6):
        #         prev_epoch = epoch - 1
        #         prev_tmp = osp.join(args.model_save_file, 'last_epoch_tmp.pth')
        #         pre_inc_path = osp.join(args.model_save_file, f'pre_increase_epoch_{prev_epoch:03d}.pth')
        #         try:
        #             if osp.exists(prev_tmp) and (not osp.exists(pre_inc_path)):
        #                 shutil.copyfile(prev_tmp, pre_inc_path)
        #         except Exception:
        #             pass
        #     # Save best-by-validation checkpoint and reset patience counter
        #     improved = False
        #     if val_total < best_val_total - 1e-6:
        #         best_val_total = val_total
        #         # Ensure we have an epoch-specific checkpoint to mark as best
        #         epoch_ckpt = osp.join(args.model_save_file, f'state_epoch_{epoch:03d}.pth')
        #         if not osp.exists(epoch_ckpt):
        #             # Save one now
        #             save_models_opt(netG, netD, netC, optimizerG, optimizerD, epoch, args.multi_gpus, args.model_save_file)
        #         best_alias = osp.join(args.model_save_file, 'best_val.pth')
        #         try:
        #             shutil.copyfile(epoch_ckpt, best_alias)
        #         except Exception:
        #             pass
        #         improved = True
        #     # Early stopping: increment patience if not improved
        #     if improved:
        #         no_improve_epochs = 0
        #     else:
        #         no_improve_epochs += 1
        #     if no_improve_epochs >= getattr(args, 'early_stop_patience', 30):
        #         try:
        #             marker = osp.join(args.model_save_file, 'EARLY_STOPPED.txt')
        #             with open(marker, 'w') as f:
        #                 f.write(f'Early stopped at epoch {epoch} after {no_improve_epochs} epochs without validation improvement.\n')
        #                 f.write(f'Best validation total loss: {best_val_total}\n')
        #         except Exception:
        #             pass
        #         print(f"Early stopping triggered at epoch {epoch} (no improvement for {no_improve_epochs} epochs)")
        #         break
        # sample
        # if epoch%gen_interval==0:
        #     sample(fixed_z, fixed_sent, netG, args.multi_gpus, epoch, args.img_save_dir, writer)
        #     torch.cuda.empty_cache()
        # generate and store validation-only samples every 15 epochs under code/saved_images
        if (epoch % 30 == 0) and ((args.multi_gpus==False) or (get_rank()==0)):
            try:
                B = fixed_z.size(0)
                val_noise = fixed_z[B//2:]
                val_sent = fixed_sent[B//2:]
                with torch.no_grad():
                    netG.eval()
                    val_imgs = netG(val_noise, val_sent, eval=True).float()
                    val_imgs = torch.clamp(val_imgs, -1., 1.)
                mkdir_p(saved_images_dir)
                out_path = osp.join(saved_images_dir, f'val_samples_epoch_{epoch:03d}.png')
                vutils.save_image(val_imgs.data, out_path, nrow=8, value_range=(-1, 1), normalize=True)
            except Exception as _:
                pass
        # test
        if epoch%test_interval==0 and (m1 is not None and s1 is not None):
            FID, TI_score = test(valid_dl, text_encoder, netG, CLIP4evl, args.device, m1, s1, epoch, args.max_epoch, args.sample_times, args.z_dim, args.batch_size)
            torch.cuda.empty_cache()
        if (args.multi_gpus==True) and (get_rank() != 0):
            None
        else:
            if epoch%test_interval==0 and (m1 is not None and s1 is not None):
                writer.add_scalar('FID', FID, epoch)
                writer.add_scalar('CLIP_Score', TI_score, epoch)
                writer.add_scalar('Loss/G_Train', avg_g_loss, epoch)
                writer.add_scalar('Loss/D_Train', avg_d_loss, epoch)
                writer.add_scalar('Loss/G_Val', val_g_loss, epoch)
                writer.add_scalar('Loss/D_Val', val_d_loss, epoch)
                print('The %d epoch FID: %.2f, CLIP_Score: %.2f' % (epoch,FID,TI_score*100))
            else:
                # Always log losses even if FID is skipped
                writer.add_scalar('Loss/G_Train', avg_g_loss, epoch)
                writer.add_scalar('Loss/D_Train', avg_d_loss, epoch)
                writer.add_scalar('Loss/G_Val', val_g_loss, epoch)
                writer.add_scalar('Loss/D_Val', val_d_loss, epoch)
            end_t = time.time()
            print('The epoch %d costs %.2fs'%(epoch, end_t-start_t))
            print('*'*40)
        # rolling tmp snapshot for previous-epoch recovery; save after logging
        if (args.multi_gpus==False) or (get_rank()==0):
            tmp_path = osp.join(args.model_save_file, 'last_epoch_tmp.pth')
            try:
                state = {'model': {'netG': netG.state_dict(), 'netD': netD.state_dict(), 'netC': netC.state_dict()},
                         'optimizers': {'optimizer_G': optimizerG.state_dict(), 'optimizer_D': optimizerD.state_dict()},
                         'epoch': epoch}
                torch.save(state, tmp_path)
            except Exception:
                pass
        prev_val_total = val_total



if __name__ == "__main__":
    args = merge_args_yaml(parse_args())
    # set seed
    if args.manual_seed is None:
        args.manual_seed = 100
        #args.manualSeed = random.randint(1, 10000)
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if args.cuda:
        if args.multi_gpus:
            torch.cuda.manual_seed_all(args.manual_seed)
            torch.distributed.init_process_group(backend="nccl")
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            args.device = torch.device("cuda", local_rank)
            args.local_rank = local_rank
        else:
            torch.cuda.manual_seed_all(args.manual_seed)
            torch.cuda.set_device(args.gpu_id)
            args.device = torch.device("cuda")
    else:
        args.device = torch.device('cpu')
    main(args)

