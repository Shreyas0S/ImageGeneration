import os, sys
import os.path as osp
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import clip
import torch
import glob
import importlib
from lib.utils import choose_model


###########   preparation   ############
ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  "..", ".."))


def _maybe_load_checkpoint_into_model(model, checkpoint_info):
    """
    Attempt to load weights into a CLIP-like model from a given path or directory.
    Supports raw state_dict or wrappers with keys like 'state_dict', 'model', 'model_state_dict'.
    """
    if not checkpoint_info:
        return model
    path = checkpoint_info
    if isinstance(checkpoint_info, dict):
        # accept any of these keys
        path = checkpoint_info.get('checkpoint') or checkpoint_info.get('checkpoint_path') \
               or checkpoint_info.get('pretrained_path') or checkpoint_info.get('weights_path')
    if not path:
        return model
    # If directory, pick latest .pt/.pth
    if osp.isdir(path):
        candidates = sorted(
            glob.glob(osp.join(path, '*.pt')) +
            glob.glob(osp.join(path, '*.pth')) +
            glob.glob(osp.join(path, '*.bin')),
            key=lambda p: osp.getmtime(p), reverse=True
        )
        if not candidates:
            return model
        path = candidates[0]
    if not osp.isfile(path):
        return model
    try:
        ckpt = torch.load(path, map_location='cpu')
        if isinstance(ckpt, dict):
            # try common keys
            state = None
            for k in ['state_dict', 'model', 'model_state_dict', 'weights']:
                if k in ckpt and isinstance(ckpt[k], dict):
                    state = ckpt[k]
                    break
            if state is None:
                # maybe the dict itself is the state_dict
                state = {k: v for k, v in ckpt.items() if hasattr(v, 'shape')}
        else:
            state = None
        if state:
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"Loaded Long-CLIP weights from {path}. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        else:
            print(f"Warning: could not find a valid state dict in {path}; using model as-is.")
    except Exception as e:
        print(f"Warning: failed to load checkpoint '{path}': {e}")
    return model


def load_clip(clip_info, device):
    """
    Load a CLIP-like model. `clip_info` is expected to be a mapping with at least:
      - 'src': name of module to import ('clip' by default, or 'longclip'/'long_clip')
      - 'type': model name, e.g. 'ViT-B/32'
    Optionally accepts 'context_length' (int) which will be forwarded to loaders that support it.
    """
    src = clip_info.get('src', 'clip')
    model_name = clip_info.get('type', 'ViT-B/32')
    ctx_len = clip_info.get('context_length', None)

    # Map common names to python module names
    if src in ('clip', 'openai_clip'):
        mod_name = 'clip'
    elif src in ('longclip', 'long-clip', 'long_clip', 'open_clip_long'):
        # Map all long-clip aliases to local module 'open_clip_long'
        mod_name = 'open_clip_long'
    else:
        # allow the user to supply a full module name
        mod_name = src

    try:
        clip_mod = __import__(mod_name)
    except Exception as e:
        # If the local Long-CLIP module isn't on sys.path, add it and retry
        if mod_name == 'open_clip_long':
            longclip_root = osp.join(ROOT_PATH, 'Long-CLIP')
            if longclip_root not in sys.path:
                sys.path.insert(0, longclip_root)
            try:
                clip_mod = __import__(mod_name)
            except Exception as e2:
                raise ImportError(f"Failed to import CLIP module '{mod_name}' (requested src '{src}').\n"
                                  f"Tried adding '{longclip_root}' to sys.path but still failed. Original error: {e2}")
        else:
            raise ImportError(f"Failed to import CLIP module '{mod_name}' (requested src '{src}').\n" \
                              f"Install it or adjust cfg.clip4trn/cfg.clip4evl. Original error: {e}")

    # Special handling for local Long-CLIP fork (open_clip_long)
    if mod_name == 'open_clip_long':
        ocl = clip_mod
        # Normalize model name for open_clip_long (it replaces '/' with '-')
        model_name_norm = model_name.replace('/', '-')
        # Determine checkpoint to load, prefer explicit, else default directory
        default_dir = osp.join(ROOT_PATH, 'Long-CLIP', 'checkpoints')
        ckpt = None
        if isinstance(clip_info, dict):
            ckpt = clip_info.get('checkpoint') or clip_info.get('checkpoint_path') or clip_info.get('pretrained_path') or clip_info.get('weights_path')
        if not ckpt and osp.isdir(default_dir):
            # pick latest .pt/.pth under default dir
            cands = sorted(
                glob.glob(osp.join(default_dir, '*.pt')) +
                glob.glob(osp.join(default_dir, '*.pth')) +
                glob.glob(osp.join(default_dir, '*.bin')),
                key=lambda p: osp.getmtime(p), reverse=True
            )
            if cands:
                ckpt = cands[0]
        # Build model; if ckpt is a file path, pass it as 'pretrained'
        pretrained_arg = ckpt if (ckpt and osp.isfile(ckpt)) else None
        # Build optional kwargs to override text context length when supported
        model_kwargs = {}
        if ctx_len is not None:
            try:
                base_cfg = ocl.get_model_config(model_name_norm)
                if base_cfg and 'text_cfg' in base_cfg:
                    new_text_cfg = dict(base_cfg['text_cfg'])
                    new_text_cfg['context_length'] = int(ctx_len)
                    model_kwargs['text_cfg'] = new_text_cfg
            except Exception:
                pass
        # Try to respect context length if provided by caller
        try:
            if ctx_len is not None:
                try:
                    model = ocl.create_model(model_name_norm, pretrained=pretrained_arg, device=device, **model_kwargs)
                except TypeError:
                    model = ocl.create_model(model_name_norm, pretrained=pretrained_arg, device=device, **model_kwargs)
            else:
                model = ocl.create_model(model_name_norm, pretrained=pretrained_arg, device=device)
        except Exception as e:
            # As a fallback, try to create without pretrained and then load checkpoint
            try:
                if ctx_len is not None:
                    try:
                        model = ocl.create_model(model_name_norm, device=device, **model_kwargs)
                    except TypeError:
                        model = ocl.create_model(model_name_norm, device=device, **model_kwargs)
                else:
                    model = ocl.create_model(model_name_norm, device=device)
            except Exception:
                # last resort: create on CPU then move
                model = ocl.create_model(model_name_norm, **model_kwargs)
                model = model.to(device)
            if pretrained_arg:
                try:
                    ocl.load_checkpoint(model, pretrained_arg)
                except Exception as _:
                    pass
        return model

    # Default: OpenAI clip module API (load)
    load_fn = getattr(clip_mod, 'load', None)
    if load_fn is None:
        raise RuntimeError(f"Module '{mod_name}' does not have a 'load' function compatible with CLIP API")
    if ctx_len is not None:
        try:
            model = load_fn(model_name, device=device, context_length=ctx_len)[0]
            model = _maybe_load_checkpoint_into_model(model, clip_info.get('checkpoint') or clip_info)
            return model
        except TypeError:
            try:
                model = load_fn(model_name, device=device, context_len=ctx_len)[0]
                model = _maybe_load_checkpoint_into_model(model, clip_info.get('checkpoint') or clip_info)
                return model
            except TypeError:
                pass
    model = load_fn(model_name, device=device)[0]
    model = _maybe_load_checkpoint_into_model(model, clip_info.get('checkpoint') or clip_info.get('checkpoint_path') or clip_info.get('pretrained_path') or clip_info.get('weights_path'))
    return model


def prepare_models(args):
    device = args.device
    local_rank = args.local_rank
    multi_gpus = args.multi_gpus
    # Load CLIP models; keep training CLIP available by not forcing requires_grad=False
    CLIP4trn = load_clip(args.clip4trn, device).train()
    CLIP4evl = load_clip(args.clip4evl, device).eval()
    NetG,NetD,NetC,CLIP_IMG_ENCODER,CLIP_TXT_ENCODER = choose_model(args.model)
    freeze_clip = getattr(args, 'freeze_clip', True)
    # image encoder (trainable)
    CLIP_img_enc = CLIP_IMG_ENCODER(CLIP4trn, freeze=freeze_clip).to(device)
    CLIP_img_enc.train()
    # text encoder (trainable)
    CLIP_txt_enc = CLIP_TXT_ENCODER(CLIP4trn, freeze=freeze_clip).to(device)
    CLIP_txt_enc.train()
    # GAN models
    netG = NetG(args.nf, args.z_dim, args.cond_dim, args.imsize, args.ch_size, args.mixed_precision, CLIP4trn, freeze_clip=freeze_clip).to(device)
    netD = NetD(args.nf, args.imsize, args.ch_size, args.mixed_precision).to(device)
    netC = NetC(args.nf, args.cond_dim, args.mixed_precision).to(device)
    if (args.multi_gpus) and (args.train):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netG = torch.nn.parallel.DistributedDataParallel(netG, broadcast_buffers=False,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank, find_unused_parameters=True)
        netD = torch.nn.parallel.DistributedDataParallel(netD, broadcast_buffers=False,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank, find_unused_parameters=True)
        netC = torch.nn.parallel.DistributedDataParallel(netC, broadcast_buffers=False,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank, find_unused_parameters=True)
    return CLIP4trn, CLIP4evl, CLIP_img_enc, CLIP_txt_enc, netG, netD, netC


def prepare_dataset(args, split, transform):
    if args.ch_size!=3:
        imsize = 256
    else:
        imsize = args.imsize
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip(),
            ])
    from lib.datasets import TextImgDataset as Dataset
    dataset = Dataset(split=split, transform=image_transform, args=args)
    return dataset


def prepare_datasets(args, transform):
    # Always start from train split then create an 80/20 train/val split
    full_train = prepare_dataset(args, split='train', transform=transform)
    # 80/20 split
    total = len(full_train)
    train_size = int(0.8 * total)
    val_size = total - train_size
    torch.manual_seed(getattr(args, 'manual_seed', 100))
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])
    print(f"Dataset split (custom 80/20): {train_size} train, {val_size} val (from {total})")
    return train_dataset, val_dataset


def prepare_dataloaders(args, transform=None):
    batch_size = args.batch_size
    num_workers = args.num_workers
    train_dataset, valid_dataset = prepare_datasets(args, transform)
    # train dataloader
    if args.multi_gpus==True:
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, drop_last=True,
            num_workers=num_workers, sampler=train_sampler)
    else:
        train_sampler = None
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, drop_last=True,
            num_workers=num_workers, shuffle='True')
    # valid dataloader
    if args.multi_gpus==True:
        valid_sampler = DistributedSampler(valid_dataset)
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, drop_last=True,
            num_workers=num_workers, sampler=valid_sampler)
    else:
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, drop_last=True,
            num_workers=num_workers, shuffle='True')
    return train_dataloader, valid_dataloader, \
            train_dataset, valid_dataset, train_sampler

