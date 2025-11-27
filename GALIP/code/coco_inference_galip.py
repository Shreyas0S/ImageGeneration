#!/usr/bin/env python3
"""
GALIP COCO-caption inference script.
- Loads GALIP generator and CLIP text encoder
- Accepts arbitrary captions (CLI or file)
- Generates images per caption

Usage:
  python coco_inference_galip.py \
    --pretrained ./saved_models/coco/pre_coco.pth \
    --captions "a person riding a bike" "a red bus on the street" \
    --num_images_per_caption 1 --truncation --gpu_id 0

For low VRAM or to avoid CLIP model GPU load, add --cpu.
"""
import os
import sys
import argparse
import time
import json
from typing import List

import torch
import torchvision.utils as vutils

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT_PATH)

from lib.perpare import prepare_models
from lib.utils import load_yaml


def build_args_from_cfg(cfg_path: str, device: torch.device):
    cfg = load_yaml(cfg_path)
    # Ensure required fields and overrides
    cfg.device = device
    cfg.local_rank = -1
    cfg.multi_gpus = False
    cfg.train = False
    cfg.model = 'GALIP'
    if not hasattr(cfg, 'mixed_precision'):
        cfg.mixed_precision = False
    # Respect coco config values (imsize=256, cond_dim=512, nf=64, z_dim=100, ch_size=3)
    return cfg


def tokenize_captions(captions: List[str], device: torch.device):
    import clip
    model, _ = clip.load('ViT-B/32', device=device)
    tokens = clip.tokenize(captions).to(device)
    return model, tokens


def load_galip_generator(args, pretrained_path: str):
    # Build models using GALIP utilities
    CLIP4trn, CLIP4evl, img_enc, txt_enc, netG, _, _ = prepare_models(args)
    # Load generator weights
    from lib.utils import load_netG
    netG = load_netG(netG, pretrained_path, args.multi_gpus, train=False)
    netG.eval().to(args.device)
    txt_enc.eval().to(args.device)
    return netG, txt_enc


def generate(netG, txt_enc, clip_model, tokens, captions: List[str], num_images: int, device: torch.device, truncation: bool, trunc_rate: float, seed: int = None):
    # Encode text with CLIP text encoder wrapper
    with torch.no_grad():
        sent_emb, _ = txt_enc(tokens)
    results = []
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    for i, cap in enumerate(captions):
        c = sent_emb[i].unsqueeze(0).repeat(num_images, 1)
        if truncation:
            from lib.utils import BICUBIC  # placeholder to ensure utils imported
            # use standard normal; GALIP has no truncated_noise util
            noise = torch.randn(num_images, 100, device=device).clamp_(-2, 2)
        else:
            noise = torch.randn(num_images, 100, device=device)
        with torch.no_grad():
            imgs = netG(noise, c, eval=True)
        results.append({'caption': cap, 'images': imgs.cpu()})
    return results


def save_outputs(results, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    images_dir = os.path.join(out_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    entries = []
    for i, entry in enumerate(results):
        cap_dir = os.path.join(images_dir, f'caption_{i:03d}')
        os.makedirs(cap_dir, exist_ok=True)
        paths = []
        for j, img in enumerate(entry['images']):
            path = os.path.join(cap_dir, f'img_{j:02d}.png')
            vutils.save_image(img, path, normalize=True)
            paths.append(path)
        entries.append({'caption': entry['caption'], 'image_paths': paths})
    with open(os.path.join(out_dir, 'inference_summary.json'), 'w') as f:
        json.dump({'captions': entries, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)
    print(f'Saved outputs to {out_dir}')


def parse_args():
    p = argparse.ArgumentParser(description='GALIP COCO-caption Inference')
    p.add_argument('--pretrained', type=str, default='./saved_models/coco/pre_coco.pth', help='Path to GALIP pretrained .pth')
    p.add_argument('--cfg', type=str, default='./cfg/coco.yml', help='Path to config YAML (use coco.yml)')
    p.add_argument('--captions', type=str, nargs='*', default=None)
    p.add_argument('--captions_file', type=str, default=None)
    p.add_argument('--num_images_per_caption', type=int, default=1)
    p.add_argument('--truncation', action='store_true')
    p.add_argument('--trunc_rate', type=float, default=0.88)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--output_dir', type=str, default='./galip_coco_inference_outputs')
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--cpu', action='store_true')
    return p.parse_args()


def main():
    args_cli = parse_args()
    # Collect captions
    caps = args_cli.captions or []
    if args_cli.captions_file:
        with open(args_cli.captions_file, 'r', encoding='utf-8') as f:
            caps += [l.strip() for l in f if l.strip()]
    caps = [c.strip() for c in caps if c and c.strip()]
    if not caps:
        print('No captions provided. Use --captions or --captions_file.')
        sys.exit(1)

    # Device
    if not args_cli.cpu and torch.cuda.is_available():
        torch.cuda.set_device(args_cli.gpu_id)
        device = torch.device('cuda')
        print(f'Using CUDA:{args_cli.gpu_id}')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    # Build GALIP args and load models (align with coco config)
    args = build_args_from_cfg(args_cli.cfg, device)
    netG, txt_enc = load_galip_generator(args, args_cli.pretrained)

    # Tokenize with CLIP and generate
    clip_model, tokens = tokenize_captions(caps, device)
    results = generate(netG, txt_enc, clip_model, tokens, caps, args_cli.num_images_per_caption, device, args_cli.truncation, args_cli.trunc_rate, seed=args_cli.seed)

    # Save
    out_dir = os.path.join(args_cli.output_dir, f'run_{time.strftime("%Y%m%d_%H%M%S")}')
    save_outputs(results, out_dir)


if __name__ == '__main__':
    main()
