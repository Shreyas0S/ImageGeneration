#!/usr/bin/env python3
"""
Quick inference wrapper: find latest checkpoint in `saved_models/custom` and generate images for given captions.

Usage:
  python infer_from_saved_custom.py --captions "a red bird on a branch" --num_images 2 --gpu_id 0

If you want to point to a specific checkpoint, use --checkpoint /path/to/state_epoch_060.pth
"""
import os
import sys
import argparse
import time
import glob
import torch
import torchvision.utils as vutils

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT_PATH)

from lib.perpare import prepare_models
from lib.utils import load_yaml, read_txt_file, load_netG

# Optional summarization (transformers). We'll gracefully fall back if unavailable.
try:
    from transformers import pipeline as hf_pipeline
except Exception:
    hf_pipeline = None


def find_latest_checkpoint(saved_models_dir: str):
    pattern = os.path.join(saved_models_dir, "*.pth")
    files = glob.glob(pattern)
    if not files:
        return None
    files = sorted(files, key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def build_args_from_cfg(cfg_path: str, device: torch.device):
    cfg = load_yaml(cfg_path)
    cfg.device = device
    cfg.local_rank = -1
    cfg.multi_gpus = False
    cfg.train = False
    cfg.model = 'GALIP'
    if not hasattr(cfg, 'mixed_precision'):
        cfg.mixed_precision = False
    return cfg


def _get_tokenizer_from_clip_info(clip_info):
    """Return (module, context_length) for tokenization based on cfg.clip4text."""
    src = None
    try:
        src = clip_info.get('src', 'clip') if isinstance(clip_info, dict) else 'clip'
    except Exception:
        src = 'clip'
    if src in ('clip', 'openai_clip'):
        mod_name = 'clip'
    elif src in ('longclip', 'long-clip', 'long_clip', 'open_clip_long'):
        mod_name = 'open_clip_long'
    else:
        mod_name = src
    # if long-clip, ensure local Long-CLIP is on sys.path
    if mod_name == 'open_clip_long':
        root_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        longclip_root = os.path.join(root_path, 'Long-CLIP')
        if os.path.isdir(longclip_root) and longclip_root not in sys.path:
            sys.path.insert(0, longclip_root)
    clip_mod = __import__(mod_name)
    ctx_len = None
    try:
        ctx_len = clip_info.get('context_length', None) if isinstance(clip_info, dict) else None
    except Exception:
        ctx_len = None
    return clip_mod, ctx_len


def tokenize_captions(captions, device, clip_info_for_text):
    clip_mod, ctx_len = _get_tokenizer_from_clip_info(clip_info_for_text)
    tokens = None
    try:
        if ctx_len is not None:
            try:
                tokens = clip_mod.tokenize(captions, truncate=True, context_length=ctx_len)
            except TypeError:
                tokens = clip_mod.tokenize(captions, truncate=True, context_len=ctx_len)
        else:
            tokens = clip_mod.tokenize(captions, truncate=True)
    except Exception:
        # Per-caption fallback
        bag = []
        for c in captions:
            t = clip_mod.tokenize(c, truncate=True)
            bag.append(t[0] if hasattr(t, '__getitem__') else t)
        tokens = torch.stack(bag, dim=0)
    return tokens.to(device)


def generate_and_save(netG, txt_enc, tokens, captions, num_images, device, out_dir, z_dim=100, truncation=False, trunc_rate=0.88, seed=None):
    with torch.no_grad():
        sent_emb, _ = txt_enc(tokens)
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    os.makedirs(out_dir, exist_ok=True)
    images_dir = os.path.join(out_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    for i, cap in enumerate(captions):
        c = sent_emb[i].unsqueeze(0).repeat(num_images, 1).to(device)
        if truncation:
            noise = torch.randn(num_images, z_dim, device=device).clamp_(-2, 2)
        else:
            noise = torch.randn(num_images, z_dim, device=device)
        with torch.no_grad():
            imgs = netG(noise, c, eval=True)
        cap_dir = os.path.join(images_dir, f'caption_{i:03d}')
        os.makedirs(cap_dir, exist_ok=True)
        paths = []
        for j, img in enumerate(imgs.cpu()):
            path = os.path.join(cap_dir, f'img_{j:02d}.png')
            vutils.save_image(img, path, normalize=True)
            paths.append(path)
        print(f'Caption {i}: "{cap}" -> saved {len(paths)} images to {cap_dir}')


def maybe_summarize(captions, do_summarize, model_name, device_is_cuda, gpu_id, max_new_tokens, min_length, context_length_hint=None):
    """
    Optionally summarize captions with a Hugging Face pipeline. If transformers isn't available
    or model loading fails, fall back to a simple truncation-based strategy that keeps the first
    N words.
    """
    if not do_summarize:
        return captions
    # Prefer a short cap length if user didn't specify context; default to 77 tokens for CLIP
    target_words = 72 if not context_length_hint else max(16, int(context_length_hint) - 5)
    if hf_pipeline is None:
        # Fallback: naive summarization by truncation
        out = []
        for c in captions:
            words = c.split()
            out.append(" ".join(words[:target_words]))
        print("[summarize] transformers not available; applied naive truncation to ~", target_words, "words")
        return out
    try:
        device_arg = 0 if device_is_cuda else -1
        if device_is_cuda and isinstance(gpu_id, int):
            device_arg = gpu_id
        summarizer = hf_pipeline(
            "summarization",
            model=model_name,
            device=device_arg
        )
        summaries = []
        for c in captions:
            # guard against extremely long inputs for smaller models
            res = summarizer(
                c,
                max_new_tokens=max_new_tokens,
                min_length=min_length,
                truncation=True
            )
            text = res[0]['summary_text'] if isinstance(res, list) and len(res) > 0 else c
            # enforce simple post-truncation if still too long
            words = text.split()
            if len(words) > target_words:
                text = " ".join(words[:target_words])
            summaries.append(text)
        print(f"[summarize] Applied model='{model_name}' to {len(captions)} captions (target ~{target_words} words)")
        return summaries
    except Exception as e:
        # Fallback: naive truncation
        out = []
        for c in captions:
            words = c.split()
            out.append(" ".join(words[:target_words]))
        print(f"[summarize] Failed to load/use transformers model '{model_name}' ({e}); applied naive truncation to ~{target_words} words")
        return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, default=None, help='Path to .pth checkpoint; if omitted, picks latest in saved_models/custom')
    p.add_argument('--cfg', type=str, default='./cfg/custom.yml')
    p.add_argument('--captions', type=str, nargs='*', default=None)
    p.add_argument('--captions_file', type=str, default=None)
    p.add_argument('--num_images', type=int, default=1)
    p.add_argument('--truncation', action='store_true')
    p.add_argument('--trunc_rate', type=float, default=0.88)
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--output_dir', type=str, default='./inference_outputs/custom')
    p.add_argument('--seed', type=int, default=None)
    # Optional summarization of long captions
    p.add_argument('--summarize', action='store_true', help='Summarize captions before tokenization to fit CLIP context length')
    p.add_argument('--summary_model', type=str, default='sshleifer/distilbart-cnn-12-6', help='HF model name for summarization')
    p.add_argument('--summary_max_new_tokens', type=int, default=64, help='Max new tokens for summarization output')
    p.add_argument('--summary_min_length', type=int, default=10, help='Minimum generated length for summarization output')
    return p.parse_args()


def main():
    args = parse_args()
    # captions
    caps = args.captions or []
    if args.captions_file:
        caps += [l.strip() for l in read_txt_file(args.captions_file) if l.strip()]
    caps = [c.strip() for c in caps if c and c.strip()]
    if not caps:
        print('No captions provided. Use --captions or --captions_file.')
        return

    # device
    if not args.cpu and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device('cuda')
        print(f'Using CUDA:{args.gpu_id}')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    cfg = build_args_from_cfg(args.cfg, device)
    z_dim = getattr(cfg, 'z_dim', 100)
    # Summarize captions if requested
    if args.summarize:
        context_hint = None
        try:
            context_hint = cfg.get('clip4text', {}).get('context_length', None)
        except Exception:
            context_hint = None
        caps = maybe_summarize(
            caps,
            do_summarize=True,
            model_name=args.summary_model,
            device_is_cuda=(device.type == 'cuda'),
            gpu_id=args.gpu_id,
            max_new_tokens=args.summary_max_new_tokens,
            min_length=args.summary_min_length,
            context_length_hint=context_hint or 77
        )

    # find checkpoint
    ckpt = args.checkpoint
    if ckpt is None:
        ckpt = find_latest_checkpoint(os.path.join(ROOT_PATH, 'saved_models', 'custom'))
    if ckpt is None or not os.path.isfile(ckpt):
        print('No checkpoint found in saved_models/custom; please provide --checkpoint path')
        return
    print('Using checkpoint:', ckpt)

    # prepare models and load generator
    CLIP4trn, CLIP4evl, img_enc, txt_enc, netG, netD, netC = prepare_models(cfg)
    netG = load_netG(netG, ckpt, cfg.multi_gpus, train=False)
    netG.eval().to(device)
    txt_enc.eval().to(device)

    # tokenize according to cfg.clip4text
    clip4text = cfg.get('clip4text', cfg.get('clip4trn'))
    tokens = tokenize_captions(caps, device, clip4text)

    # generate
    out_dir = os.path.join(args.output_dir, f'run_{time.strftime("%Y%m%d_%H%M%S")}')
    generate_and_save(netG, txt_enc, tokens, caps, args.num_images, device, out_dir, z_dim=z_dim, truncation=args.truncation, trunc_rate=args.trunc_rate, seed=args.seed)
    print('Saved outputs to', out_dir)


if __name__ == '__main__':
    main()
