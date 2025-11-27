#!/usr/bin/env python3
"""
Evaluate a model or a directory of generated images against a test dataset specified by a CSV.

Metrics:
  - FID: Fréchet Inception Distance between real images and generated images
  - CLIPScore (Long-CLIP aware): cosine similarity between generated image and its caption
  - BERTScore: between captions (candidate vs reference). If no separate reference is provided,
               this will be computed caption-vs-caption and will be trivially high; a warning is printed.

Usage examples:

1) Generate images from a checkpoint, then evaluate:
   python eval_from_csv.py \
     --csv_path ./data/custom/test/descriptions_b_withgifwebp.csv \
     --images_dir_real ./data/custom/test/Images \
     --image_col filename --caption_col caption \
     --checkpoint ./saved_models/custom/GALIP_nf64_normal_custom_256_2025_10_18_12_34_36/state_epoch_300.pth \
     --cfg ./cfg/custom.yml \
     --out_dir ./eval_outputs/run1

2) Evaluate an existing directory of generated images:
   python eval_from_csv.py \
     --csv_path ./data/custom/test/descriptions_b_withgifwebp.csv \
     --images_dir_real ./data/custom/test/Images \
     --image_col filename --caption_col caption \
     --generated_dir ./my_generated_images \
     --cfg ./cfg/custom.yml \
     --out_dir ./eval_outputs/run2

Notes:
 - Long-CLIP is loaded based on cfg.clip4text/clip4evl; ensure cfg points to your Long-CLIP checkpoint if you want long context.
 - FID implementation here uses torchvision InceptionV3 pool3 features and an eigen-decomposition
   sqrt product (no SciPy dependency). For exact parity with popular FID implementations, installing scipy or pytorch-fid is recommended.
"""

import os
import os
# Prevent Transformers from loading TF/Flax backends that can conflict in this environment
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys
import glob
import time
import math
import argparse
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models.inception import inception_v3

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT_PATH)

from lib.perpare import prepare_models, load_clip
from lib.utils import load_yaml, read_txt_file, load_netG, mkdir_p

# Optional BERTScore
try:
    from bert_score import score as bert_score
except Exception:
    bert_score = None

# Optional SBERT (sentence-transformers) with TF-IDF cosine fallback
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
except Exception:
    TfidfVectorizer = None
    sk_cosine_similarity = None


def read_csv(csv_path: str, image_col: str, caption_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if image_col not in df.columns or caption_col not in df.columns:
        raise ValueError(f"CSV must contain columns '{image_col}' and '{caption_col}' (found: {list(df.columns)})")
    return df[[image_col, caption_col]].copy()


def pil_loader(img_path: str) -> Image.Image:
    with Image.open(img_path) as im:
        return im.convert('RGB')


def list_images(dir_path: str, exts=(".png", ".jpg", ".jpeg", ".webp"), recursive: bool = True) -> Dict[str, str]:
    """Map file stem -> absolute path. Optionally search recursively."""
    out = {}
    pattern = "**/*" if recursive else "*"
    for ext in exts:
        for p in glob.glob(os.path.join(dir_path, f"{pattern}{ext}"), recursive=recursive):
            if not os.path.isfile(p):
                continue
            stem = os.path.splitext(os.path.basename(p))[0]
            out[stem] = p
    return out


def find_generated_for_stem(gen_dir: str, stem: str) -> str:
    """Return a generated image path for a given stem (matches <stem>_gen_*.png)."""
    cands = sorted(glob.glob(os.path.join(gen_dir, f"{stem}_gen_*.png")))
    return cands[0] if cands else None


def resolve_real_path(images_dir_real: str, entry: str, fallback_index: Dict[str, str] = None) -> str:
    """Try to resolve a CSV 'file name' entry to a real image path."""
    # direct join
    p = os.path.join(images_dir_real, entry)
    if os.path.isfile(p):
        return p
    # try basename match via fallback index
    stem = os.path.splitext(os.path.basename(entry))[0]
    if fallback_index is None:
        fallback_index = list_images(images_dir_real)
    return fallback_index.get(stem)


def build_inception(device: torch.device):
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = nn.Identity()
    model.Mixed_7c.register_forward_hook(lambda m, inp, out: out)  # conv feature needed if customizing
    model.eval().to(device)
    return model


def inception_activations(img_paths: List[str], device: torch.device, batch_size: int = 32) -> np.ndarray:
    model = build_inception(device)
    transform = T.Compose([
        T.Resize(299),
        T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # scale [-1,1]; inception_v3 can handle either
    ])
    feats = []
    with torch.no_grad():
        for i in range(0, len(img_paths), batch_size):
            batch = []
            for p in img_paths[i:i+batch_size]:
                img = pil_loader(p)
                batch.append(transform(img))
            x = torch.stack(batch, dim=0).to(device)
            # get pool3 features via model.forward; for torchvision inception, last before fc is 2048-dim
            f = model(x)
            if isinstance(f, (list, tuple)):
                f = f[0]
            f = f.view(f.size(0), -1)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def compute_mu_sigma(acts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    return mu, sigma


def sqrtm_product_eig(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute sqrtm(A @ B) for SPD A,B using eigen-decomposition for numerical stability (SciPy-free).
    """
    # Compute eigen of A and whiten B in A's eigenspace
    wA, VA = np.linalg.eigh(A)
    # Clamp small/neg to avoid NaNs
    wA = np.clip(wA, 1e-10, None)
    Aw_half = VA @ np.diag(np.sqrt(wA)) @ VA.T
    Aw_mhalf = VA @ np.diag(1.0 / np.sqrt(wA)) @ VA.T
    C = Aw_mhalf @ B @ Aw_mhalf
    wC, VC = np.linalg.eigh(C)
    wC = np.clip(wC, 1e-10, None)
    C_half = VC @ np.diag(np.sqrt(wC)) @ VC.T
    return Aw_half @ C_half @ Aw_half


def fid_from_acts(mu1, sigma1, mu2, sigma2) -> float:
    diff = mu1 - mu2
    covmean = sqrtm_product_eig(sigma1, sigma2)
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(np.real(fid))


def compute_fid_from_pairs(real_list: List[str], gen_list: List[str], device: torch.device) -> float:
    if not real_list or not gen_list:
        raise ValueError("Empty image lists passed to FID computation")
    acts_real = inception_activations(real_list, device)
    acts_gen = inception_activations(gen_list, device)
    if acts_real.size == 0 or acts_gen.size == 0:
        raise ValueError("No features computed for FID (empty activations)")
    mu1, sigma1 = compute_mu_sigma(acts_real)
    mu2, sigma2 = compute_mu_sigma(acts_gen)
    return fid_from_acts(mu1, sigma1, mu2, sigma2)


def load_long_clip_from_cfg(cfg: dict, device: torch.device):
    clip_info = cfg.get('clip4evl', cfg.get('clip4text', cfg.get('clip4trn')))
    if clip_info is None:
        raise RuntimeError("Config missing clip4evl/clip4text/clip4trn for CLIP scoring")
    model = load_clip(clip_info, device)
    model.eval()
    # Build image preprocess for CLIP (224, mean/var per CLIP)
    preprocess = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    return model, preprocess


def tokenize_for_cfg(captions: List[str], cfg: dict, device: torch.device) -> torch.Tensor:
    # Use the same logic as training: choose tokenizer by clip4text.src
    clip_info = cfg.get('clip4text', cfg.get('clip4trn'))
    if clip_info is None:
        # fallback to openai clip
        import clip as clip_mod
        return clip_mod.tokenize(captions, truncate=True).to(device)
    src = clip_info.get('src', 'clip') if isinstance(clip_info, dict) else 'clip'
    if src in ('longclip', 'long-clip', 'long_clip', 'open_clip_long'):
        mod_name = 'open_clip_long'
        long_root = os.path.join(ROOT_PATH, 'Long-CLIP')
        if os.path.isdir(long_root) and long_root not in sys.path:
            sys.path.insert(0, long_root)
        clip_mod = __import__(mod_name)
        ctx_len = clip_info.get('context_length', None) if isinstance(clip_info, dict) else None
        try:
            if ctx_len is not None:
                return clip_mod.tokenize(captions, truncate=True, context_length=ctx_len).to(device)
        except TypeError:
            return clip_mod.tokenize(captions, truncate=True).to(device)
        return clip_mod.tokenize(captions, truncate=True).to(device)
    else:
        import clip as clip_mod
        return clip_mod.tokenize(captions, truncate=True).to(device)


def compute_clipscore(gen_dir: str, df: pd.DataFrame, image_col: str, caption_col: str, cfg: dict, device: torch.device) -> float:
    model, preprocess = load_long_clip_from_cfg(cfg, device)
    sims = []
    with torch.no_grad():
        for _, row in df.iterrows():
            stem = os.path.splitext(os.path.basename(str(row[image_col])))[0]
            img_path = find_generated_for_stem(gen_dir, stem)
            if not img_path or not os.path.isfile(img_path):
                continue
            img = preprocess(pil_loader(img_path)).unsqueeze(0).to(device)
            tokens = tokenize_for_cfg([str(row[caption_col])], cfg, device)
            # encode
            img_feat = model.encode_image(img)
            txt_feat = model.encode_text(tokens)
            # normalize and cosine sim
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat @ txt_feat.T).squeeze().item()
            sims.append(sim)
    return float(np.mean(sims)) if sims else float('nan')


def compute_bertscore(df: pd.DataFrame, caption_col: str, ref_caption_col: str = None, model_type: str = 'bert-base-uncased', device: torch.device = None, gpu_id: int = 0) -> Dict[str, float]:
    if bert_score is None:
        print("[warn] bert-score not installed; pip install bert-score to enable. Skipping BERTScore.")
        return {"precision": float('nan'), "recall": float('nan'), "f1": float('nan')}
    cands = [str(t) for t in df[caption_col].tolist()]
    if ref_caption_col and ref_caption_col in df.columns:
        refs = [str(t) for t in df[ref_caption_col].tolist()]
    else:
        refs = cands
        print("[warn] No reference caption column provided; computing BERTScore vs self (trivial). Use --ref_caption_col to provide references.")
    try:
        # Prefer CPU for stability unless CUDA requested
        dev = 'cpu'
        if device is not None and isinstance(device, torch.device) and device.type == 'cuda':
            dev = f'cuda:{gpu_id}'
        P, R, F1 = bert_score(cands, refs, lang="en", model_type=model_type, verbose=False, device=dev, rescale_with_baseline=True)
        return {"precision": float(P.mean().item()), "recall": float(R.mean().item()), "f1": float(F1.mean().item())}
    except Exception as e:
        print(f"[warn] BERTScore failed with model '{model_type}' ({e}). Trying alternate model...")
        try:
            alt = 'roberta-large' if model_type != 'roberta-large' else 'bert-base-uncased'
            P, R, F1 = bert_score(cands, refs, lang="en", model_type=alt, verbose=False, device=dev, rescale_with_baseline=True)
            return {"precision": float(P.mean().item()), "recall": float(R.mean().item()), "f1": float(F1.mean().item())}
        except Exception as e2:
            print(f"[warn] BERTScore fallback also failed ({e2}). Skipping BERTScore.")
            return {"precision": float('nan'), "recall": float('nan'), "f1": float('nan')}


def compute_sbert(df: pd.DataFrame, caption_col: str, ref_caption_col: str = None, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', device: torch.device = None, gpu_id: int = 0) -> float:
    """
    Compute mean cosine similarity between candidate captions and reference captions using SBERT.
    If no reference column is provided, compares captions to themselves (trivial high score) with a warning.
    If sentence-transformers is unavailable, falls back to TF-IDF cosine (if available) else returns NaN.
    """
    cands = [str(t) for t in df[caption_col].tolist()]
    if ref_caption_col and ref_caption_col in df.columns:
        refs = [str(t) for t in df[ref_caption_col].tolist()]
    else:
        refs = cands
        print("[warn] No reference caption column provided; SBERT will compare captions to themselves (trivial). Use --ref_caption_col for real references.")
    try:
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers not installed")
        dev = 'cpu'
        if device is not None and device.type == 'cuda':
            dev = f'cuda:{gpu_id}'
        model = SentenceTransformer(model_name, device=dev)
        emb_c = model.encode(cands, convert_to_tensor=True, show_progress_bar=False)
        emb_r = model.encode(refs, convert_to_tensor=True, show_progress_bar=False)
        # pairwise cosine for aligned pairs
        emb_c = F.normalize(emb_c, p=2, dim=-1)
        emb_r = F.normalize(emb_r, p=2, dim=-1)
        sims = (emb_c * emb_r).sum(dim=-1).detach().cpu().numpy().tolist()
        return float(np.mean(sims)) if len(sims) > 0 else float('nan')
    except Exception as e:
        print(f"[warn] SBERT failed with model '{model_name}' ({e}). Trying TF-IDF cosine fallback...")
        if TfidfVectorizer is None or sk_cosine_similarity is None:
            print("[warn] sklearn not available for TF-IDF fallback. Returning NaN for SBERT.")
            return float('nan')
        # TF-IDF cosine per pair
        sims = []
        for a, b in zip(cands, refs):
            try:
                vec = TfidfVectorizer().fit([a, b])
                mat = vec.transform([a, b])  # 2 x V
                sim = sk_cosine_similarity(mat[0], mat[1])[0, 0]
                sims.append(sim)
            except Exception:
                continue
        return float(np.mean(sims)) if sims else float('nan')


def generate_images(checkpoint: str, cfg_path: str, df: pd.DataFrame, images_dir_real: str, image_col: str, caption_col: str,
                    out_dir: str, device: torch.device, num_per_caption: int = 1, seed: int = 1234) -> str:
    """Generate images with netG for each caption; saves as <stem>_gen.png in out_dir. Returns generated_dir."""
    from lib.perpare import prepare_models
    from lib.utils import load_netG
    cfg = load_yaml(cfg_path)
    # minimal args namespace for model prep
    import types
    Args = types.SimpleNamespace()
    Args.device = device
    Args.local_rank = -1
    Args.multi_gpus = False
    Args.train = False
    Args.model = cfg.get('model', 'GALIP')
    Args.nf = cfg.get('nf', 64)
    Args.z_dim = cfg.get('z_dim', 100)
    Args.cond_dim = cfg.get('cond_dim', 512)
    Args.imsize = cfg.get('imsize', 256)
    Args.ch_size = cfg.get('ch_size', 3)
    Args.mixed_precision = cfg.get('mixed_precision', False)
    Args.clip4trn = cfg.get('clip4trn')
    Args.clip4evl = cfg.get('clip4evl')
    # models
    CLIP4trn, CLIP4evl, img_enc, txt_enc, netG, netD, netC = prepare_models(Args)
    netG = load_netG(netG, checkpoint, Args.multi_gpus, train=False).to(device).eval()
    txt_enc = txt_enc.to(device).eval()
    # output dir
    gen_dir = os.path.join(out_dir, 'gen_images')
    mkdir_p(gen_dir)
    # tokenize all captions once in small batches
    caps = [str(t) for t in df[caption_col].tolist()]
    batch_size = 32
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for i in range(0, len(caps), batch_size):
            caps_batch = caps[i:i+batch_size]
            tokens = tokenize_for_cfg(caps_batch, cfg, device)
            sent_emb, _ = txt_enc(tokens)
            for j, emb in enumerate(sent_emb):
                stem = os.path.splitext(os.path.basename(str(df.iloc[i+j][image_col])))[0]
                for k in range(num_per_caption):
                    noise = torch.from_numpy(rng.standard_normal((1, Args.z_dim)).astype(np.float32)).to(device)
                    img = netG(noise, emb.unsqueeze(0), eval=True)
                    # save as PNG in [-1,1] -> [0,1]
                    img_np = (img.clamp(-1, 1) + 1) / 2.0
                    save_path = os.path.join(gen_dir, f"{stem}_gen_{k}.png")
                    import torchvision.utils as vutils
                    vutils.save_image(img_np.cpu(), save_path, normalize=False)
    return gen_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv_path', type=str, required=True, help='CSV with image filenames and captions')
    p.add_argument('--images_dir_real', type=str, required=True, help='Directory with real test images')
    p.add_argument('--image_col', type=str, default='filename', help='CSV column for image filenames')
    p.add_argument('--caption_col', type=str, default='caption', help='CSV column for captions')
    p.add_argument('--ref_caption_col', type=str, default=None, help='Optional CSV column with reference captions for BERTScore')
    p.add_argument('--generated_dir', type=str, default=None, help='Directory of generated images (if not generating)')
    p.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to generate images from (if provided)')
    p.add_argument('--cfg', type=str, default='./cfg/custom.yml', help='YAML config for model/CLIP settings')
    p.add_argument('--out_dir', type=str, default='./eval_outputs/run_'+time.strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--metrics', type=str, default='fid,clip,bertscore', help='Comma-separated: fid,clip,bertscore')
    p.add_argument('--bertscore_model', type=str, default='bert-base-uncased', help='Hugging Face model for BERTScore (e.g., bert-base-uncased, roberta-large)')
    p.add_argument('--sbert_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='Sentence-Transformers model for SBERT similarity')
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--num_images_per_caption', type=int, default=1, help='When generating, images per caption')
    p.add_argument('--limit', type=int, default=None, help='Optional limit for quick tests')
    return p.parse_args()


def main():
    args = parse_args()
    mkdir_p(args.out_dir)
    # device
    if not args.cpu and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device('cuda')
        print(f"Using CUDA:{args.gpu_id}")
    else:
        device = torch.device('cpu')
        print('Using CPU')

    # data
    df = read_csv(args.csv_path, args.image_col, args.caption_col)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    # generate or use provided directory
    gen_dir = args.generated_dir
    if gen_dir is None and args.checkpoint:
        gen_dir = generate_images(
            checkpoint=args.checkpoint,
            cfg_path=args.cfg,
            df=df,
            images_dir_real=args.images_dir_real,
            image_col=args.image_col,
            caption_col=args.caption_col,
            out_dir=args.out_dir,
            device=device,
            num_per_caption=args.num_images_per_caption,
        )
        print('Generated images saved to:', gen_dir)
    if gen_dir is None:
        raise ValueError('Either --generated_dir or --checkpoint must be provided')

    # metrics selection
    sel = {m.strip().lower() for m in args.metrics.split(',') if m.strip()}
    results = {}
    # FID
    if 'fid' in sel:
        # Build pairs directly from CSV rows using generated naming convention
        fallback_index = list_images(args.images_dir_real)
        real_list, gen_list = [], []
        for _, row in df.iterrows():
            name = str(row[args.image_col])
            stem = os.path.splitext(os.path.basename(name))[0]
            rp = resolve_real_path(args.images_dir_real, name, fallback_index)
            gp = find_generated_for_stem(gen_dir, stem)
            if rp and gp and os.path.isfile(rp) and os.path.isfile(gp):
                real_list.append(rp)
                gen_list.append(gp)
            if args.limit is not None and len(real_list) >= args.limit:
                break
        if not real_list or not gen_list:
            raise RuntimeError("No matching real/generated pairs found for FID. Check image_col names and file layout.")
        print(f"Computing FID on {len(gen_list)} pairs ...")
        fid = compute_fid_from_pairs(real_list, gen_list, device)
        results['fid'] = fid
        print(f"FID: {fid:.4f}")

    # CLIPScore (Long-CLIP aware via cfg)
    if 'clip' in sel:
        print("Computing CLIPScore (config-driven tokenizer/model) ...")
        cfg = load_yaml(args.cfg)
        clip_score = compute_clipscore(gen_dir, df, args.image_col, args.caption_col, cfg, device)
        results['clip_score'] = clip_score
        print(f"CLIPScore: {clip_score:.4f}")

    # BERTScore
    if 'bertscore' in sel:
        print("Computing BERTScore (caption vs reference or self if not provided) ...")
        bs = compute_bertscore(df, args.caption_col, args.ref_caption_col, model_type=args.bertscore_model, device=device, gpu_id=args.gpu_id)
        results.update({f"bertscore_{k}": v for k, v in bs.items()})
        print(f"BERTScore F1: {bs['f1']:.4f} (P={bs['precision']:.4f}, R={bs['recall']:.4f})")

    # SBERT cosine text similarity
    if 'sbert' in sel:
        print("Computing SBERT cosine similarity (caption vs reference or self) ...")
        sbert_sim = compute_sbert(df, args.caption_col, args.ref_caption_col, model_name=args.sbert_model, device=device, gpu_id=args.gpu_id)
        results['sbert'] = sbert_sim
        print(f"SBERT cosine: {sbert_sim:.4f}")

    # Save summary
    summary_path = os.path.join(args.out_dir, 'metrics_summary.csv')
    if results:
        pd.DataFrame([results]).to_csv(summary_path, index=False)
        print('Saved metrics summary to', summary_path)
    else:
        print('No metrics computed (check --metrics).')


if __name__ == '__main__':
    main()
