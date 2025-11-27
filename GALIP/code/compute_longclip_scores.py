import os
import sys
import json
from typing import List

import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

# ---- Add Long-CLIP repo path + local open_clip dependency ----
LONGCLIP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Long-CLIP'))
OPENCLIP_LOCAL = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'open_clip'))
for p in (LONGCLIP_ROOT, OPENCLIP_LOCAL):
    if p not in sys.path:
        sys.path.insert(0, p)

import open_clip_long as longclip  # from the cloned repo


# ===================== CONFIG ===================== #
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # /.../GANs/GALIP
CAPTIONS_FILE = os.path.join(ROOT, 'code', 'text_captions.txt')
IMAGES_ROOT = os.path.join(ROOT, 'code', 'inference_outputs', 'custom', 'run_20251106_160349', 'images')
CHECKPOINT_PATH = os.path.join(ROOT, 'Long-CLIP', 'checkpoints', 'LongCLIP-B', 'longclip-B.pt')
MODEL = "ViT-B-16"  # a valid model from long-clip list_models()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ================================================== #


def build_preprocess() -> T.Compose:
    return T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])


def ensure_text_mask_matches_seq(model: torch.nn.Module, seq_len: int):
    try:
        txt = getattr(model, 'text', None)
        if txt is None:
            return
        am = getattr(txt, 'attn_mask', None)
        if isinstance(am, torch.Tensor):
            if am.shape[0] != seq_len or am.shape[1] != seq_len:
                new_mask = torch.empty(seq_len, seq_len, device=next(model.parameters()).device)
                new_mask.fill_(float('-inf'))
                new_mask.triu_(1)
                txt.attn_mask = new_mask
        elif am is None:
            new_mask = torch.empty(seq_len, seq_len, device=next(model.parameters()).device)
            new_mask.fill_(float('-inf'))
            new_mask.triu_(1)
            txt.attn_mask = new_mask
    except Exception:
        pass


def score_caption_images(caption: str, image_paths: List[str], model, preprocess, device="cuda"):
    """Compute CLIP similarity for one caption and multiple images."""
    # Tokenize caption
    ctx_len = 77
    try:
        pe = getattr(model.text, 'positional_embedding', None)
        if pe is not None and hasattr(pe, 'shape'):
            ctx_len = int(pe.shape[0])
    except Exception:
        pass
    text_tokens = longclip.tokenize([caption], context_length=ctx_len).to(device)
    ensure_text_mask_matches_seq(model, seq_len=text_tokens.shape[1])

    with torch.no_grad():
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    image_feats = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠️ Skipping unreadable image: {img_path} ({e})")
            continue
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img_tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        image_feats.append(feat)

    if not image_feats:
        return None, []

    image_feats = torch.cat(image_feats, dim=0)
    sims = (image_feats @ text_feats.T).squeeze(1).cpu().tolist()  # cosine sim
    avg_sim = sum(sims) / len(sims)
    return avg_sim, sims


def main():
    print("🚀 Loading Long-CLIP model...")
    model = longclip.create_model(MODEL, device=DEVICE)
    try:
        longclip.load_checkpoint(model, CHECKPOINT_PATH, strict=False)
    except Exception as e:
        print(f"[warn] Failed to load checkpoint: {e}")
    model.eval()
    # Force a consistent short context to bypass buggy mask config in this fork
    try:
        txt = getattr(model, 'text', None)
        if txt is not None:
            seq = 77
            dev = next(model.parameters()).device
            if hasattr(txt, 'num_pos'):
                txt.num_pos = seq
            if hasattr(txt, 'context_length'):
                txt.context_length = seq
            # Create a CLS embedding to trigger attn_mask slicing in forward
            try:
                width = getattr(txt, 'width', 512)
                txt.cls_emb = torch.nn.Parameter(torch.zeros(width, device=dev))
            except Exception:
                pass
            mask = torch.empty(seq, seq, device=dev)
            mask.fill_(float('-inf'))
            mask.triu_(1)
            txt.attn_mask = mask
    except Exception:
        pass
    preprocess = build_preprocess()

    # Load captions
    with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
        captions = [line.strip() for line in f if line.strip()]

    results = []

    for idx, caption in enumerate(tqdm(captions, desc="Evaluating captions")):
        caption_dir = os.path.join(IMAGES_ROOT, f"caption_{idx:03d}")
        if not os.path.isdir(caption_dir):
            print(f"⚠️ Skipping caption_{idx:03d} (folder not found)")
            continue

        image_paths = [
            os.path.join(caption_dir, f)
            for f in os.listdir(caption_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not image_paths:
            print(f"⚠️ No images found in {caption_dir}")
            continue

        avg_score, individual_scores = score_caption_images(
            caption, image_paths, model, preprocess, DEVICE
        )

        if avg_score is None:
            continue

        print(f"📸 Caption {idx}: Avg CLIP Score = {avg_score:.4f}")

        results.append({
            "caption_index": idx,
            "caption": caption,
            "avg_clip_score": avg_score,
            "individual_scores": individual_scores
        })

    if not results:
        print("❌ No valid results computed.")
        return

    overall_avg = sum(r["avg_clip_score"] for r in results) / len(results)
    print("\n✅ Done!")
    print(f"📊 Overall Average CLIP Score: {overall_avg:.4f}")

    with open("longclip_repo_scores.json", "w") as f:
        json.dump(results, f, indent=4)
    print("💾 Saved results to longclip_repo_scores.json")


if __name__ == "__main__":
    main()
