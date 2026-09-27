"""
CaptionAI — Phase 5: precompute frozen DenseNet201 spatial features.

Runs the encoder ONCE over every image and saves spatial features
(batch, 49, 1920) so the attention decoder never touches JPEGs during
training. Outputs (into ../weights/):

    features_spatial.npy  (N, 49, 1920) float16  ordered like image_names.json

Usage:
    python precompute_spatial_features.py [--batch-size 64] [--dtype float16]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model.attention_decoder import build_spatial_encoder  # noqa: E402
from paths import DATASET_DIR, WEIGHTS_DIR                                 # noqa: E402

IMAGES_DIR = DATASET_DIR / "Images"
OUT_FILE = WEIGHTS_DIR / "features_spatial.npy"

NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    NORM,
])


def main() -> int:
    ap = argparse.ArgumentParser(description="Precompute spatial features")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--limit", type=int, default=0, help="limit images (smoke test)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> device: {device}", flush=True)

    encoder = build_spatial_encoder(str(device))
    image_names = json.loads((WEIGHTS_DIR / "image_names.json").read_text())
    if args.limit:
        image_names = image_names[:args.limit]
        print(f">> limiting to {args.limit} images", flush=True)
    print(f">> {len(image_names)} images", flush=True)

    np_dtype = np.float16 if args.dtype == "float16" else np.float32
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    t_start = time.time()
    all_feats = []
    bs = args.batch_size
    for start in range(0, len(image_names), bs):
        names = image_names[start:start + bs]
        imgs = []
        for nm in names:
            img = Image.open(IMAGES_DIR / nm).convert("RGB")
            imgs.append(TRANSFORM(img))
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = encoder(batch)                       # (B, 1920, 7, 7)
            b, ch, h, w = feats.shape
            feats = feats.view(b, ch, h * w).permute(0, 2, 1)  # (B, 49, 1920)
            if torch_dtype == torch.float16:
                feats = feats.half()
            feats = feats.cpu().numpy().astype(np_dtype)
        all_feats.append(feats)
        done = min(start + bs, len(image_names))
        if done % (bs * 5) == 0 or done == len(image_names):
            print(f"  {done}/{len(image_names)}  ({time.time() - t_start:.0f}s)", flush=True)

    features = np.concatenate(all_feats, axis=0).astype(np_dtype)
    np.save(OUT_FILE, features)
    print(f">> saved {features.shape} -> {OUT_FILE}  ({time.time() - t_start:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
