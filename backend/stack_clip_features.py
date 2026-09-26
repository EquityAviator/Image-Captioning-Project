"""
Stack per-image CLIP spatial features into one memmap-friendly .npy
indexed by weights/image_names.json order — mirrors features_spatial.npy
so train/eval code stays identical to the proven v2 pipeline.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
WEIGHTS_DIR = PROJECT_DIR / "weights"
CLIP_DIR = WEIGHTS_DIR / "clip_spatial_features"
OUT_NPY = WEIGHTS_DIR / "clip_features_spatial.npy"

image_names = json.loads((WEIGHTS_DIR / "image_names.json").read_text())
print(f">> images: {len(image_names)}")

# peek shape
first = torch.load(CLIP_DIR / f"{Path(image_names[0]).stem}.pt",
                   map_location="cpu", weights_only=False)
P, D = first.shape
print(f">> per-image: ({P}, {D}) float16 -> total "
      f"{len(image_names) * P * D * 2 / 1e9:.2f} GB")

out = np.lib.format.open_memmap(OUT_NPY, mode="w+", dtype=np.float16,
                                shape=(len(image_names), P, D))
t0 = time.time()
missing = []
for i, name in enumerate(image_names):
    p = CLIP_DIR / f"{Path(name).stem}.pt"
    if not p.exists():
        missing.append(name)
        continue
    out[i] = torch.load(p, map_location="cpu",
                        weights_only=False).numpy().astype(np.float16)
    if (i + 1) % 1000 == 0:
        print(f"  [{i+1}/{len(image_names)}] ({time.time()-t0:.0f}s)", flush=True)
out.flush()
print(f">> missing: {len(missing)} {missing[:5]}")
print(f">> saved {OUT_NPY} ({time.time()-t0:.0f}s)")
sys.exit(1 if missing else 0)