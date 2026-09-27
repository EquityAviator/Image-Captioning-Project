"""
Precompute CLIP ViT-B/16 spatial features for ALL Flickr8K images.

Run once, then train attention decoder on CLIP features.
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))

try:
    import open_clip
except ImportError:
    print("Installing open_clip...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "open_clip_torch"])
    import open_clip

# ─── paths ──────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = PROJECT_DIR / "weights"
IMAGES_DIR = PROJECT_DIR / "dataset" / "Images"
CLIP_SPATIAL_DIR = WEIGHTS_DIR / "clip_spatial_features"
CLIP_SPATIAL_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">> device: {device}")

# ─── CLIP encoder ───────────────────────────────────────────────────────
print(">> loading CLIP ViT-B/16...")
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-16', pretrained='openai', device=device
)
model.eval()

# Extract visual encoder (patches before final projection)
visual = model.visual
# ViT-B/16: 14x14 patches -> 196, 768-d
# We want spatial features: (196, 768) per image

# ─── load all image names ───────────────────────────────────────────────
train_images = json.loads((WEIGHTS_DIR / "train_images.json").read_text())
val_images = json.loads((WEIGHTS_DIR / "val_images.json").read_text())
all_images = train_images + val_images
print(f">> total images to process: {len(all_images)}")

# ─── check already done ─────────────────────────────────────────────────
existing = {p.stem for p in CLIP_SPATIAL_DIR.glob("*.pt")}
to_process = [img for img in all_images if Path(img).stem not in existing]
print(f">> already have: {len(existing)} | need: {len(to_process)}")

# ─── extract ────────────────────────────────────────────────────────────
with torch.no_grad():
    for i, img_name in enumerate(to_process, 1):
        img_path = IMAGES_DIR / img_name
        if not img_path.exists():
            print(f"  [{i}/{len(to_process)}] MISSING: {img_name}")
            continue

        img = Image.open(img_path).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)

        # Forward through CLIP visual encoder to get patch embeddings
        # forward_intermediates returns dict with 'image_intermediates' list
        # containing final block output as spatial map (1, 768, 14, 14)
        out = visual.forward_intermediates(x, indices=[-1])
        feats = out['image_intermediates'][0]  # (1, 768, 14, 14)

        # Flatten spatial map -> patch sequence: (196, 768)
        spatial_feats = feats.flatten(2).transpose(1, 2).squeeze(0).cpu()

        save_path = CLIP_SPATIAL_DIR / f"{Path(img_name).stem}.pt"
        torch.save(spatial_feats, save_path)

        if i % 500 == 0 or i == len(to_process):
            print(f"  [{i}/{len(to_process)}] saved {img_name} -> {spatial_feats.shape}")

print(">> done")