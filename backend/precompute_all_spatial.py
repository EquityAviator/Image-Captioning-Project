"""
Precompute DenseNet201 spatial features for ALL Flickr8K images.

Run once, then all downstream experiments (FT, focal, RL) can use them.
"""
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))

from tokenizers import Tokenizer

# ─── paths ──────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = PROJECT_DIR / "weights"
IMAGES_DIR = PROJECT_DIR / "dataset" / "Images"
SPATIAL_DIR = WEIGHTS_DIR / "spatial_features"
SPATIAL_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">> device: {device}")

# ─── encoder ────────────────────────────────────────────────────────────
densenet = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
encoder = torch.nn.Sequential(*list(densenet.features.children())[:-1])  # exclude norm5
encoder.eval().to(device)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ─── load all image names ───────────────────────────────────────────────
train_images = json.loads((WEIGHTS_DIR / "train_images.json").read_text())
val_images = json.loads((WEIGHTS_DIR / "val_images.json").read_text())
all_images = train_images + val_images
print(f">> total images to process: {len(all_images)}")

# ─── check already done ─────────────────────────────────────────────────
existing = {p.stem for p in SPATIAL_DIR.glob("*.pt")}
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
        x = transform(img).unsqueeze(0).to(device)

        # (1, 1920, 7, 7) -> (1, 1920, 49) -> (49, 1920)
        feats = encoder(x).flatten(2).transpose(1, 2).squeeze(0).cpu()

        save_path = SPATIAL_DIR / f"{Path(img_name).stem}.pt"
        torch.save(feats, save_path)

        if i % 100 == 0 or i == len(to_process):
            print(f"  [{i}/{len(to_process)}] saved {img_name} -> {feats.shape}")

print(">> done")