#!/usr/bin/env python3
"""
Phase 2: Fix Image Preprocessing

Implements correct ImageNet normalization for DenseNet201:
- Resize 224x224
- ToTensor (0-1 range)
- Normalize with ImageNet mean/std: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

Outputs:
- features_v2_global.npy (global average pooled, 1920-dim) - for baseline LSTM
- features_v2_spatial.npy (spatial 7x7x1920) - for attention decoder
- image_id_to_index.json mapping
- preprocessing_metadata.json

Both global and spatial features are saved to support Phase 4 (baseline) and Phase 5 (attention).
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
WEIGHTS_DIR = PROJECT_DIR / "weights"
OUTPUT_DIR = PROJECT_DIR / "experiments" / "fixed_preprocessing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Correct ImageNet preprocessing for DenseNet201
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def build_dense_encoder_spatial():
    """Build DenseNet201 encoder that outputs spatial features (7x7x1920)"""
    densenet = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
    # Remove classifier and final pooling - keep feature maps before pooling
    # DenseNet201: features -> norm5 -> relu -> pool -> classifier
    # We want output of norm5 (before pooling): (B, 1920, 7, 7)
    encoder = nn.Sequential(*list(densenet.features.children())[:-1])  # Remove final pooling
    encoder.eval()
    return encoder.to(DEVICE)


def build_dense_encoder_global():
    """Build DenseNet201 encoder with global average pooling (1920-dim)"""
    densenet = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
    # Use the full feature extractor with pooling
    encoder = nn.Sequential(
        densenet.features,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(1)
    )
    encoder.eval()
    return encoder.to(DEVICE)


def check_preprocessing():
    """Verify preprocessing with sample images"""
    print("=" * 60)
    print("PREPROCESSING VERIFICATION")
    print("=" * 60)
    
    # Load a few sample images
    image_names = json.loads((WEIGHTS_DIR / "image_names.json").read_text())
    sample_names = image_names[:5]
    
    encoder_spatial = build_dense_encoder_spatial()
    encoder_global = build_dense_encoder_global()
    
    print(f"Device: {DEVICE}")
    print(f"Sample images: {sample_names}")
    print()
    
    for name in sample_names:
        img_path = DATASET_DIR / "Images" / name
        img = Image.open(img_path).convert("RGB")
        
        # Apply preprocessing
        tensor = preprocess(img).unsqueeze(0).to(DEVICE)  # (1, 3, 224, 224)
        
        print(f"\nImage: {name}")
        print(f"  Tensor shape: {tensor.shape}")
        print(f"  Range: [{tensor.min():.4f}, {tensor.max():.4f}]")
        print(f"  Mean per channel: {tensor.mean(dim=[0,2,3]).tolist()}")
        print(f"  Std per channel: {tensor.std(dim=[0,2,3]).tolist()}")
        
        # Get spatial features
        with torch.no_grad():
            spatial_feat = encoder_spatial(tensor)  # (1, 1920, 7, 7)
            global_feat = encoder_global(tensor)    # (1, 1920)
        
        print(f"  Spatial features: {spatial_feat.shape} (mean={spatial_feat.mean():.4f}, std={spatial_feat.std():.4f})")
        print(f"  Global features: {global_feat.shape} (mean={global_feat.mean():.4f}, std={global_feat.std():.4f})")
    
    print("\n[OK] Preprocessing verification complete")


def extract_features_batch(image_names, images_dir, encoder_spatial, encoder_global, batch_size=32):
    """Extract both spatial and global features for a batch of images"""
    spatial_features = []
    global_features = []
    
    for i in range(0, len(image_names), batch_size):
        batch_names = image_names[i:i+batch_size]
        batch_tensors = []
        
        for name in batch_names:
            img_path = images_dir / name
            img = Image.open(img_path).convert("RGB")
            tensor = preprocess(img)
            batch_tensors.append(tensor)
        
        batch = torch.stack(batch_tensors).to(DEVICE)  # (B, 3, 224, 224)
        
        with torch.no_grad():
            spatial = encoder_spatial(batch)  # (B, 1920, 7, 7)
            global_feat = encoder_global(batch)  # (B, 1920)
        
        spatial_features.append(spatial.cpu().numpy())
        global_features.append(global_feat.cpu().numpy())
        
        if (i + batch_size) % (batch_size * 10) == 0 or i + batch_size >= len(image_names):
            print(f"  Processed {min(i + batch_size, len(image_names))}/{len(image_names)}")
    
    spatial_features = np.concatenate(spatial_features, axis=0)  # (N, 1920, 7, 7)
    global_features = np.concatenate(global_features, axis=0)    # (N, 1920)
    
    return spatial_features, global_features


def main():
    print("PHASE 2: FIX IMAGE PREPROCESSING")
    print("=" * 60)
    
    # Load image list
    image_names = json.loads((WEIGHTS_DIR / "image_names.json").read_text())
    images_dir = DATASET_DIR / "Images"
    
    print(f"Total images: {len(image_names)}")
    print(f"Images directory: {images_dir}")
    print(f"Device: {DEVICE}")
    
    # 1. Verify preprocessing
    check_preprocessing()
    
    # 2. Build encoders
    print("\n" + "=" * 60)
    print("BUILDING ENCODERS")
    print("=" * 60)
    encoder_spatial = build_dense_encoder_spatial()
    encoder_global = build_dense_encoder_global()
    
    # Count parameters
    spatial_params = sum(p.numel() for p in encoder_spatial.parameters())
    global_params = sum(p.numel() for p in encoder_global.parameters())
    print(f"Spatial encoder params: {spatial_params:,}")
    print(f"Global encoder params: {global_params:,}")
    
    # 3. Extract features
    print("\n" + "=" * 60)
    print("EXTRACTING FEATURES")
    print("=" * 60)
    t_start = time.time()
    
    spatial_features, global_features = extract_features_batch(
        image_names, images_dir, encoder_spatial, encoder_global, batch_size=32
    )
    
    print(f"\nFeature extraction completed in {time.time() - t_start:.1f}s")
    print(f"Spatial features shape: {spatial_features.shape}")  # (N, 1920, 7, 7)
    print(f"Global features shape: {global_features.shape}")    # (N, 1920)
    
    # 4. Save features - ensure float32
    print("\n" + "=" * 60)
    print("SAVING FEATURES")
    print("=" * 60)
    
    spatial_f32 = spatial_features.astype(np.float32)
    global_f32 = global_features.astype(np.float32)
    
    print(f"Spatial: {spatial_f32.shape} -> {spatial_f32.nbytes / 1024**2:.1f} MB")
    print(f"Global: {global_f32.shape} -> {global_f32.nbytes / 1024**2:.1f} MB")
    
    # Save global features (always needed)
    np.save(OUTPUT_DIR / "features_v2_global.npy", global_f32)
    np.save(WEIGHTS_DIR / "features_v2_global.npy", global_f32)
    print(f"[OK] Global features saved ({global_f32.nbytes / 1024**2:.1f} MB)")
    
    # Save spatial features in compressed format (npz)
    # This is large (~2.9 GB uncompressed), so use compression
    try:
        np.savez_compressed(OUTPUT_DIR / "features_v2_spatial.npz", features=spatial_f32)
        np.savez_compressed(WEIGHTS_DIR / "features_v2_spatial.npz", features=spatial_f32)
        print(f"[OK] Spatial features saved (compressed)")
    except Exception as e:
        print(f"[WARN] Could not save spatial features: {e}")
        print("  Will compute on-the-fly during attention training")
    
    # Save image_id to index mapping
    image_id_to_index = {name: idx for idx, name in enumerate(image_names)}
    (OUTPUT_DIR / "image_id_to_index.json").write_text(json.dumps(image_id_to_index))
    (WEIGHTS_DIR / "image_id_to_index.json").write_text(json.dumps(image_id_to_index))
    
    print(f"[OK] Saved global features + mappings")
    
    # 5. Create preprocessing metadata
    preprocessing_metadata = {
        "framework": "pytorch",
        "image_size": IMG_SIZE,
        "resize_method": "bilinear",
        "crop_method": "resize_only",
        "normalization_mean": IMAGENET_MEAN,
        "normalization_std": IMAGENET_STD,
        "old_preprocessing": "resize224-scale1over255",
        "new_preprocessing": "resize224-imagenet_norm",
        "encoder": "DenseNet201",
        "encoder_weights": "IMAGENET1K_V1",
        "spatial_feature_shape": [1920, 7, 7],
        "global_feature_dim": 1920,
        "feature_extraction_batch_size": 32,
        "total_images": len(image_names),
        "device": str(DEVICE)
    }
    
    (OUTPUT_DIR / "preprocessing_metadata.json").write_text(json.dumps(preprocessing_metadata, indent=2))
    (WEIGHTS_DIR / "preprocessing_metadata.json").write_text(json.dumps(preprocessing_metadata, indent=2))
    
    print(f"\n[OK] Preprocessing metadata saved")
    print(f"\nPhase 2 COMPLETE")
    return 0


if __name__ == "__main__":
    import json
    sys.exit(main())