#!/usr/bin/env python3
"""
Environment check script for CaptionAI improvement project.
Verifies GPU availability, package versions, and system resources.
"""

import sys
import platform
import subprocess
import json
from pathlib import Path


def check_python():
    print("=" * 60)
    print("PYTHON ENVIRONMENT")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    print(f"Processor: {platform.processor()}")


def check_torch():
    print("\n" + "=" * 60)
    print("PYTORCH")
    print("=" * 60)
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"Device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  Device {i}: {props.name}")
                print(f"    Capability: {props.major}.{props.minor}")
                print(f"    Total memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"    Multi-processor count: {props.multi_processor_count}")
            print(f"Current allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"Current reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        else:
            print("WARNING: CUDA not available!")
    except ImportError:
        print("PyTorch not installed")


def check_torchvision():
    print("\n" + "=" * 60)
    print("TORCHVISION")
    print("=" * 60)
    try:
        import torchvision
        print(f"Torchvision version: {torchvision.__version__}")
    except ImportError:
        print("Torchvision not installed")


def check_tensorflow():
    print("\n" + "=" * 60)
    print("TENSORFLOW")
    print("=" * 60)
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
        print(f"CPU devices: {tf.config.list_physical_devices('CPU')}")
    except ImportError:
        print("TensorFlow not installed")


def check_onnx():
    print("\n" + "=" * 60)
    print("ONNX RUNTIME")
    print("=" * 60)
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime version: {ort.__version__}")
        print(f"Available providers: {ort.get_available_providers()}")
    except ImportError:
        print("ONNX Runtime not installed")


def check_other_packages():
    print("\n" + "=" * 60)
    print("OTHER PACKAGES")
    print("=" * 60)
    packages = [
        "nltk", "rouge_score", "tqdm", "pandas", "pillow",
        "matplotlib", "numpy", "scipy", "sklearn"
    ]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            print(f"  {pkg}: {version}")
        except ImportError:
            print(f"  {pkg}: NOT INSTALLED")


def check_system():
    print("\n" + "=" * 60)
    print("SYSTEM RESOURCES")
    print("=" * 60)
    import os
    print(f"CPU count: {os.cpu_count()}")
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"RAM total: {mem.total / 1024**3:.2f} GB")
        print(f"RAM available: {mem.available / 1024**3:.2f} GB")
        print(f"RAM used: {mem.percent}%")
    except ImportError:
        print("psutil not installed, skipping memory info")


def check_dataset():
    print("\n" + "=" * 60)
    print("DATASET CHECK")
    print("=" * 60)
    project_dir = Path(__file__).resolve().parent.parent
    dataset_dir = project_dir / "dataset"
    images_dir = dataset_dir / "Images"
    captions_file = dataset_dir / "captions.txt"
    
    print(f"Dataset dir: {dataset_dir}")
    print(f"Images dir exists: {images_dir.exists()}")
    print(f"Captions file exists: {captions_file.exists()}")
    
    if images_dir.exists():
        img_count = len(list(images_dir.glob("*.jpg")))
        print(f"Image count: {img_count}")
    
    if captions_file.exists():
        with open(captions_file) as f:
            lines = f.readlines()
        print(f"Caption lines: {len(lines)}")


def check_weights():
    print("\n" + "=" * 60)
    print("WEIGHTS DIRECTORY")
    print("=" * 60)
    project_dir = Path(__file__).resolve().parent.parent
    weights_dir = project_dir / "weights"
    print(f"Weights dir: {weights_dir}")
    print(f"Exists: {weights_dir.exists()}")
    
    if weights_dir.exists():
        for f in weights_dir.iterdir():
            size_mb = f.stat().st_size / 1024**2
            print(f"  {f.name}: {size_mb:.2f} MB")


def main():
    print("CAPTIONAI ENVIRONMENT CHECK")
    print("Project: Image Captioning Improvement")
    print("Branch: feature/local-caption-improvement")
    
    check_python()
    check_torch()
    check_torchvision()
    check_tensorflow()
    check_onnx()
    check_other_packages()
    check_system()
    check_dataset()
    check_weights()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    import torch
    gpu_ok = torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (6, 1)
    print(f"GPU (GTX 1080, capability 6.1): {'PASS' if gpu_ok else 'FAIL'}")
    
    if not gpu_ok:
        print("\nERROR: GPU not available or incorrect capability!")
        print("Stopping Phase 0 - GPU is required for training.")
        sys.exit(1)
    
    print("\nPhase 0 environment check PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())