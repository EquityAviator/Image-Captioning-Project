"""Path helpers — centralised so the rest of the code never hard-codes paths."""

from __future__ import annotations

from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BACKEND_DIR.parent / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# notebook model weights (Keras .h5)
MODEL_H5 = WEIGHTS_DIR / "model.h5"
MODEL_KERAS = WEIGHTS_DIR / "model.keras"

# tokenizer pickle (saved from training)
TOKENIZER_PKL = WEIGHTS_DIR / "tokenizer.pkl"
TOKENIZER_JSON = WEIGHTS_DIR / "tokenizer.json"

# training metadata (max_length, vocab_size, etc.)
METADATA_JSON = WEIGHTS_DIR / "metadata.json"

# DenseNet201 ImageNet weights cache
DENSENET_CACHE = WEIGHTS_DIR / "densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5"
