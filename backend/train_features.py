"""
CaptionAI — Stage 2: batched DenseNet201 feature extraction + tokenizer build.

Runs on CPU (TensorFlow). Produces everything the PyTorch GPU training and
the Keras backend need, using EXACTLY the same preprocessing the backend
inference uses (resize 224x224, scale /255, DenseNet201 include_top=False
with pooling="avg" -> 1920-dim features).

Outputs (into ../weights/):
    features.npy        (N, 1920) float32  image features (ordered)
    image_names.json    [str]              image filenames in feature order
    tokenizer.pkl       Keras Tokenizer    for the Keras backend
    word_index.json     {word: id}         plain map for the PyTorch trainer
    train_images.json   [str]              train split image filenames
    val_images.json     [str]              val/test split image filenames
    metadata.json       dict               vocab_size, max_length, splits, ...

Usage:
    python train_features.py [--batch-size 32] [--threads 8]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf  # noqa: E402
from tensorflow.keras.applications import DenseNet201  # noqa: E402
from tensorflow.keras.models import Model  # noqa: E402
from tensorflow.keras.preprocessing.text import Tokenizer  # noqa: E402

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"
WEIGHTS_DIR = PROJECT_DIR / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
ENCODER_FEATURE_DIM = 1920


def text_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["caption"] = df["caption"].str.lower()
    df["caption"] = df["caption"].str.replace(r"[^a-z ]", " ", regex=True)
    df["caption"] = df["caption"].str.replace(r"\s+", " ", regex=True)
    df["caption"] = df["caption"].apply(
        lambda s: " ".join(w for w in s.split() if len(w) > 1)
    )
    df["caption"] = "startseq " + df["caption"] + " endseq"
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Batched DenseNet201 feature extraction")
    ap.add_argument("--images", default=str(DATASET_DIR / "Images"))
    ap.add_argument("--captions", default=str(DATASET_DIR / "captions.txt"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 4))
    ap.add_argument("--limit", type=int, default=0,
                    help="limit number of images (smoke test)")
    args = ap.parse_args()

    if args.threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(2)

    t_start = time.time()
    print(">> loading captions …", flush=True)
    data = pd.read_csv(args.captions)
    data = text_preprocessing(data)
    captions = data["caption"].tolist()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(c.split()) for c in captions)
    print(f">> vocab={vocab_size}  max_length={max_length}", flush=True)

    images = data["image"].unique().tolist()
    if args.limit:
        images = images[: args.limit]
    nimages = len(images)
    print(f">> extracting features for {nimages} images …", flush=True)

    print(">> building DenseNet201 encoder …", flush=True)
    base = DenseNet201(include_top=False, weights="imagenet",
                       input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg")
    encoder = Model(inputs=base.input, outputs=base.output)

    features = np.zeros((nimages, ENCODER_FEATURE_DIM), dtype="float32")
    bs = args.batch_size
    for start in range(0, nimages, bs):
        batch_names = images[start : start + bs]
        arrs = []
        for name in batch_names:
            img = tf.keras.preprocessing.image.load_img(
                os.path.join(args.images, name), target_size=(IMG_SIZE, IMG_SIZE)
            )
            arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            arrs.append(arr)
        batch = np.stack(arrs, axis=0).astype("float32")
        feats = encoder.predict(batch, verbose=0, batch_size=bs)
        features[start : start + bs] = feats
        done = min(start + bs, nimages)
        if done % (bs * 10) == 0 or done == nimages:
            print(f"  {done}/{nimages}  ({time.time() - t_start:.0f}s)", flush=True)

    # 85/15 image-level split, same as the notebook / train.py
    split = round(0.85 * nimages)
    train_images = images[:split]
    val_images = images[split:]
    print(f">> split: {len(train_images)} train / {len(val_images)} val", flush=True)

    np.save(WEIGHTS_DIR / "features.npy", features)
    (WEIGHTS_DIR / "image_names.json").write_text(json.dumps(images))
    (WEIGHTS_DIR / "train_images.json").write_text(json.dumps(train_images))
    (WEIGHTS_DIR / "val_images.json").write_text(json.dumps(val_images))

    with open(WEIGHTS_DIR / "tokenizer.pkl", "wb") as fh:
        pickle.dump(tokenizer, fh)
    (WEIGHTS_DIR / "word_index.json").write_text(
        json.dumps(tokenizer.word_index)
    )

    meta = {
        "vocab_size": vocab_size,
        "max_length": max_length,
        "embed_dim": 256,
        "lstm_units": 256,
        "dense_units": 128,
        "dropout": 0.5,
        "image_size": IMG_SIZE,
        "encoder": "DenseNet201",
        "encoder_feature_dim": ENCODER_FEATURE_DIM,
        "training_dataset": "Flickr8K",
        "n_images": nimages,
        "n_train_images": len(train_images),
        "n_val_images": len(val_images),
        "feature_extraction": "batch32-resize224-scale1over255",
    }
    (WEIGHTS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f">> done in {time.time() - t_start:.0f}s. artefacts under {WEIGHTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())