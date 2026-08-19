#!/usr/bin/env python3
"""
Enhanced training script for CaptionAI — trains the exact notebook architecture
on the local Flickr8K dataset and saves model.h5 + tokenizer.pkl + metadata.json
into ../weights/.

Features:
- Extracts image features using DenseNet201 (frozen, ImageNet weights)
- Preprocesses captions (lowercase, remove non-alpha, startseq/endseq tokens)
- Trains LSTM decoder with teacher forcing
- Validates using BLEU scores
- Saves all artifacts for production inference

Usage:
    python train_local.py [--epochs N] [--batch-size N] [--img-size N]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Quiet TF logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf  # noqa: E402
from tensorflow.keras.applications import DenseNet201  # noqa: E402
from tensorflow.keras.callbacks import (  # noqa: E402
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (  # noqa: E402
    Add,
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    Input,
    LSTM,
    Reshape,
)
from tensorflow.keras.models import Model  # noqa: E402
from tensorflow.keras.preprocessing.image import img_to_array, load_img  # noqa: E402
from tensorflow.keras.preprocessing.sequence import pad_sequences  # noqa: E402
from tensorflow.keras.preprocessing.text import Tokenizer  # noqa: E402
from tensorflow.keras.utils import Sequence, to_categorical  # noqa: E402

# Allow backend imports
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))
from utils.paths import (  # noqa: E402
    METADATA_JSON,
    MODEL_H5,
    TOKENIZER_PKL,
    WEIGHTS_DIR,
)

# Default dataset paths (local)
DEFAULT_IMAGES_DIR = BACKEND_DIR.parent / "dataset" / "Images"
DEFAULT_CAPTIONS_FILE = BACKEND_DIR.parent / "dataset" / "captions.txt"

# BLEU score calculation
def compute_bleu(reference_captions: list[str], candidate_caption: str, max_n: int = 4) -> float:
    """Compute BLEU score for a single candidate against multiple references."""
    from collections import Counter
    import math
    
    # Tokenize
    ref_tokens_list = [ref.lower().split() for ref in reference_captions]
    cand_tokens = candidate_caption.lower().split()
    
    if not cand_tokens:
        return 0.0
    
    # Modified n-gram precision
    precisions = []
    for n in range(1, max_n + 1):
        if len(cand_tokens) < n:
            precisions.append(0.0)
            continue
            
        # Candidate n-grams
        cand_ngrams = Counter(tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens) - n + 1))
        if not cand_ngrams:
            precisions.append(0.0)
            continue
            
        # Reference n-grams (max count across references)
        ref_ngrams_max = Counter()
        for ref_tokens in ref_tokens_list:
            ref_ngrams = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1))
            for ng, count in ref_ngrams.items():
                ref_ngrams_max[ng] = max(ref_ngrams_max[ng], count)
        
        # Clip counts
        clipped = sum(min(count, ref_ngrams_max[ng]) for ng, count in cand_ngrams.items())
        total = sum(cand_ngrams.values())
        precisions.append(clipped / total if total > 0 else 0.0)
    
    # Brevity penalty
    ref_len = min(len(ref) for ref in ref_tokens_list)
    cand_len = len(cand_tokens)
    bp = 1.0 if cand_len > ref_len else math.exp(1 - ref_len / cand_len) if cand_len > 0 else 0.0
    
    # Geometric mean of precisions
    if all(p > 0 for p in precisions):
        log_sum = sum(math.log(p) for p in precisions) / max_n
        bleu = bp * math.exp(log_sum)
    else:
        bleu = 0.0
    
    return bleu


def text_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize captions."""
    df = df.copy()
    df["caption"] = df["caption"].str.lower()
    df["caption"] = df["caption"].str.replace(r"[^a-z ]", " ", regex=True)
    df["caption"] = df["caption"].str.replace(r"\s+", " ", regex=True)
    df["caption"] = df["caption"].apply(
        lambda s: " ".join(w for w in s.split() if len(w) > 1)
    )
    df["caption"] = "startseq " + df["caption"] + " endseq"
    return df


class CustomDataGenerator(Sequence):
    """Keras Sequence generator for training data."""
    
    def __init__(self, df, X_col, y_col, batch_size, directory, tokenizer,
                 vocab_size, max_length, features, shuffle=True):
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.directory = directory
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.features = features
        self.shuffle = shuffle
        self.n = len(self.df)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return self.n // self.batch_size

    def __getitem__(self, index):
        batch = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size]
        X1, X2, y = [], [], []
        for image in batch[self.X_col].tolist():
            feature = self.features[image][0]
            caps = batch.loc[batch[self.X_col] == image, self.y_col].tolist()
            for cap in caps:
                seq = self.tokenizer.texts_to_sequences([cap])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)
        return (np.array(X1), np.array(X2)), np.array(y)


def greedy_decode(model, tokenizer, feature, max_length):
    """Generate caption using greedy decoding."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    in_text = "startseq"
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length, padding="post")
        y_pred = model.predict([feature, seq], verbose=0, batch_size=1)[0]
        idx = int(np.argmax(y_pred))
        word = None
        for w, index in tokenizer.word_index.items():
            if index == idx:
                word = w
                break
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    return caption


def validate_model(model, tokenizer, val_data, features, max_length, n_samples=100):
    """Compute BLEU scores on validation set."""
    print(f">> validating on {min(n_samples, len(val_data))} samples...")
    
    # Group validation captions by image
    image_to_captions = {}
    for _, row in val_data.iterrows():
        img = row["image"]
        cap = row["caption"].replace("startseq ", "").replace(" endseq", "")
        if img not in image_to_captions:
            image_to_captions[img] = []
        image_to_captions[img].append(cap)
    
    images = list(image_to_captions.keys())[:n_samples]
    bleu_scores = {f"bleu-{i}": [] for i in range(1, 5)}
    
    for img in tqdm(images, desc="BLEU validation"):
        refs = image_to_captions[img]
        feature = features[img]
        pred = greedy_decode(model, tokenizer, feature, max_length)
        
        for i in range(1, 5):
            score = compute_bleu(refs, pred, max_n=i)
            bleu_scores[f"bleu-{i}"].append(score)
    
    avg_bleu = {k: float(np.mean(v)) for k, v in bleu_scores.items()}
    print(f">> BLEU scores: {avg_bleu}")
    return avg_bleu


def main() -> int:
    ap = argparse.ArgumentParser(description="Train CaptionAI on local Flickr8K dataset")
    ap.add_argument("--images", default=str(DEFAULT_IMAGES_DIR), help="path to Flickr8K Images dir")
    ap.add_argument("--captions", default=str(DEFAULT_CAPTIONS_FILE), help="path to captions.txt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--validate", action="store_true", help="Run BLEU validation after training")
    args = ap.parse_args()

    # Verify dataset exists
    if not Path(args.images).exists():
        print(f"ERROR: Images directory not found: {args.images}")
        return 1
    if not Path(args.captions).exists():
        print(f"ERROR: Captions file not found: {args.captions}")
        return 1

    print(">> loading captions …")
    data = pd.read_csv(args.captions)
    print(f">> loaded {len(data)} caption entries for {data['image'].nunique()} images")
    
    data = text_preprocessing(data)
    captions = data["caption"].tolist()

    print(">> building tokenizer …")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(c.split()) for c in captions)
    print(f">> vocab={vocab_size}  max_length={max_length}")

    images = data["image"].unique().tolist()
    nimages = len(images)
    split = round(0.85 * nimages)
    train_imgs = images[:split]
    val_imgs = images[split:]
    train = data[data["image"].isin(train_imgs)].reset_index(drop=True)
    val = data[data["image"].isin(val_imgs)].reset_index(drop=True)
    print(f">> train images: {len(train_imgs)}, val images: {len(val_imgs)}")

    print(">> building DenseNet201 encoder (frozen, ImageNet weights) …")
    base = DenseNet201(include_top=False, weights="imagenet",
                       input_shape=(args.img_size, args.img_size, 3), pooling="avg")
    encoder = Model(inputs=base.input, outputs=base.output)

    print(">> pre-computing image features (one-time pass) …")
    features = {}
    for img_name in tqdm(images, desc="Extracting features"):
        img_path = os.path.join(args.images, img_name)
        if not os.path.exists(img_path):
            print(f"WARNING: Image not found: {img_path}")
            continue
        img = load_img(img_path, target_size=(args.img_size, args.img_size))
        arr = img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        features[img_name] = encoder.predict(arr, verbose=0)

    print(">> building decoder (exact notebook architecture) …")
    input1 = Input(shape=(1920,))
    input2 = Input(shape=(max_length,))
    img_features = Dense(256, activation="relu")(input1)
    img_features_reshaped = Reshape((1, 256))(img_features)
    sentence_features = Embedding(vocab_size, 256, mask_zero=False)(input2)
    merged = Concatenate(axis=1)([img_features_reshaped, sentence_features])
    sentence_features = LSTM(256)(merged)
    x = Dropout(0.5)(sentence_features)
    x = Add()([x, img_features])
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(vocab_size, activation="softmax")(x)
    caption_model = Model(inputs=[input1, input2], outputs=output)
    caption_model.compile(loss="categorical_crossentropy", optimizer="adam")
    
    caption_model.summary()

    train_gen = CustomDataGenerator(train, "image", "caption", args.batch_size,
                                    args.images, tokenizer, vocab_size,
                                    max_length, features)
    val_gen = CustomDataGenerator(val, "image", "caption", args.batch_size,
                                  args.images, tokenizer, vocab_size,
                                  max_length, features, shuffle=False)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(str(MODEL_H5), monitor="val_loss", mode="min",
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, verbose=1,
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=3, verbose=1,
                          factor=0.2, min_lr=1e-8),
    ]

    print(f">> training for {args.epochs} epochs (batch size: {args.batch_size}) …")
    history = caption_model.fit(
        train_gen, validation_data=val_gen,
        epochs=args.epochs, callbacks=callbacks,
    )

    print(">> saving tokenizer + metadata …")
    with open(TOKENIZER_PKL, "wb") as fh:
        pickle.dump(tokenizer, fh)
    
    # Save tokenizer as JSON too (alternative format)
    TOKENIZER_JSON = WEIGHTS_DIR / "tokenizer.json"
    TOKENIZER_JSON.write_text(tokenizer.to_json())
    
    bleu_scores = {}
    if args.validate:
        bleu_scores = validate_model(caption_model, tokenizer, val, features, max_length)

    metadata = {
        "vocab_size": vocab_size,
        "max_length": max_length,
        "embed_dim": 256,
        "lstm_units": 256,
        "dense_units": 128,
        "dropout": 0.5,
        "image_size": args.img_size,
        "encoder": "DenseNet201",
        "training_dataset": "Flickr8K",
        "epochs_trained": len(history.history["loss"]),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "bleu_scores": bleu_scores,
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2))
    
    print(f">> done. weights saved under {WEIGHTS_DIR}")
    print(f">> model: {MODEL_H5}")
    print(f">> tokenizer: {TOKENIZER_PKL} (and {TOKENIZER_JSON})")
    print(f">> metadata: {METADATA_JSON}")
    if bleu_scores:
        print(f">> BLEU scores: {bleu_scores}")
    return 0


if __name__ == "__main__":
    sys.exit(main())