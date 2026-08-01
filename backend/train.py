"""
Training script — trains the notebook's exact architecture on Flickr8K
and saves model.h5 + tokenizer.pkl + metadata.json into ../weights/.

Usage
-----
    python train.py --images /path/to/flickr8k/Images \
                    --captions /path/to/flickr8k/captions.txt \
                    --epochs 20 --batch-size 32

This script is intentionally a faithful port of the notebook — it does NOT
introduce new modelling ideas. Run it once on a GPU machine, then copy the
resulting weights/ folder next to the backend deployment.
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

# allow backend imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.paths import (  # noqa: E402
    METADATA_JSON,
    MODEL_H5,
    TOKENIZER_PKL,
    WEIGHTS_DIR,
)


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


class CustomDataGenerator(Sequence):
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Train CaptionAI on Flickr8K")
    ap.add_argument("--images", required=True, help="path to Flickr8K Images dir")
    ap.add_argument("--captions", required=True, help="path to captions.txt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()

    print(">> loading captions …")
    data = pd.read_csv(args.captions)
    data = text_preprocessing(data)
    captions = data["caption"].tolist()

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

    print(">> building DenseNet201 encoder …")
    base = DenseNet201(include_top=False, weights="imagenet",
                       input_shape=(args.img_size, args.img_size, 3), pooling="avg")
    encoder = Model(inputs=base.input, outputs=base.output)

    print(">> pre-computing image features (one-time pass) …")
    features = {}
    for img_name in tqdm(images):
        img = load_img(os.path.join(args.images, img_name),
                       target_size=(args.img_size, args.img_size))
        arr = img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        features[img_name] = encoder.predict(arr, verbose=0)

    print(">> building decoder (notebook cell 20) …")
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

    print(">> training …")
    history = caption_model.fit(
        train_gen, validation_data=val_gen,
        epochs=args.epochs, callbacks=callbacks,
    )

    print(">> saving tokenizer + metadata …")
    with open(TOKENIZER_PKL, "wb") as fh:
        pickle.dump(tokenizer, fh)
    METADATA_JSON.write_text(json.dumps({
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
    }, indent=2))
    print(f">> done. weights saved under {WEIGHTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
