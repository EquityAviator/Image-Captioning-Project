"""
CaptionAI — Stage 3: decoder training on GPU (PyTorch).

Trains the EXACT notebook architecture (cell 20) in PyTorch:
    Dense(256, relu) -> Reshape(1,256) -> concat(Embedding(vocab,256))
    -> LSTM(256) -> Dropout(0.5) -> Add(residual img feats)
    -> Dense(128, relu) -> Dropout(0.5) -> Dense(vocab, softmax)

Improved training recipe vs the notebook:
    - Adam lr=1e-3 with gradient clipping (clipnorm=5.0)
    - label smoothing 0.1
    - 100 epochs, EarlyStopping patience 10, ReduceLROnPlateau factor 0.2
    - teacher forcing with the same (prefix -> next-token) pairs as notebook
      cell 17, but precomputed once and batched vectorised for the GPU.

Consumes artefacts from train_features.py:
    weights/features.npy, image_names.json, word_index.json,
    train_images.json, val_images.json, metadata.json

Outputs:
    weights/decoder_torch.pt   (torch checkpoint of the best epoch)
    weights/metadata.json      (updated with real training losses)

Usage:
    python train_torch.py [--epochs 100] [--batch-size 512] [--limit 0]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"
WEIGHTS_DIR = PROJECT_DIR.parent / "weights"

EMBED_DIM = 256
LSTM_UNITS = 256
DENSE_UNITS = 128
DROPOUT = 0.5
FEATURE_DIM = 1920


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


class CaptionDecoder(nn.Module):
    """Exact mirror of notebook cell 20, in PyTorch."""

    def __init__(self, vocab_size: int, max_length: int,
                 dropout: float = DROPOUT) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.img_fc = nn.Linear(FEATURE_DIM, EMBED_DIM)          # Dense(256, relu)
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)     # Embedding(vocab, 256)
        self.lstm = nn.LSTM(EMBED_DIM, LSTM_UNITS, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.fc128 = nn.Linear(LSTM_UNITS, DENSE_UNITS)          # Dense(128, relu)
        self.dropout2 = nn.Dropout(DROPOUT)
        self.out = nn.Linear(DENSE_UNITS, vocab_size)            # Dense(vocab, softmax)

    def forward(self, features: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        img_feats = torch.relu(self.img_fc(features))            # (B, 256)
        img_seq = img_feats.unsqueeze(1)                         # (B, 1, 256)
        emb = self.embedding(tokens)                             # (B, T, 256)
        merged = torch.cat([img_seq, emb], dim=1)                # (B, T+1, 256)
        lstm_out, _ = self.lstm(merged)                          # (B, T+1, 256)
        h = lstm_out[:, -1, :]                                   # last timestep
        h = self.dropout1(h)
        h = h + img_feats                                        # residual add
        h = torch.relu(self.fc128(h))
        h = self.dropout2(h)
        return self.out(h)                                       # (B, vocab)


def build_pairs(
    images: list[str],
    captions_by_image: dict[str, list[list[int]]],
    feat_index: dict[str, int],
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute (feature_index, padded_prefix, target) arrays for a split.

    Mirrors the notebook generator: for every caption position i in 1..len(seq),
    a training pair (prefix=seq[:i], target=seq[i]) is created.
    """
    feat_ids: list[int] = []
    prefixes: list[np.ndarray] = []
    targets: list[int] = []
    for img in images:
        fi = feat_index[img]
        for seq in captions_by_image[img]:
            for i in range(1, len(seq)):
                prefix = np.zeros(max_length, dtype="int64")
                prefix[:i] = np.asarray(seq[:i], dtype="int64")
                feat_ids.append(fi)
                prefixes.append(prefix)
                targets.append(seq[i])
    if not feat_ids:
        return (np.zeros((0,), dtype="int64"),
                np.zeros((0, max_length), dtype="int64"),
                np.zeros((0,), dtype="int64"))
    return (np.asarray(feat_ids, dtype="int64"),
            np.stack(prefixes),
            np.asarray(targets, dtype="int64"))


def main() -> int:
    ap = argparse.ArgumentParser(description="PyTorch GPU decoder training")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=512, help="pairs per step")
    ap.add_argument("--limit", type=int, default=0,
                    help="limit images per split (smoke test)")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clipnorm", type=float, default=5.0)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--rlr-patience", type=int, default=3)
    ap.add_argument("--rlr-factor", type=float, default=0.2)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--dropout", type=float, default=DROPOUT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})", flush=True)

    # ---- load artefacts -------------------------------------------------
    meta = json.loads((WEIGHTS_DIR / "metadata.json").read_text())
    vocab_size = int(meta["vocab_size"])
    max_length = int(meta["max_length"])
    word_index = json.loads((WEIGHTS_DIR / "word_index.json").read_text())
    image_names = json.loads((WEIGHTS_DIR / "image_names.json").read_text())
    train_images = json.loads((WEIGHTS_DIR / "train_images.json").read_text())
    val_images = json.loads((WEIGHTS_DIR / "val_images.json").read_text())
    features = np.load(WEIGHTS_DIR / "features.npy")
    feat_index = {name: i for i, name in enumerate(image_names)}
    print(f">> vocab={vocab_size} max_length={max_length} "
          f"features={features.shape}", flush=True)

    if args.limit:
        train_images = train_images[: args.limit]
        val_images = val_images[: args.limit]

    # ---- tokenise captions ---------------------------------------------
    data = pd.read_csv(DATASET_DIR / "captions.txt")
    data = text_preprocessing(data)
    captions_by_image: dict[str, list[list[int]]] = {}
    for _, row in data.iterrows():
        seq = [word_index[w] for w in row["caption"].split() if w in word_index]
        if not seq:
            continue
        captions_by_image.setdefault(row["image"], []).append(seq)

    t_prep = time.time()
    train_ids, train_x, train_y = build_pairs(
        train_images, captions_by_image, feat_index, max_length)
    val_ids, val_x, val_y = build_pairs(
        val_images, captions_by_image, feat_index, max_length)
    print(f">> pairs: {len(train_y):,} train / {len(val_y):,} val "
          f"(built in {time.time() - t_prep:.0f}s)", flush=True)

    # ---- model ----------------------------------------------------------
    model = CaptionDecoder(vocab_size, max_length, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f">> decoder params: {n_params:,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.rlr_factor, patience=args.rlr_patience,
        min_lr=args.min_lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def run_epoch(ids, x, y, shuffle: bool) -> float:
        n = len(y)
        order = np.random.permutation(n) if shuffle else np.arange(n)
        total, count = 0.0, 0
        bs = args.batch_size
        for start in range(0, n, bs):
            sel = order[start : start + bs]
            f = torch.tensor(features[ids[sel]], device=device, dtype=torch.float32)
            t = torch.tensor(x[sel], device=device)
            tg = torch.tensor(y[sel], device=device)
            logits = model(f, t)
            loss = criterion(logits, tg)
            if shuffle:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipnorm)
                optimizer.step()
            total += loss.item() * len(sel)
            count += len(sel)
        return total / max(count, 1)

    # ---- training loop --------------------------------------------------
    best_val = float("inf")
    best_epoch = -1
    patience_counter = 0
    history: dict[str, list[float]] = {"loss": [], "val_loss": []}
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = run_epoch(train_ids, train_x, train_y, shuffle=True)
        model.eval()
        with torch.no_grad():
            val_loss = run_epoch(val_ids, val_x, val_y, shuffle=False)
        history["loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  "
              f"lr {lr_now:.2e}  ({time.time() - t_start:.0f}s)", flush=True)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {"state_dict": model.state_dict(),
                 "config": {"vocab_size": vocab_size, "max_length": max_length,
                            "embed_dim": EMBED_DIM, "lstm_units": LSTM_UNITS,
                            "dense_units": DENSE_UNITS, "dropout": args.dropout}},
                WEIGHTS_DIR / "decoder_torch.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f">> early stopping at epoch {epoch} "
                      f"(best epoch {best_epoch}, val_loss {best_val:.4f})", flush=True)
                break

    print(f">> training done in {time.time() - t_start:.0f}s. "
          f"best val_loss {best_val:.4f} (epoch {best_epoch})", flush=True)

    # ---- update metadata ------------------------------------------------
    meta["training"] = {
        "framework": "pytorch",
        "epochs_trained": len(history["loss"]),
        "best_epoch": best_epoch,
        "final_train_loss": history["loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "best_val_loss": round(best_val, 4),
        "batch_size": args.batch_size,
        "optimizer": "adam",
        "lr": args.lr,
        "clipnorm": args.clipnorm,
        "label_smoothing": args.label_smoothing,
        "patience": args.patience,
        "history": history,
    }
    (WEIGHTS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())