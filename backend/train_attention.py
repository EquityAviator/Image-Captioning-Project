"""
CaptionAI — Phase 5: Train Attention Decoder with PRECOMPUTED spatial features.

No raw images are loaded during training. Features come from
features_spatial.npy (N, 49, 1920). Each caption is one training sequence
(full-sequence teacher forcing), not many prefix pairs.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model.attention_decoder import AttentionDecoder      # noqa: E402
from paths import DATASET_DIR, WEIGHTS_DIR                # noqa: E402

PAD_ID = 0
START_ID = 1
END_ID = 2


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


def build_sequences(images, captions_by_image, feat_index, max_length):
    """One sample per caption (full-sequence teacher forcing)."""
    feat_ids, xs, ys = [], [], []
    for img in images:
        fi = feat_index[img]
        for seq in captions_by_image[img]:
            if len(seq) < 3:
                continue
            inp = seq[:-1]                      # [START] + caption[:-1]
            tgt = seq[1:]                       # caption[1:] + [END]
            inp = inp[:max_length] + [PAD_ID] * max(0, max_length - len(inp))
            tgt = tgt[:max_length] + [PAD_ID] * max(0, max_length - len(tgt))
            feat_ids.append(fi)
            xs.append(inp)
            ys.append(tgt)
    if not feat_ids:
        return (np.zeros((0,), "int64"), np.zeros((0, max_length), "int64"),
                np.zeros((0, max_length), "int64"))
    return (np.asarray(feat_ids, "int64"),
            np.asarray(xs, "int64"),
            np.asarray(ys, "int64"))


def train_attention(
    model, features,            # features: (N_images, 49, 1920) float16 numpy
    train_ids, train_x, train_y,
    val_ids, val_x, val_y,
    epochs=100, batch_size=64, lr=5e-4, clipnorm=5.0,
    label_smoothing=0.05, patience=15, rlr_patience=5, rlr_factor=0.5,
    min_lr=1e-6, start_epoch=1, best_val=float("inf"), device=None,
    optimizer=None, scheduler=None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=rlr_factor, patience=rlr_patience, min_lr=min_lr
        )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=label_smoothing)

    def run_epoch(ids, x, y, shuffle, training):
        n = len(ids)
        order = np.random.permutation(n) if shuffle else np.arange(n)
        total, count = 0.0, 0
        model.train(training)
        ctx = torch.no_grad() if not training else torch.enable_grad()
        with ctx:
            for start in range(0, n, batch_size):
                sel = order[start:start + batch_size]
                feats = torch.from_numpy(features[ids[sel]]).to(device).float()
                t = torch.tensor(x[sel], device=device)
                tg = torch.tensor(y[sel], device=device)
                logits, _ = model(feats, t)                 # (B, L, V)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tg.reshape(-1))
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], clipnorm
                    )
                    optimizer.step()
                total += loss.item() * len(sel)
                count += len(sel)
        return total / max(count, 1)

    history = {"loss": [], "val_loss": []}
    t_start = time.time()
    patience_counter = 0

    for epoch in range(start_epoch, epochs + 1):
        train_loss = run_epoch(train_ids, train_x, train_y, True, True)
        val_loss = run_epoch(val_ids, val_x, val_y, False, False)
        history["loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  lr {lr_now:.2e}  ({time.time() - t_start:.0f}s)", flush=True)
        scheduler.step(val_loss)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "config": {"vocab_size": model.vocab_size, "max_length": model.max_length},
            "history": history,
        }, WEIGHTS_DIR / "attention_last.pt")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": {"vocab_size": model.vocab_size, "max_length": model.max_length},
            }, WEIGHTS_DIR / "attention_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f">> early stopping at epoch {epoch} (best val {best_val:.4f})", flush=True)
                break

    print(f">> training done in {time.time() - t_start:.0f}s. best val_loss {best_val:.4f}", flush=True)
    return history, best_val


def main():
    ap = argparse.ArgumentParser(description="Train attention decoder (precomputed features)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--clipnorm", type=float, default=5.0)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--rlr-patience", type=int, default=5)
    ap.add_argument("--rlr-factor", type=float, default=0.5)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="limit images per split (smoke test)")
    ap.add_argument("--resume", type=str, default="", help="path to attention_last.pt")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> device: {device}", flush=True)

    meta = json.loads((WEIGHTS_DIR / "metadata.json").read_text())
    vocab_size = int(meta["vocab_size"])
    max_length = int(meta["max_length"])
    word_index = json.loads((WEIGHTS_DIR / "word_index.json").read_text())
    image_names = json.loads((WEIGHTS_DIR / "image_names.json").read_text())
    train_images = json.loads((WEIGHTS_DIR / "train_images.json").read_text())
    val_images = json.loads((WEIGHTS_DIR / "val_images.json").read_text())
    feat_index = {name: i for i, name in enumerate(image_names)}
    print(f">> vocab={vocab_size} max_length={max_length}", flush=True)

    if not (WEIGHTS_DIR / "features_spatial.npy").exists():
        print("!! features_spatial.npy missing. Run precompute_spatial_features.py first.", flush=True)
        return 1
    features = np.load(WEIGHTS_DIR / "features_spatial.npy", mmap_mode="r")
    print(f">> features: {features.shape}  ({features.dtype})", flush=True)

    if args.limit:
        train_images = train_images[:args.limit]
        val_images = val_images[:args.limit]
        print(f">> limiting to {args.limit} images per split", flush=True)

    data = pd.read_csv(DATASET_DIR / "captions.txt")
    data = text_preprocessing(data)
    captions_by_image = {}
    for _, row in data.iterrows():
        seq = [word_index[w] for w in row["caption"].split() if w in word_index]
        if seq:
            captions_by_image.setdefault(row["image"], []).append(seq)

    train_ids, train_x, train_y = build_sequences(
        train_images, captions_by_image, feat_index, max_length)
    val_ids, val_x, val_y = build_sequences(
        val_images, captions_by_image, feat_index, max_length)
    print(f">> images: {len(train_images)} train / {len(val_images)} val", flush=True)
    print(f">> sequences: {len(train_y):,} train / {len(val_y):,} val", flush=True)

    model = AttentionDecoder(vocab_size=vocab_size, max_length=max_length)
    print(f">> model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    start_epoch = 1
    best_val = float("inf")
    optimizer = None
    scheduler = None
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 1) + 1
        best_val = ckpt.get("val_loss", float("inf"))
        print(f">> resumed from {args.resume} (epoch {start_epoch-1})", flush=True)

    history, best_val = train_attention(
        model, features,
        train_ids, train_x, train_y,
        val_ids, val_x, val_y,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        clipnorm=args.clipnorm, label_smoothing=args.label_smoothing,
        patience=args.patience, rlr_patience=args.rlr_patience,
        rlr_factor=args.rlr_factor, min_lr=args.min_lr,
        start_epoch=start_epoch, best_val=best_val,
        device=device, optimizer=optimizer, scheduler=scheduler,
    )

    meta = json.loads((WEIGHTS_DIR / "metadata.json").read_text())
    meta["attention_training"] = {
        "epochs_trained": len(history["loss"]),
        "best_val_loss": round(best_val, 4),
        "batch_size": args.batch_size,
        "optimizer": "adamw",
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
