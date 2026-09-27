"""
STEP B — Train the global-feature LSTM decoder with the anti-overfitting
recipe (Tier-2 v2).

Changes vs the epoch-26-walled baseline run (everything else identical):
  * BPE-6k subword targets          (build_bpe.py artefacts)
  * AdamW + weight_decay=0.01       (baseline: plain Adam, wd=0)
  * embedding dropout 0.3 + padding_idx
  * cosine LR schedule with 2-epoch warmup (min_lr 1e-5)
  * EMA of weights (decay 0.999) — validation & best-checkpoint use EMA
  * label smoothing 0.1 (kept — it worked)
  * SAFE checkpoints: everything under experiments/v2_global/, best saved
    only on strict improvement — can never clobber production weights/

Usage:
    python train_v2.py [--epochs 60] [--batch-size 512] [--patience 12]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
WEIGHTS_DIR = PROJECT_DIR / "weights"
OUT_DIR = PROJECT_DIR / "experiments" / "v2_global"

EMBED_DIM = 256
LSTM_UNITS = 256
DENSE_UNITS = 128
FEATURE_DIM = 1920
MAX_LENGTH = 38  # BPE max observed = 36 (+2 headroom)


class CaptionDecoderV2(nn.Module):
    """Same forward graph as the production decoder; regularised embeddings."""

    def __init__(self, vocab_size: int, emb_dropout: float = 0.3) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.img_fc = nn.Linear(FEATURE_DIM, EMBED_DIM)
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.lstm = nn.LSTM(EMBED_DIM, LSTM_UNITS, batch_first=True)
        self.dropout1 = nn.Dropout(0.5)
        self.fc128 = nn.Linear(LSTM_UNITS, DENSE_UNITS)
        self.dropout2 = nn.Dropout(0.5)
        self.out = nn.Linear(DENSE_UNITS, vocab_size)

    def forward(self, features: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        img_feats = torch.relu(self.img_fc(features))
        img_seq = img_feats.unsqueeze(1)
        emb = self.emb_dropout(self.embedding(tokens))
        merged = torch.cat([img_seq, emb], dim=1)
        lstm_out, _ = self.lstm(merged)
        h = lstm_out[:, -1, :]
        h = self.dropout1(h)
        h = h + img_feats
        h = torch.relu(self.fc128(h))
        h = self.dropout2(h)
        return self.out(h)


def build_pairs(encoded: dict, images: list[str], feat_index: dict[str, int],
                max_length: int):
    feat_ids, prefixes, targets = [], [], []
    for img in images:
        fi = feat_index[img]
        for seq in encoded[img]:
            for i in range(1, len(seq)):
                p = np.zeros(max_length, dtype="int64")
                p[:i] = seq[:i]
                feat_ids.append(fi)
                prefixes.append(p)
                targets.append(seq[i])
    return (np.asarray(feat_ids, "int64"),
            np.stack(prefixes), np.asarray(targets, "int64"))


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--emb-dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> device: {device}", flush=True)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(WEIGHTS_DIR / "bpe" / "tokenizer.json"))
    vocab_size = tok.get_vocab_size(with_added_tokens=True)
    encoded = json.loads((WEIGHTS_DIR / "bpe" / "encoded_captions.json").read_text())
    image_names = json.loads((WEIGHTS_DIR / "image_names.json").read_text())
    train_images = json.loads((WEIGHTS_DIR / "train_images.json").read_text())
    val_images = json.loads((WEIGHTS_DIR / "val_images.json").read_text())
    feat_index = {n: i for i, n in enumerate(image_names)}
    if args.limit:
        train_images, val_images = train_images[:args.limit], val_images[:args.limit]

    print(">> building (prefix -> next) pairs …", flush=True)
    tr_ids, tr_x, tr_y = build_pairs(encoded, train_images, feat_index, MAX_LENGTH)
    va_ids, va_x, va_y = build_pairs(encoded, val_images, feat_index, MAX_LENGTH)
    print(f">> pairs: {len(tr_y):,} train / {len(va_y):,} val", flush=True)

    features = np.load(WEIGHTS_DIR / "features.npy")
    model = CaptionDecoderV2(vocab_size, args.emb_dropout).to(device)
    print(f">> params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    warmup_epochs = 2
    def lr_lambda(step):
        pass  # replaced below
    # cosine with linear warmup over first 2 epochs
    steps_per_epoch = math.ceil(len(tr_y) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        p = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=0)
    ema = EMA(model, 0.999)

    best_val = float("inf")
    patience = 0
    history = {"loss": [], "val_loss": [], "val_acc": [], "lr": []}
    t_start = time.time()

    def evaluate(m: nn.Module) -> tuple[float, float]:
        m.eval()
        tot, cnt, correct = 0.0, 0, 0
        with torch.no_grad():
            for s in range(0, len(va_y), args.batch_size):
                e = min(s + args.batch_size, len(va_y))
                f = torch.from_numpy(features[va_ids[s:e]]).to(device).float()
                x = torch.from_numpy(va_x[s:e]).to(device)
                y = torch.from_numpy(va_y[s:e]).to(device)
                logits = m(f, x)
                tot += criterion(logits, y).item() * (e - s)
                correct += (logits.argmax(-1) == y).sum().item()
                cnt += e - s
        return tot / cnt, correct / cnt

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(tr_y))
        tot, cnt = 0.0, 0
        for s in range(0, len(tr_y), args.batch_size):
            sel = order[s:s + args.batch_size]
            f = torch.from_numpy(features[tr_ids[sel]]).to(device).float()
            x = torch.from_numpy(tr_x[sel]).to(device)
            y = torch.from_numpy(tr_y[sel]).to(device)
            logits = model(f, x)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            ema.update(model)
            tot += loss.item() * len(sel)
            cnt += len(sel)

        # evaluate BOTH raw and EMA weights; track EMA as candidate
        val_raw, acc_raw = evaluate(model)
        raw_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(ema.shadow)
        val_ema, acc_ema = evaluate(model)
        model.load_state_dict(raw_state)

        use_ema = val_ema <= val_raw
        val, acc = (val_ema, acc_ema) if use_ema else (val_raw, acc_raw)
        history["loss"].append(round(tot / cnt, 4))
        history["val_loss"].append(round(val, 4))
        history["val_acc"].append(round(acc, 4))
        history["lr"].append(optimizer.param_groups[0]["lr"])
        print(f"epoch {epoch:3d}  train {tot/cnt:.4f}  val {val:.4f} "
              f"(ema={val_ema:.4f} raw={val_raw:.4f})  acc {acc:.4f}  "
              f"lr {optimizer.param_groups[0]['lr']:.2e}  "
              f"({time.time()-t_start:.0f}s)", flush=True)

        ckpt_last = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.shadow,
            "config": {"vocab_size": vocab_size, "max_length": MAX_LENGTH,
                       "emb_dropout": args.emb_dropout, "tokenizer": "bpe-6k"},
            "val_loss": val, "history": history,
        }
        torch.save(ckpt_last, OUT_DIR / "ckpt_last.pt")
        if val < best_val - 1e-4:
            best_val = val
            patience = 0
            best = dict(ckpt_last)
            best["model_state_dict"] = (ema.shadow if use_ema else model.state_dict())
            best["used_ema"] = use_ema
            torch.save(best, OUT_DIR / "best.pt")
            print(f"   ✓ new best ({val:.4f}) saved", flush=True)
        else:
            patience += 1
            if patience >= args.patience:
                print(f">> early stop at epoch {epoch} (best {best_val:.4f})",
                      flush=True)
                break

    (OUT_DIR / "train_summary.json").write_text(json.dumps({
        "best_val_loss": round(best_val, 4),
        "epochs_run": len(history["loss"]),
        "config": vars(args) | {"max_length": MAX_LENGTH, "vocab": vocab_size},
        "baseline_best_val_loss": 4.7517,
        "history": history,
    }, indent=2))
    print(f">> done. best val {best_val:.4f} (baseline was 4.7517)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
