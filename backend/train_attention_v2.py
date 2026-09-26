"""
STEP C — Attention decoder v2 (Tier-2): spatial features + BPE + scheduled
sampling + anti-overfitting recipe.

Fixes every failure of the Phase-5 attempt:
  * uses the SAME pre-norm5 spatial features it trains on (provider now
    matches — bug fixed earlier today)
  * SAFE checkpoints under experiments/v2_attention/ (strict-improvement
    best.pt — can never clobber production weights/)
  * weight decay + embedding regularisation + small logit init
  * scheduled sampling (ramp 0 -> 0.25 over epochs 5..30) against exposure bias
  * encoder_proj precomputed once per batch (not per timestep)

Usage:
    python train_attention_v2.py [--epochs 60] [--batch-size 64] [--patience 12]
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
WEIGHTS_DIR = PROJECT_DIR / "weights"
OUT_DIR = PROJECT_DIR / "experiments" / "v2_attention"

MAX_LENGTH = 38


class Bahdanau(nn.Module):
    def __init__(self, enc_dim: int, dec_dim: int, att_dim: int):
        super().__init__()
        self.enc_proj = nn.Linear(enc_dim, att_dim)
        self.dec_proj = nn.Linear(dec_dim, att_dim)
        self.v = nn.Linear(att_dim, 1)

    def context(self, proj: torch.Tensor, feats: torch.Tensor,
                h: torch.Tensor) -> torch.Tensor:
        # proj: (B,P,att) precomputed | feats: (B,P,enc) | h: (B,dec)
        d = self.dec_proj(h).unsqueeze(1)
        scores = self.v(torch.tanh(proj + d)).squeeze(-1)
        alpha = F.softmax(scores, dim=1)
        return (feats * alpha.unsqueeze(-1)).sum(1)


class AttentionDecoderV2(nn.Module):
    def __init__(self, vocab_size: int, enc_dim: int = 1920, embed_dim: int = 256,
                 dec_dim: int = 512, att_dim: int = 512, emb_dropout: float = 0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.dec_dim = dec_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.attention = Bahdanau(enc_dim, dec_dim, att_dim)
        self.lstm = nn.LSTM(embed_dim + enc_dim, dec_dim, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(dec_dim, vocab_size)
        nn.init.normal_(self.fc.weight, std=0.01)   # small logit init
        nn.init.zeros_(self.fc.bias)

    def forward(self, feats: torch.Tensor, tokens: torch.Tensor,
                ss_prob: float = 0.0) -> torch.Tensor:
        """Full-sequence pass with optional scheduled sampling (training only).

        feats: (B,49,enc) | tokens: (B,T) input ids | returns (B,T,V) logits.
        """
        B, T = tokens.shape
        device = feats.device
        proj = self.attention.enc_proj(feats)  # once per batch
        h = torch.zeros(1, B, self.dec_dim, device=device)
        c = torch.zeros(1, B, self.dec_dim, device=device)
        emb_all = self.emb_dropout(self.embedding(tokens))
        logits = []
        prev_tok = tokens[:, 0]
        for t in range(T):
            ctx = self.attention.context(proj, feats, h[0])
            inp = emb_all[:, t]
            if ss_prob > 0 and t > 0 and self.training:
                mask = torch.rand(B, device=device) < ss_prob
                if mask.any():
                    inp = torch.where(mask.unsqueeze(-1),
                                      self.embedding(prev_tok), inp)
            out, (h, c) = self.lstm(
                torch.cat([inp.unsqueeze(1), ctx.unsqueeze(1)], -1), (h, c))
            step_logits = self.fc(self.dropout(out.squeeze(1)))
            logits.append(step_logits)
            prev_tok = step_logits.argmax(-1).detach()
        return torch.stack(logits, dim=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> device: {device}", flush=True)

    vocab_size = 6000  # fixed by build_bpe.py
    encoded = json.loads((WEIGHTS_DIR / "bpe" / "encoded_captions.json").read_text())
    image_names = json.loads((WEIGHTS_DIR / "image_names.json").read_text())
    train_images = json.loads((WEIGHTS_DIR / "train_images.json").read_text())
    val_images = json.loads((WEIGHTS_DIR / "val_images.json").read_text())
    feat_index = {n: i for i, n in enumerate(image_names)}
    if args.limit:
        train_images, val_images = train_images[:args.limit], val_images[:args.limit]

    def build_seqs(images):
        fids, xs, ys = [], [], []
        for img in images:
            fi = feat_index[img]
            for seq in encoded[img]:
                if len(seq) < 3:
                    continue
                # seq = [START, w1..wn, END] -> input [START, w1..wn], target [w1..wn, END]
                inp = seq[:-1][:MAX_LENGTH]
                tgt = seq[1:][:MAX_LENGTH]
                inp = inp + [0] * (MAX_LENGTH - len(inp))
                tgt = tgt + [0] * (MAX_LENGTH - len(tgt))
                fids.append(fi)
                xs.append(inp)
                ys.append(tgt)
        return (np.asarray(fids, "int64"), np.asarray(xs, "int64"),
                np.asarray(ys, "int64"))

    tr_ids, tr_x, tr_y = build_seqs(train_images)
    va_ids, va_x, va_y = build_seqs(val_images)
    print(f">> seqs: {len(tr_y):,} train / {len(va_y):,} val", flush=True)

    features = np.load(WEIGHTS_DIR / "features_spatial.npy", mmap_mode="r")
    model = AttentionDecoderV2(vocab_size).to(device)
    print(f">> params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.05)

    def ss_for_epoch(ep: int) -> float:
        return float(np.clip((ep - 5) / 25.0, 0.0, 1.0) * 0.25)

    def evaluate():
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(va_ids), args.batch_size):
                e = min(s + args.batch_size, len(va_ids))
                f = torch.from_numpy(features[va_ids[s:e]].copy()).to(device).float()
                x = torch.from_numpy(va_x[s:e]).to(device)
                y = torch.from_numpy(va_y[s:e]).to(device)
                logits = model(f, x, ss_prob=0.0)
                tot += criterion(logits.reshape(-1, vocab_size),
                                 y.reshape(-1)).item() * (e - s)
                cnt += e - s
        return tot / cnt

    best_val = float("inf")
    patience = 0
    history = {"loss": [], "val_loss": [], "lr": []}
    start_epoch, resume_opt = 1, None
    t_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        ss = ss_for_epoch(epoch)
        order = np.random.permutation(len(tr_ids))
        tot, cnt = 0.0, 0
        for s in range(0, len(tr_ids), args.batch_size):
            sel = order[s:s + args.batch_size]
            f = torch.from_numpy(features[tr_ids[sel]].copy()).to(device).float()
            x = torch.from_numpy(tr_x[sel]).to(device)
            y = torch.from_numpy(tr_y[sel]).to(device)
            logits = model(f, x, ss_prob=ss)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tot += loss.item() * len(sel)
            cnt += len(sel)
        val = evaluate()
        scheduler.step(val)
        history["loss"].append(round(tot / cnt, 4))
        history["val_loss"].append(round(val, 4))
        history["lr"].append(optimizer.param_groups[0]["lr"])
        print(f"epoch {epoch:3d}  train {tot/cnt:.4f}  val {val:.4f}  "
              f"ss {ss:.2f}  lr {optimizer.param_groups[0]['lr']:.2e}  "
              f"({time.time()-t_start:.0f}s)", flush=True)

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val": best_val, "patience": patience,
                    "history": history,
                    "config": {"vocab": vocab_size, "max_length": MAX_LENGTH}},
                   OUT_DIR / "ckpt_last.pt")
        if val < best_val - 1e-4:
            best_val = val
            patience = 0
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_loss": val,
                        "config": {"vocab": vocab_size, "max_length": MAX_LENGTH},
                        "tokenizer": "bpe-6k"}, OUT_DIR / "best.pt")
            print(f"   ✓ new best ({val:.4f}) saved", flush=True)
        else:
            patience += 1
            if patience >= args.patience:
                print(f">> early stop at epoch {epoch} (best {best_val:.4f})",
                      flush=True)
                break

    (OUT_DIR / "train_summary.json").write_text(json.dumps({
        "best_val_loss": round(best_val, 4), "epochs_run": len(history["loss"]),
        "config": vars(args), "history": history}, indent=2))
    print(f">> done. best val {best_val:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
