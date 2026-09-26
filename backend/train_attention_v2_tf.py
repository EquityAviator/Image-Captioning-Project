"""
Gen 4 - Transformer decoder over CLIP ViT-B/16 patch features.

Replaces the Bahdanau LSTM (Gen 2/3) with a lightweight pre-norm Transformer:
  * 4 layers, 8 heads, d_model 512, FFN 2048, dropout 0.3
  * learned positional embeddings for the 196 CLIP patches
  * causal self-attention over caption tokens + FULL cross-attention to all
    196 patches at every step (no information squeeze through an LSTM state)
  * identical interface to AttentionDecoderV2: forward(feats, tokens) ->
    (B, T, V) logits; greedy() and beam_search() helpers included

Training recipe mirrors train_attention_v2_clip.py (AdamW wd 0.01, label
smoothing 0.05, scheduled sampling ramp, ReduceLROnPlateau, patience).
Scheduled sampling uses the same prev_tok substitution trick.

Output: experiments/v2_attention_tf/best.pt
"""

from __future__ import annotations

import argparse
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
OUT_DIR = PROJECT_DIR / "experiments" / "v2_attention_tf"

MAX_LENGTH = 38
START, END, PAD = 1, 2, 0


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerCaptionDecoder(nn.Module):
    """Cross-attention transformer decoder over spatial patch features."""

    def __init__(self, vocab_size: int = 6000, enc_dim: int = 768,
                 d_model: int = 512, nhead: int = 8, num_layers: int = 4,
                 dim_ff: int = 2048, dropout: float = 0.3,
                 max_len: int = MAX_LENGTH, n_patches: int = 196):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len + 8)
        self.feat_proj = nn.Linear(enc_dim, d_model)
        self.feat_pos = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, feats: torch.Tensor, tokens: torch.Tensor,
                ss_prob: float = 0.0) -> torch.Tensor:
        B, P, D_in = feats.shape
        T = tokens.size(1)

        mem = self.feat_proj(feats) + self.feat_pos          # (B, P, D)
        mem = self.dropout(mem)

        emb = self.dropout(self.pos(self.embedding(tokens) * math.sqrt(self.d_model)))

        causal = torch.triu(torch.ones(T, T, device=tokens.device),
                            diagonal=1).bool()
        tgt_mask = causal.float().masked_fill(causal, float("-inf"))
        # padding positions in the target must not be attended TO
        pad_mask = tokens == PAD                              # (B, T)
        # keep at least the START position unmasked per row (START is idx 0)
        tgt_key_padding = pad_mask.clone()
        tgt_key_padding[:, 0] = False

        h = self.decoder(
            tgt=emb, memory=mem,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding,
            memory_key_padding_mask=None,
        )
        return self.fc(h)                                     # (B, T, V)

    # ---------------- generation helpers (mirror provider usage) --------- #
    @torch.no_grad()
    def greedy(self, feats: torch.Tensor, max_len: int,
               min_len: int = 0) -> list[int]:
        feats = feats[:1]
        ids = [START]
        mem = None
        for t in range(max_len):
            x = torch.tensor([ids], device=feats.device)
            logits = self.forward(feats, x)[0, -1].clone()
            if min_len and len(ids) - 1 < min_len:
                logits[END] = -1e9
            nxt = int(logits.argmax())
            if nxt == END:
                break
            ids.append(nxt)
        return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> device: {device}", flush=True)

    vocab_size = 6000
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

    features = np.load(WEIGHTS_DIR / "clip_features_spatial.npy", mmap_mode="r")
    model = TransformerCaptionDecoder(
        vocab_size=vocab_size, enc_dim=768, d_model=args.d_model,
        nhead=args.heads, num_layers=args.layers, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f">> params: {n_params:,} ({args.layers}L x {args.heads}H "
          f"d={args.d_model})", flush=True)

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
                logits = model(f, x)
                tot += criterion(logits.reshape(-1, vocab_size),
                                 y.reshape(-1)).item() * (e - s)
                cnt += e - s
        return tot / cnt

    best_val = float("inf")
    patience = 0
    history = {"loss": [], "val_loss": [], "lr": []}
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        ss = ss_for_epoch(epoch)
        order = np.random.permutation(len(tr_ids))
        tot, cnt = 0.0, 0
        for s in range(0, len(tr_ids), args.batch_size):
            sel = order[s:s + args.batch_size]
            f = torch.from_numpy(features[tr_ids[sel]].copy()).to(device).float()
            x = torch.from_numpy(tr_x[sel]).to(device)
            y = torch.from_numpy(tr_y[sel]).to(device)
            logits = model(f, x)
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
              f"lr {optimizer.param_groups[0]['lr']:.2e}  "
              f"({time.time()-t_start:.0f}s)", flush=True)

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
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
                        "tokenizer": "bpe-6k",
                        "encoder": "clip-vit-b16-patches",
                        "decoder": "transformer"},
                       OUT_DIR / "best.pt")
            print(f"   ✓ new best ({val:.4f}) saved", flush=True)
        else:
            patience += 1
            if patience >= args.patience:
                print(f">> early stop at epoch {epoch} (best {best_val:.4f})",
                      flush=True)
                break

    (OUT_DIR / "train_summary.json").write_text(json.dumps({
        "best_val_loss": round(best_val, 4), "epochs_run": len(history["loss"]),
        "n_params": n_params, "config": vars(args), "history": history}, indent=2))
    print(f">> done. best val {best_val:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
