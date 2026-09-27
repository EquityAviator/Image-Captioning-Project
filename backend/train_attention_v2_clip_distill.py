"""
Experiment A phase 2 — retrain Gen-3 CE decoder on original + BLIP
pseudo-captions (knowledge distillation). Identical to train_attention_v2_clip.py
except the caption pool includes pseudo-captions for training images.

Pseudo-captions are BPE-encoded with the same tokenizer pipeline as the
original captions: raw text -> lowercase/strip -> "startseq <tokens> endseq"
-> BPE encode. We reuse encoded_captions.json's encoding function via
train_torch helpers if present; otherwise replicate with the tokenizers lib.

Output: experiments/v2_attention_clip_distill/best.pt
"""
from __future__ import annotations

import argparse
import json
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
OUT_DIR = PROJECT_DIR / "experiments" / "v2_attention_clip_distill"

sys.path.insert(0, str(BACKEND_DIR))
from train_attention_v2_clip import AttentionDecoderV2, MAX_LENGTH  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bpe_encode_captions(raw_by_img):
    """Encode {img: [raw captions]} -> {img: [token-id seqs incl START/END]}."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(WEIGHTS_DIR / "bpe" / "tokenizer.json"))
    out = {}
    for img, caps in raw_by_img.items():
        seqs = []
        for c in caps:
            c = c.lower().strip()
            if not c:
                continue
            ids = tok.encode(c).ids
            seq = [1] + ids[: MAX_LENGTH - 2] + [2]   # START + ids + END
            if len(seq) >= 3:
                seqs.append(seq)
        if seqs:
            out[img] = seqs
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pseudo-weight", type=float, default=1.0,
                    help="fraction of pseudo-captions to include (0-1)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f">> device: {device}", flush=True)

    vocab_size = 6000
    encoded_orig = json.loads((WEIGHTS_DIR / "bpe" / "encoded_captions.json").read_text())
    pseudo_raw = json.loads((WEIGHTS_DIR / "blip_pseudo_captions.json").read_text())

    # encode pseudo-captions with the same BPE tokenizer
    keep_n = max(1, int(round(len(next(iter(pseudo_raw.values()))) * args.pseudo_weight)))
    pseudo_subset = {img: caps[:keep_n] for img, caps in pseudo_raw.items()}
    encoded_pseudo = bpe_encode_captions(pseudo_subset)

    image_names = json.loads((WEIGHTS_DIR / "image_names.json").read_text())
    train_images = json.loads((WEIGHTS_DIR / "train_images.json").read_text())
    val_images = json.loads((WEIGHTS_DIR / "val_images.json").read_text())
    feat_index = {n: i for i, n in enumerate(image_names)}
    if args.limit:
        train_images = train_images[:args.limit]
        val_images = val_images[:args.limit]

    # validation uses ONLY human references (never synthetic)
    def build_seqs(images, pools):
        fids, xs, ys = [], [], []
        for img in images:
            fi = feat_index[img]
            n_seq = 0
            for pool in pools:
                for seq in pool.get(img, []):
                    if len(seq) < 3:
                        continue
                    inp = seq[:-1][:MAX_LENGTH]
                    tgt = seq[1:][:MAX_LENGTH]
                    inp = inp + [0] * (MAX_LENGTH - len(inp))
                    tgt = tgt + [0] * (MAX_LENGTH - len(tgt))
                    fids.append(fi); xs.append(inp); ys.append(tgt)
                    n_seq += 1
            if n_seq == 0:
                raise RuntimeError(f"no sequences for {img}")
        return (np.asarray(fids, "int64"), np.asarray(xs, "int64"),
                np.asarray(ys, "int64"))

    tr_ids, tr_x, tr_y = build_seqs(train_images, [encoded_orig, encoded_pseudo])
    va_ids, va_x, va_y = build_seqs(val_images, [encoded_orig])
    print(f">> seqs: {len(tr_y):,} train (orig+pseudo) / {len(va_y):,} val "
          f"(human only)", flush=True)

    features = np.load(WEIGHTS_DIR / "clip_features_spatial.npy", mmap_mode="r")
    model = AttentionDecoderV2(vocab_size).to(device)
    print(f">> params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.05)

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
              f"({time.time()-t_start:.0f}s)", flush=True)

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
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
                        "distilled_from": "BLIP-base offline pseudo-captions"},
                       OUT_DIR / "best.pt")
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
        "n_train_seqs": int(len(tr_y)), "config": vars(args),
        "history": history}, indent=2))
    print(f">> done. best val {best_val:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
