"""
Priority 2 — Partial DenseNet201 encoder fine-tuning (roadmap §3).

Unfreezes ONLY the final dense block (denseblock4) of the frozen spatial
encoder and fine-tunes it jointly with the attention decoder, starting from
the GRPO checkpoint (experiments/v2_attention_rl/best.pt).

Recipe (roadmap §3 + project conventions):
  * differential LR: encoder block 1e-5, decoder 1e-4 (separate param groups)
  * slightly higher weight decay on encoder params (0.02 vs 0.01)
  * BatchNorm stats stay FROZEN (encoder kept in eval mode) — standard
    small-dataset practice; only the conv weights adapt
  * CE loss, label smoothing 0.05, ignore PAD; one caption per image per epoch
  * gradient clip 5.0; NO scheduled sampling (RL gains preserved via low LR)
  * EARLY STOP / BEST-BY val CIDEr-D (fast fixed-df scorer) on a val subset —
    training loss is explicitly NOT the stopping criterion (roadmap §3)
  * SAFE checkpoints under experiments/v2_attention_ft/; the checkpoint
    stores BOTH decoder and fine-tuned encoder state dicts (serving must
    load matching encoder weights)

Usage:
    python train_attention_v2_ft.py [--epochs 5] [--batch-size 16]
        [--enc-lr 1e-5] [--dec-lr 1e-4] [--val-subset 300] [--limit 0]
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
from PIL import Image
from torchvision import transforms

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "Images"
WEIGHTS_DIR = PROJECT_DIR / "weights"
OUT_DIR = PROJECT_DIR / "experiments" / "v2_attention_ft"

sys.path.insert(0, str(BACKEND_DIR))
from train_torch import text_preprocessing  # noqa: E402
from train_attention_v2 import AttentionDecoderV2  # noqa: E402
from model.attention_decoder import build_spatial_encoder  # noqa: E402
from train_attention_v2_rl import FastCiderD, build_refs_and_scorer  # noqa: E402

MAX_LENGTH = 38
START, END, PAD = 1, 2, 0

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--enc-lr", type=float, default=1e-5)
    ap.add_argument("--dec-lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--val-subset", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> device: {device}", flush=True)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(WEIGHTS_DIR / "bpe" / "tokenizer.json"))
    encoded = json.loads((WEIGHTS_DIR / "bpe" / "encoded_captions.json").read_text())
    image_names = json.loads((WEIGHTS_DIR / "image_names.json").read_text())
    train_images = json.loads((WEIGHTS_DIR / "train_images.json").read_text())
    val_images = json.loads((WEIGHTS_DIR / "val_images.json").read_text())
    if args.limit:
        train_images, val_images = train_images[:args.limit], val_images[:args.limit]

    data = pd.read_csv(DATASET_DIR / "captions.txt")
    data = text_preprocessing(data)
    train_refs = build_refs_and_scorer(data, train_images)
    val_refs = build_refs_and_scorer(data, val_images)
    scorer = FastCiderD(list(train_refs.values()))

    # ---------- model: RL decoder + partially unfrozen encoder ----------
    rl_ckpt = PROJECT_DIR / "experiments" / "v2_attention_rl" / "best.pt"
    ckpt = torch.load(rl_ckpt, map_location="cpu", weights_only=False)
    vocab = int(ckpt["config"].get("vocab_size", ckpt["config"]["vocab"]))
    decoder = AttentionDecoderV2(vocab).to(device)
    decoder.load_state_dict(ckpt["model_state_dict"])
    print(f">> decoder from {rl_ckpt} (RL epoch {ckpt.get('epoch')})", flush=True)

    encoder = build_spatial_encoder(str(device))   # all frozen, eval mode
    blocks = list(encoder.children())
    n_blocks = len(blocks)
    final_block = blocks[-1]                       # denseblock4 (pre-norm5 head)
    for p in final_block.parameters():
        p.requires_grad = True
    n_ft = sum(p.numel() for p in final_block.parameters())
    print(f">> encoder: {n_blocks} blocks, unfroze final block "
          f"({n_ft:,} params); BN stats stay frozen (eval mode)", flush=True)

    enc_params = [p for p in final_block.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": args.enc_lr, "weight_decay": 0.02},
        {"params": decoder.parameters(), "lr": args.dec_lr, "weight_decay": 0.01},
    ])
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.05)

    # ---------- validation: greedy CIDEr on subset (the ONLY stop signal) ----
    def val_cider():
        encoder.eval(); decoder.eval()
        scores = []
        subset = val_images[:args.val_subset]
        with torch.no_grad():
            for img in subset:
                img_t = _TRANSFORM(
                    Image.open(IMAGES_DIR / img).convert("RGB")).unsqueeze(0).to(device)
                feat = encoder(img_t).flatten(2).transpose(1, 2).float()
                ids = [START]
                for _ in range(MAX_LENGTH):
                    x = torch.tensor([ids], dtype=torch.long, device=device)
                    logits = decoder(feat, x)
                    nxt = int(logits[0, -1].argmax())
                    if nxt == END:
                        break
                    ids.append(nxt)
                words = tok.decode([t for t in ids[1:] if t not in (PAD, START, END)]).strip()
                scores.append(scorer.score(words.split(), val_refs[img]))
        return float(np.mean(scores))

    base_cider = val_cider()
    print(f">> pre-FT val CIDEr-D ({args.val_subset} imgs): {base_cider:.4f}", flush=True)

    # ---------- training loop ----------
    history = {"loss": [], "val_cider": []}
    best_cider = base_cider
    patience = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        encoder.eval()          # frozen BN stats, always
        decoder.train()
        random.shuffle(train_images)
        ep_loss, n_steps = 0.0, 0
        for s in range(0, len(train_images) - args.batch_size + 1, args.batch_size):
            imgs = train_images[s:s + args.batch_size]
            batch, xs, ys = [], [], []
            for img in imgs:
                try:
                    img_t = _TRANSFORM(Image.open(IMAGES_DIR / img).convert("RGB"))
                except Exception:
                    continue
                seq = random.choice(encoded[img])[:MAX_LENGTH + 1]
                x = seq[:-1]; y = seq[1:MAX_LENGTH + 1]
                x = x + [PAD] * (MAX_LENGTH - len(x))
                y = y + [PAD] * (MAX_LENGTH - len(y))
                batch.append(img_t); xs.append(x); ys.append(y)
            if not batch:
                continue
            feats = encoder(torch.stack(batch).to(device))
            feats = feats.flatten(2).transpose(1, 2).float()
            x = torch.tensor(xs, dtype=torch.long, device=device)
            y = torch.tensor(ys, dtype=torch.long, device=device)
            logits = decoder(feats, x)
            loss = criterion(logits.reshape(-1, vocab), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]], 5.0)
            optimizer.step()
            ep_loss += loss.item(); n_steps += 1
            if n_steps % 100 == 0:
                print(f"  ep {epoch} step {n_steps}: loss {loss.item():.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

        cider = val_cider()
        history["loss"].append(round(ep_loss / max(n_steps, 1), 4))
        history["val_cider"].append(round(cider, 4))
        print(f"epoch {epoch}: train loss {ep_loss/max(n_steps,1):.4f}  "
              f"val CIDEr-D {cider:.4f} (pre-FT {base_cider:.4f}) "
              f"({time.time()-t0:.0f}s)", flush=True)

        torch.save({"epoch": epoch,
                    "decoder_state_dict": decoder.state_dict(),
                    "encoder_state_dict": encoder.state_dict(),
                    "config": ckpt["config"]}, OUT_DIR / "ckpt_last.pt")
        if cider > best_cider + 1e-4:
            best_cider = cider
            patience = 0
            torch.save({"epoch": epoch,
                        "decoder_state_dict": decoder.state_dict(),
                        "encoder_state_dict": encoder.state_dict(),
                        "val_cider": cider, "base_val_cider": base_cider,
                        "config": ckpt["config"], "tokenizer": "bpe-6k"},
                       OUT_DIR / "best.pt")
            print(f"   ✓ new best CIDEr {cider:.4f} saved", flush=True)
        else:
            patience += 1
            if patience >= args.patience:
                print(f">> early stop at epoch {epoch} (best {best_cider:.4f})",
                      flush=True)
                break

    (OUT_DIR / "train_summary.json").write_text(json.dumps({
        "base_val_cider": base_cider, "best_val_cider": best_cider,
        "config": vars(args), "history": history}, indent=2))
    print(f">> done. best val CIDEr-D {best_cider:.4f} (pre-FT {base_cider:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
