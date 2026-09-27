"""
CaptionAI — Phase 5: evaluate the trained attention decoder.

Loads attention_best.pt + precomputed spatial features, generates one
caption per validation image (greedy), and reports BLEU-1..4 + 2-gram
repetition, comparing against the references from captions.txt.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model.attention_decoder import AttentionDecoder                    # noqa: E402
from paths import DATASET_DIR, WEIGHTS_DIR                                # noqa: E402

START_ID = 1
END_ID = 2


def load_references():
    data = pd.read_csv(DATASET_DIR / "captions.txt")
    refs = {}
    for _, row in data.iterrows():
        cap = str(row["caption"]).lower()
        cap = re.sub(r"[^a-z ]", " ", cap)
        cap = " ".join(w for w in cap.split() if len(w) > 1)
        refs.setdefault(row["image"], []).append(cap.split())
    return refs


def generate_captions(model, features, val_images, feat_index, max_length, device, beam=0):
    id_to_word = json.loads((WEIGHTS_DIR / "word_index.json").read_text())
    id_to_word = {int(v): k for k, v in id_to_word.items()}
    model.eval()
    preds = {}
    for img in val_images:
        fi = feat_index[img]
        feat = torch.from_numpy(features[fi].copy()).unsqueeze(0).to(device).float()
        if beam and beam > 1:
            seq = model.beam_search(feat, START_ID, END_ID, max_length, device, beam_width=beam)
        else:
            seq = model.generate(feat, START_ID, END_ID, max_length, device)
        # strip START/END/PAD
        toks = [t for t in seq if t not in (START_ID, END_ID, 0)]
        words = [id_to_word.get(t, "") for t in toks]
        words = [w for w in words if w]
        preds[img] = words
    return preds


def bleu_n(pred, refs, n):
    def ngrams(toks, n):
        return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    bp = 1.0
    tot = 0.0
    cnt = 0
    for p, rs in zip(pred, refs):
        if not p:
            continue
        pgrams = Counter(ngrams(p, n))
        best = 0.0
        for r in rs:
            rgrams = Counter(ngrams(r, n))
            overlap = sum((pgrams & rgrams).values())
            denom = max(len(pgrams), 1)
            score = overlap / denom
            best = max(best, score)
        tot += best
        cnt += 1
    return tot / max(cnt, 1)


def repetition_rate(preds):
    twice = 0
    total = 0
    for toks in preds.values():
        if len(toks) >= 2:
            bigrams = [tuple(toks[i:i + 2]) for i in range(len(toks) - 1)]
            if bigrams and len(set(bigrams)) < len(bigrams):
                twice += 1
        if toks:
            total += 1
    return twice / max(total, 1)


def main():
    ap = argparse.ArgumentParser(description="Evaluate attention decoder")
    ap.add_argument("--checkpoint", default=str(WEIGHTS_DIR / "attention_best.pt"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--beam", type=int, default=0, help="beam width (>1 = beam search)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.loads((WEIGHTS_DIR / "metadata.json").read_text())
    vocab_size = int(meta["vocab_size"])
    max_length = int(meta["max_length"])
    image_names = json.loads((WEIGHTS_DIR / "image_names.json").read_text())
    val_images = json.loads((WEIGHTS_DIR / "val_images.json").read_text())
    feat_index = {name: i for i, name in enumerate(image_names)}
    features = np.load(WEIGHTS_DIR / "features_spatial.npy", mmap_mode="r")

    if args.limit:
        val_images = val_images[:args.limit]

    model = AttentionDecoder(vocab_size=vocab_size, max_length=max_length).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f">> loaded {args.checkpoint} (epoch {ckpt.get('epoch', '?')})", flush=True)

    preds = generate_captions(model, features, val_images, feat_index, max_length, device, beam=args.beam)
    refs_map = load_references()
    refs = [refs_map[img] for img in val_images if img in refs_map]

    pred_list = [preds[img] for img in val_images if img in refs_map]
    scores = {f"bleu_{i}": round(bleu_n(pred_list, refs, i), 4) for i in (1, 2, 3, 4)}
    scores["repetition_2gram"] = round(repetition_rate(preds), 4)
    scores["n_images"] = len(pred_list)
    print(json.dumps(scores, indent=2), flush=True)

    sample_imgs = [v for v in val_images if v in refs_map][:5]
    for img in sample_imgs:
        pred = " ".join(preds[img])
        ref = " | ".join(" ".join(r) for r in refs_map[img][:2])
        print(f"[sample] {img}\n   pred: {pred}\n   refs: {ref}", flush=True)

    meta = json.loads((WEIGHTS_DIR / "metadata.json").read_text())
    meta.setdefault("attention_evaluation", {})
    meta["attention_evaluation"].update(scores)
    (WEIGHTS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    sys.exit(main())
