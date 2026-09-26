"""
Tier-1 confidence work: fit temperature scaling for the deployed decoder.

Teacher-forces the PyTorch twin over ALL validation (prefix -> next) pairs
and, in ONE streaming pass:
  * computes exact token-level NLL for a grid of temperatures T where the
    tempered distribution is  p_T(j) = p(j)^(1/T) / sum_k p(k)^(1/T)
  * collects (max-prob, top1-correct) pairs to build the reliability
    diagram + ECE at T=1 and at the fitted T*

Answers two questions with data:
  1. Is the softmax over-confident or under-confident? -> T* < / > 1
  2. What does "mean confidence" actually predict?     -> ECE + bins

Output: experiments/confidence/temperature.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"
WEIGHTS_DIR = PROJECT_DIR / "weights"
OUT_DIR = PROJECT_DIR / "experiments" / "confidence"

sys.path.insert(0, str(BACKEND_DIR))
from train_torch import CaptionDecoder, build_pairs, text_preprocessing  # noqa: E402

TEMPS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0]


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(conf)
    rows = []
    ece_val = 0.0
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        m = (conf > lo) & (conf <= hi) if b else (conf >= lo) & (conf <= hi)
        if not m.any():
            continue
        acc = float(correct[m].mean())
        avg = float(conf[m].mean())
        w = m.sum() / total
        ece_val += w * abs(acc - avg)
        rows.append({"bin": f"({lo:.1f},{hi:.1f}]", "n": int(m.sum()),
                     "avg_conf": round(avg, 4), "empirical_acc": round(acc, 4)})
    return float(ece_val), rows


def main() -> int:
    import torch

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = json.loads((WEIGHTS_DIR / "metadata.json").read_text())
    vocab_size = int(meta["vocab_size"])
    max_length = int(meta["max_length"])
    word_index = json.loads((WEIGHTS_DIR / "word_index.json").read_text())
    image_names = json.loads((WEIGHTS_DIR / "image_names.json").read_text())
    val_images = json.loads((WEIGHTS_DIR / "val_images.json").read_text())
    feat_index = {name: i for i, name in enumerate(image_names)}

    data = pd.read_csv(DATASET_DIR / "captions.txt")
    data = text_preprocessing(data)
    captions_by_image: dict[str, list[list[int]]] = {}
    for _, row in data.iterrows():
        seq = [word_index[w] for w in row["caption"].split() if w in word_index]
        if seq:
            captions_by_image.setdefault(row["image"], []).append(seq)

    val_ids, val_x, val_y = build_pairs(
        val_images, captions_by_image, feat_index, max_length)
    print(f">> val pairs: {len(val_y):,}", flush=True)

    features = np.load(WEIGHTS_DIR / "features.npy")
    model = CaptionDecoder(vocab_size, max_length)
    blob = torch.load(WEIGHTS_DIR / "decoder_torch.pt", map_location="cpu",
                      weights_only=False)
    sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    model.load_state_dict(sd)
    model.eval().to(device)

    # streaming accumulators
    nll_sum = {T: 0.0 for T in TEMPS}
    conf_parts: list[np.ndarray] = []
    corr_parts: list[np.ndarray] = []
    n_total = 0

    bs = 4096
    with torch.no_grad():
        for s in range(0, len(val_ids), bs):
            e = min(s + bs, len(val_ids))
            f = torch.from_numpy(features[val_ids[s:e]]).to(device).float()
            x = torch.from_numpy(val_x[s:e]).to(device)
            y = torch.from_numpy(val_y[s:e]).to(device)
            logits = model(f, x)
            probs = torch.softmax(logits, dim=-1).double().cpu().numpy()
            y_np = y.cpu().numpy()

            for T in TEMPS:
                q = np.power(probs, 1.0 / T)
                z = q.sum(axis=1, keepdims=True)
                picked = q[np.arange(len(y_np)), y_np] / z[:, 0]
                nll_sum[T] += float(-np.log(np.clip(picked, 1e-12, None)).sum())

            top1 = probs.argmax(axis=1)
            conf_parts.append(probs[np.arange(len(y_np)), top1])
            corr_parts.append(top1 == y_np)

            n_total += e - s
            if (s // bs) % 3 == 0:
                print(f"   {e:,}/{len(val_ids):,}", flush=True)

    conf = np.concatenate(conf_parts)
    correct = np.concatenate(corr_parts)
    nlls = {T: nll_sum[T] / n_total for T in TEMPS}
    T_best = min(nlls, key=nlls.get)

    ece_raw, bins_raw = ece(conf, correct)
    # apply fitted temperature to the collected confidences (monotone map)
    conf_T = np.power(np.clip(conf, 1e-12, 1.0), 1.0 / T_best)
    conf_T = conf_T / (conf_T + (1.0 - np.clip(conf, 1e-12, 1.0)) ** (1.0 / T_best))
    ece_fit, _ = ece(conf_T, correct)

    result = {
        "n_val_pairs": int(n_total),
        "token_nll_by_temperature": {str(t): round(v, 4) for t, v in nlls.items()},
        "T_star": T_best,
        "top1_accuracy": round(float(correct.mean()), 4),
        "mean_max_prob_displayed": round(float(conf.mean()), 4),
        "ece_at_T1": round(ece_raw, 4),
        "ece_at_Tstar": round(ece_fit, 4),
        "reliability_bins_T1": bins_raw,
        "interpretation": (
            "T* < 1 => softmax UNDER-confident (raise displayed values); "
            "T* > 1 => OVER-confident (shrink). ECE = |displayed - accuracy|."
        ),
    }
    (OUT_DIR / "temperature.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
