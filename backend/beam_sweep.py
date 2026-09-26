"""
Beam search hyperparameter sweep (Tier-1 optimisation).

Sweeps beam width x GNMT length penalty on the FULL validation split
(1214 images) using the PyTorch twin of the deployed Keras decoder
(numerically identical: conversion gate max diff 1.97e-06).

Scoring semantics are copied verbatim from backend/evaluate.py +
inference/decoding.py (same BLEU smoothing, same ROUGE-L, same
normalised beam score, same 2-gram repeat penalty), so the anchor
config (width=5, alpha=0.6) must reproduce the published numbers:
    bleu1 0.5334 / bleu2 0.3203 / bleu3 0.1890 / bleu4 0.1218

Output: experiments/beam_sweep/results.json + console ranking table.

Usage:
    .venv/Scripts/python.exe beam_sweep.py [--device cuda] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"
WEIGHTS_DIR = PROJECT_DIR / "weights"
OUT_DIR = PROJECT_DIR / "experiments" / "beam_sweep"

sys.path.insert(0, str(BACKEND_DIR))
from train_torch import CaptionDecoder  # noqa: E402  (exact arch mirror)

SMOOTH = SmoothingFunction().method1


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


def rouge_l(hyp: list[str], ref: list[str]) -> float:
    n, m = len(hyp), len(ref)
    dp = np.zeros((n + 1, m + 1), dtype="int32")
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if hyp[i - 1] == ref[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    lcs = int(dp[n, m])
    if lcs == 0:
        return 0.0
    prec, rec = lcs / n, lcs / m
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


class TorchPredictor:
    """predict_fn compatible with decoding semantics; batched, on GPU/CPU."""

    def __init__(self, weights_dir: Path, device: str):
        import torch
        self.torch = torch
        meta = json.loads((weights_dir / "metadata.json").read_text())
        self.vocab_size = int(meta["vocab_size"])
        self.max_length = int(meta["max_length"])
        self.model = CaptionDecoder(self.vocab_size, self.max_length)
        blob = torch.load(weights_dir / "decoder_torch.pt", map_location="cpu",
                          weights_only=False)
        sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        self.model.load_state_dict(sd)
        self.model.eval()
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

    def __call__(self, feats: np.ndarray, seqs: np.ndarray) -> np.ndarray:
        # feats: (B, 1920) float32 ; seqs: (B, T) int64 -> (B, vocab) probs
        t = self.torch
        with t.no_grad():
            f = t.from_numpy(feats.astype("float32")).to(self.device)
            s = t.from_numpy(seqs.astype("int64")).to(self.device)
            logits = self.model(f, s)
            probs = t.softmax(logits, dim=-1)
            return probs.cpu().numpy()


def greedy_batch(predict, feats: np.ndarray, max_length: int, start_id: int,
                 end_id: int) -> tuple[list[list[int]], list[float]]:
    n = len(feats)
    seqs = np.zeros((n, max_length), dtype="int64")
    seqs[:, 0] = start_id
    dones = np.zeros(n, dtype=bool)
    out: list[list[int]] = [[] for _ in range(n)]
    conf_sum = np.zeros(n)
    conf_cnt = np.zeros(n, dtype="int64")
    for _ in range(max_length):
        probs = predict(feats, seqs)
        idx = probs.argmax(1)
        for i in range(n):
            if dones[i]:
                continue
            conf_sum[i] += float(probs[i, idx[i]])
            conf_cnt[i] += 1
            if idx[i] == end_id:
                dones[i] = True
            else:
                out[i].append(int(idx[i]))
        if bool(dones.all()):
            break
        seqs[:, 0] = start_id
        seqs[:, 1:] = 0
        for i in range(n):
            take = min(len(out[i]), max_length - 1)
            if take:
                seqs[i, 1 : 1 + take] = out[i][:take]
    return out, (conf_sum / np.maximum(conf_cnt, 1)).tolist()


def beam_batch(predict, feats: np.ndarray, max_length: int, beam_width: int,
               length_penalty: float, start_id: int, end_id: int,
               repeat_ngram: int = 2, repeat_penalty: float = 0.4
               ) -> tuple[list[list[int]], list[float]]:
    """Lockstep beam search — semantics copied from evaluate.beam_batch."""
    n = len(feats)
    alpha = length_penalty
    hyps: list[list[tuple[list[int], float, bool, list[float]]]] = [
        [([start_id], 0.0, False, [])] for _ in range(n)]

    def norm_score(c):
        tokens, logprob, _, _ = c
        denom = ((5 + max(len(tokens), 1)) / 6) ** alpha
        return logprob / denom

    for _ in range(max_length):
        if all(all(h[2] for h in hyps[i]) for i in range(n)):
            break
        rows: list[tuple[int, int]] = []
        seq_list: list[np.ndarray] = []
        feat_list: list[np.ndarray] = []
        for i in range(n):
            for h, hyp in enumerate(hyps[i]):
                if hyp[2]:
                    continue
                rows.append((i, h))
                seq = np.zeros(max_length, dtype="int64")
                seq[: len(hyp[0])] = np.asarray(hyp[0][:max_length], dtype="int64")
                seq_list.append(seq)
                feat_list.append(feats[i])
        if not rows:
            break
        probs = predict(np.stack(feat_list), np.stack(seq_list))
        probs = np.clip(probs, 1e-12, 1.0)

        # mirror evaluate.py exactly: finished hyps carry over unchanged,
        # each active parent is REPLACED by its beam_width expansions
        new_hyps: list[list[tuple[list[int], float, bool, list[float]]]] = [
            [] for _ in range(n)]
        for i in range(n):
            for hyp in hyps[i]:
                if hyp[2]:
                    new_hyps[i].append(hyp)
        for r, (i, h) in enumerate(rows):
            tokens, logprob, _, confs = hyps[i][h]
            topk = np.argsort(probs[r])[::-1][:beam_width]
            for idx in topk:
                p = float(probs[r, idx])
                lp = np.log(p)
                if repeat_ngram > 1 and len(tokens) >= repeat_ngram:
                    tail = tuple(tokens[-(repeat_ngram - 1):]) + (int(idx),)
                    existing = {tuple(tokens[k:k + repeat_ngram])
                                for k in range(len(tokens) - repeat_ngram + 1)}
                    if tail in existing:
                        lp += np.log(repeat_penalty)
                new_tokens = tokens + [int(idx)]
                finished_new = int(idx) == end_id
                new_hyps[i].append((new_tokens, logprob + lp, finished_new,
                                    confs + [p]))
        hyps = [sorted(im, key=norm_score, reverse=True)[:beam_width]
                for im in new_hyps]

    caps: list[list[int]] = []
    confs_out: list[float] = []
    for i in range(n):
        best = min(hyps[i], key=lambda c: (0 if c[2] else 1, -norm_score(c)))
        tokens, _, _, c = best
        caps.append(tokens)
        confs_out.append(float(np.mean(c)) if c else 0.0)
    return caps, confs_out


def score_captions(token_ids: list[list[int]], rev_index: dict[int, str],
                   refs: dict[str, list[list[str]]], val_images: list[str],
                   ) -> dict:
    agg = {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0,
           "rouge_l": 0.0}
    samples: list[dict] = []
    for i, img in enumerate(val_images):
        words: list[str] = []
        for idx in token_ids[i]:
            w = rev_index.get(int(idx))
            if w is None:
                break
            if w in ("startseq", "endseq"):
                continue
            words.append(w)
        gt = refs[img]
        if words:
            agg["bleu1"] += sentence_bleu(gt, words, weights=(1, 0, 0, 0), smoothing_function=SMOOTH)
            agg["bleu2"] += sentence_bleu(gt, words, weights=(0.5, 0.5, 0, 0), smoothing_function=SMOOTH)
            agg["bleu3"] += sentence_bleu(gt, words, weights=(1 / 3,) * 3, smoothing_function=SMOOTH)
            agg["bleu4"] += sentence_bleu(gt, words, weights=(0.25,) * 4, smoothing_function=SMOOTH)
            agg["rouge_l"] += rouge_l(words, gt[0])
        if len(samples) < 12:
            samples.append({"image": img, "caption": " ".join(words),
                            "ref": " ".join(gt[0])})
    n = max(len(val_images), 1)
    return {m: v / n for m, v in agg.items()} | {"n_images": len(val_images)}, samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--quick", action="store_true",
                    help="only run the 4 widths at alpha=0.6")
    args = ap.parse_args()

    import torch  # local import for device report only
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    meta = json.loads((WEIGHTS_DIR / "metadata.json").read_text())
    val_images_all = json.loads((WEIGHTS_DIR / "val_images.json").read_text())
    image_names = json.loads((WEIGHTS_DIR / "image_names.json").read_text())
    feat_index = {name: i for i, name in enumerate(image_names)}
    with open(WEIGHTS_DIR / "tokenizer.pkl", "rb") as fh:
        tokenizer = pickle.load(fh)
    rev_index = {v: k for k, v in tokenizer.word_index.items()}
    start_id = tokenizer.word_index.get("startseq", 1)
    end_id = tokenizer.word_index.get("endseq", 2)

    print(">> loading features …", flush=True)
    features = np.load(WEIGHTS_DIR / "features.npy")

    print(">> building references …", flush=True)
    data = pd.read_csv(DATASET_DIR / "captions.txt")
    data = text_preprocessing(data)
    refs: dict[str, list[list[str]]] = {}
    for img in val_images_all:
        rows = data[data["image"] == img]
        refs[img] = [
            [w for w in c.split() if w not in ("startseq", "endseq")]
            for c in rows["caption"].tolist()
        ]

    val_images = val_images_all[: args.limit] if args.limit else val_images_all
    val_ids = [feat_index[img] for img in val_images]
    feats = features[val_ids]

    print(">> building torch decoder …", flush=True)
    predictor = TorchPredictor(WEIGHTS_DIR, args.device)
    dev = str(predictor.device)
    if predictor.device.type == "cuda":
        print(f"   device: {dev} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"   device: {dev}")

    if args.quick:
        grid = [(w, 0.6) for w in (3, 5, 7, 10)]
    else:
        grid = [(w, a) for w in (3, 5, 7, 10) for a in (0.6, 0.8, 1.0, 1.2)]

    results = []
    print("\n>> greedy anchor …", flush=True)
    t0 = time.time()
    ids_g, conf_g = greedy_batch(predictor, feats, meta["max_length"], start_id, end_id)
    g_metrics, g_samples = score_captions(ids_g, rev_index, refs, val_images)
    g_metrics["mean_confidence"] = float(np.mean(conf_g))
    g_metrics["decode_time_ms_per_image"] = (time.time() - t0) * 1000 / len(val_images)
    results.append({"config": "greedy", "width": 1, "alpha": 0.0, **g_metrics})
    print(f"   greedy bleu1={g_metrics['bleu1']:.4f} bleu4={g_metrics['bleu4']:.4f}")

    for width, alpha in grid:
        t0 = time.time()
        ids_b, conf_b = beam_batch(predictor, feats, meta["max_length"],
                                   width, alpha, start_id, end_id)
        dt_ms = (time.time() - t0) * 1000 / len(val_images)
        m, samples = score_captions(ids_b, rev_index, refs, val_images)
        m["mean_confidence"] = float(np.mean(conf_b))
        m["decode_time_ms_per_image"] = dt_ms
        tag = f"w={width} a={alpha}"
        print(f"   {tag:12s} bleu1={m['bleu1']:.4f} bleu2={m['bleu2']:.4f} "
              f"bleu3={m['bleu3']:.4f} bleu4={m['bleu4']:.4f} "
              f"rougeL={m['rouge_l']:.4f} conf={m['mean_confidence']:.3f} "
              f"{dt_ms:.1f}ms/img", flush=True)
        results.append({"config": f"beam_w{width}_a{alpha}", "width": width,
                        "alpha": alpha, **m})
        if (width, alpha) == (5, 0.6):
            (OUT_DIR / "anchor_samples.json").write_text(
                json.dumps(samples, indent=2))

    results.sort(key=lambda r: (r["bleu4"], r["bleu1"]), reverse=True)
    payload = {
        "device": dev,
        "torch_version": torch.__version__,
        "vocab_size": meta["vocab_size"],
        "max_length": meta["max_length"],
        "results": results,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(payload, indent=2))

    print("\n" + "=" * 76)
    print("RANKING (by bleu4, then bleu1)")
    print("=" * 76)
    hdr = f"{'config':22s} {'bleu1':>7s} {'bleu2':>7s} {'bleu3':>7s} {'bleu4':>7s} {'rougeL':>7s} {'conf':>6s} {'ms/img':>7s}"
    print(hdr)
    for r in results:
        name = r["config"]
        print(f"{name:22s} {r['bleu1']:7.4f} {r['bleu2']:7.4f} {r['bleu3']:7.4f} "
              f"{r['bleu4']:7.4f} {r['rouge_l']:7.4f} {r['mean_confidence']:6.3f} "
              f"{r['decode_time_ms_per_image']:7.1f}")
    print(f"\n>> saved {OUT_DIR / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
