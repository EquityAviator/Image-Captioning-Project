"""
CaptionAI — Stage 5: evaluation on the held-out test split.

Loads weights/model.h5 + tokenizer.pkl, decodes captions for the val split
with greedy and beam search, and reports BLEU-1..4 and ROUGE-L against the
5 reference captions per image.

Decoding is BATCHED across images (lockstep), so the whole 1214-image test
split needs only ~72 predict calls instead of ~87k. Scoring semantics match
inference/decoding.py exactly (GNMT length normalisation, 2-gram repeat
penalty, confidence = mean token probability along the chosen path).

Usage:
    python evaluate.py [--limit N] [--beam 5]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"
WEIGHTS_DIR = PROJECT_DIR / "weights"

sys.path.insert(0, str(BACKEND_DIR))
from model.architecture import build_decoder  # noqa: E402

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


def rouge_l(hyp: list[str], ref: list[str]) -> float:
    """ROUGE-L F-measure over the longest common subsequence of words."""
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
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def greedy_batch(model, feats: np.ndarray, max_length: int) -> tuple[list[str], list[float]]:
    """Lockstep greedy decode over all images at once."""
    n = len(feats)
    seqs = np.zeros((n, max_length), dtype="int64")
    seqs[:, 0] = START_ID
    dones = np.zeros(n, dtype=bool)
    out: list[list[int]] = [[] for _ in range(n)]
    conf_sum = np.zeros(n, dtype="float64")
    conf_cnt = np.zeros(n, dtype="int64")
    for t in range(1, max_length):
        probs = model.predict([feats, seqs], verbose=0, batch_size=n)
        idx = probs.argmax(1)
        ended = idx == END_ID
        for i in range(n):
            if dones[i]:
                continue
            conf_sum[i] += float(probs[i, idx[i]])
            conf_cnt[i] += 1
            if ended[i]:
                dones[i] = True
            else:
                out[i].append(int(idx[i]))
        if bool(dones.all()):
            break
        seqs[:, t] = idx
    caps = [" ".join(_ids_to_words(tokenizer, tok)) for tok in out]
    confs = [(conf_sum[i] / conf_cnt[i] if conf_cnt[i] else 0.0) for i in range(n)]
    return caps, confs


def beam_batch(model, feats: np.ndarray, max_length: int, beam_width: int,
               length_penalty: float = 0.6, repeat_ngram: int = 2,
               repeat_penalty: float = 0.4) -> tuple[list[str], list[float]]:
    """Lockstep beam search over all images; per-step predictions batched."""
    n = len(feats)
    alpha = length_penalty
    # hyps[i] = [(tokens, logprob, finished, confs)]
    hyps: list[list[tuple[list[int], float, bool, list[float]]]] = [
        [([START_ID], 0.0, False, [])] for _ in range(n)]

    for _ in range(max_length):
        active = any(not all(h[2] for h in hyps[i]) for i in range(n))
        if not active:
            break
        rows: list[tuple[int, int]] = []
        seq_list: list[np.ndarray] = []
        feat_list: list[np.ndarray] = []
        for i in range(n):
            for h, hyp in enumerate(hyps[i]):
                rows.append((i, h))
                seq = np.zeros(max_length, dtype="int64")
                seq[: len(hyp[0])] = np.asarray(hyp[0][:max_length], dtype="int64")
                seq_list.append(seq)
                feat_list.append(feats[i])
        probs = model.predict([np.stack(feat_list), np.stack(seq_list)],
                              verbose=0, batch_size=len(rows))
        probs = np.clip(probs, 1e-12, 1.0)

        new_hyps: list[list[tuple[list[int], float, bool, list[float]]]] = [[] for _ in range(n)]
        for r, (i, h) in enumerate(rows):
            tokens, logprob, finished, confs = hyps[i][h]
            if finished:
                new_hyps[i].append((tokens, logprob, True, confs))
                continue
            topk = np.argsort(probs[r])[::-1][:beam_width]
            for idx in topk:
                p = float(probs[r, idx])
                lp = np.log(p)
                if repeat_ngram > 1 and len(tokens) >= repeat_ngram:
                    tail = tuple(tokens[-(repeat_ngram - 1):]) + (int(idx),)
                    if tail in _ngrams(tokens, repeat_ngram):
                        lp += np.log(repeat_penalty)
                new_tokens = tokens + [int(idx)]
                finished_new = int(idx) == END_ID
                new_hyps[i].append((new_tokens, logprob + lp, finished_new,
                                    confs + [p]))

        def norm_score(c):
            tokens, logprob, _, _ = c
            denom = ((5 + max(len(tokens), 1)) / 6) ** alpha
            return logprob / denom

        hyps = [sorted(im, key=norm_score, reverse=True)[:beam_width] for im in new_hyps]

    caps: list[str] = []
    confs: list[float] = []
    for i in range(n):
        best = min(hyps[i], key=lambda c: (0 if c[2] else 1, -norm_score(c)))
        tokens, _, _, c = best
        caps.append(" ".join(_ids_to_words(tokenizer, tokens)))
        confs.append(float(np.mean(c)) if c else 0.0)
    return caps, confs


def _ids_to_words(tokenizer, tokens: list[int]) -> list[str]:
    words: list[str] = []
    for idx in tokens:
        word = None
        for w, j in tokenizer.word_index.items():
            if j == idx:
                word = w
                break
        if word is None:
            break
        if word in ("startseq", "endseq"):
            continue
        words.append(word)
    return words


def _ngrams(tokens: list[int], n: int) -> set[tuple[int, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--beam", type=int, default=5)
    args = ap.parse_args()

    meta = json.loads((WEIGHTS_DIR / "metadata.json").read_text())
    vocab_size = int(meta["vocab_size"])
    max_length = int(meta["max_length"])
    val_images = json.loads((WEIGHTS_DIR / "val_images.json").read_text())
    features = np.load(WEIGHTS_DIR / "features.npy")
    image_names = json.loads((WEIGHTS_DIR / "image_names.json").read_text())
    feat_index = {name: i for i, name in enumerate(image_names)}

    global tokenizer
    with open(WEIGHTS_DIR / "tokenizer.pkl", "rb") as fh:
        tokenizer = pickle.load(fh)

    print(">> building Keras decoder …", flush=True)
    model = build_decoder(vocab_size, max_length)
    model.load_weights(str(WEIGHTS_DIR / "model.h5"))

    print(">> building references …", flush=True)
    data = pd.read_csv(DATASET_DIR / "captions.txt")
    data = text_preprocessing(data)
    refs: dict[str, list[list[str]]] = {}
    for img in val_images:
        rows = data[data["image"] == img]
        refs[img] = [
            [w for w in c.split() if w not in ("startseq", "endseq")]
            for c in rows["caption"].tolist()
        ]

    if args.limit:
        val_images = val_images[: args.limit]
    val_ids = [feat_index[img] for img in val_images]
    feats = features[val_ids]

    print(f">> decoding {len(val_images)} images (greedy + beam{args.beam}) …", flush=True)
    cap_g, conf_g = greedy_batch(model, feats, max_length)
    cap_b, conf_b = beam_batch(model, feats, max_length, args.beam)

    smooth = SmoothingFunction().method1
    res = {"greedy": {m: 0.0 for m in ("bleu1", "bleu2", "bleu3", "bleu4", "rouge_l")},
           "beam": {m: 0.0 for m in ("bleu1", "bleu2", "bleu3", "bleu4", "rouge_l")}}
    for i, img in enumerate(val_images):
        gt = refs[img]
        for key, cap in (("greedy", cap_g[i]), ("beam", cap_b[i])):
            words = cap.split()
            if not words:
                continue
            res[key]["bleu1"] += sentence_bleu(gt, words, weights=(1, 0, 0, 0), smoothing_function=smooth)
            res[key]["bleu2"] += sentence_bleu(gt, words, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
            res[key]["bleu3"] += sentence_bleu(gt, words, weights=(1 / 3,) * 3, smoothing_function=smooth)
            res[key]["bleu4"] += sentence_bleu(gt, words, weights=(0.25,) * 4, smoothing_function=smooth)
            res[key]["rouge_l"] += rouge_l(words, gt[0])

    n = max(len(val_images), 1)
    print("\n" + "=" * 60)
    print(f"EVALUATION on {len(val_images)} images (test split)")
    print("=" * 60)
    for key in ("greedy", "beam"):
        print(f"\n[{key}]")
        for metric in ("bleu1", "bleu2", "bleu3", "bleu4", "rouge_l"):
            print(f"  {metric:8s} = {res[key][metric] / n:.4f}")

    print("\nsample outputs:")
    for i in range(min(10, n)):
        print(f"  img {val_images[i]}")
        print(f"    greedy : {cap_g[i]}  (conf {conf_g[i]:.3f})")
        print(f"    beam   : {cap_b[i]}  (conf {conf_b[i]:.3f})")
        print(f"    ref    : {' '.join(refs[val_images[i]][0])}")

    results = {k: {m: round(v / n, 4) for m, v in r.items()} for k, r in res.items()}
    meta["evaluation"] = {
        "n_images": n,
        "greedy": results["greedy"],
        "beam": results["beam"],
    }
    (WEIGHTS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("\n>> results saved to weights/metadata.json")
    return 0


tokenizer = None  # set in main, used by decode helpers


if __name__ == "__main__":
    sys.exit(main())