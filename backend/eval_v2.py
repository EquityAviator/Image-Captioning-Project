"""
Evaluate v2 models (BPE-based) on the full validation split.

Decodes with greedy + a small beam grid, converts BPE ids back to text,
and scores word-level BLEU/ROUGE against the same references used for the
baseline (0.5358 BLEU-1 / 0.1253 BLEU-4) — apples-to-apples.

Usage:
    python eval_v2.py --model global|attention [--ckpt PATH] [--full]
"""

from __future__ import annotations

import argparse
import json
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

sys.path.insert(0, str(BACKEND_DIR))
from train_torch import text_preprocessing  # noqa: E402
from train_v2 import CaptionDecoderV2, MAX_LENGTH as G_MAX  # noqa: E402
from train_attention_v2 import AttentionDecoderV2, MAX_LENGTH as A_MAX  # noqa: E402

SMOOTH = SmoothingFunction().method1
START, END, PAD = 1, 2, 0


def rouge_l(hyp: list[str], ref: list[str]) -> float:
    n, m = len(hyp), len(ref)
    dp = np.zeros((n + 1, m + 1), dtype="int32")
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = (dp[i - 1, j - 1] + 1) if hyp[i - 1] == ref[j - 1] \
                else max(dp[i - 1, j], dp[i, j - 1])
    lcs = int(dp[n, m])
    if not lcs:
        return 0.0
    p, r = lcs / n, lcs / m
    return 2 * p * r / (p + r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["global", "attention"], required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--beam-grid", default="5:1.2,10:1.2")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-repeat-ngram", type=int, default=0,
                    help="hard-block repeated n-grams in beam (0 = soft penalty)")
    args = ap.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(WEIGHTS_DIR / "bpe" / "tokenizer.json"))

    out_dir = (PROJECT_DIR / "experiments" /
               ("v2_global" if args.model == "global" else "v2_attention"))
    ckpt_path = Path(args.ckpt) if args.ckpt else out_dir / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vocab = ckpt["config"].get("vocab_size", ckpt["config"]["vocab"])
    max_len = ckpt["config"]["max_length"]

    if args.model == "global":
        model = CaptionDecoderV2(vocab).to(device)
    else:
        model = AttentionDecoderV2(vocab).to(device)
    # Support both 'model' (focal) and 'model_state_dict' (RL/CE) keys
    state_key = "model" if "model" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[state_key])
    model.eval()
    print(f">> loaded {ckpt_path} (epoch {ckpt.get('epoch', '?')}, "
          f"val {ckpt.get('val_loss', ckpt.get('best_val', '?'))})", flush=True)

    image_names = json.loads((WEIGHTS_DIR / "image_names.json").read_text())
    val_images = json.loads((WEIGHTS_DIR / "val_images.json").read_text())
    if args.limit:
        val_images = val_images[:args.limit]
    feat_index = {n: i for i, n in enumerate(image_names)}

    data = pd.read_csv(DATASET_DIR / "captions.txt")
    data = text_preprocessing(data)
    refs: dict[str, list[list[str]]] = {}
    for img in val_images:
        rows = data[data["image"] == img]
        refs[img] = [[w for w in c.split() if w not in ("startseq", "endseq")]
                     for c in rows["caption"].tolist()]

    if args.model == "global":
        feats_all = np.load(WEIGHTS_DIR / "features.npy")
        feats = feats_all[[feat_index[i] for i in val_images]]
    else:
        feats_all = np.load(WEIGHTS_DIR / "features_spatial.npy", mmap_mode="r")

    def ids_to_text(ids: list[int]) -> str:
        return tok.decode([i for i in ids if i not in (PAD, START, END)]).strip()

    # ---------- greedy (batched lockstep, global only) ----------
    def greedy_global() -> list[str]:
        n = len(val_images)
        seqs = np.zeros((n, max_len), dtype="int64")
        seqs[:, 0] = START
        dones = np.zeros(n, dtype=bool)
        out: list[list[int]] = [[] for _ in range(n)]
        with torch.no_grad():
            for _ in range(max_len):
                logits = model(torch.from_numpy(feats).to(device).float(),
                               torch.from_numpy(seqs).to(device))
                idx = logits.argmax(-1).cpu().numpy()
                for i in range(n):
                    if dones[i]:
                        continue
                    if idx[i] == END:
                        dones[i] = True
                    else:
                        out[i].append(int(idx[i]))
                if dones.all():
                    break
                seqs[:, 0] = START
                seqs[:, 1:] = 0
                for i in range(n):
                    take = min(len(out[i]), max_len - 1)
                    if take:
                        seqs[i, 1:1 + take] = out[i][:take]
        return [ids_to_text(o) for o in out]

    # ---------- per-image decode (used for attention greedy + all beams) ----------
    def decode_one(feat_row: np.ndarray, beam: int, alpha: float) -> str:
        f = torch.from_numpy(np.asarray(feat_row, dtype=np.float32)[None]).to(device)

        def next_logits(ids: list[int]) -> torch.Tensor:
            """Return the (vocab,) next-token logits for the current prefix."""
            if args.model == "global":
                # trained on prefixes PADDED to max_len (like prod decoding)
                padded = np.zeros(max_len, dtype="int64")
                padded[:len(ids)] = ids[:max_len]
                x = torch.from_numpy(padded[None]).to(device)
                return model(f, x)[0]            # (V,)
            x = torch.tensor([ids], device=device)
            return model(f, x)[0, -1]            # (V,)

        if beam <= 1:
            ids = [START]
            with torch.no_grad():
                for _ in range(max_len):
                    nxt = int(next_logits(ids).argmax())
                    if nxt == END:
                        break
                    ids.append(nxt)
            return ids_to_text(ids[1:])
        # beam search (log-prob, ids) — GNMT normalisation, matching
        # inference/decoding.py + beam_sweep.py exactly:
        #   score(hyp) = sum(log p_tok) / ((5 + len) / 6) ** alpha
        def gnmt(lp: float, ids: list[int]) -> float:
            return lp / (((5 + max(len(ids), 1)) / 6) ** alpha)

        beams = [(0.0, [START])]
        finished: list[tuple[float, list[int]]] = []
        repeat_ngram, repeat_penalty = 2, 0.4   # EXACTLY as inference/decoding.py
        hard_block = int(getattr(args, "no_repeat_ngram", 0) or 0)

        with torch.no_grad():
            for _ in range(max_len):
                cands = []
                for lp, ids in beams:
                    logp = torch.log_softmax(next_logits(ids).float(), -1)
                    if hard_block >= 2:
                        # hard-block: ban tokens that would repeat an n-gram
                        banned = set()
                        for n_b in (repeat_ngram, hard_block):
                            if n_b < 2 or len(ids) < n_b - 1:
                                continue
                            tail = tuple(ids[-(n_b - 1):])
                            seen = {tuple(ids[k:k + n_b]) for k in range(len(ids) - n_b + 1)}
                            banned |= {t[-1] for t in (g for g in seen if g[:-1] == tail)}
                        for b_idx in banned:
                            logp[b_idx] = -1e9
                    top = torch.topk(logp, beam)
                    for v, i in zip(top.values.tolist(), top.indices.tolist()):
                        lp_new = lp + v
                        # n-gram repetition penalty (mirrors production decoding)
                        if len(ids) >= repeat_ngram:
                            tail = tuple(ids[-(repeat_ngram - 1):]) + (i,)
                            existing = {tuple(ids[k:k + repeat_ngram])
                                        for k in range(len(ids) - repeat_ngram + 1)}
                            if tail in existing:
                                lp_new += np.log(repeat_penalty)
                        cands.append((lp_new, ids + [i]))
                beams = sorted(cands, key=lambda c: gnmt(c[0], c[1]),
                               reverse=True)[:beam]
                still = []
                for lp, ids in beams:
                    if ids[-1] == END:
                        finished.append((gnmt(lp, ids), ids))
                    else:
                        still.append((lp, ids))
                beams = still
                if not beams:
                    break
        # ONLY END-terminated hypotheses may win; fall back to best active beam
        if finished:
            best = max(finished, key=lambda c: c[0])[1]
        else:
            best = max(beams, key=lambda c: gnmt(c[0], c[1]))[1]
        return ids_to_text(best[1:])

    def score_all(caps: list[str]) -> dict:
        agg = {k: 0.0 for k in ("bleu1", "bleu2", "bleu3", "bleu4", "rouge_l")}
        for img, cap in zip(val_images, caps):
            words = cap.split()
            gt = refs[img]
            if words:
                agg["bleu1"] += sentence_bleu(gt, words, weights=(1, 0, 0, 0), smoothing_function=SMOOTH)
                agg["bleu2"] += sentence_bleu(gt, words, weights=(0.5, 0.5, 0, 0), smoothing_function=SMOOTH)
                agg["bleu3"] += sentence_bleu(gt, words, weights=(1/3,)*3, smoothing_function=SMOOTH)
                agg["bleu4"] += sentence_bleu(gt, words, weights=(0.25,)*4, smoothing_function=SMOOTH)
                agg["rouge_l"] += rouge_l(words, gt[0])
        n = len(val_images)
        return {k: round(v / n, 4) for k, v in agg.items()}

    results: dict[str, dict] = {}
    samples: dict[str, list] = {}

    if args.model == "global":
        t0 = time.time()
        caps = greedy_global()
        results["greedy"] = score_all(caps)
        print(f"greedy            {results['greedy']}  ({time.time()-t0:.0f}s)", flush=True)
        samples["greedy"] = caps[:args.samples]
    else:
        t0 = time.time()
        caps = [decode_one(feats_all[feat_index[i]], 1, 0.0) for i in val_images]
        results["greedy"] = score_all(caps)
        print(f"greedy            {results['greedy']}  ({time.time()-t0:.0f}s)", flush=True)
        samples["greedy"] = caps[:args.samples]

    for spec in filter(None, args.beam_grid.split(",")):
        w, a = spec.split(":")
        w, a = int(w), float(a)
        t0 = time.time()
        caps = [decode_one(feats_all[feat_index[i]], w, a) for i in val_images]
        results[f"beam{w}_a{a}"] = score_all(caps)
        print(f"beam{w} alpha{a}      {results[f'beam{w}_a{a}']}  ({time.time()-t0:.0f}s)",
              flush=True)
        samples[f"beam{w}_a{a}"] = caps[:args.samples]

    out = {"checkpoint": str(ckpt_path), "epoch": ckpt.get("epoch"),
           "results": results,
           "samples": [{"img": val_images[i],
                        **{k: v[i] for k, v in samples.items()},
                        "ref": " | ".join(" ".join(r) for r in refs[val_images[i]][:2])}
                       for i in range(min(args.samples, len(val_images)))]}
    suffix = f"_nr{args.no_repeat_ngram}" if args.no_repeat_ngram else ""
    # PROVENANCE FIX: save next to the checkpoint that was actually
    # evaluated (out_dir tracks --model, but --ckpt can point anywhere —
    # the Aug-2025 audit found RL numbers written into v2_attention/).
    save_dir = ckpt_path.parent
    (save_dir / f"eval_results{suffix}.json").write_text(json.dumps(out, indent=2))
    print(f"\n>> saved {save_dir / f'eval_results{suffix}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
