"""
Priority 0 — Close the measurement gaps for attention-v2 (roadmap §1).
v2: hang-proof edition.

Changes vs v1 (both v1 runs deadlocked writing into the OneDrive folder
right after METEOR's failed Java subprocess):
  * METEOR REMOVED — its Java subprocess fails on Windows ([Errno 22]) and
    the broken handle wedged the process. BLEU/ROUGE-L/CIDEr-D + CHAIR-lite
    cover the roadmap requirement.
  * ALL outputs go to a LOCAL directory (%LOCALAPPDATA%/captionai_p0) first;
    copying into the repo is best-effort afterwards (never fatal).
  * Incremental prediction checkpoints every 200 images during decoding.
  * faulthandler dumps the exact stack trace if anything stalls > 300 s.

Outputs (local dir, copied best-effort to experiments/v2_attention/):
    predictions.json         beam-5 captions for all val images
    priority0_report.json    metrics + CHAIR + calibration + END diagnostics
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

# NOTE: no faulthandler watchdog — its 5-minute stderr dumps fired while the
# main thread was inside torch's native LSTM call and destabilised/wedged the
# process on Windows (access violations + deadlocks). Diagnosis complete; the
# stable configuration is plain CUDA execution as in eval_v2.py.

import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"
WEIGHTS_DIR = PROJECT_DIR / "weights"
OUT_DIR = PROJECT_DIR / "experiments" / "v2_attention"
SAFE_DIR = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "captionai_p0"

sys.path.insert(0, str(BACKEND_DIR))
from train_torch import text_preprocessing  # noqa: E402
from train_attention_v2 import AttentionDecoderV2  # noqa: E402

SMOOTH = SmoothingFunction().method1
START, END, PAD = 1, 2, 0


def log(msg: str):
    print(msg, flush=True)


def safe_write(name: str, text: str):
    """Write locally first (never fails), then best-effort copy to repo."""
    local = SAFE_DIR / name
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(text, encoding="utf-8")
    log(f">> wrote {local}")
    try:
        repo = OUT_DIR / name
        repo.parent.mkdir(parents=True, exist_ok=True)
        repo.write_text(text, encoding="utf-8")
        log(f">> copied to {repo}")
    except Exception as exc:  # noqa: BLE001
        log(f"!! repo copy skipped for {name}: {exc}")


def rouge_l(hyp, ref):
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


def ece(conf, correct, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    total, ece_val, rows = len(conf), 0.0, []
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        m = (conf > lo) & (conf <= hi) if b else (conf >= lo) & (conf <= hi)
        if not m.any():
            continue
        acc, avg = float(correct[m].mean()), float(conf[m].mean())
        ece_val += (m.sum() / total) * abs(acc - avg)
        rows.append({"bin": f"({lo:.1f},{hi:.1f}]", "n": int(m.sum()),
                     "avg_conf": round(avg, 4), "acc": round(acc, 4)})
    return float(ece_val), rows


def main() -> int:
    # CUDA is the proven-stable configuration here: eval_v2.py ran hours of
    # identical beam decoding on this GPU without a crash, and the original
    # P0 v1 run (no faulthandler watchdog) completed its entire 1,214-image
    # decode phase. CPU was only tried during the watchdog misdiagnosis.
    device = torch.device(os.environ.get("P0_DEVICE", "cuda"
                                         if torch.cuda.is_available() else "cpu"))
    SAFE_DIR.mkdir(parents=True, exist_ok=True)
    log(f">> device: {device} | safe out dir: {SAFE_DIR}")

    # ---------- data ----------
    image_names = json.loads((WEIGHTS_DIR / "image_names.json").read_text())
    val_images = json.loads((WEIGHTS_DIR / "val_images.json").read_text())
    feat_index = {n: i for i, n in enumerate(image_names)}
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(WEIGHTS_DIR / "bpe" / "tokenizer.json"))

    log(">> building references …")
    data = pd.read_csv(DATASET_DIR / "captions.txt")
    data = text_preprocessing(data)
    refs: dict[str, list[list[str]]] = {}
    for img in val_images:
        rows = data[data["image"] == img]
        refs[img] = [[w for w in c.split() if w not in ("startseq", "endseq")]
                     for c in rows["caption"].tolist()]

    ckpt_path = OUT_DIR / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vocab = int(ckpt["config"].get("vocab_size", ckpt["config"]["vocab"]))
    max_len = int(ckpt["config"]["max_length"])
    model = AttentionDecoderV2(vocab).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log(f">> loaded {ckpt_path} (epoch {ckpt.get('epoch')})")

    feats_all = np.load(WEIGHTS_DIR / "features_spatial.npy", mmap_mode="r")

    def next_logp(feat, ids):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(feat, x)
        return torch.log_softmax(logits[0, -1].float(), -1)

    def decode_beam(feat_row, bw=5, alpha_=1.2):
        f = torch.from_numpy(np.asarray(feat_row, dtype=np.float32)[None]).to(device)
        gnmt = lambda lp, ids: lp / (((5 + max(len(ids), 1)) / 6) ** alpha_)
        beams = [(0.0, [START])]
        finished = []
        with torch.no_grad():
            for _ in range(max_len):
                cands = []
                for lp, ids in beams:
                    logp = next_logp(f, ids)
                    top = torch.topk(logp, bw)
                    for v, i in zip(top.values.tolist(), top.indices.tolist()):
                        lp_new = lp + v
                        if len(ids) >= 2:
                            tail = (ids[-1], i)
                            if tail in {(ids[k], ids[k + 1]) for k in range(len(ids) - 1)}:
                                lp_new += math.log(0.4)
                        cands.append((lp_new, ids + [i]))
                beams = sorted(cands, key=lambda c: gnmt(c[0], c[1]), reverse=True)[:bw]
                still = []
                for lp, ids in beams:
                    if ids[-1] == END:
                        finished.append((gnmt(lp, ids), ids))
                    else:
                        still.append((lp, ids))
                beams = still
                if not beams:
                    break
        if finished:
            best = max(finished, key=lambda c: c[0])[1]
        else:
            best = max(beams, key=lambda c: gnmt(c[0], c[1]))[1]
        return tok.decode([t for t in best[1:] if t not in (PAD, START, END)]).strip()

    # ---------- A: decode (with incremental checkpoints + resume) ----------
    log(">> decoding val split (beam-5, alpha 1.2, CPU) …")
    partial = SAFE_DIR / "predictions_partial.json"
    preds_beam: dict[str, str] = {}
    done_set: set[str] = set()
    if partial.exists():
        try:
            preds_beam = json.loads(partial.read_text(encoding="utf-8"))
            done_set = set(preds_beam)
            log(f">> resuming: {len(done_set)} already decoded")
        except Exception:
            pass
    todo = [i for i in val_images if i not in done_set]
    for k, img in enumerate(todo):
        preds_beam[img] = decode_beam(feats_all[feat_index[img]])
        if (k + 1) % 100 == 0:
            log(f"   {len(preds_beam)}/{len(val_images)}")
            partial.write_text(json.dumps(preds_beam, indent=1), encoding="utf-8")
    partial.write_text(json.dumps(preds_beam, indent=1), encoding="utf-8")
    log(">> decoding complete — saving predictions FIRST")
    safe_write("predictions.json", json.dumps(preds_beam, indent=1))
    try:
        partial.unlink()
    except Exception:
        pass

    # ---------- classic metrics + CHAIR-lite ----------
    log(">> computing BLEU/ROUGE + CHAIR-lite …")
    agg = {k: 0.0 for k in ("bleu1", "bleu2", "bleu3", "bleu4", "rouge_l")}
    hyp_for_cider: dict[str, list[str]] = {}
    refs_for_cider: dict[str, list[str]] = {}
    chair_imgs = halluc_total = noun_total = 0

    try:
        import nltk
        for pkg in ("averaged_perceptron_tagger_eng", "punkt", "punkt_tab"):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        from nltk import pos_tag
        nltk_ok = True
    except Exception:
        nltk_ok = False

    def nouns_of(words):
        try:
            return {w.lower() for w, t in pos_tag(list(words)) if t.startswith("NN")}
        except Exception:
            return set()

    for img in val_images:
        words = preds_beam[img].split()
        gt = refs[img]
        hyp_for_cider[img] = [preds_beam[img]]
        refs_for_cider[img] = [" ".join(r) for r in gt]
        if words:
            agg["bleu1"] += sentence_bleu(gt, words, weights=(1, 0, 0, 0), smoothing_function=SMOOTH)
            agg["bleu2"] += sentence_bleu(gt, words, weights=(0.5, 0.5, 0, 0), smoothing_function=SMOOTH)
            agg["bleu3"] += sentence_bleu(gt, words, weights=(1 / 3,) * 3, smoothing_function=SMOOTH)
            agg["bleu4"] += sentence_bleu(gt, words, weights=(0.25,) * 4, smoothing_function=SMOOTH)
            agg["rouge_l"] += rouge_l(words, gt[0])
        if nltk_ok:
            gen_nouns = nouns_of(words)
            if gen_nouns:
                ref_nouns = set()
                for r in gt:
                    ref_nouns |= nouns_of(r)
                hallu = (gen_nouns - ref_nouns) if ref_nouns else set()
                noun_total += len(gen_nouns)
                halluc_total += len(hallu)
                if hallu:
                    chair_imgs += 1
    n = len(val_images)
    metrics = {k: round(v / n, 4) for k, v in agg.items()}
    chair = {
        "chair_image_rate": round(chair_imgs / n, 4),
        "hallucinated_noun_rate": round(halluc_total / max(noun_total, 1), 4),
        "note": "CHAIR-lite: generated nouns missing from the image's reference-caption noun set",
    }
    log(f">> classic metrics: {metrics}")
    log(f">> CHAIR-lite: {chair}")

    # ---------- CIDEr-D ----------
    log(">> computing CIDEr-D …")
    try:
        from pycocoevalcap.cider.cider import Cider
        cider_score, _ = Cider().compute_score(refs_for_cider, hyp_for_cider)
        metrics["cider_d"] = round(float(cider_score), 4)
        log(f">> CIDEr-D = {metrics['cider_d']}")
    except Exception as exc:  # noqa: BLE001
        metrics["cider_d"] = None
        log(f"!! CIDEr failed: {exc}")
    # METEOR deliberately skipped: Java subprocess breaks on Windows.

    # ---------- C/D: teacher-forced accuracy, ECE, T*, END diagnostic ----------
    log(">> teacher-forcing attention-v2 (accuracy/ECE/T*) …")
    encoded = json.loads((WEIGHTS_DIR / "bpe" / "encoded_captions.json").read_text())
    confs, corrects, p_end_at_end, p_end_mid = [], [], [], []
    bs = 48
    TEMPS = [0.6, 0.8, 0.9, 1.0, 1.1, 1.3, 1.6]
    nll_sum = {t: 0.0 for t in TEMPS}
    n_tok = 0
    with torch.no_grad():
        for s in range(0, len(val_images), bs):
            imgs = val_images[s:s + bs]
            f = torch.from_numpy(feats_all[[feat_index[i] for i in imgs]].copy()).to(device).float()
            xs, lens = [], []
            for img in imgs:
                seq = encoded[img][0][:max_len]
                xs.append(seq + [PAD] * (max_len - len(seq)))
                lens.append(min(len(encoded[img][0]), max_len))
            x = torch.tensor(xs, dtype=torch.long, device=device)
            logits = model(f, x)
            probs = torch.softmax(logits.float(), -1).cpu().numpy()
            for bi, img in enumerate(imgs):
                L = lens[bi]
                tgt = encoded[img][0][1:L]
                for t_i, t_true in enumerate(tgt):
                    p = probs[bi, t_i]
                    confs.append(float(p.max()))
                    corrects.append(int(p.argmax() == t_true))
                    for T in TEMPS:
                        q = np.power(np.clip(p, 1e-12, 1.0), 1.0 / T)
                        picked = q[t_true] / q.sum()
                        nll_sum[T] += -math.log(max(picked, 1e-12))
                    if t_true == END:
                        p_end_at_end.append(float(p[END]))
                    elif t_i >= 2:
                        p_end_mid.append(float(p[END]))
                    n_tok += 1
            if (s // bs) % 6 == 0:
                log(f"   tf {s + len(imgs):,}/{len(val_images)}")
    conf = np.array(confs)
    corr = np.array(corrects)
    acc = float(corr.mean())
    mean_conf = float(conf.mean())
    ece_val, bins_rows = ece(conf, corr)
    nlls = {T: nll_sum[T] / n_tok for T in TEMPS}
    t_star = min(nlls, key=nlls.get)

    report = {
        "checkpoint": str(ckpt_path),
        "decode": {"beam": 5, "alpha": 1.2},
        "metrics": metrics,
        "chair_lite": chair,
        "attention_v2_calibration": {
            "top1_accuracy": round(acc, 4),
            "mean_max_prob": round(mean_conf, 4),
            "ece_at_T1": round(ece_val, 4),
            "temperature_grid_nll": {str(t): round(v, 4) for t, v in nlls.items()},
            "T_star": t_star,
            "reliability_bins": bins_rows,
            "n_tokens": int(n_tok),
        },
        "end_token_diagnostic": {
            "mean_p_end_at_true_end": round(float(np.mean(p_end_at_end)), 4),
            "mean_p_end_mid_caption": round(float(np.mean(p_end_mid)), 4),
            "n_end_positions": len(p_end_at_end),
        },
        "meteor": "skipped on Windows (pycocoevalcap Java subprocess incompatible)",
    }
    safe_write("priority0_report.json", json.dumps(report, indent=2))
    log(json.dumps(report, indent=2)[:2600])
    log(">> P0 COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
