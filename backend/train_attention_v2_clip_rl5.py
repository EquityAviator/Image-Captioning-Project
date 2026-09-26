"""
GRPO RL v5 — CHAIR-aware reward with REAL-EVALUATION epoch scoring.

Fixes the rl4 evaluation gap: rl4 tracked CHAIR with greedy decoding on a
300-image subset, which showed a huge "fix" (0.68 -> 0.28) that vanished on
the real beam-5 full-split eval (49.5% vs champion 47.3%). rl5 closes that
gap:

  * Every epoch ends with a BATCHED beam-5 decode (standard protocol: bw 5,
    GNMT alpha 1.2, soft 2-gram penalty, no min-len guard — identical to
    eval_v2_clip.py) over the FULL 1,214-image validation split.
  * Full metrics each epoch: BLEU-1/2/3/4, ROUGE-L, CIDEr-D
    (pycocoevalcap), CHAIR-lite (chair_image_rate + hallucinated_noun_rate).
  * Checkpoint selection on the combined full-split objective
        J = CIDEr-D  -  chair_image_rate
    (both in [0,1] scale; strict improvement over pre-RL baseline J).
  * Calibration regularizer UNCHANGED from rl4 (proven): + calib_gamma *
    mean max-prob on rollouts. Final pass reports ECE at T=1.0 / 1.3.

Reward (unchanged from rl4): R = CIDEr-D - chair_lambda * hallucinated_nouns
on GRPO rollouts (4 imgs x 5 samples per step — host-memory-safe).

Success criteria (printed every epoch, from the champion's verified numbers):
  1. chair_image_rate  < 0.473   (full-split beam-5)
  2. bleu4 >= 0.1727 and cider_d >= 0.5243
  3. (final) ECE at T=1.3 <= 0.0862
Source ckpt: v2_attention_clip/best.pt (same-init lineage).
Output:      experiments/v2_attention_clip_rl5/best.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"
WEIGHTS_DIR = PROJECT_DIR / "weights"
OUT_DIR = PROJECT_DIR / "experiments" / "v2_attention_clip_rl5"

sys.path.insert(0, str(BACKEND_DIR))
from train_torch import text_preprocessing  # noqa: E402
from train_attention_v2_clip import AttentionDecoderV2  # noqa: E402
from train_attention_v2_clip_rl4 import (  # noqa: E402
    FastCiderD, FastChairLite, build_refs)

START, END, PAD = 1, 2, 0


# --------------------------------------------------------------------------- #
# Batched beam-5, standard protocol — N images decoded together.
# Each image keeps its own beam pool; sequences from all live beams share one
# LSTM forward per step (chunked to bound memory). No DBS, no min-len guard —
# byte-identical semantics to eval_v2_clip.py's per-image decode.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def beam5_batched(model, feats_list, tok, bw=5, alpha=1.2, max_len=38,
                  chunk=24, log=None):
    device = next(model.parameters()).device
    dec = model
    N = len(feats_list)
    results = [None] * N

    def gnmt(lp, ids):
        return lp / (((5 + max(len(ids), 1)) / 6) ** alpha)

    for c0 in range(0, N, chunk):
        chunk_idx = list(range(c0, min(c0 + chunk, N)))
        feats = torch.from_numpy(
            np.stack(feats_list[c0:c0 + chunk])).to(device).float()
        proj = dec.attention.enc_proj(feats)               # (B, P, att)
        B = len(chunk_idx)

        lps = [[0.0] for _ in range(B)]
        ids = [[[START]] for _ in range(B)]
        Hs = [[torch.zeros(1, 1, dec.dec_dim, device=device)] for _ in range(B)]
        Cs = [[torch.zeros(1, 1, dec.dec_dim, device=device)] for _ in range(B)]
        finished = [[] for _ in range(B)]

        for t in range(max_len):
            live = [(bi, j) for bi in range(B) for j in range(len(ids[bi]))]
            if not live:
                break
            img_of = [x[0] for x in live]
            f_rep = feats[img_of]
            p_rep = proj[img_of]
            h_all = torch.cat([Hs[bi][j] for bi, j in live], dim=1)
            c_all = torch.cat([Cs[bi][j] for bi, j in live], dim=1)
            prev = torch.tensor([ids[bi][j][-1] for bi, j in live],
                                dtype=torch.long, device=device)
            ctx = dec.attention.context(p_rep, f_rep, h_all[0])
            inp = torch.cat([dec.embedding(prev), ctx], dim=-1).unsqueeze(1)
            out, (hN, cN) = dec.lstm(inp, (h_all.contiguous(),
                                           c_all.contiguous()))
            step_logp = torch.log_softmax(dec.fc(out[:, 0]).float(), -1)
            tv, ti = torch.topk(step_logp, bw, dim=-1)

            new_lps = [[] for _ in range(B)]
            new_ids = [[] for _ in range(B)]
            new_h = [[] for _ in range(B)]
            new_c = [[] for _ in range(B)]
            for k, (bi, j) in enumerate(live):
                base_lp, base_ids = lps[bi][j], ids[bi][j]
                hb = hN[:, k:k + 1, :]
                cb = cN[:, k:k + 1, :]
                existing = {(base_ids[m], base_ids[m + 1])
                            for m in range(len(base_ids) - 1)}
                for v, i in zip(tv[k].tolist(), ti[k].tolist()):
                    lp_new = base_lp + v
                    if len(base_ids) >= 2 and (base_ids[-1], i) in existing:
                        lp_new += float(np.log(0.4))
                    new_lps[bi].append(lp_new)
                    new_ids[bi].append(base_ids + [i])
                    new_h[bi].append(hb)
                    new_c[bi].append(cb)

            for bi in range(B):
                if not new_ids[bi]:
                    continue
                order = sorted(range(len(new_lps[bi])),
                               key=lambda k: gnmt(new_lps[bi][k], new_ids[bi][k]),
                               reverse=True)[:bw]
                lps[bi] = [new_lps[bi][k] for k in order]
                ids[bi] = [new_ids[bi][k] for k in order]
                Hs[bi] = [new_h[bi][k] for k in order]
                Cs[bi] = [new_c[bi][k] for k in order]
                still_lp, still_ids, still_H, still_C = [], [], [], []
                for k in range(len(ids[bi])):
                    if ids[bi][k][-1] == END:
                        finished[bi].append(
                            (gnmt(lps[bi][k], ids[bi][k]), ids[bi][k]))
                    else:
                        still_lp.append(lps[bi][k])
                        still_ids.append(ids[bi][k])
                        still_H.append(Hs[bi][k])
                        still_C.append(Cs[bi][k])
                lps[bi], ids[bi] = still_lp, still_ids
                Hs[bi], Cs[bi] = still_H, still_C

        for bi in range(B):
            pool = finished[bi] or [(gnmt(lps[bi][j], ids[bi][j]),
                                     ids[bi][j]) for j in range(len(ids[bi]))]
            best = max(pool, key=lambda c: c[0])[1]
            results[chunk_idx[bi]] = tok.decode(
                [t for t in best[1:] if t not in (PAD, START, END)]).strip()
        if log:
            log(f"   beam5 {min(c0 + B, N)}/{N}", flush=True)
    return results


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


def log(msg, **kw):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--images-per-step", type=int, default=4)
    ap.add_argument("--group", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chair-lambda", type=float, default=0.05)
    ap.add_argument("--calib-gamma", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> device {device} | R = CIDEr - {args.chair_lambda}*hallu | "
          f"calib γ={args.calib_gamma} | epoch eval = beam-5 FULL split",
          flush=True)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(WEIGHTS_DIR / "bpe" / "tokenizer.json"))

    data = pd.read_csv(DATASET_DIR / "captions.txt")
    data = text_preprocessing(data)
    train_images = json.loads((WEIGHTS_DIR / "train_images.json").read_text())
    val_images = json.loads((WEIGHTS_DIR / "val_images.json").read_text())
    if args.limit:
        train_images = train_images[:args.limit]
        val_images = val_images[:args.limit]
    train_refs = build_refs(data, train_images)
    val_refs = build_refs(data, val_images)

    print(">> building scorers …", flush=True)
    scorer = FastCiderD(list(train_refs.values()))
    chair = FastChairLite(data, train_images + val_images)

    src_ckpt = PROJECT_DIR / "experiments" / "v2_attention_clip" / "best.pt"
    ckpt = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    vocab = int(ckpt["config"].get("vocab_size", ckpt["config"]["vocab"]))
    max_len = int(ckpt["config"]["max_length"])
    model = AttentionDecoderV2(vocab).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train()
    print(f">> loaded {src_ckpt} (epoch {ckpt.get('epoch')})", flush=True)

    ema = copy.deepcopy(model.state_dict())
    feats_all = np.load(WEIGHTS_DIR / "clip_features_spatial.npy", mmap_mode="r")
    feat_index = {n: i for i, n in
                  enumerate(json.loads((WEIGHTS_DIR / "image_names.json").read_text()))}

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def greedy_caption(feat_row):
        f = torch.from_numpy(np.asarray(feat_row, dtype=np.float32)[None]).to(device)
        ids = [START]
        with torch.no_grad():
            for _ in range(max_len):
                x = torch.tensor([ids], dtype=torch.long, device=device)
                nxt = int(model(f, x)[0, -1].argmax())
                if nxt == END:
                    break
                ids.append(nxt)
        return tok.decode([t for t in ids[1:] if t not in (PAD, START, END)]).strip()

    # ---------------- REAL-EVAL epoch scoring (full split, beam-5) ---------- #
    def full_split_eval(tag):
        model.eval()
        t0 = time.time()
        feats_list = [feats_all[feat_index[i]] for i in val_images]
        caps = beam5_batched(model, feats_list, tok, bw=5, alpha=1.2,
                             max_len=max_len, log=log)
        agg = {k: 0.0 for k in ("bleu1", "bleu2", "bleu3", "bleu4", "rouge_l")}
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        sm = SmoothingFunction().method1
        hyp_c, ref_c = {}, {}
        chair_imgs = 0
        hallu_n = nouns_n = 0
        lens = []
        from nltk import pos_tag
        for img, cap in zip(val_images, caps):
            words = cap.split()
            lens.append(len(words))
            gt = val_refs[img]
            hyp_c[img] = [cap]
            ref_c[img] = [" ".join(r) for r in gt]
            if words:
                agg["bleu1"] += sentence_bleu(gt, words, weights=(1, 0, 0, 0), smoothing_function=sm)
                agg["bleu2"] += sentence_bleu(gt, words, weights=(0.5, 0.5, 0, 0), smoothing_function=sm)
                agg["bleu3"] += sentence_bleu(gt, words, weights=(1 / 3,) * 3, smoothing_function=sm)
                agg["bleu4"] += sentence_bleu(gt, words, weights=(0.25,) * 4, smoothing_function=sm)
                agg["rouge_l"] += rouge_l(words, gt[0])
            try:
                gen = {w.lower() for w, t in pos_tag(words) if t.startswith("NN")}
            except Exception:
                gen = set()
            ref_nouns = chair.ref_nouns[img]
            if gen and ref_nouns:
                h = len(gen - ref_nouns)
                nouns_n += len(gen)
                hallu_n += h
                if h:
                    chair_imgs += 1
        n = len(val_images)
        metrics = {k: round(v / n, 4) for k, v in agg.items()}
        try:
            from pycocoevalcap.cider.cider import Cider
            cs, _ = Cider().compute_score(ref_c, hyp_c)
            metrics["cider_d"] = round(float(cs), 4)
        except Exception as exc:  # noqa: BLE001
            metrics["cider_d"] = None
            log(f"!! CIDEr failed: {exc}")
        metrics["chair_img_rate"] = round(chair_imgs / n, 4)
        metrics["hallu_noun_rate"] = round(hallu_n / max(nouns_n, 1), 4)
        metrics["len"] = round(float(np.mean(lens)), 2)
        metrics["eval_seconds"] = round(time.time() - t0, 1)
        model.train()
        return metrics

    def calib_ece(temps=(1.0, 1.3)):
        """Teacher-forced ECE on full val at given temperatures."""
        model.eval()
        encoded = json.loads((WEIGHTS_DIR / "bpe" / "encoded_captions.json").read_text())
        per = {t: {"conf": [], "cor": []} for t in temps}
        bs = 48
        with torch.no_grad():
            for s in range(0, len(val_images), bs):
                imgs = val_images[s:s + bs]
                f = torch.from_numpy(
                    feats_all[[feat_index[i] for i in imgs]].copy()).to(device).float()
                xs, lens = [], []
                for img in imgs:
                    seq = encoded[img][0][:max_len]
                    xs.append(seq + [0] * (max_len - len(seq)))
                    lens.append(min(len(encoded[img][0]), max_len))
                x = torch.tensor(xs, dtype=torch.long, device=device)
                logits = model(f, x)
                for T in temps:
                    probs = torch.softmax(logits.float() / T, -1).cpu().numpy()
                    for bi, img in enumerate(imgs):
                        L = lens[bi]
                        tgt = encoded[img][0][1:L]
                        for t_i, t_true in enumerate(tgt):
                            p = probs[bi, t_i]
                            per[T]["conf"].append(float(p.max()))
                            per[T]["cor"].append(int(p.argmax() == t_true))
        model.train()
        out = {}
        for T in temps:
            conf = np.array(per[T]["conf"]); corr = np.array(per[T]["cor"])
            bins = np.linspace(0, 1, 11)
            e, total = 0.0, len(conf)
            for b in range(10):
                lo, hi = bins[b], bins[b + 1]
                m = (conf > lo) & (conf <= hi) if b else (conf >= lo) & (conf <= hi)
                if m.any():
                    e += (m.sum() / total) * abs(float(corr[m].mean())
                                                 - float(conf[m].mean()))
            out[str(T)] = round(e, 4)
        return out

    # ---------------- pre-RL baseline on the SAME protocol ---------------- #
    log(">> pre-RL baseline: beam-5 FULL split …", flush=True)
    base = full_split_eval("pre")
    baseJ = base["cider_d"] - base["chair_img_rate"]
    log(f">> pre-RL FULL-SPLIT: B1={base['bleu1']} B4={base['bleu4']} "
        f"RL={base['rouge_l']} CIDEr={base['cider_d']} "
        f"CHAIR-img={base['chair_img_rate']} hallu={base['hallu_noun_rate']} "
        f"len={base['len']} J={baseJ:.4f}", flush=True)
    (OUT_DIR / "pre_rl_fullsplit.json").write_text(json.dumps(base, indent=2))

    best_J = baseJ
    history = []
    t_start = time.time()
    order = train_images[:]
    step = 0

    for epoch in range(1, args.epochs + 1):
        random.shuffle(order)
        ep_rewards = []
        for s in range(0, len(order) - args.images_per_step + 1,
                       args.images_per_step):
            imgs = order[s:s + args.images_per_step]
            B, G = len(imgs), args.group
            feats_np = np.stack([feats_all[feat_index[i]] for i in imgs])
            feats_1 = torch.from_numpy(feats_np).to(device).float()
            feat_rep = feats_1.unsqueeze(1).expand(
                B, G, *feats_1.shape[1:]).reshape(B * G, *feats_1.shape[1:])
            N = B * G

            ids = torch.full((N, 1), START, dtype=torch.long, device=device)
            done = torch.zeros(N, dtype=torch.bool, device=device)
            token_logps, max_probs = [], []

            model.train()
            with torch.enable_grad():
                for t in range(args.max_len):
                    logits = model(feat_rep, ids)
                    logp_all = torch.log_softmax(logits[:, -1].float(), -1)
                    max_probs.append(logp_all.exp().max(-1).values.detach())
                    probs = logp_all.exp()
                    nxt = torch.multinomial(probs.detach(), 1).squeeze(1)
                    chosen = torch.where(done, torch.full_like(nxt, PAD), nxt)
                    lp_t = logp_all.gather(1, chosen.unsqueeze(1)).squeeze(1)
                    token_logps.append(lp_t)
                    ids = torch.cat([ids, chosen.unsqueeze(1)], dim=1)
                    done = done | (chosen == END)
                    if bool(done.all()):
                        break

            ids_cpu = ids[:, 1:].cpu().numpy()
            rewards = np.zeros(N)
            for bi in range(B):
                img = imgs[bi]
                for g in range(G):
                    seq = ids_cpu[bi * G + g]
                    toks = [int(t) for t in seq if t not in (PAD, START, END)]
                    words = tok.decode(toks).strip().split()
                    cider = scorer.score(words, train_refs[img])
                    hallu_n = chair.hallucinated_count(words, img)
                    rewards[bi * G + g] = cider - args.chair_lambda * hallu_n
            ep_rewards.append(float(rewards.mean()))

            adv = torch.from_numpy(
                (rewards.reshape(B, G) - rewards.reshape(B, G).mean(1, keepdims=True))
                / (rewards.reshape(B, G).std(1, keepdims=True) + 1e-6)
            ).float().reshape(N).to(device)

            seq_lp = torch.stack(token_logps, 0)
            pad_mask = (ids[:, 1:].transpose(0, 1) != PAD).float()
            seq_lp = seq_lp * pad_mask
            loss = -(seq_lp.sum(0) * adv).mean()
            if args.calib_gamma > 0:
                maxp = torch.stack(max_probs, 0)
                loss = loss + args.calib_gamma * maxp.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if step % 100 == 0:
                torch.cuda.empty_cache()

            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(0.999).add_(v, alpha=0.001)
                    else:
                        ema[k].copy_(v)
            step += 1
            if step % 50 == 0:
                print(f"  ep {epoch} step {step}: reward "
                      f"{np.mean(ep_rewards[-50:]):.4f} "
                      f"({time.time()-t_start:.0f}s)", flush=True)

        # ---- REAL-EVAL epoch scoring ----
        m = full_split_eval(f"ep{epoch}")
        m["reward"] = round(float(np.mean(ep_rewards)), 4)
        m["epoch"] = epoch
        J = m["cider_d"] - m["chair_img_rate"]
        m["J"] = round(J, 4)
        history.append(m)
        crit1 = m["chair_img_rate"] < 0.473
        crit2 = (m["bleu4"] >= 0.1727) and (m["cider_d"] >= 0.5243)
        print(f"epoch {epoch}: B1={m['bleu1']} B4={m['bleu4']} RL={m['rouge_l']} "
              f"CIDEr={m['cider_d']} CHAIR-img={m['chair_img_rate']} "
              f"hallu={m['hallu_noun_rate']} len={m['len']} J={m['J']} | "
              f"crit1(CHAIR<47.3%)={'PASS' if crit1 else 'fail'} "
              f"crit2(no-regress)={'PASS' if crit2 else 'fail'} "
              f"({time.time()-t_start:.0f}s)", flush=True)

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "config": ckpt["config"]}, OUT_DIR / "ckpt_last.pt")
        if J > best_J + 1e-4:
            best_J = J
            torch.save({"epoch": epoch, "model_state_dict": ema,
                        "used_ema": True, "val_cider": m["cider_d"],
                        "chair_img_rate": m["chair_img_rate"],
                        "config": ckpt["config"],
                        "reward": f"cider-{args.chair_lambda}*hallu+calib",
                        "selection": "J = cider - chair_img_rate (full-split beam5)",
                        "encoder": "clip-vit-b16-patches"},
                       OUT_DIR / "best.pt")
            print(f"   ✓ new best J {J:.4f} saved", flush=True)
        (OUT_DIR / "train_history.json").write_text(json.dumps(
            {"pre": base, "pre_J": baseJ, "history": history,
             "config": vars(args)}, indent=2))

    # ---------------- final: calibration of the best checkpoint ------------- #
    log(">> final calibration pass on best checkpoint …", flush=True)
    best_ckpt = torch.load(OUT_DIR / "best.pt", map_location="cpu",
                           weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()
    eces = calib_ece(temps=(1.0, 1.3))
    crit3 = eces["1.3"] <= 0.0862
    print(f">> FINAL: ECE@T1.0={eces['1.0']} ECE@T1.3={eces['1.3']} | "
          f"crit3(calib<=0.0862)={'PASS' if crit3 else 'fail'}", flush=True)

    summary = {
        "reward": f"cider-{args.chair_lambda}*hallu+calib",
        "selection": "J = cider_d - chair_img_rate (beam-5 full split)",
        "pre": base, "pre_J": baseJ,
        "best_J": round(best_J, 4),
        "final_ece": eces,
        "success_criteria": {
            "crit1_chair_lt_0473 (full-split beam5)": "see per-epoch history",
            "crit2_bleu4_cider_no_regress": "see per-epoch history",
            "crit3_ece_T1.3_le_0.0862": "PASS" if crit3 else "fail",
        },
        "history": history, "config": vars(args)}
    (OUT_DIR / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(">> done.", flush=True)


if __name__ == "__main__":
    main()
