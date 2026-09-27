"""
Experiment #3 — LM reranking of beam hypotheses.

For each val image: generate top-5 diverse beam hypotheses with the champion
model, score each under a small masked-LM (distilbert-base-uncased, local),
pick the one with the lowest pseudo-perplexity. Compare BLEU/ROUGE of
reranked vs original beam-5 output on the full 1,214-image split.

Zero training. If it fails, nothing changes in production.
"""
import json, math, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BACKEND = Path(__file__).resolve().parent
PROJECT = BACKEND.parent
sys.path.insert(0, str(BACKEND))
import os
os.chdir(BACKEND)

from inference.attention_provider import AttentionProvider
from PIL import Image
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

device = "cuda" if torch.cuda.is_available() else "cpu"
LM_NAME = os.environ.get("RERANK_LM", "distilbert-base-uncased")

# ---------------- load models ----------------
p = AttentionProvider()
p.load()

from transformers import AutoModelForMaskedLM, AutoTokenizer
lm_tok = AutoTokenizer.from_pretrained(LM_NAME, local_files_only=True)
lm = AutoModelForMaskedLM.from_pretrained(LM_NAME, local_files_only=True).to(device).eval()
print(f">> reranker: {LM_NAME} loaded", flush=True)


@torch.no_grad()
def pseudo_perplexity(sentence: str) -> float:
    """Mean negative log-likelihood per token under the MLM (lower = more fluent)."""
    words = sentence.split()
    if len(words) < 2:
        return 0.0
    nll_sum, n_tokens = 0.0, 0
    for i in range(len(words)):
        masked = words[:]; masked[i] = lm_tok.mask_token
        enc = lm_tok(" ".join(masked), return_tensors="pt",
                     truncation=True, max_length=64).to(device)
        logits = lm(**enc).logits[0]
        mask_pos = (enc["input_ids"][0] == lm_tok.mask_token_id).nonzero()
        if mask_pos.numel() == 0:
            continue
        pos = int(mask_pos[0])
        logprobs = torch.log_softmax(logits[pos], -1)
        tgt_ids = lm_tok.encode(words[i], add_special_tokens=False)
        if not tgt_ids:
            continue
        tgt = torch.tensor(tgt_ids, device=device)
        lp = torch.log_softmax(logits[pos], -1)
        # sum log-probs of the word's wordpieces (approximate token prob)
        nll = -lp[tgt].mean().item()
        nll_sum += nll
        n_tokens += 1
    return nll_sum / max(n_tokens, 1)


def rouge_l(hyp, ref):
    n, m = len(hyp), len(ref)
    if n == 0 or m == 0:
        return 0.0
    dp = np.zeros((n + 1, m + 1), dtype="int32")
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = (dp[i - 1, j - 1] + 1) if hyp[i - 1] == ref[j - 1] \
                else max(dp[i - 1, j], dp[i, j - 1])
    lcs = int(dp[n, m])
    if not lcs:
        return 0.0
    p_, r_ = lcs / n, lcs / m
    return 2 * p_ * r_ / (p_ + r_)


# ---------------- data ----------------
data = pd.read_csv(PROJECT / "dataset/captions.txt")
with open(PROJECT / "weights/val_images.json") as fh:
    val_images = json.load(fh)
refs = {}
for img in val_images:
    rows = data[data["image"] == img]["caption"].tolist()
    refs[img] = [[w for w in c.split() if w not in ("startseq", "endseq")]
                 for c in rows]

SM = SmoothingFunction().method1


def score_set(pairs):
    b1s, b4s, rls = [], [], []
    for img, cap in pairs:
        words = cap.split()
        gt = refs[img]
        b1s.append(sentence_bleu(gt, words, weights=(1, 0, 0, 0),
                                 smoothing_function=SM) if words else 0)
        b4s.append(sentence_bleu(gt, words, weights=(.25,) * 4,
                                 smoothing_function=SM) if words else 0)
        rls.append(rouge_l(words, gt[0]))
    return (float(np.mean(b1s)), float(np.mean(b4s)), float(np.mean(rls)))


# ---------------- generate hypotheses ----------------
print(">> generating beam hypotheses …", flush=True)
hypotheses = {}
t0 = time.time()

# use the provider's internals for beam diversity: monkeypatch DBS to get
# distinct beams, then take the top-5 finished candidates
import inference.attention_provider as ap
ap.DBS_LAMBDA = 0.3   # mild diversity so the 5 hypotheses differ

orig_beam_search = ap.AttentionProvider._beam_search
def beam_search_topk(self, feat, k=5):
    """Return list of (score, caption_words) for top-k finished beams."""
    dec = self.decoder
    bw = self.beam_width
    alpha = self.length_penalty
    gnmt = lambda lp, ids: lp / (((5 + max(len(ids), 1)) / 6) ** alpha)
    proj = dec.attention.enc_proj(feat)
    lps = [0.0]
    ids_list = [[ap.START_ID]]
    probs_list = [[]]
    H = torch.zeros(1, 1, dec.dec_dim, device=self.device)
    C = torch.zeros(1, 1, dec.dec_dim, device=self.device)
    finished = []
    while ids_list:
        B = len(ids_list)
        f_rep = feat.expand(B, -1, -1)
        p_rep = proj.expand(B, -1, -1)
        h_all = H[:, :B, :]; c_all = C[:, :B, :]
        prev = torch.tensor([ids[-1] for ids in ids_list],
                            dtype=torch.long, device=self.device)
        ctx = dec.attention.context(p_rep, f_rep, h_all[0])
        inp = torch.cat([dec.embedding(prev), ctx], dim=-1).unsqueeze(1)
        out, (h_new, c_new) = dec.lstm(inp, (h_all.contiguous(), c_all.contiguous()))
        step_logp = torch.log_softmax(dec.fc(out[:, 0]).float(), dim=-1)
        if ap.MIN_LENGTH and len(ids_list[0]) - 1 < ap.MIN_LENGTH:
            step_logp[:, ap.END_ID] = -1e9
        cand_lp, cand_ids, cand_h, cand_c = [], [], [], []
        tv, ti = torch.topk(step_logp, bw, dim=-1)
        for b in range(B):
            base_lp, base_ids = lps[b], ids_list[b]
            hb = h_new[:, b:b+1, :]; cb = c_new[:, b:b+1, :]
            existing = {(base_ids[k], base_ids[k+1]) for k in range(len(base_ids)-1)}
            for v, i in zip(tv[b].tolist(), ti[b].tolist()):
                lp_new = base_lp + v
                if len(base_ids) >= 2 and (base_ids[-1], i) in existing:
                    lp_new += float(np.log(0.4))
                cand_lp.append(lp_new); cand_ids.append(base_ids + [i])
                cand_h.append(hb); cand_c.append(cb)
        scored = [(gnmt(cand_lp[k], cand_ids[k]), k) for k in range(len(cand_lp))]
        scored.sort(key=lambda t_: t_[0], reverse=True)
        sel, sel_ng = [], set()
        for s0, k in scored:
            if len(sel) >= bw:
                break
            ngs = {tuple(cand_ids[k][j:j+ap.DBS_NGRAM])
                   for j in range(len(cand_ids[k]) - ap.DBS_NGRAM + 1)}
            ov = len(ngs & sel_ng)
            cand_lp[k] -= ap.DBS_LAMBDA * ov
            sel.append(k); sel_ng |= ngs
        order = sorted(sel, key=lambda k: gnmt(cand_lp[k], cand_ids[k]), reverse=True)
        lps = [cand_lp[k] for k in order]
        ids_list = [cand_ids[k] for k in order]
        H = torch.cat([cand_h[k] for k in order], dim=1)
        C = torch.cat([cand_c[k] for k in order], dim=1)
        still_lp, still_ids, still_h, still_c = [], [], [], []
        for j, ids in enumerate(ids_list):
            if ids[-1] == ap.END_ID:
                finished.append((gnmt(lps[j], ids), ids))
            else:
                still_lp.append(lps[j]); still_ids.append(ids)
                still_h.append(H[:, j:j+1, :]); still_c.append(C[:, j:j+1, :])
        lps, ids_list = still_lp, still_ids
        if still_h:
            H = torch.cat(still_h, dim=1); C = torch.cat(still_c, dim=1)
        else:
            break
        if len(finished) >= bw:
            break
    if not finished:
        finished = [(gnmt(lps[j], ids_list[j]), ids_list[j]) for j in range(len(lps))]
    finished.sort(key=lambda c: c[0], reverse=True)
    out = []
    seen = set()
    for sc, ids in finished:
        cap = self.tokenizer.decode(
            [t for t in ids if t not in (ap.PAD_ID, ap.START_ID, ap.END_ID)]).strip()
        ws = cap.split()
        FUNC = {"a","an","the","on","in","at","of","with","and","or","to","by","for"}
        while len(ws) > 2 and ws[-1] in FUNC:
            ws.pop()
        key = " ".join(ws)
        if key not in seen:
            seen.add(key)
            out.append((sc, key))
        if len(out) >= k:
            break
    return out

ap.AttentionProvider._beam_search = orig_beam_search  # keep class intact
p2 = AttentionProvider(); p2.load()
p2.beam_width = 5

for n, img in enumerate(val_images):
    image = Image.open(str(PROJECT / "dataset/Images" / img)).convert("RGB")
    feat = p2._encode_image(image)
    hyps = beam_search_topk(p2, feat, k=5)
    hypotheses[img] = hyps
    if (n + 1) % 300 == 0:
        print(f"  {n+1}/{len(val_images)} ({time.time()-t0:.0f}s)", flush=True)

print(f">> hypotheses done ({time.time()-t0:.0f}s)", flush=True)

# baseline = top beam as-is
baseline_pairs = [(img, hyps[0][1]) for img, hyps in hypotheses.items()]
b1, b4, rl = score_set(baseline_pairs)
print(f"BASELINE beam-top   BLEU-1 {b1:.4f} | BLEU-4 {b4:.4f} | ROUGE-L {rl:.4f}",
      flush=True)

# ---------------- rerank ----------------
print(">> reranking with MLM fluency …", flush=True)
t1 = time.time()
reranked_pairs, changed, tie = [], 0, 0
for n, (img, hyps) in enumerate(hypotheses.items()):
    best_cap, best_s = None, None
    for _, cap in hyps:
        s = pseudo_perplexity(cap)
        if best_s is None or s < best_s:
            best_s, best_cap = s, cap
    if best_cap == hyps[0][1]:
        tie += 1
    else:
        changed += 1
    reranked_pairs.append((img, best_cap))
    if (n + 1) % 200 == 0:
        print(f"  {n+1}/{len(hypotheses)} ({time.time()-t1:.0f}s)", flush=True)

b1r, b4r, rlr = score_set(reranked_pairs)
print(f"RERANKED MLM        BLEU-1 {b1r:.4f} | BLEU-4 {b4r:.4f} | ROUGE-L {rlr:.4f}",
      flush=True)
print(f"\nchanged: {changed}/{len(hypotheses)} ({changed/len(hypotheses):.1%}), "
      f"unchanged: {tie}")
print(f"deltas: BLEU-1 {b1r-b1:+.4f} | BLEU-4 {b4r-b4:+.4f} | ROUGE-L {rlr-rl:+.4f}")

out = {"lm": LM_NAME, "n": len(hypotheses), "changed": changed,
       "baseline": {"bleu1": b1, "bleu4": b4, "rouge_l": rl},
       "reranked": {"bleu1": b1r, "bleu4": b4r, "rouge_l": rlr},
       "delta": {"bleu1": b1r-b1, "bleu4": b4r-b4, "rouge_l": rlr-rl}}
Path("../experiments/rerank_eval.json").write_text(json.dumps(out, indent=2))
print("saved ../experiments/rerank_eval.json")
