"""
Step 2 — Length-scaled CIDEr GRPO:  R = CIDEr-D * sqrt(len_cand / len_ref_mean)

Rationale (from the failed rl2 mixed-reward experiment): ROUGE-L mixing added
noise without countering the length bias. This reward instead MULTIPLICATIVELY
penalizes captions shorter than the reference mean, directly attacking the
root cause while keeping CIDEr's precision signal intact.

Source ckpt: experiments/v2_attention_clip/best.pt (CLIP CE)
Output:      experiments/v2_attention_clip_rl3/best.pt
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
OUT_DIR = PROJECT_DIR / "experiments" / "v2_attention_clip_rl3"

sys.path.insert(0, str(BACKEND_DIR))
from train_torch import text_preprocessing  # noqa: E402
from train_attention_v2_clip import AttentionDecoderV2  # noqa: E402

START, END, PAD = 1, 2, 0


def _ngrams(words, n):
    return Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


class FastCiderD:
    def __init__(self, refs_corpus):
        self.df = [Counter() for _ in range(4)]
        n_docs = len(refs_corpus)
        for refs in refs_corpus:
            seen = [set() for _ in range(4)]
            for r in refs:
                for n in range(1, 5):
                    for g in _ngrams(r, n):
                        seen[n - 1].add(g)
            for n in range(4):
                self.df[n].update(seen[n])
        self.n_docs = n_docs
        self.weights = [1.0] * 4

    def _idf(self, n, gram):
        return math.log10(max(1.0, self.n_docs) / (1.0 + self.df[n - 1][gram]))

    def score_one(self, cand, refs):
        total_w, acc, c_len = 0.0, 0.0, len(cand)
        for n in range(1, 5):
            c_ng = _ngrams(cand, n)
            if not c_ng:
                continue
            sims = []
            for r in refs:
                r_ng = _ngrams(r, n)
                r_len = len(r)
                if not r_ng:
                    sims.append(0.0)
                    continue
                num = sum(min(cc, r_ng.get(g, 0)) * self._idf(n, g)
                          for g, cc in c_ng.items())

                def norm(counts):
                    return math.sqrt(sum((cnt * self._idf(n, g)) ** 2
                                         for g, cnt in counts.items()))

                denom = norm(c_ng) * norm(r_ng)
                sim = (num / denom) if denom > 0 else 0.0
                lp = math.exp(-((c_len - r_len) ** 2) / (2 * 6.0 ** 2))
                sims.append(sim * lp)
            acc += sum(sims) / len(sims)
            total_w += 1
        return 10.0 * acc / max(total_w, 1e-9)

    def score(self, cand, refs):
        return self.score_one(cand, refs)


def build_refs(data, images):
    refs = {}
    for img in images:
        rows = data[data["image"] == img]
        refs[img] = [[w for w in c.split() if w not in ("startseq", "endseq")]
                     for c in rows["caption"].tolist()]
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--images-per-step", type=int, default=8)
    ap.add_argument("--group", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-subset", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> device: {device} | reward = CIDEr * sqrt(len/ref_mean)",
          flush=True)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(WEIGHTS_DIR / "bpe" / "tokenizer.json"))

    data = pd.read_csv(DATASET_DIR / "captions.txt")
    data = text_preprocessing(data)
    train_images = json.loads((WEIGHTS_DIR / "train_images.json").read_text())
    val_images = json.loads((WEIGHTS_DIR / "val_images.json").read_text())
    if args.limit:
        train_images = train_images[:args.limit]
    train_refs = build_refs(data, train_images)
    val_refs = build_refs(data, val_images)

    # precompute reference length means for the length-scaling term
    ref_len_mean = {img: float(np.mean([len(r) for r in rs]))
                    for img, rs in train_refs.items()}
    val_ref_len_mean = {img: float(np.mean([len(r) for r in rs]))
                        for img, rs in val_refs.items()}

    print(">> building fixed-df CIDEr-D …", flush=True)
    scorer = FastCiderD(list(train_refs.values()))

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
                logits = model(f, x)
                nxt = int(logits[0, -1].argmax())
                if nxt == END:
                    break
                ids.append(nxt)
        return tok.decode([t for t in ids[1:] if t not in (PAD, START, END)]).strip()

    def val_metrics(subset):
        model.eval()
        cs, b1s, b4s, lens = [], [], [], []
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        sm = SmoothingFunction().method1
        with torch.no_grad():
            for img in subset:
                cap = greedy_caption(feats_all[feat_index[img]])
                words = cap.split()
                lens.append(len(words))
                cs.append(scorer.score(words, val_refs[img]))
                b1s.append(sentence_bleu(val_refs[img], words, weights=(1, 0, 0, 0),
                                         smoothing_function=sm) if words else 0.0)
                b4s.append(sentence_bleu(val_refs[img], words, weights=(0.25,) * 4,
                                         smoothing_function=sm) if words else 0.0)
        model.train()
        return (float(np.mean(cs)), float(np.mean(b1s)), float(np.mean(b4s)),
                float(np.mean(lens)))

    val_subset = val_images[:args.val_subset]
    base = val_metrics(val_subset)
    print(f">> pre-RL val subset: CIDEr={base[0]:.4f} BLEU-1={base[1]:.4f} "
          f"BLEU-4={base[2]:.4f} len={base[3]:.1f}", flush=True)

    history = {"cider": [], "bleu1": [], "bleu4": [], "len": [], "reward": []}
    best_cider = base[0]
    t0 = time.time()
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
            feats_1 = torch.from_numpy(feats_np.astype(np.float32)).to(device)
            feat_rep = feats_1.unsqueeze(1).expand(
                B, G, *feats_1.shape[1:]).reshape(B * G, *feats_1.shape[1:])
            N = B * G

            ids = torch.full((N, 1), START, dtype=torch.long, device=device)
            done = torch.zeros(N, dtype=torch.bool, device=device)
            token_logps = []

            model.train()
            with torch.enable_grad():
                for t in range(args.max_len):
                    logits = model(feat_rep, ids)
                    logp_all = torch.log_softmax(logits[:, -1].float(), -1)
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
                ref_lm = ref_len_mean[img]
                for g in range(G):
                    seq = ids_cpu[bi * G + g]
                    toks = [int(t) for t in seq if t not in (PAD, START, END)]
                    words = tok.decode(toks).strip().split()
                    cider = scorer.score(words, train_refs[img])
                    ratio = len(words) / max(ref_lm, 1e-6)
                    rewards[bi * G + g] = cider * math.sqrt(min(ratio, 2.0))
            ep_rewards.append(float(rewards.mean()))

            adv = torch.from_numpy(
                (rewards.reshape(B, G) - rewards.reshape(B, G).mean(1, keepdims=True))
                / (rewards.reshape(B, G).std(1, keepdims=True) + 1e-6)
            ).float().reshape(N).to(device)

            seq_lp = torch.stack(token_logps, 0)
            pad_mask = (ids[:, 1:].transpose(0, 1) != PAD).float()
            seq_lp = seq_lp * pad_mask
            loss = -(seq_lp.sum(0) * adv).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(0.999).add_(v, alpha=0.001)
                    else:
                        ema[k].copy_(v)
            step += 1
            if step % 25 == 0:
                print(f"  ep {epoch} step {step}: mean reward "
                      f"{np.mean(ep_rewards[-25:]):.4f} ({time.time()-t0:.0f}s)",
                      flush=True)

        c, b1, b4, L = val_metrics(val_subset)
        history["cider"].append(round(c, 4))
        history["bleu1"].append(round(b1, 4))
        history["bleu4"].append(round(b4, 4))
        history["len"].append(round(L, 1))
        history["reward"].append(round(float(np.mean(ep_rewards)), 4))
        print(f"epoch {epoch}: val CIDEr={c:.4f} BLEU-1={b1:.4f} BLEU-4={b4:.4f} "
              f"len={L:.1f} (pre {base[0]:.4f}) ({time.time()-t0:.0f}s)", flush=True)

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "config": ckpt["config"]}, OUT_DIR / "ckpt_last.pt")
        if c > best_cider + 1e-4:
            best_cider = c
            torch.save({"epoch": epoch, "model_state_dict": ema,
                        "used_ema": True, "val_cider": c,
                        "config": ckpt["config"],
                        "reward": "cider*sqrt(len/ref_mean)",
                        "encoder": "clip-vit-b16-patches"},
                       OUT_DIR / "best.pt")
            print(f"   ✓ new best CIDEr {c:.4f} saved", flush=True)

    (OUT_DIR / "train_summary.json").write_text(json.dumps({
        "reward": "cider*sqrt(len/ref_mean)",
        "best_val_cider_subset": best_cider,
        "history": history, "config": vars(args)}, indent=2))
    print(f">> done. best subset CIDEr {best_cider:.4f}", flush=True)


if __name__ == "__main__":
    main()
