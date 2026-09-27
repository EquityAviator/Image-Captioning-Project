#!/usr/bin/env python3
"""
Phase 3: Full Evaluation Harness

Computes all required metrics for any provider:
- Caption metrics: BLEU-1/2/3/4, ROUGE-L, CIDEr (optional), METEOR (optional)
- Token metrics: unsmoothed val loss, perplexity, top-1 prob, top-5 mass (SUM), top-1/5 accuracy, entropy, END prob, PAD prob
- Repetition metrics: % captions with repeated 2-gram/3-gram, max repeated n-gram count, avg caption length
- Latency metrics: encoder time, decoder time, total time, p50/p95

Usage:
    python scripts/evaluate.py --provider torch_baseline --split val --decode beam --beam_size 5 --output_dir experiments/corrected_lstm_baseline/
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
WEIGHTS_DIR = PROJECT_DIR / "weights"
SCRIPTS_DIR = PROJECT_DIR / "scripts"

sys.path.insert(0, str(PROJECT_DIR / "backend"))

from model.architecture import build_decoder, DEFAULT_IMG_SIZE
from inference.decoding import beam_search_decode, greedy_decode

# Special tokens - use tokenizer's actual indices
# Will be set after tokenizer loads
PAD_ID = 0
START_ID = None
END_ID = None
UNK_ID = None
VOCAB_SIZE = None


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
    """ROUGE-L F-measure over longest common subsequence of words."""
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


def load_model_and_tokenizer(checkpoint_path, vocab_size, max_length, device):
    """Load model from checkpoint"""
    import sys
    sys.path.insert(0, str(PROJECT_DIR / "backend"))
    from train_torch import CaptionDecoder
    
    model = CaptionDecoder(vocab_size, max_length).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, "rb") as f:
        return pickle.load(f)


def ids_to_words(tokenizer, tokens: list[int]) -> list[str]:
    words = []
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


def compute_token_metrics(model, feats, val_data, tokenizer, max_length, device, batch_size=64):
    """Compute token-level metrics on validation set"""
    model.eval()
    
    total_nll = 0.0
    total_tokens = 0
    top1_correct = 0
    top5_correct = 0
    top1_probs = []
    top5_masses = []
    entropies = []
    end_probs_at_end = []
    pad_probs = []
    total_preds = 0
    
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum")
    
    val_images = val_data["image"].unique()
    # feats is already indexed by val_images order, so create a simple index
    val_feat_index = {img: i for i, img in enumerate(val_images)}
    
    # Build references
    data = pd.read_csv(DATASET_DIR / "captions.txt")
    data = text_preprocessing(data)
    
    with torch.no_grad():
        for img in val_images:
            feat = feats[val_feat_index[img]][None, :].to(device)  # (1, 1920)
            rows = data[data["image"] == img]
            for _, row in rows.iterrows():
                tokens = [tokenizer.word_index[w] for w in row["caption"].split() if w in tokenizer.word_index]
                if len(tokens) < 2:
                    continue
                
                # Teacher forcing: predict next token at each position
                for i in range(1, len(tokens)):
                    prefix = tokens[:i]
                    target = tokens[i]
                    
                    seq = torch.zeros(1, max_length, dtype=torch.long, device=device)
                    seq[0, :len(prefix)] = torch.tensor(prefix, device=device)
                    
                    with torch.no_grad():
                        logits = model(feat, seq)  # (1, vocab)
                        probs = F.softmax(logits, dim=-1)[0]  # (vocab,)
                    
                    # NLL
                    nll = -torch.log(probs[target] + 1e-12).item()
                    total_nll += nll
                    total_tokens += 1
                    
                    # Top-1
                    pred = probs.argmax().item()
                    top1_correct += (pred == target)
                    top1_probs.append(probs[pred].item())
                    
                    # Top-5
                    top5_vals, top5_idx = probs.topk(5)
                    top5_correct += (target in top5_idx)
                    top5_masses.append(top5_vals.sum().item())
                    
                    # Entropy
                    ent = -(probs * torch.log(probs + 1e-12)).sum().item()
                    entropies.append(ent)
                    
                    # END probability at true END positions
                    if target == END_ID:
                        end_probs_at_end.append(probs[END_ID].item())
                    
                    # PAD probability (should be near 0)
                    pad_probs.append(probs[PAD_ID].item())
                    
                    total_preds += 1
    
    if total_tokens == 0:
        return {}
    
    return {
        "val_loss_unsmoothed": total_nll / total_tokens,
        "val_perplexity_unsmoothed": np.exp(total_nll / total_tokens),
        "top1_accuracy": top1_correct / total_preds,
        "top5_accuracy": top5_correct / total_preds,
        "avg_top1_probability": float(np.mean(top1_probs)),
        "avg_top5_probability_mass": float(np.mean(top5_masses)),
        "avg_entropy": float(np.mean(entropies)),
        "avg_end_prob_at_end": float(np.mean(end_probs_at_end)) if end_probs_at_end else 0.0,
        "avg_pad_prob": float(np.mean(pad_probs)),
        "total_predictions": total_preds
    }


def compute_caption_metrics(generated_captions, references, tokenizer):
    """Compute BLEU, ROUGE-L, repetition metrics"""
    smooth = SmoothingFunction().method1
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []
    rouge_l_scores = []
    rep_2gram = 0
    rep_3gram = 0
    max_rep_ngram = 0
    lengths = []
    
    for gen, refs in zip(generated_captions, references):
        gen_words = gen.split()
        ref_words_list = refs  # Already pre-split lists of words
        
        if not gen_words:
            continue
        
        # BLEU
        bleu1_scores.append(sentence_bleu(ref_words_list, gen_words, weights=(1,0,0,0), smoothing_function=smooth))
        bleu2_scores.append(sentence_bleu(ref_words_list, gen_words, weights=(0.5,0.5,0,0), smoothing_function=smooth))
        bleu3_scores.append(sentence_bleu(ref_words_list, gen_words, weights=(1/3,1/3,1/3,0), smoothing_function=smooth))
        bleu4_scores.append(sentence_bleu(ref_words_list, gen_words, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth))
        
        # ROUGE-L
        rouge_l_scores.append(rouge_l(gen_words, ref_words_list[0]))
        
        # Repetition
        gen_2grams = set()
        gen_3grams = set()
        has_rep_2 = False
        has_rep_3 = False
        max_rep = 0
        
        for n in [2, 3]:
            grams = [tuple(gen_words[i:i+n]) for i in range(len(gen_words)-n+1)]
            if len(grams) != len(set(grams)):
                if n == 2:
                    has_rep_2 = True
                else:
                    has_rep_3 = True
                # Count max repetition
                for g in set(grams):
                    count = grams.count(g)
                    if count > max_rep:
                        max_rep = count
        
        if has_rep_2:
            rep_2gram += 1
        if has_rep_3:
            rep_3gram += 1
        if max_rep > max_rep_ngram:
            max_rep_ngram = max_rep
        
        lengths.append(len(gen_words))
    
    n = len(generated_captions)
    return {
        "bleu1": float(np.mean(bleu1_scores)) if bleu1_scores else 0.0,
        "bleu2": float(np.mean(bleu2_scores)) if bleu2_scores else 0.0,
        "bleu3": float(np.mean(bleu3_scores)) if bleu3_scores else 0.0,
        "bleu4": float(np.mean(bleu4_scores)) if bleu4_scores else 0.0,
        "rouge_l": float(np.mean(rouge_l_scores)) if rouge_l_scores else 0.0,
        "repetition_rate_2gram": rep_2gram / n if n > 0 else 0.0,
        "repetition_rate_3gram": rep_3gram / n if n > 0 else 0.0,
        "max_repeated_ngram": max_rep_ngram,
        "avg_caption_length": float(np.mean(lengths)) if lengths else 0.0
    }


def evaluate_provider(provider_name, model, tokenizer, feats, val_data, max_length, device, 
                      decode="beam", beam_size=5, output_dir=None):
    """Run full evaluation for a provider"""
    print(f"\n{'='*60}")
    print(f"EVALUATING PROVIDER: {provider_name} ({decode}, beam={beam_size})")
    print(f"{'='*60}")
    
    val_images = val_data["image"].unique()
    image_names = json.loads((WEIGHTS_DIR / "image_names.json").read_text())
    feat_index = {name: i for i, name in enumerate(image_names)}
    
    # Build references
    data = pd.read_csv(DATASET_DIR / "captions.txt")
    data = text_preprocessing(data)
    
    refs = {}
    for img in val_images:
        rows = data[data["image"] == img]
        refs[img] = [
            [w for w in c.split() if w not in ("startseq", "endseq")]
            for c in rows["caption"].tolist()
        ]
    
    # Decode
    print(f"Decoding {len(val_images)} images...")
    t_start = time.time()
    
    if decode == "greedy":
        caps, confs = greedy_batch(model, feats, tokenizer, max_length)
    else:
        caps, confs = beam_batch(model, feats, tokenizer, max_length, beam_size)
    
    decode_time = time.time() - t_start
    
    # Format for metrics
    generated = []
    references = []
    for i, img in enumerate(val_images):
        generated.append(caps[i])
        references.append(refs[img])
    
    # Compute metrics
    print("Computing metrics...")
    token_metrics = compute_token_metrics(model, feats, val_data, tokenizer, max_length, device)
    caption_metrics = compute_caption_metrics(generated, references, tokenizer)
    
    # Latency
    avg_decode_time = decode_time / len(val_images) * 1000  # ms
    
    # Combine all metrics
    all_metrics = {
        "provider": provider_name,
        "decode": decode,
        "beam_size": beam_size if decode == "beam" else 1,
        "n_images": len(val_images),
        "decode_time_ms": avg_decode_time,
        **token_metrics,
        **caption_metrics
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {provider_name} ({decode})")
    print(f"{'='*60}")
    print(f"  BLEU-1: {caption_metrics['bleu1']:.4f}")
    print(f"  BLEU-2: {caption_metrics['bleu2']:.4f}")
    print(f"  BLEU-3: {caption_metrics['bleu3']:.4f}")
    print(f"  BLEU-4: {caption_metrics['bleu4']:.4f}")
    print(f"  ROUGE-L: {caption_metrics['rouge_l']:.4f}")
    print(f"  Repetition 2-gram: {caption_metrics['repetition_rate_2gram']:.2%}")
    print(f"  Repetition 3-gram: {caption_metrics['repetition_rate_3gram']:.2%}")
    print(f"  Max repeated n-gram: {caption_metrics['max_repeated_ngram']}")
    print(f"  Avg caption length: {caption_metrics['avg_caption_length']:.1f}")
    print(f"  Val loss (unsmoothed): {token_metrics.get('val_loss_unsmoothed', 'N/A')}")
    print(f"  Perplexity: {token_metrics.get('val_perplexity_unsmoothed', 'N/A')}")
    print(f"  Top-1 prob: {token_metrics.get('avg_top1_probability', 'N/A'):.4f}")
    print(f"  Top-5 mass: {token_metrics.get('avg_top5_probability_mass', 'N/A'):.4f}")
    print(f"  Top-1 acc: {token_metrics.get('top1_accuracy', 'N/A'):.4f}")
    print(f"  Top-5 acc: {token_metrics.get('top5_accuracy', 'N/A'):.4f}")
    print(f"  Avg decode time: {avg_decode_time:.1f} ms")
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)
        
        # Save sample captions
        samples = []
        for i in range(min(10, len(val_images))):
            samples.append({
                "image": val_images[i],
                "generated": generated[i],
                "references": [" ".join(r) for r in references[i]],
                "confidence": confs[i] if isinstance(confs, list) else 0
            })
        with open(output_dir / "sample_captions.json", "w") as f:
            json.dump(samples, f, indent=2)
        
        print(f"\n[OK] Results saved to {output_dir}/")
    
    return all_metrics


def greedy_batch(model, feats, tokenizer, max_length):
    """Lockstep greedy decode over all images"""
    global START_ID, END_ID, VOCAB_SIZE
    n = len(feats)
    seqs = np.zeros((n, max_length), dtype="int64")
    seqs[:, 0] = START_ID
    dones = np.zeros(n, dtype=bool)
    out = [[] for _ in range(n)]
    conf_sum = np.zeros(n, dtype="float64")
    conf_cnt = np.zeros(n, dtype="int64")
    
    for t in range(1, max_length):
        seqs_t = torch.tensor(seqs, device=feats.device, dtype=torch.long)
        # Aggressive clamp to prevent any out-of-bounds indices
        seqs_t = torch.clamp(seqs_t, 0, VOCAB_SIZE - 1)
        with torch.no_grad():
            logits = model(feats, seqs_t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        idx = probs.argmax(1)
        # Clamp to valid vocab range
        idx = np.clip(idx, 0, VOCAB_SIZE - 1)
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
        if dones.all():
            break
        seqs[:, t] = idx
    
    caps = [" ".join(ids_to_words(tokenizer, tok)) for tok in out]
    confs = [(conf_sum[i] / conf_cnt[i] if conf_cnt[i] else 0.0) for i in range(n)]
    return caps, confs


def beam_batch(model, feats, tokenizer, max_length, beam_width=5, length_penalty=0.6, repeat_ngram=2, repeat_penalty=0.4):
    """Lockstep beam search over all images"""
    global START_ID, END_ID, VOCAB_SIZE
    n = len(feats)
    alpha = length_penalty
    hyps = [[([START_ID], 0.0, False, [])] for _ in range(n)]
    
    for step in range(max_length):
        active = any(not all(h[2] for h in hyps[i]) for i in range(n))
        if not active:
            break
        print(f"  Beam step {step+1}: {sum(len(h) for h in hyps)} total hypotheses")
        # Debug: check hypothesis tokens
        for i in range(n):
            for h, hyp in enumerate(hyps[i]):
                invalid = [t for t in hyp[0] if t >= VOCAB_SIZE or t < 0]
                if invalid:
                    print(f"  ERROR: Invalid tokens in hyp {i},{h}: {invalid}")
        
        # Debug: check hypothesis counts per image
        for i in range(n):
            print(f"  Image {i}: {len(hyps[i])} hypotheses")
        
        rows = []
        seq_list = []
        feat_list = []
        for i in range(n):
            for h, hyp in enumerate(hyps[i]):
                rows.append((i, h))
                seq = np.zeros(max_length, dtype="int64")
                seq[:len(hyp[0])] = np.asarray(hyp[0][:max_length], dtype="int64")
                seq_list.append(seq)
                # feats[i] has shape (1, 1920), squeeze to (1920,)
                feat_list.append(feats[i].squeeze(0).cpu().numpy())
        
        seqs_t = torch.tensor(np.stack(seq_list), device=feats.device, dtype=torch.long)
        # Aggressive clamp to prevent any out-of-bounds indices
        seqs_t = torch.clamp(seqs_t, 0, VOCAB_SIZE - 1)
        feats_t = torch.tensor(np.stack(feat_list), device=feats.device, dtype=torch.float32)
        
        with torch.no_grad():
            logits = model(feats_t, seqs_t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        probs = np.clip(probs, 1e-12, 1.0)

        new_hyps = [[] for _ in range(n)]
        for r, (i, h) in enumerate(rows):
            tokens, logprob, finished, confs = hyps[i][h]
            if finished:
                new_hyps[i].append((tokens, logprob, True, confs))
                continue
            topk = np.argsort(probs[r])[::-1][:beam_width]
            # Debug: check topk values
            if topk.max() >= VOCAB_SIZE or topk.min() < 0:
                print(f"ERROR: topk out of bounds: min={topk.min()}, max={topk.max()}")
            topk = np.clip(topk, 0, VOCAB_SIZE - 1)
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
            denom = ((5 + max(len(tokens), 1)) / 6) ** 0.6
            return logprob / denom

        hyps = [sorted(im, key=norm_score, reverse=True)[:beam_width] for im in new_hyps]
    
    caps = []
    confs = []
    for i in range(n):
        best = min(hyps[i], key=lambda c: (0 if c[2] else 1, -norm_score(c)))
        tokens, _, _, c = best
        caps.append(" ".join(ids_to_words(tokenizer, tokens)))
        confs.append(float(np.mean(c)) if c else 0.0)
    return caps, confs


def _ngrams(tokens, n):
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


def main():
    ap = argparse.ArgumentParser(description="CaptionAI Evaluation Harness")
    ap.add_argument("--provider", required=True, choices=["torch_baseline", "torch_attention", "onnx", "legacy"])
    ap.add_argument("--checkpoint", help="Path to model checkpoint (.pt)")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--decode", default="beam", choices=["greedy", "beam"])
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--vocab_size", type=int, default=6000)
    ap.add_argument("--max_length", type=int, default=35)
    ap.add_argument("--limit", type=int, default=0, help="Limit images for quick test")
    args = ap.parse_args()
    
    print(f"PHASE 3: EVALUATION HARNESS")
    print(f"Provider: {args.provider}")
    print(f"Split: {args.split}")
    print(f"Decode: {args.decode} (beam={args.beam_size})")
    print(f"Output: {args.output_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    val_images = json.loads((WEIGHTS_DIR / "val_images.json").read_text())
    if args.limit:
        val_images = val_images[:args.limit]
    
    image_names = json.loads((WEIGHTS_DIR / "image_names.json").read_text())
    feat_index = {name: i for i, name in enumerate(image_names)}
    val_ids = [feat_index[img] for img in val_images]
    
    # Load features (global for baseline) - use OLD features that decoder was trained on
    features = np.load(WEIGHTS_DIR / "features.npy")
    feats = torch.tensor(features[val_ids], device=device, dtype=torch.float32)
    
    val_data = pd.read_csv(DATASET_DIR / "captions.txt")
    val_data = text_preprocessing(val_data)
    val_data = val_data[val_data["image"].isin(val_images)]
    
    # Load tokenizer
    tokenizer = load_tokenizer(WEIGHTS_DIR / "tokenizer.pkl")
    
    # Set special token IDs from tokenizer's word_index
    global START_ID, END_ID, UNK_ID, VOCAB_SIZE
    START_ID = tokenizer.word_index.get("startseq", 1)
    END_ID = tokenizer.word_index.get("endseq", 2)
    UNK_ID = tokenizer.word_index.get("<UNK>", 3)
    VOCAB_SIZE = len(tokenizer.word_index) + 1  # +1 for PAD at index 0
    print(f"Special tokens: PAD={PAD_ID}, START={START_ID}, END={END_ID}, UNK={UNK_ID}")
    print(f"Vocab size: {VOCAB_SIZE} (max index: {VOCAB_SIZE-1})")
    
    # Load model
    if args.provider in ["torch_baseline", "torch_attention"]:
        model = load_model_and_tokenizer(args.checkpoint, args.vocab_size, args.max_length, device)
    else:
        raise ValueError(f"Provider {args.provider} not yet implemented in evaluation harness")
    
    # Run evaluation
    metrics = evaluate_provider(
        args.provider, model, tokenizer, feats, val_data, 
        args.max_length, device, args.decode, args.beam_size, args.output_dir
    )
    
    # Save final metrics
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[OK] Evaluation complete. Results in {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())