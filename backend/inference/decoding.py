"""
CaptionAI — decoding strategies (greedy + beam search).

These functions are framework-agnostic: they take a `predict_fn` callable
```
    predict_fn(feature, sequences) -> probabilities
```
where `feature` is shape (1, 1920) and `sequences` is shape (B, max_length)
of padded token ids; it returns a (B, vocab_size) probability matrix.

Both the Keras backend (`inference/manager.py`) and the evaluation harness
(`evaluate.py`) use them, so metrics are measured on the exact code path
that runs in production.

Beam search implements GNMT-style length normalisation plus a mild
no-repeat n-gram penalty to suppress the "man man man" degeneration that
plagues greedy argmax on this architecture.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np

PredFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _pad(tokens: Sequence[int], max_length: int) -> np.ndarray:
    """Post-pad a token list to max_length (value 0), matching pad_sequences."""
    seq = np.zeros(max_length, dtype="int64")
    take = min(len(tokens), max_length)
    seq[:take] = np.asarray(tokens[:take], dtype="int64")
    return seq


def _idx_to_word(tokenizer, idx: int) -> Optional[str]:
    for word, index in tokenizer.word_index.items():
        if index == idx:
            return word
    return None


def greedy_decode(
    predict_fn: PredFn,
    feature: np.ndarray,
    tokenizer,
    max_length: int,
) -> tuple[str, float]:
    """Greedy argmax decoding — identical to the notebook's predict_caption."""
    start_id = tokenizer.word_index.get("startseq", 1)
    end_id = tokenizer.word_index.get("endseq", 2)
    tokens = [start_id]
    confidences: list[float] = []
    for _ in range(max_length):
        seq = _pad(tokens, max_length)
        probs = predict_fn(feature, seq[None, :])[0]  # (vocab,)
        idx = int(np.argmax(probs))
        confidences.append(float(probs[idx]))
        if idx == end_id:
            break
        word = _idx_to_word(tokenizer, idx)
        if word is None:
            break
        tokens.append(idx)
    caption = _tokens_to_caption(tokenizer, tokens)
    confidence = float(np.mean(confidences)) if confidences else 0.0
    return caption, confidence


def beam_search_decode(
    predict_fn: PredFn,
    feature: np.ndarray,
    tokenizer,
    max_length: int,
    beam_width: int = 5,
    length_penalty: float = 0.6,
    repeat_ngram: int = 2,
    repeat_penalty: float = 0.4,
) -> tuple[str, float]:
    """
    Beam search decoding with GNMT-style length normalisation and a
    no-repeat n-gram penalty.

    score(hyp) = sum(log p_tok) / ((5 + len) / 6) ** length_penalty
    """
    start_id = tokenizer.word_index.get("startseq", 1)
    end_id = tokenizer.word_index.get("endseq", 2)
    vocab_size = len(tokenizer.word_index) + 1

    # hypotheses: (tokens, logprob, finished)
    hyps: list[tuple[list[int], float, bool]] = [([start_id], 0.0, False)]
    alpha = length_penalty

    for _ in range(max_length):
        if all(h[2] for h in hyps):
            break
        candidates: list[tuple[list[int], float, bool]] = []
        for tokens, logprob, finished in hyps:
            if finished:
                candidates.append((tokens, logprob, True))
                continue
            seq = _pad(tokens, max_length)
            probs = predict_fn(feature, seq[None, :])[0]
            probs = np.clip(probs, 1e-12, 1.0)
            topk = np.argsort(probs)[::-1][:beam_width]
            for idx in topk:
                p = float(probs[idx])
                lp = np.log(p)
                # n-gram repetition penalty
                if repeat_ngram > 1 and len(tokens) >= repeat_ngram:
                    tail = tuple(tokens[-(repeat_ngram - 1):]) + (int(idx),)
                    if tail in _ngrams(tokens, repeat_ngram):
                        lp += np.log(repeat_penalty)
                new_tokens = tokens + [int(idx)]
                finished = int(idx) == end_id
                candidates.append((new_tokens, logprob + lp, finished))

        # prune to beam_width by normalised score
        def norm_score(c):
            tokens, logprob, _ = c
            length = max(len(tokens), 1)
            denom = ((5 + length) / 6) ** alpha
            return logprob / denom

        candidates.sort(key=norm_score, reverse=True)
        hyps = candidates[:beam_width]

    # pick best: prefer finished, then by score
    def norm_score(c):
        tokens, logprob, _ = c
        length = max(len(tokens), 1)
        denom = ((5 + length) / 6) ** alpha
        return logprob / denom

    best = min(hyps, key=lambda c: (0 if c[2] else 1, -norm_score(c)))
    tokens, _, _ = best

    # confidence: mean softmax probability along the chosen path
    confidences: list[float] = []
    cur = [start_id]
    for nxt in tokens[1:]:
        seq = _pad(cur, max_length)
        probs = predict_fn(feature, seq[None, :])[0]
        confidences.append(float(probs[nxt]))
        cur.append(nxt)
    confidence = float(np.mean(confidences)) if confidences else 0.0

    caption = _tokens_to_caption(tokenizer, tokens)
    return caption, confidence


def _ngrams(tokens: Sequence[int], n: int) -> set[tuple[int, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _tokens_to_caption(tokenizer, tokens: Sequence[int]) -> str:
    words: list[str] = []
    for idx in tokens:
        word = _idx_to_word(tokenizer, idx)
        if word is None:
            break
        if word in ("startseq", "endseq"):
            continue
        words.append(word)
    return " ".join(words)