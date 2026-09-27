"""
STEP A — Build the BPE-6k subword tokenizer (Tier-2 v2 pipeline).

Fits a ByteLevel-BPE tokenizer with vocab_size=6000 on the TRAIN SPLIT ONLY
(no validation leakage), encodes every caption, and validates that
encode -> decode round-trips every caption exactly.

Why BPE: rare words ("kayakers" seen twice) become learnable subword
sequences ("kayak"+"ers") instead of unlearnable hapax targets — removes
the memorisation shortcut behind the epoch-26 overfitting wall while
keeping every caption perfectly representable (no <unk>, ever).

Outputs (weights/bpe/):
    tokenizer.json          full serialised tokenizer
    encoded_captions.json   {image_name: [[START, ...ids..., END], ...]}
    bpe_stats.json          fit/compression/round-trip statistics

Usage: python build_bpe.py [--vocab 6000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"
WEIGHTS_DIR = PROJECT_DIR / "weights"
OUT_DIR = WEIGHTS_DIR / "bpe"

sys.path.insert(0, str(BACKEND_DIR))
from train_torch import text_preprocessing  # noqa: E402  (identical cleanup)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=6000)
    args = ap.parse_args()

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_images = set(json.loads((WEIGHTS_DIR / "train_images.json").read_text()))

    data = pd.read_csv(DATASET_DIR / "captions.txt")
    data = text_preprocessing(data)  # lowercase, clean, startseq/endseq wrapped

    # strip the wrapper tokens — BPE adds its own specials
    train_caps = [
        row["caption"].replace("startseq ", "").replace(" endseq", "")
        for _, row in data.iterrows() if row["image"] in train_images
    ]
    all_caps_by_image: dict[str, list[str]] = {}
    for _, row in data.iterrows():
        all_caps_by_image.setdefault(row["image"], []).append(
            row["caption"].replace("startseq ", "").replace(" endseq", ""))

    print(f">> fitting BPE(vocab={args.vocab}) on {len(train_caps):,} train captions …",
          flush=True)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab,
        show_progress=False,
        special_tokens=["[PAD]", "[START]", "[END]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(train_caps, trainer)
    print(f">> final vocab (incl. 3 specials): {tok.get_vocab_size(with_added_tokens=True)}",
          flush=True)

    pad_id, start_id, end_id = (
        tok.token_to_id("[PAD]"), tok.token_to_id("[START]"), tok.token_to_id("[END]"))
    assert (pad_id, start_id, end_id) == (0, 1, 2), "special IDs must be 0/1/2"

    # ---- encode all captions + round-trip check --------------------------
    encoded: dict[str, list[list[int]]] = {}
    n_seqs = n_frag_tokens = n_word_tokens = 0
    roundtrip_fail = 0
    for img, caps in all_caps_by_image.items():
        seqs = []
        for c in caps:
            ids = tok.encode(c).ids
            if tok.decode(ids).strip() != c.strip():
                roundtrip_fail += 1
            seqs.append([start_id] + list(ids) + [end_id])
            n_frag_tokens += len(ids)
            n_word_tokens += len(c.split())
            n_seqs += 1
        encoded[img] = seqs

    max_len = max(len(s) for seqs in encoded.values() for s in seqs)
    stats = {
        "vocab_size_with_specials": tok.get_vocab_size(with_added_tokens=True),
        "train_captions_fit": len(train_caps),
        "total_sequences_encoded": n_seqs,
        "max_bpe_length": int(max_len),
        "avg_tokens_per_caption": round(n_frag_tokens / n_seqs, 2),
        "avg_words_per_caption": round(n_word_tokens / n_seqs, 2),
        "compression_vs_words": round(n_word_tokens / n_frag_tokens, 3),
        "roundtrip_failures": roundtrip_fail,
        "special_ids": {"PAD": pad_id, "START": start_id, "END": end_id},
    }
    (OUT_DIR / "tokenizer.json").write_text(tok.to_str())
    (OUT_DIR / "encoded_captions.json").write_text(json.dumps(encoded))
    (OUT_DIR / "bpe_stats.json").write_text(json.dumps(stats, indent=2))

    print("\n" + "=" * 60)
    print("BPE BUILD SUMMARY")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:28s} = {v}")
    demo = ["a black dog catches a red frisbee in midair",
            "two kayakers paddle past sunbathers at sunset"]
    for d in demo:
        ids = tok.encode(d).ids
        print(f"\n  '{d}'")
        print(f"    -> {ids} ({len(d.split())} words -> {len(ids)} tokens)")
    ok = stats["roundtrip_failures"] == 0
    print(f"\n>> ROUND-TRIP {'PASSED' if ok else 'FAILED'} "
          f"({stats['roundtrip_failures']} failures / {n_seqs} captions)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
