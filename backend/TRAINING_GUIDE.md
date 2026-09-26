# CaptionAI — GPU Training Pipeline (DenseNet201 + LSTM)

## Overview
This guide documents the **actual** training pipeline used to produce the deployed model: PyTorch GPU decoder training → Keras weight conversion → BLEU/ROUGE evaluation. The DenseNet201 encoder is frozen; only the LSTM decoder is trained.

## Pipeline Stages

### Stage 0: Environment (Python 3.11 + uv)
```bash
cd Image-Captioning-Project/backend
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe tensorflow-cpu==2.21.0 numpy pandas Pillow tqdm nltk
# Install torch 2.5.1+cu121 from local wheel (GTX 1080 requires cu121 wheel)
uv pip install --python .venv/Scripts/python.exe <torch-wheel-path>
```

### Stage 1: Feature Extraction (CPU, one-time)
```bash
.venv/Scripts/python.exe train_features.py [--limit N]
```
- Precomputes DenseNet201 features (224×224, /255, pooling="avg" → 1920-dim) for all images
- Builds Keras Tokenizer on all captions (preprocessing: lowercase, strip non-[a-z ], drop len-1 words, wrap startseq/endseq)
- 85/15 image-level split
- Outputs: `weights/features.npy`, `tokenizer.pkl`, `word_index.json`, `train/val_images.json`, `metadata.json`

### Stage 2: PyTorch GPU Decoder Training
```bash
.venv/Scripts/python.exe train_torch.py \
    --epochs 100 --batch-size 512 \
    --patience 20 --rlr-patience 6 --rlr-factor 0.5 --min-lr 1e-6 \
    --dropout 0.5 --seed 42
```
- Exact notebook architecture (cell 20) in PyTorch
- Adam lr=1e-3, clipnorm=5.0, label_smoothing=0.1, CrossEntropyLoss
- Precomputes all (prefix→next) pairs once; batch=512 pairs/step
- EarlyStopping patience=20, ReduceLROnPlateau patience=6 factor=0.5
- Saves best `weights/decoder_torch.pt` + updates `metadata.json` with losses

### Stage 3: Torch → Keras Weight Conversion
```bash
.venv/Scripts/python.exe convert_torch_to_keras.py
```
- Maps state_dict to Keras `build_decoder()` (Dense transpose, LSTM gates [i,f,c,o], Embedding copy)
- **Gate**: torch vs Keras softmax on 50 random (feature, seq) pairs; max |diff| < 1e-4
- Saves `weights/model.h5` + updates `metadata.json` with conversion stats

### Stage 4: Evaluation
```bash
.venv/Scripts/python.exe evaluate.py [--beam 5] [--limit N]
```
- Batched greedy + beam search (lockstep across images, ~72 predict calls total)
- Reports BLEU-1..4 + ROUGE-L on val split (1214 images)
- Updates `metadata.json` with metrics

## Actual Results (GTX 1080, 8GB)

| Metric | Value |
|--------|-------|
| Feature extraction | 8,091 images, 14.5 min (batch 32) |
| GPU training (best) | 26 epochs, 13.5 min, val_loss 4.7517 (with label smoothing 0.1) |
| Conversion gate | max |torch−keras| = 1.97e-06 |
| **Beam BLEU-1** | **0.5334** |
| **Beam BLEU-2** | **0.3203** |
| **Beam BLEU-3** | **0.1890** |
| **Beam BLEU-4** | **0.1218** |
| **Beam ROUGE-L** | **0.2216** |
| Greedy BLEU-1 | 0.4372 (repetitive) |

## Architecture Details (unchanged from notebook)
- Encoder: DenseNet201 `include_top=False, pooling="avg"` → 1920-dim
- Decoder: `Dense(256,relu) → Reshape(1,256) → concat(Embedding(vocab,256)) → LSTM(256) → Dropout(0.5) → Add(residual) → Dense(128,relu) → Dropout(0.5) → Dense(vocab,softmax)`
- Vocab: 8,427 | Max length: 35 tokens (incl. start/end) | Split: 6,877 / 1,214 images

## Decoding in Production
Backend uses `inference/decoding.py`:
- **Beam search** (width 5, GNMT length penalty α=0.6, 2-gram repeat penalty 0.4)
- Greedy fallback
- Identical scoring to evaluation harness

## Quick Start (reproduce best model)
```bash
cd backend
.venv/Scripts/python.exe train_features.py          # ~15 min
.venv/Scripts/python.exe train_torch.py --patience 20 --rlr-patience 6 --rlr-factor 0.5  # ~15 min
.venv/Scripts/python.exe convert_torch_to_keras.py  # ~10 sec
.venv/Scripts/python.exe evaluate.py                # ~2 min
# model.h5 + tokenizer.pkl + metadata.json now in weights/
```

## Files in This Repo
- `train_features.py` — batched DenseNet201 feature extraction + tokenizer
- `train_torch.py` — PyTorch GPU decoder training
- `convert_torch_to_keras.py` — weight conversion with logit gate
- `evaluate.py` — batched BLEU/ROUGE evaluation
- `inference/decoding.py` — greedy + beam search (production + eval)
- `inference/manager.py` — NotebookProvider (loads model.h5, runs beam search)
- `model/architecture.py` — `build_decoder()` (Keras) / `build_encoder()`
- `weights/` — generated artefacts (gitignored)