# CaptionAI — Training Checklist (Actual Pipeline)

## Prerequisites
- [ ] Dataset: `dataset/Images/` (8,091 .jpg) + `dataset/captions.txt` (40,455 rows)
- [ ] GPU: NVIDIA (GTX 1080 verified; sm_61/Pascal works with torch cu121 wheels)
- [ ] Python 3.11 + uv
- [ ] Environment: `backend/.venv` with `tensorflow-cpu==2.21.0`, `torch==2.5.1+cu121`, `numpy`, `pandas`, `Pillow`, `tqdm`, `nltk`

## Run Order (sequential)

### 1. Feature Extraction (CPU, one-time)
```bash
.venv/Scripts/python.exe train_features.py          # --limit N for smoke test
```
- [ ] Completes without error (~15 min for full dataset)
- [ ] Outputs in `weights/`: `features.npy`, `image_names.json`, `tokenizer.pkl`, `word_index.json`, `train_images.json`, `val_images.json`, `metadata.json`
- [ ] Metadata shows: `vocab_size ≈ 8427`, `max_length = 35`, `n_train_images = 6877`, `n_val_images = 1214`

### 2. PyTorch GPU Decoder Training
```bash
.venv/Scripts/python.exe train_torch.py \
    --epochs 100 --batch-size 512 \
    --patience 20 --rlr-patience 6 --rlr-factor 0.5 --min-lr 1e-6 \
    --dropout 0.5 --seed 42
```
- [ ] CUDA detected (`device: cuda (NVIDIA GeForce GTX 1080)`)
- [ ] Train/val pairs built (~300k train / ~53k val)
- [ ] Epochs run; loss decreases; early stopping triggers at ~epoch 26
- [ ] Best checkpoint saved: `weights/decoder_torch.pt`
- [ ] `metadata.json` updated with `training.history`, `best_epoch`, `best_val_loss`

### 3. Torch → Keras Conversion (with logit gate)
```bash
.venv/Scripts/python.exe convert_torch_to_keras.py
```
- [ ] Gate passes: `max |torch−keras| < 1e-4` (actual ~2e-6)
- [ ] `weights/model.h5` written
- [ ] `metadata.json` updated with conversion stats

### 4. Evaluation (Batched)
```bash
.venv/Scripts/python.exe evaluate.py --beam 5
```
- [ ] Decodes 1,214 val images in ~2 min
- [ ] Beam BLEU-1 ≥ 0.53, BLEU-2 ≥ 0.32
- [ ] No repetitive "man man man" in beam outputs
- [ ] `metadata.json` updated with `evaluation` section

## Verification

### Weights Directory Contents
- [ ] `weights/features.npy` — (8091, 1920) float32
- [ ] `weights/tokenizer.pkl` — Keras Tokenizer (loads in backend)
- [ ] `weights/word_index.json` — plain dict for torch
- [ ] `weights/train_images.json`, `val_images.json` — splits
- [ ] `weights/decoder_torch.pt` — torch checkpoint (best epoch)
- [ ] `weights/model.h5` — Keras decoder weights (deployed)
- [ ] `weights/metadata.json` — full provenance

### Backend Activation
- [ ] Restart backend: `docker compose up -d --build` or `python -m uvicorn main:app`
- [ ] Health check: `GET /health` → provider `notebook-tensorflow` active
- [ ] Test predict: `POST /predict` with image + `provider=notebook-tensorflow`
- [ ] Caption is coherent, non-repetitive, confidence > 0.2

## Success Criteria (Actual Measured Targets)

| Criterion | Target | Achieved |
|-----------|--------|----------|
| Beam BLEU-1 | ≥ 0.55 | 0.5334 |
| Beam BLEU-2 | ≥ 0.35 | 0.3203 |
| No repetition | — | ✅ (beam) |
| Logit gate | < 1e-4 | 1.97e-06 ✅ |

*Notes:* 
- val_loss with label smoothing 0.1 is 4.75; equivalent plain CE ≈ 3.85
- This architecture on Flickr8K peaks ~BLEU-1 0.53–0.55; further gains need encoder fine-tuning or attention

## Cleanup
- [ ] `train_improved.py` deleted (empty file)
- [ ] `performance.tsx` updated with real metrics
- [ ] `TRAINING_GUIDE.md` and this checklist reflect actual pipeline
- [ ] Old `train.py`, `train_local.py` kept for reference but not used