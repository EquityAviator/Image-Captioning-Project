# CaptionAI: Local Image Captioning with CLIP + GRPO

**CaptionAI** is a production-ready, locally-running image captioning system that achieves state-of-the-art results on Flickr8K using only locally-trained models. The project demonstrates that with the right feature backbone (CLIP ViT-B/16) and reinforcement learning (GRPO), a modest 8.4M parameter LSTM decoder can outperform much larger models on in-domain data — while running entirely on consumer hardware (CPU: ~590 ms/caption, no GPU required).

The system includes a **hybrid routing** mechanism: a CLIP-based zero-shot classifier detects out-of-domain images (cartoons, screenshots, food photos) and automatically routes them to a local BLIP fallback, while in-domain photos are handled by the specialist CLIP+GRPO model. The entire pipeline runs locally with no external API dependencies — images never leave your machine.

### Key Highlights
- 🎯 **State-of-the-art on Flickr8K**: BLEU-1 0.6559, BLEU-4 0.1727, ROUGE-L 0.2782
- ⚡ **Fast inference**: ~590 ms/caption on CPU (no GPU required)
- 🛡️ **Hybrid OOD routing**: CLIP zero-shot + BLIP fallback (100% OOD recall)
- 🎯 **Calibrated confidence**: Temperature scaling T=1.3, ECE 0.0862
- 🏠 **Fully local**: No external APIs, images never leave your machine
- ⚙️ **uv-first setup**: 3–10× faster installs, auto CPU/GPU detection
- 📚 **Full reproducibility**: Every experiment tracked with eval artifacts

---

## Quick Start

### Prerequisites
- **Git** and **Node.js 20+**
- **uv** (recommended) or **Python 3.11+**

### Installation (Recommended: uv)

```bash
# 1. Install uv (one-time)
# Windows:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and run
git clone https://github.com/EquityAviator/Image-Captioning-Project.git
cd Image-Captioning-Project

# Run the setup script (handles everything: deps, model downloads, prompts)
uv run --python 3.11 backend/setup.py
```

### Installation (Fallback: plain Python)

```bash
git clone https://github.com/EquityAviator/Image-Captioning-Project.git
cd Image-Captioning-Project

# Ensure Python 3.11 and Node.js 20+ are installed
python backend/setup.py
```

### What the Setup Script Does
1. **Engine choice**: CPU (default, 10s timeout) or GPU (CUDA 12.1)
2. **Installs dependencies**: Python packages + `npm ci` for frontend
3. **CLIP ViT-B/16** (352 MB): auto-downloads from Hugging Face (default ON)
4. **BLIP fallback** (945 MB): optional prompt — **default SKIP after 10s**
5. **Smoke test** → opens dashboard at `http://localhost:3001`

**Headless/CI mode**: `uv run --python 3.11 backend/setup.py --yes` — zero prompts, all defaults.

---

## Dataset: Flickr8K

| Property | Value |
|----------|-------|
| **Images** | 8,091 photographs (Flickr) |
| **Captions** | 5 human-written per image (40,455 total) |
| **Split** | 6,877 train / 1,214 validation (fixed 85/15 split) |
| **Preprocessing** | Lowercase, strip non-alphabetic, BPE-6k tokenizer |
| **Max length** | 38 tokens |

The same fixed validation split is used across all experiments for fair comparison. All metrics reported are on the **full 1,214-image validation split** with beam-5 (α=1.2), NLTK smoothing.

---

## Model Performance

### Champion Model: CLIP ViT-B/16 + GRPO (Production)

| Metric | Score |
|--------|-------|
| **BLEU-1** | **0.6559** |
| **BLEU-2** | 0.4344 |
| **BLEU-3** | 0.2698 |
| **BLEU-4** | **0.1727** |
| **ROUGE-L** | **0.2782** |
| **CIDEr-D** | 0.5243 |
| **Word Precision** | 70.7% |
| **CHAIR (hallucination)** | 47.3% images / 29.8% nouns |
| **Inference Time** | ~590 ms (CPU, beam-5) |
| **Model Size** | 8.42M params (32 MB) |

### Generational Progression

| Generation | Architecture | BLEU-1 | BLEU-4 | ROUGE-L | CIDEr-D | Notes |
|------------|--------------|--------|--------|---------|---------|-------|
| Gen 1 | DenseNet201 + LSTM (TF) | 0.5334 | 0.1218 | 0.2216 | — | Baseline |
| Gen 2 | DenseNet + Attention (CE) | 0.5649 | 0.1663 | 0.2548 | 0.5397 | +37% BLEU-4 |
| Gen 2-RL | + GRPO | 0.5971 | 0.1612 | 0.2566 | — | RL gain |
| Gen 3 | CLIP ViT-B/16 + CE | 0.6092 | 0.1755 | 0.2747 | 0.6207 | Feature swap |
| **Gen 3-RL ★** | **CLIP + GRPO** | **0.6559** | **0.1727** | **0.2782** | **0.5243** | **Production** |

### Key Findings
- **Feature swap > architecture**: CLIP features alone (+GRPO) beat all prior DenseNet models with same decoder
- **GRPO improves precision**: Word precision 63.9% → 70.7%, but shortens captions (CIDEr bias)
- **GRPO reduces hallucination**: CHAIR 61.9% (CLIP-CE) → 47.3% (champion)
- **Capacity isn't the bottleneck**: 23M Transformer overfit badly vs 8.4M LSTM

---

## Fallback & Routing

| Provider | In-Domain BLEU-1 | OOD Recall | Latency |
|----------|------------------|------------|---------|
| **CLIP+GRPO (champion)** | **0.574** | — | ~590 ms |
| BLIP (fallback) | 0.503 | **100%** (10/10 synthetic) | ~2,500 ms |
| Notebook TF (Gen-1) | 0.427 | — | ~7,000 ms |

**Router**: CLIP zero-shot over 6 prompts (photo, screenshot, food, cartoon, painting, document). Margin rule ≥0.02 cosine keeps 90% of real photos with specialist; 100% OOD recall on 10 synthetic images.

---

## Calibration & Confidence

| Metric | Champion |
|--------|----------|
| **ECE @ T=1 (raw)** | 0.2542 |
| **ECE @ T=1.3 (served)** | **0.0862** |
| Optimal temperature (T*) | 1.3 |
| Top-1 accuracy (teacher-forced) | 31.6% |

Temperature scaling (T=1.3) applied **post-decode only** — beam search unchanged, metrics bit-exact.

---

## Ablation & Rejected Techniques (Documented)

| Technique | Result | Reason |
|-----------|--------|--------|
| DenseNet fine-tune (P2) | CIDEr 0.418 → 0.399 | 8k images too few |
| Focal loss (P3b) | BLEU-1 0.0038 (gibberish) | Down-weights structural tokens |
| Mixed reward (CIDEr+ROUGE) | BLEU-1 0.6559 → 0.6514 | ROUGE also precision-biased |
| Length-scaled reward | Length fixed but BLEU-4 -1.3 pts | Trade-off moved, not resolved |
| Transformer decoder (Gen 4) | BLEU-1 -10.1 pts | Overfit (23M params vs 8k images) |
| BLIP distillation | BLEU-1 0.609 → 0.489 | Style mismatch (val loss better, gen worse) |
| Diverse Beam Search | BLEU-1 0.570 → 0.531 | Flickr8K refs too similar |
| CHAIR-aware RL (rl4/rl5) | CHAIR drops but CIDEr collapses | Degeneration via noun avoidance |

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                         Browser (client)                                │
│                                                                        │
│   Next.js 16 + React 19 + TypeScript + Tailwind 4 + shadcn/ui          │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  Landing page → Hero / Features / Architecture / How-it-works / │  │
│   │                 Model / Performance / Demo / Footer              │  │
│   │  Dashboard  → Sidebar + Navbar + Upload + Prediction + History   │  │
│   │  Model Info → Pipeline diagram + hyperparameters                 │  │
│   │  Settings   → Theme / animations / API URL / toggles             │  │
│   └────────────────────────┬────────────────────────────────────────┘  │
└────────────────────────────┼────────────────────────────────────────────┘
                             │ HTTPS (relative paths + XTransformPort)
                             ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       FastAPI backend  (:8010)                         │
│                                                                        │
│   backend/main.py                                                      │
│   ├── /health                                                          │
│   ├── /model-info                                                      │
│   ├── /predict            (single image)                               │
│   └── /predict-batch      (up to 8 images)                             │
│                                                                        │
│   backend/inference/manager.py                                         │
│   ┌──────────────────────────────────────────────────────────────┐    │
│   │  ModelManager (singleton)                                    │    │
│   │   ├── AttentionProvider  (CLIP + GRPO, primary)              │    │
│   │   ├── BLIPProvider  (fallback for OOD)                       │    │
│   │   └── TFProvider  (Gen-1 DenseNet+LSTM, comparison only)     │    │
│   └──────────────────────────────────────────────────────────────┘    │
│                                                                        │
│   backend/inference/attention_provider.py  ← CLIP+GRPO serving        │
│   backend/inference/manager.py           ← ModelManager + routing    │
│   backend/inference/decoding.py          ← Beam search + guards      │
└────────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
                  weights/ (gitignored)
                  ├── v2_attention_clip_rl/best.pt   (champion, 32 MB)
                  ├── v2_attention_clip/best.pt       (fallback #1)
                  ├── v2_attention_rl/best.pt          (fallback #2)
                  ├── v2_attention/best.pt             (fallback #3)
                  ├── bpe/tokenizer.json               (BPE-6k, 3 MB)
                  ├── image_names.json                  (index)
                  ├── image_id_to_index.json            (index)
                  ├── clip_features_spatial.npy        # 2.27 GB (training cache)
                  ├── features_spatial.npy             # 1.45 GB (training cache)
                  └── blip/                            # BLIP fallback (auto-download)
```

### Inference Pipeline

```
image bytes
   │
   ▼
[Pillow decode + resize to 224×224]
   │
   ▼
[CLIP ViT-B/16 encoder (frozen, pretrained="openai")]
   │  outputs (1, 196, 768) patch tokens
   ▼
[CLIP zero-shot router: cosine vs 6 prompts]
   │  photo? → AttentionProvider
   │  other  → BLIPProvider
   ▼
[AttentionProvider: Bahdanau attention over 196 patches]
   │  LSTM(512) + GRPO-tuned weights
   ▼
[Incremental beam-5 search (GNMT α=1.2, soft 2-gram penalty)]
   │  min-length guard (8 tokens)
   ▼
[Caption + tempered confidence (T=1.3, display-only)]
```

---

## Frontend Features

### Landing Page
- **Hero section** with animated gradient blobs and CTA buttons
- **Feature cards** (6) with hover animations
- **Architecture pipeline** diagram with animated arrows
- **How-it-works** 4-step flow
- **Model section** with interactive metrics
- **Performance** animated count-up statistics
- **Demo CTA** with animated gradient buttons

### Dashboard
- **Sidebar navigation**: Dashboard, History, Model Info, Settings
- **Upload zone**: Drag-and-drop with preview, file validation
- **Prediction panel**: Real-time stage animation (Upload → Extract → Encode → Decode → Caption)
- **Result card**: Caption, confidence bar, inference time, provider badge
- **Actions**: Copy, Download, Share, Speak (TTS), Regenerate
- **History panel**: Recent predictions (localStorage, capped at 50), reuse/delete

### Model Info Page
- **Champion card**: Live metrics from `/model-info`
- **Generation history**: 5 generations with metrics
- **Techniques ledger**: 20 expandable cards (10 wins, 10 failures) with metrics tables
- **Progress comparison table**: All generations × metrics
- **Live pipeline diagram**: 7-step visual flow
- **API documentation**: All endpoints with request/response schemas
- **Backend health**: Routing, latency, confidence, operational quirks
- **Live hyperparameters**: Read directly from running backend

### Settings
- **Theme**: Dark / Light / System
- **Animations**: Toggle all Framer Motion animations
- **History**: Enable/disable localStorage persistence
- **Typing effect**: Caption typewriter reveal
- **Speech synthesis**: Browser TTS for captions
- **Auto-copy**: Copy caption to clipboard on generation
- **Provider selector**: Auto (recommended) / Attention / Notebook TF / HF BLIP
- **Custom API URL**: Override backend endpoint

### Design System
- **Tailwind CSS 4** with CSS variables for theming
- **shadcn/ui** component library (Radix UI primitives)
- **Framer Motion** for all animations
- **Dark-first palette**: indigo/purple/blue + emerald accent
- **Accessibility**: ARIA labels, keyboard navigation, focus rings, screen-reader text

---

## Backend API

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness + active provider + weights status |
| `GET` | `/model-info` | Full architecture metadata + val metrics |
| `GET` | `/providers` | List available providers + current |
| `POST` | `/providers/{name}` | Hot-swap active provider |
| `POST` | `/predict` | Single image → caption + metadata |
| `POST` | `/predict-batch` | Up to 8 images → batch results |

### Request/Response

**POST /predict**
```bash
curl -X POST http://localhost:8010/predict \
  -F "file=@image.jpg"
```
```json
{
  "caption": "a young boy is playing in the grass",
  "inference_time": "416.9ms",
  "confidence": 0.322,
  "provider": "attention",
  "success": true
}
```

**Response fields:**
- `caption`: Generated caption string
- `inference_time`: Wall-clock time (ms)
- `confidence`: Tempered confidence (T=1.3)
- `provider`: `attention` | `attention+blip` | `huggingface-blip`
- `semantic_score`: null (unless `ENABLE_BLIP_SCORE=1`)
- `semantic_pmi`: null

---

## Training Pipeline

### Dataset: Flickr8K
| Property | Value |
|----------|-------|
| Images | 8,091 photographs (Flickr) |
| Captions | 5 human-written per image (40,455 total) |
| Split | 6,877 train / 1,214 validation (85/15 fixed) |
| Tokenizer | BPE-6k (6,000 subwords) |
| Max length | 38 tokens |

### Training Scripts
```bash
# Gen 2: DenseNet + Attention (CE)
python backend/train_attention_v2.py

# Gen 2-RL: GRPO on DenseNet
python backend/train_attention_v2_rl.py

# Gen 3: CLIP-CE (feature swap)
python backend/train_attention_v2_clip.py

# Gen 3-RL ★: CLIP + GRPO (champion)
python backend/train_attention_v2_clip_rl.py

# Transformer decoder (Gen 4)
python backend/train_attention_v2_tf.py

# BLIP distillation
python backend/train_attention_v2_clip_distill.py

# CHAIR-aware RL (rl4/rl5)
python backend/train_attention_v2_clip_rl4.py
python backend/train_attention_v2_clip_rl5.py
```

### Key Training Configs
| Config | Value |
|--------|-------|
| Batch size | 64 |
| Learning rate | 3e-4 (CE), 1e-5 (RL) |
| Optimizer | AdamW |
| Weight decay | 0.01 (CE), 0 (RL) |
| Patience | 12 epochs |
| GRPO group size | 5 |
| Images/step (RL) | 8 |
| Max length | 24 tokens |
| Seed | 42 |

### Evaluation Protocol
- **Split**: Fixed 1,214 validation images
- **Decoding**: Beam-5, GNMT α=1.2, soft 2-gram penalty
- **Metrics**: BLEU-1/2/3/4, ROUGE-L, CIDEr-D (pycocoevalcap), CHAIR-lite, ECE
- **Seed**: 42 (all runs)

---

## Evaluation Methodology

### Metrics Computed
| Metric | Tool | Purpose |
|--------|------|---------|
| BLEU-1/2/3/4 | NLTK (method-1 smoothing) | N-gram precision |
| ROUGE-L | rouge-score | Longest common subsequence |
| CIDEr-D | pycocoevalcap | TF-IDF weighted n-gram consensus |
| CHAIR-lite | Custom (NLTK POS) | Hallucination rate |
| ECE | Custom (13 bins) | Calibration error |
| Word precision | Custom | % predicted words in references |

### CHAIR-lite Methodology
- POS-tag nouns in generated captions (NLTK)
- Noun is "unsupported" if absent from union of nouns in 5 references
- **CHAIR-img**: % images with ≥1 unsupported noun
- **Hallucinated noun rate**: % all generated nouns unsupported
- Note: Conservative — synonyms ("dog"/"puppy") count as hallucinated

### Calibration
- **ECE**: Expected Calibration Error (13 bins, teacher-forced tokens)
- **T\***: Optimal temperature via NLL grid search (0.5–2.0)
- **Served**: T=1.3 applied post-decode (display only, beam unchanged)
- **Verification**: Bit-exact metric reproduction after tempering

---

## Ablation & Rejected Techniques (Documented)

| Technique | Result | Reason |
|-----------|--------|--------|
| DenseNet fine-tune (P2) | CIDEr 0.418 → 0.399 | 8k images too few |
| Focal loss (P3b) | BLEU-1 0.0038 (gibberish) | Down-weights structural tokens |
| Mixed reward (CIDEr+ROUGE) | BLEU-1 0.6559 → 0.6514 | ROUGE also precision-biased |
| Length-scaled reward | Length fixed but BLEU-4 -1.3 pts | Trade-off moved, not resolved |
| Transformer decoder (Gen 4) | BLEU-1 -10.1 pts | Overfit (23M params vs 8k images) |
| BLIP distillation | BLEU-1 0.609 → 0.489 | Style mismatch (val loss better, gen worse) |
| Diverse Beam Search | BLEU-1 0.570 → 0.531 | Flickr8K refs too similar |
| CHAIR-aware RL (rl4/rl5) | CHAIR drops but CIDEr collapses | Degeneration via noun avoidance |

---

## Project Structure

```
Image-Captioning-Project/
├── backend/                 # FastAPI server
│   ├── inference/           # Model providers, routing, decoding
│   ├── setup.py             # Bootstrap script (uv/uvx or python)
│   ├── requirements.txt     # Common deps (no torch/tf)
│   ├── requirements-cpu.txt # torch CPU
│   ├── requirements-gpu.txt # torch CUDA 12.1
│   └── requirements-legacy.txt  # tensorflow-cpu (optional)
├── src/                     # Next.js 16 frontend
│   ├── components/dashboard/   # Dashboard, ModelInfo, Settings
│   └── app/                    # Next.js app router
├── experiments/             # All training runs (best.pt + eval JSONs)
│   ├── v2_attention_clip_rl/    # ★ Champion
│   ├── v2_attention_clip/       # CLIP-CE (fallback #1)
│   ├── v2_attention_rl/         # DenseNet-RL (fallback #2)
│   └── ... (rejected runs)
├── weights/
│   ├── bpe/                    # BPE tokenizer (6k vocab)
│   ├── clip_features_spatial.npy   # 2.27 GB (training cache)
│   ├── features_spatial.npy        # 1.45 GB (training cache)
│   └── blip/                   # BLIP fallback (auto-downloads)
├── experiments/               # Training checkpoints + eval JSONs
├── dataset/
│   ├── Images/               # 8,091 Flickr8K images (gitignored)
│   └── captions.txt          # All captions
├── docs/
│   ├── SETUP_REQUIREMENTS.md    # Full setup spec
│   ├── SETUP_REQUIREMENTS.md    # Client summary
│   └── CaptionAI_Research_Report_v2.pdf
└── docs/CaptionAI_Research_Report_v2.pdf   # Full technical report
```

---

## Commands Reference

```bash
# Full setup (interactive)
uv run --python 3.11 backend/setup.py

# Headless (CI/defaults)
uv run --python 3.11 backend/setup.py --yes

# Force CPU/GPU
uv run --python 3.11 backend/setup.py --cpu
uv run --python 3.11 backend/setup.py --gpu

# Skip BLIP explicitly
uv run --python 3.11 backend/setup.py --blip off

# Force CLIP download
uv run --python 3.11 backend/setup.py --no-clip

# Air-gapped (pre-copied HF cache)
CAPTIONAI_HF_OFFLINE=1 uv run --python 3.11 backend/setup.py

# Development servers
uv run --python 3.11 -m uvicorn backend.main:app --port 8010 --reload  # backend
npm run dev                                                  # frontend (port 3001)
```

---

## Environment Variables

| Variable | Values | Default | Purpose |
|----------|--------|---------|---------|
| `CAPTIONAI_BLIP` | `download` \| `local:PATH` \| `off` | — | Override BLIP prompt |
| `CAPTIONAI_CLIP` | `skip` | — | Skip CLIP prefetch |
| `CAPTIONAI_HF_OFFLINE` | `1` | — | Force offline mode |
| `CAPTIONAI_BLIP` | `download` \| `local` \| `off` | — | BLIP mode |
| `ATT_CKPT` | path | auto | Override checkpoint |
| `ATT_DEVICE` | `cpu` \| `cuda` | `cpu` | Inference device |

---

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Run linting: `bun run lint` (frontend), `ruff check` (backend)
4. Run tests: `pytest` (backend), `bun run test` (frontend)
5. Commit with conventional commits: `feat:`, `fix:`, `docs:`, etc.
5. Open PR against `main`

### Code Style
- **Frontend**: TypeScript strict, ESLint + Prettier, shadcn/ui patterns
- **Backend**: Python 3.11+, type hints, ruff, mypy
- **Commits**: Conventional Commits 1.0.0

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `pip install` fails on torch | Use `requirements-cpu.txt` or `requirements-gpu.txt` explicitly |
| BLIP not loading | Check `weights/blip/` exists or run setup with BLIP=download |
| Port 8000/3001 in use | Change `PORT` env var or kill existing process |
| CUDA OOM | Use CPU mode (`--cpu`) or reduce batch size |
| OneDrive sync issues | Ensure `.venv` and `node_modules` are junctions to local disk |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{captionai2026,
  title={CaptionAI: Local Image Captioning with CLIP + GRPO},
  author={Muhammad Hamza Mushtaq},
  year={2026},
  url={https://github.com/EquityAviator/Image-Captioning-Project}
}
```

---

## Acknowledgments

- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [BLIP](https://github.com/salesforce/BLIP) by Salesforce Research
- [GRPO](https://arxiv.org/abs/2402.03300) by DeepSeek AI
- [open_clip](https://github.com/mlfoundations/open_clip) by ML Foundations
- [Flickr8K](https://www.kaggle.com/datasets/adityajn105/flickr8k) dataset
- [shadcn/ui](https://ui.shadcn.com/) component library

---

**Made with ❤️ for the local-first ML community** — proving you don't need massive GPUs or cloud APIs for production-grade captioning.
