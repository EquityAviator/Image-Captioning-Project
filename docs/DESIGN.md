# CaptionAI — Design Document

> A single consolidated reference for **all design** in the Image Captioning Project:
> product design, system architecture, UI/UX design system, ML model design,
> data pipeline, API contract, and deployment topology.
>
> Sources merged into this document: `README.md`, `docs/PROJECT_ANALYSIS.md`,
> `backend/TRAINING_GUIDE.md`, `src/app/globals.css`, `src/lib/types.ts`.

---

## Table of contents

1. [Product design](#1-product-design)
2. [System architecture](#2-system-architecture)
3. [Frontend design](#3-frontend-design)
4. [UI / UX design system](#4-ui--ux-design-system)
5. [Backend design](#5-backend-design)
6. [ML model design](#6-ml-model-design)
7. [Data pipeline design](#7-data-pipeline-design)
8. [API design contract](#8-api-design-contract)
9. [Deployment design](#9-deployment-design)
10. [Storage & artefacts design](#10-storage--artefacts-design)
11. [Design decisions & rationale](#11-design-decisions--rationale)
12. [Known design gaps](#12-known-design-gaps)
13. [Future design directions](#13-future-design-directions)

---

## 1. Product design

### 1.1 Purpose

Given an uploaded image, produce a fluent natural-language description and present
it inside a polished web application that also explains *how* the model works.

### 1.2 Three layered goals

| Goal | Description | Primary artefacts |
| --- | --- | --- |
| **Portfolio / product** | Wrap a research model in a production-grade web app (landing, dashboard, history, model-info, settings, dark mode) | `src/` |
| **Serving resilience** | Always return a caption, even with no trained weights, via a provider fallback chain | `backend/inference/manager.py` |
| **Research** | Substantially beat the notebook baseline on standard captioning metrics | `backend/train_*.py`, `experiments/` |

### 1.3 Product evolution (V1 → V2)

The project began as a faithful port of a Flickr8K Kaggle notebook
(**DenseNet201 encoder + LSTM decoder**, Keras) and evolved over roughly one week
of iteration into a research-grade pipeline:

> **CLIP ViT-B/16 patch encoder + Bahdanau-attention LSTM decoder + BPE tokenizer,
> fine-tuned with GRPO reinforcement learning on CIDEr-D.**

Headline result on the 1,214-image validation split:

| Metric | V1 baseline | V2 (served) | Δ |
| --- | --- | --- | --- |
| BLEU-1 | 0.534 | **0.656** | +22.9% |
| BLEU-4 | 0.122 | **0.173** | +41.8% |
| ROUGE-L | 0.222 | **0.278** | +25.5% |

…achieved with a *smaller* decoder than the original.

### 1.4 Target users

- Recruiters / portfolio viewers → consume the **landing page** narrative.
- End users → upload images on the **dashboard**, read/copy/speak captions.
- Technical reviewers → inspect the **model-info** pipeline and hyperparameters.

---

## 2. System architecture

### 2.1 High-level topology

```
┌───────────────────────────────────────────────────────────────────────────┐
│  BROWSER                                                                   │
│  Next.js 16 · React 19 · TypeScript · Tailwind 4 · shadcn/ui · Framer      │
│  Single route "/", client-side view switching via Zustand                  │
│  landing │ dashboard │ history │ model-info │ settings                     │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │  relative path + ?XTransformPort=8000
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
      ┌──────────────────────────┐    ┌──────────────────────────────┐
      │  Caddy gateway  :81      │    │  Next.js  :3000              │
      │  query XTransformPort=*  │    │  rewrites() → :8000          │
      │  → localhost:8000        │    │  (redundant fallback path)   │
      │  else → localhost:3000   │    └──────────────────────────────┘
      └────────────┬─────────────┘
                   ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  FastAPI backend  :8000                                                    │
│  main.py → api/routes.py → inference/manager.py (ModelManager singleton)   │
│                                                                            │
│    ┌──────────────────────────────────────────────────────────────────┐   │
│    │ tier 0  AttentionProvider      (PyTorch CLIP + attn LSTM) ← DEFAULT│  │
│    │ tier 1  NotebookProvider       (Keras DenseNet201 + LSTM)  V1      │  │
│    │ tier 2  HuggingFaceProvider    (BLIP-base)                         │  │
│    │ tier 3  _HeuristicProvider     (colour histogram)                  │  │
│    └──────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    ▼
     weights/  (3.9 GB)        experiments/ (9 checkpoints)    dataset/ (8,091 jpg)
```

### 2.2 Component responsibilities

| Component | Port | Responsibility |
| --- | --- | --- |
| Next.js frontend | 3000 | UI, view switching, upload, prediction flow, local history |
| Caddy gateway | 81 | Route by `?XTransformPort=` query; else serve frontend |
| FastAPI backend | 8000 | Validation, provider resolution, inference dispatch |
| `weights/` | — | Feature caches, tokenizers, checkpoints, BLIP snapshot |
| `experiments/` | — | Per-run checkpoints + evaluation JSONs |

### 2.3 Request lifecycle — `POST /predict`

1. Caddy matches `?XTransformPort=8000` → proxy to `:8000` (Next.js bypassed).
2. Multipart parse → validate: content-type allow-list (**415**), empty payload (**400**), >10 MB (**413**).
3. If `?provider=` differs from the active one, **re-initialise the singleton on the event loop**.
4. Dispatch heavy work to `loop.run_in_executor(None, …)` (default `ThreadPoolExecutor`).
5. `AttentionProvider.predict`:
   - **Hybrid OOD router** (CLIP branch only): cosine-similarity of the image's CLIP
     pooled embedding against 6 zero-shot prompts. In-domain (people/animals/outdoors)
     → trained model; else → hand off to **BLIP**, return `provider="attention+blip"`, `confidence=0.90`.
   - In-domain: CLIP `forward_intermediates(indices=[-1])` → `(1, 196, 768)` patch tokens,
     memoised in a 16-entry FIFO cache.
6. **Beam search** (width 5): incremental state-carrying LSTM, GNMT length norm α=1.2,
   2-gram repeat penalty ×0.4, Diverse Beam Search λ=0.7 on 3-grams, END suppressed until 8 tokens.
7. Post-process: BPE decode → strip trailing function words.
8. Optional BLIP semantic scoring (PMI) — **off by default** (`ENABLE_BLIP_SCORE=0`).
9. Response **always HTTP 200**; failures are `success=false` with an `error` string.

---

## 3. Frontend design

### 3.1 Information architecture

A **single-route SPA** (`/`) with client-side view switching driven by a
non-persisted Zustand `nav-store`. Refresh always returns to the landing view by design.

```
landing ──▶ dashboard ──▶ history
   │             │
   └──▶ model-info ──▶ settings
```

`ViewKey = "landing" | "dashboard" | "history" | "model-info" | "settings"`

### 3.2 View composition

| View | Sections / components |
| --- | --- |
| **Landing** | hero → features (6 cards) → architecture (animated pipeline) → how-it-works (4 steps) → model-section → performance (count-up stats) → demo-cta → footer |
| **Dashboard** | sidebar + navbar + upload-zone (drag&drop) + prediction-stages (5-stage) + prediction-card + history-panel |
| **History** | localStorage-backed list with search, reuse, delete (≤50 entries) |
| **Model Info** | step-by-step pipeline diagram + hyperparameter grid + live `/model-info` status |
| **Settings** | theme picker, animation/history/typing/speech/auto-copy toggles, API URL, provider picker |

### 3.3 Directory layout (`src/`)

```
src/
├── app/
│   ├── layout.tsx        root layout, 3 Google fonts, ThemeProvider
│   ├── page.tsx          single route — client-side view switcher
│   └── globals.css       oklch theme tokens (275° indigo) + custom utilities
├── components/
│   ├── ui/               shadcn/ui primitives (Radix)
│   ├── theme-provider.tsx, theme-toggle.tsx, navbar.tsx, footer.tsx
│   ├── landing/          hero, features, architecture, how-it-works,
│   │                     model-section, performance, demo-cta, landing-page
│   └── dashboard/        shell, sidebar, navbar, dashboard-view, upload-zone,
│                         prediction-stages, prediction-card, history-panel,
│                         model-info-view, settings-view
├── hooks/                use-caption, use-typing-effect, use-speech,
│                         use-count-up, use-mounted
└── lib/                  api.ts, types.ts, settings-store.ts,
                          history-store.ts, nav-store.ts, utils.ts (cn helper)
```

### 3.4 State design (Zustand)

| Store | Persisted key | Contents |
| --- | --- | --- |
| `settings-store` | `captionai:settings` | `AppSettings` (theme, toggles, apiBaseUrl, provider) |
| `history-store` | `captionai:history` | ≤50 `HistoryEntry` (full base64 dataURLs) |
| `nav-store` | *not persisted* | active `ViewKey` — refresh returns to landing |

### 3.5 Prediction flow design

`use-caption.ts` orchestrates the request and a **5-stage animated progress**
indicator. The stage progress is presentational (hardcoded `setTimeout` beats) and
runs alongside the real network request; the caption, confidence, inference time and
provider come from the API response.

---

## 4. UI / UX design system

### 4.1 Visual language

A **dark-first, startup-grade** aesthetic: glassmorphism, animated gradient blobs,
grid/dot patterns, and an indigo / purple / blue palette with an emerald accent.

### 4.2 Colour tokens (oklch, hue 275° indigo)

| Token | Dark (default) | Light |
| --- | --- | --- |
| `--background` | `oklch(0.08 0.012 275)` | `oklch(0.99 0.002 275)` |
| `--foreground` | `oklch(0.97 0.005 275)` | `oklch(0.15 0.01 275)` |
| `--card` | `oklch(0.13 0.015 275)` | `oklch(1 0 0)` |
| `--primary` | `oklch(0.62 0.22 275)` (indigo) | `oklch(0.55 0.22 275)` |
| `--secondary` | `oklch(0.20 0.015 275)` | `oklch(0.96 0.005 275)` |
| `--accent` | `oklch(0.78 0.18 160)` (emerald) | `oklch(0.70 0.18 160)` |
| `--destructive` | `oklch(0.65 0.22 25)` | `oklch(0.58 0.24 25)` |
| `--border` | `oklch(1 0 0 / 8%)` | `oklch(0.20 0.015 275 / 12%)` |
| `--ring` | `oklch(0.62 0.22 275)` | `oklch(0.55 0.22 275)` |

Chart ramp: indigo (275) → purple (305) → blue (240) → emerald (160) → cyan (200).

Radius scale: base `--radius: 0.875rem` with `sm/md/lg/xl/2xl` derived offsets.

### 4.3 Signature utility classes

| Utility | Effect |
| --- | --- |
| `.glass` / `.glass-strong` | backdrop-blur 16/24px + saturate 180%, translucent tint |
| `.text-gradient` / `.text-gradient-accent` | clipped 135° gradient text |
| `.glow-primary` / `.glow-accent` | coloured box-shadow glow |
| `.animated-gradient-bg` | 400% gradient, 18s `gradientShift` loop |
| `.animate-blob` / `.animate-blob-slow` | 14s / 22s floating blob transform |
| `.shimmer` | skeleton loader sweep |
| `.pulse-glow` | 2.4s pulsing ring |
| `.typing-cursor::after` | blinking `▋` caret (emerald) |
| `.grid-pattern` / `.dot-pattern` | 56px grid / 24px dot backgrounds |
| `.animate-marquee` / `.animate-spin-slow` | 30s marquee / 8s slow spin |

### 4.4 Motion design

Framer Motion drives page/view transitions, hover micro-interactions, skeleton
loaders, count-up statistics, and the typing cursor. All motion respects the
`animationsEnabled` setting.

### 4.5 Accessibility & responsiveness

- Semantic HTML, ARIA labels, screen-reader-only text.
- Keyboard-navigable upload zone, visible focus rings, high-contrast tokens.
- Mobile-first responsive layout (phone / tablet / desktop).
- Theme modes: `dark | light | system` via `next-themes`.

---

## 5. Backend design

### 5.1 Module map

| Module | Role |
| --- | --- |
| `main.py` | App factory; sets `CUDA_VISIBLE_DEVICES=-1`; startup warmup; registers `/`, `/health`; mounts router |
| `paths.py` | Canonical `DATASET_DIR` / `WEIGHTS_DIR` |
| `api/routes.py` | 7 endpoints; validation constants (5 MIME types, 10 MB); optional provider hot-swap; threadpool dispatch; **errors never 4xx/5xx** |
| `api/schemas.py` | 6 Pydantic models; `PredictResponse` carries `semantic_score` / `semantic_pmi`; no request-body model (decoding params not client-controllable) |
| `inference/manager.py` | **Central abstraction**: `ModelManager` singleton, `CaptionResult` dataclass, `_BaseProvider` interface, 4 providers, `auto` + explicit-name resolution |
| `inference/attention_provider.py` | **V2 serving path** (711 lines): checkpoint auto-selection, encoder construction, hybrid OOD router, incremental beam search (DBS + repeat penalty + min-length), confidence, trailing-word trimmer, `metadata()` |
| `model/architecture.py` | V1 Keras — verbatim port of notebook cell 20 |
| `model/attention_decoder.py` | Phase-5 `BahdanauAttention` + `AttentionDecoder` (13.2M params); only `build_spatial_encoder()` still used |
| `inference/blip_scorer.py` | BLIP PMI semantic scoring; thread-safe lazy load; **disabled by default** |
| `inference/decoding.py` | V1 greedy/beam utilities |
| `utils/*` | `paths.py`, `logger.py`, `timer.py` (dead) |

### 5.2 Provider fallback chain (design pattern)

A **tiered strategy pattern** guarantees the API always returns a caption:

```
tier 0  AttentionProvider    PyTorch CLIP ViT-B/16 + Bahdanau LSTM-512  ← DEFAULT
tier 1  NotebookProvider     Keras DenseNet201 + LSTM-256 (V1)
tier 2  HuggingFaceProvider  Salesforce/blip-image-captioning-base
tier 3  _HeuristicProvider   colour-histogram captioner (keeps API alive)
```

All providers implement the `_BaseProvider` interface (`name`, `predict`, `metadata`).
`auto` resolution walks the tiers; an explicit `?provider=` name selects one directly.

### 5.3 Concurrency design

`async` handlers offload CPU-heavy inference to a `ThreadPoolExecutor` via
`run_in_executor`, so the event loop is never blocked by model forwards.

---

## 6. ML model design

### 6.1 Model generations

| Gen | Encoder | Decoder | Objective | Key params |
| --- | --- | --- | --- | --- |
| **1** | DenseNet201 GAP (frozen), 1920-d | LSTM-256 + residual, word vocab 8,427 | CE + LS 0.1, teacher forcing | Adam 1e-3, clip 5.0, batch 512 pairs |
| **2** | DenseNet201 pre-norm5 (frozen), 49×1920 | Bahdanau LSTM-512 + BPE-6k | CE + LS 0.05 + scheduled sampling | AdamW 3e-4, wd 0.01, batch 64 |
| **2-RL** | ″ | ″ | GRPO / CIDEr-D | AdamW 1e-5, G=5, EMA 0.999 |
| **3** | **CLIP ViT-B/16 patches** (frozen), 196×768 | Bahdanau LSTM-512 + BPE-6k | CE + LS 0.05 + SS | AdamW 3e-4, wd 0.01, batch 64 |
| **3-RL ★** | ″ | ″ | **GRPO / CIDEr-D** | AdamW 1e-5, G=5, 8 img/step, EMA 0.999 |
| 4 ✗ | ″ | Transformer 4L/8H d512 | CE + LS 0.05 | 23.4M params — rejected |

★ = served in production. ✗ = trained and abandoned.

### 6.2 V1 inference pipeline (notebook-faithful)

```
image bytes
   ▼ [Pillow decode + resize to 224×224]
   ▼ [DenseNet201 encoder (frozen, ImageNet)]  → (1, 1920)
   ▼ [Dense(256, relu) + Reshape((1, 256))]
   ├──▶ [Embedding(vocab_size, 256)]  ◀── partial caption "startseq …"
   ▼            ▼
   [Concatenate(axis=1)] → (1, 1+max_length, 256)
   ▼ [LSTM(256)]
   ▼ [Dropout(0.5) → Add(image_features) → Dense(128, relu) → Dropout(0.5)]
   ▼ [Dense(vocab_size, softmax)]
   ▼ argmax → idx_to_word → append → loop until "endseq" or max_length
   ▼ generated caption + mean softmax confidence
```

### 6.3 V2 serving pipeline (CLIP + attention)

```
image bytes
   ▼ [Hybrid OOD router: CLIP pooled emb vs 6 zero-shot prompts]
   │        out-of-domain ──▶ BLIP hand-off (provider="attention+blip")
   ▼ in-domain
   [CLIP ViT-B/16 forward_intermediates] → (1, 196, 768) patch tokens (FIFO-16 cache)
   ▼ [Bahdanau attention over patches, per decode step]
   ▼ [LSTM-512 decoder + BPE-6k vocab]
   ▼ [Beam search width 5: GNMT α=1.2, 2-gram repeat ×0.4, DBS λ=0.7, MIN_LENGTH 8]
   ▼ [BPE decode → strip trailing function words]
   ▼ caption + confidence (+ optional BLIP PMI semantic score)
```

### 6.4 Decoding design

| Parameter | Value | Purpose |
| --- | --- | --- |
| Beam width | 5 | search breadth (sweep optimum was 10, not transferred) |
| GNMT length norm α | 1.2 | penalise overly short sequences |
| 2-gram repeat penalty | ×0.4 | suppress repetition |
| Diverse Beam Search λ | 0.7 | diversity on 3-grams |
| MIN_LENGTH | 8 | suppress END early (decode-time fix for CIDEr shortening) |
| Trailing-word trimmer | on | strip dangling function words |

### 6.5 Reinforcement learning design (GRPO)

Group Relative Policy Optimisation (Liang, arXiv:2503.01333):

- Sample **G=5** captions per image; reward each with **CIDEr-D**.
- Advantage `A = (r − group_mean) / (group_std + 1e-6)`.
- Loss `−Σ log π(token) · A`.
- A **fixed-DF fast CIDEr scorer** is cross-checked against `pycocoevalcap`;
  the run **aborts if |diff| > 0.15**.

---

## 7. Data pipeline design

### 7.1 End-to-end flow

```
dataset/Images (8,091 jpg) + captions.txt (40,455 rows)
   │
   ├─ clean: lowercase → strip non-alpha → collapse spaces → drop len≤1
   │         → "startseq … endseq"
   ├─ split: df.image.unique() in file order (NOT shuffled)
   │         split = round(0.85 × 8091) → 6,877 train / 1,214 val   [seed 42]
   ├─ tokenize:
   │     Gen 1      word-level Keras Tokenizer, vocab 8,427, max_len 35
   │     Gen 2+     ByteLevel BPE vocab 6,000, fit on TRAIN ONLY,
   │                max_bpe_len 36 → MAX_LENGTH 38, 0 round-trip failures
   ├─ feature extraction (one-time, cached as fp16 memmap):
   │     train_features.py           → features.npy            (8091, 1920)
   │     precompute_spatial_features → features_spatial.npy    (8091, 49, 1920)
   │     precompute_clip_spatial     → clip_spatial/*.pt       (196, 768)
   │     stack_clip_features         → clip_features_spatial.npy (8091,196,768)
   └─ training: np.load(mmap_mode="r") → features[ids].copy().float() → GPU
```

### 7.2 Dataset

Flickr8K — **8,091 images, 40,455 captions** (5 references per image),
85/15 image-level split (seed 42), split in file order (not shuffled).

### 7.3 Tokenization design

The move from word-level to **ByteLevel BPE-6k** was a deliberate attack on the
*target space* rather than the model: rare words (hapax legomena like "kayakers")
become learnable subword sequences instead of unlearnable one-hot targets,
removing the memorisation shortcut behind the epoch-26 overfitting wall.

Specials: `[PAD]=0`, `[START]=1`, `[END]=2`.

---

## 8. API design contract

### 8.1 Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | root health check |
| GET | `/health` | service status + active provider |
| GET | `/model-info` | architecture + training metadata |
| GET | `/providers` | list providers |
| POST | `/providers` | set active provider |
| POST | `/predict` | caption a single image |
| POST | `/predict-batch` | caption up to 8 images |
| GET | `/docs`, `/redoc` | Swagger / ReDoc |

### 8.2 Validation rules

| Condition | Status |
| --- | --- |
| Unsupported content type | **415** |
| Empty payload | **400** |
| File > 10 MB | **413** |
| Inference failure | **200** with `success=false` + `error` |

Allowed MIME: PNG / JPG / JPEG / WEBP (+1). Max size 10 MB.

### 8.3 Response schemas (from `types.ts`)

```ts
PredictResponse {
  caption: string;
  inference_time: string;   // e.g. "1711.4ms"
  confidence: number;
  semantic_score?: number | null;   // BLIP PMI (off by default → null)
  semantic_pmi?: number | null;
  provider: string;                 // "attention" | "attention+blip" | …
  success: boolean;
  error?: string | null;
}

ModelInfoResponse {
  ready, active_provider, weights_loaded, provider, encoder,
  encoder_feature_dim, decoder, embed_dim, lstm_units, dense_units,
  dropout, training_dataset, image_size, vocab_size, max_length,
  tensorflow_version, training_metadata
}

HealthResponse { status, model_loaded, provider, weights_loaded, backend }
```

### 8.4 Client transport design

`src/lib/api.ts` builds every URL with `?XTransformPort=8000` (unless
`settings.apiBaseUrl` overrides) so calls route transparently through Caddy
without exposing internal ports. **No request timeouts are configured.**

---

## 9. Deployment design

| Path | Mechanism |
| --- | --- |
| Backend | `backend/start.sh` (one-shot uvicorn) or `backend/run.sh` (supervisor: `while true`, 3 s restart, `/tmp/backend.log`, 5 MB truncation) |
| Frontend | `bun run dev` (:3000) or `next build` + `bun .next/standalone/server.js` |
| Gateway | `Caddyfile` — `:81`, `@transform_port_query` matcher → `localhost:{query.XTransformPort}`, else → `:3000` |

### 9.1 Prerequisites

- Node.js ≥ 20 + Bun (frontend)
- Python ≥ 3.10 / 3.11 + uv (backend)
- ~5 GB disk (TensorFlow CPU + transformers + torch CPU)
- Training GPU reference: GTX 1080, 8 GB

### 9.2 Training pipeline (reproduce V1 best)

```bash
cd backend
.venv/Scripts/python.exe train_features.py          # ~15 min  (DenseNet201 features + tokenizer)
.venv/Scripts/python.exe train_torch.py --patience 20 --rlr-patience 6 --rlr-factor 0.5   # ~15 min
.venv/Scripts/python.exe convert_torch_to_keras.py  # ~10 sec  (logit gate < 1e-4)
.venv/Scripts/python.exe evaluate.py                # ~2 min   (BLEU/ROUGE on 1,214 val)
# → weights/model.h5 + tokenizer.pkl + metadata.json
```

V1 architecture (unchanged from notebook cell 20):
Encoder DenseNet201 `include_top=False, pooling="avg"` → 1920-d;
Decoder `Dense(256,relu) → Reshape(1,256) → concat(Embedding(vocab,256)) → LSTM(256)
→ Dropout(0.5) → Add(residual) → Dense(128,relu) → Dropout(0.5) → Dense(vocab,softmax)`;
Vocab 8,427 | Max length 35 | Split 6,877 / 1,214.

---

## 10. Storage & artefacts design

### 10.1 `weights/` inventory (3.9 GB)

| File | Size | Purpose |
| --- | --- | --- |
| `clip_features_spatial.npy` | 2.44 GB | (8091, 196, 768) fp16 — CLIP ViT-B/16 patch tokens, memmap |
| `features_spatial.npy` | 1.52 GB | (8091, 49, 1920) fp16 — DenseNet201 pre-norm5 grid |
| `blip/` | ~1 GB | local BLIP-base snapshot (`model.safetensors`) |
| `attention_best.pt`, `baseline_best.pt`, `decoder_torch.pt` | 53/27/17 MB | Gen-1 & Phase-5 checkpoints |
| `bpe/tokenizer.json` | — | BPE-6k, specials `[PAD]=0 [START]=1 [END]=2` |
| `blip_pseudo_captions.json` | 1.36 MB | ≤5 BLIP pseudo-captions per train image |
| `model.h5`, `tokenizer.pkl`, `metadata.json` | 17 MB | V1 Keras artefacts — still loadable |

Feature caches use NumPy `open_memmap` fp16 (`.npy`). `metadata.json` doubles as the
experiment log (loss histories, conversion gate result 1.97e-06, BLEU scores).

### 10.2 Client-side storage

`localStorage` holds settings and up to 50 history entries with **full base64
dataURLs**. There is **no database** — Prisma/SQLite is dead code.

---

## 11. Design decisions & rationale

The research narrative is unusually well documented in code docstrings. Key decisions:

1. **BPE over word-level vocab** — attacks the target space, not the model; removes
   the memorisation shortcut behind the epoch-26 overfitting wall.
2. **Attention over global features** — same frozen encoder + Bahdanau attention over
   the 7×7×1920 grid moved BLEU-4 0.1218 → 0.1663 (+37%). The decoder, not the
   features, was the first bottleneck.
3. **GRPO over plain CE** — CE optimises likelihood, not the metric; GRPO on CIDEr-D
   yielded +10.2% CIDEr.
4. **CLIP encoder swap** — better val loss with **26% fewer parameters**; greedy CLIP
   alone (0.6520 BLEU-1) beat the entire previous beam-searched pipeline. *A feature
   swap outperformed every decoder-side innovation combined.*
5. **Decode-time length fix** — CIDEr reward hacking systematically shortens captions;
   the shipped fix is `MIN_LENGTH = 8` + a trailing-function-word trimmer rather than
   a training-side change.

### 11.1 Documented dead ends

| Attempt | Outcome |
| --- | --- |
| Focal loss (γ=2, α=0.25) | greedy BLEU-1 0.0038 — gibberish |
| Encoder fine-tune (unfreeze denseblock4) | CIDEr 0.418 → 0.399; 8k images too few, BN destabilises |
| Transformer decoder (4L/8H, 23.4M) | overfits from epoch 12, BLEU-1 −10.1 pts; LSTM recurrence was a regulariser |
| BLIP distillation | best val loss (3.6643) but worst CLIP-family BLEU-1 (0.4889) |
| Mixed reward (CIDEr + 0.5·ROUGE-L) | worse on every axis |
| Length-scaled reward (rl3) | best subset CIDEr but BLEU-4 −0.013 + surface degeneration |
| BERT reranking | BLEU-1 +0.0124 but BLEU-4 −0.0056; rejected |
| DBS at λ=0.3 | negative; Flickr8K references are already similar |
| Hard 3-gram blocking | zero delta; soft 2-gram penalty sufficient |

---

## 12. Known design gaps

### Critical

1. **All V2 work is uncommitted** — 41 untracked + 15 modified files; the entire
   research programme is one `git clean -fd` from loss.
2. **`requirements.txt` is incomplete** — `pandas`, `tokenizers`, `open_clip_torch`,
   `nltk` are imported at boot but undeclared; a clean install fails.

### High

3. **Landing metrics are hardcoded and stale** — shows Gen-1 numbers (BLEU-1 0.53)
   while the served model scores 0.6559; mixes beam-w5 BLEU with beam-w10 latency;
   says "test split" for the val split; unsourced "Repetition Rate 2.8%".
4. **Landing page documents Gen-1** while the backend serves Gen-3 (6 files contradict
   `model-info-view` and the backend).
5. **Lock-free singleton + hot-swap** — provider rebind can occur under in-flight
   inference; no semaphore guards shared torch modules across executor threads.
6. **Hallucination unsolved** — 50.1% of generated nouns unsupported, 77.6% of images affected.
7. **Val loss is no longer a useful selection signal** — best-loss model is worst-BLEU.

### Medium (selected)

8. `/health` registered twice; `main.py` shadows the schema-validated `routes.py` version.
9. `semantic_score` / `semantic_pmi` structurally dead (gated off by default).
10. `CUDA_VISIBLE_DEVICES=-1` set process-wide defeats `ATT_DEVICE=cuda`.
11. `_beam_search` has no hard step cap (V1 did).
12. `/predict-batch` skips the 10 MB guard and is strictly serial.
13. No request timeouts in `api.ts` — a hung backend leaves the UI spinning.
14. History stores 50 full base64 dataURLs; quota failures silently swallowed.

### Dead code

Prisma + `src/lib/db.ts`, `src/app/api/route.ts` stub, `use-toast.ts`,
`utils/timer.py`, `getProviders`/`setProvider`/`predictBatch` (no batch UI),
`ATT_HYBRID_THRESHOLD`, `typingEffectEnabled` no-op toggle.

---

## 13. Future design directions

- **Beam search decoding** tuned to the sweep optimum (width 10, α 1.2) for V2.
- **Attention visualisation** — expose the discarded `alpha` as an image heatmap overlay.
- **Multi-image batch UI** — wire the existing `/predict-batch` typing into the upload zone.
- **Isotonic confidence recalibration** — the high-confidence region is 5–7 points over-confident;
  a single scalar temperature cannot fix a sign-flipping error.
- **CHAIR-aware / CLIPScore GRPO reward** — target the hallucination gap that BLEU/CIDEr cannot see.
- **Add METEOR / SPICE / BERTScore** — the biggest gap against published captioning work.
- **Auth + cloud history** — replace localStorage with a Supabase/Postgres backend.
- **PWA / offline** — client-side inference via `transformers.js`.
- **CI/CD** — GitHub Actions running `bun run lint`, `pytest`, and deploying both services.

---

*End of design document.*
