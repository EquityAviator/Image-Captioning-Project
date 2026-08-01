# CaptionAI — AI Image Caption Generator

> Convert any image into a natural language description using a Deep Learning pipeline (DenseNet201 encoder + LSTM decoder) trained on the Flickr8K dataset.

A production-grade, portfolio-ready web application built around a Jupyter notebook. The notebook's exact model architecture is exposed through a FastAPI inference service and consumed by a modern Next.js 16 frontend with animations, dark mode, accessibility, and a polished startup-grade UI.

---

## Table of contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [Folder structure](#folder-structure)
5. [Installation](#installation)
6. [Running the backend](#running-the-backend)
7. [Running the frontend](#running-the-frontend)
8. [API documentation](#api-documentation)
9. [Training the model](#training-the-model)
10. [Screenshots](#screenshots)
11. [Future improvements](#future-improvements)
12. [License](#license)

---

## Overview

**CaptionAI** wraps a Flickr8K image-captioning notebook in a complete, deployable AI product. The frontend is a single-page Next.js app with client-side view switching (Landing → Dashboard → History → Model Info → Settings). The backend is a FastAPI service that loads the **exact** Keras model architecture from the notebook (`backend/model/architecture.py`) and exposes three HTTP endpoints: `/health`, `/model-info`, and `/predict`.

The original notebook lives at `upload/flickr8k-image-captioning-using-cnns-lstms.ipynb`. The model architecture is replicated faithfully in `backend/model/architecture.py` — no rewrite, no extra layers, no renamed hyperparameters. Only the inference path is exposed; training is moved into an optional `backend/train.py` script that produces `weights/model.h5` + `weights/tokenizer.pkl` + `weights/metadata.json`.

### Provider fallback

The backend supports two inference providers, selected automatically at startup:

| Provider                | When it activates                                              | Notes                                                                                     |
| ----------------------- | -------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `notebook-tensorflow`   | `weights/model.h5` + `weights/tokenizer.pkl` + `weights/metadata.json` all exist | The exact DenseNet201 + LSTM architecture from the notebook. Real notebook inference.      |
| `huggingface-blip`      | Default fallback when notebook weights are absent              | Uses Salesforce/blip-image-captioning-base so the demo works end-to-end out of the box.    |
| `heuristic-fallback`    | Only if neither TensorFlow nor Transformers can be imported    | Lightweight colour-histogram captioner — keeps the API alive in resource-constrained envs. |

The active provider is reported by `/health` and `/model-info`.

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
│                       Caddy gateway (:81)                              │
│   Routes /?XTransformPort=8000 → http://localhost:8000                 │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  FastAPI backend  (:8000)                              │
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
│   │   ├── NotebookProvider  (TF + DenseNet201 + LSTM)            │    │
│   │   ├── HuggingFaceProvider  (BLIP, fallback)                  │    │
│   │   └── _HeuristicProvider  (colour-only, ultimate fallback)   │    │
│   └──────────────────────────────────────────────────────────────┘    │
│                                                                        │
│   backend/model/architecture.py  ← EXACT notebook architecture         │
│   backend/train.py               ← optional training script            │
└────────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
                  weights/ (gitignored)
                  ├── model.h5                  (Keras weights)
                  ├── tokenizer.pkl             (Keras Tokenizer)
                  ├── metadata.json             (vocab_size, max_length, …)
                  └── densenet201_*.h5          (ImageNet weights cache)
```

### Inference pipeline

```
image bytes
   │
   ▼
[Pillow decode + resize to 224×224]
   │
   ▼
[DenseNet201 encoder (frozen, ImageNet weights)]
   │  outputs (1, 1920) feature vector
   ▼
[Dense(256, relu) + Reshape((1, 256))]
   │
   ├──▶ [Embedding(vocab_size, 256)]   ◀── partial caption "startseq …"
   │            │
   ▼            ▼
[Concatenate(axis=1)] → (1, 1+max_length, 256)
   │
   ▼
[LSTM(256)]
   │
   ▼
[Dropout(0.5) → Add(image_features) → Dense(128, relu) → Dropout(0.5)]
   │
   ▼
[Dense(vocab_size, softmax)]
   │
   ▼
argmax → idx_to_word → append to caption
   │
   ▼
loop until "endseq" or max_length iterations
   │
   ▼
generated caption + mean softmax confidence
```

---

## Features

### Frontend

- **Modern startup-grade UI** with glassmorphism, animated gradient blobs, grid/dot patterns, and a dark-first palette (indigo / purple / blue + emerald accent).
- **Landing page** with hero, feature cards, animated architecture pipeline, how-it-works steps, model deep-dive, animated performance statistics, and a final CTA.
- **Dashboard** with sidebar, top navbar (search + notifications + theme toggle + avatar), large drag-and-drop upload zone, animated prediction stages, and a recent-predictions side panel.
- **Prediction card** with copy / download / share / speak (browser TTS) / regenerate buttons, plus a typing-effect caption reveal and an animated confidence bar.
- **History panel** persisted to `localStorage` (capped to 50 entries) with search, reuse, and delete.
- **Model Info page** with a step-by-step pipeline diagram, hyperparameter grid, and live backend status pulled from `/model-info`.
- **Settings page** with theme picker (dark / light / system), animation toggle, history toggle, typing-effect toggle, speech-synthesis toggle, auto-copy toggle, and a custom API URL field.
- **Animations everywhere** via Framer Motion: page transitions, view transitions, hover micro-interactions, skeleton loaders, count-up statistics, and a typing cursor.
- **Accessibility**: semantic HTML, ARIA labels, keyboard-navigable upload zone, focus rings, screen-reader-only text, and high-contrast theme tokens.
- **Responsive**: mobile-first, tested on phone / tablet / desktop widths.

### Backend

- **FastAPI service** with `async` handlers that dispatch CPU-heavy inference to a threadpool via `run_in_executor` — never blocks the event loop.
- **Exact notebook architecture** in `backend/model/architecture.py` — `build_encoder()` returns DenseNet201 with the head removed, `build_decoder()` returns the exact same `Dense(256) → Reshape → Embedding → Concatenate → LSTM(256) → Dropout → Add → Dense(128) → Dropout → Dense(vocab, softmax)` stack as notebook cell 20.
- **Provider fallback chain** so the demo always works even before you've trained the model.
- **Batch endpoint** (`POST /predict-batch`) accepts up to 8 images at once.
- **CORS enabled** for cross-origin development.
- **Auto-restart supervisor** (`backend/run.sh`) — if the Python process dies, it restarts within 3 seconds.
- **Optional training script** (`backend/train.py`) that ports the notebook's training loop verbatim and saves `model.h5` + `tokenizer.pkl` + `metadata.json` into `weights/`.

---

## Folder structure

```
.
├── README.md                       ← you are here
├── package.json                    ← Next.js 16 + React 19 + Tailwind 4 + shadcn/ui
├── next.config.ts
├── tailwind.config.ts
├── tsconfig.json
├── Caddyfile                       ← gateway config (port 81 → 3000 / 8000)
│
├── src/                            ← Next.js frontend (App Router)
│   ├── app/
│   │   ├── layout.tsx              ← root layout + ThemeProvider + fonts
│   │   ├── page.tsx                ← single route, client-side view switching
│   │   └── globals.css             ← Tailwind 4 theme tokens + custom utilities
│   │
│   ├── components/
│   │   ├── ui/                     ← shadcn/ui component library
│   │   ├── theme-provider.tsx
│   │   ├── theme-toggle.tsx
│   │   ├── navbar.tsx              ← landing navbar
│   │   ├── footer.tsx              ← shared footer
│   │   ├── landing/
│   │   │   ├── landing-page.tsx    ← landing composition
│   │   │   ├── hero.tsx            ← animated hero with gradient blobs
│   │   │   ├── features.tsx        ← 6 feature cards
│   │   │   ├── architecture.tsx    ← pipeline diagram with animated arrows
│   │   │   ├── how-it-works.tsx    ← 4-step cards
│   │   │   ├── model-section.tsx   ← encoder/decoder/embedding/dataset
│   │   │   ├── performance.tsx     ← animated count-up statistics
│   │   │   └── demo-cta.tsx        ← final CTA
│   │   └── dashboard/
│   │       ├── dashboard-shell.tsx ← sidebar + navbar + view switcher
│   │       ├── dashboard-navbar.tsx
│   │       ├── sidebar.tsx
│   │       ├── dashboard-view.tsx  ← main prediction panel
│   │       ├── upload-zone.tsx     ← drag&drop + preview + progress
│   │       ├── prediction-stages.tsx ← 5-stage animated progress
│   │       ├── prediction-card.tsx ← copy/download/share/speak/regenerate
│   │       ├── history-panel.tsx   ← localStorage history
│   │       ├── model-info-view.tsx ← pipeline diagram + hyperparams
│   │       └── settings-view.tsx   ← theme/animation/API URL settings
│   │
│   ├── hooks/
│   │   ├── use-caption.ts          ← predict API + stage management
│   │   ├── use-typing-effect.ts    ← typewriter caption reveal
│   │   ├── use-speech.ts           ← browser TTS wrapper
│   │   ├── use-count-up.ts         ← animated number on scroll
│   │   └── use-mounted.ts          ← hydration-safe mounted flag
│   │
│   └── lib/
│       ├── api.ts                  ← typed FastAPI client (XTransformPort)
│       ├── types.ts                ← shared TypeScript types
│       ├── settings-store.ts       ← Zustand + persist (settings)
│       ├── history-store.ts        ← Zustand + persist (history)
│       ├── nav-store.ts            ← Zustand (active view)
│       └── utils.ts                ← shadcn cn() helper
│
├── backend/                        ← FastAPI service
│   ├── main.py                     ← app + startup + CORS + routes
│   ├── requirements.txt
│   ├── start.sh                    ← one-shot launcher
│   ├── run.sh                      ← auto-restart supervisor
│   ├── train.py                    ← optional Flickr8K training script
│   ├── api/
│   │   ├── routes.py               ← /health, /model-info, /predict, /predict-batch
│   │   └── schemas.py              ← Pydantic models
│   ├── model/
│   │   └── architecture.py         ← EXACT notebook architecture (cell 20)
│   ├── inference/
│   │   └── manager.py              ← provider fallback chain
│   ├── utils/
│   │   ├── paths.py                ← centralised weight paths
│   │   ├── logger.py               ← structured logging
│   │   └── timer.py                ← context-manager timer
│   └── routes/                     ← (reserved for future route modules)
│
├── weights/                        ← trained model artefacts (gitignored)
│   ├── model.h5                    ← Keras weights (place here after training)
│   ├── tokenizer.pkl               ← Keras Tokenizer
│   ├── metadata.json               ← vocab_size, max_length, etc.
│   └── densenet201_*.h5            ← ImageNet weights cache
│
└── upload/
    └── flickr8k-image-captioning-using-cnns-lstms.ipynb   ← source notebook
```

---

## Installation

### Prerequisites

- **Node.js ≥ 20** and **Bun** (for the frontend)
- **Python ≥ 3.10** (for the backend)
- ~5 GB free disk space (for TensorFlow CPU + transformers + torch CPU)

### 1. Clone & install frontend dependencies

```bash
git clone <your-repo-url> captionai
cd captionai
bun install
```

### 2. Install backend dependencies

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> If you only want the BLIP fallback (no TensorFlow notebook provider), you can skip `tensorflow-cpu` and save ~600 MB. The `HuggingFaceProvider` only needs `transformers` + `torch` (CPU build).

---

## Running the backend

### Option A — Auto-restart supervisor (recommended for dev)

```bash
bash backend/run.sh
```

Restarts the uvicorn process automatically if it dies. Logs to `/tmp/backend.log`.

### Option B — Direct uvicorn

```bash
cd backend
PORT=8000 python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Option C — One-shot launcher

```bash
bash backend/start.sh
```

The backend listens on **port 8000** and exposes:

- `GET  /`            → root health check
- `GET  /health`      → service status + active provider
- `GET  /model-info`  → architecture + training metadata
- `POST /predict`     → caption a single image
- `POST /predict-batch` → caption up to 8 images at once
- `GET  /docs`        → Swagger UI
- `GET  /redoc`       → ReDoc UI

---

## Running the frontend

```bash
# from the project root
bun run dev
```

The Next.js dev server listens on **port 3000**. In production you would typically front both services with the Caddy gateway (already configured in `Caddyfile`):

- `https://your-domain/`  → Next.js (port 3000)
- `https://your-domain/<api>?XTransformPort=8000`  → FastAPI (port 8000)

All API calls from the frontend use **relative paths** with the `XTransformPort` query parameter, so they work transparently through the gateway without exposing internal ports.

---

## API documentation

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "provider": "huggingface-blip",
  "weights_loaded": false,
  "backend": "TensorFlow n/a (transformers)"
}
```

### `GET /model-info`

```json
{
  "ready": true,
  "active_provider": "huggingface-blip",
  "weights_loaded": false,
  "provider": "huggingface-blip",
  "encoder": "Vision Transformer (BLIP)",
  "encoder_feature_dim": 768,
  "decoder": "BERT-style text decoder",
  "embed_dim": 768,
  "lstm_units": 0,
  "dense_units": 0,
  "dropout": 0.0,
  "training_dataset": "BLIP pretraining corpus (COYO + LAION)",
  "image_size": 384,
  "vocab_size": 30522,
  "max_length": 50,
  "tensorflow_version": "n/a (transformers)",
  "training_metadata": { "note": "HuggingFace fallback provider" }
}
```

### `POST /predict`

**Request** — `multipart/form-data` with a single field:

| field | type   | description                            |
| ----- | ------ | -------------------------------------- |
| file  | binary | PNG / JPG / JPEG / WEBP, ≤ 10 MB       |

**Response — 200 OK**

```json
{
  "caption": "a house with a red roof and a green tree",
  "inference_time": "1711.4ms",
  "confidence": 0.92,
  "provider": "huggingface-blip",
  "success": true,
  "error": null
}
```

**Example with curl**

```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@/path/to/image.jpg"
```

### `POST /predict-batch`

**Request** — `multipart/form-data` with up to 8 `files` fields.

**Response**

```json
{
  "results": [
    { "filename": "a.jpg", "caption": "...", "inference_time": "...", "confidence": 0.9, "success": true, "error": null },
    { "filename": "b.jpg", "caption": "...", "inference_time": "...", "confidence": 0.9, "success": true, "error": null }
  ],
  "total": 2,
  "success_count": 2
}
```

---

## Training the model

If you want real notebook inference (instead of the BLIP fallback), train the model on Flickr8K and drop the artefacts into `weights/`:

### 1. Download Flickr8K

- Images: <https://www.kaggle.com/datasets/adityajn105/flickr8k/download?datasetVersionNumber=1>
- Captions: same archive, `captions.txt`

### 2. Run the training script

```bash
cd backend
python3 train.py \
    --images /path/to/flickr8k/Images \
    --captions /path/to/flickr8k/captions.txt \
    --epochs 20 \
    --batch-size 32
```

This is a verbatim port of the notebook's training loop (cells 9–28). After training you'll have:

- `weights/model.h5` — best decoder weights (EarlyStopping-monitored)
- `weights/tokenizer.pkl` — pickled Keras Tokenizer
- `weights/metadata.json` — `vocab_size`, `max_length`, hyperparameters, final losses

### 3. Restart the backend

On next startup, `ModelManager` will detect the three files and automatically switch from `huggingface-blip` to `notebook-tensorflow`. The `/health` endpoint will report `weights_loaded: true`.

---

## Screenshots

Screenshots are saved to `download/` as you run the app:

| File                                | What it shows                              |
| ----------------------------------- | ------------------------------------------ |
| `download/landing-screenshot.png`   | Full landing page (hero + features)        |
| `download/dashboard-screenshot.png` | Empty dashboard with upload zone           |
| `download/prediction-result.png`    | Dashboard with a successful prediction     |

---

## Future improvements

- **Beam search decoding** instead of greedy argmax for higher-quality captions.
- **Attention visualisation** — overlay the image with a heatmap showing which regions influenced each generated word.
- **Multi-image batch UI** — drag multiple files into the upload zone and caption them in parallel via `/predict-batch`.
- **Translation** — pipe the generated caption through an MT API (e.g. DeepL) for non-English output.
- **Latency graph** — record inference times in `localStorage` and plot them with Recharts (already a project dependency).
- **Auth + cloud history** — replace the local-history store with a Supabase/Postgres backend so captions sync across devices.
- **Image comparison slider** — drag handle to compare the original image with an attention-overlay version.
- **PWA / offline mode** — cache the model in the browser via `transformers.js` for fully client-side inference.
- **CI/CD** — GitHub Actions pipeline that runs `bun run lint`, `pytest`, and deploys both services on merge to `main`.

---

## License

MIT — see `LICENSE` (or feel free to use this as a portfolio piece without one).
