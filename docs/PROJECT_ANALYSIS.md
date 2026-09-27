# CaptionAI — Comprehensive Project Analysis

**Repository:** `Image-Captioning-Project`
**Analysis date:** 2026-08-30
**Scope:** full repository, both branches (`main`, `feature/local-caption-improvement`)

---

## 1. Executive summary

CaptionAI is an end-to-end **image captioning** product: a research-trained encoder–decoder model wrapped in a FastAPI inference service and a Next.js 16 web UI. It takes the classic Flickr8K notebook architecture (DenseNet201 encoder + LSTM decoder) as its starting point and then — across roughly one week of iteration — evolves it into a **CLIP ViT-B/16 patch encoder + Bahdanau-attention LSTM decoder + BPE tokenizer, fine-tuned with GRPO reinforcement learning on CIDEr-D**.

Headline result: **BLEU-1 0.534 → 0.656, BLEU-4 0.122 → 0.173, ROUGE-L 0.222 → 0.278** (+22.9% / +41.8% / +25.5%) on the 1,214-image validation split, achieved with a *smaller* decoder than the original.

The single most important structural finding of this analysis:

> **The two branches are not two versions of the code in git — they are one commit plus uncommitted work.**
> `main` and `feature/local-caption-improvement` both point at commit `36783b3`. `git diff main feature/…` is **empty**. All of the "V2" work lives in the *working tree* as **41 untracked files** and **15 modified tracked files**. Nothing has been committed.

This is a material risk: the entire research programme (12 training scripts, 8 eval scripts, 9 experiment checkpoints, ~4 GB of feature caches) is **one `git clean -fd` away from permanent loss.**

---

## 2. Branch reality check

```
$ git rev-parse main feature/local-caption-improvement origin/main
36783b3657aec1004750afab5aa469de2eaf139f
36783b3657aec1004750afab5aa469de2eaf139f
36783b3657aec1004750afab5aa469de2eaf139f

$ git diff --stat main feature/local-caption-improvement
(empty)
```

Commit history is 3 commits deep: `6e875bc Initial commit` → `801afba` → `36783b3 Commiting V1`.

| | `main` (committed) | `feature/…` working tree (uncommitted) |
|---|---|---|
| Model | V1 — Keras DenseNet201-GAP + LSTM-256 | V2 — PyTorch CLIP patches + Bahdanau LSTM-512 + BPE-6k + GRPO |
| Training scripts | 3 | 15 |
| Eval scripts | 0 | 8 |
| Experiment dirs | 0 | 14 |
| Provider served | `notebook-tensorflow` (or BLIP fallback) | `attention` (tier 0) |
| Frontend model-info | basic hyperparameter grid | 702-line "generations / hurdles / optimizations" narrative |

**Interpretation for the "analyze both branches" request:** V1 is what git knows about; V2 is what actually runs. The delta below is the real content of the feature branch.

**V2 delta — 15 modified tracked files (+921/−733):**

| File | Change |
|---|---|
| `backend/inference/manager.py` | +118/−… new `attention` provider tier inserted at the top of the fallback chain |
| `src/components/dashboard/model-info-view.tsx` | +616 lines — generation history, hurdles, live pipeline diagram |
| `backend/api/routes.py` | +33 — `/providers` GET/POST, `?provider=` query param |
| `backend/api/schemas.py` | +5 — `semantic_score`, `semantic_pmi` |
| `src/components/landing/performance.tsx` | +43 metrics updated (but see §11 — they are stale) |
| `src/components/dashboard/prediction-card.tsx` | +45 — semantic-match card |
| `src/lib/types.ts` | +8 — new response fields, `"attention"` provider |
| `src/components/dashboard/settings-view.tsx` | +6 — provider picker relabelled |
| `next.config.ts`, `package.json`, `backend/main.py` | rewires/pins |
| `backend/TRAINING_GUIDE.md`, `TRAINING_CHECKLIST.md` | rewritten for the V2 pipeline |
| `backend/train_improved.py` | **deleted** (0-byte) |
| `package-lock.json` | +292/−… dependency churn |

**V2 delta — 41 untracked paths:** 24 new Python modules under `backend/`, plus `docs/`, `experiments/`, `scripts/`, `weights/`.

---

## 3. Project purpose

Given an uploaded image, produce a fluent natural-language description, and present it in a polished web application that also explains *how* the model works.

Three layered goals, in tension with each other:

1. **Portfolio/product goal** — wrap a notebook model in a production-grade web app (landing page, dashboard, history, model-info, settings, dark mode).
2. **Serving goal** — always return a caption, even with no trained weights (provider fallback chain).
3. **Research goal** — substantially beat the notebook baseline on standard captioning metrics. This is where nearly all the recent effort went, and it is what the feature branch is about.

The README documents only goals 1 and 2. It is **superseded** and no longer describes the running system.

---

## 4. Overall architecture

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

### Request lifecycle — `POST /predict`

1. Caddy matches `?XTransformPort=8000` → proxy to `:8000` (Next.js bypassed entirely).
2. Multipart parse → validate: content-type allow-list (**415**), empty payload (**400**), >10 MB (**413**).
3. If `?provider=` differs from active, **re-initialise the singleton on the event loop** (blocks it for the full `torch.load`).
4. Dispatch heavy work to `loop.run_in_executor(None, …)` — default `ThreadPoolExecutor`.
5. `AttentionProvider.predict`:
   - **Hybrid OOD router** (CLIP branch only): cosine-similarity the image's CLIP pooled embedding against 6 zero-shot prompts. `argmax == 0` (in-domain: people/animals/outdoors) → trained model; else → **hand off to BLIP**, return `provider="attention+blip"`, `confidence=0.90`.
   - In-domain: CLIP `forward_intermediates(indices=[-1])` → `(1, 196, 768)` patch tokens. Memoised in a 16-entry FIFO cache.
6. **Beam search** (width 5): incremental state-carrying LSTM, GNMT length norm α=1.2, 2-gram repeat penalty ×0.4, Diverse Beam Search λ=0.7 on 3-grams, END suppressed until 8 tokens.
7. Post-process: BPE decode → strip trailing function words.
8. Optional BLIP semantic scoring (PMI) — **off by default** (`ENABLE_BLIP_SCORE=0`).
9. Response **always HTTP 200**; failures are `success=false` with an `error` string.

---

## 5. Technology stack

| Layer | Technology |
|---|---|
| **Frontend** | Next.js 16.1 (App Router), React 19, TypeScript 5.9, Tailwind CSS 4, shadcn/ui (Radix), Framer Motion 12, Zustand 5 (persist), sonner, lucide-react, recharts, next-themes |
| **Backend** | FastAPI 0.128, uvicorn 0.44, Pydantic 2.9, python-multipart, Pillow, NumPy 2.4 |
| **ML — training** | PyTorch 2.13 (GTX 1080, 8 GB), `open_clip` (ViT-B/16 OpenAI), HuggingFace `transformers` 4.46, `tokenizers` (ByteLevel BPE), NLTK, pycocoevalcap |
| **ML — V1 legacy** | TensorFlow-CPU 2.21, Keras (DenseNet201) |
| **Data** | Flickr8K — 8,091 images, 40,455 captions (5 refs/image) |
| **Gateway** | Caddy (port 81) |
| **Feature storage** | NumPy `open_memmap` fp16 (`.npy`) |
| **Unused/dead** | Prisma + SQLite (`src/lib/db.ts`, `prisma/schema.prisma` — zero importers) |

**`requirements.txt` is incomplete.** `pandas`, `tokenizers`, `open_clip_torch`, and `nltk` are imported at module scope on the app's critical import path but are not declared. A clean install will fail at boot.

---

## 6. Directory structure

```
.
├── src/                          Next.js frontend (App Router)
│   ├── app/
│   │   ├── layout.tsx            root layout, 3 Google fonts, ThemeProvider
│   │   ├── page.tsx              single route — client-side view switcher
│   │   ├── globals.css           327 lines — oklch theme tokens (275° indigo)
│   │   └── api/route.ts          DEAD — 5-line "Hello, world!" stub
│   ├── components/
│   │   ├── ui/                   48 shadcn/ui primitives
│   │   ├── landing/              8 sections (hero→features→architecture→
│   │   │                         how-it-works→model→performance→CTA)
│   │   └── dashboard/            shell, sidebar, upload-zone, prediction-card,
│   │                             history-panel, model-info-view, settings-view
│   ├── hooks/                    use-caption, use-typing-effect, use-speech,
│   │                             use-count-up, use-mounted, use-toast (dead)
│   └── lib/                      api.ts, types.ts, 3 Zustand stores, db.ts (dead)
│
├── backend/                      FastAPI service
│   ├── main.py                   app factory, CORS, startup, /health, /
│   ├── paths.py                  canonical DATASET_DIR / WEIGHTS_DIR
│   ├── api/{routes,schemas}.py   endpoints + Pydantic models
│   ├── model/
│   │   ├── architecture.py       V1 Keras (notebook cell 20, verbatim)
│   │   └── attention_decoder.py  Phase-5 decoder + shared build_spatial_encoder()
│   ├── inference/
│   │   ├── manager.py            ModelManager + 4 providers  ← central abstraction
│   │   ├── attention_provider.py V2 serving path (711 lines) ← the real workhorse
│   │   ├── blip_scorer.py        BLIP PMI semantic scoring
│   │   └── decoding.py           V1 greedy/beam utilities
│   ├── train*.py                 15 training scripts (see §8)
│   ├── eval*.py, evaluate*.py    8 evaluation scripts (see §9)
│   ├── precompute*.py            4 feature-extraction scripts
│   ├── build_bpe.py              BPE-6k tokenizer
│   ├── generate_blip_pseudo.py   offline pseudo-caption generation
│   ├── convert_torch_to_keras.py torch→Keras weight port w/ logit gate
│   ├── fit_temperature.py, calibrate_semantic.py, beam_sweep.py,
│   │   rerank_bert_test.py       calibration & decoding experiments
│   └── utils/{paths,logger,timer}.py
│
├── weights/           3.9 GB — feature caches, tokenizers, BLIP snapshot
├── experiments/       14 dirs — 9 checkpoints + eval JSONs
├── scripts/           audit_dataset.py, check_env.py, check_preprocessing.py, evaluate.py
├── dataset/           8,091 JPEGs + captions.txt
├── docs/              2 PDFs (research report)
├── upload/            the original Kaggle notebook (.ipynb)
├── Caddyfile          gateway :81
└── prisma/            DEAD — untouched starter schema (User/Post)
```

### `weights/` inventory (3.9 GB)

| File | Size | Purpose |
|---|---|---|
| `clip_features_spatial.npy` | 2.44 GB | **(8091, 196, 768) fp16** — CLIP ViT-B/16 patch tokens, memmap |
| `features_spatial.npy` | 1.52 GB | **(8091, 49, 1920) fp16** — DenseNet201 pre-norm5 grid |
| `blip/` | ~1 GB | local BLIP-base snapshot (`model.safetensors`) |
| `attention_best.pt`, `baseline_best.pt`, `decoder_torch.pt` | 53/27/17 MB | Gen-1 & Phase-5 checkpoints |
| `bpe/tokenizer.json` | — | BPE-6k, specials `[PAD]=0 [START]=1 [END]=2` |
| `blip_pseudo_captions.json` | 1.36 MB | ≤5 BLIP pseudo-captions per train image |
| `model.h5`, `tokenizer.pkl`, `metadata.json` | 17 MB | **V1 Keras artefacts — still loadable** |

`metadata.json` doubles as the **experiment log** for V1/Phase-5 (loss histories, conversion gate result 1.97e-06, BLEU scores).

---

## 7. Module responsibilities

| Module | Role |
|---|---|
| `backend/main.py` | App factory. Sets `CUDA_VISIBLE_DEVICES=-1` at import (hides GPU from PyTorch too). Legacy `@app.on_event("startup")` warmup — loads the provider but runs **no dummy inference**. Registers `/` and `/health`, then mounts the router. |
| `backend/api/routes.py` | 7 endpoints. Validation constants: 5 MIME types, 10 MB. Optional provider hot-swap. Threadpool dispatch. **Inference errors never 4xx/5xx.** |
| `backend/api/schemas.py` | 6 Pydantic models. `PredictResponse` carries `semantic_score` / `semantic_pmi`. **No request-body model exists** — decoding params are not client-controllable. |
| `backend/inference/manager.py` | **Central abstraction.** `ModelManager` singleton, `CaptionResult` dataclass, `_BaseProvider` interface (`name`, `predict`, `metadata`), 4 concrete providers, `auto` resolution + explicit-name resolution. |
| `backend/inference/attention_provider.py` | **The V2 serving path and the most important file in the project.** Checkpoint auto-selection, CLIP/DenseNet encoder construction, hybrid OOD router, incremental beam search with DBS + repeat penalty + min-length, confidence, trailing-word trimmer, `metadata()` for the UI. |
| `backend/model/architecture.py` | V1 Keras — verbatim port of notebook cell 20. DenseNet201-GAP → 1920-d → `Dense(256)→Reshape→Concat(Embedding)→LSTM(256)→Dropout→Add→Dense(128)→Dropout→Dense(V,softmax)`. |
| `backend/model/attention_decoder.py` | Phase-5 `BahdanauAttention` + `AttentionDecoder` (13.2M params). **Superseded** — only `build_spatial_encoder()` (frozen pre-norm5 DenseNet201) is still used, for the DenseNet branch. |
| `backend/inference/blip_scorer.py` | `PMI = mean logP(caption \| image) − mean logP(caption \| gray)`. `semantic = sigmoid((PMI − 0.35)/0.45)`. Thread-safe lazy load. **Disabled by default.** |
| `backend/utils/*` | `paths.py` (centralised weight paths), `logger.py`, `timer.py` (**dead** — imported by nothing). |
| `src/lib/api.ts` | Typed client. `buildUrl` appends `?XTransformPort=8000` unless `settings.apiBaseUrl` overrides. **No timeouts anywhere.** |
| `src/hooks/use-caption.ts` | Prediction flow. **Stage progress is fake** — hardcoded `setTimeout(400/1100/1900)` running alongside the real request. |
| `src/lib/*-store.ts` | Zustand + persist. Settings (`captionai:settings`) and history (`captionai:history`, ≤50 entries, **full base64 dataURLs**). `nav-store` is deliberately **not** persisted → refresh always returns to landing. |

---

## 8. Training pipeline — the research progression

### 8.1 Data pipeline (end to end)

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
   │     train_features.py          → features.npy           (8091, 1920)
   │     precompute_spatial_features→ features_spatial.npy   (8091, 49, 1920)
   │     precompute_clip_spatial    → clip_spatial/*.pt      (196, 768)  [purged]
   │     stack_clip_features        → clip_features_spatial.npy (8091,196,768)
   └─ training: np.load(mmap_mode="r") → features[ids].copy().float() → GPU
```

### 8.2 Model generations

| Gen | Encoder | Decoder | Training objective | Key params |
|---|---|---|---|---|
| **1** | DenseNet201 GAP (frozen), 1920-d | LSTM-256 + residual, word vocab 8,427 | CE + LS 0.1, teacher forcing | Adam 1e-3, clip 5.0, batch 512 pairs |
| **2** | DenseNet201 pre-norm5 (frozen), 49×1920 | Bahdanau LSTM-512 + BPE-6k | CE + LS 0.05 + **scheduled sampling** | AdamW 3e-4, wd 0.01, batch 64 |
| **2-RL** | ″ | ″ | **GRPO / CIDEr-D** | AdamW 1e-5, G=5, EMA 0.999 |
| **3** | **CLIP ViT-B/16 patches** (frozen), 196×768 | Bahdanau LSTM-512 + BPE-6k | CE + LS 0.05 + SS | AdamW 3e-4, wd 0.01, batch 64 |
| **3-RL** ★ | ″ | ″ | **GRPO / CIDEr-D** | AdamW 1e-5, G=5, 8 img/step, EMA 0.999 |
| 4 ✗ | ″ | **Transformer 4L/8H d512** | CE + LS 0.05 | 23.4M params — rejected |

★ = served in production. ✗ = trained and abandoned.

### 8.3 Why each step was taken

The narrative is unusually well documented in the docstrings — this is a genuine experimental log, not guesswork.

1. **Gen 1 hits an "epoch-26 overfitting wall."** Val loss bottoms at 4.7517 then climbs to 5.0132 while train keeps falling. Diagnosis: rare words (hapax legomena like "kayakers", seen twice) are unlearnable one-hot targets — the model memorises instead of generalising.
2. **BPE-6k attacks the target space, not the model.** `build_bpe.py:8-11`: *"rare words become learnable subword sequences instead of unlearnable hapax targets — removes the memorisation shortcut behind the epoch-26 overfitting wall."*
3. **A control experiment proves the recipe alone is insufficient.** `train_v2.py` applies the full anti-overfitting recipe (AdamW+wd, embedding dropout, cosine+warmup, EMA) to the *global-feature* LSTM: val loss 4.4205, BLEU-1 0.5358 — barely better than 4.7517/0.5334.
4. **Attention is what moves the needle.** Same frozen encoder + Bahdanau attention over the 7×7×1920 grid: **BLEU-4 0.1218 → 0.1663 (+37%)**. The decoder, not the features, was the first bottleneck.
5. **A Phase-5 attempt failed first, and taught four lessons.** `train_attention.py` achieved the *best* val loss yet (3.6845) but BLEU-1 collapsed to 0.3531 with 2-gram repetition at 0.2315. `train_attention_v2.py:5-13` lists the repairs: feature/provider mismatch, production-clobber risk, insufficient regularisation, and **exposure bias → scheduled sampling**.
6. **CE optimises likelihood, not the metric → GRPO.** `train_attention_v2_rl.py` implements Group Relative Policy Optimisation (Liang, arXiv:2503.01333): sample G=5 captions per image, reward each with CIDEr-D, advantage `A = (r − group_mean)/(group_std + 1e-6)`, loss `−Σ log π(token)·A`. A **fixed-DF fast CIDEr scorer** is cross-checked against `pycocoevalcap` and the run **aborts if |diff| > 0.15**. CIDEr +10.2%.
7. **The encoder is the last lever → CLIP.** `train_attention_v2_clip.py` is a near-verbatim copy of the Gen-2 trainer with `enc_dim 1920→768`. Only the features change. Result: better val loss **with 26% fewer parameters**; greedy CLIP alone (0.6520 BLEU-1) beats the entire previous beam-searched pipeline. *A feature swap outperformed every decoder-side innovation combined.*

### 8.4 Dead ends (all explicitly documented)

| Attempt | Outcome |
|---|---|
| **Focal loss** (γ=2, α=0.25) | train loss falls but **greedy BLEU-1 0.0038 — gibberish**. Script never completed; `validate_cider()` is a stub returning `0.0`. |
| **Encoder fine-tune** (unfreeze denseblock4, BN frozen, differential LR) | CIDEr 0.418 → 0.399 @ ep2 → stopped. *"8k images too few; BatchNorm stats destabilize."* Never ran to completion. |
| **Transformer decoder** (4L/8H, 23.4M) | val loss 3.7941, overfits from epoch 12, BLEU-1 −10.1 pts. LSTM recurrence was acting as a regulariser. Rejected. |
| **BLIP distillation** | **Best val loss in the project (3.6643) and worst BLEU-1 of the CLIP family (0.4889).** The model learned BLIP's phrasing, not Flickr8K's. |
| **Mixed reward** (CIDEr + 0.5·ROUGE-L) | Worse on every axis. ROUGE-L's LCS F-measure is itself precision-dominated on short captions. |
| **Length-scaled reward** (rl3) | Best subset CIDEr (0.5306) and length restored — but BLEU-4 −0.013 and visible surface degeneration (`"woman in black is standing in the in the of the"`). Not promoted. |
| **BERT reranking** (distilbert pseudo-perplexity over 5 DBS beams) | BLEU-1 +0.0124 but **BLEU-4 −0.0056**, ROUGE-L −0.0013; changed 812/1,214 captions. Rejected. |
| **Diverse Beam Search at λ=0.3** | Negative; Flickr8K's references are themselves similar. |
| **Hard 3-gram blocking** | Zero delta — the soft 2-gram penalty was already sufficient. |

### 8.5 The unsolved problem: CIDEr reward hacking

GRPO on CIDEr-D systematically **shortens captions** (precision-over-recall bias): 8.2→7.4 words (Gen 2-RL), 7.8→6.3 (Gen 3-RL). The rl2 trace is textbook:

| GRPO step | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Reward | 0.3141 | 0.4036 | 0.4640 | 0.5041 | **0.5345** |
| CIDEr (subset) | 0.5012 | 0.5129 | **0.5159** | 0.5115 | 0.5016 |
| Mean length | 7.7 | 6.9 | 6.3 | 6.1 | **6.0** |

Reward rises monotonically while CIDEr peaks at step 3 and decays. Two training-side fixes were tried and rejected (§8.4). **The shipped fix is decode-time**: `MIN_LENGTH = 8` (suppress END) plus a trailing-function-word trimmer in `attention_provider.py`.

---

## 9. Evaluation

### 9.1 Master results table

Full validation split (1,214 images), **beam-5, GNMT α=1.2**:

| Variant | Encoder | BLEU-1 | BLEU-2 | BLEU-3 | **BLEU-4** | ROUGE-L | Val loss |
|---|---|---|---|---|---|---|---|
| Gen-1 baseline | DenseNet GAP | 0.5334 | 0.3203 | 0.1890 | 0.1218 | 0.2216 | 4.7517 |
| v2_global (control) | DenseNet GAP | 0.5358 | 0.3383 | 0.2092 | 0.1373 | 0.2236 | 4.4205 |
| **v2_attention** (CE) | DenseNet 7×7 | 0.5649 | 0.3704 | 0.2447 | 0.1663 | 0.2548 | 3.8161 |
| v2_attention_rl | ″ | 0.5971 | 0.3872 | 0.2456 | 0.1612 | 0.2566 | — |
| **v2_attention_clip_rl ★** | **CLIP patches** | **0.6559** | **0.4344** | **0.2698** | **0.1727** | **0.2782** | — |
| v2_attention_clip_rl2 | ″ | 0.6514 | 0.4319 | 0.2709 | 0.1722 | 0.2774 | — |
| v2_attention_clip_rl3 | ″ | 0.6538 | 0.4280 | 0.2590 | 0.1601 | 0.2782 | — |
| v2_attention_clip_distill ✗ | ″ | 0.4889 | 0.2843 | 0.1740 | 0.1153 | 0.2142 | **3.6643** |
| v2_attention_tf ✗ | ″ | *never evaluated* | | | | | 3.7941 |

**⚠ Two provenance defects in the artifacts:**
1. `experiments/v2_attention/eval_results.json` does **not** measure `v2_attention/best.pt` — it measures `v2_attention_rl/best.pt`. `eval_v2.py:69-71` derives `out_dir` from `--model`, not `--ckpt`.
2. `experiments/v2_attention_clip/eval_results.json` is **byte-identical** to `v2_attention_clip_rl`'s. `eval_v2_clip.py` writes to `ckpt_path.parent`, so evaluating RL into the CLIP directory **overwrote** the plain-CLIP CE eval, which no longer exists on disk.
3. The headline CLIP-CE numbers (0.6092 / 0.1755 / 0.2747) are **hardcoded** in `attention_provider.py:59,582` and `model-info-view.tsx:358-360` — no artifact backs them.

### 9.2 Metrics that do not exist

**METEOR, BERTScore, CLIPScore, SPICE, and self-retrieval are never computed anywhere in this repository.** METEOR appears only as the string `"skipped on Windows (pycocoevalcap Java subprocess incompatible)"`. CIDEr-D is measured on the full split for exactly one checkpoint (`v2_attention` CE, 0.5397); every RL variant reports CIDEr only on a 400-image training subset, so the CIDEr column is not comparable across variants.

### 9.3 Priority-0 diagnostic — hallucination

`eval_priority0.py` adds the metrics the earlier runs lacked. Its headline finding on `v2_attention/best.pt`:

| Metric | Value |
|---|---|
| **CHAIR-lite `chair_image_rate`** | **0.7759** — 77.6% of images get ≥1 noun absent from all 5 references |
| **CHAIR-lite `hallucinated_noun_rate`** | **0.5011** — half of all generated nouns are unsupported |
| CIDEr-D (full split) | 0.5397 |
| ECE @ T=1 / fitted T* | 0.0119 / **1.0** (already calibrated) |
| mean p(END) at true END / mid-caption | 0.5782 / 0.0672 |

*Caveat: a noun absent from an image's 5 references is not necessarily absent from the image — 0.5011 is an upper bound.*

The END diagnostic is the sharpest clue: under teacher forcing the model clearly knows where to stop (0.5782 vs 0.0672, ~8.6×), yet free-running outputs are visibly truncated (`"woman in white dress is standing in front of the man in the"`). Classic **exposure bias**.

### 9.4 Confidence calibration

`fit_temperature.py` (62,515 teacher-forced pairs, 12-temperature grid) concludes **T\* = 1.0, ECE = 0.013 → no rescaling needed.** But the 10-bin reliability table reveals a hidden problem: the error is concentrated in the high-confidence region.

| Bin | n | avg conf | empirical acc | error |
|---|---|---|---|---|
| (0.4,0.5] | 3,455 | 0.4458 | 0.4263 | −0.0195 |
| (0.5,0.6] | 2,126 | 0.5457 | 0.5061 | −0.0396 |
| **(0.6,0.7]** | 1,979 | 0.6523 | 0.5841 | **−0.0682** |
| **(0.7,0.8]** | 2,986 | 0.7547 | 0.6849 | **−0.0698** |
| **(0.8,0.9]** | 2,511 | 0.8379 | 0.7845 | **−0.0534** |

A user shown "85% confidence" should expect ~78%. A single scalar temperature cannot fix this (the error's sign flips across the range) — it needs isotonic recalibration.

**Note:** `temperature.json` was fitted on the **V1** decoder, not the V2 model that serves traffic, and it is **never read by any serving module**.

### 9.5 Beam search sweep

`beam_sweep.py` — full val split, grid of width {3,5,7,10} × α {0.6,0.8,1.0,1.2} + greedy.

- **Optimum: width 10, α 1.2** (BLEU-4 0.1253) — best on all five metrics simultaneously, monotone in both width and α.
- **But beam-10 costs 35.9 ms/img vs 1.9 ms for greedy** — 18× latency for +30.7% relative BLEU-4. Width 7/α 1.0 is the knee.
- **The whole grid spans only 0.76 BLEU-4 points.** Greedy→beam is worth ~2.6 points; choosing among beam settings is worth <1.
- **The winning (10, 1.2) setting was never transferred to the V2 models**, which ship beam 5 / α 1.2 — rank 11 of 16 in this sweep.
- `anchor_samples.json` is a **regression guard**, not a results file: it dumps 12 captions only at (5, 0.6) so a human can verify the sweep reproduces the published 0.5334/0.3203/0.1890/0.1218 before trusting the novel recommendation.

---

## 10. Deployment

| Path | Mechanism |
|---|---|
| Backend | `backend/start.sh` (one-shot uvicorn) or `backend/run.sh` (supervisor: `while true`, 3 s restart, `/tmp/backend.log`, 5 MB truncation) |
| Frontend | `bun run dev` (:3000) or `next build` + `bun .next/standalone/server.js` |
| Gateway | `Caddyfile` — `:81`, `@transform_port_query` matcher → `localhost:{query.XTransformPort}`, else → `:3000` |

**Both shell scripts are POSIX-only** (`/tmp`, `pkill`, `wc`) and this checkout is Windows/OneDrive — the committed logs show the backend was launched ad hoc from `backend/.venv` instead.

**Deployment gap:** the platform entrypoint `.zscripts/start.sh` starts Next.js, mini-services, and Caddy — **it never starts the Python backend.** In a packaged deploy the API at :8000 would not exist.

---

## 11. Defects and risks

### Critical

| # | Issue |
|---|---|
| 1 | **All V2 work is uncommitted.** 41 untracked files + 15 modified = the entire research programme. One `git clean -fd` destroys it. |
| 2 | **`requirements.txt` is incomplete** — `pandas`, `tokenizers`, `open_clip_torch`, `nltk` are imported at module scope on the boot path but undeclared. A recorded `MemoryError` crash at startup traces to this chain. |

### High

| # | Issue | Location |
|---|---|---|
| 3 | **Landing page metrics are hardcoded and stale.** BLEU-1 shown as **0.53**; the served model scores **0.6559** (understated 19%). BLEU-2 0.32 vs 0.4344 (−26%). Also: mixes beam-w5 BLEU with beam-w10 latency (35.7 ms labelled "beam=5"; beam-5 is 15.9 ms), says "test split" when it is the **val** split, and the "Repetition Rate 2.8%" figure has **no source** in `experiments/`. | `performance.tsx:17-90` |
| 4 | **The entire landing page documents Gen-1** (DenseNet201 / 256-unit LSTM / TensorFlow / Keras tokenizer / greedy decoding) while the backend serves Gen-3 (CLIP / 512-unit LSTM / PyTorch / BPE / beam-5). `hero`, `features`, `architecture`, `how-it-works`, `model-section`, `footer` all contradict `model-info-view` and the backend. | 6 files |
| 5 | **Lock-free singleton + hot-swap.** `ModelManager.get_instance()` and `initialize_with_provider()` are unsynchronised; a `?provider=` request rebinds the model under in-flight inference. No semaphore guards the shared torch modules across up to 32 executor threads. | `manager.py:371-375, 451-503` |
| 6 | **Hallucination is unsolved** — 50.1% of generated nouns unsupported, 77.6% of images affected. |
| 7 | **Val loss has stopped being a useful selection signal** — the best-loss model (`_distill`, 3.6643) is the worst-BLEU model (0.4889). |

### Medium

| # | Issue |
|---|---|
| 8 | `/health` is registered twice; `main.py:85` **shadows** the schema-validated `routes.py:39` and reports `"TensorFlow n/a (torch)"`. |
| 9 | `semantic_score` / `semantic_pmi` are **structurally dead** — gated off by `ENABLE_BLIP_SCORE` default `"0"`, so they are always `null`. |
| 10 | `CUDA_VISIBLE_DEVICES=-1` is set process-wide at import, defeating `ATT_DEVICE=cuda`. |
| 11 | The auto-mode gate probes `experiments/v2_attention/best.pt` but the loader independently picks `v2_attention_clip_rl/best.pt` — gate and artifact can diverge. |
| 12 | `_beam_search` has **no hard step cap** (V1's `for _ in range(max_length)` did). |
| 13 | `ATT_CKPT` and all decoding constants are read at **import time**, not per request. |
| 14 | `/predict-batch` skips the 10 MB guard and is strictly serial (no `gather`). |
| 15 | **No request timeouts anywhere** in `api.ts` — a hung backend leaves the UI spinning. |
| 16 | History stores **50 full base64 dataURLs** in `localStorage`; quota failures are silently swallowed, so history can silently stop persisting. |
| 17 | `train_torch.py:45` — `WEIGHTS_DIR = PROJECT_DIR.parent / "weights"` resolves **outside the project**. |
| 18 | Preprocessing inconsistency: `precompute_all_spatial.py` uses `Resize(256)+CenterCrop(224)`; `precompute_spatial_features.py` uses `Resize((224,224))`. |

### Low / dead code

- `Prisma` + `src/lib/db.ts` + `prisma/schema.prisma` — zero importers, untouched starter schema.
- `src/app/api/route.ts` — 5-line "Hello, world!" stub.
- `use-toast.ts` (194 lines) — never imported; everything uses `sonner`.
- `utils/timer.py`, `getProviders`, `setProvider`, `predictBatch` — never called. Batch UI does not exist despite full typing.
- `ATT_HYBRID_THRESHOLD` — defined, never read. `blip_scorer` docstring says PMI centre 0.75; code uses 0.35.
- `train_attention_v2_tf.py:217` computes `ss` but never passes it → **the Gen-4 run had no scheduled sampling** despite the docstring.
- `typingEffectEnabled` toggle is a no-op (`? true : true`).
- `dashboard-navbar` "Model Online" badge is hardcoded green; the search input is non-functional.
- `use-caption` stage labels hardcode Gen-1 internals ("Running DenseNet201 encoder…", "256-dim").

---

## 12. Recommendations (priority order)

1. **Commit immediately.** At minimum: `git add -A && git commit` on the feature branch. Then split into two PRs — research artifacts (`experiments/`, `weights/`, `train*.py`, `eval*.py`) vs. serving/UI changes. Add `weights/*.npy`, `weights/blip/`, and `backend/.venv/` to `.gitignore` (3.9 GB + venv should never be staged).
2. **Fix `requirements.txt`** — declare `pandas`, `tokenizers`, `open_clip_torch`, `nltk`. Verify a clean venv boots.
3. **Re-sync the landing page with reality.** Either pull metrics live from `/model-info` or replace the hardcoded block with the Gen-3 numbers (BLEU-1 0.6559, BLEU-4 0.1727, ROUGE-L 0.2782, val split, beam-5). Fix the "test split" → "validation split" error and drop the unsourced 2.8% figure.
4. **Re-evaluate cleanly.** Fix the two artifact-provenance bugs in `eval_v2.py` / `eval_v2_clip.py` (output dir follows `--model`, not `--ckpt`), then re-run plain-CLIP CE eval and the transformer eval, and add CIDEr-D to every variant so the RL story is actually comparable.
5. **Add METEOR/SPICE/BERTScore** — their absence is the biggest gap against published captioning work.
6. **Concurrency:** add a lock around provider hot-swap and a semaphore around torch forwards, or set the executor to 1 worker for inference.
7. **Next research lever, ranked:** (a) the hallucination rate is the largest quality gap and CIDEr/BLEU cannot see it — CHAIR-aware or CLIPScore-based reward is the natural next GRPO iteration; (b) transfer the sweep's winning beam-10/α1.2 to the V2 models; (c) isotonic (not scalar) confidence recalibration, since the high-confidence region is 5–7 points over-confident.
8. **Either ship or delete the dead code** — Prisma, the batch UI, `use-toast`, `/api/route.ts`, `utils/timer.py`. They inflate install size and mislead readers.

---

## 13. Quick reference

| Question | Answer |
|---|---|
| What runs in production? | `AttentionProvider` → `experiments/v2_attention_clip_rl/best.pt` (CLIP ViT-B/16 patches + Bahdanau LSTM-512 + BPE-6k + GRPO) |
| Best measured metrics | BLEU-1 0.6559 · BLEU-4 0.1727 · ROUGE-L 0.2782 (beam-5, α1.2, 1,214 val images) |
| Most recent experiment | `v2_attention_clip_rl3` (2026-08-25 10:00) — wins subset CIDEr, loses BLEU-4, not promoted |
| Can I control beam size / temperature from the UI? | **No.** Only a 4-way provider picker. All decoding params are server-side env vars read at import. |
| Is the V1 Keras model still usable? | Yes — selectable at runtime via `?provider=notebook-tensorflow`; weights present. No longer the default. |
| Are attention heatmaps exposed? | **No.** `alpha` is computed inside `Bahdanau.context()` and discarded — it returns only the weighted sum. |
| Is BLIP a reranker? | No. It is (a) fallback captioner, (b) semantic PMI scorer, (c) OOD hand-off target. Reranking was tested and rejected. |
| Is there a database? | No. Prisma/SQLite is dead code. History is `localStorage`. |
