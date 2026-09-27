# CaptionAI — Setup & Bootstrap Requirements

**Version 1.1 (decisions locked by project owner, Sept 2026)** · Owner: Muhammad Hamza Mushtaq — Machine Learning Project
Companion to: `CaptionAI_Research_Report_v2.pdf` (technical record), `CaptionAI_Client_Summary.pdf`

## 0. Locked decisions (v1.1)

| # | Decision | Consequence |
|---|---|---|
| D1 | Setup is script-driven (`backend/setup.py`); requirements files are pip-consumed lists and **cannot prompt** — all choices/timeouts live in the script |
| D2 | Engine choice: ask CPU/GPU, **10 s timeout → CPU** (CPU is also the reference serving config: ~590 ms) |
| D3 | `tensorflow` **excluded from all default installs** — only the legacy Gen-1 "Notebook TF" comparison provider needs it (~600 MB); documented as optional add-on; requires the lazy-import fix (C1) if ever used |
| D4 | CLIP ViT-B/16 weights (352 MB): **default-on auto-download** at setup (library arrives via pip; weights fetched by script from the public hub, cached; same mechanism the app already uses at runtime) |
| D5 | BLIP (945 MB): **ask with 10 s timeout → default SKIP**. When skipped: router auto-disables, /health reports `blip_fallback:false`, UI badge shows "fallback not installed" — never a silent failure (requires code change C2) |
| D6 | Feature caches (3.7 GB `.npy`): **out of scope for setup entirely** — optional retraining material, provided by author later as Hugging Face links in README; nothing in requirements/script |
| D7 | One-file CPU/GPU switching rejected: torch CPU (`torch==2.5.1` @ PyPI) vs GPU (`+cu121` @ PyTorch index) need different pins AND indexes; declarative requirements have no logic → 3-file split (common + cpu + gpu), script composes them |
| D8 | **uv is the primary setup path** (`uv run --python 3.11 backend/setup.py` — installs its own pinned Python, ~3–10× faster, cache makes re-runs instant); plain `python backend/setup.py` remains fully supported (script auto-detects: uv-managed installs vs venv+pip fallback). Node.js stays the only manual prerequisite either way — setup checks and links. |

---

## 1. Purpose

Define exactly what a new machine must acquire and do to go from
`git clone` → **working captioning app**, including:

- all software libraries (pip / npm),
- all model assets (own champion weights, CLIP, BLIP) from their public sources,
- a **first-run choice prompt**: download BLIP from the public hub **or** use a local
  weights folder (or skip it entirely).

Guiding rule: **the repo carries only what is ours** (code, the 32 MB champion, the
tokenizer, documents — everything a clone needs). Anything that already exists
publicly (PyPI, npm, Hugging Face) is *fetched at setup*, never committed.

---

## 2. Target scenarios

| S | Scenario | Needs |
|---|---|---|
| A | Reviewer/visitor: clone, run, caption images (the common case) | libraries + CLIP + champion; **BLIP optional** |
| B | Developer on a GPU machine (this project's author setup) | same, GPU torch variant |
| C | Offline / air-gapped machine | pre-copied HF cache folder or local `weights/blip/` (scenario A works fully offline for in-domain captions once CLIP is cached) |

---

## 3. Prerequisites

**Recommended path (D8, uv-first):** only **uv** + **Node.js 20+**. uv downloads its own
pinned Python 3.11 — no Python install, no PATH risk, no version drift.

**Fallback path (plain Python):** the rows below apply.

| Item | Minimum | Recommended |
|---|---|---|
| OS | Windows 10/11 x64, Ubuntu 22.04+, macOS 13+ | — |
| Python | — (uv provides it) | 3.11.x if not using uv |
| Node.js | 20+ | 24.x (matches development) |
| RAM | 4 GB free | 8 GB (BLIP peak ≈ 1.2 GB + torch base ≈ 0.9 GB) |
| Disk (project after clone) | ~0.5 GB | — |
| Disk (downloads at setup) | 1.4 GB minimal / 4.4 GB full | 8 GB free headroom |
| Network | first run only | — |

---

## 4. Software dependency requirements

### 4.1 R-LIB-1 · Three-file requirements split (D7; fixes the cu121 bug)

Current `backend/requirements.txt` pins **`torch==2.5.1+cu121`** — that wheel exists
only on PyTorch's CUDA index, so a fresh `pip install` fails on any clean machine.
New layout (no torch/tf in the common file; all non-torch deps ~150 MB):

| File | Contents | Selected by |
|---|---|---|
| `requirements.txt` | fastapi, uvicorn, transformers, open_clip_torch, tokenizers, numpy, pandas, scikit-learn, Pillow, nltk, tqdm, pycocoevalcap | always |
| `requirements-cpu.txt` | `torch==2.5.1` + `torchvision==0.20.1` (PyPI CPU wheels) | default after 10 s timeout (D2) or `--cpu` |
| `requirements-gpu.txt` | `--extra-index-url https://download.pytorch.org/whl/cu121` + cu121 pins | `--gpu` or prompt answer 2 |
| `requirements-legacy.txt` | `tensorflow-cpu==2.21.0` | never by setup (D3); README note for the Gen-1 demo |

### 4.2 R-LIB-2 · Legacy TensorFlow provider becomes an optional extra

`tensorflow-cpu` ≈ **600 MB** and is used ONLY by the "Notebook TF (Gen-1)" provider
(comparison/education feature). Move to `backend/requirements-legacy.txt`. The TF
provider must degrade gracefully when absent (it already imports lazily in
`manager.py` — confirm + health endpoint reports it as unavailable, not broken).

### 4.3 R-LIB-3 · Node packages

`package-lock.json` is committed → fresh install is `npm ci` (deterministic, ~90 MB).
No global installs, no bun requirement (dev script uses plain `next dev`).

---

## 5. Model asset requirements

| # | Asset | Size | Source of truth | Where it lands | Committed to git? |
|---|---|---|---|---|---|
| M1 | Champion decoder `best.pt` | 32 MB | **ours** — GitHub repo | `experiments/v2_attention_clip_rl/` | ✅ YES (required) |
| M2 | BPE tokenizer + id maps | ~3 MB | **ours** — GitHub | `weights/bpe/`, `weights/*.json` | ✅ YES (required) — needs `.gitignore` un-ignore for the 2 runtime JSONs |
| M3 | CLIP ViT-B/16 (openai) | 352 MB | HF hub (public, existing) | HF cache `~/.cache/huggingface` | ❌ fetched |
| M4 | BLIP base caption model | ~945 MB | HF `Salesforce/blip-image-captioning-base` (public, existing) | local `weights/blip/` if user supplies it, else HF cache | ❌ fetched (only if chosen) |
| M5 | Feature caches `*.npy` | 3.7 GB | regenerable via committed scripts | `weights/` | ❌ never — training-only, irrelevant to scenarios A/B |

Facts verified in code (Sept 2026): serving reads M1–M4 only; `attention_provider.py`
never touches the `.npy` caches; `manager.py` BLIP loader currently **fails hard** if
`weights/blip/` is missing → R-BEH-2 below fixes that.

---

## 6. First-run bootstrap requirements

New committed script **`backend/setup.py`** — the ONLY setup entry point (D1); safe to
re-run. R-requirements:

- **R-BEH-0 · Non-interactive guard.** If no TTY (CI/pipe), or `--yes`: take ALL
  defaults with no waiting (CPU + CLIP + no-BLIP) and print what was chosen.
- **R-BEH-1 · Engine prompt (D2).**
  `Compute engine? [1] CPU (default, works everywhere, ~590 ms) [2] GPU (CUDA 12.1) — answering in 10 s; default: CPU`
  Then: create venv → `pip install -r requirements.txt` → + `requirements-cpu.txt` or
  `-gpu.txt` accordingly → `npm ci` in repo root.
- **R-BEH-2 · CLIP weights (D4, default-on, no prompt).**
  `open_clip` hub fetch of ViT-B/16 (openai, 352 MB) into the standard HF cache;
  skipped silently when already cached (idempotent). `--no-clip` / `CAPTIONAI_CLIP=skip`
  opts out (first caption then downloads lazily at runtime, as today).
- **R-BEH-3 · BLIP prompt (D5, default-SKIP after 10 s).**
  ```
  Optional general-fallback model (BLIP, ~945 MB) — used to auto-route
  cartoons/screenshots/food photos away from the specialist:
    [1] Download now from Hugging Face (Salesforce/blip-image-captioning-base)
    [2] Use local weights → enter folder path (e.g. weights/blip/)
    [3] Skip — in-domain captions only (default in 10 s)
  Choose 1/2/3:
  ```
  Answer 3/timeout → install proceeds; app behaviour per §6.1 (honest, disabled
  fallback — NOT an error). Flags/env: `--blip download|local:PATH|off`,
  `CAPTIONAI_BLIP=…` (env > flag > prompt > default-off).
- **R-BEH-4 · Code-level local-first loading (implements D5 for later toggles).**
  BLIP resolution order at runtime: local `weights/blip/` → HF hub id → absent
  (fallback disabled). Requires change C2.
- **R-BEH-5 · Verify.** Expected files present, one smoke caption through the real
  pipeline; prints engine, provider list, CLIP/BLIP status + cache locations. Non-zero
  exit on failure.
- **R-BEH-6 · Idempotent.** Second run: zero downloads, < 10 s to complete.
- **R-BEH-7 · Offline path.** Pre-copied HF cache or `weights/blip/` present +
  `CAPTIONAI_HF_OFFLINE=1` → no network needed.
- **R-BEH-8 · Out of scope (D6).** Feature caches (`weights/*.npy`, 3.7 GB):
  never fetched, referenced or checked by setup/requirements — README links only,
  for the retraining path.

### 6.1 What "skip BLIP" means (behaviour, not an error)

- Hybrid router auto-disables after first out-of-domain detection attempt; ALL
  requests served by champion (in-domain quality unchanged);
- `/health` reports `"blip_fallback": false`; UI badge shows
  `attention (fallback not installed)` on OOD images instead of `attention+blip`;
- Settings list greys out the `huggingface-blip` provider with a "not installed —
  run setup to add" hint.

---

## 7. Network/disk budget per scenario (post-D3: TF never in defaults)

| Path | Downloads | Time (typical 50 Mbps) |
|---|---|---|
| **Minimal (locked defaults: CPU torch, CLIP, no BLIP)** | ~1.4 GB | ~4 min |
| Standard (+ BLIP via prompt answer 1) | ~2.3 GB | ~7 min |
| GPU (CUDA torch instead of CPU build) | ~2.9 GB | ~9 min |
| Optional afterthought: Gen-1 legacy demo | +600 MB (`requirements-legacy.txt`) | +2 min |

---

## 8. Acceptance criteria (definition of done)

1. On a clean machine (git + Node 20 + either **uv** or Python 3.11): **`git clone` then
   `uv run --python 3.11 backend/setup.py`** (or `python backend/setup.py`) produces a running
   system (`http://127.0.0.1:8010/health` = ok) — no other manual commands.
2. Engine prompt defaults to **CPU after 10 s**; both [1] and [2] yield a working app;
   the installed torch variant matches the choice (`torch.__version__` / `torch.cuda.is_available()` reported at verify).
3. CLIP (M3) downloaded without being asked; present after run #1 so the first caption is warm-speed.
4. BLIP prompt defaults to **skip after 10 s**; answers 1/2/3 all produce a working app
   with §6.1 behaviour; skipped state is visible (health + badge), never a silent failure.
5. Headless run (`--yes`/no TTY): zero prompts, zero hangs, documented defaults, exit 0.
6. Second run: zero downloads, < 10 s total.
7. `pip install -r backend/requirements.txt` alone succeeds against default PyPI (cu121 bug gone);
   `tensorflow` importable-or-absent never crashes backend startup (lazy import per C1).
8. Air-gapped: pre-copied HF cache (or `weights/blip/`) + `CAPTIONAI_HF_OFFLINE=1` → setup
   verifies offline and the app serves.
9. Fresh-clone repo packed size ≤ 200 MB (code + champion M1 + tokenizer M2 + docs + dataset already in history); `setup.py` never references the 3.7 GB caches (D6).

---

## 9. Required changes derived from this document

| # | Change | File(s) | Effort |
|---|---|---|---|
| C1 | 3-file requirements split per §4.1 (common / cpu / gpu / legacy); remove torch+cu121 bug; tensorflow → legacy-only. NOTE: `manager.py` imports tensorflow at module top-level — wrap that provider's registration in try/except (import at call time) so absence never crashes startup | `backend/requirements*.txt`, `manager.py` | 25 min |
| C2 | BLIP hub-fallback load (local-first → `Salesforce/blip-image-captioning-base` → absent) in `manager.py` + `attention_provider.py` router fallback. (`blip_scorer.py` deliberately stays local-only: it's the optional semantic-score feature, already gated by `ENABLE_BLIP_SCORE`.) | `manager.py`, `attention_provider.py` | 25 min |
| C3 | New `setup.py`: uv-auto-detect path (D8) with venv+pip fallback, TTY guard, engine prompt (10 s→CPU), pip/npx installs, CLIP prefetch (default-on), BLIP prompt (10 s→skip), verify/smoke, flags+env precedence, idempotency | `backend/` | ~1 h |
| C4 | Un-ignore M2 runtime JSONs; commit M1; ignore everything else under `weights/`, `experiments/` | `.gitignore` | 10 min |
| C5 | Health/UI reporting of fallback availability (§6.1: `blip_fallback` flag + honest badge) | `main.py` health + prediction badge | 30 min |
| C6 | README quickstart rewrite: uv line + two commands total (fallback path documented too); §7 budgets; HF-links section reserved for D6 feature caches | `README.md` | 20 min |

No new third-party services required — downloads use PyPI, npm, and Hugging Face
public hubs exactly as CLIP already does today.
