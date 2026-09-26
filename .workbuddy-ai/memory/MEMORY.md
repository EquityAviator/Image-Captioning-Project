# CaptionAI — Project Memory

## Project: Flickr8K image captioning (Next.js + FastAPI + PyTorch)

### Critical fact: branch state
- `main` and `feature/local-caption-improvement` **both point at commit `36783b3`**. `git diff main feature/...` is EMPTY.
- All V2 work is **uncommitted in the working tree**: 41 untracked paths + 15 modified tracked files.
- V1 = what git knows (Keras DenseNet201-GAP + LSTM-256). V2 = what actually runs (PyTorch CLIP + attention LSTM + GRPO).
- **Risk: commit before any `git clean`.**

### Model generations (chronological research log)
| Gen | Encoder | Decoder | BLEU-1 | BLEU-4 |
|---|---|---|---|---|
| 1 | DenseNet201 GAP 1920-d | LSTM-256, word vocab 8427 | 0.5334 | 0.1218 |
| 2 | DenseNet 7×7×1920 | Bahdanau LSTM-512 + BPE-6k + sched. sampling | 0.5649 | 0.1663 |
| 2-RL | ″ | + GRPO / CIDEr-D | 0.5971 | 0.1612 |
| **3-RL (PRODUCTION)** | **CLIP ViT-B/16 patches 196×768** | Bahdanau LSTM-512 + GRPO | **0.6559** | **0.1727** |
| 4 ✗ | ″ | Transformer 4L/8H (23.4M) | rejected | |

Metrics: full 1,214-image val split, beam-5, GNMT α=1.2. ROUGE-L prod = 0.2782.

### Key architecture facts
- Serving entry: `backend/main.py` → `api/routes.py` → `inference/manager.py` (ModelManager singleton, 4-tier provider chain: attention → notebook-tensorflow → BLIP → heuristic).
- Production checkpoint auto-selected by `attention_provider.py::_select_checkpoint()` — order `(_CLIP_RL, _CLIP, _RL, _CE)`, i.e. **`experiments/v2_attention_clip_rl/best.pt`**.
- Decoding params are **server-side env vars read at import time** (`ATT_BEAM_WIDTH=5`, `ATT_LENGTH_PENALTY=1.2`, `ATT_MIN_LEN=8`, `ATT_DBS_LAMBDA=0.7`). **No client control** — no request-body model exists.
- Attention maps are computed then **discarded** (`Bahdanau.context()` returns only the weighted sum). Not exposed.
- Frontend reaches backend via relative path + `?XTransformPort=8000` (Caddy :81); Next `rewrites()` is a redundant second path. No env var.
- Prisma/`src/lib/db.ts`, `src/app/api/route.ts`, `use-toast.ts`, `utils/timer.py` are **dead code**.

### Known unsolved problems
1. **Hallucination**: CHAIR-lite `hallucinated_noun_rate` 0.5011, `chair_image_rate` 0.7759.
2. **CIDEr reward hacking** → captions shrink (7.8→6.3 words). Shipped fix is decode-time `MIN_LENGTH=8` + trailing-word trimmer, not a training fix.
3. **Val loss is no longer a valid selection signal** — best-loss model (`clip_distill`, 3.6643) is worst-BLEU (0.4889).
4. **Landing page metrics are hardcoded and stale** (`performance.tsx`): shows BLEU-1 0.53 vs real 0.6559.

### Artifact provenance traps (verified)
- `experiments/v2_attention/eval_results.json` measures `v2_attention_rl/best.pt`, NOT `v2_attention` (eval_v2.py derives out_dir from `--model`, not `--ckpt`).
- `experiments/v2_attention_clip/eval_results.json` is byte-identical to `clip_rl`'s — the plain-CLIP CE eval was **overwritten and lost**.
- CLIP-CE headline numbers (0.6092/0.1755/0.2747) are **hardcoded** in `attention_provider.py:59,582`, no artifact backs them.
- METEOR / BERTScore / CLIPScore / SPICE are **never computed** anywhere in this repo.

### Environment
- Repo lives under OneDrive — eval scripts write to `%LOCALAPPDATA%` first because OneDrive + METEOR's Java subprocess deadlocked.
- `backend/requirements.txt` is INCOMPLETE: `pandas`, `tokenizers`, `open_clip_torch`, `nltk` are imported at module scope on the boot path but undeclared.
- `backend/.venv/` exists (Python 3.11). Hardware: GTX 1080, 8 GB VRAM.
- `weights/` is 3.9 GB (`clip_features_spatial.npy` 2.44 GB, `features_spatial.npy` 1.52 GB) — should be gitignored.

### Deliverables produced
- `docs/PROJECT_ANALYSIS.md` — full comprehensive analysis (2026-08-30).
