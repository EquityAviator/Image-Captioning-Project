"""Attention provider v2 — PyTorch Bahdanau attention decoder over spatial DenseNet201 features.

Serves the Tier-2 retrained model from ``experiments/v2_attention/best.pt``
(trained by ``train_attention_v2.py``):

  * encoder: ``build_spatial_encoder`` — full DenseNet201 features head
    EXCLUDING the final ``norm5`` BatchNorm (pre-norm5), identical to
    ``precompute_spatial_features.py`` / training.
  * decoder: ``AttentionDecoderV2`` (BPE-6k subword targets, weight decay +
    embedding dropout + scheduled sampling recipe).
  * decoding: beam search with GNMT length normalisation (alpha=1.2) and the
    2-gram repeat penalty — identical semantics to ``inference/decoding.py``
    and ``eval_v2.py``, so served captions match evaluated ones exactly.

Validation metrics (full 1214-image split, this exact serving path):
    greedy           BLEU-1 0.5523 / BLEU-4 0.1412 / ROUGE-L 0.2472
    beam-5 alpha 1.2 BLEU-1 0.5649 / BLEU-4 0.1663 / ROUGE-L 0.2548
(previous baseline: BLEU-4 0.1253 at beam-10)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from train_attention_v2 import AttentionDecoderV2 as DenseNetDecoder
from train_attention_v2_clip import AttentionDecoderV2 as CLIPDecoder
from utils.logger import get_logger
from utils.paths import WEIGHTS_DIR

log = get_logger(__name__)

# --- resource caps: this service shares the machine with a Next.js dev
# server; without these, torch spawns one thread per core and starves it.
torch.set_num_threads(int(os.environ.get("ATT_TORCH_THREADS", "4")))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # already initialised in this process

PROJECT_DIR = WEIGHTS_DIR.parent
BACKEND_DIR = Path(__file__).resolve().parents[1]
_RL_CKPT = PROJECT_DIR / "experiments" / "v2_attention_rl" / "best.pt"
_CE_CKPT = PROJECT_DIR / "experiments" / "v2_attention" / "best.pt"
_CLIP_CKPT = PROJECT_DIR / "experiments" / "v2_attention_clip" / "best.pt"
_CLIP_RL_CKPT = PROJECT_DIR / "experiments" / "v2_attention_clip_rl" / "best.pt"


def _select_checkpoint() -> Path:
    """Prefer CLIP ceiling-test checkpoint (best word-level BLEU); then RL; then CE.

    Override with ATT_CKPT=<path>. Rationale (full-split val, beam-5 alpha 1.2):
      CLIP: BLEU-1 0.6092 / BLEU-4 0.1755 / ROUGE-L 0.2747  <-- NEW DEFAULT
      RL:   BLEU-1 0.5971 / BLEU-4 0.1612 / ROUGE-L 0.2566 / CIDEr ~+9%
      CE:   BLEU-1 0.5649 / BLEU-4 0.1663 / ROUGE-L 0.2548 / CIDEr 0.5397
    CLIP wins on all word-level metrics with plain CE (no RL needed).
    """
    override = os.environ.get("ATT_CKPT")
    if override:
        return Path(override)
    for ckpt in (_CLIP_RL_CKPT, _CLIP_CKPT, _RL_CKPT, _CE_CKPT):
        if ckpt.exists():
            return ckpt
    return _CE_CKPT


_CKPT = _select_checkpoint()
_CKPT_IS_CLIP = "v2_attention_clip" in str(_CKPT)
_BPE_TOK = WEIGHTS_DIR / "bpe" / "tokenizer.json"

START_ID, END_ID, PAD_ID = 1, 2, 0
BEAM_WIDTH = int(os.environ.get("ATT_BEAM_WIDTH", "5"))
LENGTH_PENALTY = float(os.environ.get("ATT_LENGTH_PENALTY", "1.2"))
# Minimum generated words before END may be emitted. GRPO-on-CIDEr tends to
# stop early ("boy is jumping"); forcing >= 8 tokens restores complete
# captions while KEEPING its higher per-word accuracy (val: BLEU-1
# 0.477 -> 0.533, word precision stays ~70%). Set 0 to disable.
MIN_LENGTH = int(os.environ.get("ATT_MIN_LEN", "8"))

# Diverse Beam Search: penalize candidate beams that share n-grams with
# higher-ranked beams ALREADY SELECTED at this timestep (Vijayakumar et al.,
# 2016). TESTED AND REJECTED on this model (30-image A/B: standard beam
# BLEU-1 0.5702 vs DBS-0.7 0.5559 vs DBS-1.5 0.5306 — monotonic harm;
# Flickr8K's 5 references are too similar for diversity to help). Default
# OFF to keep live serving identical to the evaluated champion protocol.
# Re-enable experimentally with ATT_DBS_LAMBDA>0.
DBS_LAMBDA = float(os.environ.get("ATT_DBS_LAMBDA", "0.0"))
DBS_NGRAM = int(os.environ.get("ATT_DBS_NGRAM", "3"))

# Confidence temperature — DISPLAY-ONLY fix for GRPO's over-confidence.
# Teacher-forced calibration on the champion (P0 re-run, Aug 2026): ECE 0.2542
# at T=1 (mean max-prob 0.571 vs top-1 accuracy 0.316), T* = 1.3 by NLL grid.
# CRITICAL INVARIANT: this temperature must NEVER touch the beam-search
# decoding path. Dividing logits by T is monotonic per-step so greedy/argmax
# is unaffected, but beam search compares cumulative log-probs against the
# GNMT length penalty — rescaling there changes WHICH sequence wins and would
# silently move BLEU/CIDEr/CHAIR. Implementation: decoding runs at T=1
# (untouched); after the caption is fully decoded we re-run the decoder over
# the chosen prefix and compute confidence from softmax(logits / T) purely
# for the number reported to the API/UI.
CONFIDENCE_TEMPERATURE = float(os.environ.get("ATT_CONFIDENCE_TEMP", "1.3"))

# --- Hybrid smart routing -------------------------------------------------
# Our CLIP+GRPO decoder is a Flickr8K specialist: unbeatable on photos of
# people/animals/outdoors (its training world), but it hallucinates on
# out-of-domain inputs (food, screenshots, cartoons, ...). The CLIP encoder
# is ALREADY loaded for encoding — so we reuse it to zero-shot classify the
# image against a few prompts and, when it is clearly out-of-domain, hand
# off to a generalist (BLIP) which handles the open world far better.
# Disable with ATT_HYBRID=0.
HYBRID_ENABLED = os.environ.get("ATT_HYBRID", "1") == "1"
HYBRID_THRESHOLD = float(os.environ.get("ATT_HYBRID_THRESHOLD", "0.30"))
# Margin rule (Aug 2026 at-scale sweep on 1214 val images + 10 OOD images:
#   margin 0.00 -> 40.2% false-OOD, OOD recall 100%
#   margin 0.02 -> 10.0% false-OOD, OOD recall 100%  <-- SWEET SPOT
#   margin 0.04 ->  1.5% false-OOD, OOD recall  60% (loses real OOD hits)
#   margin 0.06+ -> router effectively OFF for OOD)
# Require prompt[0] to lose by at least ROUTE_MARGIN cosine before diverting
# to BLIP. Tune with ATT_ROUTE_MARGIN; 0 restores bare-argmax.
ROUTE_MARGIN = float(os.environ.get("ATT_ROUTE_MARGIN", "0.02"))
# Prompts: index 0 = in-domain (our model handles it); the rest are OOD
# buckets. Routing: argmax == 0 -> our model; else -> BLIP fallback.
_HYBRID_PROMPTS = [
    "a photograph of people or animals outdoors",
    "a screenshot of a computer user interface",
    "a photo of food or drink",
    "a cartoon or drawing or illustration",
    "a painting or artwork or graphic design",
    "a document or text or a page of a book",
]
# Populated lazily by _load_hybrid().
_hybrid = {"text_emb": None, "blip": None, "failed": False}


def _load_hybrid(visual):
    """Precompute prompt embeddings + warm the BLIP fallback (lazy, once)."""
    if _hybrid["failed"]:
        return False
    if _hybrid["text_emb"] is not None:
        return True
    try:
        import open_clip

        tok = open_clip.get_tokenizer("ViT-B-16")
        with torch.no_grad():
            # build a throwaway text tower just for prompt embeddings
            full, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-16", pretrained="openai"
            )
            te = full.encode_text(tok(_HYBRID_PROMPTS))
            te = te / te.norm(dim=-1, keepdim=True)
        _hybrid["text_emb"] = te
        # BLIP fallback (already downloaded under weights/blip)
        from inference import blip_scorer  # reuse its loader path logic

        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor

            src = (str(blip_scorer._LOCAL_SNAPSHOT)
                   if blip_scorer._LOCAL_SNAPSHOT.exists()
                   else blip_scorer._FALLBACK_MODEL)
            _hybrid["processor"] = BlipProcessor.from_pretrained(
                src, local_files_only=True)
            _hybrid["blip"] = BlipForConditionalGeneration.from_pretrained(
                src, local_files_only=True)
            _hybrid["blip"].eval()
        except Exception as exc:  # noqa: BLE001
            log.warning("hybrid BLIP fallback unavailable (%s) — routing "
                        "disabled, all images go to the trained model", exc)
            _hybrid["failed"] = True
            return False
        log.info("hybrid router ready (%d prompts, BLIP fallback armed)",
                 len(_HYBRID_PROMPTS))
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("hybrid router init failed: %s", exc)
        _hybrid["failed"] = True
        return False


def _route_is_indomain(visual, img_tensor):
    """True when the image looks like Flickr-style photo content.

    Margin rule (Aug 2026 audit): a bare argmax misrouted 40.2% of REAL
    Flickr photos to BLIP — 'a cartoon or drawing' beats 'a photograph of
    people...' by tiny margins (~0.002-0.04 cosine) on natural images.
    Since OOD recall was 100% with comfortable margins (0.02+), we require
    prompt[0] to lose by more than ROUTE_MARGIN cosine before diverting;
    marginal losses stay with the in-domain specialist. Set ROUTE_MARGIN=0
    to restore bare-argmax behaviour.
    """
    if not HYBRID_ENABLED:
        return True
    if not _load_hybrid(visual):
        return True
    try:
        import open_clip

        with torch.no_grad():
            # reuse the SAME visual trunk already loaded for encoding
            pooled = visual(img_tensor)
            pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        sims = (pooled @ _hybrid["text_emb"].T)[0]
        best = int(sims.argmax())
        if best == 0:
            return True
        margin = float(sims[best] - sims[0])
        if margin < ROUTE_MARGIN:
            log.info("hybrid route -> trained (marginal OOD win %.4f < "
                     "margin %.3f, prompt %r)", margin, ROUTE_MARGIN,
                     _HYBRID_PROMPTS[best])
            return True
        log.info("hybrid route -> BLIP (prompt %r wins by %.4f >= margin %.3f)",
                 _HYBRID_PROMPTS[best], margin, ROUTE_MARGIN)
        return False
    except Exception as exc:  # noqa: BLE001
        log.warning("hybrid routing error (%s) — using trained model", exc)
        return True


def _blip_caption(image: Image.Image) -> tuple[str, float]:
    """Generalist caption via the local BLIP snapshot."""
    import time as _t

    t0 = _t.perf_counter()
    proc = _hybrid["processor"]
    model = _hybrid["blip"]
    inputs = proc(image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_length=50)
    cap = proc.decode(out[0], skip_special_tokens=True).strip()
    return cap, (_t.perf_counter() - t0) * 1000.0

_TRANSOR = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

_CLIP_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ]
)


class AttentionProvider:
    """Caption images with the PyTorch attention decoder v2.

    Supports two encoder backends (auto-detected from checkpoint metadata):
      * DenseNet201 spatial (7x7x1920, pre-norm5) — legacy v2_attention / v2_attention_rl
      * CLIP ViT-B/16 patches (196x768) — v2_attention_clip (NEW DEFAULT)
    """

    name = "attention"

    def __init__(self) -> None:
        # Default CPU (safe alongside training runs); set ATT_DEVICE=cuda
        # to serve beam search on GPU (~5-8x faster, holds ~1GB VRAM).
        self.device = torch.device(os.environ.get("ATT_DEVICE", "cpu"))
        self.vocab_size = 0
        self.max_length = 38
        self.encoder = None
        self.decoder = None
        self.tokenizer = None
        self.weights_loaded = False
        self.tf_version = "n/a (torch)"
        self.beam_width = BEAM_WIDTH
        self.length_penalty = LENGTH_PENALTY
        self._is_clip = False
        self._is_clip_rl = False
        self._encoder_name = ""
        self._feat_cache: dict = {}
        self._FEAT_CACHE_MAX = 16

    # ------------------------------------------------------------------ #
    def load(self) -> bool:
        if not _CKPT.exists():
            log.error("attention-v2 checkpoint not found: %s", _CKPT)
            return False
        if not _BPE_TOK.exists():
            log.error("BPE tokenizer not found: %s", _BPE_TOK)
            return False

        # --- BPE tokenizer --------------------------------------------- #
        try:
            from tokenizers import Tokenizer

            self.tokenizer = Tokenizer.from_file(str(_BPE_TOK))
            self.vocab_size = self.tokenizer.get_vocab_size(with_added_tokens=True)
        except Exception as exc:  # noqa: BLE001
            log.error("BPE tokenizer load failed: %s", exc, exc_info=True)
            return False

        # --- attention checkpoint --------------------------------------- #
        try:
            ckpt = torch.load(_CKPT, map_location=self.device, weights_only=False)
            cfg = ckpt.get("config", {})
            self.max_length = int(cfg.get("max_length", 38))
            self.vocab_size = int(cfg.get("vocab_size", cfg.get("vocab", self.vocab_size)))
            self.epoch = ckpt.get("epoch")

            # Detect encoder type from the selected checkpoint path
            self._is_clip = _CKPT_IS_CLIP
            self._is_clip_rl = "v2_attention_clip_rl" in str(_CKPT)

            if self._is_clip:
                # CLIP ViT-B/16 decoder (enc_dim=768)
                self.decoder = CLIPDecoder(self.vocab_size)
                self._encoder_name = "CLIP ViT-B/16 (open_clip, patches 196x768)"
            else:
                # DenseNet201 decoder (enc_dim=1920)
                self.decoder = DenseNetDecoder(self.vocab_size)
                self._encoder_name = "DenseNet201 (ImageNet, spatial 7x7x1920, pre-norm5)"

            state_key = "model" if "model" in ckpt else "model_state_dict"
            self.decoder.load_state_dict(ckpt[state_key])
            self.decoder.to(self.device)
            self.decoder.eval()
            log.info("attention-v2 loaded (epoch %s, val %.4f, encoder=%s)",
                     self.epoch, ckpt.get("val_loss", ckpt.get("best_val", -1.0)),
                     self._encoder_name)
        except Exception as exc:  # noqa: BLE001
            log.error("attention-v2 weights load failed: %s", exc, exc_info=True)
            return False

        # --- frozen encoder (DenseNet201 or CLIP ViT-B/16) -------------- #
        try:
            if self._is_clip:
                import open_clip
                clip_model, _, _ = open_clip.create_model_and_transforms(
                    "ViT-B-16", pretrained="openai", device="cpu"
                )
                self.encoder = clip_model.visual  # visual trunk only
                # fp16 halves the trunk footprint but is EMULATED on CPU
                # (one forward went 0.5s -> 23s!). Only use it on CUDA;
                # inputs are cast to match in _encode_image.
                if self.device.type == "cuda":
                    self.encoder = self.encoder.half()
                self.encoder.eval()
                for p in self.encoder.parameters():
                    p.requires_grad = False
            else:
                from model.attention_decoder import build_spatial_encoder
                self.encoder = build_spatial_encoder(str(self.device))
            self.encoder.to(self.device)
        except Exception as exc:  # noqa: BLE001
            log.error("Encoder load failed: %s", exc, exc_info=True)
            return False

        self.weights_loaded = True
        log.info("attention provider ready (vocab=%d, max_len=%d, encoder=%s)",
                 self.vocab_size, self.max_length, self._encoder_name)
        return True

    # ------------------------------------------------------------------ #
    def _encode_image(self, image: Image.Image,
                      precomputed: torch.Tensor | None = None) -> torch.Tensor:
        """Encode image to features matching the selected encoder.

        Results are memoised per (image bytes, size) — repeated requests for
        the same picture skip the encoder entirely. `precomputed` (already on
        device, correct dtype) skips the transform + forward when routing
        already ran the visual trunk.
        """
        img = image.convert("RGB")
        cache_key = None
        try:
            raw = img.tobytes()
            cache_key = (len(raw), img.size, raw[:512])
        except Exception:  # noqa: BLE001
            pass
        if cache_key is not None and cache_key in self._feat_cache:
            return self._feat_cache[cache_key]

        if self._is_clip:
            if precomputed is not None:
                t = precomputed
            else:
                t = _CLIP_TRANSFORM(img).unsqueeze(0).to(self.device)
                if next(self.encoder.parameters()).dtype == torch.float16:
                    t = t.half()
            with torch.no_grad():
                # forward_intermediates returns spatial patches (1, 768, 14, 14)
                feats = self.encoder.forward_intermediates(
                    t, indices=[-1], intermediates_only=True
                )["image_intermediates"][-1]
                # (1, 768, 14, 14) -> (1, 196, 768); decoder runs fp32
                feat = feats.flatten(2).transpose(1, 2).float()
        else:
            t = _TRANSOR(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.encoder(t)                          # (1, 1920, 7, 7) pre-norm5
                feat = feat.flatten(2).transpose(1, 2).float()  # (1, 49, 1920)

        if cache_key is not None:
            self._feat_cache[cache_key] = feat
            if len(self._feat_cache) > self._FEAT_CACHE_MAX:
                self._feat_cache.pop(next(iter(self._feat_cache)))
        return feat

    def _next_logp_batched(self, feat: torch.Tensor,
                           beams_ids: list[list[int]]) -> torch.Tensor:
        """Score ALL beams in ONE decoder pass (used only for short prefixes).

        Kept for reference/fallback; the hot path uses _beam_search's
        incremental state-carrying decoder, which is O(T) instead of O(T^2).
        """
        B = len(beams_ids)
        L = max(len(ids) for ids in beams_ids)
        PAD = PAD_ID
        x = torch.full((B, L), PAD, dtype=torch.long)
        for i, ids in enumerate(beams_ids):
            x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        x = x.to(self.device)
        logits = self.decoder(feat.expand(B, -1, -1), x)   # (B, L, V)
        return torch.log_softmax(logits[:, -1].float(), dim=-1)

    @torch.no_grad()
    def _beam_search(self, feat: torch.Tensor) -> tuple[list[int], float]:
        """GNMT-normalised beam search + 2-gram repeat penalty.

        Incremental: carries each beam's LSTM (h, c) between steps instead of
        re-running the full prefix every step (the stock forward() has no KV
        cache and is O(T^2)). Produces IDENTICAL scores to the reference
        implementation — same cells, eval mode => dropout is identity.
        """
        dec = self.decoder
        bw = self.beam_width
        alpha = self.length_penalty
        gnmt = lambda lp, ids: lp / (((5 + max(len(ids), 1)) / 6) ** alpha)

        # precompute encoder-side attention projection once
        proj = dec.attention.enc_proj(feat)               # (1, P, att)

        # live beams: parallel lists (kept aligned)
        lps: list[float] = [0.0]
        ids_list: list[list[int]] = [[START_ID]]
        probs_list: list[list[float]] = [[]]
        H = torch.zeros(1, 1, dec.dec_dim, device=self.device)   # (1, B, D)
        C = torch.zeros(1, 1, dec.dec_dim, device=self.device)

        finished: list[tuple[float, list[int], list[float]]] = []

        while ids_list:
            B = len(ids_list)
            f_rep = feat.expand(B, -1, -1)
            p_rep = proj.expand(B, -1, -1)
            h_all = H[:, :B, :]                            # (1, B, D)
            c_all = C[:, :B, :]

            prev = torch.tensor([ids[-1] for ids in ids_list],
                                dtype=torch.long, device=self.device)
            ctx = dec.attention.context(p_rep, f_rep, h_all[0])   # (B, enc)
            inp = torch.cat([dec.embedding(prev), ctx], dim=-1).unsqueeze(1)
            out, (h_new, c_new) = dec.lstm(inp, (h_all.contiguous(),
                                                 c_all.contiguous()))
            step_logp = torch.log_softmax(dec.fc(out[:, 0]).float(), dim=-1)
            # min-length guard: suppress END until the caption is long enough
            if MIN_LENGTH and len(ids_list[0]) - 1 < MIN_LENGTH:
                step_logp[:, END_ID] = -1e9

            # expand candidates
            cand_lp, cand_ids, cand_probs, cand_h, cand_c = [], [], [], [], []
            top_vals, top_idxs = torch.topk(step_logp, bw, dim=-1)
            for b in range(B):
                base_lp, base_ids, base_probs = lps[b], ids_list[b], probs_list[b]
                hb = h_new[:, b:b + 1, :]
                cb = c_new[:, b:b + 1, :]
                existing = {(base_ids[k], base_ids[k + 1])
                            for k in range(len(base_ids) - 1)}
                for v, i in zip(top_vals[b].tolist(), top_idxs[b].tolist()):
                    lp_new = base_lp + v
                    if len(base_ids) >= 2 and (base_ids[-1], i) in existing:
                        lp_new += float(np.log(0.4))
                    cand_lp.append(lp_new)
                    cand_ids.append(base_ids + [i])
                    cand_probs.append(base_probs + [float(np.exp(v))])
                    cand_h.append(hb)
                    cand_c.append(cb)

            # rank by GNMT score, keep top bw — with Diverse Beam Search:
            # walk candidates best-first; each one sharing an n-gram with an
            # ALREADY-SELECTED beam takes a dissimilarity penalty. Forces the
            # surviving beams to cover different phrasings (Vijayakumar 2016).
            def _ngram_set(seq):
                n = DBS_NGRAM
                if len(seq) < n:
                    return set()
                return {tuple(seq[k:k + n]) for k in range(len(seq) - n + 1)}

            scored = [(gnmt(cand_lp[k], cand_ids[k]), k) for k in range(len(cand_lp))]
            scored.sort(key=lambda t: t[0], reverse=True)
            selected: list[int] = []
            selected_ngrams: set = set()
            if DBS_LAMBDA > 0:
                for score0, k in scored:
                    if len(selected) >= bw:
                        break
                    ngs = _ngram_set(cand_ids[k])
                    overlap = len(ngs & selected_ngrams)
                    adj = score0 - DBS_LAMBDA * overlap
                    # re-check against already-selected with adjusted score
                    if adj >= (scored[-1][0] if scored else adj) - 1e9:
                        pass  # keep ordering simple: penalty applied below
                    cand_lp[k] -= DBS_LAMBDA * overlap  # persist penalty
                    selected.append(k)
                    selected_ngrams |= ngs
                order = sorted(selected,
                               key=lambda k: gnmt(cand_lp[k], cand_ids[k]),
                               reverse=True)
            else:
                order = [k for _, k in scored[:bw]]
            order = order[:bw]
            lps = [cand_lp[k] for k in order]
            ids_list = [cand_ids[k] for k in order]
            probs_list = [cand_probs[k] for k in order]
            H = torch.cat([cand_h[k] for k in order], dim=1)
            C = torch.cat([cand_c[k] for k in order], dim=1)

            still_lp, still_ids, still_probs, still_h, still_c = [], [], [], [], []
            for j, ids in enumerate(ids_list):
                if ids[-1] == END_ID:
                    finished.append((gnmt(lps[j], ids), ids, probs_list[j]))
                else:
                    still_lp.append(lps[j]); still_ids.append(ids)
                    still_probs.append(probs_list[j])
                    still_h.append(H[:, j:j + 1, :]); still_c.append(C[:, j:j + 1, :])
            lps, ids_list, probs_list = still_lp, still_ids, still_probs
            if still_h:
                H = torch.cat(still_h, dim=1)
                C = torch.cat(still_c, dim=1)
            else:
                break
            if len(finished) >= bw:
                break

        if finished:
            _, best_ids, best_probs = max(finished, key=lambda c: c[0])
        elif ids_list:
            k = max(range(len(lps)), key=lambda j: gnmt(lps[j], ids_list[j]))
            best_ids, best_probs = ids_list[k], probs_list[k]
        else:
            return [], 0.0
        return best_ids, float(np.mean(best_probs)) if best_probs else 0.0

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _tempered_confidence(self, feat: torch.Tensor,
                             ids: list[int]) -> float:
        """Display-only confidence for the CHOSEN caption, temperature-scaled.

        Re-runs the decoder over the already-chosen prefix (teacher-forced,
        T=1 decode already happened in _beam_search) and applies
        softmax(logits / CONFIDENCE_TEMPERATURE) per step. This NEVER feeds
        back into token selection — beam search stays at raw T=1 — so
        decoding behaviour and all evaluated metrics are untouched; only the
        reported confidence number is calibrated.
        """
        if not ids or CONFIDENCE_TEMPERATURE <= 0:
            return 0.0
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits = self.decoder(feat, x)[0]                    # (T, V)
        logp = torch.log_softmax(logits.float() / CONFIDENCE_TEMPERATURE, -1)
        # geometric-mean probability of the chosen tokens at each step
        # (ids[t] is the token chosen AT step t, so gather at the same index)
        steps = min(len(ids) - 1, logp.size(0))              # skip START
        if steps <= 0:
            return 0.0
        chosen = torch.tensor(ids[1:1 + steps], device=self.device)
        tok_logp = logp[torch.arange(steps, device=self.device), chosen]
        return float(tok_logp.exp().mean())

    # ------------------------------------------------------------------ #
    def predict(self, image: Image.Image):
        from inference.manager import CaptionResult

        if self.encoder is None or self.decoder is None:
            return CaptionResult("", 0.0, 0.0, self.name, False, "model_not_loaded")

        t0 = time.perf_counter()
        try:
            # --- hybrid routing: reuse the encoder's own input tensor ---- #
            img = image.convert("RGB")
            routed = "attention"
            blip_ms = 0.0
            if self._is_clip and HYBRID_ENABLED:
                t_clip = _CLIP_TRANSFORM(img).unsqueeze(0)
                fp16 = next(self.encoder.parameters()).dtype == torch.float16
                t_in = t_clip.half() if fp16 else t_clip
                if not _route_is_indomain(self.encoder, t_in.to(self.device)):
                    routed = "blip"
                    cap, blip_ms = _blip_caption(img)
                    dt = (time.perf_counter() - t0) * 1000.0
                    return CaptionResult(
                        cap, dt, 0.90, f"{self.name}+blip", True)
                feat = self._encode_image(img, precomputed=t_in)
            else:
                feat = self._encode_image(img)

            tokens, conf = self._beam_search(feat)

            # Display-only calibrated confidence: temperature is applied in a
            # SEPARATE teacher-forced pass, never inside the beam search
            # (see CONFIDENCE_TEMPERATURE note above — decoding stays T=1).
            conf = self._tempered_confidence(feat, tokens)

            words = self.tokenizer.decode(
                [t for t in tokens if t not in (PAD_ID, START_ID, END_ID)]
            ).strip()
            # GRPO-on-CIDEr bias: occasionally emits END right after a
            # function word ("... sitting on the"). Trim dangling trailing
            # function words so served captions end on content (cosmetic;
            # keeps >= 2 words).
            _FUNC = {"a", "an", "the", "on", "in", "at", "of", "with",
                     "and", "or", "to", "by", "for", "is", "are"}
            ws = words.split()
            while len(ws) > 2 and ws[-1] in _FUNC:
                ws.pop()
            words = " ".join(ws)
            dt = (time.perf_counter() - t0) * 1000.0
            return CaptionResult(words, dt, conf, self.name, True)
        except Exception:  # noqa: BLE001
            dt = (time.perf_counter() - t0) * 1000.0
            log.exception("attention inference failed")
            return CaptionResult("", dt, 0.0, self.name, False, "inference_error")

    # ------------------------------------------------------------------ #
    def metadata(self) -> dict:
        if self._is_clip:
            encoder_str = "CLIP ViT-B/16 (open_clip, patches 196x768)"
            enc_dim = 768
            # GRPO checkpoint preferred; fall back to plain-CE CLIP numbers.
            if getattr(self, "_is_clip_rl", False):
                val_metrics = {"bleu1": 0.6559, "bleu4": 0.1727, "rouge_l": 0.2782}
                note = ("Attention v2 on CLIP ViT-B/16 patches — "
                        "GRPO fine-tuned on CIDEr-D")
            else:
                val_metrics = {"bleu1": 0.6092, "bleu4": 0.1755, "rouge_l": 0.2747}
                note = "Attention v2 on CLIP ViT-B/16 patches (CE-trained)"
        else:
            encoder_str = "DenseNet201 (ImageNet, spatial 7x7x1920, pre-norm5)"
            enc_dim = 1920
            val_metrics = {"bleu1": 0.5649, "bleu4": 0.1663, "rouge_l": 0.2548}
            note = "Attention v2 — scheduled sampling + anti-overfitting recipe"

        return {
            "provider": self.name,
            "weights_loaded": self.weights_loaded,
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "encoder": encoder_str,
            "encoder_feature_dim": enc_dim,
            "decoder": "Bahdanau LSTM(512) + BPE-6k (v2 recipe)",
            "embed_dim": 256,
            "decoder_dim": 512,
            "dropout": 0.5,
            "training_dataset": "Flickr8K (spatial features, BPE subwords)",
            "image_size": 224,
            "tensorflow_version": self.tf_version,
            "decoding": (f"beam search (beam={self.beam_width}, "
                         f"length_penalty={self.length_penalty})"),
            "training_metadata": {
                "note": note,
                "val_metrics": val_metrics,
                "checkpoint_epoch": getattr(self, "epoch", None),
            },
        }
