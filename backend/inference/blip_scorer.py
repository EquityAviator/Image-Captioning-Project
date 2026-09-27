"""
BLIP semantic-agreement scorer (PMI-based).

An independent vision-language judge (Salesforce/blip-image-captioning-base,
already on disk under weights/blip/) re-reads our generated caption:

    PMI(caption | image)  =  logP_BLIP(caption | image)
                           - logP_BLIP(caption | neutral-gray image)

The second term cancels the language-model prior, i.e. the reward for
writing a *bland, generic* sentence that fits ANY photo. What remains is
the evidence that THIS image specifically supports THIS caption.

Empirically validated on Flickr8K samples (backend/test_pmi_validation.py):
true image captions score PMI ≈ +0.5 … +1.9, captions from other images
≈ +0.3 … +0.7. Displayed score maps PMI through a fixed logistic so the
API returns something in [0, 1]:

    semantic_score = sigmoid((PMI - 0.75) / 0.45)

Loaded lazily on first use; failures are non-fatal (score simply omitted).
Set ENABLE_BLIP_SCORE=0 to disable (saves ~2 s CPU per request).
"""

from __future__ import annotations

import math
import os
import threading

import numpy as np
from PIL import Image

from utils.logger import get_logger
from utils.paths import WEIGHTS_DIR

log = get_logger(__name__)

_LOCAL_SNAPSHOT = WEIGHTS_DIR / "blip"
_FALLBACK_MODEL = "Salesforce/blip-image-captioning-base"

# logistic mapping constants — centred on the MEASURED distribution of
# generated captions (150 val images): median PMI +0.33, IQR [-0.07, +0.93].
# => typical output scores ~50%, strong image-evidence 80-95%, weak <30%.
_PMI_CENTER = 0.35
_PMI_SCALE = 0.45

_lock = threading.Lock()
_model = None
_processor = None
_device = None
_gray: Image.Image | None = None


def _load() -> bool:
    global _model, _processor, _device, _gray
    if _model is not None:
        return True
    with _lock:
        if _model is not None:
            return True
        try:
            import torch
            from transformers import BlipForConditionalGeneration, BlipProcessor

            src = str(_LOCAL_SNAPSHOT) if _LOCAL_SNAPSHOT.exists() else _FALLBACK_MODEL
            log.info("loading BLIP scorer from %s …", src)
            _processor = BlipProcessor.from_pretrained(src, local_files_only=True)
            _model = BlipForConditionalGeneration.from_pretrained(
                src, local_files_only=True)
            _device = torch.device("cpu")
            _model.eval().to(_device)
            _gray = Image.new("RGB", (224, 224), (128, 128, 128))
            log.info("BLIP scorer ready on %s", _device)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("BLIP scorer unavailable (%s) — semantic scores disabled", exc)
            return False


def enabled() -> bool:
    # Default OFF: BLIP adds ~1 GB RAM and slows every request. The caption
    # provider itself is fully self-contained. Opt in with ENABLE_BLIP_SCORE=1
    # when semantic_score/semantic_pmi are actually needed.
    return os.environ.get("ENABLE_BLIP_SCORE", "0") == "1"


def _cond_logp(image: Image.Image, caption: str) -> float:
    """Mean token log-probability of `caption` under BLIP given `image`."""
    import torch

    inputs = _processor(images=image, text=caption, return_tensors="pt",
                        padding=True).to(_device)
    with torch.no_grad():
        out = _model(**inputs)
    # shift: logits[t] predicts token t+1 (verified against HF labels loss)
    logits = out.logits[:, :-1, :]
    targets = inputs["input_ids"][:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(tok.mean())


def score(image: Image.Image, caption: str) -> tuple[float, float] | None:
    """Semantic agreement for `caption` given `image`.

    Returns (semantic_score, pmi):
      semantic_score in [0,1] — logistic-mapped PMI, for display
      pmi — raw log-evidence difference (image minus neutral baseline)

    Returns None when disabled or unavailable.
    """
    if not enabled() or not caption.strip():
        return None
    if not _load():
        return None
    try:
        img = image.convert("RGB")
        lp_img = _cond_logp(img, caption)
        lp_gray = _cond_logp(_gray, caption)
        pmi = lp_img - lp_gray
        sem = 1.0 / (1.0 + math.exp(-((pmi - _PMI_CENTER) / _PMI_SCALE)))
        return round(sem, 4), round(pmi, 4)
    except Exception as exc:  # noqa: BLE001
        log.warning("BLIP scoring failed: %s", exc)
        return None
