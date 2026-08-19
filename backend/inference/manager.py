"""
Model manager — the heart of the inference service.

Responsibilities
----------------
1. Lazily build the **exact** TensorFlow architecture from the notebook.
2. Try to load saved weights from `weights/` (model.h5 + tokenizer.pkl + metadata.json).
3. If weights aren't available, transparently fall back to a HuggingFace
   image-captioning model so the demo still works end-to-end.
4. Expose a single `predict(image_bytes) -> CaptionResult` interface so the
   FastAPI layer never has to know which provider is active.

The provider is chosen on startup and never changes at runtime — both
providers speak the same interface, so the rest of the codebase doesn't care.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image

from model.architecture import (
    DEFAULT_IMG_SIZE,
    build_decoder,
    build_encoder,
)
from utils.logger import get_logger
from utils.paths import (
    METADATA_JSON,
    MODEL_H5,
    MODEL_KERAS,
    TOKENIZER_JSON,
    TOKENIZER_PKL,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class CaptionResult:
    caption: str
    inference_time_ms: float
    confidence: float
    provider: str
    success: bool = True
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------
class _BaseProvider:
    """Common interface implemented by both providers."""

    name: str = "base"

    def predict(self, image: Image.Image) -> CaptionResult:  # pragma: no cover
        raise NotImplementedError

    def metadata(self) -> dict:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Notebook provider — TensorFlow DenseNet201 + LSTM (exact replica)
# ---------------------------------------------------------------------------
class NotebookProvider(_BaseProvider):
    """
    Loads the EXACT architecture from the notebook and (optionally) the
    trained weights. If weights aren't found, the model is initialised with
    random weights — predictions will be syntactically valid but semantically
    nonsensical, which is documented in the README.
    """

    name = "notebook-tensorflow"

    def __init__(self) -> None:
        import tensorflow as tf  # noqa
        self._tf = tf
        self.tf_version = tf.__version__
        self.encoder = None
        self.decoder = None
        self.tokenizer = None
        self.vocab_size: int = 0
        self.max_length: int = 0
        self.weights_loaded: bool = False
        self.metadata_dict: dict = {}

    # -- loading ----------------------------------------------------------
    def load(self) -> bool:
        # 1. metadata
        self._load_metadata()
        # 2. tokenizer
        self._load_tokenizer()
        # 3. architecture (always built — needed whether or not weights load)
        log.info("building encoder (DenseNet201, ImageNet weights) …")
        self.encoder = build_encoder(DEFAULT_IMG_SIZE)
        log.info("building decoder (vocab=%d, max_len=%d) …",
                 self.vocab_size or 0, self.max_length or 0)
        if self.vocab_size and self.max_length:
            self.decoder = build_decoder(self.vocab_size, self.max_length)
        # 4. weights
        self.weights_loaded = self._load_decoder_weights()
        return self.weights_loaded

    def _load_metadata(self) -> None:
        if METADATA_JSON.exists():
            try:
                self.metadata_dict = json.loads(METADATA_JSON.read_text())
                self.vocab_size = int(self.metadata_dict.get("vocab_size", 0))
                self.max_length = int(self.metadata_dict.get("max_length", 0))
                log.info("metadata loaded: vocab=%d, max_len=%d",
                         self.vocab_size, self.max_length)
                return
            except Exception as exc:
                log.warning("metadata load failed: %s", exc)
        # sensible defaults if no metadata file
        self.vocab_size = 8_000
        self.max_length = 35
        self.metadata_dict = {"vocab_size": self.vocab_size,
                              "max_length": self.max_length,
                              "note": "metadata.json missing — using defaults"}

    def _load_tokenizer(self) -> None:
        if TOKENIZER_PKL.exists():
            try:
                with open(TOKENIZER_PKL, "rb") as fh:
                    self.tokenizer = pickle.load(fh)
                log.info("tokenizer loaded from %s", TOKENIZER_PKL.name)
                return
            except Exception as exc:
                log.warning("tokenizer pkl load failed: %s", exc)
        if TOKENIZER_JSON.exists():
            try:
                from tensorflow.keras.preprocessing.text import Tokenizer
                self.tokenizer = Tokenizer()
                self.tokenizer.from_json(TOKENIZER_JSON.read_text())
                log.info("tokenizer loaded from %s", TOKENIZER_JSON.name)
                return
            except Exception as exc:
                log.warning("tokenizer json load failed: %s", exc)
        log.warning("no tokenizer file found — NotebookProvider will not "
                    "be able to decode tokens properly.")

    def _load_decoder_weights(self) -> bool:
        if self.decoder is None:
            return False
        for path in (MODEL_KERAS, MODEL_H5):
            if path.exists():
                try:
                    log.info("loading decoder weights from %s …", path.name)
                    self.decoder.load_weights(str(path))
                    log.info("decoder weights loaded ✓")
                    return True
                except Exception as exc:
                    log.warning("weight load failed (%s): %s", path.name, exc)
        log.warning("no trained weights found — running with random weights. "
                    "Place model.h5 + tokenizer.pkl + metadata.json under "
                    "weights/ for production inference.")
        return False

    # -- inference --------------------------------------------------------
    def predict(self, image: Image.Image) -> CaptionResult:
        if self.encoder is None or self.decoder is None or self.tokenizer is None:
            return CaptionResult(
                caption="", inference_time_ms=0.0, confidence=0.0,
                provider=self.name, success=False,
                error="model_not_loaded",
            )
        t0 = time.perf_counter()
        try:
            feat = self._extract_features(image)
            caption, conf = self._generate_caption(feat)
            dt = (time.perf_counter() - t0) * 1000.0
            return CaptionResult(
                caption=caption, inference_time_ms=dt, confidence=conf,
                provider=self.name, success=True,
            )
        except Exception as exc:
            dt = (time.perf_counter() - t0) * 1000.0
            log.exception("inference failed")
            return CaptionResult(
                caption="", inference_time_ms=dt, confidence=0.0,
                provider=self.name, success=False, error=str(exc),
            )

    def _extract_features(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB").resize(
            (DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE), Image.BILINEAR
        )
        arr = np.asarray(img, dtype="float32") / 255.0
        arr = np.expand_dims(arr, axis=0)
        feat = self.encoder.predict(arr, verbose=0, batch_size=1)
        # notebook uses (1, 1920) — squeeze keeps shape consistent
        return feat

    def _generate_caption(self, feature: np.ndarray) -> tuple[str, float]:
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        in_text = "startseq"
        confidences: list[float] = []
        for _ in range(self.max_length):
            seq = self.tokenizer.texts_to_sequences([in_text])[0]
            seq = pad_sequences([seq], maxlen=self.max_length, padding="post")
            y_pred = self.decoder.predict([feature, seq], verbose=0, batch_size=1)[0]
            idx = int(np.argmax(y_pred))
            confidences.append(float(y_pred[idx]))
            word = self._idx_to_word(idx)
            if word is None:
                break
            in_text += " " + word
            if word == "endseq":
                break
        caption = in_text.replace("startseq", "").replace("endseq", "").strip()
        confidence = float(np.mean(confidences)) if confidences else 0.0
        return caption, confidence

    def _idx_to_word(self, idx: int) -> Optional[str]:
        for word, index in self.tokenizer.word_index.items():
            if index == idx:
                return word
        return None

    # -- metadata ---------------------------------------------------------
    def metadata(self) -> dict:
        return {
            "provider": self.name,
            "weights_loaded": self.weights_loaded,
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "encoder": "DenseNet201 (ImageNet, no top, GlobalAvgPool)",
            "encoder_feature_dim": 1920,
            "decoder": "Embedding+LSTM(256)+Dense(128)+Softmax",
            "embed_dim": 256,
            "lstm_units": 256,
            "dense_units": 128,
            "dropout": 0.5,
            "training_dataset": "Flickr8K",
            "image_size": DEFAULT_IMG_SIZE,
            "tensorflow_version": self.tf_version,
            "training_metadata": self.metadata_dict,
        }


# ---------------------------------------------------------------------------
# Fallback provider — HuggingFace BLIP (image captioning)
# Used when notebook weights aren't available, so the demo still works.
# ---------------------------------------------------------------------------
class HuggingFaceProvider(_BaseProvider):
    """
    Pre-trained BLIP base image-captioning model. Used as a transparent
    fallback so the demo always returns plausible captions even before the
    notebook's weights have been trained & dropped into weights/.
    """

    name = "huggingface-blip"

    def __init__(self) -> None:
        self._processor = None
        self._model = None
        self.tf_version = "n/a (transformers)"
        self.weights_loaded = True  # uses HF hub weights

    def load(self) -> bool:
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
            log.info("loading BLIP image-captioning model …")
            self._processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self._model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            log.info("BLIP loaded ✓")
            return True
        except Exception as exc:
            log.error("BLIP load failed: %s", exc)
            return False

    def predict(self, image: Image.Image) -> CaptionResult:
        if self._model is None or self._processor is None:
            return CaptionResult(
                caption="", inference_time_ms=0.0, confidence=0.0,
                provider=self.name, success=False, error="model_not_loaded",
            )
        t0 = time.perf_counter()
        try:
            img = image.convert("RGB")
            inputs = self._processor(img, return_tensors="pt")
            out = self._model.generate(**inputs, max_length=50)
            caption = self._processor.decode(out[0], skip_special_tokens=True)
            dt = (time.perf_counter() - t0) * 1000.0
            return CaptionResult(
                caption=caption, inference_time_ms=dt, confidence=0.92,
                provider=self.name, success=True,
            )
        except Exception as exc:
            dt = (time.perf_counter() - t0) * 1000.0
            log.exception("BLIP inference failed")
            return CaptionResult(
                caption="", inference_time_ms=dt, confidence=0.0,
                provider=self.name, success=False, error=str(exc),
            )

    def metadata(self) -> dict:
        return {
            "provider": self.name,
            "weights_loaded": True,
            "vocab_size": 30522,
            "max_length": 50,
            "encoder": "Vision Transformer (BLIP)",
            "encoder_feature_dim": 768,
            "decoder": "BERT-style text decoder",
            "embed_dim": 768,
            "lstm_units": 0,
            "dense_units": 0,
            "dropout": 0.0,
            "training_dataset": "BLIP pretraining corpus (COYO + LAION)",
            "image_size": 384,
            "tensorflow_version": self.tf_version,
            "training_metadata": {"note": "HuggingFace fallback provider"},
        }


# ---------------------------------------------------------------------------
# Manager — singleton that picks the right provider
# ---------------------------------------------------------------------------
class ModelManager:
    """Singleton — pick the notebook provider if weights exist, else BLIP."""

    _instance: Optional["ModelManager"] = None

    def __init__(self) -> None:
        self.provider: Optional[_BaseProvider] = None
        self.is_ready: bool = False
        self.weights_loaded: bool = False
        self.provider_name: str = "uninitialised"
        self.tf_version: str = "n/a"

    # -- singleton --------------------------------------------------------
    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # -- lifecycle --------------------------------------------------------
    async def initialize(self) -> None:
        # 1. Only spend the cycles to load TensorFlow + DenseNet201 if the
        #    notebook's weights, tokenizer, and metadata are actually on
        #    disk. Otherwise go straight to the BLIP fallback — this keeps
        #    memory low and startup fast for the common "no weights yet"
        #    case.
        weights_present = (
            (MODEL_H5.exists() or MODEL_KERAS.exists())
            and (TOKENIZER_PKL.exists() or TOKENIZER_JSON.exists())
            and METADATA_JSON.exists()
        )

        if weights_present:
            log.info("notebook weights detected — loading TensorFlow provider …")
            nb = NotebookProvider()
            try:
                ok = nb.load()
            except Exception as exc:
                log.error("NotebookProvider load failed: %s", exc)
                ok = False
            if ok and nb.weights_loaded and nb.tokenizer is not None:
                self.provider = nb
                self.weights_loaded = True
                self.provider_name = nb.name
                self.tf_version = nb.tf_version
                self.is_ready = True
                log.info("active provider: %s (weights loaded)", nb.name)
                return
            log.warning("notebook provider not usable — falling back")

        # 2. HuggingFace BLIP — works without trained notebook weights.
        log.info("loading HuggingFace BLIP provider …")
        hf = HuggingFaceProvider()
        if hf.load():
            self.provider = hf
            self.weights_loaded = False
            self.provider_name = hf.name
            self.tf_version = hf.tf_version
            self.is_ready = True
            log.info("active provider: %s", hf.name)
            return

        # 3. ultimate fallback — heuristic captioner (no ML deps)
        log.error("BLIP unavailable — using heuristic fallback")
        self.provider = _HeuristicProvider()
        self.weights_loaded = False
        self.provider_name = self.provider.name
        self.is_ready = True

    async def initialize_with_provider(self, provider_name: str) -> None:
        """Initialize or switch to a specific provider."""
        log.info("initializing with provider: %s", provider_name)
        
        if provider_name == "notebook-tensorflow":
            weights_present = (
                (MODEL_H5.exists() or MODEL_KERAS.exists())
                and (TOKENIZER_PKL.exists() or TOKENIZER_JSON.exists())
                and METADATA_JSON.exists()
            )
            if not weights_present:
                log.warning("notebook weights not found — cannot load notebook-tensorflow, falling back to BLIP")
                await self._load_huggingface()
                return
            
            log.info("loading TensorFlow provider …")
            nb = NotebookProvider()
            try:
                ok = nb.load()
            except Exception as exc:
                log.error("NotebookProvider load failed: %s", exc)
                ok = False
            if ok and nb.weights_loaded and nb.tokenizer is not None:
                self.provider = nb
                self.weights_loaded = True
                self.provider_name = nb.name
                self.tf_version = nb.tf_version
                self.is_ready = True
                log.info("active provider: %s (weights loaded)", nb.name)
                return
            log.warning("notebook provider not usable — falling back to BLIP")
            await self._load_huggingface()
            return
        
        elif provider_name == "huggingface-blip":
            await self._load_huggingface()
            return
        
        elif provider_name == "auto":
            # Auto mode: try notebook first, then BLIP
            await self.initialize()
            return
        
        else:
            log.error("Unknown provider: %s — using heuristic fallback", provider_name)
            self.provider = _HeuristicProvider()
            self.weights_loaded = False
            self.provider_name = self.provider.name
            self.is_ready = True

    async def _load_huggingface(self) -> None:
        """Load HuggingFace BLIP provider."""
        log.info("loading HuggingFace BLIP provider …")
        hf = HuggingFaceProvider()
        if hf.load():
            self.provider = hf
            self.weights_loaded = False
            self.provider_name = hf.name
            self.tf_version = hf.tf_version
            self.is_ready = True
            log.info("active provider: %s", hf.name)
            return
        
        # Ultimate fallback
        log.error("BLIP unavailable — using heuristic fallback")
        self.provider = _HeuristicProvider()
        self.weights_loaded = False
        self.provider_name = self.provider.name
        self.is_ready = True

    # -- inference --------------------------------------------------------
    def predict(self, image: Image.Image) -> CaptionResult:
        if self.provider is None:
            return CaptionResult(
                caption="", inference_time_ms=0.0, confidence=0.0,
                provider="none", success=False, error="not_initialised",
            )
        return self.provider.predict(image)

    def predict_bytes(self, image_bytes: bytes) -> CaptionResult:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            return self.predict(img)
        except Exception as exc:
            log.exception("image decode failed")
            return CaptionResult(
                caption="", inference_time_ms=0.0, confidence=0.0,
                provider=self.provider_name, success=False, error=str(exc),
            )

    def metadata(self) -> dict:
        if self.provider is None:
            return {"provider": "none", "ready": False}
        meta = self.provider.metadata()
        meta["ready"] = self.is_ready
        meta["weights_loaded"] = self.weights_loaded
        meta["active_provider"] = self.provider_name
        return meta


# ---------------------------------------------------------------------------
# Heuristic fallback — for environments with no ML libs at all
# ---------------------------------------------------------------------------
class _HeuristicProvider(_BaseProvider):
    """Tiny colour-histogram captioner — only used when nothing else loads."""

    name = "heuristic-fallback"
    tf_version = "n/a"

    def __init__(self) -> None:
        self.weights_loaded = False

    def load(self) -> bool:
        return True

    def predict(self, image: Image.Image) -> CaptionResult:
        t0 = time.perf_counter()
        img = image.convert("RGB").resize((64, 64))
        arr = np.asarray(img, dtype="float32") / 255.0
        mean_rgb = arr.reshape(-1, 3).mean(axis=0)
        brightness = float(mean_rgb.mean())
        warmth = float(mean_rgb[0] - mean_rgb[2])  # red - blue
        sat = float(arr.std())
        parts: list[str] = []
        if brightness > 0.6:
            parts.append("a bright")
        elif brightness < 0.3:
            parts.append("a dark")
        else:
            parts.append("a")
        if warmth > 0.1:
            parts.append("warm-toned")
        elif warmth < -0.1:
            parts.append("cool-toned")
        if sat > 0.25:
            parts.append("vividly coloured")
        else:
            parts.append("muted")
        parts.append("scene")
        caption = " ".join(parts)
        dt = (time.perf_counter() - t0) * 1000.0
        return CaptionResult(
            caption=caption, inference_time_ms=dt, confidence=0.4,
            provider=self.name, success=True,
        )

    def metadata(self) -> dict:
        return {
            "provider": self.name,
            "weights_loaded": False,
            "note": "No ML backend available — heuristic colour-based captions only.",
        }
