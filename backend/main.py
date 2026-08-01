"""
CaptionAI Backend — FastAPI service for image captioning inference.

Architecture mirrors the original Kaggle notebook:
  - Encoder: DenseNet201 (pretrained, last layer removed) -> 1920-dim feature
  - Decoder: Dense(256) + Reshape + Embedding + Concat + LSTM(256)
             + Dropout + Add + Dense(128) + Dense(vocab_size, softmax)

Endpoints
---------
GET  /health         -> service status + model loaded flag
GET  /model-info     -> architecture + training metadata
POST /predict        -> upload an image, returns {caption, inference_time, success}
POST /predict-batch  -> upload multiple images (up to 8)
GET  /               -> root health check
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---- paths ---------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BACKEND_DIR.parent / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# allow sibling imports
sys.path.insert(0, str(BACKEND_DIR))

# ---- env -----------------------------------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # CPU-only by default

# ---- fastapi -------------------------------------------------------------
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router as api_router
from inference.manager import ModelManager
from utils.logger import get_logger

log = get_logger(__name__)

# ---- app -----------------------------------------------------------------
app = FastAPI(
    title="CaptionAI API",
    version="1.0.0",
    description="Deep Learning image captioning service (DenseNet201 + LSTM, Flickr8K).",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    """Pre-load the model so the first request is fast."""
    log.info("startup: loading model manager …")
    mgr = ModelManager.get_instance()
    await mgr.initialize()
    log.info(
        "startup: ready (provider=%s, weights=%s)",
        mgr.provider_name,
        mgr.weights_loaded,
    )


@app.get("/")
async def root() -> dict:
    return {"service": "CaptionAI", "status": "ok", "docs": "/docs"}


@app.get("/health")
async def health() -> dict:
    mgr = ModelManager.get_instance()
    return {
        "status": "ok",
        "model_loaded": mgr.is_ready,
        "provider": mgr.provider_name,
        "weights_loaded": mgr.weights_loaded,
        "backend": "TensorFlow " + mgr.tf_version,
    }


app.include_router(api_router, prefix="")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
