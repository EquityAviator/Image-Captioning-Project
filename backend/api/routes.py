"""
HTTP routes — thin wrappers around the ModelManager singleton.
"""

from __future__ import annotations

import time
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from api.schemas import (
    BatchPredictItem,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictResponse,
)
from inference.manager import ModelManager
from utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter()

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/gif",
}
MAX_BYTES = 10 * 1024 * 1024  # 10 MB


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    mgr = ModelManager.get_instance()
    return HealthResponse(
        status="ok",
        model_loaded=mgr.is_ready,
        provider=mgr.provider_name,
        weights_loaded=mgr.weights_loaded,
        backend=mgr.tf_version,
    )


@router.get("/model-info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    mgr = ModelManager.get_instance()
    meta = mgr.metadata()
    # Map to schema (some fields may be missing depending on provider)
    return ModelInfoResponse(
        ready=meta.get("ready", False),
        active_provider=meta.get("active_provider", "none"),
        weights_loaded=meta.get("weights_loaded", False),
        provider=meta.get("provider", "none"),
        encoder=meta.get("encoder", "n/a"),
        encoder_feature_dim=meta.get("encoder_feature_dim", 0),
        decoder=meta.get("decoder", "n/a"),
        embed_dim=meta.get("embed_dim", 0),
        lstm_units=meta.get("lstm_units", 0),
        dense_units=meta.get("dense_units", 0),
        dropout=meta.get("dropout", 0.0),
        training_dataset=meta.get("training_dataset", "n/a"),
        image_size=meta.get("image_size", 0),
        vocab_size=meta.get("vocab_size", 0),
        max_length=meta.get("max_length", 0),
        tensorflow_version=meta.get("tensorflow_version", "n/a"),
        training_metadata=meta.get("training_metadata", {}),
    )


@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    """Generate a caption for a single uploaded image.

    The handler is async for fast file reading, but the heavy ML inference
    is dispatched to a worker thread via `run_in_executor` so we never
    block the uvicorn event loop (which can cause the server to be killed
    by reverse-proxy health probes on slow CPU inference).
    """
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"unsupported content type: {file.content_type}. "
                   f"Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )
    blob = await file.read()
    if not blob:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="empty file payload",
        )
    if len(blob) > MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"file too large: {len(blob)} bytes (max {MAX_BYTES})",
        )

    import asyncio
    import concurrent.futures

    mgr = ModelManager.get_instance()
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    # Run CPU-heavy inference in a threadpool so the event loop stays alive.
    result = await loop.run_in_executor(None, mgr.predict_bytes, blob)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    if not result.success:
        # still return 200 so the frontend can render the error gracefully
        return PredictResponse(
            caption="",
            inference_time=f"{wall_ms:.1f}ms",
            confidence=0.0,
            provider=result.provider,
            success=False,
            error=result.error,
        )

    return PredictResponse(
        caption=result.caption,
        inference_time=f"{result.inference_time_ms:.1f}ms",
        confidence=result.confidence,
        provider=result.provider,
        success=True,
    )


@router.post("/predict-batch", response_model=BatchPredictResponse)
async def predict_batch(files: List[UploadFile] = File(...)) -> BatchPredictResponse:
    """Generate captions for up to 8 images at once."""
    if not files:
        raise HTTPException(status_code=400, detail="no files provided")
    if len(files) > 8:
        raise HTTPException(status_code=400, detail="max 8 files per batch")

    import asyncio

    mgr = ModelManager.get_instance()
    loop = asyncio.get_running_loop()
    items: list[BatchPredictItem] = []
    for f in files:
        blob = await f.read()
        if not blob or (f.content_type and f.content_type not in ALLOWED_CONTENT_TYPES):
            items.append(BatchPredictItem(
                filename=f.filename or "unknown",
                caption="",
                inference_time="0.0ms",
                confidence=0.0,
                success=False,
                error="invalid file",
            ))
            continue
        res = await loop.run_in_executor(None, mgr.predict_bytes, blob)
        items.append(BatchPredictItem(
            filename=f.filename or "unknown",
            caption=res.caption,
            inference_time=f"{res.inference_time_ms:.1f}ms",
            confidence=res.confidence,
            success=res.success,
            error=res.error,
        ))

    return BatchPredictResponse(
        results=items,
        total=len(items),
        success_count=sum(1 for i in items if i.success),
    )
