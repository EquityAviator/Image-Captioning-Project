"""
Pydantic schemas — request/response shapes for the CaptionAI API.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    provider: str
    weights_loaded: bool
    backend: str


class PredictResponse(BaseModel):
    caption: str
    inference_time: str
    confidence: float = Field(..., description="Mean softmax confidence in [0,1].")
    semantic_score: Optional[float] = Field(
        None, description="Semantic agreement in [0,1]: how strongly THIS image supports the caption (BLIP PMI, logistic-mapped).")
    semantic_pmi: Optional[float] = Field(
        None, description="Raw BLIP PMI log-evidence (image-conditioned minus neutral-baseline mean log-prob).")
    provider: str
    success: bool
    error: Optional[str] = None


class BatchPredictItem(BaseModel):
    filename: str
    caption: str
    inference_time: str
    confidence: float
    semantic_score: Optional[float] = None
    success: bool
    error: Optional[str] = None


class BatchPredictResponse(BaseModel):
    results: List[BatchPredictItem]
    total: int
    success_count: int


class ModelInfoResponse(BaseModel):
    ready: bool
    active_provider: str
    weights_loaded: bool
    provider: str
    encoder: str
    encoder_feature_dim: int
    decoder: str
    embed_dim: int
    lstm_units: int
    dense_units: int
    dropout: float
    training_dataset: str
    image_size: int
    vocab_size: int
    max_length: int
    tensorflow_version: str
    training_metadata: dict
