#!/usr/bin/env bash
# CaptionAI FastAPI backend launcher — runs on port 8000
set -e
cd "$(dirname "$0")/.."
export PORT="${PORT:-8000}"
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=-1
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONUNBUFFERED=1
exec python3 -m uvicorn main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level info \
    --app-dir backend
