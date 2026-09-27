"""
CaptionAI — shared canonical paths for all backend scripts.

Single source of truth so WEIGHTS_DIR never gets assigned inconsistently
across scripts. Every backend script should do:

    from paths import DATASET_DIR, WEIGHTS_DIR
"""

from __future__ import annotations

from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent

DATASET_DIR = PROJECT_DIR / "dataset"
WEIGHTS_DIR = PROJECT_DIR / "weights"

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
