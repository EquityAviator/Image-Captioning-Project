"""Logging helpers — quiet TF / absl chatter, structured console output."""

from __future__ import annotations

import logging
import os
import sys


def _configure_root() -> logging.Logger:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    fmt = "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        stream=sys.stdout,
        force=True,
    )
    # quiet noisy libs
    for noisy in ("absl", "tensorflow", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("captionai")


_root = _configure_root()


def get_logger(name: str = "captionai") -> logging.Logger:
    return _root.getChild(name) if name != "captionai" else _root
