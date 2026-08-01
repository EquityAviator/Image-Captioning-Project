"""Small shared helpers."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator


class Timer:
    """Context manager that measures elapsed wall-clock seconds."""

    def __init__(self) -> None:
        self._t0: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._t0

    @property
    def ms(self) -> float:
        return self.elapsed * 1000.0


@asynccontextmanager
async def async_timer() -> AsyncIterator[tuple[Timer, None]]:
    t = Timer()
    with t:
        yield t, None
