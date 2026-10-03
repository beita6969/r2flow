from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path


class RunWallClock:
    def __init__(
        self,
        root: Path,
        *,
        wall_clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        path = root / "run-clock.json"
        now = wall_clock()
        if not path.exists():
            config = root / "formal-config.json"
            start = min(now, config.stat().st_mtime) if config.exists() else now
            value = {
                "format": "skillev-run-wall-clock@1",
                "started_unix_seconds": start,
                "origin": "existing-formal-config-mtime" if config.exists() else "run-start",
            }
            with path.open("x", encoding="utf-8") as handle:
                json.dump(value, handle)
                handle.flush()
                os.fsync(handle.fileno())
            descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        value = json.loads(path.read_text(encoding="utf-8"))
        start = value["started_unix_seconds"]
        if type(start) not in (int, float) or not math.isfinite(start) or start > now:
            raise ValueError("run wall-clock origin is invalid or ahead of this host")
        self.origin = value["origin"]
        self._prior = now - float(start)
        self._monotonic = monotonic
        self._entered = monotonic()

    def elapsed(self) -> float:
        return self._prior + self._monotonic() - self._entered
