from __future__ import annotations

import signal
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Event


class TrainingPausedError(Exception):
    def __init__(self, optimizer_step: int, checkpoint: Path) -> None:
        super().__init__(f"training paused after committed step {optimizer_step}")
        self.optimizer_step = optimizer_step
        self.checkpoint = checkpoint


class StopAfterCheckpoint:
    def __init__(self, request_file: Path) -> None:
        self.request_file = request_file
        self._requested = Event()

    def request(self, _signal: int = 0, _frame: object = None) -> None:
        self._requested.set()

    def __call__(self) -> bool:
        return self._requested.is_set() or self.request_file.is_file()

    @contextmanager
    def signals(self) -> Iterator[None]:
        watched = (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)
        previous = {kind: signal.getsignal(kind) for kind in watched}
        try:
            for kind in watched:
                signal.signal(kind, self.request)
            yield
        finally:
            for kind, handler in previous.items():
                signal.signal(kind, handler)
