from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationRequestWindow:
    work_deadline: float
    cleanup_deadline: float
    record: Callable[[dict[str, object]], None]


REQUEST_WINDOW: ContextVar[EvaluationRequestWindow | None] = ContextVar(
    "evaluation_request_window", default=None
)


@contextmanager
def request_window(window: EvaluationRequestWindow | None) -> Iterator[None]:
    token = REQUEST_WINDOW.set(window)
    try:
        yield
    finally:
        REQUEST_WINDOW.reset(token)
