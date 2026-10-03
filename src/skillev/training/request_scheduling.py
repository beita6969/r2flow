from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from skillev.diagnostics.rollout_progress import current_progress

FAIR_MODEL_REQUESTS = "horizon-action-aging@1"


@dataclass(slots=True)
class _Waiting:
    future: asyncio.Future[None]
    queued: float
    bonus_seconds: float
    sequence: int
    tokens: int = 0


class FairRequestGate:
    def __init__(self, capacity: int, *, token_capacity: int | None = None) -> None:
        if type(capacity) is not int or capacity < 1:
            raise ValueError("request capacity must be positive")
        if token_capacity is not None and (type(token_capacity) is not int or token_capacity < 1):
            raise ValueError("token capacity must be positive")
        self.capacity = capacity
        self.token_capacity = token_capacity
        self.active_tokens = 0
        self.active = 0
        self._sequence = 0
        self._waiting: list[_Waiting] = []

    async def __aenter__(self) -> None:
        await self.acquire()

    async def acquire(self, tokens: int = 0) -> None:
        if type(tokens) is not int or tokens < 0:
            raise ValueError("token reservation must be nonnegative")
        if self.token_capacity is not None and tokens > self.token_capacity:
            raise ValueError("one request exceeds the declared service token capacity")
        row = current_progress()
        horizon, phase = (1, None) if row is None else row.request_priority
        bonus = min(4.0, max(0, horizon - 1) / 12.25) + (2.0 if phase == "action" else 0.0)
        future = asyncio.get_running_loop().create_future()
        item = _Waiting(future, time.perf_counter(), bonus, self._sequence, tokens)
        self._sequence += 1
        self._waiting.append(item)
        self._dispatch()
        try:
            await future
        except asyncio.CancelledError:
            if item in self._waiting:
                self._waiting.remove(item)
            elif future.done() and not future.cancelled():
                self.active -= 1
                self.active_tokens -= tokens
            self._dispatch()
            raise

    async def __aexit__(self, *args: object) -> None:
        self.release()

    def release(self, tokens: int = 0) -> None:
        self.active -= 1
        self.active_tokens -= tokens
        self._dispatch()

    def _dispatch(self) -> None:
        while self._waiting and self.active < self.capacity:
            item = min(self._waiting, key=lambda w: (w.queued - w.bonus_seconds, w.sequence))
            if (
                self.token_capacity is not None
                and self.active_tokens + item.tokens > self.token_capacity
            ):
                return
            self._waiting.remove(item)
            if not item.future.cancelled():
                self.active += 1
                self.active_tokens += item.tokens
                item.future.set_result(None)
