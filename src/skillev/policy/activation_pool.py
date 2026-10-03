from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock

import torch

BufferKey = tuple[torch.dtype, tuple[int, ...]]


@dataclass(slots=True, weakref_slot=True)
class PinnedActivation:
    device: torch.device
    saved: torch.Tensor
    ready: torch.cuda.Event
    restores: list[torch.cuda.Event] = field(default_factory=list)


class PinnedActivationPool:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._lock = RLock()
        self.allocated_bytes = self.live_bytes = self.peak_live_bytes = 0
        self.reuses = self.allocations = self.fallbacks = 0
        self._free: OrderedDict[BufferKey, list[torch.Tensor]] = OrderedDict()
        self._pending: list[tuple[torch.Tensor, tuple[torch.cuda.Event, ...]]] = []

    def _reclaim(self) -> None:
        pending = []
        for tensor, events in self._pending:
            if all(event.query() for event in events):
                self.live_bytes -= tensor.numel() * tensor.element_size()
                self._free.setdefault((tensor.dtype, tuple(tensor.shape)), []).append(tensor)
            else:
                pending.append((tensor, events))
        self._pending = pending

    def acquire(self, source: torch.Tensor) -> torch.Tensor | None:
        with self._lock:
            return self._acquire(source)

    def _acquire(self, source: torch.Tensor) -> torch.Tensor | None:
        self._reclaim()
        key = source.dtype, tuple(source.shape)
        size = source.numel() * source.element_size()
        available = self._free.get(key)
        if available:
            result = available.pop()
            if not available:
                del self._free[key]
            self.reuses += 1
        else:
            while self._free and self.allocated_bytes + size > self.limit:
                _, buffers = self._free.popitem(last=False)
                self.allocated_bytes -= sum(t.numel() * t.element_size() for t in buffers)
            if self.allocated_bytes + size > self.limit:
                self.fallbacks += 1
                return None
            try:
                result = torch.empty(
                    source.shape, dtype=source.dtype, device="cpu", pin_memory=True
                )
            except (torch.OutOfMemoryError, RuntimeError):
                self.fallbacks += 1
                return None
            self.allocated_bytes += size
            self.allocations += 1
        self.live_bytes += size
        self.peak_live_bytes = max(self.peak_live_bytes, self.live_bytes)
        return result

    def release(
        self, tensor: torch.Tensor, ready: torch.cuda.Event, restores: list[torch.cuda.Event]
    ) -> None:
        with self._lock:
            self._pending.append((tensor, (ready, *restores)))
            self._reclaim()

    def close(self) -> None:
        with self._lock:
            for _, events in self._pending:
                for event in events:
                    event.synchronize()
            self._reclaim()
            self._free.clear()
            self.allocated_bytes = self.live_bytes

    def __del__(self) -> None:
        try:
            self.close()
        except (RuntimeError, AttributeError):
            pass
