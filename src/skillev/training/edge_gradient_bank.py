from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Final

import torch

EDGE_GRADIENT_BANK_ID: Final = "per-edge-gradient-bank@1"

_Buffer = dict[str, torch.Tensor]


class EdgeGradientBank:
    def __init__(
        self, named: Mapping[str, torch.nn.Parameter], *, device_budget_bytes: int
    ) -> None:
        if not named:
            raise ValueError("an edge gradient bank needs trainable parameters")
        if type(device_budget_bytes) is not int or device_budget_bytes < 0:
            raise ValueError("edge gradient device budget must be a non-negative integer")
        self.named = dict(named)
        self.device_budget_bytes = device_budget_bytes
        self._edges: dict[int, _Buffer] = {}
        self._candidates: dict[tuple[int, int], _Buffer] = {}
        self.peak_device_bytes = 0

    def _buffers(self) -> tuple[_Buffer, ...]:
        return (*self._edges.values(), *self._candidates.values())

    def _bytes(self, *, device: bool) -> int:
        return sum(
            value.numel() * 4
            for buffer in self._buffers()
            for value in buffer.values()
            if (value.device.type != "cpu") == device
        )

    @property
    def device_bytes(self) -> int:
        return self._bytes(device=True)

    @property
    def host_bytes(self) -> int:
        return self._bytes(device=False)

    def _store(self, gradient: torch.Tensor, resident: int) -> torch.Tensor:
        value = gradient.detach().to(dtype=torch.float32)
        size = value.numel() * 4
        if value.device.type == "cpu" or resident + size <= self.device_budget_bytes:
            return value
        host = torch.empty(value.shape, dtype=torch.float32, pin_memory=torch.cuda.is_available())
        host.copy_(value)
        return host

    def _take_grads(self) -> _Buffer:
        taken: _Buffer = {}
        resident = self.device_bytes
        for name, parameter in self.named.items():
            if parameter.grad is not None:
                taken[name] = self._store(parameter.grad, resident)
                if taken[name].device.type != "cpu":
                    resident += taken[name].numel() * 4
                parameter.grad = None
        return taken

    def _track_peak(self) -> None:
        self.peak_device_bytes = max(self.peak_device_bytes, self.device_bytes)

    @staticmethod
    def _add_into(target: _Buffer, source: _Buffer, alpha: float = 1.0) -> None:
        for name, value in source.items():
            if name in target:
                target[name].add_(value.to(target[name].device), alpha=alpha)
            else:
                target[name] = value.mul(alpha) if alpha != 1.0 else value

    def capture(self, t: int) -> None:
        if type(t) is not int or t < 1:
            raise ValueError("edge index must be a positive integer")
        taken = self._take_grads()
        buffer = self._edges.setdefault(t, {})
        self._add_into(buffer, taken)
        self._track_peak()

    def capture_candidate(self, t: int, c: int) -> None:
        if (t, c) in self._candidates:
            raise ValueError("candidate gradient captured twice")
        self._candidates[(t, c)] = self._take_grads()
        self._track_peak()

    def fold_candidates(self, t: int, weights: Mapping[int, float]) -> None:
        keys = sorted(c for (edge, c) in self._candidates if edge == t)
        if keys != sorted(weights):
            raise ValueError("candidate weights must cover exactly the captured candidates")
        buffer = self._edges.setdefault(t, {})
        for c in keys:
            weight = float(weights[c])
            if not math.isfinite(weight):
                raise ValueError("candidate fold weight must be finite")
            self._add_into(buffer, self._candidates.pop((t, c)), alpha=weight)

    @property
    def edges(self) -> tuple[int, ...]:
        return tuple(sorted(self._edges))

    def combine(self, coefficients: Sequence[float], scale: float) -> dict[str, torch.Tensor]:
        if self._candidates:
            raise RuntimeError("unfolded candidate gradients remain in the bank")
        horizon = len(coefficients)
        if any(t > horizon for t in self._edges):
            raise ValueError("bank holds edges beyond the coefficient horizon")
        out: dict[str, torch.Tensor] = {}
        for name, parameter in self.named.items():
            total = torch.zeros(parameter.shape, dtype=torch.float32, device=parameter.device)
            for t in range(1, horizon + 1):
                value = self._edges.get(t, {}).get(name)
                if value is not None:
                    total.add_(value.to(parameter.device), alpha=float(coefficients[t - 1]) * scale)
            out[name] = total
        return out

    def clear(self) -> None:
        self._edges.clear()
        self._candidates.clear()

    def metrics(self) -> dict[str, int]:
        return {
            "device_bytes": self.device_bytes,
            "host_bytes": self.host_bytes,
            "peak_device_bytes": self.peak_device_bytes,
            "device_budget_bytes": self.device_budget_bytes,
        }


__all__ = ["EDGE_GRADIENT_BANK_ID", "EdgeGradientBank"]
