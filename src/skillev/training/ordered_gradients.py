from __future__ import annotations

import time
from collections.abc import Mapping

import torch

from .gradient_buckets import PackedGradients, pack_gradients
from .step_math import TTBArtifactMath, TTBGradientShard, accumulate_ttb_gradients


class OrderedGradientAccumulator:
    def __init__(
        self,
        positions: tuple[int, ...],
        parameters: Mapping[str, torch.Tensor],
        max_buffer_bytes: int,
    ) -> None:
        if type(max_buffer_bytes) is not int or max_buffer_bytes < 1:
            raise ValueError("gradient buffer byte limit must be positive")
        self.positions = positions
        self.parameters = parameters
        self.max_buffer_bytes = max_buffer_bytes
        self.contribution_bytes = sum(p.numel() * p.element_size() for p in parameters.values())
        self.gradients: dict[str, torch.Tensor] = {}
        self.items: list[TTBArtifactMath] = []
        self.pending: dict[int, tuple[TTBArtifactMath, PackedGradients]] = {}
        self.buffer_bytes = 0
        self.peak_buffer_bytes = 0
        self.cursor = 0
        self.host_pack_seconds = 0.0
        self.host_restore_seconds = 0.0

    @property
    def next_position(self) -> int | None:
        return self.positions[self.cursor] if self.cursor < len(self.positions) else None

    def may_compute(self, position: int) -> bool:
        return position == self.next_position or (
            self.buffer_bytes + self.contribution_bytes <= self.max_buffer_bytes
        )

    def add(self, contribution: TTBGradientShard) -> None:
        (item,) = contribution.artifacts
        self._check_position(item)
        if set(contribution.gradients) != set(self.parameters):
            raise ValueError("gradient contribution parameter set differs")
        if item.position != self.next_position:
            started = time.perf_counter()
            packed = pack_gradients(contribution.gradients)
            self.host_pack_seconds += time.perf_counter() - started
            self.add_packed(item, packed)
        else:
            self._append(item, contribution.gradients)
            self._drain()

    def _check_position(self, item: TTBArtifactMath) -> None:
        if item.position not in self.positions[self.cursor :] or item.position in self.pending:
            raise ValueError("gradient position was not assigned or was already consumed")

    def add_packed(self, item: TTBArtifactMath, packed: PackedGradients) -> None:
        self._check_position(item)
        if {name for bucket in packed.names for name in bucket} != set(self.parameters):
            raise ValueError("gradient buffer parameter set differs")
        if item.position != self.next_position:
            if self.buffer_bytes + packed.nbytes > self.max_buffer_bytes:
                raise RuntimeError("out-of-order gradient buffer is full")
            self.pending[item.position] = (item, packed)
            self.buffer_bytes += packed.nbytes
            self.peak_buffer_bytes = max(self.peak_buffer_bytes, self.buffer_bytes)
            return
        self._restore(item, packed)
        self._drain()

    def _append(self, item: TTBArtifactMath, gradients: Mapping[str, torch.Tensor]) -> None:
        accumulate_ttb_gradients(self.gradients, gradients)
        self.items.append(item)
        self.cursor += 1

    def _restore(self, item: TTBArtifactMath, packed: PackedGradients) -> None:
        started = time.perf_counter()
        self._append(item, packed.on_device(self.parameters))
        self.host_restore_seconds += time.perf_counter() - started

    def _drain(self) -> None:
        while self.next_position in self.pending:
            assert self.next_position is not None
            item, packed = self.pending.pop(self.next_position)
            self.buffer_bytes -= packed.nbytes
            self._restore(item, packed)

    def clear(self) -> None:
        self.pending.clear()
        self.gradients.clear()
        self.items.clear()
        self.buffer_bytes = 0
