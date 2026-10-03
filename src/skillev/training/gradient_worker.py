from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True, slots=True)
class GradientWorkItem:
    item_id: str
    token_cost: int
    payload_ref: str

    def __post_init__(self) -> None:
        if not self.item_id.strip() or not self.payload_ref.strip():
            raise ValueError("gradient work item identity and payload reference are required")
        if type(self.token_cost) is not int or self.token_cost <= 0:
            raise ValueError("token_cost must be a positive integer")


@dataclass(frozen=True, slots=True)
class SealedGradientBatch:
    batch_id: str
    optimizer_step: int
    items: tuple[GradientWorkItem, ...]

    def __post_init__(self) -> None:
        if not self.batch_id.strip():
            raise ValueError("batch_id must be non-empty")
        if type(self.optimizer_step) is not int or self.optimizer_step <= 0:
            raise ValueError("optimizer_step must be positive")
        if not self.items:
            raise ValueError("a sealed gradient batch cannot be empty")
        item_ids = tuple(item.item_id for item in self.items)
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("sealed batch item IDs must be unique")


@dataclass(frozen=True, slots=True)
class GradientRequest:
    batch_id: str
    optimizer_step: int
    profile: str
    attempt: int
    items: tuple[GradientWorkItem, ...]

    def __post_init__(self) -> None:
        if not self.batch_id.strip() or not self.profile.strip():
            raise ValueError("gradient request identity and profile are required")
        if type(self.optimizer_step) is not int or self.optimizer_step <= 0:
            raise ValueError("optimizer_step must be positive")
        if type(self.attempt) is not int or self.attempt <= 0:
            raise ValueError("attempt must be positive")
        if not self.items:
            raise ValueError("a worker request cannot be empty")


@dataclass(frozen=True, slots=True)
class GradientTensorSummary:
    l2_norm: float
    maximum_absolute_value: float
    finite: bool

    def __post_init__(self) -> None:
        for value in (self.l2_norm, self.maximum_absolute_value):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError("gradient summary values must be numeric")
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError("gradient summary values must be finite and non-negative")
        if type(self.finite) is not bool:
            raise TypeError("finite must be boolean")


@dataclass(frozen=True, slots=True)
class GradientResult:
    worker_id: str
    profile: str
    attempt: int
    processed_item_ids: tuple[str, ...]
    loss_contribution: float
    gradients: Mapping[str, torch.Tensor]
    summaries: Mapping[str, GradientTensorSummary]
    peak_allocated_bytes: int
    peak_reserved_bytes: int

    def __post_init__(self) -> None:
        if not self.worker_id.strip() or not self.profile.strip():
            raise ValueError("worker and profile identity are required")
        if type(self.attempt) is not int or self.attempt <= 0:
            raise ValueError("attempt must be positive")
        if not self.processed_item_ids or len(set(self.processed_item_ids)) != len(
            self.processed_item_ids
        ):
            raise ValueError("processed item IDs must be non-empty and unique")
        if (
            isinstance(self.loss_contribution, bool)
            or not isinstance(self.loss_contribution, int | float)
            or not math.isfinite(float(self.loss_contribution))
        ):
            raise ValueError("loss contribution must be finite")
        if set(self.gradients) != set(self.summaries):
            raise ValueError("gradient tensors and summaries must name the same parameters")
        if any(not isinstance(tensor, torch.Tensor) for tensor in self.gradients.values()):
            raise TypeError("gradients must contain tensors")
        for field_name in ("peak_allocated_bytes", "peak_reserved_bytes"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class AggregatedGradientResult:
    batch_id: str
    optimizer_step: int
    profile: str
    attempt: int
    worker_ids: tuple[str, ...]
    processed_item_ids: tuple[str, ...]
    loss: float
    gradients: Mapping[str, torch.Tensor]
    peak_allocated_bytes: int
    peak_reserved_bytes: int


class GradientWorkerClient(Protocol):
    @property
    def worker_id(self) -> str: ...

    def compute_weighted_grads(self, request: GradientRequest) -> GradientResult: ...


class GradientWorkerOOMError(RuntimeError):
    def __init__(self, *, worker_id: str, batch_id: str, microbatch_index: int) -> None:
        super().__init__(f"gradient worker {worker_id} exhausted CUDA memory")
        self.worker_id = worker_id
        self.batch_id = batch_id
        self.microbatch_index = microbatch_index


@dataclass(slots=True)
class CostBalancedGradientWorkerPool:
    workers: tuple[GradientWorkerClient, ...]
    profile: str

    def __post_init__(self) -> None:
        if not self.workers:
            raise ValueError("a gradient worker pool cannot be empty")
        if not self.profile.strip():
            raise ValueError("profile must be non-empty")
        worker_ids = tuple(worker.worker_id for worker in self.workers)
        if len(set(worker_ids)) != len(worker_ids):
            raise ValueError("gradient worker IDs must be unique")

    def compute(self, batch: SealedGradientBatch, *, attempt: int) -> AggregatedGradientResult:
        partitions = partition_items_by_token_cost(batch.items, len(self.workers))
        results: list[GradientResult] = []
        for worker, items in zip(self.workers, partitions, strict=True):
            request = GradientRequest(
                batch_id=batch.batch_id,
                optimizer_step=batch.optimizer_step,
                profile=self.profile,
                attempt=attempt,
                items=items,
            )
            try:
                result = worker.compute_weighted_grads(request)
            except torch.cuda.OutOfMemoryError as error:
                raise GradientWorkerOOMError(
                    worker_id=worker.worker_id,
                    batch_id=batch.batch_id,
                    microbatch_index=0,
                ) from error
            if result.worker_id != worker.worker_id:
                raise ValueError("gradient result came from the wrong worker")
            if result.profile != self.profile or result.attempt != attempt:
                raise ValueError("gradient result profile or attempt differs")
            if result.processed_item_ids != tuple(item.item_id for item in items):
                raise ValueError("gradient worker did not process its exact assigned items")
            results.append(result)
        return aggregate_gradient_results(batch, results, profile=self.profile, attempt=attempt)


def partition_items_by_token_cost(
    items: Sequence[GradientWorkItem],
    worker_count: int,
) -> tuple[tuple[GradientWorkItem, ...], ...]:
    if type(worker_count) is not int or worker_count <= 0:
        raise ValueError("worker_count must be positive")
    if len(items) < worker_count:
        raise ValueError("each gradient worker must receive a non-empty partition")
    indexed = list(enumerate(items))
    indexed.sort(key=lambda pair: (-pair[1].token_cost, pair[0]))
    bins: list[list[tuple[int, GradientWorkItem]]] = [[] for _ in range(worker_count)]
    costs = [0] * worker_count
    for original_index, item in indexed:
        target = min(range(worker_count), key=lambda index: (costs[index], index))
        bins[target].append((original_index, item))
        costs[target] += item.token_cost
    return tuple(
        tuple(item for _, item in sorted(worker_items, key=lambda pair: pair[0]))
        for worker_items in bins
    )


def aggregate_gradient_results(
    batch: SealedGradientBatch,
    results: Sequence[GradientResult],
    *,
    profile: str,
    attempt: int,
) -> AggregatedGradientResult:
    if not results:
        raise ValueError("cannot aggregate an empty gradient result set")
    processed = tuple(item_id for result in results for item_id in result.processed_item_ids)
    expected = tuple(item.item_id for item in batch.items)
    if len(processed) != len(set(processed)) or set(processed) != set(expected):
        raise ValueError("gradient results must process every sealed item exactly once")
    parameter_names = set(results[0].gradients)
    if any(set(result.gradients) != parameter_names for result in results):
        raise ValueError("all gradient workers must return the same parameter set")
    gradients: dict[str, torch.Tensor] = {}
    for name in sorted(parameter_names):
        tensors = [result.gradients[name].detach().to(device="cpu") for result in results]
        reference = tensors[0]
        if any(
            tensor.shape != reference.shape or tensor.dtype != reference.dtype for tensor in tensors
        ):
            raise ValueError("gradient shard tensors have incompatible shape or dtype")
        combined = torch.zeros_like(reference)
        for tensor in tensors:
            combined.add_(tensor)
        if not bool(torch.isfinite(combined).all()):
            raise ValueError("aggregated gradient contains a non-finite value")
        gradients[name] = combined
    return AggregatedGradientResult(
        batch_id=batch.batch_id,
        optimizer_step=batch.optimizer_step,
        profile=profile,
        attempt=attempt,
        worker_ids=tuple(result.worker_id for result in results),
        processed_item_ids=expected,
        loss=math.fsum(float(result.loss_contribution) for result in results),
        gradients=gradients,
        peak_allocated_bytes=max(result.peak_allocated_bytes for result in results),
        peak_reserved_bytes=max(result.peak_reserved_bytes for result in results),
    )


__all__ = [
    "AggregatedGradientResult",
    "CostBalancedGradientWorkerPool",
    "GradientRequest",
    "GradientResult",
    "GradientTensorSummary",
    "GradientWorkItem",
    "GradientWorkerClient",
    "GradientWorkerOOMError",
    "SealedGradientBatch",
    "aggregate_gradient_results",
    "partition_items_by_token_cost",
]
