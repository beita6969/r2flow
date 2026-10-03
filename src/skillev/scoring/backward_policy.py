from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

_SUM_TOLERANCE = 1e-6


def _check(actual: int, count: int) -> None:
    if type(count) is not int or count < 1:
        raise ValueError("In(s') of a non-root state is non-empty")
    if type(actual) is not int or not 0 <= actual < count:
        raise ValueError("actual in-edge index out of range")


def needs_scores(count: int) -> bool:
    _check(0, count)
    return count > 1


def log_pb_without_scores(actual: int, count: int) -> float:
    _check(actual, count)
    if count == 1:
        return 0.0
    raise ValueError("the learned in-edge softmax needs candidate scores")


def log_pb_from_scores(scores: Sequence[float], actual: int) -> float:
    count = len(scores)
    _check(actual, count)
    if not needs_scores(count):
        return log_pb_without_scores(actual, count)
    values = [float(value) for value in scores]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("candidate scores must be finite")
    top = max(values)
    normaliser = top + math.log(sum(math.exp(value - top) for value in values))
    log_probs = [value - normaliser for value in values]
    total = math.fsum(math.exp(value) for value in log_probs)
    if abs(total - 1.0) > _SUM_TOLERANCE:
        raise AssertionError("Σ_In P_B must equal 1")
    return log_probs[actual]


def log_pb_tensor(scores: torch.Tensor, actual: int) -> torch.Tensor:
    import torch

    if scores.ndim != 1:
        raise ValueError("candidate scores must be a 1-D tensor")
    count = int(scores.shape[0])
    _check(actual, count)
    if not needs_scores(count):
        return torch.tensor(
            log_pb_without_scores(actual, count), dtype=scores.dtype, device=scores.device
        )
    log_probs = scores - torch.logsumexp(scores, 0)
    if abs(float(torch.logsumexp(log_probs.detach().double(), 0))) > _SUM_TOLERANCE:
        raise AssertionError("Σ_In P_B must equal 1")
    return log_probs[actual]


__all__ = [
    "log_pb_from_scores",
    "log_pb_tensor",
    "log_pb_without_scores",
    "needs_scores",
]
