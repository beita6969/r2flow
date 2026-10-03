from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

SUBTB_RESIDUAL_ID: Final = "subtb-all-pairs-geometric@1"
SUBTB_WEIGHT_NORMALIZATION: Final = "per-trajectory-included-pairs@1"
SUBTB_HUBER_GRAD_SUFFIX: Final = "+huber-grad@1"
SUBTB_HUBER_GRAD_ID: Final = SUBTB_RESIDUAL_ID + SUBTB_HUBER_GRAD_SUFFIX
SUBTB_HUBER_GRAD_DELTA: Final = 20.0


@dataclass(frozen=True, slots=True)
class SubTBTerms:
    pairs: tuple[tuple[int, int], ...]
    weights: tuple[float, ...]
    deltas: tuple[float, ...]
    loss: float
    edge_coefficients: tuple[float, ...]
    flow_coefficients: tuple[float, ...]
    delta_0T: float
    edge_residuals: tuple[float, ...]
    clipped_pairs: int = 0


def subtb_pair_count(horizon: int) -> int:
    if type(horizon) is not int or horizon < 1:
        raise ValueError("SubTB horizon must be a positive integer")
    return horizon * (horizon + 1) // 2


def subtb_terms(
    *,
    log_flows: Sequence[float],
    edge_log_ratios: Sequence[float],
    lam: float,
    gradient_clip: float | None = None,
) -> SubTBTerms:
    flows = tuple(float(value) for value in log_flows)
    ratios = tuple(float(value) for value in edge_log_ratios)
    horizon = len(ratios)
    if horizon < 1 or len(flows) != horizon + 1:
        raise ValueError("SubTB requires T >= 1 edge terms and exactly T + 1 log flows")
    if not all(math.isfinite(value) for value in (*flows, *ratios)):
        raise ValueError("SubTB log flows and edge terms must be finite")
    if isinstance(lam, bool) or not math.isfinite(lam) or not 0 < lam <= 1:
        raise ValueError("SubTB lambda must satisfy 0 < lambda <= 1")
    if gradient_clip is not None and (
        isinstance(gradient_clip, bool) or not math.isfinite(gradient_clip) or gradient_clip <= 0
    ):
        raise ValueError("SubTB gradient clip must be a positive finite number")
    pairs = tuple((i, j) for i in range(horizon) for j in range(i + 1, horizon + 1))
    raw = tuple(lam ** (j - i) for i, j in pairs)
    norm = math.fsum(raw)
    weights = tuple(value / norm for value in raw)
    deltas = tuple(math.fsum((flows[i], *ratios[i:j], -flows[j])) for i, j in pairs)
    loss = math.fsum(w * d * d for w, d in zip(weights, deltas, strict=True))
    bound = math.inf if gradient_clip is None else float(gradient_clip)
    scaled = tuple(2 * w * max(-bound, min(bound, d)) for w, d in zip(weights, deltas, strict=True))
    edge_coefficients = tuple(
        math.fsum(g for (i, j), g in zip(pairs, scaled, strict=True) if i < t <= j)
        for t in range(1, horizon + 1)
    )
    flow_coefficients = tuple(
        math.fsum(
            [g for (i, _), g in zip(pairs, scaled, strict=True) if i == k]
            + [-g for (_, j), g in zip(pairs, scaled, strict=True) if j == k]
        )
        for k in range(horizon)
    )
    by_pair = dict(zip(pairs, deltas, strict=True))
    values = (loss, *deltas, *edge_coefficients, *flow_coefficients)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("SubTB arithmetic overflowed")
    return SubTBTerms(
        pairs=pairs,
        weights=weights,
        deltas=deltas,
        loss=loss,
        edge_coefficients=edge_coefficients,
        flow_coefficients=flow_coefficients,
        delta_0T=by_pair[(0, horizon)],
        edge_residuals=tuple(by_pair[(t - 1, t)] for t in range(1, horizon + 1)),
        clipped_pairs=sum(1 for d in deltas if abs(d) > bound),
    )


__all__ = [
    "SUBTB_HUBER_GRAD_DELTA",
    "SUBTB_HUBER_GRAD_ID",
    "SUBTB_HUBER_GRAD_SUFFIX",
    "SUBTB_RESIDUAL_ID",
    "SUBTB_WEIGHT_NORMALIZATION",
    "SubTBTerms",
    "subtb_pair_count",
    "subtb_terms",
]
