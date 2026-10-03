from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

import torch

from skillev.contracts import JsonValue

FLOW_TRAJECTORY_SCORE_FORMAT: Final = "r2flow-heldout-trajectory-score@1"


@dataclass(frozen=True, slots=True)
class FlowTrajectoryScore:
    trajectory_id: str
    task_id: str
    delta_0T: float
    horizon: int
    format: str = FLOW_TRAJECTORY_SCORE_FORMAT

    def __post_init__(self) -> None:
        if self.format != FLOW_TRAJECTORY_SCORE_FORMAT:
            raise ValueError("unsupported held-out trajectory score format")
        if not self.trajectory_id or not self.task_id:
            raise ValueError("held-out score identities must be non-empty")
        if not math.isfinite(self.delta_0T):
            raise ValueError("held-out delta_0T must be finite")
        if type(self.horizon) is not int or self.horizon < 1:
            raise ValueError("held-out horizon must be a positive integer")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "delta_0T": self.delta_0T,
            "format": self.format,
            "horizon": self.horizon,
            "task_id": self.task_id,
            "trajectory_id": self.trajectory_id,
        }

    @classmethod
    def from_value(cls, value: object) -> FlowTrajectoryScore:
        if not isinstance(value, dict) or set(value) != {
            "delta_0T",
            "format",
            "horizon",
            "task_id",
            "trajectory_id",
        }:
            raise ValueError("held-out trajectory score has an incompatible field set")
        return cls(
            str(value["trajectory_id"]),
            str(value["task_id"]),
            float(value["delta_0T"]),
            int(value["horizon"]),
            str(value["format"]),
        )


class FlowTrajectoryScorer(Protocol):
    def __call__(
        self, backbone: Any, artifact: Any, method: Any, *, requires_grad: bool
    ) -> object: ...


def resolve_flow_scorer() -> FlowTrajectoryScorer:
    try:
        from skillev.training import step_math

        scorer = step_math.score_flow_trajectory
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "score_flow_trajectory (S32) is not integrated; V_q held-out scoring is unavailable"
        ) from error
    return scorer


def flow_trajectory_score(value: object, artifact: Any) -> FlowTrajectoryScore:
    if isinstance(value, tuple) and value:
        value = value[0]
    residual = getattr(value, "residual", None)
    if residual is not None and hasattr(residual, "delta_0T"):
        value = residual
    if isinstance(value, FlowTrajectoryScore):
        score = value
    else:
        raw: object
        if isinstance(value, Mapping):
            raw = value.get("delta_0T", value.get("delta_0t"))
        else:
            raw = getattr(value, "delta_0T", getattr(value, "delta_0t", None))
        if isinstance(raw, torch.Tensor):
            raw = float(raw.detach().to("cpu", torch.float64).item())
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            raise TypeError("flow scorer must expose a numeric delta_0T")
        score = FlowTrajectoryScore(
            artifact.manifest.trajectory_id,
            artifact.manifest.task_id,
            float(raw),
            artifact.record.horizon,
        )
    if (score.trajectory_id, score.task_id) != (
        artifact.manifest.trajectory_id,
        artifact.manifest.task_id,
    ):
        raise ValueError("flow score belongs to another trajectory")
    return score


def _grad_state(parameters: Sequence[torch.Tensor]) -> tuple[object, ...]:
    return tuple(None if p.grad is None else (id(p.grad), p.grad.data_ptr()) for p in parameters)


def score_flow_no_grad(
    backbone: Any,
    artifacts: Sequence[Any],
    method: Any,
    *,
    scorer: FlowTrajectoryScorer | None = None,
    parameters: Iterable[torch.Tensor] = (),
) -> tuple[FlowTrajectoryScore, ...]:
    resolved = resolve_flow_scorer() if scorer is None else scorer
    watched = tuple(parameters)
    before = _grad_state(watched)
    with torch.no_grad():
        scores = tuple(
            flow_trajectory_score(
                resolved(backbone, artifact, method, requires_grad=False), artifact
            )
            for artifact in artifacts
        )
    if _grad_state(watched) != before:
        raise RuntimeError("held-out scoring populated a parameter gradient")
    return scores


def token_cost_partition(costs: Sequence[int], worker_count: int) -> tuple[tuple[int, ...], ...]:
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    bins: list[list[int]] = [[] for _ in range(worker_count)]
    loads = [0] * worker_count
    for position in sorted(range(len(costs)), key=lambda i: (-costs[i], i)):
        target = min(range(worker_count), key=lambda r: (loads[r], r))
        bins[target].append(position)
        loads[target] += max(1, int(costs[position]))
    return tuple(tuple(sorted(b)) for b in bins)


ArtifactCost = Callable[[Any], int]


__all__ = [
    "FLOW_TRAJECTORY_SCORE_FORMAT",
    "ArtifactCost",
    "FlowTrajectoryScore",
    "FlowTrajectoryScorer",
    "flow_trajectory_score",
    "resolve_flow_scorer",
    "score_flow_no_grad",
    "token_cost_partition",
]
