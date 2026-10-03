from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, cast

from .canonical import JsonValue, normalize_json, stable_hash
from .subtb import SUBTB_RESIDUAL_ID, SubTBTerms, subtb_pair_count, subtb_terms
from .ttb_common import (
    require_finite_number,
    require_iso_timestamp,
    require_non_empty_text,
)
from .ttb_training import EdgeScoreContext

R2FLOW_EDGE_RECORD_FORMAT: Final = "r2flow-edge-record@1"
R2FLOW_TRAJECTORY_RESIDUAL_FORMAT: Final = "r2flow-trajectory-residual@1"
R2FLOW_BATCH_STATS_FORMAT: Final = "r2flow-batch-stats@1"
RELATIVE_TOLERANCE: Final = 1e-12
ABSOLUTE_TOLERANCE: Final = 1e-9


def require_close_relative(
    actual: float,
    expected: float,
    *,
    field_name: str,
    rel_tol: float = RELATIVE_TOLERANCE,
    abs_tol: float = ABSOLUTE_TOLERANCE,
) -> None:
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise ValueError(f"{field_name} must be finite")
    if not math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol):
        raise ValueError(f"{field_name} is inconsistent with its component values")


def _object(value: object, *, label: str, fields: frozenset[str]) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or set(normalized) != fields:
        raise ValueError(f"{label} has incompatible fields")
    return normalized


def _number(value: object, field: str) -> float:
    return require_finite_number(value, field=field)


def _count(value: object, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _format(value: object, expected: str, label: str) -> None:
    if value != expected:
        raise ValueError(f"unsupported {label} format")


@dataclass(frozen=True, slots=True)
class R2FlowEdgeRecord:
    trajectory_id: str
    step_index: int
    context: EdgeScoreContext
    log_pf_reasoning: float
    log_pf_event: float
    log_q_reasoning: float
    log_pb_in_edge: float
    in_edge_count: int
    edge_log_ratio: float
    log_flow_source: float
    log_flow_target: float
    edge_residual: float
    edge_coefficient: float
    raw_logp_event: float
    log_mask_mass_event: float
    event_token_count: int
    forced_event_token_count: int
    reasoning_token_count: int
    reasoning_stopped: bool
    forward_adapter_version: str
    backward_adapter_version: str
    flow_head_version: str
    scoring_stack_id: str
    format: str = R2FLOW_EDGE_RECORD_FORMAT
    backward_scores: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        _format(self.format, R2FLOW_EDGE_RECORD_FORMAT, "R2 Flow edge record")
        for name in (
            "trajectory_id",
            "forward_adapter_version",
            "backward_adapter_version",
            "flow_head_version",
            "scoring_stack_id",
        ):
            require_non_empty_text(getattr(self, name), field=name)
        if not isinstance(self.context, EdgeScoreContext):
            raise TypeError("R2 Flow edge context must be EdgeScoreContext")
        _count(self.step_index, "step_index", 1)
        _count(self.in_edge_count, "in_edge_count", 1)
        _count(self.event_token_count, "event_token_count", 1)
        _count(self.forced_event_token_count, "forced_event_token_count")
        _count(self.reasoning_token_count, "reasoning_token_count")
        if self.forced_event_token_count > self.event_token_count:
            raise ValueError("forced event tokens exceed the event length")
        if self.context.action_token_count != self.event_token_count:
            raise ValueError("edge context and event token counts differ")
        if type(self.reasoning_stopped) is not bool:
            raise TypeError("reasoning_stopped must be boolean")
        for name in (
            "log_pf_reasoning",
            "log_pf_event",
            "log_q_reasoning",
            "log_pb_in_edge",
            "edge_log_ratio",
            "log_flow_source",
            "log_flow_target",
            "edge_residual",
            "edge_coefficient",
            "raw_logp_event",
            "log_mask_mass_event",
        ):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        for name in ("log_pf_reasoning", "log_pf_event", "log_q_reasoning", "log_pb_in_edge"):
            if getattr(self, name) > 0.0:
                raise ValueError(f"{name} is a log-probability and must be <= 0")
        if self.log_mask_mass_event > 0.0:
            raise ValueError("log_mask_mass_event must be <= 0")
        if self.in_edge_count == 1 and self.log_pb_in_edge != 0.0:
            raise ValueError("|In(s')| = 1 requires log P_B = 0 exactly")
        if not isinstance(self.backward_scores, tuple | list):
            raise TypeError("backward_scores must be a sequence of f_phi scores")
        object.__setattr__(
            self,
            "backward_scores",
            tuple(_number(v, "backward_scores") for v in self.backward_scores),
        )
        if self.backward_scores and len(self.backward_scores) != self.in_edge_count:
            raise ValueError("backward_scores must score every in-edge of In(s')")
        require_close_relative(
            self.log_pf_event,
            self.raw_logp_event - self.log_mask_mass_event,
            field_name="log_pf_event (masked log q = raw log p - log mask mass)",
            rel_tol=1e-9,
        )
        require_close_relative(
            self.edge_log_ratio,
            math.fsum(
                (
                    self.log_pf_reasoning,
                    self.log_pf_event,
                    -self.log_pb_in_edge,
                    -self.log_q_reasoning,
                )
            ),
            field_name="edge_log_ratio",
        )
        require_close_relative(
            self.edge_residual,
            math.fsum((self.log_flow_source, self.edge_log_ratio, -self.log_flow_target)),
            field_name="edge_residual",
        )

    def to_value(self) -> dict[str, JsonValue]:
        value: dict[str, JsonValue] = {
            "backward_adapter_version": self.backward_adapter_version,
            "context": self.context.to_value(),
            "edge_coefficient": self.edge_coefficient,
            "edge_log_ratio": self.edge_log_ratio,
            "edge_residual": self.edge_residual,
            "event_token_count": self.event_token_count,
            "flow_head_version": self.flow_head_version,
            "forced_event_token_count": self.forced_event_token_count,
            "format": self.format,
            "forward_adapter_version": self.forward_adapter_version,
            "in_edge_count": self.in_edge_count,
            "log_flow_source": self.log_flow_source,
            "log_flow_target": self.log_flow_target,
            "log_mask_mass_event": self.log_mask_mass_event,
            "log_pb_in_edge": self.log_pb_in_edge,
            "log_pf_event": self.log_pf_event,
            "log_pf_reasoning": self.log_pf_reasoning,
            "log_q_reasoning": self.log_q_reasoning,
            "raw_logp_event": self.raw_logp_event,
            "reasoning_stopped": self.reasoning_stopped,
            "reasoning_token_count": self.reasoning_token_count,
            "scoring_stack_id": self.scoring_stack_id,
            "step_index": self.step_index,
            "trajectory_id": self.trajectory_id,
            **({"backward_scores": list(self.backward_scores)} if self.backward_scores else {}),
        }
        return value

    @classmethod
    def from_value(cls, value: object) -> R2FlowEdgeRecord:
        normalized = normalize_json(value)
        scored = isinstance(normalized, dict) and "backward_scores" in normalized
        data = _object(
            value,
            label="R2 Flow edge record",
            fields=_EDGE_FIELDS | ({"backward_scores"} if scored else set()),
        )
        fields = {name: data[name] for name in _EDGE_FIELDS if name != "context"}
        scores = data.get("backward_scores", [])
        if not isinstance(scores, list):
            raise ValueError("backward_scores must be a list")
        return cls(
            context=EdgeScoreContext.from_value(data["context"]),
            backward_scores=tuple(scores),
            **fields,
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


_EDGE_FIELDS: Final = frozenset(
    {
        "backward_adapter_version",
        "context",
        "edge_coefficient",
        "edge_log_ratio",
        "edge_residual",
        "event_token_count",
        "flow_head_version",
        "forced_event_token_count",
        "format",
        "forward_adapter_version",
        "in_edge_count",
        "log_flow_source",
        "log_flow_target",
        "log_mask_mass_event",
        "log_pb_in_edge",
        "log_pf_event",
        "log_pf_reasoning",
        "log_q_reasoning",
        "raw_logp_event",
        "reasoning_stopped",
        "reasoning_token_count",
        "scoring_stack_id",
        "step_index",
        "trajectory_id",
    }
)


@dataclass(frozen=True, slots=True)
class R2FlowTrajectoryResidual:
    trajectory_id: str
    query_hash: str
    horizon: int
    log_z: float
    log_flows: tuple[float, ...]
    edge_log_ratios: tuple[float, ...]
    log_reward_eta: float
    raw_reward: float
    eta: float
    epsilon: float
    subtb_lambda: float
    delta_0T: float
    loss: float
    pair_count: int
    format: str = R2FLOW_TRAJECTORY_RESIDUAL_FORMAT

    def __post_init__(self) -> None:
        _format(self.format, R2FLOW_TRAJECTORY_RESIDUAL_FORMAT, "R2 Flow residual")
        require_non_empty_text(self.trajectory_id, field="trajectory_id")
        require_non_empty_text(self.query_hash, field="query_hash")
        _count(self.horizon, "horizon", 1)
        if not isinstance(self.log_flows, tuple) or len(self.log_flows) != self.horizon - 1:
            raise ValueError("R2 Flow residual needs T - 1 intermediate log flows")
        if not isinstance(self.edge_log_ratios, tuple) or len(self.edge_log_ratios) != (
            self.horizon
        ):
            raise ValueError("R2 Flow residual needs T edge log ratios")
        object.__setattr__(
            self, "log_flows", tuple(_number(v, "log_flows") for v in self.log_flows)
        )
        object.__setattr__(
            self,
            "edge_log_ratios",
            tuple(_number(v, "edge_log_ratios") for v in self.edge_log_ratios),
        )
        for name in (
            "log_z",
            "log_reward_eta",
            "raw_reward",
            "eta",
            "epsilon",
            "subtb_lambda",
            "delta_0T",
            "loss",
        ):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if not 0.0 <= self.raw_reward <= 1.0 or self.eta <= 0.0 or self.epsilon <= 0.0:
            raise ValueError("R2 Flow residual reward condition is invalid")
        require_close_relative(
            self.log_reward_eta,
            self.eta * math.log(self.raw_reward + self.epsilon),
            field_name="log_reward_eta",
        )
        if self.pair_count != subtb_pair_count(self.horizon):
            raise ValueError("pair_count must be T(T+1)/2")
        terms = self.terms
        if self.loss != terms.loss or self.delta_0T != terms.delta_0T:
            raise ValueError("R2 Flow residual loss/delta_0T differ from subtb_terms")

    @property
    def terms(self) -> SubTBTerms:
        return self.terms_with_gradient_clip(None)

    def terms_with_gradient_clip(self, gradient_clip: float | None) -> SubTBTerms:
        return subtb_terms(
            log_flows=(self.log_z, *self.log_flows, self.log_reward_eta),
            edge_log_ratios=self.edge_log_ratios,
            lam=self.subtb_lambda,
            gradient_clip=gradient_clip,
        )

    @classmethod
    def from_terms(
        cls,
        *,
        trajectory_id: str,
        query_hash: str,
        log_z: float,
        log_flows: tuple[float, ...],
        edge_log_ratios: tuple[float, ...],
        raw_reward: float,
        eta: float,
        epsilon: float,
        subtb_lambda: float,
    ) -> R2FlowTrajectoryResidual:
        log_reward_eta = eta * math.log(raw_reward + epsilon)
        terms = subtb_terms(
            log_flows=(log_z, *log_flows, log_reward_eta),
            edge_log_ratios=edge_log_ratios,
            lam=subtb_lambda,
        )
        return cls(
            trajectory_id=trajectory_id,
            query_hash=query_hash,
            horizon=len(edge_log_ratios),
            log_z=log_z,
            log_flows=log_flows,
            edge_log_ratios=edge_log_ratios,
            log_reward_eta=log_reward_eta,
            raw_reward=raw_reward,
            eta=eta,
            epsilon=epsilon,
            subtb_lambda=subtb_lambda,
            delta_0T=terms.delta_0T,
            loss=terms.loss,
            pair_count=len(terms.pairs),
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "delta_0T": self.delta_0T,
            "edge_log_ratios": list(self.edge_log_ratios),
            "epsilon": self.epsilon,
            "eta": self.eta,
            "format": self.format,
            "horizon": self.horizon,
            "log_flows": list(self.log_flows),
            "log_reward_eta": self.log_reward_eta,
            "log_z": self.log_z,
            "loss": self.loss,
            "pair_count": self.pair_count,
            "query_hash": self.query_hash,
            "raw_reward": self.raw_reward,
            "subtb_lambda": self.subtb_lambda,
            "trajectory_id": self.trajectory_id,
        }

    @classmethod
    def from_value(cls, value: object) -> R2FlowTrajectoryResidual:
        data = _object(value, label="R2 Flow residual", fields=_RESIDUAL_FIELDS)
        for name in ("log_flows", "edge_log_ratios"):
            if not isinstance(data[name], list):
                raise ValueError(f"{name} must be an array")
        fields: dict[str, object] = dict(data)
        fields["log_flows"] = tuple(cast(list[float], data["log_flows"]))
        fields["edge_log_ratios"] = tuple(cast(list[float], data["edge_log_ratios"]))
        return cls(**fields)

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


_RESIDUAL_FIELDS: Final = frozenset(
    {
        "delta_0T",
        "edge_log_ratios",
        "epsilon",
        "eta",
        "format",
        "horizon",
        "log_flows",
        "log_reward_eta",
        "log_z",
        "loss",
        "pair_count",
        "query_hash",
        "raw_reward",
        "subtb_lambda",
        "trajectory_id",
    }
)


@dataclass(frozen=True, slots=True)
class R2FlowBatchStats:
    batch_id: str
    optimizer_step: int
    library_version: str
    objective_id: str
    residuals: tuple[R2FlowTrajectoryResidual, ...]
    batch_loss: float
    mean_reward: float
    created_at: str
    format: str = R2FLOW_BATCH_STATS_FORMAT

    def __post_init__(self) -> None:
        _format(self.format, R2FLOW_BATCH_STATS_FORMAT, "R2 Flow batch stats")
        require_non_empty_text(self.batch_id, field="batch_id")
        require_non_empty_text(self.library_version, field="library_version")
        require_iso_timestamp(self.created_at, field="created_at")
        _count(self.optimizer_step, "optimizer_step")
        if self.objective_id != SUBTB_RESIDUAL_ID:
            raise ValueError("unsupported R2 Flow objective id")
        if (
            not isinstance(self.residuals, tuple)
            or not self.residuals
            or any(not isinstance(r, R2FlowTrajectoryResidual) for r in self.residuals)
        ):
            raise ValueError("R2 Flow batch stats need R2FlowTrajectoryResidual records")
        ids = [r.trajectory_id for r in self.residuals]
        if len(ids) != len(set(ids)):
            raise ValueError("R2 Flow batch stats repeat a trajectory")
        n = len(self.residuals)
        if self.batch_loss != math.fsum(r.loss for r in self.residuals) / n:
            raise ValueError("batch_loss must equal fsum(trajectory loss) / n exactly")
        if self.mean_reward != math.fsum(r.raw_reward for r in self.residuals) / n:
            raise ValueError("mean_reward must equal the mean raw reward exactly")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "batch_id": self.batch_id,
            "batch_loss": self.batch_loss,
            "created_at": self.created_at,
            "format": self.format,
            "library_version": self.library_version,
            "mean_reward": self.mean_reward,
            "objective_id": self.objective_id,
            "optimizer_step": self.optimizer_step,
            "residuals": [r.to_value() for r in self.residuals],
        }

    @classmethod
    def from_value(cls, value: object) -> R2FlowBatchStats:
        data = _object(value, label="R2 Flow batch stats", fields=_BATCH_FIELDS)
        residuals = data["residuals"]
        if not isinstance(residuals, list):
            raise ValueError("residuals must be an array")
        fields: dict[str, object] = dict(data)
        fields["residuals"] = tuple(R2FlowTrajectoryResidual.from_value(r) for r in residuals)
        return cls(**fields)

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


_BATCH_FIELDS: Final = frozenset(
    {
        "batch_id",
        "batch_loss",
        "created_at",
        "format",
        "library_version",
        "mean_reward",
        "objective_id",
        "optimizer_step",
        "residuals",
    }
)


__all__ = [
    "ABSOLUTE_TOLERANCE",
    "R2FLOW_BATCH_STATS_FORMAT",
    "R2FLOW_EDGE_RECORD_FORMAT",
    "R2FLOW_TRAJECTORY_RESIDUAL_FORMAT",
    "RELATIVE_TOLERANCE",
    "R2FlowBatchStats",
    "R2FlowEdgeRecord",
    "R2FlowTrajectoryResidual",
    "require_close_relative",
]
