from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from .answer_writer import written_answer
from .canonical import JsonValue, stable_hash
from .r2flow_training import R2FlowEdgeRecord, R2FlowTrajectoryResidual
from .ttb_trajectory import TrajectoryRecord

FLOW_RECORD_FORMAT: Final = "r2flow-flow-record@1"
FLOW_STEP_EVENT_FORMAT: Final = "r2flow-flow-step-event@1"
REASONING_CONDITIONED: Final = "sampled-reasoning-conditioned@1"


@dataclass(frozen=True, slots=True)
class FlowEdgeRecord:
    step_index: int
    state_key: str
    predecessor_key: str
    in_edge_count: int
    actual_in_edge: int
    legal_event_set_sha256: str
    event_label: str
    event_function: str
    log_q_event: float
    raw_logp_event: float
    log_mask_mass_event: float
    log_pb_in_edge: float
    edge_log_ratio: float
    edge_residual: float
    edge_coefficient: float
    event_token_count: int
    forced_event_token_count: int
    reasoning_token_count: int
    executor_cost: JsonValue
    verifier_event_ids: tuple[str, ...]
    verifier_ids: tuple[str, ...]
    in_edges: tuple[tuple[str, str], ...] = ()
    backward_scores: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.in_edge_count < 1 or not 0 <= self.actual_in_edge < self.in_edge_count:
            raise ValueError("flow edge in-edge index out of range")
        if self.in_edge_count == 1 and self.log_pb_in_edge != 0.0:
            raise ValueError("in_degree 1 requires log P_B = 0")
        if self.in_edges and (
            len(self.in_edges) != self.in_edge_count
            or self.in_edges[self.actual_in_edge][0] != self.predecessor_key
        ):
            raise ValueError("flow edge in-edge order disagrees with |In| or the actual edge")
        if self.backward_scores is not None and (
            len(self.backward_scores) != self.in_edge_count
            or not all(math.isfinite(v) for v in self.backward_scores)
        ):
            raise ValueError("backward scores must be finite, one per in-edge")
        for name in (
            "log_q_event",
            "raw_logp_event",
            "log_mask_mass_event",
            "log_pb_in_edge",
            "edge_log_ratio",
            "edge_residual",
            "edge_coefficient",
        ):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"flow edge {name} must be finite")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "step_index": self.step_index,
            "state_key": self.state_key,
            "predecessor_key": self.predecessor_key,
            "in_edge_count": self.in_edge_count,
            "actual_in_edge": self.actual_in_edge,
            "legal_event_set_sha256": self.legal_event_set_sha256,
            "event_label": self.event_label,
            "event_function": self.event_function,
            "log_q_event": self.log_q_event,
            "raw_logp_event": self.raw_logp_event,
            "log_mask_mass_event": self.log_mask_mass_event,
            "log_pf_reasoning": None,
            "log_pb_in_edge": self.log_pb_in_edge,
            "log_q_reasoning": None,
            "edge_log_ratio": self.edge_log_ratio,
            "edge_residual": self.edge_residual,
            "edge_coefficient": self.edge_coefficient,
            "event_token_count": self.event_token_count,
            "forced_event_token_count": self.forced_event_token_count,
            "reasoning_token_count": self.reasoning_token_count,
            "executor_cost": self.executor_cost,
            "verifier_event_ids": list(self.verifier_event_ids),
            "verifier_ids": list(self.verifier_ids),
            "in_edges": [
                {"predecessor_key": pred, "label_hash": label} for pred, label in self.in_edges
            ],
            "backward_scores": None if self.backward_scores is None else list(self.backward_scores),
        }


@dataclass(frozen=True, slots=True)
class FlowStepRecord:
    trajectory_id: str
    task_id: str
    library_version: str
    state_map: str
    horizon: int
    raw_reward: float
    log_z: float
    log_flows: tuple[float, ...]
    log_reward_eta: float
    delta_0T: float
    loss: float
    edges: tuple[FlowEdgeRecord, ...]
    episode: Mapping[str, JsonValue]
    format: str = FLOW_RECORD_FORMAT

    def __post_init__(self) -> None:
        if self.format != FLOW_RECORD_FORMAT:
            raise ValueError("unsupported flow record format")
        if len(self.edges) != self.horizon or len(self.log_flows) != self.horizon - 1:
            raise ValueError("flow record needs T edges and T - 1 intermediate log flows")
        if [edge.step_index for edge in self.edges] != list(range(1, self.horizon + 1)):
            raise ValueError("flow record edges must be ordered by step")
        total = math.fsum(edge.edge_residual for edge in self.edges)
        if not math.isclose(total, self.delta_0T, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("unit-edge residuals do not telescope into delta_0T")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "trajectory_id": self.trajectory_id,
            "task_id": self.task_id,
            "library_version": self.library_version,
            "state_map": self.state_map,
            "horizon": self.horizon,
            "raw_reward": self.raw_reward,
            "log_z": self.log_z,
            "log_flows": list(self.log_flows),
            "log_reward_eta": self.log_reward_eta,
            "delta_0T": self.delta_0T,
            "loss": self.loss,
            "edges": [edge.to_value() for edge in self.edges],
            "episode": dict(self.episode),
            "reasoning_scoring": REASONING_CONDITIONED,
        }

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


def _verifier_rows(
    verifier_records: Sequence[Any] | None, step_index: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    rows = [r for r in verifier_records or () if getattr(r, "step_index", None) == step_index]
    events = tuple(sorted(str(r.event_id) for r in rows if hasattr(r, "event_id")))
    ids = tuple(sorted({str(c.verifier_id) for r in rows for c in getattr(r, "components", ())}))
    return events, ids


def build_flow_step_record(
    *,
    record: TrajectoryRecord,
    residual: R2FlowTrajectoryResidual,
    edges: Sequence[R2FlowEdgeRecord],
    task_id: str,
    library_version: str,
    termination: str,
    wall_seconds: float | None,
    verifier_records: Sequence[Any] | None = None,
    executor_costs: Mapping[int, JsonValue] | None = None,
) -> FlowStepRecord:
    if residual.trajectory_id != record.trajectory_id or len(edges) != record.horizon:
        raise ValueError("flow record inputs belong to different trajectories")
    rows = []
    reasoning_tokens = 0
    executor_calls = 0
    answer_executor_calls = 0
    for step, edge in zip(record.steps, edges, strict=True):
        flow = step.r2flow
        if flow is None or edge.step_index != step.index:
            raise ValueError("flow records require sigma-mode step-r2flow@1 records")
        events, ids = _verifier_rows(verifier_records, step.index)
        reasoning_tokens += edge.reasoning_token_count
        executor_calls += flow.event_function == "invoke_skill"
        answer_executor_calls += flow.event_function == "submit_answer" and _written(
            step.observation_text
        )
        rows.append(
            FlowEdgeRecord(
                step_index=step.index,
                state_key=flow.state_key,
                predecessor_key=flow.predecessor_key,
                in_edge_count=edge.in_edge_count,
                actual_in_edge=flow.actual_in_edge,
                legal_event_set_sha256=flow.legal_event_set_sha256,
                event_label=flow.event_label,
                event_function=flow.event_function,
                log_q_event=edge.log_pf_event,
                raw_logp_event=edge.raw_logp_event,
                log_mask_mass_event=edge.log_mask_mass_event,
                log_pb_in_edge=edge.log_pb_in_edge,
                edge_log_ratio=edge.edge_log_ratio,
                edge_residual=edge.edge_residual,
                edge_coefficient=edge.edge_coefficient,
                event_token_count=edge.event_token_count,
                forced_event_token_count=edge.forced_event_token_count,
                reasoning_token_count=edge.reasoning_token_count,
                executor_cost=(executor_costs or {}).get(step.index),
                verifier_event_ids=events,
                verifier_ids=ids,
                in_edges=flow.in_edges,
                backward_scores=getattr(edge, "backward_scores", ()) or None,
            )
        )
    costs = [c for c in (executor_costs or {}).values() if isinstance(c, dict)]
    executor_tokens = [c.get("output_tokens") for c in costs]
    return FlowStepRecord(
        trajectory_id=record.trajectory_id,
        task_id=task_id,
        library_version=library_version,
        state_map=record.steps[0].r2flow.state_map if record.steps[0].r2flow else "",
        horizon=record.horizon,
        raw_reward=residual.raw_reward,
        log_z=residual.log_z,
        log_flows=residual.log_flows,
        log_reward_eta=residual.log_reward_eta,
        delta_0T=residual.delta_0T,
        loss=residual.loss,
        edges=tuple(rows),
        episode={
            "policy_reasoning_tokens": reasoning_tokens,
            "policy_action_tokens": sum(e.event_token_count for e in edges),
            "executor_calls": executor_calls,
            "executor_tokens": (
                sum(int(t) for t in executor_tokens if type(t) is int)
                if costs and all(type(t) is int for t in executor_tokens)
                else None
            ),
            "wall_seconds": wall_seconds,
            "termination": termination,
            **({"answer_executor_calls": answer_executor_calls} if answer_executor_calls else {}),
        },
    )


def _written(observation_text: str) -> bool:
    try:
        return written_answer(json.loads(observation_text)) is not None
    except json.JSONDecodeError:
        return False


def flow_step_event_payload(
    *, batch_id: str, optimizer_step: int, records: Sequence[FlowStepRecord]
) -> dict[str, JsonValue]:
    return {
        "format": FLOW_STEP_EVENT_FORMAT,
        "batch_id": batch_id,
        "optimizer_step": optimizer_step,
        "records": [record.to_value() for record in records],
    }


__all__ = [
    "FLOW_RECORD_FORMAT",
    "FLOW_STEP_EVENT_FORMAT",
    "FlowEdgeRecord",
    "FlowStepRecord",
    "build_flow_step_record",
    "flow_step_event_payload",
]
