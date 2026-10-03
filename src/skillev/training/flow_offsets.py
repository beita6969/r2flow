from __future__ import annotations

import json
import math
import os
import statistics
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from skillev.contracts import JsonValue
from skillev.policy.flow_head import (
    FLOW_HEAD_DOMAIN_OFFSET_ID,
    FLOW_OFFSET_CAP_HORIZON_UNIT,
    FLOW_OFFSET_INITIALIZATION,
    FLOW_OFFSET_LEARNING_RATE,
    FLOW_OFFSET_STEP_CAP,
    FLOW_OFFSET_TRACKING_DAMPING,
)
from skillev.scoring.r2flow_objective import trajectory_domain

FLOW_OFFSET_RECORD_FILE = "flow-offset-initialization.json"
FLOW_OFFSET_RECORD_FORMAT = "flow-domain-offset-initialization@1"


def flow_offset_domains(rollout: Any) -> tuple[str, ...]:
    domains = tuple(domain for domain, _ in getattr(rollout, "reasoning_by_domain", ()))
    if not domains:
        raise ValueError("per-domain flow offsets need the declared training domains")
    return domains


def offsets_from_scores(
    artifacts: Sequence[Any], deltas: Sequence[float]
) -> tuple[dict[str, float], dict[str, JsonValue]]:
    if len(artifacts) != len(deltas) or not artifacts:
        raise ValueError("flow offset initialisation needs one delta per trajectory")
    grouped: dict[str, list[float]] = defaultdict(list)
    for artifact, delta in zip(artifacts, deltas, strict=True):
        grouped[trajectory_domain(artifact.record)].append(float(delta))
    offsets = {domain: -statistics.median(values) for domain, values in sorted(grouped.items())}
    summary: dict[str, JsonValue] = {
        domain: {
            "trajectories": len(values),
            "median_delta_0T_at_zero_offset": statistics.median(values),
            "offset": offsets[domain],
        }
        for domain, values in sorted(grouped.items())
    }
    return offsets, summary


def exact_offset_steps(
    trajectories: Sequence[tuple[str, float, Sequence[float], Sequence[float], float, float]],
) -> dict[str, float]:
    per_trajectory: dict[str, list[float]] = defaultdict(list)
    for domain, log_z, log_flows, ratios, log_reward_eta, lam in trajectories:
        horizon = len(ratios)
        if len(log_flows) != horizon - 1 or horizon < 1:
            raise ValueError("offset step needs T - 1 intermediate flows and T edge ratios")
        flows = (log_z, *log_flows)
        total = sum(lam ** (j - i) for i in range(horizon) for j in range(i + 1, horizon + 1))
        own_numerator = own_denominator = 0.0
        for i in range(horizon):
            residual = flows[i] + math.fsum(ratios[i:]) - log_reward_eta
            weight = lam ** (horizon - i) / total
            own_numerator += weight * residual
            own_denominator += weight
        per_trajectory[domain].append(-own_numerator / own_denominator)
    return {domain: statistics.median(values) for domain, values in per_trajectory.items()}


def tracked_offset_update(prepared: Any, domains: Sequence[str]) -> dict[str, float]:
    by_id = {
        artifact.manifest.trajectory_id: trajectory_domain(artifact.record)
        for artifact in prepared.batch.artifacts
    }
    steps = exact_offset_steps(
        [
            (
                by_id[r.trajectory_id],
                r.log_z,
                r.log_flows,
                r.edge_log_ratios,
                r.log_reward_eta,
                r.subtb_lambda,
            )
            for r in prepared.residuals
        ]
    )
    damped = {d: FLOW_OFFSET_TRACKING_DAMPING * steps[d] for d in domains if d in steps}
    horizons: dict[str, list[int]] = defaultdict(list)
    for r in prepared.residuals:
        horizons[by_id[r.trajectory_id]].append(len(r.edge_log_ratios))
    caps = {
        d: FLOW_OFFSET_STEP_CAP
        * max(1.0, statistics.median(horizons[d]) / FLOW_OFFSET_CAP_HORIZON_UNIT)
        for d in damped
    }
    return {d: max(-caps[d], min(caps[d], value)) for d, value in damped.items()}


def initialize_flow_offsets(
    *,
    backbone: Any,
    batch: Any,
    score: Callable[[tuple[Any, ...]], Sequence[float]],
    record_path: Path | None,
) -> dict[str, JsonValue]:
    if backbone.flow_offsets_initialization is not None:
        raise RuntimeError("flow offsets are already initialised")
    if record_path is not None and record_path.exists():
        raise RuntimeError("this run already recorded a flow offset initialisation")
    parameter = backbone.flow_offset_parameter
    if parameter is None or bool((parameter.detach() != 0).any()):
        raise RuntimeError("flow offsets must be zero before their initialisation")
    artifacts = tuple(batch.artifacts)
    offsets, summary = offsets_from_scores(artifacts, tuple(score(artifacts)))
    record: dict[str, JsonValue] = {
        "format": FLOW_OFFSET_RECORD_FORMAT,
        "flow_head": FLOW_HEAD_DOMAIN_OFFSET_ID,
        "rule": FLOW_OFFSET_INITIALIZATION,
        "learning_rate": FLOW_OFFSET_LEARNING_RATE,
        "optimizer_step": batch.optimizer_step,
        "batch_id": batch.batch_id,
        "policy_snapshot_id": batch.policy_snapshot_id,
        "domains": list(backbone.flow_offset_domains),
        "offsets": {domain: offsets.get(domain, 0.0) for domain in backbone.flow_offset_domains},
        "per_domain": summary,
    }
    backbone.initialize_flow_offsets(offsets, record)
    if record_path is not None:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(
            os.open(record_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w"
        ) as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    return record


__all__ = [
    "FLOW_OFFSET_RECORD_FILE",
    "FLOW_OFFSET_RECORD_FORMAT",
    "exact_offset_steps",
    "flow_offset_domains",
    "initialize_flow_offsets",
    "offsets_from_scores",
    "tracked_offset_update",
]
