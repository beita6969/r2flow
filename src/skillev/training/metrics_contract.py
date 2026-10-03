from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import cast

from skillev.contracts import JsonValue

from .metric_identity import native_benchmark

FORMAT = "skillev-training-metrics@3"
STATS_FORMAT = "r2flow-batch-stats@1"
DENOMINATORS: dict[str, JsonValue] = {
    "batch_success_fraction": "terminal success_count / trajectory_count",
    "reward_mean": "sum native projected reward / trajectory_count",
    "subtb_loss": (
        "sum_tau sum_{i<j} w_ij delta_ij**2 / trajectory_count, "
        "w = lambda**(j-i) normalised per trajectory over included pairs"
    ),
    "delta_0T_sq_mean": "sum raw summed delta_0T**2 / trajectory_count (Remark B.1, not loss)",
    "delta_0T_abs_max": "max |raw delta_0T| over trajectories",
    "edge_residual_abs_mean": "sum |delta_{t-1:t}| / edge_count",
    "reasoning_log_ratio_mean": "sum (log P_F(r) - log Q_phi(r)) / edge_count (summed nats)",
    "event_log_q_mean": "sum grammar-masked log q(e) / edge_count (summed nats)",
    "log_mask_mass_mean": "sum log sum_allowed p / edge_count (<= 0, leak)",
    "forced_token_fraction": "forced (single-legal) event tokens / event tokens",
    "multi_in_edge_fraction": "edges with |In(s')| > 1 / edge_count",
    "log_z_mean": "sum log Z(q) / trajectory_count",
    "log_psi_mean": "sum log F_psi(s_k), 0<k<T / intermediate state count; null if none",
}
NATIVE_METRICS = {
    "hotpotqa": ("answer-exact-match", "answer-f1"),
    "triviaqa": ("answer-exact-match", "answer-f1"),
    "aime-2026": ("accuracy",),
    "healthbench": ("judge-medium-api-rubric-score",),
    "mbpp-plus": ("base-pass", "plus-pass", "base-plus-pass@1"),
    "alfworld": (),
}


def _object(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError("metrics source must contain an object")
    return value


def _objects(value: JsonValue) -> list[dict[str, JsonValue]]:
    if not isinstance(value, list):
        raise ValueError("metrics source must contain an array")
    return [_object(item) for item in value]


def _number(value: JsonValue) -> float:
    if type(value) not in (int, float) or not math.isfinite(cast(float, value)):
        raise ValueError("metrics source must contain a finite number")
    return float(cast(float, value))


def _text(value: JsonValue) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("metrics identity must be nonempty text")
    return value


def _count(value: JsonValue) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("committed step/horizon/token count must be positive")
    return value


def _success(record: dict[str, JsonValue]) -> bool:
    value = _object(record["reward"])["success"]
    if type(value) is not bool:
        raise ValueError("terminal success must be the native boolean label")
    return value


@dataclass(frozen=True)
class TrainingMetricsSnapshot:
    run_id: str
    condition_id: str
    batch_id: str
    commit_id: str
    sampled_policy_id: str
    sampled_policy_step: int
    optimizer_step: int
    committed_at: str
    metrics: dict[str, JsonValue]
    native_metrics_by_domain: dict[str, JsonValue]

    @property
    def identity(self) -> tuple[str, str, str]:
        return self.run_id, self.batch_id, self.commit_id

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": FORMAT,
            "run_id": self.run_id,
            "condition_id": self.condition_id,
            "batch_id": self.batch_id,
            "commit_id": self.commit_id,
            "sampled_policy_id": self.sampled_policy_id,
            "sampled_policy_step": self.sampled_policy_step,
            "optimizer_step": self.optimizer_step,
            "committed_at": self.committed_at,
            "metrics": self.metrics,
            "native_metrics_by_domain": self.native_metrics_by_domain,
            "denominator_definitions": DENOMINATORS,
            "smoothing": "none",
        }

    def wandb_values(self) -> dict[str, JsonValue]:
        return {
            "optimizer_step": self.optimizer_step,
            "sampled_policy_step": self.sampled_policy_step,
            "source_commit_id": self.commit_id,
            **{f"train/{name}": value for name, value in self.metrics.items()},
        }

    @classmethod
    def from_event(
        cls, event: dict[str, JsonValue], *, condition_id: str
    ) -> TrainingMetricsSnapshot:
        if event.get("event_type") != "training_step_committed":
            raise ValueError("metrics require a complete training commit, not a preview")
        payload = _object(event["payload"])
        records = _objects(payload["records"])
        stats = _object(payload["stats"])
        if stats.get("format") != STATS_FORMAT:
            raise ValueError("unknown training stats format")
        return _r2flow_snapshot(event, payload, records, stats, condition_id=condition_id)


def _native_summary(domain: str, records: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    summary: dict[str, JsonValue] = {
        "trajectory_count": len(records),
        "success_fraction": sum(_success(r) for r in records) / len(records),
        "reward_mean": math.fsum(_number(_object(r["reward"])["value"]) for r in records)
        / len(records),
    }
    for name in NATIVE_METRICS.get(domain, ()):
        values = []
        for record in records:
            native = _object(_object(record["reward"])["native_payload"])
            public = native.get("public_metrics", {})
            value = public.get(name) if isinstance(public, dict) else None
            if value is None and name == "passed":
                passed = native.get("passed")
                value = int(passed) if type(passed) is bool else None
            if value is not None:
                values.append(_number(value))
        summary[name] = math.fsum(values) / len(records) if len(values) == len(records) else None
        summary[f"{name}/observed_count"] = len(values)
    return summary


def _optimization_metrics(payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    report = payload.get("report")
    diagnostics = report.get("optimization_diagnostics") if isinstance(report, dict) else None
    if not isinstance(diagnostics, dict):
        return {}
    return {
        "stability_loss": diagnostics.get("stability_loss"),
        "total_loss": diagnostics.get("total_loss"),
        "gradient_groups": diagnostics.get("gradient_groups"),
    }


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _r2flow_snapshot(
    event: dict[str, JsonValue],
    payload: dict[str, JsonValue],
    records: list[dict[str, JsonValue]],
    stats: dict[str, JsonValue],
    *,
    condition_id: str,
) -> TrainingMetricsSnapshot:
    residuals = _objects(stats["residuals"])
    edges = _objects(payload["edge_records"])
    ids = [_text(record["trajectory_id"]) for record in records]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("metrics require a unique nonempty complete trajectory population")
    if ids != [item["trajectory_id"] for item in residuals]:
        raise ValueError("residual order differs from the committed population")
    count = len(records)
    losses = [_number(r["loss"]) for r in residuals]
    loss = math.fsum(losses) / count
    if loss != _number(stats["batch_loss"]):
        raise ValueError("source loss differs from the complete residual population")
    rewards = [_number(_object(record["reward"])["value"]) for record in records]
    reward_mean = math.fsum(rewards) / count
    if not math.isclose(reward_mean, _number(stats["mean_reward"]), abs_tol=1e-12):
        raise ValueError("source reward differs from the complete trajectory population")
    deltas = [_number(r["delta_0T"]) for r in residuals]
    horizons = [_count(r["horizon"]) for r in residuals]
    if len(edges) != sum(horizons):
        raise ValueError("edge records do not cover every committed edge")
    event_tokens = sum(_count(e["event_token_count"]) for e in edges)
    forced = sum(int(_number(e["forced_event_token_count"])) for e in edges)
    psi = [_number(v) for r in residuals for v in cast(list[JsonValue], r["log_flows"])]
    success_count = sum(_success(record) for record in records)
    groups: dict[str, list[dict[str, JsonValue]]] = defaultdict(list)
    for record in records:
        groups[native_benchmark(_object(_object(record["reward"])["native_payload"]))].append(
            record
        )
    step = _count(payload["optimizer_step"])
    return TrainingMetricsSnapshot(
        run_id=_text(event["run_id"]),
        condition_id=_text(condition_id),
        batch_id=_text(payload["batch_id"]),
        commit_id=_text(event["event_id"]),
        sampled_policy_id=_text(payload["policy_snapshot_before"]),
        sampled_policy_step=step - 1,
        optimizer_step=step,
        committed_at=_text(event["occurred_at"]),
        metrics={
            "trajectory_count": count,
            "success_count": success_count,
            "batch_success_fraction": success_count / count,
            "reward_sum": math.fsum(rewards),
            "reward_mean": reward_mean,
            "subtb_loss": loss,
            **_optimization_metrics(payload),
            "delta_0T_sq_mean": math.fsum(d * d for d in deltas) / count,
            "delta_0T_abs_max": max(abs(d) for d in deltas),
            "edge_residual_abs_mean": math.fsum(abs(_number(e["edge_residual"])) for e in edges)
            / len(edges),
            "reasoning_log_ratio_mean": math.fsum(
                _number(e["log_pf_reasoning"]) - _number(e["log_q_reasoning"]) for e in edges
            )
            / len(edges),
            "event_log_q_mean": math.fsum(_number(e["log_pf_event"]) for e in edges) / len(edges),
            "log_mask_mass_mean": math.fsum(_number(e["log_mask_mass_event"]) for e in edges)
            / len(edges),
            "forced_token_fraction": forced / event_tokens,
            "multi_in_edge_fraction": sum(_count(e["in_edge_count"]) > 1 for e in edges)
            / len(edges),
            "log_z_mean": math.fsum(_number(r["log_z"]) for r in residuals) / count,
            "log_psi_mean": _mean(psi),
            "horizon_mean": math.fsum(horizons) / count,
            "horizon_max": max(horizons),
            "edge_count": len(edges),
            "event_tokens": event_tokens,
        },
        native_metrics_by_domain={
            name: _native_summary(name, rows) for name, rows in sorted(groups.items())
        },
    )
