from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Literal

from skillev.contracts import JsonValue, normalize_json


@dataclass(frozen=True)
class QualityRule:
    metric: str
    minimum: float
    maximum_drop: float
    baseline_minimum: float | None = None

    def __post_init__(self) -> None:
        if not self.metric or not all(math.isfinite(v) for v in (self.minimum, self.maximum_drop)):
            raise ValueError("quality rule requires a named finite threshold")
        if self.maximum_drop < 0:
            raise ValueError("quality drop tolerance cannot be negative")
        if self.baseline_minimum is not None and (
            not math.isfinite(self.baseline_minimum) or self.baseline_minimum < self.minimum
        ):
            raise ValueError("baseline floor must be finite and at least the retention floor")


@dataclass(frozen=True)
class QualityGatePolicy:
    rule_version: str
    panel_id: str
    condition_id: str
    cadence: int
    minimum_source_questions: int
    consecutive_failures: int
    rules: tuple[QualityRule, ...]
    initial_admission_override: str | None = None
    baseline_step: int = 0
    architecture_id: str | None = None
    library_axis: Literal["fixed-initial-library", "combined-policy-library"] | None = None

    def __post_init__(self) -> None:
        if (self.architecture_id is None) != (self.library_axis is None) or (
            self.architecture_id is not None
            and (
                not self.architecture_id.strip()
                or self.library_axis not in {"fixed-initial-library", "combined-policy-library"}
            )
        ):
            raise ValueError("quality architecture and explicit library comparison axis are paired")
        if type(self.baseline_step) is not int or self.baseline_step < 0:
            raise ValueError("quality baseline step must be a nonnegative integer")
        if not all((self.rule_version, self.panel_id, self.condition_id)):
            raise ValueError("quality policy identities must be declared")
        if any(
            type(v) is not int or v < 1
            for v in (self.cadence, self.minimum_source_questions, self.consecutive_failures)
        ):
            raise ValueError("quality policy counts must be positive")
        if not self.rules or len({r.metric for r in self.rules}) != len(self.rules):
            raise ValueError("quality policy requires unique metric rules")
        if self.initial_admission_override is not None and (
            not isinstance(self.initial_admission_override, str)
            or not self.initial_admission_override.strip()
        ):
            raise ValueError("initial admission override requires an explicit owner reason")


@dataclass(frozen=True)
class ProtocolProbe:
    evidence_id: str
    panel_id: str
    condition_id: str
    policy_snapshot_id: str
    policy_step: int
    source_question_count: int
    metrics: dict[str, float | None]
    purpose: str = "diagnostic-collect-only"
    metric_notes: dict[str, str] = field(default_factory=dict)
    library_snapshot_id: str | None = None
    architecture_id: str | None = None
    execution_controls: dict[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        if self.purpose != "diagnostic-collect-only":
            raise ValueError("quality probes must never be training evidence")
        if self.policy_step < 0 or self.source_question_count < 0:
            raise ValueError("probe coordinates must be nonnegative")
        if not all((self.evidence_id, self.panel_id, self.condition_id, self.policy_snapshot_id)):
            raise ValueError("probe identities must be explicit")
        if any(
            v is not None and (type(v) not in (int, float) or not math.isfinite(v))
            for v in self.metrics.values()
        ):
            raise ValueError("probe metrics must be finite or explicitly missing")


@dataclass(frozen=True)
class QualityGateDecision:
    action: Literal["continue", "pause-after-commit"]
    status: Literal["not-due", "verified", "warning", "regressed", "metrics-missing"]
    policy_snapshot_id: str
    policy_step: int
    rule_version: str
    evidence_ids: tuple[str, ...]
    triggered_rules: tuple[str, ...]
    metrics_unavailable: tuple[str, ...]

    def to_value(self) -> dict[str, JsonValue]:
        value = normalize_json(asdict(self))
        assert isinstance(value, dict)
        return value


def evaluate_quality(
    policy: QualityGatePolicy,
    *,
    baseline: ProtocolProbe | None,
    probes: tuple[ProtocolProbe, ...],
    policy_step: int,
    policy_snapshot_id: str,
    library_snapshot_id: str | None = None,
) -> QualityGateDecision:
    if policy_step < 0 or not policy_snapshot_id:
        raise ValueError("current policy coordinate must be explicit")
    by_step: dict[int, ProtocolProbe] = {}
    for probe in probes:
        old = by_step.get(probe.policy_step)
        if old is not None and old != probe:
            raise ValueError("two different probes target the same policy step")
        by_step[probe.policy_step] = probe
    if policy_step < policy.baseline_step:
        raise ValueError("quality coordinate precedes this condition baseline")
    if policy_step != policy.baseline_step and policy_step % policy.cadence:
        return QualityGateDecision(
            "continue", "not-due", policy_snapshot_id, policy_step, policy.rule_version, (), (), ()
        )
    missing: list[str] = []

    def compatible(probe: ProtocolProbe | None, label: str) -> bool:
        if probe is None:
            missing.append(label)
            return False
        if (
            probe.panel_id != policy.panel_id
            or probe.condition_id != policy.condition_id
            or probe.source_question_count < policy.minimum_source_questions
        ):
            missing.append(label + "/panel-condition-or-sample-count")
            return False
        if policy.architecture_id is not None:
            if (
                probe.architecture_id != policy.architecture_id
                or not probe.library_snapshot_id
                or not probe.execution_controls
            ):
                missing.append(label + "/architecture-or-library-not-bound")
                return False
            if baseline is not None and (
                probe.execution_controls != baseline.execution_controls
                or (
                    policy.library_axis == "fixed-initial-library"
                    and probe.library_snapshot_id != baseline.library_snapshot_id
                )
            ):
                missing.append(label + "/execution-or-library-axis-changed")
                return False
        return True

    compatible(baseline, "baseline")
    if baseline is not None and baseline.policy_step != policy.baseline_step:
        missing.append("baseline/not-initial-policy")
    current = baseline if policy_step == policy.baseline_step else by_step.get(policy_step)
    compatible(current, "current")
    if current is not None and current.policy_snapshot_id != policy_snapshot_id:
        missing.append("current/stale-policy")
    if (
        current is not None
        and library_snapshot_id is not None
        and (current.library_snapshot_id != library_snapshot_id)
    ):
        missing.append("current/stale-library")
    evidence = []
    triggered: list[str] = []
    persistent: list[str] = []
    for rule in policy.rules:
        before = baseline.metrics.get(rule.metric) if baseline is not None else None
        now = current.metrics.get(rule.metric) if current is not None else None
        if before is None or now is None:
            missing.append(rule.metric)
            continue
        baseline_floor = rule.minimum if rule.baseline_minimum is None else rule.baseline_minimum
        if policy.initial_admission_override is not None and policy_step == policy.baseline_step:
            continue
        if before < baseline_floor and policy.initial_admission_override is None:
            triggered.append(rule.metric + "/baseline-below-floor")
            persistent.append(rule.metric)
            continue
        if now >= rule.minimum and before - now <= rule.maximum_drop:
            continue
        triggered.append(rule.metric)
        streak = 0
        for step in range(policy_step, policy.baseline_step, -policy.cadence):
            historical = by_step.get(step)
            if not compatible(historical, f"probe-{step}"):
                break
            assert historical is not None
            value = historical.metrics.get(rule.metric)
            if value is None:
                missing.append(f"probe-{step}/{rule.metric}")
                break
            evidence.append(historical.evidence_id)
            if value >= rule.minimum and before - value <= rule.maximum_drop:
                break
            streak += 1
            if streak >= policy.consecutive_failures:
                persistent.append(rule.metric)
                break
    evidence.extend(p.evidence_id for p in (baseline, current) if p is not None)
    status: Literal["verified", "warning", "regressed", "metrics-missing"] = (
        "metrics-missing"
        if missing
        else "regressed"
        if persistent
        else "warning"
        if triggered
        else "verified"
    )
    return QualityGateDecision(
        "pause-after-commit" if missing or persistent else "continue",
        status,
        policy_snapshot_id,
        policy_step,
        policy.rule_version,
        tuple(dict.fromkeys(evidence)),
        tuple(triggered),
        tuple(dict.fromkeys(missing)),
    )
