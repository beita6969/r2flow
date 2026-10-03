from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

from skillev.contracts import JsonValue
from skillev.rollout.codec import codec_for_initial_meta
from skillev.runtime.execution import ActionParseStatus
from skillev.training.action_metrics import ActionMetrics
from skillev.training.metrics_contract import _count, _native_summary, _object, _objects, _text
from skillev.training.quality_gate import ProtocolProbe
from skillev.training.source_group_metrics import (
    SOURCE_GROUP_NOTE,
    TERMINAL_NOTE,
    source_group_summary,
)

if TYPE_CHECKING:
    from skillev.rollout import RolloutArtifact

    from .zero_update_bridge import ReadOnlyCollectionResult


@dataclass(frozen=True)
class PanelSlot:
    task_id: str
    benchmark_id: str
    population_id: str
    source_question_id: str

    @property
    def canonical_source(self) -> tuple[str, str]:
        return self.benchmark_id, self.source_question_id

    @property
    def source(self) -> tuple[str, str, str]:
        return self.benchmark_id, self.population_id, self.source_question_id


@dataclass(frozen=True)
class FixedQualityPanel:
    panel_id: str
    condition_id: str
    slots: tuple[PanelSlot, ...]

    def __post_init__(self) -> None:
        if not self.panel_id or not self.condition_id or not self.slots:
            raise ValueError("quality panel requires an identity and complete population")
        if len({s.task_id for s in self.slots}) != len(self.slots):
            raise ValueError("panel rollout slots must have unique task IDs")
        if any(not all((s.task_id, *s.source)) for s in self.slots):
            raise ValueError("panel source coordinates must be known")

    def require_disjoint(self, excluded_sources: frozenset[tuple[str, str, str]]) -> None:
        canonical = {(domain, source) for domain, _, source in excluded_sources}
        if any(slot.canonical_source in canonical for slot in self.slots):
            raise ValueError("quality panel overlaps an excluded source population")


def collection_probe(
    result: ReadOnlyCollectionResult,
    *,
    panel: FixedQualityPanel,
    policy_step: int,
    evidence_id: str,
    events_path: Path,
    event_run_id: str,
) -> ProtocolProbe:
    artifacts = result.diagnostic_artifacts
    if result.condition.condition_id != panel.condition_id or tuple(
        a.manifest.task_id for a in artifacts
    ) != tuple(s.task_id for s in panel.slots):
        raise ValueError("collection is not the complete frozen panel condition")
    if len({a.record.trajectory_id for a in artifacts}) != len(artifacts):
        raise ValueError("panel contains duplicate trajectory evidence")
    probe = _artifact_probe(
        artifacts,
        panel=panel,
        policy_snapshot_id=result.policy_snapshot_id,
        policy_step=policy_step,
        evidence_id=evidence_id,
        origins=tuple((events_path, event_run_id) for _ in artifacts),
    )
    return replace(
        probe,
        library_snapshot_id=result.library_snapshot_id,
        architecture_id=result.architecture_id,
        execution_controls=result.execution_controls,
    )


def _artifact_probe(
    artifacts: tuple[RolloutArtifact, ...],
    *,
    panel: FixedQualityPanel,
    policy_snapshot_id: str,
    policy_step: int,
    evidence_id: str,
    origins: tuple[tuple[Path, str], ...],
) -> ProtocolProbe:
    if len(origins) != len(artifacts):
        raise ValueError("every quality observation requires its original event source")
    groups: dict[str, list[dict[str, JsonValue]]] = {}
    records = []
    for artifact, slot in zip(artifacts, panel.slots, strict=True):
        if artifact.manifest.policy_snapshot.snapshot_id != policy_snapshot_id:
            raise ValueError("panel contains another policy snapshot")
        record = artifact.record.to_value()
        native = artifact.record.reward.native_payload
        source = (
            native.get("evaluation_evidence_source", native.get("training_evidence_source"))
            if isinstance(native, dict)
            else None
        )
        if (
            not isinstance(source, dict)
            or tuple(
                source.get(key) for key in ("benchmark_id", "population_id", "source_question_id")
            )
            != slot.source
        ):
            raise ValueError("panel evaluator source differs from the declared source")
        if native.get("benchmark_id") != slot.benchmark_id:
            raise ValueError("panel evaluator domain differs from its source")
        records.append(record)
        groups.setdefault(slot.benchmark_id, []).append(record)

    metrics: dict[str, float | None] = {}
    with sqlite3.connect(":memory:") as connection:
        assessments = ActionMetrics(connection)
        unique_origins = tuple(dict.fromkeys(origins))
        join_names = {origin: f"quality-origin-{i}" for i, origin in enumerate(unique_origins)}
        record_origins = {
            id(record): join_names[origin] for record, origin in zip(records, origins, strict=True)
        }
        for origin, join_name in join_names.items():
            path, original_run = origin
            if path.exists():
                with path.open(encoding="utf-8") as stream:
                    for line in stream:
                        event = json.loads(line)
                        if (
                            event.get("run_id") == original_run
                            and isinstance(event.get("payload"), dict)
                            and "assessment" in event["payload"]
                        ):
                            assessments.observe({**event, "run_id": join_name})

        def counts_for(selected: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
            by_origin: dict[str, list[dict[str, JsonValue]]] = {}
            for record in selected:
                by_origin.setdefault(record_origins[id(record)], []).append(record)
            counts = [assessments.for_records(name, rows) for name, rows in by_origin.items()]
            return {
                name: None
                if any(row[name] is None for row in counts)
                else sum(cast(int, row[name]) for row in counts)
                for name in counts[0]
            }

        source_observations = [
            (
                slot.canonical_source,
                _record_indicators(
                    record, slot.benchmark_id, assessments, record_origins[id(record)]
                ),
            )
            for record, slot in zip(records, panel.slots, strict=True)
        ]
        for domain, selected in (("panel", records), *sorted(groups.items())):
            steps = [step for row in selected for step in _objects(row["steps"])]
            statuses = [
                codec_for_initial_meta(_object(_object(row["initial_context"])["meta"]))
                .parse(_text(step["action_text"]))
                .status
                for row in selected
                for step in _objects(row["steps"])
            ]
            valid = sum(status is ActionParseStatus.VALID for status in statuses)
            first = sum(
                codec_for_initial_meta(_object(_object(row["initial_context"])["meta"]))
                .parse(_text(_objects(row["steps"])[0]["action_text"]))
                .status
                is ActionParseStatus.VALID
                for row in selected
            )
            prefix = domain + "/"
            metrics.update(
                {
                    prefix + "trajectory_count": float(len(selected)),
                    prefix + "action_count": float(len(steps)),
                    prefix + "action_structure_valid_fraction": valid / len(steps),
                    prefix + "first_turn_structure_valid_fraction": first / len(selected),
                    prefix + "horizon_mean": len(steps) / len(selected),
                    prefix + "horizon_max": float(max(_count(row["horizon"]) for row in selected)),
                }
            )
            for name, value in _native_summary(domain, selected).items():
                metrics[prefix + name] = None if value is None else float(cast(float, value))
            counts = counts_for(selected)
            for name, value in counts.items():
                metrics[prefix + name] = None if value is None else float(cast(int, value))
            terminal = cast(int | None, counts["valid_terminal_record_count"])
            metrics[prefix + "valid_terminal_record_fraction"] = (
                None if terminal is None else terminal / len(selected)
            )
            observations = [
                (source, values)
                for source, values in source_observations
                if domain == "panel" or source[0] == domain
            ]
            if domain == "panel":
                observations = [
                    (source, {k: v for k, v in values.items() if k in _COMMON_INDICATORS})
                    for source, values in observations
                ]
            metrics.update(
                {
                    prefix + name: value
                    for name, value in source_group_summary(
                        observations,
                        allow_standard_error=len({source[0] for source, _ in observations}) == 1,
                    ).items()
                }
            )
            admitted = cast(int | None, counts["admitted_count"])
            executed = cast(int | None, counts["executed_count"])
            returned = cast(int | None, counts["execution_returned_success_count"])
            metrics[prefix + "admitted_fraction"] = (
                None if admitted is None else admitted / len(steps)
            )
            metrics[prefix + "admitted_given_structural_fraction"] = (
                None if admitted is None or not valid else admitted / valid
            )
            metrics[prefix + "execution_returned_success_fraction"] = (
                None if returned is None or not executed else returned / executed
            )
    metrics["panel/reward_sum"] = math.fsum(a.record.reward.value for a in artifacts)
    return ProtocolProbe(
        evidence_id=evidence_id,
        panel_id=panel.panel_id,
        condition_id=panel.condition_id,
        policy_snapshot_id=policy_snapshot_id,
        policy_step=policy_step,
        source_question_count=len({s.canonical_source for s in panel.slots}),
        metrics=metrics,
        metric_notes={"source_groups": SOURCE_GROUP_NOTE, "valid_terminal_records": TERMINAL_NOTE},
    )


_COMMON_INDICATORS = frozenset(
    {
        "success_fraction",
        "reward_mean",
        "action_structure_valid_fraction",
        "first_turn_structure_valid_fraction",
        "horizon_mean",
        "admitted_fraction",
        "valid_terminal_record_fraction",
    }
)


def _record_indicators(
    record: dict[str, JsonValue],
    domain: str,
    assessments: ActionMetrics,
    run_id: str,
) -> dict[str, float | None]:
    values = {
        name: None if value is None else float(cast(float, value))
        for name, value in _native_summary(domain, [record]).items()
        if name != "trajectory_count" and not name.endswith("/observed_count")
    }
    steps = _objects(record["steps"])
    codec = codec_for_initial_meta(_object(_object(record["initial_context"])["meta"]))
    valid = [
        codec.parse(_text(step["action_text"])).status is ActionParseStatus.VALID for step in steps
    ]
    counts = assessments.for_records(run_id, [record])
    admitted = cast(int | None, counts["admitted_count"])
    terminal = cast(int | None, counts["valid_terminal_record_count"])
    values.update(
        action_structure_valid_fraction=sum(valid) / len(steps),
        first_turn_structure_valid_fraction=float(valid[0]),
        horizon_mean=float(len(steps)),
        admitted_fraction=None if admitted is None else admitted / len(steps),
        valid_terminal_record_fraction=None if terminal is None else float(terminal),
    )
    return values
