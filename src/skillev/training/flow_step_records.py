from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from skillev.contracts import JsonValue
from skillev.contracts.flow_record import build_flow_step_record, flow_step_event_payload
from skillev.contracts.r2flow_training import R2FlowEdgeRecord

if TYPE_CHECKING:
    from .planning import CollectedTrainingBatch
    from .step_math import PreparedTTBStep


def _wall_seconds(started_at: str, completed_at: str) -> float | None:
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (end - start).total_seconds()


def r2flow_step_payload(
    batch: CollectedTrainingBatch,
    prepared: PreparedTTBStep,
) -> dict[str, JsonValue]:
    residuals = {r.trajectory_id: r for r in prepared.residuals}
    edges: dict[str, list[R2FlowEdgeRecord]] = {}
    for edge in prepared.edges:
        edges.setdefault(edge.trajectory_id, []).append(edge)
    records = []
    for artifact in batch.artifacts:
        record = artifact.record
        manifest = artifact.manifest
        records.append(
            build_flow_step_record(
                record=record,
                residual=residuals[record.trajectory_id],
                edges=sorted(edges[record.trajectory_id], key=lambda e: e.step_index),
                task_id=manifest.task_id,
                library_version=manifest.library_version,
                termination=manifest.termination.value,
                wall_seconds=_wall_seconds(manifest.started_at, manifest.completed_at),
                verifier_records=getattr(artifact, "verifier_records", None),
            )
        )
    return flow_step_event_payload(
        batch_id=batch.batch_id, optimizer_step=batch.optimizer_step, records=records
    )


__all__ = ["r2flow_step_payload"]
