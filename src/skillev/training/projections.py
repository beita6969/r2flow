from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from skillev.contracts import JsonValue, PosteriorBatchUpdate, TrajectoryRecord
from skillev.contracts.r2flow_training import R2FlowBatchStats, R2FlowEdgeRecord
from skillev.diagnostics import (
    DiagnosticsConfig,
    DiagnosticsState,
    FreshDiagnosticsSegment,
    diagnostics_state_from_value,
    reset_diagnostics_segment,
)
from skillev.rollout import RolloutArtifact
from skillev.rollout.generator import RolloutTokenizerProtocol

from .evidence_context import TrajectoryEvidenceContext
from .posterior_state import (
    POSTERIOR_PROVENANCE_FORMAT,
    PosteriorEventProvenance,
    PosteriorEvidenceBatch,
)


@dataclass(frozen=True, slots=True)
class TrainingStepSource:
    batch_id: str
    optimizer_step: int
    artifacts: tuple[RolloutArtifact, ...]
    stats: R2FlowBatchStats
    edge_records: tuple[R2FlowEdgeRecord, ...]

    def __post_init__(self) -> None:
        if self.stats.batch_id != self.batch_id or self.stats.optimizer_step != self.optimizer_step:
            raise ValueError("projection source differs from its batch statistics")
        if not self.artifacts:
            raise ValueError("projection requires a complete nonempty batch")
        trajectory_ids = tuple(artifact.record.trajectory_id for artifact in self.artifacts)
        if trajectory_ids != tuple(item.trajectory_id for item in self.stats.residuals):
            raise ValueError("projection residuals differ from the sealed trajectory order")
        if len(set(trajectory_ids)) != len(trajectory_ids):
            raise ValueError("projection repeats a trajectory")
        snapshot = self.artifacts[0].manifest.policy_snapshot
        for artifact in self.artifacts:
            if (
                artifact.manifest.policy_snapshot != snapshot
                or artifact.manifest.library_version != self.stats.library_version
            ):
                raise ValueError("projection mixes rollout policies or skill libraries")
        if any(
            edge.context is None
            or edge.context.policy_snapshot_id != snapshot.snapshot_id
            or edge.forward_adapter_version != snapshot.forward_adapter_version
            for edge in self.edge_records
        ):
            raise ValueError("projection scores differ from the rollout policy snapshot")

    @property
    def policy_snapshot_id(self) -> str:
        return self.artifacts[0].manifest.policy_snapshot.snapshot_id

    @property
    def records(self) -> tuple[TrajectoryRecord, ...]:
        return tuple(artifact.record for artifact in self.artifacts)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "artifacts": [artifact.to_value() for artifact in self.artifacts],
            "batch_id": self.batch_id,
            "edge_records": [edge.to_value() for edge in self.edge_records],
            "optimizer_step": self.optimizer_step,
            "stats": self.stats.to_value(),
        }

    @classmethod
    def from_value(
        cls,
        value: object,
        *,
        tokenizer: RolloutTokenizerProtocol,
    ) -> TrainingStepSource:
        if not isinstance(value, dict) or set(value) != {
            "artifacts",
            "batch_id",
            "edge_records",
            "optimizer_step",
            "stats",
        }:
            raise ValueError("TrainingStepSource has incompatible fields")
        artifacts = value["artifacts"]
        edge_records = value["edge_records"]
        batch_id = value["batch_id"]
        optimizer_step = value["optimizer_step"]
        if not isinstance(artifacts, list) or not isinstance(edge_records, list):
            raise ValueError("TrainingStepSource arrays have wrong types")
        if not isinstance(batch_id, str) or not batch_id:
            raise ValueError("TrainingStepSource batch_id must be non-empty text")
        if type(optimizer_step) is not int or optimizer_step < 1:
            raise ValueError("TrainingStepSource optimizer_step must be positive")
        return cls(
            batch_id=batch_id,
            optimizer_step=optimizer_step,
            artifacts=tuple(
                RolloutArtifact.from_value(item, tokenizer=tokenizer) for item in artifacts
            ),
            stats=R2FlowBatchStats.from_value(value["stats"]),
            edge_records=tuple(R2FlowEdgeRecord.from_value(item) for item in edge_records),
        )


PROJECTION_FORMAT = "skillev-full-projection-runtime-state@7"


@dataclass(frozen=True, slots=True)
class FullProjectionRuntimeState:
    diagnostics_state: DiagnosticsState
    posterior_provenance: PosteriorEventProvenance
    retained_batch_count: int
    revision: int = 0
    kind: str = "full"
    format: str = PROJECTION_FORMAT

    def __post_init__(self) -> None:
        if self.kind != "full" or self.format != PROJECTION_FORMAT:
            raise ValueError("unsupported full projection runtime state")
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("projection revision must be nonnegative")
        if type(self.retained_batch_count) is not int or self.retained_batch_count < 1:
            raise ValueError("retained_batch_count must be positive")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "revision": self.revision,
            "calibration_cells": [],
            "current_segment_batches": [],
            "diagnostics_state": self.diagnostics_state.to_value(),
            "format": self.format,
            "kind": self.kind,
            "latest_diagnostic": {"kind": "none"},
            "posterior_provenance": self.posterior_provenance.to_value(),
            "retained_batch_count": self.retained_batch_count,
        }

    @classmethod
    def from_value(cls, value: object) -> FullProjectionRuntimeState:
        expected = {
            "revision",
            "calibration_cells",
            "current_segment_batches",
            "diagnostics_state",
            "format",
            "kind",
            "latest_diagnostic",
            "posterior_provenance",
            "retained_batch_count",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("FullProjectionRuntimeState has incompatible fields")
        if (
            value["calibration_cells"] != []
            or value["current_segment_batches"] != []
            or value["latest_diagnostic"] != {"kind": "none"}
        ):
            raise ValueError("R2 Flow projections carry no diagnostics or calibration cells")
        retained = value["retained_batch_count"]
        if type(retained) is not int:
            raise TypeError("retained_batch_count must be an integer")
        return cls(
            revision=value["revision"],
            diagnostics_state=diagnostics_state_from_value(value["diagnostics_state"]),
            posterior_provenance=PosteriorEventProvenance.from_value(value["posterior_provenance"]),
            retained_batch_count=retained,
            kind=value["kind"],
            format=value["format"],
        )


@dataclass(frozen=True, slots=True)
class ProjectionTransition:
    source: TrainingStepSource
    next_state: FullProjectionRuntimeState
    posterior_batch: PosteriorBatchUpdate
    base_revision: int

    def __post_init__(self) -> None:
        if (
            self.next_state.revision != self.base_revision + 1
            or self.posterior_batch.batch_id != self.source.batch_id
        ):
            raise ValueError("R2 Flow projection transition must be an identity-tagged no-op")


class TrainingProjectionPipeline(Protocol):
    def preview(self, source: TrainingStepSource) -> ProjectionTransition: ...

    def commit(self, transition: ProjectionTransition) -> None: ...

    def reset_library_segment(
        self,
        old_library_version: str,
        new_library_version: str,
    ) -> None: ...

    @property
    def diagnostics_state(self) -> DiagnosticsState: ...

    def runtime_state(self) -> FullProjectionRuntimeState: ...


class MethodProjectionPipeline:
    def __init__(self, *, state: FullProjectionRuntimeState) -> None:
        self._state = state

    @classmethod
    def fresh(
        cls,
        *,
        diagnostics_config: DiagnosticsConfig,
        library_version: str,
    ) -> MethodProjectionPipeline:
        return cls(
            state=FullProjectionRuntimeState(
                diagnostics_state=FreshDiagnosticsSegment(expected_library_version=library_version),
                posterior_provenance=PosteriorEventProvenance.empty(),
                retained_batch_count=max(1, 2 * diagnostics_config.window_size),
            )
        )

    @classmethod
    def from_runtime_state(
        cls,
        *,
        diagnostics_config: DiagnosticsConfig,
        state: FullProjectionRuntimeState,
    ) -> MethodProjectionPipeline:
        if state.retained_batch_count != max(1, 2 * diagnostics_config.window_size):
            raise ValueError("retained_batch_count differs from diagnostics config")
        return cls(state=state)

    @property
    def diagnostics_state(self) -> DiagnosticsState:
        return self._state.diagnostics_state

    @property
    def posterior_provenance(self) -> PosteriorEventProvenance:
        return self._state.posterior_provenance

    def preview(self, source: TrainingStepSource) -> ProjectionTransition:
        state = self._state
        trajectory_ids = tuple(record.trajectory_id for record in source.records)
        state.posterior_provenance.require_new_batch(
            source.batch_id, source.optimizer_step, trajectory_ids
        )
        posterior_batch = PosteriorBatchUpdate(source.batch_id)
        provenance = state.posterior_provenance.append_batch(
            PosteriorEvidenceBatch(
                optimizer_step=source.optimizer_step,
                policy_snapshot_id=source.policy_snapshot_id,
                library_version=source.stats.library_version,
                trajectory_ids=trajectory_ids,
                posterior=posterior_batch,
                trajectory_contexts=tuple(
                    TrajectoryEvidenceContext.from_artifact(item) for item in source.artifacts
                ),
            )
        )
        next_state = FullProjectionRuntimeState(
            revision=state.revision + 1,
            diagnostics_state=state.diagnostics_state,
            posterior_provenance=provenance,
            retained_batch_count=state.retained_batch_count,
        )
        return ProjectionTransition(
            base_revision=state.revision,
            source=source,
            next_state=next_state,
            posterior_batch=posterior_batch,
        )

    def commit(self, transition: ProjectionTransition) -> None:
        if (
            transition.base_revision != self._state.revision
            or transition.next_state.revision != self._state.revision + 1
        ):
            raise ValueError("projection transition was prepared from an old state")
        self._state = transition.next_state

    def reset_library_segment(
        self,
        old_library_version: str,
        new_library_version: str,
    ) -> None:
        self._state = FullProjectionRuntimeState(
            revision=self._state.revision + 1,
            diagnostics_state=reset_diagnostics_segment(
                self._state.diagnostics_state,
                old_library_version=old_library_version,
                new_library_version=new_library_version,
            ),
            posterior_provenance=self._state.posterior_provenance,
            retained_batch_count=self._state.retained_batch_count,
        )

    def runtime_state(self) -> FullProjectionRuntimeState:
        return self._state


__all__ = [
    "POSTERIOR_PROVENANCE_FORMAT",
    "FullProjectionRuntimeState",
    "MethodProjectionPipeline",
    "PosteriorEventProvenance",
    "ProjectionTransition",
    "TrainingProjectionPipeline",
    "TrainingStepSource",
]
