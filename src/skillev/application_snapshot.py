from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from skillev.policy import PolicyBackbone
from skillev.runtime import (
    AttemptRunCursorState,
    FullRuntimeExecutionState,
    OrderedTaskCursorState,
    RuntimeSnapshotIdentity,
    SkillLibrary,
)
from skillev.training.checkpoint import TrainingCheckpointSnapshot

if TYPE_CHECKING:
    import torch

    from skillev.evolution.detector import DetectorRuntimeState
    from skillev.training.config import TrainerConfig
    from skillev.training.projections import FullProjectionRuntimeState, PosteriorEventProvenance
    from skillev.training.r2flow_library_versions import PhaseCarrier


class _SnapshotTrainingLoop(Protocol):
    @property
    def optimizer_step(self) -> int: ...

    @property
    def config(self) -> TrainerConfig: ...

    @property
    def backbone(self) -> PolicyBackbone: ...

    @property
    def optimizer(self) -> torch.optim.Optimizer: ...


class _SnapshotTaskProvider(Protocol):
    @property
    def runtime_state(self) -> OrderedTaskCursorState: ...


class _SnapshotRunProgress(Protocol):
    @property
    def state(self) -> AttemptRunCursorState: ...


class _SnapshotProjections(Protocol):
    @property
    def posterior_provenance(self) -> PosteriorEventProvenance: ...

    def runtime_state(self) -> FullProjectionRuntimeState: ...


class _SnapshotDetector(Protocol):
    @property
    def state(self) -> DetectorRuntimeState: ...


@dataclass(frozen=True, slots=True)
class ApplicationSnapshotFactory:
    training_loop: _SnapshotTrainingLoop
    task_provider: _SnapshotTaskProvider
    run_progress: _SnapshotRunProgress
    snapshot_identity: RuntimeSnapshotIdentity
    library: SkillLibrary
    projections: _SnapshotProjections
    detector: _SnapshotDetector
    phase_carrier: PhaseCarrier | None = None

    def snapshot(self) -> TrainingCheckpointSnapshot:
        task_cursor = self.task_provider.runtime_state
        state = FullRuntimeExecutionState(
            task_cursor=task_cursor,
            run_cursor=self.run_progress.state,
            library=self.library.state,
            projections=self.projections.runtime_state(),
            detector=self.detector.state,
            r2flow_phase=None if self.phase_carrier is None else self.phase_carrier.value,
        )
        require_full_execution_coherence(state, self.training_loop.optimizer_step)
        return TrainingCheckpointSnapshot(
            optimizer_step=self.training_loop.optimizer_step,
            experiment_id=self.training_loop.config.execution.experiment_id,
            identity=self.snapshot_identity,
            backbone=self.training_loop.backbone,
            optimizer=self.training_loop.optimizer,
            execution_state=state,
        )


def require_full_execution_coherence(state: FullRuntimeExecutionState, optimizer_step: int) -> None:
    batches = state.projections.posterior_provenance.batches
    posterior_step = batches[-1].optimizer_step if batches else 0
    if (
        posterior_step != optimizer_step
        or state.run_cursor.completed_training_steps != optimizer_step
    ):
        raise ValueError("model, posterior and run cursor belong to different training steps")


__all__ = ["ApplicationSnapshotFactory"]
