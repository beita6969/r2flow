from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from skillev.contracts import JsonValue

from .execution_state import (
    FullRuntimeExecutionState,
)
from .snapshot_identity import RuntimeSnapshotIdentity

if TYPE_CHECKING:
    from skillev.training.checkpoint import TrainingCheckpointSnapshot

RUNTIME_SNAPSHOT_FORMAT = "skillev-runtime-snapshot@6"


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    optimizer_step: int
    experiment_id: str
    identity: RuntimeSnapshotIdentity
    policy_directory: str
    optimizer_file: str
    execution_state: FullRuntimeExecutionState
    format: str = RUNTIME_SNAPSHOT_FORMAT

    def __post_init__(self) -> None:
        if type(self.optimizer_step) is not int or self.optimizer_step < 0:
            raise ValueError("optimizer_step must be non-negative")
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must be non-empty")
        if not isinstance(self.identity, RuntimeSnapshotIdentity):
            raise TypeError("snapshot requires RuntimeSnapshotIdentity")
        if self.policy_directory != "policy":
            raise ValueError("policy_directory is fixed to 'policy'")
        if self.optimizer_file != "optimizer.pt":
            raise ValueError("optimizer_file is fixed to 'optimizer.pt'")
        if not isinstance(self.execution_state, FullRuntimeExecutionState):
            raise TypeError("execution_state must be FullRuntimeExecutionState")
        if self.execution_state.run_cursor.run_plan_hash != self.identity.run_plan_hash:
            raise ValueError("snapshot run cursor differs from identity run plan")
        if self.format != RUNTIME_SNAPSHOT_FORMAT:
            raise ValueError("unsupported runtime snapshot format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "execution_state": self.execution_state.to_value(),
            "experiment_id": self.experiment_id,
            "identity": self.identity.to_value(),
            "format": self.format,
            "optimizer_file": self.optimizer_file,
            "optimizer_step": self.optimizer_step,
            "policy_directory": self.policy_directory,
        }

    @classmethod
    def from_value(cls, value: object) -> RuntimeSnapshot:
        expected = {
            "execution_state",
            "experiment_id",
            "identity",
            "format",
            "optimizer_file",
            "optimizer_step",
            "policy_directory",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("RuntimeSnapshot has incompatible fields")
        if value["format"] != RUNTIME_SNAPSHOT_FORMAT:
            raise ValueError("snapshot has an incompatible format")
        optimizer_step = value["optimizer_step"]
        experiment_id = value["experiment_id"]
        policy_directory = value["policy_directory"]
        optimizer_file = value["optimizer_file"]
        format_name = value["format"]
        if type(optimizer_step) is not int:
            raise TypeError("optimizer_step must be an integer")
        if not all(
            isinstance(item, str)
            for item in (experiment_id, policy_directory, optimizer_file, format_name)
        ):
            raise TypeError("snapshot identity fields must be text")
        return cls(
            optimizer_step=optimizer_step,
            experiment_id=experiment_id,
            identity=RuntimeSnapshotIdentity.from_value(value["identity"]),
            policy_directory=policy_directory,
            optimizer_file=optimizer_file,
            execution_state=FullRuntimeExecutionState.from_value(value["execution_state"]),
            format=format_name,
        )


class SnapshotArtifactStore(Protocol):
    def save_as(self, snapshot: TrainingCheckpointSnapshot, *, name: str) -> Path: ...

    def load_metadata(self, directory: Path) -> RuntimeSnapshot: ...

    def retain_recent(self, *, keep_recent: int) -> tuple[Path, ...]: ...


class RuntimeSnapshotStore:
    def __init__(self, artifact_store: SnapshotArtifactStore) -> None:
        self._artifact_store = artifact_store

    @property
    def training_store(self) -> SnapshotArtifactStore:
        return self._artifact_store

    def save(self, snapshot: TrainingCheckpointSnapshot, *, name: str) -> str:
        return str(self._artifact_store.save_as(snapshot, name=name))

    def load_exact(self, directory: Path) -> RuntimeSnapshot:
        return self._artifact_store.load_metadata(directory)

    def retain_recent(self, *, keep_recent: int) -> tuple[Path, ...]:
        return self._artifact_store.retain_recent(keep_recent=keep_recent)


__all__ = [
    "RUNTIME_SNAPSHOT_FORMAT",
    "RuntimeSnapshot",
    "RuntimeSnapshotStore",
    "SnapshotArtifactStore",
]
