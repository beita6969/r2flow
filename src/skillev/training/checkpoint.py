from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from skillev.contracts import canonical_json, stable_hash
from skillev.policy import AdapterRole, PolicyBackbone
from skillev.runtime.execution_state import FullRuntimeExecutionState
from skillev.runtime.runtime_snapshot import RuntimeSnapshot
from skillev.runtime.snapshot_identity import RuntimeSnapshotIdentity

from .optimizer_state import checkpoint_optimizer_state, require_optimizer_state_layout

_POLICY_DIRECTORY = "policy"
_OPTIMIZER_FILE = "optimizer.pt"
_RUNTIME_STATE_FILE = "runtime_state.json"
_COMPLETE_FILE = "COMPLETE"


@dataclass(frozen=True, slots=True)
class TrainingCheckpointSnapshot:
    optimizer_step: int
    experiment_id: str
    identity: RuntimeSnapshotIdentity
    backbone: PolicyBackbone
    optimizer: torch.optim.Optimizer
    execution_state: FullRuntimeExecutionState

    @property
    def snapshot_id(self) -> str:
        return stable_hash(
            {
                "backward_version": self.backbone.adapter_version(AdapterRole.BACKWARD_POLICY),
                "execution_state": self.execution_state.to_value(),
                "experiment_id": self.experiment_id,
                "runtime_snapshot_identity": self.identity.to_value(),
                "forward_version": self.backbone.adapter_version(AdapterRole.FORWARD_POLICY),
                "initial_trainable_state_hash": self.backbone.initial_trainable_state_hash,
                "optimizer_step": self.optimizer_step,
                "z_version": self.backbone.z_version,
            }
        )


class TrainingCheckpointStore(Protocol):
    def save(self, snapshot: TrainingCheckpointSnapshot) -> Path: ...

    def save_as(self, snapshot: TrainingCheckpointSnapshot, *, name: str) -> Path: ...

    def load_metadata(self, directory: Path) -> RuntimeSnapshot: ...

    def restore(
        self,
        directory: str | Path,
        *,
        backbone: PolicyBackbone,
        optimizer: torch.optim.Optimizer,
        expected_experiment_id: str,
        expected_identity: RuntimeSnapshotIdentity,
    ) -> RuntimeSnapshot: ...


class FilesystemTrainingCheckpointStore:
    def __init__(self, *, root: str | Path) -> None:
        self._root = Path(root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def save(self, snapshot: TrainingCheckpointSnapshot) -> Path:
        return self.save_as(snapshot, name=f"step-{snapshot.optimizer_step:08d}")

    def save_as(self, snapshot: TrainingCheckpointSnapshot, *, name: str) -> Path:
        if Path(name).name != name or not name:
            raise ValueError("snapshot name must be one path component")
        self._root.mkdir(parents=True, exist_ok=True)
        final = self._root / name
        staging = self._root / f".{name}.staging-{snapshot.snapshot_id[7:23]}"
        if final.exists():
            raise FileExistsError(final)
        if staging.exists():
            raise FileExistsError(staging)
        staging.mkdir()

        snapshot.backbone.save_checkpoint(str(staging / _POLICY_DIRECTORY))
        torch.save(
            checkpoint_optimizer_state(snapshot.backbone, snapshot.optimizer),
            staging / _OPTIMIZER_FILE,
        )
        metadata = RuntimeSnapshot(
            optimizer_step=snapshot.optimizer_step,
            experiment_id=snapshot.experiment_id,
            identity=snapshot.identity,
            policy_directory=_POLICY_DIRECTORY,
            optimizer_file=_OPTIMIZER_FILE,
            execution_state=snapshot.execution_state,
        )
        (staging / _RUNTIME_STATE_FILE).write_text(
            canonical_json(metadata.to_value()) + "\n",
            encoding="utf-8",
        )
        (staging / _COMPLETE_FILE).write_text("complete\n", encoding="ascii")
        _fsync_tree(staging)
        os.replace(staging, final)
        _fsync_directory(self._root)
        return final

    def load_metadata(self, directory: Path) -> RuntimeSnapshot:
        path = directory.resolve()
        if path.parent != self._root:
            raise ValueError("snapshot must be an exact child of the configured root")
        required = (
            path / _COMPLETE_FILE,
            path / _POLICY_DIRECTORY,
            path / _OPTIMIZER_FILE,
            path / _RUNTIME_STATE_FILE,
        )
        if (
            not required[0].is_file()
            or not required[1].is_dir()
            or any(not item.is_file() for item in required[2:])
        ):
            raise ValueError("snapshot is incomplete")
        raw = json.loads((path / _RUNTIME_STATE_FILE).read_text(encoding="utf-8"))
        return RuntimeSnapshot.from_value(raw)

    def retain_recent(self, *, keep_recent: int) -> tuple[Path, ...]:
        if type(keep_recent) is not int or keep_recent < 1:
            raise ValueError("checkpoint retention must keep at least one recent step")
        if not self._root.exists():
            return ()
        ordinary = sorted(
            (
                path
                for path in self._root.iterdir()
                if path.is_dir()
                and path.name.startswith("step-")
                and len(path.name) == len("step-") + 8
                and path.name.removeprefix("step-").isdigit()
            ),
            key=lambda path: path.name,
        )
        removed = ordinary[:-keep_recent]
        for path in removed:
            shutil.rmtree(path)
        if removed:
            _fsync_directory(self._root)
        return tuple(removed)

    def archive_uncommitted_step(self, optimizer_step: int) -> tuple[Path, ...]:
        if type(optimizer_step) is not int or optimizer_step < 1:
            raise ValueError("uncommitted optimizer step must be positive")
        names = [f"step-{optimizer_step:08d}", f"cadence-step-{optimizer_step:08d}"]
        names.extend(
            path.name for path in self._root.glob(f"phase-????????-step-{optimizer_step:08d}")
        )
        paths: list[Path] = []
        for name in names:
            final = self._root / name
            if final.exists():
                paths.append(final)
            paths.extend(self._root.glob(f".{name}.staging-*"))
        paths.extend(self._root.glob(f".phase-????????-step-{optimizer_step:08d}.staging-*"))
        paths = list(dict.fromkeys(paths))
        if not paths:
            return ()
        ordinal = 1
        while True:
            archive = self._root / f"uncommitted-step-{optimizer_step:08d}-{ordinal:03d}"
            try:
                archive.mkdir(mode=0o700)
                break
            except FileExistsError:
                ordinal += 1
        for path in paths:
            os.replace(path, archive / path.name)
        _fsync_directory(archive)
        _fsync_directory(self._root)
        return tuple(archive / path.name for path in paths)

    def restore(
        self,
        directory: str | Path,
        *,
        backbone: PolicyBackbone,
        optimizer: torch.optim.Optimizer,
        expected_experiment_id: str,
        expected_identity: RuntimeSnapshotIdentity,
    ) -> RuntimeSnapshot:
        path = Path(directory).resolve()
        metadata = self.load_metadata(path)
        if metadata.experiment_id != expected_experiment_id:
            raise ValueError("snapshot belongs to another experiment")
        if metadata.identity != expected_identity:
            raise ValueError("snapshot method/config/protocol identity differs")
        optimizer_state = torch.load(
            path / metadata.optimizer_file, map_location="cpu", weights_only=True
        )
        require_optimizer_state_layout(optimizer_state, backbone, optimizer)
        backbone.load_checkpoint(str(path / metadata.policy_directory))
        optimizer.load_state_dict(optimizer_state)
        return metadata


def _fsync_tree(directory: Path) -> None:
    for path in directory.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "FilesystemTrainingCheckpointStore",
    "TrainingCheckpointSnapshot",
    "TrainingCheckpointStore",
]
