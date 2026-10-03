from __future__ import annotations

import json
import os
from contextlib import ExitStack
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from skillev.contracts import JsonValue, canonical_json

from .event_log import EventEnvelope

STEP_TRANSACTION_FORMAT = "skillev-step-transaction@1"


class StepTransactionState(StrEnum):
    PREPARED = "prepared"
    OPTIMIZER_APPLIED = "optimizer-applied"
    PROJECTION_INSTALLED = "projection-installed"
    EVOLUTION_RESOLVED = "evolution-resolved"
    CHECKPOINT_PUBLISHED = "checkpoint-published"
    ADAPTER_COMMITTED = "adapter-committed"
    SOURCE_EVENTS_PUBLISHED = "source-events-published"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled-back-before-checkpoint"


_ORDER = tuple(StepTransactionState)


@dataclass(frozen=True, slots=True)
class StepTransactionRecord:
    optimizer_step: int
    batch_id: str
    policy_snapshot_before: str
    policy_snapshot_after: str | None
    checkpoint_name: str | None
    adapter_revision: str | None
    source_events: tuple[EventEnvelope, ...]
    state: StepTransactionState
    format: str = STEP_TRANSACTION_FORMAT

    def __post_init__(self) -> None:
        if self.format != STEP_TRANSACTION_FORMAT:
            raise ValueError("step transaction format is unsupported")
        if type(self.optimizer_step) is not int or self.optimizer_step < 1:
            raise ValueError("step transaction optimizer step must be positive")
        if not self.batch_id.strip() or not self.policy_snapshot_before.strip():
            raise ValueError("step transaction identity is incomplete")
        if self.policy_snapshot_after is not None and not self.policy_snapshot_after.strip():
            raise ValueError("step transaction after snapshot cannot be empty")
        if (
            self.checkpoint_name is not None
            and Path(self.checkpoint_name).name != self.checkpoint_name
        ):
            raise ValueError("step transaction checkpoint must be one path component")
        if self.adapter_revision is not None and not self.adapter_revision.strip():
            raise ValueError("step transaction adapter revision cannot be empty")
        if any(not isinstance(event, EventEnvelope) for event in self.source_events):
            raise TypeError("step transaction source events are invalid")
        if self.state is StepTransactionState.ROLLED_BACK:
            if (
                self.checkpoint_name is not None
                or self.adapter_revision is not None
                or self.source_events
            ):
                raise ValueError("rolled-back transaction contains durable step evidence")
            return
        state_index = _ORDER.index(self.state)
        if state_index >= _ORDER.index(StepTransactionState.OPTIMIZER_APPLIED):
            if self.policy_snapshot_after is None:
                raise ValueError("applied step transaction lacks its policy snapshot")
        if state_index >= _ORDER.index(StepTransactionState.CHECKPOINT_PUBLISHED):
            if self.checkpoint_name is None or not self.source_events:
                raise ValueError("checkpointed transaction lacks recovery evidence")
        if state_index >= _ORDER.index(StepTransactionState.ADAPTER_COMMITTED):
            if self.adapter_revision is None:
                raise ValueError("adapter-committed transaction lacks its revision")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "adapter_revision": self.adapter_revision,
            "batch_id": self.batch_id,
            "checkpoint_name": self.checkpoint_name,
            "format": self.format,
            "optimizer_step": self.optimizer_step,
            "policy_snapshot_after": self.policy_snapshot_after,
            "policy_snapshot_before": self.policy_snapshot_before,
            "source_events": [event.to_value() for event in self.source_events],
            "state": self.state.value,
        }

    @classmethod
    def from_value(cls, value: object) -> StepTransactionRecord:
        fields = {
            "adapter_revision",
            "batch_id",
            "checkpoint_name",
            "format",
            "optimizer_step",
            "policy_snapshot_after",
            "policy_snapshot_before",
            "source_events",
            "state",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("step transaction has incompatible fields")
        events = value["source_events"]
        if not isinstance(events, list):
            raise TypeError("step transaction source events must be an array")
        nullable_text = ("adapter_revision", "checkpoint_name", "policy_snapshot_after")
        if any(
            value[field] is not None and type(value[field]) is not str for field in nullable_text
        ):
            raise TypeError("step transaction optional identities must be text")
        if any(
            type(value[field]) is not str
            for field in ("batch_id", "format", "policy_snapshot_before", "state")
        ):
            raise TypeError("step transaction identities must be text")
        if type(value["optimizer_step"]) is not int:
            raise TypeError("step transaction optimizer step must be an integer")
        return cls(
            optimizer_step=value["optimizer_step"],
            batch_id=value["batch_id"],
            policy_snapshot_before=value["policy_snapshot_before"],
            policy_snapshot_after=value["policy_snapshot_after"],
            checkpoint_name=value["checkpoint_name"],
            adapter_revision=value["adapter_revision"],
            source_events=tuple(EventEnvelope.from_value(event) for event in events),
            state=StepTransactionState(value["state"]),
            format=value["format"],
        )


class StepTransactionJournal:
    def __init__(self, directory: Path) -> None:
        if not directory.is_absolute():
            raise ValueError("step transaction directory must be absolute")
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)

    def begin(
        self,
        *,
        optimizer_step: int,
        batch_id: str,
        policy_snapshot_before: str,
    ) -> StepTransactionRecord:
        record = StepTransactionRecord(
            optimizer_step=optimizer_step,
            batch_id=batch_id,
            policy_snapshot_before=policy_snapshot_before,
            policy_snapshot_after=None,
            checkpoint_name=None,
            adapter_revision=None,
            source_events=(),
            state=StepTransactionState.PREPARED,
        )
        path = self._path(optimizer_step)
        if path.exists():
            raise FileExistsError(path)
        self._write(record, replace_existing=False)
        return record

    def advance(
        self,
        record: StepTransactionRecord,
        state: StepTransactionState,
        *,
        policy_snapshot_after: str | None = None,
        checkpoint_name: str | None = None,
        adapter_revision: str | None = None,
        source_events: tuple[EventEnvelope, ...] | None = None,
    ) -> StepTransactionRecord:
        current = self.load(record.optimizer_step)
        if current != record:
            raise ValueError("step transaction changed before advance")
        expected_index = _ORDER.index(current.state) + 1
        if expected_index >= len(_ORDER) or state is not _ORDER[expected_index]:
            raise ValueError("step transaction transition is not the next frozen state")
        updated = replace(
            current,
            state=state,
            policy_snapshot_after=(
                current.policy_snapshot_after
                if policy_snapshot_after is None
                else policy_snapshot_after
            ),
            checkpoint_name=(
                current.checkpoint_name if checkpoint_name is None else checkpoint_name
            ),
            adapter_revision=(
                current.adapter_revision if adapter_revision is None else adapter_revision
            ),
            source_events=current.source_events if source_events is None else source_events,
        )
        self._write(updated, replace_existing=True)
        return updated

    def load(self, optimizer_step: int) -> StepTransactionRecord:
        return StepTransactionRecord.from_value(
            json.loads(self._path(optimizer_step).read_text(encoding="utf-8"))
        )

    def pending(self) -> tuple[StepTransactionRecord, ...]:
        paths = tuple(
            path
            for path in sorted(self.directory.glob("step-*.json"))
            if len(path.stem) == len("step-") + 8 and path.stem.removeprefix("step-").isdigit()
        )
        records = tuple(self.load(int(path.stem.removeprefix("step-"))) for path in paths)
        return tuple(
            record
            for record in records
            if record.state
            not in {StepTransactionState.COMMITTED, StepTransactionState.ROLLED_BACK}
        )

    def rollback_before_checkpoint(
        self,
        record: StepTransactionRecord,
    ) -> StepTransactionRecord:
        current = self.load(record.optimizer_step)
        if current != record:
            raise ValueError("step transaction changed before rollback")
        if current.state not in {
            StepTransactionState.PREPARED,
            StepTransactionState.OPTIMIZER_APPLIED,
            StepTransactionState.PROJECTION_INSTALLED,
            StepTransactionState.EVOLUTION_RESOLVED,
        }:
            raise ValueError("only a pre-checkpoint transaction can be rolled back")
        rolled_back = replace(current, state=StepTransactionState.ROLLED_BACK)
        self._write(rolled_back, replace_existing=True)
        source = self._path(record.optimizer_step)
        ordinal = 1
        while True:
            archive = self.directory / (
                f"step-{record.optimizer_step:08d}.rolled-back-{ordinal:03d}.json"
            )
            if not archive.exists():
                break
            ordinal += 1
        os.replace(source, archive)
        directory_descriptor = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return rolled_back

    def _path(self, optimizer_step: int) -> Path:
        if type(optimizer_step) is not int or optimizer_step < 1:
            raise ValueError("step transaction optimizer step must be positive")
        return self.directory / f"step-{optimizer_step:08d}.json"

    def _write(self, record: StepTransactionRecord, *, replace_existing: bool) -> None:
        final = self._path(record.optimizer_step)
        staging = final.with_suffix(".json.staging")
        if staging.exists():
            raise FileExistsError(staging)
        if final.exists() and not replace_existing:
            raise FileExistsError(final)
        with ExitStack() as cleanup:
            cleanup.callback(staging.unlink, missing_ok=True)
            descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(canonical_json(record.to_value()) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(staging, final)
            directory_descriptor = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
            cleanup.pop_all()


class StepCheckpointRecovery(Protocol):
    def __call__(self, record: StepTransactionRecord) -> None: ...


class StepAdapterRecovery(Protocol):
    def __call__(self, record: StepTransactionRecord) -> str: ...


@dataclass(frozen=True, slots=True)
class StepTransactionReconciler:
    journal: StepTransactionJournal
    emitter: object
    require_checkpoint: StepCheckpointRecovery
    ensure_adapter: StepAdapterRecovery

    def reconcile(self, record: StepTransactionRecord) -> StepTransactionRecord:
        from .emitter import RuntimeEventEmitter

        if not isinstance(self.emitter, RuntimeEventEmitter):
            raise TypeError("step reconciliation requires the runtime event emitter")
        if record.state is StepTransactionState.COMMITTED:
            self.emitter.reconcile_prepared(record.source_events)
            return record
        if _ORDER.index(record.state) < _ORDER.index(StepTransactionState.CHECKPOINT_PUBLISHED):
            raise RuntimeError(
                "step has no durable checkpoint and must restart from the last committed state"
            )
        self.require_checkpoint(record)
        current = record
        if current.state is StepTransactionState.CHECKPOINT_PUBLISHED:
            revision = self.ensure_adapter(current)
            current = self.journal.advance(
                current,
                StepTransactionState.ADAPTER_COMMITTED,
                adapter_revision=revision,
            )
        else:
            revision = self.ensure_adapter(current)
            if revision != current.adapter_revision:
                raise RuntimeError("recovered adapter revision differs from the transaction")
        if current.state is StepTransactionState.ADAPTER_COMMITTED:
            self.emitter.reconcile_prepared(current.source_events)
            current = self.journal.advance(
                current,
                StepTransactionState.SOURCE_EVENTS_PUBLISHED,
            )
        elif current.state is StepTransactionState.SOURCE_EVENTS_PUBLISHED:
            self.emitter.reconcile_prepared(current.source_events)
        if current.state is StepTransactionState.SOURCE_EVENTS_PUBLISHED:
            current = self.journal.advance(current, StepTransactionState.COMMITTED)
        if current.state is not StepTransactionState.COMMITTED:
            raise RuntimeError("step reconciliation ended in an unsupported state")
        return current


__all__ = [
    "STEP_TRANSACTION_FORMAT",
    "StepTransactionJournal",
    "StepTransactionReconciler",
    "StepTransactionRecord",
    "StepTransactionState",
]
