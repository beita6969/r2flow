from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import cast

from .canonical import JsonValue, normalize_json, stable_hash
from .identity import validate_sha256
from .r2flow_training import (
    R2FlowBatchStats,
    R2FlowEdgeRecord,
)
from .posterior_batch import PosteriorBatchUpdate
from .ttb_common import require_non_empty_text
from .ttb_trajectory import TrajectoryRecord

TRAINING_STEP_REPORT_FORMAT = "skillev-training-step-report@6"
TRAINING_STEP_COMMIT_FORMAT = "skillev-training-step-commit@4"
LIBRARY_INITIALIZED_FORMAT = "skillev-library-initialized@4"
RUN_CURSOR_VALUE_FORMAT = "skillev-source-run-cursor@1"


def _object(
    value: object,
    *,
    label: str,
    fields: frozenset[str],
) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict):
        raise ValueError(f"{label} must be a JSON object")
    if set(normalized) != fields:
        raise ValueError(f"{label} has incompatible fields")
    return normalized


def _text(value: JsonValue, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    require_non_empty_text(value, field=field)
    return value


def _integer(value: JsonValue, *, field: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _number(value: JsonValue, *, field: str, upper: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be a finite non-negative number")
    if upper is not None and number > upper:
        raise ValueError(f"{field} must not exceed {upper}")
    return number


def _timestamp(value: JsonValue, *, field: str) -> tuple[str, datetime]:
    text = _text(value, field=field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a UTC offset")
    return text, parsed


def _array(value: JsonValue, *, field: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return value


def _json_objects(
    values: tuple[Mapping[str, JsonValue], ...],
    *,
    field: str,
) -> tuple[Mapping[str, JsonValue], ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"{field} must be a tuple")
    normalized: list[Mapping[str, JsonValue]] = []
    for value in values:
        item = normalize_json(value)
        if not isinstance(item, dict):
            raise ValueError(f"{field} must contain JSON objects")
        normalized.append(item)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class RunCursorValue:
    run_plan_hash: str
    completed_training_steps: int
    committed_cycles: int
    committed_actions: int
    format: str = RUN_CURSOR_VALUE_FORMAT

    def __post_init__(self) -> None:
        validate_sha256(self.run_plan_hash)
        for field in (
            "completed_training_steps",
            "committed_cycles",
            "committed_actions",
        ):
            _integer(getattr(self, field), field=field, minimum=0)
        if self.committed_actions < self.committed_cycles:
            raise ValueError("run cursor has fewer actions than cycles")
        if self.format != RUN_CURSOR_VALUE_FORMAT:
            raise ValueError("unsupported source run cursor format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "committed_actions": self.committed_actions,
            "committed_cycles": self.committed_cycles,
            "completed_training_steps": self.completed_training_steps,
            "format": self.format,
            "run_plan_hash": self.run_plan_hash,
        }

    @classmethod
    def from_value(cls, value: object) -> RunCursorValue:
        data = _object(
            value,
            label="RunCursorValue",
            fields=frozenset(
                {
                    "committed_actions",
                    "committed_cycles",
                    "completed_training_steps",
                    "format",
                    "run_plan_hash",
                }
            ),
        )
        return cls(
            run_plan_hash=_text(data["run_plan_hash"], field="run_plan_hash"),
            completed_training_steps=_integer(
                data["completed_training_steps"],
                field="completed_training_steps",
                minimum=0,
            ),
            committed_cycles=_integer(
                data["committed_cycles"], field="committed_cycles", minimum=0
            ),
            committed_actions=_integer(
                data["committed_actions"], field="committed_actions", minimum=0
            ),
            format=_text(data["format"], field="format"),
        )


@dataclass(frozen=True, slots=True)
class TrainingStepReportValue:
    optimizer_step: int
    batch_id: str
    torch_batch_loss: float
    audited_batch_loss: float
    mean_reward: float
    grad_norm_forward: float
    grad_norm_backward: float
    grad_norm_z: float
    grad_norm_psi: float
    forward_adapter_version: str
    backward_adapter_version: str
    z_version: str
    started_at: str
    completed_at: str
    optimization_diagnostics: JsonValue
    optimizer_transition: JsonValue
    format: str = TRAINING_STEP_REPORT_FORMAT

    def __post_init__(self) -> None:
        _integer(self.optimizer_step, field="optimizer_step", minimum=1)
        require_non_empty_text(self.batch_id, field="batch_id")
        for field, value in (
            ("torch_batch_loss", self.torch_batch_loss),
            ("audited_batch_loss", self.audited_batch_loss),
            ("grad_norm_forward", self.grad_norm_forward),
            ("grad_norm_backward", self.grad_norm_backward),
            ("grad_norm_z", self.grad_norm_z),
            ("grad_norm_psi", self.grad_norm_psi),
        ):
            object.__setattr__(self, field, _number(value, field=field))
        object.__setattr__(
            self,
            "mean_reward",
            _number(self.mean_reward, field="mean_reward", upper=1.0),
        )
        for field, version in (
            ("forward_adapter_version", self.forward_adapter_version),
            ("backward_adapter_version", self.backward_adapter_version),
            ("z_version", self.z_version),
        ):
            require_non_empty_text(version, field=field)
        if self.format != TRAINING_STEP_REPORT_FORMAT:
            raise ValueError("unsupported training step report format")
        for name, item in (
            ("optimization_diagnostics", self.optimization_diagnostics),
            ("optimizer_transition", self.optimizer_transition),
        ):
            if item is not None:
                if not isinstance(item, dict):
                    raise ValueError(f"{name} must be an object or null")
                normalize_json(item)
        started_at, started = _timestamp(self.started_at, field="started_at")
        completed_at, completed = _timestamp(self.completed_at, field="completed_at")
        if completed < started:
            raise ValueError("completed_at cannot precede started_at")
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "completed_at", completed_at)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "audited_batch_loss": self.audited_batch_loss,
            "backward_adapter_version": self.backward_adapter_version,
            "batch_id": self.batch_id,
            "completed_at": self.completed_at,
            "format": self.format,
            "forward_adapter_version": self.forward_adapter_version,
            "grad_norm_backward": self.grad_norm_backward,
            "grad_norm_forward": self.grad_norm_forward,
            "grad_norm_psi": self.grad_norm_psi,
            "grad_norm_z": self.grad_norm_z,
            "mean_reward": self.mean_reward,
            "optimization_diagnostics": self.optimization_diagnostics,
            "optimizer_step": self.optimizer_step,
            "optimizer_transition": self.optimizer_transition,
            "started_at": self.started_at,
            "torch_batch_loss": self.torch_batch_loss,
            "z_version": self.z_version,
        }

    @classmethod
    def from_value(cls, value: object) -> TrainingStepReportValue:
        data = _object(
            value,
            label="TrainingStepReportValue",
            fields=frozenset(
                {
                    "audited_batch_loss",
                    "backward_adapter_version",
                    "batch_id",
                    "completed_at",
                    "format",
                    "forward_adapter_version",
                    "grad_norm_backward",
                    "grad_norm_forward",
                    "grad_norm_psi",
                    "grad_norm_z",
                    "mean_reward",
                    "optimization_diagnostics",
                    "optimizer_step",
                    "optimizer_transition",
                    "started_at",
                    "torch_batch_loss",
                    "z_version",
                }
            ),
        )
        return cls(
            optimizer_step=_integer(data["optimizer_step"], field="optimizer_step", minimum=1),
            batch_id=_text(data["batch_id"], field="batch_id"),
            torch_batch_loss=_number(data["torch_batch_loss"], field="torch_batch_loss"),
            audited_batch_loss=_number(data["audited_batch_loss"], field="audited_batch_loss"),
            mean_reward=_number(data["mean_reward"], field="mean_reward", upper=1.0),
            grad_norm_forward=_number(data["grad_norm_forward"], field="grad_norm_forward"),
            grad_norm_backward=_number(data["grad_norm_backward"], field="grad_norm_backward"),
            grad_norm_z=_number(data["grad_norm_z"], field="grad_norm_z"),
            grad_norm_psi=_number(data["grad_norm_psi"], field="grad_norm_psi"),
            forward_adapter_version=_text(
                data["forward_adapter_version"], field="forward_adapter_version"
            ),
            backward_adapter_version=_text(
                data["backward_adapter_version"], field="backward_adapter_version"
            ),
            z_version=_text(data["z_version"], field="z_version"),
            started_at=_text(data["started_at"], field="started_at"),
            completed_at=_text(data["completed_at"], field="completed_at"),
            optimization_diagnostics=data["optimization_diagnostics"],
            optimizer_transition=data["optimizer_transition"],
            format=_text(data["format"], field="format"),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class TrainingStepCommit:
    batch_id: str
    optimizer_step: int
    policy_snapshot_before: str
    policy_snapshot_after: str
    library_version: str
    records: tuple[TrajectoryRecord, ...]
    edge_records: tuple[R2FlowEdgeRecord, ...]
    stats: R2FlowBatchStats
    posterior_batch: PosteriorBatchUpdate
    report: TrainingStepReportValue
    run_cursor_after: RunCursorValue
    format: str = TRAINING_STEP_COMMIT_FORMAT

    def __post_init__(self) -> None:
        require_non_empty_text(self.batch_id, field="batch_id")
        _integer(self.optimizer_step, field="optimizer_step", minimum=1)
        for field, value in (
            ("policy_snapshot_before", self.policy_snapshot_before),
            ("policy_snapshot_after", self.policy_snapshot_after),
            ("library_version", self.library_version),
        ):
            require_non_empty_text(value, field=field)
        if self.format != TRAINING_STEP_COMMIT_FORMAT:
            raise ValueError("unsupported training step commit format")
        if not isinstance(self.records, tuple) or any(
            not isinstance(record, TrajectoryRecord) for record in self.records
        ):
            raise ValueError("records must contain TrajectoryRecord values")
        if not self.records:
            raise ValueError("training step commit requires records")
        if not isinstance(self.edge_records, tuple) or any(
            not isinstance(record, R2FlowEdgeRecord) for record in self.edge_records
        ):
            raise ValueError("edge_records must contain R2FlowEdgeRecord values")
        if not isinstance(self.stats, R2FlowBatchStats):
            raise ValueError("stats must be R2FlowBatchStats")
        if not isinstance(self.posterior_batch, PosteriorBatchUpdate):
            raise ValueError("posterior_batch must be PosteriorBatchUpdate")
        if not isinstance(self.report, TrainingStepReportValue):
            raise ValueError("report must be TrainingStepReportValue")
        if not isinstance(self.run_cursor_after, RunCursorValue):
            raise TypeError("training commit requires run_cursor_after")
        if self.run_cursor_after.completed_training_steps < 1:
            raise ValueError("training commit cursor must include its completed step")
        if self.stats.batch_id != self.batch_id:
            raise ValueError("training step stats batch_id differs")
        if self.stats.optimizer_step != self.optimizer_step:
            raise ValueError("training step stats optimizer_step differs")
        record_ids = tuple(record.trajectory_id for record in self.records)
        residual_ids = tuple(residual.trajectory_id for residual in self.stats.residuals)
        if record_ids != residual_ids:
            raise ValueError("training records and residuals have different order")
        if self.posterior_batch.batch_id != self.batch_id:
            raise ValueError("posterior batch targets another training batch")
        if self.report.batch_id != self.batch_id:
            raise ValueError("training report targets another batch")
        if self.report.optimizer_step != self.optimizer_step:
            raise ValueError("training report targets another optimizer step")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "batch_id": self.batch_id,
            "edge_records": [record.to_value() for record in self.edge_records],
            "format": self.format,
            "library_version": self.library_version,
            "optimizer_step": self.optimizer_step,
            "policy_snapshot_after": self.policy_snapshot_after,
            "policy_snapshot_before": self.policy_snapshot_before,
            "posterior_batch": self.posterior_batch.to_value(),
            "records": [record.to_value() for record in self.records],
            "report": self.report.to_value(),
            "run_cursor_after": self.run_cursor_after.to_value(),
            "stats": self.stats.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> TrainingStepCommit:
        data = _object(
            value,
            label="TrainingStepCommit",
            fields=frozenset(
                {
                    "batch_id",
                    "edge_records",
                    "format",
                    "library_version",
                    "optimizer_step",
                    "policy_snapshot_after",
                    "policy_snapshot_before",
                    "posterior_batch",
                    "records",
                    "report",
                    "run_cursor_after",
                    "stats",
                }
            ),
        )
        return cls(
            batch_id=_text(data["batch_id"], field="batch_id"),
            optimizer_step=_integer(data["optimizer_step"], field="optimizer_step", minimum=1),
            policy_snapshot_before=_text(
                data["policy_snapshot_before"],
                field="policy_snapshot_before",
            ),
            policy_snapshot_after=_text(
                data["policy_snapshot_after"],
                field="policy_snapshot_after",
            ),
            library_version=_text(data["library_version"], field="library_version"),
            records=tuple(
                TrajectoryRecord.from_value(record)
                for record in _array(data["records"], field="records")
            ),
            edge_records=tuple(
                R2FlowEdgeRecord.from_value(record)
                for record in _array(data["edge_records"], field="edge_records")
            ),
            stats=R2FlowBatchStats.from_value(data["stats"]),
            posterior_batch=PosteriorBatchUpdate.from_value(data["posterior_batch"]),
            report=TrainingStepReportValue.from_value(data["report"]),
            run_cursor_after=RunCursorValue.from_value(data["run_cursor_after"]),
            format=_text(data["format"], field="format"),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class LibraryInitialized:
    documents: tuple[Mapping[str, JsonValue], ...]
    active_skill_ids: tuple[str, ...]
    library_version: str
    initial_optimizer_step: int
    method_identity_hash: str
    run_cursor: RunCursorValue
    format: str = LIBRARY_INITIALIZED_FORMAT

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "documents",
            _json_objects(self.documents, field="documents"),
        )
        if not isinstance(self.active_skill_ids, tuple) or any(
            not isinstance(skill_id, str) or not skill_id.strip()
            for skill_id in self.active_skill_ids
        ):
            raise ValueError("active_skill_ids must contain non-empty text")
        if tuple(sorted(set(self.active_skill_ids))) != self.active_skill_ids:
            raise ValueError("active_skill_ids must be sorted and unique")
        require_non_empty_text(self.library_version, field="library_version")
        _integer(self.initial_optimizer_step, field="initial_optimizer_step", minimum=0)
        validate_sha256(self.method_identity_hash)
        if not isinstance(self.run_cursor, RunCursorValue):
            raise TypeError("library initialization requires RunCursorValue")
        if self.format != LIBRARY_INITIALIZED_FORMAT:
            raise ValueError("unsupported library-initialized format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "active_skill_ids": list(self.active_skill_ids),
            "documents": [normalize_json(document) for document in self.documents],
            "format": self.format,
            "initial_optimizer_step": self.initial_optimizer_step,
            "library_version": self.library_version,
            "method_identity_hash": self.method_identity_hash,
            "run_cursor": self.run_cursor.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> LibraryInitialized:
        data = _object(
            value,
            label="LibraryInitialized",
            fields=frozenset(
                {
                    "active_skill_ids",
                    "documents",
                    "format",
                    "initial_optimizer_step",
                    "library_version",
                    "method_identity_hash",
                    "run_cursor",
                }
            ),
        )
        documents = _array(data["documents"], field="documents")
        if any(not isinstance(document, dict) for document in documents):
            raise ValueError("documents must contain JSON objects")
        return cls(
            documents=tuple(cast(dict[str, JsonValue], document) for document in documents),
            active_skill_ids=tuple(
                _text(item, field="active_skill_ids")
                for item in _array(data["active_skill_ids"], field="active_skill_ids")
            ),
            library_version=_text(data["library_version"], field="library_version"),
            initial_optimizer_step=_integer(
                data["initial_optimizer_step"], field="initial_optimizer_step", minimum=0
            ),
            method_identity_hash=_text(data["method_identity_hash"], field="method_identity_hash"),
            run_cursor=RunCursorValue.from_value(data["run_cursor"]),
            format=_text(data["format"], field="format"),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


__all__ = [
    "LIBRARY_INITIALIZED_FORMAT",
    "RUN_CURSOR_VALUE_FORMAT",
    "TRAINING_STEP_COMMIT_FORMAT",
    "TRAINING_STEP_REPORT_FORMAT",
    "LibraryInitialized",
    "RunCursorValue",
    "TrainingStepCommit",
    "TrainingStepReportValue",
]
