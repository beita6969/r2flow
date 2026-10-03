from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from skillev.contracts import JsonValue, TrainingStepReportValue


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _non_negative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


class AttemptBuilderKind(StrEnum):
    FULL = "full"


class AttemptFailureStage(StrEnum):
    EXECUTION = "execution"
    TERMINAL_EVALUATION = "terminal-evaluation"


class AttemptFailureCode(StrEnum):
    TERMINAL_EVALUATOR_FAILED = "terminal-evaluator-failed"
    EVENT_APPEND_FAILED = "event-append-failed"


@dataclass(frozen=True, slots=True)
class FullAttemptSummary:
    reports: tuple[TrainingStepReportValue, ...]
    planned_training_steps_this_attempt: int
    completed_training_steps_this_attempt: int
    actions_committed_this_attempt: int
    cycles_committed_this_attempt: int
    cycles_committed_in_run: int
    initial_optimizer_step: int
    final_optimizer_step: int
    final_library_version: str
    final_policy_snapshot_id: str

    def __post_init__(self) -> None:
        for field in (
            "planned_training_steps_this_attempt",
            "completed_training_steps_this_attempt",
            "actions_committed_this_attempt",
            "cycles_committed_this_attempt",
            "cycles_committed_in_run",
            "initial_optimizer_step",
            "final_optimizer_step",
        ):
            _non_negative_int(getattr(self, field), field=field)
        if self.completed_training_steps_this_attempt != len(self.reports):
            raise ValueError("completed step count differs from reports")
        if self.completed_training_steps_this_attempt != self.planned_training_steps_this_attempt:
            raise ValueError("successful attempt did not complete its exact run plan")
        if (
            self.final_optimizer_step - self.initial_optimizer_step
            != self.completed_training_steps_this_attempt
        ):
            raise ValueError("summary optimizer-step delta differs from completed steps")
        if self.cycles_committed_this_attempt > self.cycles_committed_in_run:
            raise ValueError("attempt cycles cannot exceed run-total cycles")
        if self.actions_committed_this_attempt < self.cycles_committed_this_attempt:
            raise ValueError("each committed cycle requires at least one action")
        _text(self.final_library_version, field="final_library_version")
        _text(self.final_policy_snapshot_id, field="final_policy_snapshot_id")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "actions_committed_this_attempt": self.actions_committed_this_attempt,
            "completed_training_steps_this_attempt": self.completed_training_steps_this_attempt,
            "cycles_committed_in_run": self.cycles_committed_in_run,
            "cycles_committed_this_attempt": self.cycles_committed_this_attempt,
            "final_library_version": self.final_library_version,
            "final_optimizer_step": self.final_optimizer_step,
            "final_policy_snapshot_id": self.final_policy_snapshot_id,
            "initial_optimizer_step": self.initial_optimizer_step,
            "kind": "full",
            "planned_training_steps_this_attempt": self.planned_training_steps_this_attempt,
            "reports": [report.to_value() for report in self.reports],
        }


__all__ = [
    "AttemptBuilderKind",
    "AttemptFailureCode",
    "AttemptFailureStage",
    "FullAttemptSummary",
]
