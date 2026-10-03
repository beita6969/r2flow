from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeAlias

from skillev.contracts import (
    JsonValue,
    TerminalReward,
    normalize_json,
    validate_sha256,
)
from skillev.runtime.attempt_failures import AttemptDomainError
from skillev.runtime.attempt_protocol import AttemptFailureCode, AttemptFailureStage
from skillev.runtime.execution import EnvironmentObservation, RolloutEnvironmentSession

from .types import RolloutTermination


class NoSubmissionReason(StrEnum):
    HORIZON_EXHAUSTED = "horizon-exhausted"


@dataclass(frozen=True, slots=True)
class SubmittedTerminalValue:
    value: JsonValue

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", normalize_json(self.value))

    def to_value(self) -> dict[str, JsonValue]:
        return {"kind": "submitted", "value": self.value}


@dataclass(frozen=True, slots=True)
class NoTerminalSubmission:
    reason: NoSubmissionReason

    def __post_init__(self) -> None:
        if not isinstance(self.reason, NoSubmissionReason):
            raise TypeError("no-submission reason must be NoSubmissionReason")

    def to_value(self) -> dict[str, JsonValue]:
        return {"kind": "no-submission", "reason": self.reason.value}


TerminalEvaluationInput: TypeAlias = SubmittedTerminalValue | NoTerminalSubmission


@dataclass(frozen=True, slots=True)
class TerminalActionEvidence:
    step_index: int
    text: str
    parse_status: str
    observation_status: str
    finish_reason: str

    def __post_init__(self) -> None:
        if (
            type(self.step_index) is not int
            or self.step_index < 1
            or not isinstance(self.text, str)
        ):
            raise ValueError("terminal action evidence requires a sampled action")
        if self.parse_status not in {"valid", "parse-error", "schema-invalid"}:
            raise ValueError("unknown terminal action parse status")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "step_index": self.step_index,
            "text": self.text,
            "parse_status": self.parse_status,
            "observation_status": self.observation_status,
            "finish_reason": self.finish_reason,
        }


@dataclass(frozen=True, slots=True)
class TerminalEvaluationRequest:
    trajectory_id: str
    task_id: str
    termination: RolloutTermination
    evaluation_input: TerminalEvaluationInput
    public_transcript_hash: str
    last_action: TerminalActionEvidence | None = None

    def __post_init__(self) -> None:
        if not self.trajectory_id or not self.task_id:
            raise ValueError("terminal evaluation identity fields must be non-empty")
        if not isinstance(self.termination, RolloutTermination):
            raise ValueError("termination must be a RolloutTermination")
        if not isinstance(
            self.evaluation_input,
            SubmittedTerminalValue | NoTerminalSubmission,
        ):
            raise TypeError("evaluation_input must be a closed terminal input")
        if self.termination is RolloutTermination.COMPLETED and not isinstance(
            self.evaluation_input, SubmittedTerminalValue
        ):
            raise ValueError("completed rollout requires a submitted terminal value")
        if (
            self.termination is RolloutTermination.HORIZON_EXHAUSTED
            and self.evaluation_input != NoTerminalSubmission(NoSubmissionReason.HORIZON_EXHAUSTED)
        ):
            raise ValueError("horizon exhaustion requires its explicit no-submission input")
        validate_sha256(self.public_transcript_hash)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "evaluation_input": self.evaluation_input.to_value(),
            "public_transcript_hash": self.public_transcript_hash,
            "task_id": self.task_id,
            "termination": self.termination.value,
            "trajectory_id": self.trajectory_id,
            **(
                {"last_action": self.last_action.to_value()} if self.last_action is not None else {}
            ),
        }


class TerminalEvaluator(Protocol):
    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward: ...


class TerminalEvaluatorError(AttemptDomainError):
    def __init__(self, private_detail: str) -> None:
        super().__init__(
            code=AttemptFailureCode.TERMINAL_EVALUATOR_FAILED,
            stage=AttemptFailureStage.TERMINAL_EVALUATION,
            private_detail=private_detail,
        )


__all__ = [
    "EnvironmentObservation",
    "NoSubmissionReason",
    "NoTerminalSubmission",
    "RolloutEnvironmentSession",
    "SubmittedTerminalValue",
    "TerminalEvaluationInput",
    "TerminalEvaluationRequest",
    "TerminalEvaluator",
    "TerminalEvaluatorError",
]
