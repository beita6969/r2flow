from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from skillev.contracts import JsonValue, normalize_json

ROLLOUT_INFRASTRUCTURE_FAILURE_FORMAT: Final = "skillev-rollout-infrastructure-failure@2"


def _wire_text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise TypeError(f"{field} must be non-empty text")
    return value


class RolloutBoundaryError(ValueError):
    pass


class EpisodeInfrastructureError(RuntimeError):
    pass


class InitialContextInfrastructureError(RuntimeError):
    pass


class GenerationInfrastructureError(RuntimeError):
    pass


class RolloutInfrastructureKind(StrEnum):
    POLICY_SNAPSHOT_MISMATCH = "policy-snapshot-mismatch"
    TOKEN_ROUNDTRIP_MISMATCH = "token-roundtrip-mismatch"
    EMPTY_ACTION = "empty-action"
    INITIAL_CONTEXT_MISMATCH = "initial-context-mismatch"
    ENVIRONMENT_SKILL_INVOCATION_MISMATCH = "environment-skill-invocation-mismatch"
    ACTION_CONSTRAINT_VIOLATION = "action-constraint-violation"
    NON_EVENT_ACTION = "non-event-action"


@dataclass(frozen=True, slots=True)
class RolloutInfrastructureFailure:
    trajectory_id: str
    task_id: str
    stage: str
    kind: RolloutInfrastructureKind
    public_message: str
    step_index: int | None = None
    format: str = ROLLOUT_INFRASTRUCTURE_FAILURE_FORMAT

    def __post_init__(self) -> None:
        for field in ("trajectory_id", "task_id", "stage", "public_message"):
            _wire_text(getattr(self, field), field=field)
        if not isinstance(self.kind, RolloutInfrastructureKind):
            raise TypeError("kind must be a RolloutInfrastructureKind")
        if self.step_index is not None and (
            type(self.step_index) is not int or self.step_index < 1
        ):
            raise ValueError("step_index must be positive when present")
        if self.format != ROLLOUT_INFRASTRUCTURE_FAILURE_FORMAT:
            raise ValueError("unsupported rollout infrastructure failure format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "kind": self.kind.value,
            "public_message": self.public_message,
            "stage": self.stage,
            "step_index": self.step_index,
            "task_id": self.task_id,
            "trajectory_id": self.trajectory_id,
        }

    @classmethod
    def from_value(cls, value: object) -> RolloutInfrastructureFailure:
        normalized = normalize_json(value)
        expected = {
            "format",
            "kind",
            "public_message",
            "stage",
            "step_index",
            "task_id",
            "trajectory_id",
        }
        if not isinstance(normalized, dict) or set(normalized) != expected:
            raise ValueError("rollout infrastructure failure has an incompatible field set")
        if normalized["format"] != ROLLOUT_INFRASTRUCTURE_FAILURE_FORMAT:
            raise ValueError("unsupported rollout infrastructure failure format")
        raw_step = normalized["step_index"]
        if raw_step is not None and type(raw_step) is not int:
            raise TypeError("step_index must be an integer or null")
        return cls(
            trajectory_id=_wire_text(
                normalized["trajectory_id"],
                field="trajectory_id",
            ),
            task_id=_wire_text(normalized["task_id"], field="task_id"),
            stage=_wire_text(normalized["stage"], field="stage"),
            kind=RolloutInfrastructureKind(_wire_text(normalized["kind"], field="kind")),
            public_message=_wire_text(
                normalized["public_message"],
                field="public_message",
            ),
            step_index=raw_step,
        )


class RolloutInfrastructureError(RuntimeError):
    def __init__(self, failure: RolloutInfrastructureFailure) -> None:
        if not isinstance(failure, RolloutInfrastructureFailure):
            raise TypeError("failure must be a RolloutInfrastructureFailure")
        self.failure = failure
        super().__init__(failure.public_message)


__all__ = [
    "ROLLOUT_INFRASTRUCTURE_FAILURE_FORMAT",
    "EpisodeInfrastructureError",
    "GenerationInfrastructureError",
    "InitialContextInfrastructureError",
    "RolloutBoundaryError",
    "RolloutInfrastructureError",
    "RolloutInfrastructureFailure",
    "RolloutInfrastructureKind",
]
