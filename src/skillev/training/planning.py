from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TypeAlias

from skillev.contracts import JsonValue, RunCursorValue
from skillev.rollout import DecodingSnapshot, RolloutArtifact, RolloutTask, RolloutTokenizerProtocol
from skillev.runtime import BudgetExceededError, BudgetVector
from skillev.runtime.attempt_run_plan import ExactAttemptRunPlan

from .config import TrainerConfig


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


@dataclass(frozen=True, slots=True)
class PlannedRollout:
    position: int
    task: RolloutTask
    trajectory_id: str
    decoding: DecodingSnapshot

    def __post_init__(self) -> None:
        if type(self.position) is not int or self.position < 1:
            raise ValueError("planned rollout position must be positive")
        _text(self.trajectory_id, field="trajectory_id")
        if not isinstance(self.decoding, DecodingSnapshot):
            raise TypeError("planned rollout requires its exact decoding snapshot")


@dataclass(frozen=True, slots=True)
class TrainingBatchPlan:
    batch_id: str
    optimizer_step: int
    policy_snapshot_id: str
    library_version: str
    rollouts: tuple[PlannedRollout, ...]

    def __post_init__(self) -> None:
        for field in ("batch_id", "policy_snapshot_id", "library_version"):
            _text(getattr(self, field), field=field)
        if type(self.optimizer_step) is not int or self.optimizer_step < 1:
            raise ValueError("optimizer_step must be positive")
        if not self.rollouts:
            raise ValueError("training batch plan cannot be empty")
        if tuple(item.position for item in self.rollouts) != tuple(
            range(1, len(self.rollouts) + 1)
        ):
            raise ValueError("planned rollout positions must be consecutive")


@dataclass(frozen=True, slots=True)
class CollectedTrainingBatch:
    batch_id: str
    optimizer_step: int
    policy_snapshot_id: str
    library_version: str
    artifacts: tuple[RolloutArtifact, ...]

    def __post_init__(self) -> None:
        for field in ("batch_id", "policy_snapshot_id", "library_version"):
            _text(getattr(self, field), field=field)
        if type(self.optimizer_step) is not int or self.optimizer_step < 1:
            raise ValueError("optimizer_step must be positive")
        if not self.artifacts:
            raise ValueError("collected training batch cannot be empty")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "artifacts": [artifact.to_value() for artifact in self.artifacts],
            "batch_id": self.batch_id,
            "library_version": self.library_version,
            "optimizer_step": self.optimizer_step,
            "policy_snapshot_id": self.policy_snapshot_id,
        }

    @classmethod
    def from_value(
        cls,
        value: object,
        *,
        tokenizer: RolloutTokenizerProtocol,
    ) -> CollectedTrainingBatch:
        if not isinstance(value, dict) or set(value) != {
            "artifacts",
            "batch_id",
            "library_version",
            "optimizer_step",
            "policy_snapshot_id",
        }:
            raise ValueError("collected training batch has incompatible fields")
        artifacts = value["artifacts"]
        if not isinstance(artifacts, list):
            raise TypeError("collected training batch artifacts must be an array")
        if any(
            type(value[field]) is not str
            for field in (
                "batch_id",
                "library_version",
                "policy_snapshot_id",
            )
        ):
            raise TypeError("collected training batch identity fields must be strings")
        if type(value["optimizer_step"]) is not int:
            raise TypeError("collected training batch optimizer step must be an integer")
        return cls(
            batch_id=value["batch_id"],
            optimizer_step=value["optimizer_step"],
            policy_snapshot_id=value["policy_snapshot_id"],
            library_version=value["library_version"],
            artifacts=tuple(
                RolloutArtifact.from_value(artifact, tokenizer=tokenizer) for artifact in artifacts
            ),
        )


@dataclass(frozen=True, slots=True)
class TrainingStepExecutionContext:
    run_cursor_after: RunCursorValue

    def __post_init__(self) -> None:
        if not isinstance(self.run_cursor_after, RunCursorValue):
            raise TypeError("training step context requires RunCursorValue")


@dataclass(frozen=True, slots=True)
class FixedAttemptBudgetPlan:
    batch_count: int
    batch_size: int
    max_turns: int
    reasoning_call_maximum: BudgetVector
    action_call_maximum: BudgetVector
    tool_call_maximum: BudgetVector
    maximum_cycles: int
    phi_per_cycle_maximum: BudgetVector
    executor_call_maximum: BudgetVector = dataclasses.field(default_factory=BudgetVector)

    def __post_init__(self) -> None:
        if type(self.batch_count) is not int or self.batch_count < 0:
            raise ValueError("batch_count must be non-negative")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if type(self.max_turns) is not int or self.max_turns < 1:
            raise ValueError("max_turns must be positive")
        if type(self.maximum_cycles) is not int or self.maximum_cycles < 0:
            raise ValueError("maximum_cycles must be non-negative")
        for field in (
            "reasoning_call_maximum",
            "action_call_maximum",
            "tool_call_maximum",
            "phi_per_cycle_maximum",
            "executor_call_maximum",
        ):
            if not isinstance(getattr(self, field), BudgetVector):
                raise TypeError(f"{field} must be BudgetVector")

    @classmethod
    def from_trainer_and_run_plan(
        cls,
        *,
        trainer: TrainerConfig,
        run_plan: ExactAttemptRunPlan,
        phi_per_cycle_maximum: BudgetVector,
    ) -> FixedAttemptBudgetPlan:
        rollout = trainer.rollout
        per_rollout = rollout.policy_call_maximum
        calls = 2 * rollout.max_turns
        input_tokens_per_call = per_rollout.input_tokens // calls
        return cls(
            batch_count=run_plan.total_training_steps,
            batch_size=trainer.execution.batch_size,
            max_turns=rollout.max_turns,
            reasoning_call_maximum=BudgetVector(
                input_tokens=input_tokens_per_call,
                output_tokens=rollout.max_reasoning_tokens,
                model_calls=1,
            ),
            action_call_maximum=BudgetVector(
                input_tokens=input_tokens_per_call,
                output_tokens=rollout.max_action_tokens,
                model_calls=1,
                agent_turns=1,
            ),
            tool_call_maximum=BudgetVector(
                tool_calls=1,
                wall_time_milliseconds=(per_rollout.wall_time_milliseconds // rollout.max_turns),
            ),
            maximum_cycles=run_plan.maximum_cycles,
            phi_per_cycle_maximum=phi_per_cycle_maximum,
            executor_call_maximum=rollout.executor_call_maximum or BudgetVector(),
        )

    def required(self) -> BudgetVector:
        rollout_count = self.batch_count * self.batch_size
        generation_count = rollout_count * self.max_turns
        tool_count = rollout_count * self.max_turns
        rollout = self.reasoning_call_maximum.scale(generation_count).add(
            self.action_call_maximum.scale(generation_count)
        )
        return (
            rollout.add(self.tool_call_maximum.scale(tool_count))
            .add(self.executor_call_maximum.scale(tool_count))
            .add(self.phi_per_cycle_maximum.scale(self.maximum_cycles))
        )

    def validate_against(self, cap: BudgetVector) -> None:
        required = self.required()
        if not required.fits_within(cap):
            raise BudgetExceededError(f"complete attempt envelope {required!r} exceeds cap {cap!r}")


@dataclass(frozen=True, slots=True)
class IdleTrainingState:
    kind: str = "idle"


@dataclass(frozen=True, slots=True)
class BatchReadyTrainingState:
    batch: CollectedTrainingBatch
    kind: str = "batch-ready"


@dataclass(frozen=True, slots=True)
class AppliedTrainingState:
    batch_id: str
    optimizer_step: int
    projection_installed: bool = False
    kind: str = "optimizer-applied"

    def __post_init__(self) -> None:
        _text(self.batch_id, field="batch_id")
        if type(self.optimizer_step) is not int or self.optimizer_step < 1:
            raise ValueError("applied optimizer step must be positive")
        if type(self.projection_installed) is not bool:
            raise TypeError("projection_installed must be boolean")


TrainingRuntimeState: TypeAlias = IdleTrainingState | BatchReadyTrainingState | AppliedTrainingState


__all__ = [
    "AppliedTrainingState",
    "BatchReadyTrainingState",
    "CollectedTrainingBatch",
    "FixedAttemptBudgetPlan",
    "IdleTrainingState",
    "PlannedRollout",
    "TrainingBatchPlan",
    "TrainingRuntimeState",
    "TrainingStepExecutionContext",
]
