from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from skillev.contracts import JsonValue, normalize_json, stable_hash
from skillev.contracts.ttb_source_events import RunCursorValue

EXACT_ATTEMPT_RUN_PLAN_FORMAT: Final = "skillev-exact-attempt-run-plan@2"
ATTEMPT_RUN_CURSOR_FORMAT: Final = "skillev-attempt-run-cursor@2"


class RunSlotKind(StrEnum):
    PHASE_SEARCH = "phase-search"
    CLOSURE = "closure"


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _non_negative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class ExactAttemptRunPlan:
    phase_search_steps: int
    closure_steps: int
    maximum_cycles: int

    def __post_init__(self) -> None:
        _positive_int(self.phase_search_steps, field="phase_search_steps")
        _positive_int(self.closure_steps, field="closure_steps")
        _positive_int(self.maximum_cycles, field="maximum_cycles")

    @property
    def total_training_steps(self) -> int:
        return self.phase_search_steps + self.closure_steps

    def slot_kind(self, ordinal: int) -> RunSlotKind:
        if type(ordinal) is not int or not 1 <= ordinal <= self.total_training_steps:
            raise ValueError("training-step ordinal is outside the frozen run plan")
        if ordinal <= self.phase_search_steps:
            return RunSlotKind.PHASE_SEARCH
        return RunSlotKind.CLOSURE

    def remaining_from(self, cursor: AttemptRunCursorState) -> RemainingAttemptCapacity:
        cursor.require_plan(self)
        return RemainingAttemptCapacity(
            training_steps=self.total_training_steps - cursor.completed_training_steps,
            possible_cycles=self.maximum_cycles - cursor.committed_cycles,
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "closure_steps": self.closure_steps,
            "format": EXACT_ATTEMPT_RUN_PLAN_FORMAT,
            "maximum_cycles": self.maximum_cycles,
            "phase_search_steps": self.phase_search_steps,
        }

    @classmethod
    def from_value(cls, value: object) -> ExactAttemptRunPlan:
        normalized = normalize_json(value)
        if not isinstance(normalized, dict) or set(normalized) != {
            "closure_steps",
            "format",
            "maximum_cycles",
            "phase_search_steps",
        }:
            raise ValueError("ExactAttemptRunPlan has incompatible fields")
        if normalized["format"] != EXACT_ATTEMPT_RUN_PLAN_FORMAT:
            raise ValueError("unsupported exact attempt run-plan format")
        return cls(
            phase_search_steps=_positive_int(
                normalized["phase_search_steps"], field="phase_search_steps"
            ),
            closure_steps=_positive_int(normalized["closure_steps"], field="closure_steps"),
            maximum_cycles=_positive_int(normalized["maximum_cycles"], field="maximum_cycles"),
        )


@dataclass(frozen=True, slots=True)
class AttemptRunCursorState:
    run_plan_hash: str
    completed_training_steps: int
    committed_cycles: int
    committed_actions: int
    format: str = ATTEMPT_RUN_CURSOR_FORMAT

    def __post_init__(self) -> None:
        if type(self.run_plan_hash) is not str or not self.run_plan_hash.startswith("sha256:"):
            raise ValueError("run_plan_hash must be a content hash")
        _non_negative_int(
            self.completed_training_steps,
            field="completed_training_steps",
        )
        _non_negative_int(self.committed_cycles, field="committed_cycles")
        _non_negative_int(self.committed_actions, field="committed_actions")
        if self.format != ATTEMPT_RUN_CURSOR_FORMAT:
            raise ValueError("unsupported attempt run cursor format")

    @classmethod
    def fresh(cls, plan: ExactAttemptRunPlan) -> AttemptRunCursorState:
        return cls(
            run_plan_hash=plan.content_hash,
            completed_training_steps=0,
            committed_cycles=0,
            committed_actions=0,
        )

    def require_plan(self, plan: ExactAttemptRunPlan) -> None:
        if self.run_plan_hash != plan.content_hash:
            raise ValueError("run cursor belongs to another exact run plan")
        if self.completed_training_steps > plan.total_training_steps:
            raise ValueError("run cursor is beyond the exact run plan")
        if self.committed_cycles > plan.maximum_cycles:
            raise ValueError("run cursor exceeds the exact cycle bound")

    def after_training_step(self, plan: ExactAttemptRunPlan) -> AttemptRunCursorState:
        self.require_plan(plan)
        next_step = self.completed_training_steps + 1
        if next_step > plan.total_training_steps:
            raise RuntimeError("run cursor exceeded exact training-step plan")
        return AttemptRunCursorState(
            run_plan_hash=self.run_plan_hash,
            completed_training_steps=next_step,
            committed_cycles=self.committed_cycles,
            committed_actions=self.committed_actions,
        )

    def after_cycle(
        self,
        plan: ExactAttemptRunPlan,
        *,
        action_count: int,
    ) -> AttemptRunCursorState:
        self.require_plan(plan)
        if type(action_count) is not int or action_count < 1:
            raise ValueError("a committed Phi cycle requires positive action_count")
        next_cycle = self.committed_cycles + 1
        if next_cycle > plan.maximum_cycles:
            raise RuntimeError("verified phases exceed predeclared cycle bound")
        return AttemptRunCursorState(
            run_plan_hash=self.run_plan_hash,
            completed_training_steps=self.completed_training_steps,
            committed_cycles=next_cycle,
            committed_actions=self.committed_actions + action_count,
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "committed_actions": self.committed_actions,
            "committed_cycles": self.committed_cycles,
            "completed_training_steps": self.completed_training_steps,
            "format": self.format,
            "run_plan_hash": self.run_plan_hash,
        }

    def to_source_value(self) -> RunCursorValue:
        return RunCursorValue(
            run_plan_hash=self.run_plan_hash,
            completed_training_steps=self.completed_training_steps,
            committed_cycles=self.committed_cycles,
            committed_actions=self.committed_actions,
        )

    @classmethod
    def from_value(cls, value: object) -> AttemptRunCursorState:
        normalized = normalize_json(value)
        fields = {
            "committed_actions",
            "committed_cycles",
            "completed_training_steps",
            "format",
            "run_plan_hash",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("AttemptRunCursorState has incompatible fields")
        if type(normalized["run_plan_hash"]) is not str:
            raise TypeError("run_plan_hash must be text")
        if type(normalized["format"]) is not str:
            raise TypeError("run cursor format must be text")
        return cls(
            run_plan_hash=normalized["run_plan_hash"],
            completed_training_steps=_non_negative_int(
                normalized["completed_training_steps"],
                field="completed_training_steps",
            ),
            committed_cycles=_non_negative_int(
                normalized["committed_cycles"], field="committed_cycles"
            ),
            committed_actions=_non_negative_int(
                normalized["committed_actions"], field="committed_actions"
            ),
            format=normalized["format"],
        )

    @classmethod
    def from_source_value(cls, value: RunCursorValue) -> AttemptRunCursorState:
        if not isinstance(value, RunCursorValue):
            raise TypeError("source cursor must be RunCursorValue")
        return cls(
            run_plan_hash=value.run_plan_hash,
            completed_training_steps=value.completed_training_steps,
            committed_cycles=value.committed_cycles,
            committed_actions=value.committed_actions,
        )


@dataclass(frozen=True, slots=True)
class RemainingAttemptCapacity:
    training_steps: int
    possible_cycles: int

    def __post_init__(self) -> None:
        _non_negative_int(self.training_steps, field="training_steps")
        _non_negative_int(self.possible_cycles, field="possible_cycles")


@dataclass(slots=True)
class AttemptRunProgress:
    plan: ExactAttemptRunPlan
    _state: AttemptRunCursorState

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ExactAttemptRunPlan):
            raise TypeError("plan must be ExactAttemptRunPlan")
        self._state.require_plan(self.plan)

    @classmethod
    def fresh(cls, plan: ExactAttemptRunPlan) -> AttemptRunProgress:
        return cls(plan=plan, _state=AttemptRunCursorState.fresh(plan))

    @classmethod
    def from_state(
        cls,
        plan: ExactAttemptRunPlan,
        state: AttemptRunCursorState,
    ) -> AttemptRunProgress:
        return cls(plan=plan, _state=state)

    @property
    def state(self) -> AttemptRunCursorState:
        return self._state

    def preview_training_step(self) -> AttemptRunCursorState:
        return self._state.after_training_step(self.plan)

    def commit_training_step(self) -> AttemptRunCursorState:
        self._state = self.preview_training_step()
        return self._state

    def commit_training_step_state(self, state: AttemptRunCursorState) -> None:
        state.require_plan(self.plan)
        if state.completed_training_steps != self._state.completed_training_steps + 1:
            raise ValueError("training cursor must advance exactly one step")
        if state.committed_cycles != self._state.committed_cycles:
            raise ValueError("training cursor cannot change committed cycles")
        if state.committed_actions != self._state.committed_actions:
            raise ValueError("training cursor cannot change committed actions")
        self._state = state

    def preview_cycle(self, *, action_count: int) -> AttemptRunCursorState:
        return self._state.after_cycle(self.plan, action_count=action_count)

    def commit_cycle(self, state: AttemptRunCursorState) -> None:
        state.require_plan(self.plan)
        if state.completed_training_steps != self._state.completed_training_steps:
            raise ValueError("cycle cursor changes training-step position")
        if state.committed_cycles != self._state.committed_cycles + 1:
            raise ValueError("cycle cursor must advance exactly one cycle")
        if state.committed_actions <= self._state.committed_actions:
            raise ValueError("cycle cursor must advance committed actions")
        self._state = state


__all__ = [
    "ATTEMPT_RUN_CURSOR_FORMAT",
    "EXACT_ATTEMPT_RUN_PLAN_FORMAT",
    "AttemptRunCursorState",
    "AttemptRunProgress",
    "ExactAttemptRunPlan",
    "RemainingAttemptCapacity",
    "RunSlotKind",
]
