from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

from skillev.contracts.answer_writer import accepted_observation
from skillev.contracts.canonical import JsonValue, normalize_json

if TYPE_CHECKING:
    from skillev.rollout.action_surface import ActionSurface

from .budget_ledger import BudgetLedger
from .contracts import (
    BudgetReservation,
    BudgetSettlement,
    BudgetVector,
    StructuredAction,
)
from .emitter import RuntimeEventEmitter
from .event_log import EventType
from .execution import (
    ActionParseResult,
    ActionParseStatus,
    EnvironmentObservation,
    RolloutEnvironmentSession,
)


@dataclass(frozen=True, slots=True)
class BoundedAgentState:
    invocation_id: str
    turns_used: int = 0
    completed: bool = False
    completion_value: JsonValue = None

    def __post_init__(self) -> None:
        if not self.invocation_id:
            raise ValueError("Bounded agent state requires an invocation ID")
        if type(self.turns_used) is not int or self.turns_used < 0:
            raise ValueError("Bounded agent turn count cannot be negative")
        if type(self.completed) is not bool:
            raise ValueError("Bounded agent completion status must be boolean")
        if not self.completed and self.completion_value is not None:
            raise ValueError("Incomplete bounded agent state cannot contain a completion")
        if normalize_json(self.completion_value) != self.completion_value:
            raise ValueError("Completion value must be normalized JSON")


@dataclass(frozen=True, slots=True)
class BoundedAgentTurnRequest:
    trajectory_id: str
    step_index: int
    action_text: str
    action_token_ids: tuple[int, ...]
    parse_result: ActionParseResult
    retrieved_skill_ids: tuple[str, ...]
    active_skill_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.trajectory_id:
            raise ValueError("Bounded agent turn requires a trajectory ID")
        if type(self.step_index) is not int or self.step_index < 1:
            raise ValueError("Bounded agent step index must be positive")
        if not isinstance(self.action_text, str) or not self.action_text:
            raise ValueError("Bounded agent action text cannot be empty")
        if (
            not isinstance(self.action_token_ids, tuple)
            or not self.action_token_ids
            or any(type(token_id) is not int or token_id < 0 for token_id in self.action_token_ids)
        ):
            raise ValueError("Bounded agent action tokens must be non-negative integers")
        if not isinstance(self.parse_result, ActionParseResult):
            raise TypeError("Bounded agent parse result has an incompatible shape")
        if (
            not isinstance(self.retrieved_skill_ids, tuple)
            or any(
                type(skill_id) is not str or not skill_id for skill_id in self.retrieved_skill_ids
            )
            or len(set(self.retrieved_skill_ids)) != len(self.retrieved_skill_ids)
        ):
            raise ValueError("retrieved_skill_ids must be unique non-empty text")
        if (
            not isinstance(self.active_skill_ids, tuple)
            or tuple(sorted(set(self.active_skill_ids))) != self.active_skill_ids
            or any(type(skill_id) is not str or not skill_id for skill_id in self.active_skill_ids)
        ):
            raise ValueError("active_skill_ids must be sorted unique non-empty text")
        if not set(self.retrieved_skill_ids) <= set(self.active_skill_ids):
            raise ValueError("retrieved_skill_ids reference inactive skills")


@dataclass(frozen=True, slots=True)
class BoundedAgentTurnResult:
    state: BoundedAgentState
    observation: EnvironmentObservation
    admitted: bool = False
    executed: bool = False


@dataclass(frozen=True, slots=True)
class BoundedAgentPolicy:
    max_turns: int

    def __post_init__(self) -> None:
        if type(self.max_turns) is not int or self.max_turns < 1:
            raise ValueError("Bounded agent must allow at least one turn")


class AgentTurnsExhaustedError(RuntimeError):
    pass


class EnvironmentMethodTimeoutError(RuntimeError):
    def __init__(
        self,
        *,
        budget_usage: BudgetVector,
        public_error_code: str = "environment_timeout",
    ) -> None:
        self.budget_usage = budget_usage
        self.public_error_code = public_error_code
        super().__init__("environment method timed out")


class EnvironmentMethodFailedError(RuntimeError):
    def __init__(
        self,
        *,
        budget_usage: BudgetVector,
        public_error_code: str = "environment_tool_error",
    ) -> None:
        self.budget_usage = budget_usage
        self.public_error_code = public_error_code
        super().__init__("environment method failed")


class EnvironmentSkillInvocationMismatchError(RuntimeError):
    pass


class BoundedAgent:
    def __init__(
        self,
        *,
        environment: RolloutEnvironmentSession,
        policy: BoundedAgentPolicy,
        ledger: BudgetLedger,
        tool_call_maximum: BudgetVector,
        emitter: RuntimeEventEmitter,
        action_surface: ActionSurface | None = None,
    ) -> None:
        if tool_call_maximum.tool_calls != 1:
            raise ValueError("Each executable action must reserve one tool call")
        self._environment = environment
        self._policy = policy
        self._ledger = ledger
        self._tool_call_maximum = tool_call_maximum
        self._emitter = emitter
        self._action_surface = action_surface

    @property
    def environment_id(self) -> str:
        return self._environment.environment_id

    @property
    def task_family(self) -> str:
        return self._environment.task_family

    @property
    def max_turns(self) -> int:
        return self._policy.max_turns

    async def execute_turn(
        self,
        state: BoundedAgentState,
        request: BoundedAgentTurnRequest,
    ) -> BoundedAgentTurnResult:
        if state.completed:
            raise ValueError("A completed bounded agent cannot execute another turn")
        if request.step_index != state.turns_used + 1:
            raise ValueError("Bounded agent step index is not contiguous")
        if state.turns_used >= self._policy.max_turns:
            raise AgentTurnsExhaustedError("Bounded agent exhausted its turn budget")

        self._emitter.emit(
            EventType.AGENT_TURN_STARTED,
            {
                "invocation_id": state.invocation_id,
                "trajectory_id": request.trajectory_id,
                "turn": request.step_index,
            },
        )
        parse_status = request.parse_result.status
        self._emitter.emit(
            EventType.AGENT_ACTION_PARSED,
            {
                "invocation_id": state.invocation_id,
                "status": parse_status.value,
                "turn": request.step_index,
            },
        )

        from skillev.rollout.action_contract import ActionContract

        contract = ActionContract.freeze(
            self._action_surface,
            retrieved_skill_ids=request.retrieved_skill_ids,
            active_skill_ids=request.active_skill_ids,
        )
        admitted = executed = accepted_submission = False
        working_state = state
        if parse_status is ActionParseStatus.PARSE_ERROR:
            observation = _invalid_observation(
                status="parse_error",
                error_code=cast(str, request.parse_result.public_error_code),
            )
        elif parse_status is ActionParseStatus.SCHEMA_INVALID:
            observation = _invalid_observation(
                status="schema_invalid",
                error_code=cast(str, request.parse_result.public_error_code),
            )
        else:
            action = cast(StructuredAction, request.parse_result.action)
            admission = contract.validate(
                action,
                validate_completion=self._environment.validate_completion,
            )
            if admission.error is not None:
                status, error_code = admission.error
                observation = _invalid_observation(status=status, error_code=error_code)
            elif admission.completion:
                admitted = accepted_submission = True
                observation = EnvironmentObservation(
                    public_value=accepted_observation(
                        cast(dict[str, str], admission.submission)["answer"]
                    )
                    if self._action_surface is not None
                    and self._action_surface.completion_writer is not None
                    else {"status": "accepted_for_evaluation"},
                    observation_status="success",
                    terminal=True,
                    terminal_submission=admission.submission,
                )
            else:
                admitted = executed = True
                observation, working_state = await self._execute_action(
                    state=working_state,
                    request=request,
                    action=admission.action,
                    expected_skill_ids=admission.invoked_skill_ids,
                )

        completed = observation.terminal
        updated = replace(
            working_state,
            turns_used=request.step_index,
            completed=completed,
            completion_value=(observation.terminal_submission if completed else None),
        )
        self._emitter.emit(
            EventType.AGENT_STEP_RECORDED,
            {
                "trajectory_id": request.trajectory_id,
                "action_token_ids": list(request.action_token_ids),
                "observation_text": observation.observation_text,
                "assessment": {
                    "parse_status": parse_status.value,
                    "admitted": admitted,
                    "executed": executed,
                    "execution_status": observation.observation_status if executed else None,
                    "accepted_submission": accepted_submission,
                    "environment_terminal": executed and observation.terminal,
                    "terminal_task_success": None,
                },
                "action_token_count": len(request.action_token_ids),
                "invocation_id": state.invocation_id,
                "observation": observation.to_value(),
                "turn": request.step_index,
            },
        )
        if completed:
            self._emitter.emit(
                EventType.AGENT_COMPLETED,
                {
                    "invocation_id": state.invocation_id,
                    "turns_used": updated.turns_used,
                },
            )
        return BoundedAgentTurnResult(
            state=updated, observation=observation, admitted=admitted, executed=executed
        )

    async def _execute_action(
        self,
        *,
        state: BoundedAgentState,
        request: BoundedAgentTurnRequest,
        action: StructuredAction,
        expected_skill_ids: tuple[str, ...],
    ) -> tuple[EnvironmentObservation, BoundedAgentState]:
        from skillev.rollout.errors import EpisodeInfrastructureError

        reservation = self._reserve(state, step_index=request.step_index)
        try:
            observation = await self._environment.execute(
                action,
                step_index=request.step_index,
            )
        except EnvironmentMethodTimeoutError as error:
            observation = EnvironmentObservation(
                public_value={"error": error.public_error_code},
                observation_status="timeout",
                invoked_skill_ids=expected_skill_ids,
                budget_usage=error.budget_usage,
            )
        except EnvironmentMethodFailedError as error:
            observation = EnvironmentObservation(
                public_value={"error": error.public_error_code},
                observation_status="tool_error",
                invoked_skill_ids=expected_skill_ids,
                budget_usage=error.budget_usage,
            )
        except EpisodeInfrastructureError:
            self._settle(reservation, reservation.maximum)
            raise

        if observation.invoked_skill_ids != expected_skill_ids:
            raise EnvironmentSkillInvocationMismatchError(
                "environment invocation credit differs from the structured action"
            )
        if observation.budget_usage.tool_calls != 1:
            raise ValueError("environment must report exactly one measured tool call")
        self._settle(reservation, observation.budget_usage)
        return observation, state

    def _settle(self, reservation: BudgetReservation, actual: BudgetVector) -> None:
        settlement = BudgetSettlement(
            reservation_id=reservation.reservation_id,
            actual=actual,
        )
        self._ledger.settle(settlement)
        self._emitter.emit(
            EventType.BUDGET_SETTLED,
            {
                "actual": settlement.actual.to_value(),
                "reservation_id": settlement.reservation_id,
            },
        )

    def _reserve(
        self,
        state: BoundedAgentState,
        *,
        step_index: int,
    ) -> BudgetReservation:
        reservation = BudgetReservation(
            reservation_id=f"{state.invocation_id}:{step_index}:tool",
            run_id=self._ledger.run_id,
            attempt_id=self._ledger.attempt_id,
            invocation_id=state.invocation_id,
            maximum=self._tool_call_maximum,
        )
        self._ledger.reserve(reservation)
        self._emitter.emit(
            EventType.BUDGET_RESERVED,
            {
                "maximum": reservation.maximum.to_value(),
                "reservation_id": reservation.reservation_id,
            },
        )
        return reservation


def _invalid_observation(*, status: str, error_code: str) -> EnvironmentObservation:
    return EnvironmentObservation(
        public_value={"error": error_code},
        observation_status=status,
    )


__all__ = [
    "AgentTurnsExhaustedError",
    "BoundedAgent",
    "BoundedAgentPolicy",
    "BoundedAgentState",
    "BoundedAgentTurnRequest",
    "BoundedAgentTurnResult",
    "EnvironmentMethodFailedError",
    "EnvironmentMethodTimeoutError",
    "EnvironmentSkillInvocationMismatchError",
]
