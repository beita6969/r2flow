from dataclasses import dataclass
from typing import cast

from skillev.contracts import JsonValue, TrajectoryRecord
from skillev.contracts.skill_invocation import parse_action_invocation


@dataclass(frozen=True, slots=True)
class InvocationExecutionLink:
    step_index: int
    declared_skill_id: str
    admitted: bool
    observation_status: str
    following_execution_steps: tuple[int, ...]
    terminal_success: bool
    later_execution_steps: tuple[int, ...]
    read_ordinal: int
    preceding_observation_status: str | None
    following_task_action_steps: tuple[int, ...]

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "read_ordinal": self.read_ordinal,
            "repeat_same_version": None,
            "preceding_observation_status": self.preceding_observation_status,
            "following_task_action_steps": list(self.following_task_action_steps),
            "read_followed_by_task_action": bool(self.following_task_action_steps),
            "efficacy": "not-inferred-from-read-or-temporal-following",
            "later_execution_steps": cast(JsonValue, list(self.later_execution_steps)),
            "step_index": self.step_index,
            "declared_skill_id": self.declared_skill_id,
            "admitted": self.admitted,
            "observation_status": self.observation_status,
            "following_execution_steps": list(self.following_execution_steps),
            "terminal_success": self.terminal_success,
            "label_source": "TerminalReward.success",
            "interpretation": "explicit-strategy-declaration-not-independent-skill-execution",
        }

    @classmethod
    def from_value(cls, value: object) -> "InvocationExecutionLink":
        if not isinstance(value, dict):
            raise TypeError("invocation evidence must be an object")
        if value.get("label_source") != "TerminalReward.success":
            raise ValueError("invocation evidence must use the terminal Bernoulli label")
        return cls(
            value["step_index"],
            value["declared_skill_id"],
            value["admitted"],
            value["observation_status"],
            tuple(value["following_execution_steps"]),
            value["terminal_success"],
            tuple(value["later_execution_steps"]),
            value["read_ordinal"],
            value["preceding_observation_status"],
            tuple(value["following_task_action_steps"]),
        )


def invocation_execution_links(record: TrajectoryRecord) -> tuple[InvocationExecutionLink, ...]:
    parsed = [
        parse_action_invocation(step.action_text, initial_meta=record.initial_context.meta)
        for step in record.steps
    ]
    result: list[InvocationExecutionLink] = []
    for offset, (step, action) in enumerate(zip(record.steps, parsed, strict=True)):
        if action is None or action.skill_id is None:
            continue
        following = []
        for later_step, later_action in zip(
            record.steps[offset + 1 :], parsed[offset + 1 :], strict=True
        ):
            if later_action is not None and later_action.skill_id is not None:
                break
            following.append(later_step.index)
        later = tuple(
            later_step.index
            for later_step, later_action in zip(
                record.steps[offset + 1 :], parsed[offset + 1 :], strict=True
            )
            if later_action is not None and later_action.skill_id is None
        )
        following_task = tuple(
            candidate.index
            for candidate, parsed_action in zip(
                record.steps[offset + 1 :], parsed[offset + 1 :], strict=True
            )
            if candidate.index in following
            and parsed_action is not None
            and parsed_action.skill_id is None
        )
        result.append(
            InvocationExecutionLink(
                step.index,
                action.skill_id,
                action.skill_id in step.invoked_skill_ids,
                step.observation_status,
                tuple(following),
                record.reward.success,
                later,
                len(result) + 1,
                record.steps[offset - 1].observation_status if offset else None,
                following_task,
            )
        )
    return tuple(result)
