from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Protocol

from skillev.contracts import JsonValue, normalize_json, stable_hash
from skillev.rollout import RolloutTask
from skillev.runtime import (
    ActionKind,
    BudgetVector,
    EnvironmentObservation,
    OrderedTaskCursorState,
    StructuredAction,
)

from .task_family import require_benchmark_task_family


def _text(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{field_name} must be non-empty text without NUL")
    normalized = normalize_json(value)
    if type(normalized) is not str:
        raise TypeError(f"{field_name} must normalize to text")
    return normalized


def _positive_int(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class EmbodiedPublicItem:
    dataset_revision: str
    environment_snapshot_id: str
    split: str
    task_id: str
    task_family: str
    query: str
    public_context: JsonValue
    seed: int
    max_steps: int

    BENCHMARK_ID: ClassVar[str] = ""
    RESOURCE_ID: ClassVar[str] = ""

    def __post_init__(self) -> None:
        if not self.BENCHMARK_ID or not self.RESOURCE_ID:
            raise TypeError("embodied public item must declare benchmark and resource identities")
        for field_name in (
            "dataset_revision",
            "environment_snapshot_id",
            "split",
            "task_id",
            "task_family",
            "query",
        ):
            object.__setattr__(
                self,
                field_name,
                _text(getattr(self, field_name), field_name=field_name),
            )
        require_benchmark_task_family(
            benchmark_id=self.BENCHMARK_ID,
            task_family=self.task_family,
        )
        object.__setattr__(self, "public_context", normalize_json(self.public_context))
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("embodied simulator seed must be a non-negative integer")
        _positive_int(self.max_steps, field_name="max_steps")

    @property
    def environment_id(self) -> str:
        return (
            f"benchmark:{self.BENCHMARK_ID}@{self.dataset_revision}:"
            f"environment:{self.environment_snapshot_id}:seed:{self.seed}"
        )

    def to_rollout_task(self) -> RolloutTask:
        return RolloutTask(
            task_id=self.task_id,
            environment_id=self.environment_id,
            task_family=self.task_family,
            context_id=self.environment_snapshot_id,
            query=self.query,
            available_tools=("act",),
            public_context={
                "benchmark_id": self.BENCHMARK_ID,
                "dataset_revision": self.dataset_revision,
                "environment_snapshot_id": self.environment_snapshot_id,
                "max_steps": self.max_steps,
                "payload": self.public_context,
                "seed": self.seed,
                "split": self.split,
                "tools": {"act": {"arguments": ["command"]}},
            },
        )


@dataclass(slots=True)
class OrderedEmbodiedTaskProvider:
    items: tuple[EmbodiedPublicItem, ...]
    cursor: int = 0

    ITEM_TYPE: ClassVar[type[EmbodiedPublicItem]] = EmbodiedPublicItem

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("embodied task provider requires at least one item")
        if any(not isinstance(item, self.ITEM_TYPE) for item in self.items):
            raise TypeError("embodied task provider contains an incompatible item")
        identities = tuple(item.task_id for item in self.items)
        if len(set(identities)) != len(identities):
            raise ValueError("embodied task identities must be unique")
        if type(self.cursor) is not int or not 0 <= self.cursor <= len(self.items):
            raise ValueError("embodied task provider cursor is invalid")

    def next_task(self) -> RolloutTask:
        if self.cursor >= len(self.items):
            raise RuntimeError("embodied public curriculum is exhausted")
        task = self.items[self.cursor].to_rollout_task()
        self.cursor += 1
        return task

    @property
    def runtime_state(self) -> OrderedTaskCursorState:
        return OrderedTaskCursorState(
            curriculum_id=stable_hash(
                {
                    "kind": self.ITEM_TYPE.__name__,
                    "task_ids": [item.task_id for item in self.items],
                }
            ),
            cursor=self.cursor,
        )


@dataclass(frozen=True, slots=True)
class EmbodiedCommand:
    text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _text(self.text, field_name="command"))

    def to_value(self) -> dict[str, JsonValue]:
        return {"text": self.text}


@dataclass(frozen=True, slots=True)
class EmbodiedPublicStep:
    public_observation: JsonValue
    terminal: bool
    budget_usage: BudgetVector

    def __post_init__(self) -> None:
        object.__setattr__(self, "public_observation", normalize_json(self.public_observation))
        if type(self.terminal) is not bool:
            raise TypeError("embodied terminal flag must be boolean")
        if not isinstance(self.budget_usage, BudgetVector):
            raise TypeError("embodied budget usage must be a BudgetVector")


class EmbodiedEpisode(Protocol):
    @property
    def task_id(self) -> str: ...

    @property
    def environment_id(self) -> str: ...

    @property
    def seed(self) -> int: ...

    @property
    def max_steps(self) -> int: ...

    async def execute(
        self,
        command: EmbodiedCommand,
        *,
        step_index: int,
    ) -> EmbodiedPublicStep: ...


@dataclass(frozen=True, slots=True)
class _ExecutionRecord:
    step_index: int
    action: StructuredAction
    observation: EnvironmentObservation

    def __post_init__(self) -> None:
        _positive_int(self.step_index, field_name="step_index")


@dataclass(slots=True)
class EmbodiedTextEnvironment:
    item: EmbodiedPublicItem
    episode: EmbodiedEpisode
    _records: list[_ExecutionRecord] = field(default_factory=list, init=False, repr=False)

    ITEM_TYPE: ClassVar[type[EmbodiedPublicItem]] = EmbodiedPublicItem

    def __post_init__(self) -> None:
        if not isinstance(self.item, self.ITEM_TYPE):
            raise TypeError("embodied environment received an incompatible public item")
        if self.episode.task_id != self.item.task_id:
            raise ValueError("embodied episode belongs to another task")
        if self.episode.environment_id != self.item.environment_id:
            raise ValueError("embodied episode belongs to another environment snapshot")
        if self.episode.seed != self.item.seed:
            raise ValueError("embodied episode uses a different deterministic seed")
        if self.episode.max_steps != self.item.max_steps:
            raise ValueError("embodied episode uses a different step limit")

    @property
    def environment_id(self) -> str:
        return self.item.environment_id

    @property
    def task_family(self) -> str:
        return self.item.task_family

    async def execute(
        self,
        action: StructuredAction,
        *,
        step_index: int,
    ) -> EnvironmentObservation:
        if not isinstance(action, StructuredAction):
            raise TypeError("embodied action must be StructuredAction")
        _positive_int(step_index, field_name="step_index")
        if self._records and step_index <= self._records[-1].step_index:
            raise ValueError("embodied execution steps must be strictly increasing")
        if step_index > self.item.max_steps:
            raise ValueError("embodied execution exceeded its configured step limit")

        limit_reached = len(self._records) + 1 == self.item.max_steps
        if action.kind is ActionKind.SKILL and action.skill_id is not None:
            terminal = limit_reached
            observation = EnvironmentObservation(
                public_value={"status": "skill-invoked"},
                observation_status="success",
                invoked_skill_ids=(action.skill_id,),
                terminal_submission=self._terminal_submission("step-limit") if terminal else None,
                terminal=terminal,
                budget_usage=BudgetVector(tool_calls=1),
            )
        else:
            command = self._command_from_action(action)
            if command is None:
                terminal = limit_reached
                observation = EnvironmentObservation(
                    public_value={"error": f"unsupported_{self.item.BENCHMARK_ID}_action"},
                    observation_status="tool_error",
                    terminal_submission=(
                        self._terminal_submission("step-limit") if terminal else None
                    ),
                    terminal=terminal,
                    budget_usage=BudgetVector(tool_calls=1),
                )
            else:
                public_step = await self.episode.execute(
                    command,
                    step_index=step_index,
                )
                if not isinstance(public_step, EmbodiedPublicStep):
                    raise TypeError("embodied episode returned an incompatible step")
                if public_step.budget_usage.tool_calls != 1:
                    raise ValueError("embodied episode must report exactly one measured tool call")
                terminal = public_step.terminal or limit_reached
                reason = "simulator" if public_step.terminal else "step-limit"
                observation = EnvironmentObservation(
                    public_value=public_step.public_observation,
                    observation_status="success",
                    terminal_submission=(self._terminal_submission(reason) if terminal else None),
                    terminal=terminal,
                    budget_usage=public_step.budget_usage,
                )

        self._records.append(
            _ExecutionRecord(
                step_index=step_index,
                action=action,
                observation=observation,
            )
        )
        return observation

    def _command_from_action(self, action: StructuredAction) -> EmbodiedCommand | None:
        if (
            action.kind is not ActionKind.TOOL
            or action.resource_id != self.item.RESOURCE_ID
            or action.name != "act"
            or not isinstance(action.arguments, dict)
            or set(action.arguments) != {"command"}
        ):
            return None
        command = action.arguments["command"]
        if type(command) is not str or not command.strip():
            return None
        return EmbodiedCommand(command)

    def _terminal_submission(self, reason: str) -> dict[str, JsonValue]:
        return {
            "benchmark_id": self.item.BENCHMARK_ID,
            "episode_terminal": True,
            "reason": reason,
            "task_id": self.item.task_id,
        }

    def validate_completion(self, submission: JsonValue) -> bool:
        del submission
        return False


__all__ = [
    "EmbodiedCommand",
    "EmbodiedEpisode",
    "EmbodiedPublicItem",
    "EmbodiedPublicStep",
    "EmbodiedTextEnvironment",
    "OrderedEmbodiedTaskProvider",
]
