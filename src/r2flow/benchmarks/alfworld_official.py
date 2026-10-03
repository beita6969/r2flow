from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from skillev.benchmarks.alfworld import (
    ALFWorldCommand,
    ALFWorldPublicItem,
    ALFWorldPublicStep,
)
from skillev.contracts import JsonValue, canonical_json, normalize_json
from skillev.runtime import BudgetVector

from .alfworld import (
    ALFWorldOutcomeUnavailableError,
    PrivateALFWorldCase,
    PrivateALFWorldEpisodeSession,
)
from .alfworld_public_goal import reset_public_goal

_PUBLIC_CONTEXT_FIELDS = {"admissible_commands", "initial_observation"}


def _text(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{field_name} must be non-empty text without NUL")
    return value


def _positive_int(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _seed(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("official ALFWorld seed must be a non-negative integer")
    return value


def _object(value: object, *, fields: set[str], label: str) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or set(normalized) != fields:
        raise ValueError(f"{label} has an incompatible field set")
    return normalized


def _admissible_commands(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError("official ALFWorld admissible commands must be a tuple")
    commands = tuple(_text(command, field_name="admissible command") for command in value)
    if len(set(commands)) != len(commands):
        raise ValueError("official ALFWorld admissible commands must be unique")
    return commands


def _public_context(item: ALFWorldPublicItem) -> tuple[str, tuple[str, ...]]:
    data = _object(
        item.public_context,
        fields=_PUBLIC_CONTEXT_FIELDS,
        label="official ALFWorld public context",
    )
    raw_commands = data["admissible_commands"]
    if not isinstance(raw_commands, list):
        raise TypeError("official ALFWorld public admissible commands must be an array")
    commands = _admissible_commands(tuple(raw_commands))
    return (
        _text(data["initial_observation"], field_name="public initial observation"),
        commands,
    )


@dataclass(frozen=True, slots=True)
class OfficialALFWorldTask:
    task_id: str
    environment_id: str
    game_id: str
    seed: int
    max_steps: int
    payload: object = field(repr=False)

    def __post_init__(self) -> None:
        for field_name in ("task_id", "environment_id", "game_id"):
            _text(getattr(self, field_name), field_name=field_name)
        _seed(self.seed)
        _positive_int(self.max_steps, field_name="max_steps")
        if self.payload is None:
            raise ValueError("official ALFWorld task payload cannot be absent")


@dataclass(frozen=True, slots=True)
class OfficialALFWorldResetResult:
    observation_text: str
    instruction_text: str
    admissible_commands: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_text",
            _text(self.observation_text, field_name="official reset observation"),
        )
        object.__setattr__(
            self,
            "instruction_text",
            _text(self.instruction_text, field_name="official instruction"),
        )
        object.__setattr__(
            self,
            "admissible_commands",
            _admissible_commands(self.admissible_commands),
        )


@dataclass(frozen=True, slots=True)
class OfficialALFWorldStepResult:
    observation_text: str
    admissible_commands: tuple[str, ...]
    terminal: bool
    success: bool | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_text",
            _text(self.observation_text, field_name="official observation"),
        )
        object.__setattr__(
            self,
            "admissible_commands",
            _admissible_commands(self.admissible_commands),
        )
        if type(self.terminal) is not bool:
            raise TypeError("official ALFWorld terminal flag must be boolean")
        if self.terminal:
            if type(self.success) is not bool:
                raise TypeError("terminal official ALFWorld step must carry a boolean success")
        elif self.success is not None:
            raise ValueError("non-terminal official ALFWorld step cannot carry success")


class OfficialALFWorldTextEnv(Protocol):
    @property
    def game_id(self) -> str: ...

    @property
    def seed(self) -> int: ...

    @property
    def max_steps(self) -> int: ...

    def reset(self, seed: int) -> OfficialALFWorldResetResult: ...

    def step(self, action: str) -> OfficialALFWorldStepResult: ...

    async def close(self) -> None: ...


class OfficialALFWorldTextEnvFactory(Protocol):
    def create(self, task: OfficialALFWorldTask) -> OfficialALFWorldTextEnv: ...


def _public_step(
    *,
    observation_text: str,
    admissible_commands: tuple[str, ...],
    terminal: bool,
    budget_usage: BudgetVector,
) -> ALFWorldPublicStep:
    return ALFWorldPublicStep(
        public_observation={
            "admissible_commands": list(admissible_commands),
            "terminal": terminal,
            "text": observation_text,
        },
        terminal=terminal,
        budget_usage=budget_usage,
    )


@dataclass(frozen=True, slots=True)
class _ExecutionEntry:
    step_index: int
    command: ALFWorldCommand
    public_step: ALFWorldPublicStep

    def __post_init__(self) -> None:
        _positive_int(self.step_index, field_name="official ALFWorld execution step")
        if not isinstance(self.command, ALFWorldCommand):
            raise TypeError("official ALFWorld command is invalid")
        if not isinstance(self.public_step, ALFWorldPublicStep):
            raise TypeError("official ALFWorld public step is invalid")


@dataclass(slots=True)
class _OfficialEpisodeState:
    case: PrivateALFWorldCase
    task: OfficialALFWorldTask
    env_factory: OfficialALFWorldTextEnvFactory
    env: OfficialALFWorldTextEnv
    observed_reset: OfficialALFWorldResetResult
    entries: list[_ExecutionEntry] = field(default_factory=list)
    terminal_success: bool | None = None

    @classmethod
    def create(
        cls,
        case: PrivateALFWorldCase,
        task: OfficialALFWorldTask,
        env_factory: OfficialALFWorldTextEnvFactory,
    ) -> _OfficialEpisodeState:
        env, observed_reset = _create_pinned_env(case, task, env_factory)
        return cls(
            case=case, task=task, env_factory=env_factory, env=env, observed_reset=observed_reset
        )

    async def execute(
        self,
        command: ALFWorldCommand,
        *,
        step_index: int,
    ) -> ALFWorldPublicStep:
        if self.entries and step_index <= self.entries[-1].step_index:
            raise ValueError("official ALFWorld steps must be strictly increasing")
        if len(self.entries) >= self.task.max_steps:
            raise ValueError("official ALFWorld execution exceeded its step limit")
        if self.terminal_success is not None:
            raise ValueError("official ALFWorld episode already terminated")

        result, elapsed_ns = await asyncio.to_thread(_timed_step, self.env, command.text)
        if not isinstance(result, OfficialALFWorldStepResult):
            raise TypeError("official ALFWorld env returned an incompatible step")
        if len(self.entries) + 1 == self.task.max_steps and not result.terminal:
            raise ValueError("official ALFWorld env did not terminate at its pinned step limit")
        public_step = _public_step(
            observation_text=result.observation_text,
            admissible_commands=result.admissible_commands,
            terminal=result.terminal,
            budget_usage=BudgetVector(
                tool_calls=1,
                wall_time_milliseconds=(elapsed_ns + 999_999) // 1_000_000,
            ),
        )
        self.entries.append(_ExecutionEntry(step_index, command, public_step))
        if result.terminal:
            self.terminal_success = result.success
        return public_step


def _timed_step(env: OfficialALFWorldTextEnv, text: str) -> tuple[Any, int]:
    started_ns = time.perf_counter_ns()
    result = env.step(text)
    return result, time.perf_counter_ns() - started_ns


def _create_pinned_env(
    case: PrivateALFWorldCase,
    task: OfficialALFWorldTask,
    env_factory: OfficialALFWorldTextEnvFactory,
) -> tuple[OfficialALFWorldTextEnv, OfficialALFWorldResetResult]:
    env = env_factory.create(task)
    try:
        if _text(env.game_id, field_name="official game_id") != task.game_id:
            raise ValueError("official ALFWorld env has another game identity")
        if _seed(env.seed) != task.seed:
            raise ValueError("official ALFWorld env has another seed")
        if _positive_int(env.max_steps, field_name="official max_steps") != task.max_steps:
            raise ValueError("official ALFWorld env has another step limit")
        reset = env.reset(task.seed)
        if not isinstance(reset, OfficialALFWorldResetResult):
            raise TypeError("official ALFWorld env returned an incompatible reset result")
        if env.game_id != task.game_id or env.seed != task.seed or env.max_steps != task.max_steps:
            raise ValueError("official ALFWorld reset changed its pinned identity")
        if reset_public_goal(reset.observation_text) != reset.instruction_text:
            raise ValueError("official ALFWorld reset instruction is not its observed public goal")
        if reset.instruction_text != case.public.query:
            raise ValueError("official ALFWorld instruction differs from the public query")
        expected_observation, expected_commands = _public_context(case.public)
        if (
            reset.observation_text != expected_observation
            or reset.admissible_commands != expected_commands
        ):
            raise ValueError("official ALFWorld reset differs from the public context")
    except (ValueError, TypeError):
        close_failed = getattr(env, "close_after_preparation_failure", None)
        if close_failed is not None:
            close_failed()
        raise
    return env, reset


@dataclass(frozen=True, slots=True)
class _OfficialEpisode:
    state: _OfficialEpisodeState = field(repr=False)

    @property
    def task_id(self) -> str:
        return self.state.task.task_id

    @property
    def environment_id(self) -> str:
        return self.state.task.environment_id

    @property
    def seed(self) -> int:
        return self.state.task.seed

    @property
    def max_steps(self) -> int:
        return self.state.task.max_steps

    async def execute(
        self,
        command: ALFWorldCommand,
        *,
        step_index: int,
    ) -> ALFWorldPublicStep:
        return await self.state.execute(command, step_index=step_index)


@dataclass(frozen=True, slots=True)
class _OfficialOutcomeView:
    state: _OfficialEpisodeState = field(repr=False)

    @property
    def task_id(self) -> str:
        return self.state.task.task_id

    @property
    def environment_id(self) -> str:
        return self.state.task.environment_id

    @property
    def seed(self) -> int:
        return self.state.task.seed

    def final_success(self) -> bool:
        if self.state.terminal_success is None:
            raise ALFWorldOutcomeUnavailableError(
                "official ALFWorld success is unavailable before terminal state"
            )
        return self.state.terminal_success


@dataclass(frozen=True, slots=True)
class OfficialALFWorldEpisodeFactory:
    env_factory: OfficialALFWorldTextEnvFactory

    def __post_init__(self) -> None:
        if not callable(getattr(self.env_factory, "create", None)):
            raise TypeError("official ALFWorld environment factory must implement create")

    def create(self, case: PrivateALFWorldCase) -> PrivateALFWorldEpisodeSession:
        if not isinstance(case.private_task, OfficialALFWorldTask):
            raise TypeError("official ALFWorld bridge requires OfficialALFWorldTask")
        task = case.private_task
        if task.task_id != case.public.task_id:
            raise ValueError("official ALFWorld task belongs to another public task")
        if task.environment_id != case.public.environment_id:
            raise ValueError("official ALFWorld task belongs to another environment snapshot")
        if task.seed != case.public.seed:
            raise ValueError("official ALFWorld task uses another seed")
        if task.max_steps != case.public.max_steps:
            raise ValueError("official ALFWorld task uses another step limit")
        state = _OfficialEpisodeState.create(case, task, self.env_factory)
        return PrivateALFWorldEpisodeSession(
            episode=_OfficialEpisode(state),
            outcome_view=_OfficialOutcomeView(state),
            cleanup=state.env.close,
            observed_reset_json=canonical_json(
                {
                    "game_id": state.env.game_id,
                    "seed": state.env.seed,
                    "max_steps": state.env.max_steps,
                    "instruction_text": state.observed_reset.instruction_text,
                    "observation_text": state.observed_reset.observation_text,
                    "admissible_commands": list(state.observed_reset.admissible_commands),
                }
            ),
        )


def bind_reset_public_item(
    public: ALFWorldPublicItem,
    reset: OfficialALFWorldResetResult,
) -> ALFWorldPublicItem:
    query = reset_public_goal(reset.observation_text)
    if reset.instruction_text != query:
        raise ValueError("ALFWorld reset instruction differs from its bound public goal")
    return replace(
        public,
        query=query,
        public_context={
            "admissible_commands": list(reset.admissible_commands),
            "initial_observation": reset.observation_text,
        },
    )


__all__ = [
    "OfficialALFWorldEpisodeFactory",
    "OfficialALFWorldResetResult",
    "OfficialALFWorldStepResult",
    "OfficialALFWorldTask",
    "OfficialALFWorldTextEnv",
    "OfficialALFWorldTextEnvFactory",
]
