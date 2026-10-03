from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from .canonical import JsonValue, normalize_json, stable_hash
from .identity import validate_identifier, validate_sha256
from .skill_invocation import validate_trajectory_skill_invocations
from .ttb_common import (
    FLOAT_TOLERANCE,
    TokenizerProtocol,
    require_canonical_mapping,
    require_finite_number,
    require_iso_timestamp,
    require_non_empty_text,
)
from .ttb_reward import TerminalReward

OBSERVATION_STATUSES = frozenset(
    {
        "success",
        "tool_error",
        "schema_invalid",
        "timeout",
        "parse_error",
        "other",
    }
)


def _require_json_object(value: object, *, label: str) -> dict[str, JsonValue]:
    try:
        normalized = normalize_json(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a JSON object") from error
    if not isinstance(normalized, dict):
        raise ValueError(f"{label} must be a JSON object")
    return normalized


def _require_exact_fields(
    value: dict[str, JsonValue],
    *,
    fields: set[str],
    label: str,
) -> None:
    if set(value) != fields:
        raise ValueError(f"{label} has incompatible fields")


def _require_string(value: JsonValue, *, field: str, location: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{location}: {field} must be text")
    return value


def _require_int(value: JsonValue, *, field: str, location: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{location}: {field} must be an integer")
    return value


def _require_float(value: JsonValue, *, field: str, location: str) -> float:
    return require_finite_number(value, field=field, location=location)


def _require_string_tuple(value: JsonValue, *, field: str, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{location}: {field} must be an array of strings")
    return tuple(cast(list[str], value))


@dataclass(frozen=True, slots=True)
class InitialContext:
    query: str
    retrieved_skill_ids: tuple[str, ...]
    active_skill_ids: tuple[str, ...]
    meta: Mapping[str, JsonValue]
    assembler_version: str
    assembled_hash: str
    assembled_token_count: int

    def __post_init__(self) -> None:
        location = "initial context"
        require_non_empty_text(self.query, field="query", location=location)
        if not isinstance(self.retrieved_skill_ids, tuple):
            raise ValueError(f"{location}: retrieved_skill_ids must be a tuple")
        for skill_id in self.retrieved_skill_ids:
            if not isinstance(skill_id, str):
                raise ValueError(f"{location}: retrieved_skill_ids must contain strings")
            validate_identifier(skill_id)
        if len(set(self.retrieved_skill_ids)) != len(self.retrieved_skill_ids):
            raise ValueError(f"{location}: retrieved_skill_ids must be unique")
        if not isinstance(self.active_skill_ids, tuple):
            raise ValueError(f"{location}: active_skill_ids must be a tuple")
        for skill_id in self.active_skill_ids:
            if not isinstance(skill_id, str):
                raise ValueError(f"{location}: active_skill_ids must contain strings")
            validate_identifier(skill_id)
        if tuple(sorted(set(self.active_skill_ids))) != self.active_skill_ids:
            raise ValueError(f"{location}: active_skill_ids must be sorted and unique")
        if not set(self.retrieved_skill_ids) <= set(self.active_skill_ids):
            raise ValueError(f"{location}: retrieved_skill_ids reference inactive skills")
        require_non_empty_text(
            self.assembler_version,
            field="assembler_version",
            location=location,
        )
        require_non_empty_text(self.assembled_hash, field="assembled_hash", location=location)
        validate_sha256(self.assembled_hash)
        if type(self.assembled_token_count) is not int or self.assembled_token_count <= 0:
            raise ValueError(f"{location}: assembled_token_count must be positive")
        normalized_meta = require_canonical_mapping(self.meta, field="meta", location=location)
        object.__setattr__(self, "meta", normalized_meta)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "assembled_hash": self.assembled_hash,
            "assembled_token_count": self.assembled_token_count,
            "assembler_version": self.assembler_version,
            "active_skill_ids": list(self.active_skill_ids),
            "meta": normalize_json(self.meta),
            "query": self.query,
            "retrieved_skill_ids": list(self.retrieved_skill_ids),
        }

    @classmethod
    def from_value(cls, value: object) -> InitialContext:
        normalized = _require_json_object(value, label="initial context")
        _require_exact_fields(
            normalized,
            fields={
                "assembled_hash",
                "assembled_token_count",
                "assembler_version",
                "active_skill_ids",
                "meta",
                "query",
                "retrieved_skill_ids",
            },
            label="initial context",
        )
        meta = normalized["meta"]
        if not isinstance(meta, dict):
            raise ValueError("initial context: meta must be a JSON object")
        return cls(
            query=_require_string(normalized["query"], field="query", location="initial context"),
            retrieved_skill_ids=_require_string_tuple(
                normalized["retrieved_skill_ids"],
                field="retrieved_skill_ids",
                location="initial context",
            ),
            active_skill_ids=_require_string_tuple(
                normalized["active_skill_ids"],
                field="active_skill_ids",
                location="initial context",
            ),
            meta=meta,
            assembler_version=_require_string(
                normalized["assembler_version"],
                field="assembler_version",
                location="initial context",
            ),
            assembled_hash=_require_string(
                normalized["assembled_hash"],
                field="assembled_hash",
                location="initial context",
            ),
            assembled_token_count=_require_int(
                normalized["assembled_token_count"],
                field="assembled_token_count",
                location="initial context",
            ),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


STEP_R2FLOW_RECORD_FORMAT = "step-r2flow@1"
_R2FLOW_TERMINALS = frozenset({"submitted", "environment-terminal", "horizon"})


def _token_tuple(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"step-r2flow: {field} must be a tuple of non-negative integers")
    return value


@dataclass(frozen=True, slots=True)
class StepR2FlowRecord:
    state_map: str
    event_label: str
    event_function: str
    event_args_sha256: str
    event_identity: str
    state_key: str
    predecessor_key: str
    rank: int
    in_edges: tuple[tuple[str, str], ...]
    actual_in_edge: int
    legal_event_set_sha256: str
    action_grammar_version: str
    action_grammar_sha256: str
    action_budget_tokens: int
    action_stop_token_ids: tuple[int, ...]
    action_finish_reason: str
    budget_forced_positions: tuple[int, ...]
    action_masks_digest: str
    reasoning_token_ids: tuple[int, ...]
    reasoning_token_ids_include_stop: bool
    reasoning_stop_token_ids: tuple[int, ...]
    reasoning_finish_reason: str
    terminal: str | None
    format: str = STEP_R2FLOW_RECORD_FORMAT

    def __post_init__(self) -> None:
        if self.format != STEP_R2FLOW_RECORD_FORMAT:
            raise ValueError("unsupported step-r2flow format")
        for name in (
            "state_map",
            "event_label",
            "event_function",
            "event_identity",
            "action_grammar_version",
            "action_finish_reason",
            "reasoning_finish_reason",
        ):
            require_non_empty_text(getattr(self, name), field=name, location="step-r2flow")
        for name in (
            "event_args_sha256",
            "state_key",
            "predecessor_key",
            "legal_event_set_sha256",
            "action_grammar_sha256",
            "action_masks_digest",
        ):
            value = getattr(self, name)
            if type(value) is not str or len(value) != 64 or value.strip("0123456789abcdef"):
                raise ValueError(f"step-r2flow: {name} must be a sha256 hex digest")
        if type(self.rank) is not int or self.rank < 1:
            raise ValueError("step-r2flow: rank must be a positive integer")
        if (
            not isinstance(self.in_edges, tuple)
            or not self.in_edges
            or any(
                not isinstance(edge, tuple)
                or len(edge) != 2
                or any(type(item) is not str for item in edge)
                for edge in self.in_edges
            )
        ):
            raise ValueError("step-r2flow: in_edges must be non-empty (predecessor, label) pairs")
        if type(self.actual_in_edge) is not int or not 0 <= self.actual_in_edge < len(
            self.in_edges
        ):
            raise ValueError("step-r2flow: actual_in_edge out of range")
        if self.in_edges[self.actual_in_edge][0] != self.predecessor_key:
            raise ValueError("step-r2flow: the actual in-edge must start at the predecessor")
        if type(self.action_budget_tokens) is not int or self.action_budget_tokens < 1:
            raise ValueError("step-r2flow: action_budget_tokens must be positive")
        for name in (
            "action_stop_token_ids",
            "budget_forced_positions",
            "reasoning_token_ids",
            "reasoning_stop_token_ids",
        ):
            _token_tuple(getattr(self, name), field=name)
        if type(self.reasoning_token_ids_include_stop) is not bool:
            raise ValueError("step-r2flow: reasoning_token_ids_include_stop must be boolean")
        if self.terminal is not None and self.terminal not in _R2FLOW_TERMINALS:
            raise ValueError("step-r2flow: unsupported terminal kind")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "state_map": self.state_map,
            "event_label": self.event_label,
            "event_function": self.event_function,
            "event_args_sha256": self.event_args_sha256,
            "event_identity": self.event_identity,
            "state_key": self.state_key,
            "predecessor_key": self.predecessor_key,
            "rank": self.rank,
            "in_edges": [
                {"predecessor_key": pred, "label_hash": label} for pred, label in self.in_edges
            ],
            "actual_in_edge": self.actual_in_edge,
            "legal_event_set_sha256": self.legal_event_set_sha256,
            "action_grammar_version": self.action_grammar_version,
            "action_grammar_sha256": self.action_grammar_sha256,
            "action_budget_tokens": self.action_budget_tokens,
            "action_stop_token_ids": list(self.action_stop_token_ids),
            "action_finish_reason": self.action_finish_reason,
            "budget_forced_positions": list(self.budget_forced_positions),
            "action_masks_digest": self.action_masks_digest,
            "reasoning_token_ids": list(self.reasoning_token_ids),
            "reasoning_token_ids_include_stop": self.reasoning_token_ids_include_stop,
            "reasoning_stop_token_ids": list(self.reasoning_stop_token_ids),
            "reasoning_finish_reason": self.reasoning_finish_reason,
            "terminal": self.terminal,
        }

    @classmethod
    def from_value(cls, value: object) -> StepR2FlowRecord:
        data = _require_json_object(value, label="step-r2flow")
        _require_exact_fields(
            data,
            fields={
                "format",
                "state_map",
                "event_label",
                "event_function",
                "event_args_sha256",
                "event_identity",
                "state_key",
                "predecessor_key",
                "rank",
                "in_edges",
                "actual_in_edge",
                "legal_event_set_sha256",
                "action_grammar_version",
                "action_grammar_sha256",
                "action_budget_tokens",
                "action_stop_token_ids",
                "action_finish_reason",
                "budget_forced_positions",
                "action_masks_digest",
                "reasoning_token_ids",
                "reasoning_token_ids_include_stop",
                "reasoning_stop_token_ids",
                "reasoning_finish_reason",
                "terminal",
            },
            label="step-r2flow",
        )
        edges = data["in_edges"]
        if not isinstance(edges, list) or any(
            not isinstance(edge, dict) or set(edge) != {"predecessor_key", "label_hash"}
            for edge in edges
        ):
            raise ValueError("step-r2flow: in_edges must be objects")

        def ids(name: str) -> tuple[int, ...]:
            raw = data[name]
            if not isinstance(raw, list):
                raise ValueError(f"step-r2flow: {name} must be an array")
            return tuple(_require_int(item, field=name, location="step-r2flow") for item in raw)

        terminal = data["terminal"]
        if terminal is not None and not isinstance(terminal, str):
            raise ValueError("step-r2flow: terminal must be text or null")
        return cls(
            state_map=cast(str, data["state_map"]),
            event_label=cast(str, data["event_label"]),
            event_function=cast(str, data["event_function"]),
            event_args_sha256=cast(str, data["event_args_sha256"]),
            event_identity=cast(str, data["event_identity"]),
            state_key=cast(str, data["state_key"]),
            predecessor_key=cast(str, data["predecessor_key"]),
            rank=cast(int, data["rank"]),
            in_edges=tuple(
                (cast(str, edge["predecessor_key"]), cast(str, edge["label_hash"]))
                for edge in cast(list[dict[str, JsonValue]], edges)
            ),
            actual_in_edge=cast(int, data["actual_in_edge"]),
            legal_event_set_sha256=cast(str, data["legal_event_set_sha256"]),
            action_grammar_version=cast(str, data["action_grammar_version"]),
            action_grammar_sha256=cast(str, data["action_grammar_sha256"]),
            action_budget_tokens=cast(int, data["action_budget_tokens"]),
            action_stop_token_ids=ids("action_stop_token_ids"),
            action_finish_reason=cast(str, data["action_finish_reason"]),
            budget_forced_positions=ids("budget_forced_positions"),
            action_masks_digest=cast(str, data["action_masks_digest"]),
            reasoning_token_ids=ids("reasoning_token_ids"),
            reasoning_token_ids_include_stop=cast(bool, data["reasoning_token_ids_include_stop"]),
            reasoning_stop_token_ids=ids("reasoning_stop_token_ids"),
            reasoning_finish_reason=cast(str, data["reasoning_finish_reason"]),
            terminal=terminal,
            format=cast(str, data["format"]),
        )


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    index: int
    reasoning_text: str
    action_text: str
    action_token_ids: tuple[int, ...]
    action_token_count: int
    observation_text: str
    observation_status: str
    invoked_skill_ids: tuple[str, ...]
    forward_prefix_hash: str
    hindsight_prefix_hash: str
    r2flow: StepR2FlowRecord | None = None

    def __post_init__(self) -> None:
        location = f"trajectory step {self.index!r}"
        if self.r2flow is not None and not isinstance(self.r2flow, StepR2FlowRecord):
            raise ValueError(f"{location}: r2flow must be a StepR2FlowRecord")
        if self.r2flow is not None and self.r2flow.rank != self.index:
            raise ValueError(f"{location}: r2flow rank must equal the step index")
        if type(self.index) is not int or self.index < 1:
            raise ValueError(f"{location}: index must start at one")
        if not isinstance(self.reasoning_text, str):
            raise ValueError(f"{location}: reasoning_text must be text")
        require_non_empty_text(self.action_text, field="action_text", location=location)
        if not isinstance(self.action_token_ids, tuple) or not self.action_token_ids:
            raise ValueError(f"{location}: action_token_ids must be a non-empty tuple")
        if any(type(token_id) is not int or token_id < 0 for token_id in self.action_token_ids):
            raise ValueError(f"{location}: action_token_ids must contain non-negative integers")
        if type(self.action_token_count) is not int or self.action_token_count <= 0:
            raise ValueError(f"{location}: action_token_count must be positive")
        if self.action_token_count != len(self.action_token_ids):
            raise ValueError(f"{location}: action_token_count does not match action_token_ids")
        require_non_empty_text(
            self.observation_text,
            field="observation_text",
            location=location,
        )
        if self.observation_status not in OBSERVATION_STATUSES:
            raise ValueError(f"{location}: observation_status is unsupported")
        if not isinstance(self.invoked_skill_ids, tuple):
            raise ValueError(f"{location}: invoked_skill_ids must be a tuple")
        for skill_id in self.invoked_skill_ids:
            if not isinstance(skill_id, str):
                raise ValueError(f"{location}: invoked_skill_ids must contain strings")
            validate_identifier(skill_id)
        if len(set(self.invoked_skill_ids)) != len(self.invoked_skill_ids):
            raise ValueError(f"{location}: invoked_skill_ids must be unique")
        require_non_empty_text(
            self.forward_prefix_hash,
            field="forward_prefix_hash",
            location=location,
        )
        require_non_empty_text(
            self.hindsight_prefix_hash,
            field="hindsight_prefix_hash",
            location=location,
        )
        validate_sha256(self.forward_prefix_hash)
        validate_sha256(self.hindsight_prefix_hash)

    def to_value(self) -> dict[str, JsonValue]:
        value: dict[str, JsonValue] = {
            "action_text": self.action_text,
            "action_token_count": self.action_token_count,
            "action_token_ids": list(self.action_token_ids),
            "forward_prefix_hash": self.forward_prefix_hash,
            "hindsight_prefix_hash": self.hindsight_prefix_hash,
            "index": self.index,
            "invoked_skill_ids": list(self.invoked_skill_ids),
            "observation_status": self.observation_status,
            "observation_text": self.observation_text,
            "reasoning_text": self.reasoning_text,
        }
        if self.r2flow is not None:
            value["r2flow"] = self.r2flow.to_value()
        return value

    @classmethod
    def from_value(cls, value: object) -> TrajectoryStep:
        normalized = _require_json_object(value, label="trajectory step")
        has_r2flow = "r2flow" in normalized
        r2flow_value = normalized.pop("r2flow", None)
        if has_r2flow and not isinstance(r2flow_value, dict):
            raise ValueError("trajectory step: r2flow must be an object when present")
        _require_exact_fields(
            normalized,
            fields={
                "action_text",
                "action_token_count",
                "action_token_ids",
                "forward_prefix_hash",
                "hindsight_prefix_hash",
                "index",
                "invoked_skill_ids",
                "observation_status",
                "observation_text",
                "reasoning_text",
            },
            label="trajectory step",
        )
        raw_token_ids = normalized["action_token_ids"]
        if not isinstance(raw_token_ids, list):
            raise ValueError("trajectory step: action_token_ids must be an array")
        token_ids = tuple(
            _require_int(item, field="action_token_ids", location="trajectory step")
            for item in raw_token_ids
        )
        return cls(
            index=_require_int(normalized["index"], field="index", location="trajectory step"),
            reasoning_text=_require_string(
                normalized["reasoning_text"],
                field="reasoning_text",
                location="trajectory step",
            ),
            action_text=_require_string(
                normalized["action_text"],
                field="action_text",
                location="trajectory step",
            ),
            action_token_ids=token_ids,
            action_token_count=_require_int(
                normalized["action_token_count"],
                field="action_token_count",
                location="trajectory step",
            ),
            observation_text=_require_string(
                normalized["observation_text"],
                field="observation_text",
                location="trajectory step",
            ),
            observation_status=_require_string(
                normalized["observation_status"],
                field="observation_status",
                location="trajectory step",
            ),
            invoked_skill_ids=_require_string_tuple(
                normalized["invoked_skill_ids"],
                field="invoked_skill_ids",
                location="trajectory step",
            ),
            forward_prefix_hash=_require_string(
                normalized["forward_prefix_hash"],
                field="forward_prefix_hash",
                location="trajectory step",
            ),
            hindsight_prefix_hash=_require_string(
                normalized["hindsight_prefix_hash"],
                field="hindsight_prefix_hash",
                location="trajectory step",
            ),
            r2flow=None if r2flow_value is None else StepR2FlowRecord.from_value(r2flow_value),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class TrajectoryRecord:
    trajectory_id: str
    environment_id: str
    task_family: str
    initial_context: InitialContext
    steps: tuple[TrajectoryStep, ...]
    horizon: int
    reward: TerminalReward
    shifted_reward: float
    epsilon_min: float
    tokenizer_id: str
    decoding_snapshot_id: str
    created_at: str

    def __post_init__(self) -> None:
        location = f"trajectory {self.trajectory_id!r}"
        require_non_empty_text(self.trajectory_id, field="trajectory_id", location=location)
        validate_identifier(self.trajectory_id)
        require_non_empty_text(self.environment_id, field="environment_id", location=location)
        require_non_empty_text(self.task_family, field="task_family", location=location)
        if not isinstance(self.initial_context, InitialContext):
            raise ValueError(f"{location}: initial_context has an incompatible type")
        context_environment_id = self.initial_context.meta.get("environment_id")
        context_task_family = self.initial_context.meta.get("task_family")
        if context_environment_id != self.environment_id:
            raise ValueError(f"{location}: environment_id differs from initial context")
        if context_task_family != self.task_family:
            raise ValueError(f"{location}: task_family differs from initial context")
        if not isinstance(self.steps, tuple) or not self.steps:
            raise ValueError(f"{location}: steps must be a non-empty tuple")
        for expected_index, step in enumerate(self.steps, start=1):
            if not isinstance(step, TrajectoryStep):
                raise ValueError(f"{location}: steps must contain TrajectoryStep records")
            if step.index != expected_index:
                raise ValueError(
                    f"{location}: step index {step.index} is not contiguous at "
                    f"position {expected_index}"
                )
        validate_trajectory_skill_invocations(
            retrieved_skill_ids=self.initial_context.retrieved_skill_ids,
            active_skill_ids=self.initial_context.active_skill_ids,
            steps=self.steps,
            initial_meta=self.initial_context.meta,
        )
        if type(self.horizon) is not int or self.horizon < 1:
            raise ValueError(f"{location}: horizon must be positive")
        if self.horizon != len(self.steps):
            raise ValueError(f"{location}: horizon does not match steps")
        if not isinstance(self.reward, TerminalReward):
            raise ValueError(f"{location}: reward has an incompatible type")
        if not 0.0 <= self.reward.value <= 1.0:
            raise ValueError(f"{location}: reward.value must lie in [0, 1]")

        epsilon_min = require_finite_number(
            self.epsilon_min,
            field="epsilon_min",
            location=location,
        )
        if epsilon_min <= 0.0:
            raise ValueError(f"{location}: epsilon_min must be positive")
        shifted_reward = require_finite_number(
            self.shifted_reward,
            field="shifted_reward",
            location=location,
        )
        expected_shifted = self.reward.value + epsilon_min
        if abs(shifted_reward - expected_shifted) > FLOAT_TOLERANCE:
            raise ValueError(f"{location}: shifted_reward is inconsistent with reward")
        object.__setattr__(self, "epsilon_min", epsilon_min)
        object.__setattr__(self, "shifted_reward", shifted_reward)

        require_non_empty_text(self.tokenizer_id, field="tokenizer_id", location=location)
        require_non_empty_text(
            self.decoding_snapshot_id,
            field="decoding_snapshot_id",
            location=location,
        )
        require_iso_timestamp(self.created_at, field="created_at", location=location)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "created_at": self.created_at,
            "decoding_snapshot_id": self.decoding_snapshot_id,
            "environment_id": self.environment_id,
            "epsilon_min": self.epsilon_min,
            "horizon": self.horizon,
            "initial_context": self.initial_context.to_value(),
            "reward": self.reward.to_value(),
            "shifted_reward": self.shifted_reward,
            "steps": [step.to_value() for step in self.steps],
            "task_family": self.task_family,
            "tokenizer_id": self.tokenizer_id,
            "trajectory_id": self.trajectory_id,
        }

    @classmethod
    def from_value(cls, value: object) -> TrajectoryRecord:
        normalized = _require_json_object(value, label="trajectory")
        _require_exact_fields(
            normalized,
            fields={
                "created_at",
                "decoding_snapshot_id",
                "environment_id",
                "epsilon_min",
                "horizon",
                "initial_context",
                "reward",
                "shifted_reward",
                "steps",
                "task_family",
                "tokenizer_id",
                "trajectory_id",
            },
            label="trajectory",
        )
        raw_steps = normalized["steps"]
        if not isinstance(raw_steps, list):
            raise ValueError("trajectory: steps must be an array")
        return cls(
            trajectory_id=_require_string(
                normalized["trajectory_id"],
                field="trajectory_id",
                location="trajectory",
            ),
            environment_id=_require_string(
                normalized["environment_id"],
                field="environment_id",
                location="trajectory",
            ),
            task_family=_require_string(
                normalized["task_family"],
                field="task_family",
                location="trajectory",
            ),
            initial_context=InitialContext.from_value(normalized["initial_context"]),
            steps=tuple(TrajectoryStep.from_value(step) for step in raw_steps),
            horizon=_require_int(normalized["horizon"], field="horizon", location="trajectory"),
            reward=TerminalReward.from_value(normalized["reward"]),
            shifted_reward=_require_float(
                normalized["shifted_reward"],
                field="shifted_reward",
                location="trajectory",
            ),
            epsilon_min=_require_float(
                normalized["epsilon_min"],
                field="epsilon_min",
                location="trajectory",
            ),
            tokenizer_id=_require_string(
                normalized["tokenizer_id"],
                field="tokenizer_id",
                location="trajectory",
            ),
            decoding_snapshot_id=_require_string(
                normalized["decoding_snapshot_id"],
                field="decoding_snapshot_id",
                location="trajectory",
            ),
            created_at=_require_string(
                normalized["created_at"],
                field="created_at",
                location="trajectory",
            ),
        )

    def _content_value(self) -> dict[str, JsonValue]:
        value = self.to_value()
        del value["created_at"]
        return value

    @property
    def content_hash(self) -> str:
        return stable_hash(self._content_value())


def _tokenizer_identity(tokenizer: TokenizerProtocol, *, trajectory_id: str) -> str:
    identity = tokenizer.tokenizer_id
    require_non_empty_text(
        identity,
        field="tokenizer.tokenizer_id",
        location=f"trajectory {trajectory_id!r}",
    )
    return identity


@dataclass(frozen=True, slots=True)
class _AdmittedActionSpan:
    token_ids: tuple[int, ...]
    token_count: int


def _admit_action_span(
    tokenizer: TokenizerProtocol,
    action_text: str,
    recorded_token_ids: tuple[int, ...],
    recorded_token_count: int,
    *,
    trajectory_id: str,
    step_index: int,
) -> _AdmittedActionSpan:
    location = f"trajectory {trajectory_id!r}, step {step_index}"
    if not isinstance(recorded_token_ids, tuple) or not recorded_token_ids:
        raise ValueError(f"{location}: recorded action token ids must be non-empty")
    if any(type(token_id) is not int or token_id < 0 for token_id in recorded_token_ids):
        raise ValueError(f"{location}: recorded action token ids are invalid")
    if recorded_token_count != len(recorded_token_ids):
        raise ValueError(f"{location}: action_token_count differs from admitted token length")
    decoded = tokenizer.decode(recorded_token_ids)
    if not isinstance(decoded, str):
        raise ValueError(f"{location}: tokenizer.decode must return text")
    if decoded != action_text:
        raise ValueError(f"{location}: recorded action text differs from decoded token span")
    return _AdmittedActionSpan(
        token_ids=recorded_token_ids,
        token_count=len(recorded_token_ids),
    )


def build_trajectory_record(
    *,
    tokenizer: TokenizerProtocol,
    trajectory_id: str,
    environment_id: str,
    task_family: str,
    initial_context: InitialContext,
    steps: tuple[TrajectoryStep, ...],
    horizon: int,
    reward: TerminalReward,
    shifted_reward: float,
    epsilon_min: float,
    tokenizer_id: str,
    decoding_snapshot_id: str,
    created_at: str,
) -> TrajectoryRecord:
    actual_tokenizer_id = _tokenizer_identity(tokenizer, trajectory_id=trajectory_id)
    if actual_tokenizer_id != tokenizer_id:
        raise ValueError(
            f"trajectory {trajectory_id!r}: tokenizer_id {tokenizer_id!r} "
            f"does not match injected tokenizer {actual_tokenizer_id!r}"
        )

    for step in steps:
        _admit_action_span(
            tokenizer,
            step.action_text,
            step.action_token_ids,
            step.action_token_count,
            trajectory_id=trajectory_id,
            step_index=step.index,
        )

    return TrajectoryRecord(
        trajectory_id=trajectory_id,
        environment_id=environment_id,
        task_family=task_family,
        initial_context=initial_context,
        steps=steps,
        horizon=horizon,
        reward=reward,
        shifted_reward=shifted_reward,
        epsilon_min=epsilon_min,
        tokenizer_id=tokenizer_id,
        decoding_snapshot_id=decoding_snapshot_id,
        created_at=created_at,
    )
