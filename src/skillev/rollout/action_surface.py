from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, cast

from skillev.contracts import JsonValue, normalize_json
from skillev.contracts.answer_writer import COMPLETION_WRITERS

ACTION_SURFACE_FORMAT = "skillev-action-surface@4"
SKILL_INVOCATION = "invoke-skill@1"
ROLLOUT_BUDGET_PROFILE_FORMAT = "skillev-rollout-budget-profile@1"


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _object(value: object, *, fields: set[str], label: str) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or set(normalized) != fields:
        raise ValueError(f"{label} has an incompatible field set")
    return normalized


def _object_value(value: object, *, label: str) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict):
        raise ValueError(f"{label} must be a JSON object")
    return normalized


class TerminalMode(StrEnum):
    EXPLICIT_COMPLETION = "explicit-completion"
    ENVIRONMENT = "environment-terminal"


class ArgumentType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"


@dataclass(frozen=True, slots=True)
class ArgumentFieldSpec:
    value_type: ArgumentType
    required: bool
    nullable: bool = False
    default: JsonValue = None
    minimum: int | None = None
    maximum: int | None = None
    choices: tuple[str, ...] | None = None
    allow_blank: bool = False

    def __post_init__(self) -> None:
        if type(self.required) is not bool or type(self.nullable) is not bool:
            raise TypeError("argument required/nullable flags must be boolean")
        if type(self.allow_blank) is not bool:
            raise TypeError("argument allow_blank flag must be boolean")
        if (self.choices is not None or self.allow_blank) and (
            self.value_type is not ArgumentType.STRING
        ):
            raise ValueError("only string fields may declare choices or allow blanks")
        if self.choices is not None and (
            not isinstance(self.choices, tuple)
            or not self.choices
            or any(type(item) is not str or not item for item in self.choices)
            or len(set(self.choices)) != len(self.choices)
        ):
            raise ValueError("argument choices must be a non-empty tuple of unique text")
        if self.required and self.default is not None:
            raise ValueError("required field cannot have an implicit default")
        if self.value_type is not ArgumentType.INTEGER and (
            self.minimum is not None or self.maximum is not None
        ):
            raise ValueError("only integer fields may have numeric bounds")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("argument bounds are inverted")

    def to_value(self) -> dict[str, JsonValue]:
        value: dict[str, JsonValue] = {
            "default": self.default,
            "maximum": self.maximum,
            "minimum": self.minimum,
            "nullable": self.nullable,
            "required": self.required,
            "value_type": self.value_type.value,
        }
        if self.choices is not None:
            value["choices"] = list(self.choices)
        if self.allow_blank:
            value["allow_blank"] = True
        return value

    @classmethod
    def from_value(cls, value: object) -> ArgumentFieldSpec:
        base = {"default", "maximum", "minimum", "nullable", "required", "value_type"}
        extras = set(value) - base if isinstance(value, dict) else set()
        if not extras <= {"choices", "allow_blank"}:
            raise ValueError("argument field spec has an incompatible field set")
        data = _object(value, fields=base | extras, label="argument field spec")
        raw_choices = data.get("choices")
        if "choices" in data and not isinstance(raw_choices, list):
            raise TypeError("argument choices must be an array")
        if "allow_blank" in data and data["allow_blank"] is not True:
            raise ValueError("allow_blank is serialised only when true")
        if type(data["required"]) is not bool or type(data["nullable"]) is not bool:
            raise TypeError("argument required/nullable flags must be boolean")
        for bound in ("minimum", "maximum"):
            if data[bound] is not None and type(data[bound]) is not int:
                raise TypeError("argument bounds must be integers or null")
        return cls(
            value_type=ArgumentType(_text(data["value_type"], field="argument value type")),
            required=data["required"],
            nullable=data["nullable"],
            default=data["default"],
            minimum=cast(int | None, data["minimum"]),
            maximum=cast(int | None, data["maximum"]),
            choices=(
                None
                if raw_choices is None
                else tuple(cast(str, item) for item in cast(list[JsonValue], raw_choices))
            ),
            allow_blank="allow_blank" in data,
        )


@dataclass(frozen=True, slots=True)
class ToolActionSpec:
    resource_id: str
    name: str
    arguments: dict[str, ArgumentFieldSpec]
    example_arguments: dict[str, JsonValue]

    def __post_init__(self) -> None:
        _text(self.resource_id, field="tool resource_id")
        _text(self.name, field="tool name")
        if set(self.arguments) != set(self.example_arguments):
            raise ValueError("tool example keys must match the v2 argument schema")
        if any(
            not key.strip() or not isinstance(value, ArgumentFieldSpec)
            for key, value in self.arguments.items()
        ):
            raise ValueError("v2 argument schema is invalid")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "arguments": {name: value.to_value() for name, value in self.arguments.items()},
            "example_arguments": self.example_arguments,
            "name": self.name,
            "resource_id": self.resource_id,
        }

    @classmethod
    def from_value(cls, value: object) -> ToolActionSpec:
        data = _object(
            value,
            fields={"arguments", "example_arguments", "name", "resource_id"},
            label="v2 tool action spec",
        )
        arguments = _object_value(data["arguments"], label="v2 arguments")
        return cls(
            resource_id=_text(data["resource_id"], field="tool resource_id"),
            name=_text(data["name"], field="tool name"),
            arguments={name: ArgumentFieldSpec.from_value(raw) for name, raw in arguments.items()},
            example_arguments=_object_value(data["example_arguments"], label="example arguments"),
        )


@dataclass(frozen=True, slots=True)
class ActionAdmission:
    admitted: bool
    normalized_action: Any | None
    error_code: str | None


def _matches_type(value: JsonValue, expected: ArgumentType) -> bool:
    if expected is ArgumentType.STRING:
        return type(value) is str and bool(value.strip())
    if expected is ArgumentType.INTEGER:
        return type(value) is int
    return type(value) is bool


def admit_arguments(action: Any, spec: ToolActionSpec) -> ActionAdmission:
    if not isinstance(action.arguments, dict):
        return ActionAdmission(False, None, "invalid_arguments")
    unknown = set(action.arguments) - set(spec.arguments)
    if unknown:
        return ActionAdmission(False, None, "unknown_arguments")
    normalized: dict[str, JsonValue] = {}
    for name, field in spec.arguments.items():
        if name not in action.arguments:
            if field.required:
                return ActionAdmission(False, None, "missing_required_argument")
            normalized[name] = field.default
            continue
        value = action.arguments[name]
        if value is None and field.nullable:
            normalized[name] = None
            continue
        if field.allow_blank and type(value) is str:
            pass
        elif not _matches_type(value, field.value_type):
            return ActionAdmission(False, None, "argument_type_mismatch")
        if field.choices is not None and value not in field.choices:
            return ActionAdmission(False, None, "argument_not_in_choices")
        if field.value_type is ArgumentType.INTEGER:
            integer = int(value)
            if field.minimum is not None and integer < field.minimum:
                return ActionAdmission(False, None, "argument_out_of_range")
            if field.maximum is not None and integer > field.maximum:
                return ActionAdmission(False, None, "argument_out_of_range")
        normalized[name] = value
    return ActionAdmission(True, replace(action, arguments=normalize_json(normalized)), None)


@dataclass(frozen=True, slots=True)
class CompletionSpec:
    value_schema: dict[str, JsonValue]
    example_value: dict[str, JsonValue]

    def __post_init__(self) -> None:
        value_schema = normalize_json(self.value_schema)
        example_value = normalize_json(self.example_value)
        if not isinstance(value_schema, dict) or not isinstance(example_value, dict):
            raise ValueError("completion schema and example must be JSON objects")
        object.__setattr__(self, "value_schema", value_schema)
        object.__setattr__(self, "example_value", example_value)
        if set(self.value_schema) != set(self.example_value):
            raise ValueError("completion example keys must match the value schema")

    def to_value(self) -> dict[str, JsonValue]:
        return {"example_value": self.example_value, "value_schema": self.value_schema}

    @classmethod
    def from_value(cls, value: object) -> CompletionSpec:
        data = _object(
            value,
            fields={"example_value", "value_schema"},
            label="completion spec",
        )
        return cls(
            value_schema=_object_value(data["value_schema"], label="completion value schema"),
            example_value=_object_value(data["example_value"], label="completion example"),
        )


@dataclass(frozen=True, slots=True)
class ActionSurface:
    terminal_mode: TerminalMode
    tools: tuple[ToolActionSpec, ...] = ()
    completion: CompletionSpec | None = None
    dynamic_choice_fields: tuple[str, ...] = ()
    instructions: tuple[str, ...] = ()
    completion_writer: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.terminal_mode, TerminalMode):
            raise TypeError("action surface terminal_mode is invalid")
        if not isinstance(self.tools, tuple) or any(
            not isinstance(item, ToolActionSpec) for item in self.tools
        ):
            raise TypeError("action surface tools are invalid")
        identities = tuple((item.resource_id, item.name) for item in self.tools)
        if len(set(identities)) != len(identities):
            raise ValueError("action surface tools must be unique")
        if self.completion is not None and not isinstance(self.completion, CompletionSpec):
            raise TypeError("action surface completion is invalid")
        if self.terminal_mode is TerminalMode.EXPLICIT_COMPLETION and self.completion is None:
            raise ValueError("explicit-completion surfaces require a completion spec")
        if self.terminal_mode is TerminalMode.ENVIRONMENT and self.completion is not None:
            raise ValueError("environment-terminal surfaces cannot expose completion")
        for field, values in (
            ("dynamic_choice_fields", self.dynamic_choice_fields),
            ("instructions", self.instructions),
        ):
            if not isinstance(values, tuple) or any(
                type(item) is not str or not item.strip() for item in values
            ):
                raise ValueError(f"{field} must be a tuple of non-empty text")
        if len(set(self.dynamic_choice_fields)) != len(self.dynamic_choice_fields):
            raise ValueError("dynamic choice fields must be unique")
        if self.completion_writer is not None and (
            self.completion_writer not in COMPLETION_WRITERS
            or self.terminal_mode is not TerminalMode.EXPLICIT_COMPLETION
            or self.completion is None
            or self.completion.value_schema != {}
            or self.completion.example_value != {}
        ):
            raise ValueError(
                "completion writer executor-answer@1 requires an explicit completion "
                "without parameters"
            )

    def to_value(self) -> dict[str, JsonValue]:
        value: dict[str, JsonValue] = {
            "completion": None if self.completion is None else self.completion.to_value(),
            "dynamic_choice_fields": list(self.dynamic_choice_fields),
            "format": ACTION_SURFACE_FORMAT,
            "instructions": {"semantic": list(self.instructions), "json_wire": []},
            "skill_invocation": SKILL_INVOCATION,
            "terminal_mode": self.terminal_mode.value,
            "tools": [item.to_value() for item in self.tools],
        }
        if self.completion_writer is not None:
            value["completion_writer"] = self.completion_writer
        return value

    @classmethod
    def from_value(cls, value: object) -> ActionSurface:
        written = isinstance(value, dict) and "completion_writer" in value
        data = _object(
            value,
            fields={
                "completion",
                "dynamic_choice_fields",
                "format",
                "instructions",
                "skill_invocation",
                "terminal_mode",
                "tools",
            }
            | ({"completion_writer"} if written else set()),
            label="action surface",
        )
        if data["format"] != ACTION_SURFACE_FORMAT or data["skill_invocation"] != SKILL_INVOCATION:
            raise ValueError("unsupported action surface format")
        tools = data["tools"]
        choices = data["dynamic_choice_fields"]
        instructions = _object(
            data["instructions"], fields={"semantic", "json_wire"}, label="public instructions"
        )
        semantic = instructions["semantic"]
        if (
            not isinstance(tools, list)
            or not isinstance(choices, list)
            or not isinstance(semantic, list)
            or instructions["json_wire"] != []
        ):
            raise TypeError("action surface arrays are invalid")
        raw_completion = data["completion"]
        return cls(
            terminal_mode=TerminalMode(_text(data["terminal_mode"], field="terminal_mode")),
            tools=tuple(ToolActionSpec.from_value(item) for item in tools),
            completion=(
                None if raw_completion is None else CompletionSpec.from_value(raw_completion)
            ),
            dynamic_choice_fields=tuple(
                _text(item, field="dynamic choice field") for item in choices
            ),
            instructions=tuple(_text(item, field="semantic instruction") for item in semantic),
            completion_writer=(
                _text(data["completion_writer"], field="completion writer") if written else None
            ),
        )


@dataclass(frozen=True, slots=True)
class RolloutBudgetProfile:
    profile_id: str
    max_turns: int
    max_reasoning_tokens: int
    max_action_tokens: int
    format: str = ROLLOUT_BUDGET_PROFILE_FORMAT

    def __post_init__(self) -> None:
        _text(self.profile_id, field="budget profile_id")
        for field in ("max_turns", "max_reasoning_tokens", "max_action_tokens"):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field} must be positive")
        if self.format != ROLLOUT_BUDGET_PROFILE_FORMAT:
            raise ValueError("unsupported rollout budget profile format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "max_action_tokens": self.max_action_tokens,
            "max_reasoning_tokens": self.max_reasoning_tokens,
            "max_turns": self.max_turns,
            "profile_id": self.profile_id,
        }

    @classmethod
    def from_value(cls, value: object) -> RolloutBudgetProfile:
        data = _object(
            value,
            fields={
                "format",
                "max_action_tokens",
                "max_reasoning_tokens",
                "max_turns",
                "profile_id",
            },
            label="rollout budget profile",
        )
        for field in ("max_turns", "max_reasoning_tokens", "max_action_tokens"):
            if type(data[field]) is not int:
                raise TypeError(f"{field} must be an integer")
        return cls(
            profile_id=_text(data["profile_id"], field="budget profile_id"),
            max_turns=data["max_turns"],
            max_reasoning_tokens=data["max_reasoning_tokens"],
            max_action_tokens=data["max_action_tokens"],
            format=_text(data["format"], field="rollout budget profile format"),
        )


@dataclass(frozen=True, slots=True)
class ModelVisibleMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError("model-visible message role is unsupported")
        _text(self.content, field="model-visible message content")

    def to_value(self) -> dict[str, JsonValue]:
        return {"content": self.content, "role": self.role}

    @classmethod
    def from_value(cls, value: object) -> ModelVisibleMessage:
        data = _object(value, fields={"content", "role"}, label="model-visible message")
        return cls(
            role=_text(data["role"], field="model-visible message role"),
            content=_text(data["content"], field="model-visible message content"),
        )


__all__ = [
    "ACTION_SURFACE_FORMAT",
    "ROLLOUT_BUDGET_PROFILE_FORMAT",
    "ActionAdmission",
    "ActionSurface",
    "ArgumentFieldSpec",
    "ArgumentType",
    "CompletionSpec",
    "ModelVisibleMessage",
    "RolloutBudgetProfile",
    "TerminalMode",
    "ToolActionSpec",
    "admit_arguments",
]
