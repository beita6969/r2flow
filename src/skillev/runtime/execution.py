from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from skillev.contracts.canonical import JsonValue, canonical_json, normalize_json
from skillev.contracts.identity import validate_identifier
from skillev.contracts.ttb_trajectory import OBSERVATION_STATUSES

from .contracts import BudgetVector, StructuredAction


def _require_object(value: object, *, label: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or normalized != value:
        raise TypeError(f"{label} must be a JSON object")
    return normalized


def _require_fields(
    value: object,
    *,
    label: str,
    expected: set[str],
) -> dict[str, JsonValue]:
    normalized = _require_object(value, label=label)
    if set(normalized) != expected:
        raise ValueError(f"{label} has an incompatible field set")
    return normalized


def _wire_text(value: object, *, field: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field} must be text")
    return value


def _optional_wire_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _wire_text(value, field=field)


class ActionParseStatus(StrEnum):
    VALID = "valid"
    PARSE_ERROR = "parse-error"
    SCHEMA_INVALID = "schema-invalid"


@dataclass(frozen=True, slots=True)
class ActionParseResult:
    status: ActionParseStatus
    action: StructuredAction | None
    public_error_code: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ActionParseStatus):
            raise TypeError("Action parse status has an incompatible type")
        if self.status is ActionParseStatus.VALID:
            if self.action is None or self.public_error_code is not None:
                raise ValueError("A valid parse result requires only a structured action")
            return
        if self.action is not None or not self.public_error_code:
            raise ValueError("An invalid parse result requires only a public error code")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "action": None if self.action is None else self.action.to_value(),
            "public_error_code": self.public_error_code,
            "status": self.status.value,
        }

    @classmethod
    def from_value(cls, value: object) -> ActionParseResult:
        normalized = _require_fields(
            value,
            label="Action parse result",
            expected={"action", "public_error_code", "status"},
        )
        action = normalized["action"]
        raw_status = _wire_text(normalized["status"], field="status")
        return cls(
            status=ActionParseStatus(raw_status),
            action=None if action is None else StructuredAction.from_value(action),
            public_error_code=_optional_wire_text(
                normalized["public_error_code"],
                field="public_error_code",
            ),
        )


@dataclass(frozen=True, slots=True)
class EnvironmentObservation:
    public_value: JsonValue
    observation_status: str
    invoked_skill_ids: tuple[str, ...] = ()
    terminal_submission: JsonValue = None
    terminal: bool = False
    budget_usage: BudgetVector = field(default_factory=BudgetVector)

    def __post_init__(self) -> None:
        if normalize_json(self.public_value) != self.public_value:
            raise ValueError("Environment public value must be normalized JSON")
        if self.observation_status not in OBSERVATION_STATUSES:
            raise ValueError("Environment observation status is unsupported")
        if not isinstance(self.invoked_skill_ids, tuple):
            raise ValueError("Invoked skill IDs must be a tuple")
        if len(set(self.invoked_skill_ids)) != len(self.invoked_skill_ids):
            raise ValueError("Invoked skill IDs must be unique")
        for skill_id in self.invoked_skill_ids:
            if not isinstance(skill_id, str):
                raise ValueError("Invoked skill IDs must contain strings")
            validate_identifier(skill_id)
        if type(self.terminal) is not bool:
            raise ValueError("Environment terminal status must be boolean")
        if not self.terminal and self.terminal_submission is not None:
            raise ValueError("A non-terminal observation cannot carry a submission")
        if normalize_json(self.terminal_submission) != self.terminal_submission:
            raise ValueError("Terminal submission must be normalized JSON")
        if not isinstance(self.budget_usage, BudgetVector):
            raise TypeError("Environment budget usage must be a BudgetVector")

    @property
    def observation_text(self) -> str:
        return canonical_json(self.public_value)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "budget_usage": normalize_json(self.budget_usage.to_value()),
            "invoked_skill_ids": list(self.invoked_skill_ids),
            "observation_status": self.observation_status,
            "public_value": self.public_value,
            "terminal": self.terminal,
            "terminal_submission": self.terminal_submission,
        }

    @classmethod
    def from_value(cls, value: object) -> EnvironmentObservation:
        normalized = _require_fields(
            value,
            label="Environment observation",
            expected={
                "budget_usage",
                "invoked_skill_ids",
                "observation_status",
                "public_value",
                "terminal",
                "terminal_submission",
            },
        )
        invoked_skill_ids = normalized["invoked_skill_ids"]
        if type(invoked_skill_ids) is not list:
            raise TypeError("Environment invoked skill IDs must be an array")
        skill_ids = tuple(
            _wire_text(skill_id, field="invoked_skill_ids item") for skill_id in invoked_skill_ids
        )
        terminal = normalized["terminal"]
        if type(terminal) is not bool:
            raise TypeError("Environment terminal status must be boolean")
        public_value = normalized["public_value"]
        terminal_submission = normalized["terminal_submission"]
        if normalize_json(public_value) != public_value:
            raise TypeError("Environment public value must be normalized JSON")
        if normalize_json(terminal_submission) != terminal_submission:
            raise TypeError("Environment terminal submission must be normalized JSON")
        return cls(
            public_value=public_value,
            observation_status=_wire_text(
                normalized["observation_status"],
                field="observation_status",
            ),
            invoked_skill_ids=skill_ids,
            terminal_submission=terminal_submission,
            terminal=terminal,
            budget_usage=BudgetVector.from_value(normalized["budget_usage"]),
        )


class RolloutEnvironmentSession(Protocol):
    @property
    def environment_id(self) -> str: ...

    @property
    def task_family(self) -> str: ...

    async def execute(
        self,
        action: StructuredAction,
        *,
        step_index: int,
    ) -> EnvironmentObservation: ...

    def validate_completion(self, submission: JsonValue) -> bool: ...


__all__ = [
    "ActionParseResult",
    "ActionParseStatus",
    "EnvironmentObservation",
    "RolloutEnvironmentSession",
]
