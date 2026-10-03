from __future__ import annotations

from dataclasses import dataclass, fields
from enum import StrEnum

from skillev.contracts.canonical import JsonValue, normalize_json
from skillev.contracts.identity import validate_identifier, validate_sha256


def _require_json_object(value: object, *, label: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or normalized != value:
        raise TypeError(f"{label} must be a JSON object")
    return normalized


def _require_exact_fields(
    value: object,
    *,
    label: str,
    expected: set[str],
) -> dict[str, JsonValue]:
    normalized = _require_json_object(value, label=label)
    if set(normalized) != expected:
        raise ValueError(f"{label} has an incompatible field set")
    return normalized


def _require_wire_text(value: object, *, field: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field} must be text")
    return value


def _require_optional_wire_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _require_wire_text(value, field=field)


@dataclass(frozen=True, slots=True)
class BudgetVector:
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0
    agent_turns: int = 0
    tool_calls: int = 0
    wall_time_milliseconds: int = 0

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if type(value) is not int or value < 0:
                raise ValueError("Budget counters must be non-negative integers")

    def fits_within(self, cap: BudgetVector) -> bool:
        return all(getattr(self, item.name) <= getattr(cap, item.name) for item in fields(self))

    def add(self, other: BudgetVector) -> BudgetVector:
        return BudgetVector(
            **{
                item.name: getattr(self, item.name) + getattr(other, item.name)
                for item in fields(self)
            }
        )

    def subtract(self, other: BudgetVector) -> BudgetVector:
        if not other.fits_within(self):
            raise ValueError("Budget subtraction would produce a negative counter")
        return BudgetVector(
            **{
                item.name: getattr(self, item.name) - getattr(other, item.name)
                for item in fields(self)
            }
        )

    def scale(self, multiplier: int) -> BudgetVector:
        if type(multiplier) is not int or multiplier < 0:
            raise ValueError("Budget multiplier must be a non-negative integer")
        return BudgetVector(
            **{item.name: getattr(self, item.name) * multiplier for item in fields(self)}
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_value(cls, value: object) -> BudgetVector:
        expected = {item.name for item in fields(cls)}
        normalized = _require_exact_fields(
            value,
            label="Budget vector",
            expected=expected,
        )
        counters: dict[str, int] = {}
        for name in expected:
            raw = normalized[name]
            if type(raw) is not int:
                raise TypeError("Budget vector counters must be integers")
            counters[name] = raw
        return cls(**counters)


@dataclass(frozen=True, slots=True)
class BudgetReservation:
    reservation_id: str
    run_id: str
    attempt_id: str
    invocation_id: str
    maximum: BudgetVector

    def __post_init__(self) -> None:
        if not all((self.reservation_id, self.run_id, self.attempt_id, self.invocation_id)):
            raise ValueError("Budget reservation identity fields cannot be empty")


@dataclass(frozen=True, slots=True)
class BudgetSettlement:
    reservation_id: str
    actual: BudgetVector

    def __post_init__(self) -> None:
        if type(self.reservation_id) is not str or not self.reservation_id.strip():
            raise ValueError("reservation_id must be non-empty text")
        if not isinstance(self.actual, BudgetVector):
            raise TypeError("actual must be BudgetVector")


class ActionKind(StrEnum):
    TOOL = "tool"
    SKILL = "skill"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class StructuredAction:
    kind: ActionKind
    name: str
    arguments: JsonValue
    resource_id: str | None = None
    skill_id: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Structured action name cannot be empty")
        normalized = normalize_json(self.arguments)
        if normalized != self.arguments:
            raise ValueError("Structured action arguments must be normalized JSON")
        if self.kind in {ActionKind.TOOL, ActionKind.SKILL} and not self.resource_id:
            raise ValueError("Executable actions require a resource ID")
        if self.kind is ActionKind.SKILL and not self.skill_id:
            raise ValueError("Skill actions require a skill ID")
        if self.kind is ActionKind.SKILL and self.skill_id is not None:
            validate_identifier(self.skill_id)
        if self.kind is not ActionKind.SKILL and self.skill_id is not None:
            raise ValueError("Only skill actions may carry a skill ID")
        if self.kind is ActionKind.COMPLETE and self.resource_id is not None:
            raise ValueError("Completion is not dispatched to a resource")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "arguments": self.arguments,
            "kind": self.kind.value,
            "name": self.name,
            "resource_id": self.resource_id,
            "skill_id": self.skill_id,
        }

    @classmethod
    def from_value(cls, value: object) -> StructuredAction:
        normalized = _require_exact_fields(
            value,
            label="Structured action",
            expected={"arguments", "kind", "name", "resource_id", "skill_id"},
        )
        raw_kind = _require_wire_text(normalized["kind"], field="kind")
        raw_arguments = normalized["arguments"]
        if normalize_json(raw_arguments) != raw_arguments:
            raise TypeError("arguments must be normalized JSON")
        return cls(
            kind=ActionKind(raw_kind),
            name=_require_wire_text(normalized["name"], field="name"),
            arguments=raw_arguments,
            resource_id=_require_optional_wire_text(
                normalized["resource_id"],
                field="resource_id",
            ),
            skill_id=_require_optional_wire_text(
                normalized["skill_id"],
                field="skill_id",
            ),
        )


@dataclass(frozen=True, slots=True)
class SkillManifest:
    skill_id: str
    version: str
    content_hash: str
    input_schema_id: str
    output_schema_id: str
    license_id: str
    provenance_hash: str

    def __post_init__(self) -> None:
        if not all(
            (
                self.skill_id,
                self.version,
                self.input_schema_id,
                self.output_schema_id,
                self.license_id,
            )
        ):
            raise ValueError("Skill manifest fields cannot be empty")
        validate_identifier(self.skill_id)
        validate_sha256(self.content_hash)
        validate_sha256(self.provenance_hash)

    def to_value(self) -> dict[str, str]:
        return {
            "content_hash": self.content_hash,
            "input_schema_id": self.input_schema_id,
            "license_id": self.license_id,
            "output_schema_id": self.output_schema_id,
            "provenance_hash": self.provenance_hash,
            "skill_id": self.skill_id,
            "version": self.version,
        }

    @classmethod
    def from_value(cls, value: object) -> SkillManifest:
        normalized = _require_exact_fields(
            value,
            label="Skill manifest",
            expected={
                "content_hash",
                "input_schema_id",
                "license_id",
                "output_schema_id",
                "provenance_hash",
                "skill_id",
                "version",
            },
        )
        return cls(
            skill_id=_require_wire_text(normalized["skill_id"], field="skill_id"),
            version=_require_wire_text(normalized["version"], field="version"),
            content_hash=_require_wire_text(normalized["content_hash"], field="content_hash"),
            input_schema_id=_require_wire_text(
                normalized["input_schema_id"], field="input_schema_id"
            ),
            output_schema_id=_require_wire_text(
                normalized["output_schema_id"], field="output_schema_id"
            ),
            license_id=_require_wire_text(normalized["license_id"], field="license_id"),
            provenance_hash=_require_wire_text(
                normalized["provenance_hash"], field="provenance_hash"
            ),
        )
