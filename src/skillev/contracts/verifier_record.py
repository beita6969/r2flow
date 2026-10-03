from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from .canonical import JsonValue, canonical_json, normalize_json, stable_hash
from .failure_buckets import FailureMode, HorizonBucket, TokenBucket

VERIFIER_RECORD_FORMAT: Final = "r2flow-verifier-record@1"
EVENT_LABEL_FORMAT: Final = "r2flow-event-label@1"
VERIFIER_Z_FORMAT: Final = "r2flow-z@1"
VERIFIER_COMBINATION_RULE: Final = "verifier-conjunction@1"
NO_LAGGED_FAILURE: Final = "none"

CALL_SCHEMA: Final = "call-schema@1"
EXECUTION_STATUS: Final = "execution-status@1"
MBPP_PUBLIC_ASSERTS: Final = "mbpp-public-asserts@1"
ALFWORLD_ACT: Final = "alfworld-act-admissibility-execution@1"
QA_GROUNDING: Final = "qa-answer-grounding@1"
ALFWORLD_SKILL_ADVICE: Final = "alfworld-skill-advice-execution@2"
REFERENCE_AGREEMENT: Final = "reference-model-agreement@1"
REFERENCE_AGREEMENT_SUBMIT: Final = "reference-model-agreement=submit-answer@1"
QA_GROUNDING_SUBMIT: Final = "qa-answer-grounding=submit-context@1"
GATE_ELIGIBILITY_RULE: Final = "gate-eligible-soft-only-needs-evidential@1"

REGISTERED_VERIFIERS: Final[Mapping[str, tuple[float, bool]]] = {
    CALL_SCHEMA: (1.0, False),
    EXECUTION_STATUS: (1.0, False),
    MBPP_PUBLIC_ASSERTS: (1.0, True),
    ALFWORLD_ACT: (1.0, True),
    QA_GROUNDING: (0.5, True),
    ALFWORLD_SKILL_ADVICE: (0.5, True),
    REFERENCE_AGREEMENT: (0.5, True),
    REFERENCE_AGREEMENT_SUBMIT: (0.5, True),
    QA_GROUNDING_SUBMIT: (0.5, True),
}
SOFT_ONLY_DOMAINS: Final = frozenset({"aime-2026", "healthbench"})

SKILL_FUNCTION: Final = "invoke_skill"
SUBMIT_FUNCTION: Final = "submit_answer"
UNPARSED_FUNCTION: Final = "unparsed"


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be non-empty text")
    return value


def _exact(value: object, *, label: str, fields: frozenset[str]) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or set(normalized) != fields:
        raise ValueError(f"{label} has an incompatible field set")
    return normalized


def content_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


class EventClass(StrEnum):
    SKILL = "skill"
    TOOL = "tool"
    SUBMIT = "submit"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class EventLabel:
    event_class: EventClass
    function: str
    unit: str
    arguments_hash: str
    format: str = EVENT_LABEL_FORMAT

    def __post_init__(self) -> None:
        if not isinstance(self.event_class, EventClass):
            raise ValueError("event_class must be an EventClass")
        _text(self.function, field="event function")
        _text(self.unit, field="event unit")
        _text(self.arguments_hash, field="event arguments_hash")
        if self.format != EVENT_LABEL_FORMAT:
            raise ValueError("unsupported event label format")

    @classmethod
    def from_call(cls, function: str, arguments: Mapping[str, str]) -> EventLabel:
        arguments_hash = stable_hash(dict(arguments))
        if function == SKILL_FUNCTION:
            return cls(
                EventClass.SKILL,
                function,
                _text(arguments.get("skill_id"), field="skill_id"),
                arguments_hash,
            )
        if function == SUBMIT_FUNCTION:
            return cls(EventClass.SUBMIT, function, function, arguments_hash)
        return cls(EventClass.TOOL, function, function, arguments_hash)

    @classmethod
    def unparsed(cls, action_text: str) -> EventLabel:
        return cls(
            EventClass.INVALID, UNPARSED_FUNCTION, UNPARSED_FUNCTION, content_hash(action_text)
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "arguments_hash": self.arguments_hash,
            "event_class": self.event_class.value,
            "format": self.format,
            "function": self.function,
            "unit": self.unit,
        }

    @classmethod
    def from_value(cls, value: object) -> EventLabel:
        raw = _exact(
            value,
            label="event label",
            fields=frozenset({"arguments_hash", "event_class", "format", "function", "unit"}),
        )
        return cls(
            EventClass(str(raw["event_class"])),
            str(raw["function"]),
            str(raw["unit"]),
            str(raw["arguments_hash"]),
            str(raw["format"]),
        )


@dataclass(frozen=True, slots=True)
class VerifierContext:
    context_class: str
    lagged_failure_mode: str
    token_bucket: TokenBucket
    turn_bucket: HorizonBucket
    format: str = VERIFIER_Z_FORMAT

    def __post_init__(self) -> None:
        _text(self.context_class, field="context_class")
        if self.lagged_failure_mode != NO_LAGGED_FAILURE:
            FailureMode(self.lagged_failure_mode)
        TokenBucket(self.token_bucket)
        HorizonBucket(self.turn_bucket)
        if self.format != VERIFIER_Z_FORMAT:
            raise ValueError("unsupported z format")

    def cell_key(self) -> str:
        return "|".join(
            (
                self.context_class,
                self.lagged_failure_mode,
                str(self.token_bucket),
                str(self.turn_bucket),
            )
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "context_class": self.context_class,
            "format": self.format,
            "lagged_failure_mode": self.lagged_failure_mode,
            "token_bucket": str(self.token_bucket),
            "turn_bucket": str(self.turn_bucket),
        }

    @classmethod
    def from_value(cls, value: object) -> VerifierContext:
        raw = _exact(
            value,
            label="z",
            fields=frozenset(
                {"context_class", "format", "lagged_failure_mode", "token_bucket", "turn_bucket"}
            ),
        )
        return cls(
            str(raw["context_class"]),
            str(raw["lagged_failure_mode"]),
            TokenBucket(str(raw["token_bucket"])),
            HorizonBucket(str(raw["turn_bucket"])),
            str(raw["format"]),
        )


def pre_invocation_context(
    *,
    context_class: str,
    step_index: int,
    previous_status: str | None,
    forward_prefix_token_count: int,
) -> VerifierContext:
    if type(step_index) is not int or step_index < 1:
        raise ValueError("step_index must be a positive integer")
    if (previous_status is None) != (step_index == 1):
        raise ValueError("exactly the first event has no lagged status")
    lagged = (
        NO_LAGGED_FAILURE
        if previous_status is None
        else FailureMode.from_observation_status(previous_status).value
    )
    return VerifierContext(
        context_class=context_class,
        lagged_failure_mode=lagged,
        token_bucket=TokenBucket.from_count(forward_prefix_token_count),
        turn_bucket=HorizonBucket.from_horizon(step_index),
    )


@dataclass(frozen=True, slots=True)
class VerifierComponent:
    verifier_id: str
    passed: bool
    confidence: float
    evidential: bool
    reason_code: str
    checked_content_hash: str

    def __post_init__(self) -> None:
        if self.verifier_id not in REGISTERED_VERIFIERS:
            raise ValueError(f"unregistered verifier {self.verifier_id!r}")
        if type(self.passed) is not bool or type(self.evidential) is not bool:
            raise ValueError("passed and evidential must be booleans")
        confidence, evidential = REGISTERED_VERIFIERS[self.verifier_id]
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, int | float)
            or not math.isfinite(self.confidence)
            or float(self.confidence) != confidence
            or self.evidential is not evidential
        ):
            raise ValueError("component confidence/evidential differ from the declared verifier")
        _text(self.reason_code, field="reason_code")
        _text(self.checked_content_hash, field="checked_content_hash")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "checked_content_hash": self.checked_content_hash,
            "confidence": float(self.confidence),
            "evidential": self.evidential,
            "passed": self.passed,
            "reason_code": self.reason_code,
            "verifier_id": self.verifier_id,
        }

    @classmethod
    def from_value(cls, value: object) -> VerifierComponent:
        raw = _exact(
            value,
            label="verifier component",
            fields=frozenset(
                {
                    "checked_content_hash",
                    "confidence",
                    "evidential",
                    "passed",
                    "reason_code",
                    "verifier_id",
                }
            ),
        )
        confidence = raw["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, int | float):
            raise ValueError("confidence must be a number")
        return cls(
            str(raw["verifier_id"]),
            raw["passed"] is True,
            float(confidence),
            raw["evidential"] is True,
            str(raw["reason_code"]),
            str(raw["checked_content_hash"]),
        )


def component(verifier_id: str, passed: bool, reason_code: str, checked: str) -> VerifierComponent:
    confidence, evidential = REGISTERED_VERIFIERS[verifier_id]
    return VerifierComponent(
        verifier_id, passed, confidence, evidential, reason_code, content_hash(checked)
    )


def combine_verifier_components(
    components: tuple[VerifierComponent, ...],
) -> tuple[bool | None, float | None]:
    failed = [item.confidence for item in components if not item.passed]
    if failed:
        return False, float(max(failed))
    evidential = [item.confidence for item in components if item.evidential]
    if evidential:
        return True, float(min(evidential))
    return None, None


def gate_eligible(
    domain: str,
    event: EventLabel,
    combined_passed: bool | None,
    components: tuple[VerifierComponent, ...] = (),
) -> bool:
    return (
        combined_passed is not None
        and event.event_class is EventClass.SKILL
        and (domain not in SOFT_ONLY_DOMAINS or any(item.evidential for item in components))
    )


_RECORD_FIELDS = frozenset(
    {
        "combination_rule",
        "combined_confidence",
        "combined_passed",
        "components",
        "domain",
        "event",
        "event_id",
        "format",
        "gate_eligible",
        "not_applicable",
        "post_invocation_status",
        "step_index",
        "task_id",
        "trajectory_id",
        "z",
    }
)


def event_id(trajectory_id: str, step_index: int) -> str:
    return stable_hash([trajectory_id, step_index])


@dataclass(frozen=True, slots=True)
class VerifierRecord:
    trajectory_id: str
    step_index: int
    task_id: str
    domain: str
    event: EventLabel
    z: VerifierContext
    components: tuple[VerifierComponent, ...]
    not_applicable: tuple[tuple[str, str], ...]
    post_invocation_status: str
    combination_rule: str = VERIFIER_COMBINATION_RULE
    format: str = VERIFIER_RECORD_FORMAT
    event_id: str = field(init=False)
    combined_passed: bool | None = field(init=False)
    combined_confidence: float | None = field(init=False)
    gate_eligible: bool = field(init=False)

    def __post_init__(self) -> None:
        _text(self.trajectory_id, field="trajectory_id")
        _text(self.task_id, field="task_id")
        _text(self.domain, field="domain")
        if type(self.step_index) is not int or self.step_index < 1:
            raise ValueError("step_index must be a positive integer")
        if not isinstance(self.event, EventLabel) or not isinstance(self.z, VerifierContext):
            raise ValueError("record needs an EventLabel and a VerifierContext")
        if (
            type(self.components) is not tuple
            or not self.components
            or any(not isinstance(item, VerifierComponent) for item in self.components)
        ):
            raise ValueError("record needs a non-empty tuple of components")
        ids = [item.verifier_id for item in self.components]
        if ids != sorted(set(ids)):
            raise ValueError("components must be unique and sorted by verifier_id")
        if type(self.not_applicable) is not tuple or any(
            type(item) is not tuple
            or len(item) != 2
            or item[0] not in REGISTERED_VERIFIERS
            or item[0] in ids
            or type(item[1]) is not str
            or not item[1]
            for item in self.not_applicable
        ):
            raise ValueError("not_applicable must be (registered verifier id, reason) pairs")
        FailureMode.from_observation_status(self.post_invocation_status)
        if self.format != VERIFIER_RECORD_FORMAT:
            raise ValueError("unsupported verifier record format")
        if self.combination_rule != VERIFIER_COMBINATION_RULE:
            raise ValueError("unsupported verifier combination rule")
        combined_passed, combined_confidence = combine_verifier_components(self.components)
        object.__setattr__(self, "event_id", event_id(self.trajectory_id, self.step_index))
        object.__setattr__(self, "combined_passed", combined_passed)
        object.__setattr__(self, "combined_confidence", combined_confidence)
        object.__setattr__(
            self,
            "gate_eligible",
            gate_eligible(self.domain, self.event, combined_passed, self.components),
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "combination_rule": self.combination_rule,
            "combined_confidence": self.combined_confidence,
            "combined_passed": self.combined_passed,
            "components": [item.to_value() for item in self.components],
            "domain": self.domain,
            "event": self.event.to_value(),
            "event_id": self.event_id,
            "format": self.format,
            "gate_eligible": self.gate_eligible,
            "not_applicable": [list(item) for item in self.not_applicable],
            "post_invocation_status": self.post_invocation_status,
            "step_index": self.step_index,
            "task_id": self.task_id,
            "trajectory_id": self.trajectory_id,
            "z": self.z.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> VerifierRecord:
        raw = _exact(value, label="verifier record", fields=_RECORD_FIELDS)
        components = raw["components"]
        skipped = raw["not_applicable"]
        confidence = raw["combined_confidence"]
        if not isinstance(components, list) or not isinstance(skipped, list):
            raise ValueError("components and not_applicable must be lists")
        if confidence is not None and (
            isinstance(confidence, bool) or not isinstance(confidence, int | float)
        ):
            raise ValueError("combined_confidence must be a number or null")
        passed = raw["combined_passed"]
        if passed is not None and type(passed) is not bool:
            raise ValueError("combined_passed must be a boolean or null")
        step_index = raw["step_index"]
        if type(step_index) is not int or type(raw["gate_eligible"]) is not bool:
            raise ValueError("step_index/gate_eligible have incompatible types")
        pairs: list[tuple[str, str]] = []
        for item in skipped:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("not_applicable entries must be pairs")
            pairs.append((str(item[0]), str(item[1])))
        record = cls(
            trajectory_id=str(raw["trajectory_id"]),
            step_index=step_index,
            task_id=str(raw["task_id"]),
            domain=str(raw["domain"]),
            event=EventLabel.from_value(raw["event"]),
            z=VerifierContext.from_value(raw["z"]),
            components=tuple(VerifierComponent.from_value(item) for item in components),
            not_applicable=tuple(pairs),
            post_invocation_status=str(raw["post_invocation_status"]),
            combination_rule=str(raw["combination_rule"]),
            format=str(raw["format"]),
        )
        if (
            record.event_id,
            record.combined_passed,
            record.combined_confidence,
            record.gate_eligible,
        ) != (raw["event_id"], passed, confidence, raw["gate_eligible"]):
            raise ValueError("stored derived fields differ from verifier-conjunction@1")
        return record

    def canonical_text(self) -> str:
        return canonical_json(self.to_value())


__all__ = [
    "ALFWORLD_ACT",
    "ALFWORLD_SKILL_ADVICE",
    "CALL_SCHEMA",
    "EVENT_LABEL_FORMAT",
    "EXECUTION_STATUS",
    "GATE_ELIGIBILITY_RULE",
    "MBPP_PUBLIC_ASSERTS",
    "NO_LAGGED_FAILURE",
    "QA_GROUNDING",
    "QA_GROUNDING_SUBMIT",
    "REFERENCE_AGREEMENT",
    "REFERENCE_AGREEMENT_SUBMIT",
    "REGISTERED_VERIFIERS",
    "SOFT_ONLY_DOMAINS",
    "VERIFIER_COMBINATION_RULE",
    "VERIFIER_RECORD_FORMAT",
    "VERIFIER_Z_FORMAT",
    "EventClass",
    "EventLabel",
    "VerifierComponent",
    "VerifierContext",
    "VerifierRecord",
    "combine_verifier_components",
    "component",
    "content_hash",
    "event_id",
    "gate_eligible",
    "pre_invocation_context",
]
