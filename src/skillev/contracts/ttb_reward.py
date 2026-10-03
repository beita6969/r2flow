from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from .canonical import JsonValue, normalize_json, stable_hash
from .ttb_common import (
    FLOAT_TOLERANCE,
    require_canonical_mapping,
    require_finite_number,
    require_non_empty_text,
)


class SuccessRule(StrEnum):
    R_EQUALS_ONE = "r-equals-one"
    R_AT_THRESHOLD = "r-at-threshold"
    TRUSTED_NATIVE_PROJECTION = "trusted-native-projection"


def _read_text(value: JsonValue, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"terminal reward {field} must be text")
    return value


@dataclass(frozen=True, slots=True)
class TerminalReward:
    value: float
    success: bool
    success_rule: SuccessRule
    success_threshold: float | None
    native_metric_name: str
    native_payload: Mapping[str, JsonValue]
    environment_id: str
    verifier_version: str

    def __post_init__(self) -> None:
        location = f"terminal reward for environment {self.environment_id!r}"
        value = require_finite_number(self.value, field="value", location=location)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{location}: value must lie in [0, 1]")
        object.__setattr__(self, "value", value)

        if type(self.success) is not bool:
            raise ValueError(f"{location}: success must be a boolean")
        if not isinstance(self.success_rule, SuccessRule):
            raise ValueError(f"{location}: success_rule must be a SuccessRule")

        threshold: float | None
        if self.success_rule is SuccessRule.R_EQUALS_ONE:
            if self.success_threshold is not None:
                raise ValueError(f"{location}: r-equals-one does not accept success_threshold")
            threshold = None
            expected_success = abs(value - 1.0) <= FLOAT_TOLERANCE
        elif self.success_rule is SuccessRule.R_AT_THRESHOLD:
            if self.success_threshold is None:
                raise ValueError(f"{location}: success_threshold is required")
            threshold = require_finite_number(
                self.success_threshold,
                field="success_threshold",
                location=location,
            )
            if not 0.0 < threshold <= 1.0:
                raise ValueError(f"{location}: success_threshold must lie in (0, 1]")
            expected_success = value >= threshold
        elif self.success_rule is SuccessRule.TRUSTED_NATIVE_PROJECTION:
            if self.success_threshold is not None:
                raise ValueError(
                    f"{location}: trusted-native-projection does not accept success_threshold"
                )
            threshold = None
            expected_success = self.success
        else:
            raise ValueError(f"{location}: unsupported success rule")

        if self.success is not expected_success:
            raise ValueError(f"{location}: success is inconsistent with success_rule")

        object.__setattr__(self, "success_threshold", threshold)
        require_non_empty_text(
            self.native_metric_name,
            field="native_metric_name",
            location=location,
        )
        require_non_empty_text(self.environment_id, field="environment_id", location=location)
        require_non_empty_text(
            self.verifier_version,
            field="verifier_version",
            location=location,
        )
        normalized_payload = require_canonical_mapping(
            self.native_payload,
            field="native_payload",
            location=location,
        )
        object.__setattr__(self, "native_payload", normalized_payload)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "environment_id": self.environment_id,
            "native_metric_name": self.native_metric_name,
            "native_payload": normalize_json(self.native_payload),
            "success": self.success,
            "success_rule": self.success_rule.value,
            "success_threshold": self.success_threshold,
            "value": self.value,
            "verifier_version": self.verifier_version,
        }

    @classmethod
    def from_value(cls, value: object) -> TerminalReward:
        normalized = normalize_json(value)
        if not isinstance(normalized, dict):
            raise ValueError("terminal reward must be a JSON object")
        expected_fields = {
            "environment_id",
            "native_metric_name",
            "native_payload",
            "success",
            "success_rule",
            "success_threshold",
            "value",
            "verifier_version",
        }
        if set(normalized) != expected_fields:
            raise ValueError("terminal reward has incompatible fields")

        payload = normalized["native_payload"]
        if not isinstance(payload, dict):
            raise ValueError("terminal reward native_payload must be a JSON object")
        raw_success = normalized["success"]
        if type(raw_success) is not bool:
            raise ValueError("terminal reward success must be a boolean")
        raw_threshold = normalized["success_threshold"]
        threshold = (
            None
            if raw_threshold is None
            else require_finite_number(raw_threshold, field="success_threshold")
        )
        reward_value = require_finite_number(normalized["value"], field="value")
        raw_rule = _read_text(normalized["success_rule"], field="success_rule")
        try:
            success_rule = SuccessRule(raw_rule)
        except ValueError as error:
            raise ValueError("terminal reward success_rule is unsupported") from error

        return cls(
            value=reward_value,
            success=raw_success,
            success_rule=success_rule,
            success_threshold=threshold,
            native_metric_name=_read_text(
                normalized["native_metric_name"],
                field="native_metric_name",
            ),
            native_payload=payload,
            environment_id=_read_text(normalized["environment_id"], field="environment_id"),
            verifier_version=_read_text(
                normalized["verifier_version"],
                field="verifier_version",
            ),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())
