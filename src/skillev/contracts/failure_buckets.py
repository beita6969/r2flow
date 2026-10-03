from __future__ import annotations

from enum import StrEnum


def _require_integer(value: object, *, field: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer greater than or equal to {minimum}")
    return value


class FailureMode(StrEnum):
    SUCCESS = "success"
    TOOL_ERROR = "tool_error"
    SCHEMA_INVALID = "schema_invalid"
    TIMEOUT = "timeout"
    PARSE_ERROR = "parse_error"
    OTHER = "other"

    @classmethod
    def from_observation_status(cls, status: str) -> FailureMode:
        if type(status) is not str:
            raise ValueError("observation status must be text")
        try:
            return cls(status)
        except ValueError as error:
            raise ValueError(
                f"unknown observation_status {status!r}; producers must emit a closed status"
            ) from error


class TokenBucket(StrEnum):
    LE_1K = "le_1k"
    K1_TO_4K = "k1_to_4k"
    GT_4K = "gt_4k"

    @classmethod
    def from_count(cls, n: int) -> TokenBucket:
        count = _require_integer(n, field="token count", minimum=0)
        if count <= 1_000:
            return cls.LE_1K
        if count <= 4_000:
            return cls.K1_TO_4K
        return cls.GT_4K


class HorizonBucket(StrEnum):
    LE_3 = "le_3"
    H4_TO_8 = "h4_to_8"
    GT_8 = "gt_8"

    @classmethod
    def from_horizon(cls, t: int) -> HorizonBucket:
        horizon = _require_integer(t, field="trajectory horizon", minimum=1)
        if horizon <= 3:
            return cls.LE_3
        if horizon <= 8:
            return cls.H4_TO_8
        return cls.GT_8
