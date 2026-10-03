from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime
from typing import Final, Protocol

from .canonical import JsonValue, canonical_json, normalize_json

FLOAT_TOLERANCE: Final[float] = 1e-9


class TokenizerProtocol(Protocol):
    @property
    def tokenizer_id(self) -> str: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: tuple[int, ...]) -> str: ...


def require_non_empty_text(
    value: object,
    *,
    field: str,
    location: str = "record",
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: {field} must be non-empty text")
    return value


def require_finite_number(
    value: object,
    *,
    field: str,
    location: str = "record",
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{location}: {field} must be a finite number")
    try:
        normalized = float(value)
    except OverflowError as error:
        raise ValueError(f"{location}: {field} must be a finite number") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{location}: {field} must be a finite number")
    return normalized


def require_iso_timestamp(
    value: object,
    *,
    field: str,
    location: str = "record",
) -> str:
    text = require_non_empty_text(value, field=field, location=location)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{location}: {field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{location}: {field} must include a UTC offset")
    return text


def require_canonical_mapping(
    value: object,
    *,
    field: str,
    location: str = "record",
) -> dict[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location}: {field} must be a JSON mapping")
    try:
        canonical_json(value)
        normalized = normalize_json(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{location}: {field} must be canonical-JSON serializable") from error
    if not isinstance(normalized, dict):
        raise ValueError(f"{location}: {field} must be a JSON mapping")
    return normalized
