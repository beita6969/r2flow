from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from .canonical import stable_hash

IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*$")
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class EntityKind(StrEnum):
    PROJECT = "project"
    EXPERIMENT = "experiment"
    LINEAGE = "lineage"
    SKILL = "skill"
    SKILL_VERSION = "skill-version"
    SKILL_LIBRARY = "skill-library"
    TASK = "task"
    TRAJECTORY = "trajectory"
    ROLLOUT = "rollout"
    TRAINING_RUN = "training-run"
    RUN = "run"
    ATTEMPT = "attempt"
    EVALUATION = "evaluation"
    INVOCATION = "invocation"
    EVENT = "event"
    ARTIFACT = "artifact"


def validate_identifier(value: str) -> str:
    if not IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"Invalid stable identifier: {value!r}")
    return value


def validate_sha256(value: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"Invalid SHA-256 identifier: {value!r}")
    return value


def artifact_hash(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Timestamps must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class Seed:
    value: int

    def __post_init__(self) -> None:
        if not 0 <= self.value < 2**64:
            raise ValueError("Seed must be an unsigned 64-bit integer")


def deterministic_id(kind: str | EntityKind, fields: object) -> str:
    kind_value = kind.value if isinstance(kind, EntityKind) else kind
    validate_identifier(kind_value)
    digest = stable_hash(fields).removeprefix("sha256:")
    return f"{kind_value}_{digest[:32]}"


def attempt_id(*, run_id: str, attempt_no: int, lease_epoch: int) -> str:
    if attempt_no < 1 or lease_epoch < 1:
        raise ValueError("Attempt number and lease epoch start at one")
    return deterministic_id(
        EntityKind.ATTEMPT,
        {"attempt_no": attempt_no, "lease_epoch": lease_epoch, "run_id": run_id},
    )


def event_id(*, producer_id: str, producer_seq: int, payload_hash: str) -> str:
    if producer_seq < 1:
        raise ValueError("Producer sequence starts at one")
    validate_sha256(payload_hash)
    return deterministic_id(
        EntityKind.EVENT,
        {
            "payload_hash": payload_hash,
            "producer_id": producer_id,
            "producer_seq": producer_seq,
        },
    )
