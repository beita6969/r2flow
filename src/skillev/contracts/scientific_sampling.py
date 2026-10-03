from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .canonical import JsonValue, normalize_json, stable_hash

SCIENTIFIC_SAMPLING_ALGORITHM: Final = "skillev-scientific-sampling@1"
_UINT64_LIMIT: Final = 2**64


def scientific_sampling_schedule_hash(*, base_seed: int) -> str:
    if type(base_seed) is not int or not 0 <= base_seed < _UINT64_LIMIT:
        raise ValueError("base_seed must be an unsigned 64-bit integer")
    return stable_hash(
        {
            "algorithm": SCIENTIFIC_SAMPLING_ALGORITHM,
            "base_seed": base_seed,
        }
    )


@dataclass(frozen=True, slots=True)
class ScientificSamplingCoordinate:
    sampling_schedule_hash: str
    schedule_purpose: str
    ordered_sequence_hash: str
    sequence_position: int
    task_id: str
    optimizer_step_or_anchor_ordinal: int
    format: str = "skillev-scientific-sampling-coordinate@1"

    def __post_init__(self) -> None:
        for field in (
            "sampling_schedule_hash",
            "ordered_sequence_hash",
            "schedule_purpose",
            "task_id",
        ):
            value = getattr(self, field)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{field} must be non-empty text")
        if type(self.sequence_position) is not int or self.sequence_position < 0:
            raise ValueError("sequence_position must be non-negative")
        if (
            type(self.optimizer_step_or_anchor_ordinal) is not int
            or self.optimizer_step_or_anchor_ordinal < 0
        ):
            raise ValueError("optimizer_step_or_anchor_ordinal must be non-negative")
        if self.format != "skillev-scientific-sampling-coordinate@1":
            raise ValueError("unsupported scientific sampling coordinate format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "optimizer_step_or_anchor_ordinal": self.optimizer_step_or_anchor_ordinal,
            "ordered_sequence_hash": self.ordered_sequence_hash,
            "sampling_schedule_hash": self.sampling_schedule_hash,
            "schedule_purpose": self.schedule_purpose,
            "sequence_position": self.sequence_position,
            "task_id": self.task_id,
        }

    @classmethod
    def from_value(cls, value: object) -> ScientificSamplingCoordinate:
        normalized = normalize_json(value)
        fields = {
            "format",
            "optimizer_step_or_anchor_ordinal",
            "ordered_sequence_hash",
            "sampling_schedule_hash",
            "schedule_purpose",
            "sequence_position",
            "task_id",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("scientific sampling coordinate has incompatible fields")
        if any(
            type(normalized[field]) is not str
            for field in (
                "format",
                "ordered_sequence_hash",
                "sampling_schedule_hash",
                "schedule_purpose",
                "task_id",
            )
        ):
            raise TypeError("scientific sampling coordinate text fields must be text")
        if any(
            type(normalized[field]) is not int
            for field in ("optimizer_step_or_anchor_ordinal", "sequence_position")
        ):
            raise TypeError("scientific sampling coordinate positions must be integers")
        return cls(
            sampling_schedule_hash=normalized["sampling_schedule_hash"],
            schedule_purpose=normalized["schedule_purpose"],
            ordered_sequence_hash=normalized["ordered_sequence_hash"],
            sequence_position=normalized["sequence_position"],
            task_id=normalized["task_id"],
            optimizer_step_or_anchor_ordinal=normalized["optimizer_step_or_anchor_ordinal"],
            format=normalized["format"],
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


__all__ = [
    "SCIENTIFIC_SAMPLING_ALGORITHM",
    "ScientificSamplingCoordinate",
    "scientific_sampling_schedule_hash",
]
