from __future__ import annotations

from dataclasses import dataclass

from .canonical import JsonValue, normalize_json, stable_hash


@dataclass(frozen=True, slots=True)
class PosteriorBatchUpdate:
    batch_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.batch_id, str) or not self.batch_id.strip():
            raise ValueError("batch_id must be non-empty text")

    def to_value(self) -> dict[str, JsonValue]:
        return {"batch_id": self.batch_id, "updates": []}

    @classmethod
    def from_value(cls, value: object) -> PosteriorBatchUpdate:
        normalized = normalize_json(value)
        if (
            not isinstance(normalized, dict)
            or set(normalized) != {"batch_id", "updates"}
            or normalized["updates"] != []
        ):
            raise ValueError("posterior batch update has an incompatible field set")
        batch_id = normalized["batch_id"]
        if not isinstance(batch_id, str):
            raise ValueError("batch_id must be non-empty text")
        return cls(batch_id=batch_id)

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())
