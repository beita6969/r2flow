from __future__ import annotations

from dataclasses import dataclass

from skillev.contracts import JsonValue, normalize_json

from .skills import RetrievalInclusionReason, SkillMetadata


def _exact_object(value: object) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or set(normalized) != {
        "content",
        "inclusion_reason",
        "metadata",
    }:
        raise ValueError("FullRetrievedSkillContext has an incompatible field set")
    return normalized


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


@dataclass(frozen=True, slots=True)
class FullRetrievedSkillContext:
    metadata: SkillMetadata
    content: str
    inclusion_reason: RetrievalInclusionReason

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, SkillMetadata):
            raise TypeError("metadata must be SkillMetadata")
        _text(self.content, field="content")
        if not isinstance(self.inclusion_reason, RetrievalInclusionReason):
            raise TypeError("inclusion_reason must be RetrievalInclusionReason")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "content": self.content,
            "inclusion_reason": self.inclusion_reason.value,
            "metadata": self.metadata.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> FullRetrievedSkillContext:
        data = _exact_object(value)
        return cls(
            metadata=SkillMetadata.from_value(data["metadata"]),
            content=_text(data["content"], field="content"),
            inclusion_reason=RetrievalInclusionReason(
                _text(data["inclusion_reason"], field="inclusion_reason")
            ),
        )


__all__ = ["FullRetrievedSkillContext"]
