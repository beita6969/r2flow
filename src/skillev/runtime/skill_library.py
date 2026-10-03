from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from skillev.contracts import JsonValue, stable_hash

from .skills import SkillDocument


def skill_library_version(
    *,
    documents: Mapping[str, SkillDocument],
    active_skill_ids: tuple[str, ...],
) -> str:
    return stable_hash([documents[skill_id].to_value() for skill_id in active_skill_ids])


EMPTY_LIBRARY_VERSION = stable_hash([])


def require_seed_library(seed_documents: tuple[SkillDocument, ...]) -> None:
    if not isinstance(seed_documents, tuple) or any(
        not isinstance(document, SkillDocument) for document in seed_documents
    ):
        raise TypeError("seed library must contain SkillDocument values")
    if not seed_documents:
        raise ValueError("the full method requires a non-empty seed library")
    skill_ids = tuple(document.manifest.skill_id for document in seed_documents)
    if len(set(skill_ids)) != len(skill_ids):
        raise ValueError("seed library repeats a skill ID")


@dataclass(frozen=True, slots=True)
class SkillLibraryState:
    documents: Mapping[str, SkillDocument]
    active_skill_ids: tuple[str, ...]
    current_version: str

    def __post_init__(self) -> None:
        copied = dict(self.documents)
        for key, document in copied.items():
            if key != document.manifest.skill_id:
                raise ValueError("skill-library document key differs from skill ID")
        if tuple(sorted(set(self.active_skill_ids))) != self.active_skill_ids:
            raise ValueError("active_skill_ids must be sorted and unique")
        if not set(self.active_skill_ids) <= set(copied):
            raise ValueError("active_skill_ids reference unknown documents")
        expected = skill_library_version(
            documents=copied,
            active_skill_ids=self.active_skill_ids,
        )
        if self.current_version != expected:
            raise ValueError("current_version differs from active content")
        object.__setattr__(self, "documents", MappingProxyType(copied))

    @classmethod
    def from_seed_documents(
        cls,
        documents: tuple[SkillDocument, ...],
    ) -> SkillLibraryState:
        by_id = {document.manifest.skill_id: document for document in documents}
        if len(by_id) != len(documents):
            raise ValueError("seed documents repeat a skill ID")
        active = tuple(sorted(by_id))
        return cls(
            documents=by_id,
            active_skill_ids=active,
            current_version=skill_library_version(
                documents=by_id,
                active_skill_ids=active,
            ),
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "active_skill_ids": list(self.active_skill_ids),
            "current_version": self.current_version,
            "documents": [
                self.documents[skill_id].to_value() for skill_id in sorted(self.documents)
            ],
        }

    @property
    def state_hash(self) -> str:
        return stable_hash(self.to_value())

    @classmethod
    def from_value(cls, value: object) -> SkillLibraryState:
        if not isinstance(value, dict) or set(value) != {
            "active_skill_ids",
            "current_version",
            "documents",
        }:
            raise ValueError("SkillLibraryState has incompatible fields")
        active = value["active_skill_ids"]
        raw_documents = value["documents"]
        current_version = value["current_version"]
        if not isinstance(active, list) or any(not isinstance(item, str) for item in active):
            raise ValueError("active_skill_ids must be an array of text")
        if not isinstance(raw_documents, list):
            raise ValueError("documents must be an array")
        if not isinstance(current_version, str):
            raise ValueError("current_version must be text")
        documents = tuple(SkillDocument.from_value(item) for item in raw_documents)
        by_id = {document.manifest.skill_id: document for document in documents}
        if len(by_id) != len(documents):
            raise ValueError("documents repeat a skill ID")
        return cls(
            documents=by_id,
            active_skill_ids=tuple(active),
            current_version=current_version,
        )


class SkillLibrary:
    def __init__(self, state: SkillLibraryState) -> None:
        self._state = state

    @property
    def state(self) -> SkillLibraryState:
        return self._state

    @property
    def current_version(self) -> str:
        return self._state.current_version

    @property
    def active_skill_ids(self) -> tuple[str, ...]:
        return self._state.active_skill_ids

    def document(self, skill_id: str) -> SkillDocument:
        return self._state.documents[skill_id]

    def active_documents(self) -> tuple[SkillDocument, ...]:
        return tuple(self._state.documents[skill_id] for skill_id in self._state.active_skill_ids)

    def all_documents(self) -> tuple[SkillDocument, ...]:
        return tuple(self._state.documents[skill_id] for skill_id in sorted(self._state.documents))

    def checkpoint_value(self) -> dict[str, JsonValue]:
        return self._state.to_value()

    def apply(self, state: SkillLibraryState) -> None:
        self._state = state


__all__ = [
    "EMPTY_LIBRARY_VERSION",
    "SkillLibrary",
    "SkillLibraryState",
    "require_seed_library",
    "skill_library_version",
]
