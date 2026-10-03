from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import cast

from skillev.contracts.canonical import JsonValue, normalize_json, stable_hash
from skillev.contracts.identity import validate_identifier, validate_sha256

from .contracts import SkillManifest
from .skill_md import TRANSFERABLE_FAMILIES, SkillMd, parse_skill_md, render_skill_md

SKILL_DOCUMENT_FORMAT = "skillev-skill-document@3"
SKILL_MD_CONTEXTS = ("*",)


def _exact_object(
    value: object,
    *,
    label: str,
    expected: set[str],
) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or normalized != value:
        raise TypeError(f"{label} must be a normalized JSON object")
    if set(normalized) != expected:
        raise ValueError(f"{label} has an incompatible field set")
    return normalized


def _wire_text(value: object, *, field: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field} must be text")
    return value


def _text_tuple(
    values: tuple[str, ...],
    *,
    field: str,
    allow_empty: bool = True,
) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{field} must be a tuple")
    if not allow_empty and not values:
        raise ValueError(f"{field} cannot be empty")
    if any(type(value) is not str or not value.strip() for value in values):
        raise ValueError(f"{field} must contain non-empty text")
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"{field} must be sorted and unique")


@dataclass(frozen=True, slots=True)
class SkillApplicability:
    task_families: tuple[str, ...]
    contexts: tuple[str, ...]
    required_tools: tuple[str, ...]
    excluded_contexts: tuple[str, ...]

    def __post_init__(self) -> None:
        _text_tuple(self.task_families, field="task_families")
        _text_tuple(self.contexts, field="contexts")
        _text_tuple(self.required_tools, field="required_tools")
        _text_tuple(self.excluded_contexts, field="excluded_contexts")
        if not self.task_families and not self.contexts:
            raise ValueError("Skill applicability requires task_families or contexts")
        for field, values in (
            ("task_families", self.task_families),
            ("contexts", self.contexts),
        ):
            if "*" in values and values != ("*",):
                raise ValueError(f"{field} wildcard must be its only value")
        if set(self.contexts) & set(self.excluded_contexts):
            raise ValueError("included and excluded contexts cannot overlap")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "contexts": list(self.contexts),
            "excluded_contexts": list(self.excluded_contexts),
            "required_tools": list(self.required_tools),
            "task_families": list(self.task_families),
        }

    @classmethod
    def from_value(cls, value: object) -> SkillApplicability:
        normalized = _exact_object(
            value,
            label="Skill applicability",
            expected={
                "contexts",
                "excluded_contexts",
                "required_tools",
                "task_families",
            },
        )
        return cls(
            task_families=_wire_text_tuple(normalized["task_families"], field="task_families"),
            contexts=_wire_text_tuple(normalized["contexts"], field="contexts"),
            required_tools=_wire_text_tuple(normalized["required_tools"], field="required_tools"),
            excluded_contexts=_wire_text_tuple(
                normalized["excluded_contexts"], field="excluded_contexts"
            ),
        )


def _wire_text_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a JSON array")
    return tuple(_wire_text(item, field=field) for item in value)


@dataclass(frozen=True, slots=True)
class SkillDocument:
    manifest: SkillManifest
    title: str
    summary: str
    instructions: str
    applicability: SkillApplicability

    def __post_init__(self) -> None:
        if not isinstance(self.applicability, SkillApplicability):
            raise TypeError("Skill applicability must be SkillApplicability")
        if self.title != self.manifest.skill_id:
            raise ValueError("SKILL.md document title must equal its skill ID")
        families = self.applicability.task_families
        if (
            self.applicability
            != SkillApplicability(
                task_families=families,
                contexts=SKILL_MD_CONTEXTS,
                required_tools=(),
                excluded_contexts=(),
            )
            or not set(families) <= TRANSFERABLE_FAMILIES
        ):
            raise ValueError("SKILL.md applicability must be transferable families only")
        self.skill_md()
        if self.manifest.content_hash != stable_hash(self.content_value()):
            raise ValueError("Skill document does not match its immutable manifest")

    def skill_md(self) -> SkillMd:
        return SkillMd(
            name=self.manifest.skill_id,
            description=self.summary,
            version=self.manifest.version,
            families=self.applicability.task_families,
            body=self.instructions,
        )

    @classmethod
    def from_skill_md(cls, manifest: SkillManifest, skill: SkillMd) -> SkillDocument:
        if skill.name != manifest.skill_id or skill.version != manifest.version:
            raise ValueError("SKILL.md name/version must match the manifest")
        return cls(
            manifest=manifest,
            title=skill.name,
            summary=skill.description,
            instructions=skill.body,
            applicability=SkillApplicability(
                task_families=skill.families,
                contexts=SKILL_MD_CONTEXTS,
                required_tools=(),
                excluded_contexts=(),
            ),
        )

    def content_value(self) -> dict[str, JsonValue]:
        return {"skill_md": render_skill_md(self.skill_md())}

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": SKILL_DOCUMENT_FORMAT,
            "manifest": cast(dict[str, JsonValue], self.manifest.to_value()),
            "skill_md": render_skill_md(self.skill_md()),
        }

    @classmethod
    def from_value(cls, value: object) -> SkillDocument:
        normalized = _exact_object(
            value, label="Skill document", expected={"format", "manifest", "skill_md"}
        )
        if normalized["format"] != SKILL_DOCUMENT_FORMAT:
            raise ValueError("Skill document has an incompatible format")
        raw_manifest = normalized["manifest"]
        if not isinstance(raw_manifest, dict):
            raise TypeError("Skill manifest must be a JSON object")
        return cls.from_skill_md(
            SkillManifest.from_value(raw_manifest),
            parse_skill_md(_wire_text(normalized["skill_md"], field="skill_md")),
        )


def model_visible_skill_content(document: SkillDocument) -> str:
    if not isinstance(document, SkillDocument):
        raise TypeError("model-visible skill content requires a SkillDocument")
    return render_skill_md(document.skill_md())


@dataclass(frozen=True, slots=True)
class SkillMetadata:
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
            raise ValueError("Skill metadata fields cannot be empty")
        validate_identifier(self.skill_id)
        validate_sha256(self.content_hash)
        validate_sha256(self.provenance_hash)

    @classmethod
    def from_document(cls, document: SkillDocument) -> SkillMetadata:
        manifest = document.manifest
        return cls(
            skill_id=manifest.skill_id,
            version=manifest.version,
            content_hash=manifest.content_hash,
            input_schema_id=manifest.input_schema_id,
            output_schema_id=manifest.output_schema_id,
            license_id=manifest.license_id,
            provenance_hash=manifest.provenance_hash,
        )

    def to_value(self) -> dict[str, JsonValue]:
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
    def from_value(cls, value: object) -> SkillMetadata:
        normalized = _exact_object(
            value,
            label="Skill metadata",
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
            skill_id=_wire_text(normalized["skill_id"], field="skill_id"),
            version=_wire_text(normalized["version"], field="version"),
            content_hash=_wire_text(normalized["content_hash"], field="content_hash"),
            input_schema_id=_wire_text(
                normalized["input_schema_id"],
                field="input_schema_id",
            ),
            output_schema_id=_wire_text(
                normalized["output_schema_id"],
                field="output_schema_id",
            ),
            license_id=_wire_text(normalized["license_id"], field="license_id"),
            provenance_hash=_wire_text(
                normalized["provenance_hash"],
                field="provenance_hash",
            ),
        )


class RetrievalInclusionReason(str, Enum):
    APPLICABILITY_MATCH = "applicability-match"


class SkillCatalog:
    def __init__(self, documents: tuple[SkillDocument, ...]) -> None:
        by_id: dict[str, SkillDocument] = {}
        for document in documents:
            skill_id = document.manifest.skill_id
            if skill_id in by_id:
                raise ValueError("Skill IDs must be unique within a catalog")
            by_id[skill_id] = document
        self._documents: Mapping[str, SkillDocument] = MappingProxyType(by_id)

    @property
    def skill_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._documents))

    def document(self, skill_id: str) -> SkillDocument:
        return self._documents[skill_id]

    def metadata(self, skill_id: str) -> SkillMetadata:
        return SkillMetadata.from_document(self.document(skill_id))


__all__ = [
    "SKILL_DOCUMENT_FORMAT",
    "RetrievalInclusionReason",
    "SkillApplicability",
    "SkillCatalog",
    "SkillDocument",
    "SkillMetadata",
    "model_visible_skill_content",
]
