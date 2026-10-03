from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from skillev.contracts import JsonValue, canonical_json, normalize_json, stable_hash
from skillev.r2flow_evolution.types import LibraryVersion, PhaseState, SkillSpec
from skillev.runtime.skill_library import SkillLibraryState, skill_library_version
from skillev.runtime.skill_md import SkillMd, normalise_body
from skillev.runtime.skills import SkillDocument

LIBRARY_VERSION_FORMAT: Final = "r2flow-library-version@1"
PHASE_RECORD_FORMAT: Final = "r2flow-phase-state-record@1"
TRANSITION_RECORD_FORMAT: Final = "r2flow-phase-transition-commit@1"
LIBRARY_DIRECTORY: Final = "library"
EVOLVED_SKILL_PROVENANCE: Final = "r2flow-evolved-skill-md@1"


def skill_spec_to_value(spec: SkillSpec) -> dict[str, JsonValue]:
    return {
        "body": spec.body,
        "description": spec.description,
        "families": list(spec.families),
        "name": spec.name,
        "parent_id": spec.parent_id,
        "skill_id": spec.skill_id,
        "version": spec.version,
    }


def skill_spec_from_value(value: object) -> SkillSpec:
    if not isinstance(value, dict) or set(value) != {
        "body",
        "description",
        "families",
        "name",
        "parent_id",
        "skill_id",
        "version",
    }:
        raise ValueError("skill spec has an incompatible field set")
    families = value["families"]
    if not isinstance(families, list) or any(type(item) is not str for item in families):
        raise ValueError("skill spec families must be text")
    if type(value["version"]) is not int or not (
        value["parent_id"] is None or type(value["parent_id"]) is str
    ):
        raise ValueError("skill spec version/parent have incompatible types")
    return SkillSpec(
        skill_id=str(value["skill_id"]),
        name=str(value["name"]),
        description=str(value["description"]),
        body=str(value["body"]),
        families=tuple(families),
        version=value["version"],
        parent_id=value["parent_id"],
    )


def library_version_to_value(library: LibraryVersion) -> dict[str, JsonValue]:
    return {
        "skills": [skill_spec_to_value(spec) for spec in library.skills],
        "version": library.version,
    }


def library_version_from_value(value: object) -> LibraryVersion:
    if not isinstance(value, dict) or set(value) != {"skills", "version"}:
        raise ValueError("library version has an incompatible field set")
    if type(value["version"]) is not int or value["version"] < 0:
        raise ValueError("library version index must be a non-negative integer")
    skills = value["skills"]
    if not isinstance(skills, list):
        raise ValueError("library version skills must be a list")
    return LibraryVersion(value["version"], tuple(skill_spec_from_value(item) for item in skills))


def encode_phase_state(state: PhaseState) -> dict[str, JsonValue]:
    counts = sorted(state.carried_counts.items(), key=lambda item: (item[0][0], item[0][1]))
    for (_, _), (success, failure) in counts:
        if not (math.isfinite(success) and math.isfinite(failure)):
            raise ValueError("carried posterior counts must be finite")
    return {
        "carried_counts": [
            [skill, list(z), float(success), float(failure)]
            for (skill, z), (success, failure) in counts
        ],
        "cooldown": {skill: int(n) for skill, n in sorted(state.cooldown.items())},
        "phase": state.phase,
        "retired": list(state.retired),
    }


def decode_phase_state(value: object) -> PhaseState:
    if not isinstance(value, dict) or set(value) != {
        "carried_counts",
        "cooldown",
        "phase",
        "retired",
    }:
        raise ValueError("phase state has an incompatible field set")
    counts: dict[tuple[str, tuple[str, ...]], tuple[float, float]] = {}
    for row in value["carried_counts"]:
        if not isinstance(row, list) or len(row) != 4 or not isinstance(row[1], list):
            raise ValueError("carried counts are [skill, z, success, failure] rows")
        counts[(str(row[0]), tuple(str(item) for item in row[1]))] = (
            float(row[2]),
            float(row[3]),
        )
    cooldown = value["cooldown"]
    if not isinstance(cooldown, dict) or any(type(n) is not int for n in cooldown.values()):
        raise ValueError("cooldown must map skill ids to integers")
    if type(value["phase"]) is not int or not isinstance(value["retired"], list):
        raise ValueError("phase state phase/retired have incompatible types")
    return PhaseState(
        phase=value["phase"],
        carried_counts=counts,
        cooldown={str(skill): n for skill, n in cooldown.items()},
        retired=tuple(str(item) for item in value["retired"]),
    )


def skill_spec_document(spec: SkillSpec) -> SkillDocument:
    from skillev.experiments.skill_md_candidates import skill_md_document

    body = spec.body if spec.is_empty_slot else normalise_body(spec.body)
    skill = SkillMd(
        name=spec.skill_id,
        description=spec.description,
        version=str(spec.version),
        families=tuple(sorted(set(spec.families))),
        body=body,
    )
    document = skill_md_document(skill, provenance_kind=EVOLVED_SKILL_PROVENANCE)
    return document


def committed_spec(spec: SkillSpec) -> SkillSpec:
    document = skill_spec_document(spec)
    skill = document.skill_md()
    return SkillSpec(
        skill_id=spec.skill_id,
        name=spec.name,
        description=skill.description,
        body=skill.body,
        families=skill.families,
        version=spec.version,
        parent_id=spec.parent_id,
    )


def same_skill_content(left: SkillDocument, right: SkillDocument) -> bool:
    return left.content_value() == right.content_value() and (
        left.manifest.version == right.manifest.version
    )


def runtime_library_state(library: LibraryVersion, base: SkillLibraryState) -> SkillLibraryState:
    ids = library.skill_ids
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("a library version needs unique skill ids and at least one skill")
    documents = dict(base.documents)
    for spec in library.skills:
        document = skill_spec_document(spec)
        existing = documents.get(spec.skill_id)
        if existing is None:
            documents[spec.skill_id] = document
        elif not same_skill_content(existing, document):
            raise ValueError(
                f"skill {spec.skill_id!r} changed content under the same id "
                "(every edit must yield a new skill id)"
            )
    active = tuple(sorted(ids))
    return SkillLibraryState(
        documents=documents,
        active_skill_ids=active,
        current_version=skill_library_version(documents=documents, active_skill_ids=active),
    )


def library_matches_runtime(library: LibraryVersion, state: SkillLibraryState) -> bool:
    if tuple(sorted(library.skill_ids)) != state.active_skill_ids:
        return False
    return all(
        same_skill_content(state.documents[spec.skill_id], skill_spec_document(spec))
        for spec in library.skills
    )


def initial_library_version(
    state: SkillLibraryState, specs: tuple[SkillSpec, ...] | None = None
) -> LibraryVersion:
    if specs is None:
        converted = []
        for skill_id in state.active_skill_ids:
            skill = state.documents[skill_id].skill_md()
            converted.append(
                SkillSpec(
                    skill_id=skill_id,
                    name="empty slot" if skill.is_empty_slot else skill_id,
                    description=skill.description,
                    body=skill.body,
                    families=skill.families,
                    version=int(skill.version) if skill.version.isdigit() else 1,
                    parent_id=None,
                )
            )
        specs = tuple(converted)
    library = LibraryVersion(0, tuple(sorted(specs, key=lambda spec: spec.skill_id)))
    if not library_matches_runtime(library, state):
        raise ValueError("the initial library version differs from the runtime seed library")
    return library


@dataclass(frozen=True, slots=True)
class PhaseRecord:
    phase: int
    library: LibraryVersion
    library_digest: str
    phase_state: PhaseState
    since_step: int
    last_transition: dict[str, JsonValue] | None = None
    format: str = PHASE_RECORD_FORMAT

    def __post_init__(self) -> None:
        if self.format != PHASE_RECORD_FORMAT:
            raise ValueError("unsupported phase record format")
        if type(self.phase) is not int or self.phase < 0:
            raise ValueError("phase index must be a non-negative integer")
        if type(self.since_step) is not int or self.since_step < 0:
            raise ValueError("phase window start must be a non-negative optimizer step")
        if not isinstance(self.library, LibraryVersion) or not isinstance(
            self.phase_state, PhaseState
        ):
            raise TypeError("phase record needs a LibraryVersion and a PhaseState")
        if not self.library_digest.startswith("sha256:"):
            raise ValueError("library digest must be the runtime library version hash")

    @property
    def segment_label(self) -> str:
        return f"phase-{self.phase:04d}/v{self.library.version:04d}/{self.library_digest}"

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "last_transition": self.last_transition,
            "library": library_version_to_value(self.library),
            "library_digest": self.library_digest,
            "phase": self.phase,
            "phase_state": encode_phase_state(self.phase_state),
            "since_step": self.since_step,
        }

    @classmethod
    def from_value(cls, value: object) -> PhaseRecord:
        if not isinstance(value, dict) or set(value) != {
            "format",
            "last_transition",
            "library",
            "library_digest",
            "phase",
            "phase_state",
            "since_step",
        }:
            raise ValueError("phase record has an incompatible field set")
        last = value["last_transition"]
        if last is not None and not isinstance(last, dict):
            raise ValueError("last transition must be an object or null")
        return cls(
            phase=value["phase"],
            library=library_version_from_value(value["library"]),
            library_digest=str(value["library_digest"]),
            phase_state=decode_phase_state(value["phase_state"]),
            since_step=value["since_step"],
            last_transition=last,
            format=str(value["format"]),
        )


@dataclass(slots=True)
class PhaseCarrier:
    value: dict[str, JsonValue] | None = None

    def set(self, record: PhaseRecord) -> None:
        value = normalize_json(record.to_value())
        assert isinstance(value, dict)
        self.value = value

    def record(self) -> PhaseRecord | None:
        return None if self.value is None else PhaseRecord.from_value(self.value)


def _write_json(path: Path, value: Mapping[str, Any], *, exclusive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    text = canonical_json(normalize_json(dict(value))) + "\n"
    if exclusive and path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"{path} exists with different content (write-once)")
        return
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class LibraryVersionStore:
    def __init__(self, run_root: Path) -> None:
        self.root = Path(run_root) / LIBRARY_DIRECTORY

    def version_path(self, version: int) -> Path:
        return self.root / f"v{version:04d}.json"

    def transition_path(self, optimizer_step: int) -> Path:
        return self.root / f"transition-{optimizer_step:08d}.json"

    @property
    def active_path(self) -> Path:
        return self.root / "active.json"

    def save_version(self, library: LibraryVersion, *, digest: str, committed_at_step: int) -> Path:
        path = self.version_path(library.version)
        _write_json(
            path,
            {
                "format": LIBRARY_VERSION_FORMAT,
                "library": library_version_to_value(library),
                "library_digest": digest,
                "committed_at_optimizer_step": committed_at_step,
                "effective_from_optimizer_step": committed_at_step + 1,
                "content_hash": stable_hash(library_version_to_value(library)),
            },
            exclusive=True,
        )
        return path

    def load_version(self, version: int) -> LibraryVersion:
        value = json.loads(self.version_path(version).read_text(encoding="utf-8"))
        if value.get("format") != LIBRARY_VERSION_FORMAT:
            raise ValueError("unsupported library version file")
        return library_version_from_value(value["library"])

    def save_transition(self, optimizer_step: int, value: Mapping[str, Any]) -> Path:
        path = self.transition_path(optimizer_step)
        _write_json(path, {**value, "format": TRANSITION_RECORD_FORMAT}, exclusive=True)
        return path

    def load_transition(self, optimizer_step: int) -> dict[str, Any] | None:
        path = self.transition_path(optimizer_step)
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("format") != TRANSITION_RECORD_FORMAT:
            raise ValueError("unsupported transition commit record")
        return dict(value)

    def save_active(self, record: PhaseRecord) -> None:
        _write_json(self.active_path, record.to_value(), exclusive=False)


__all__ = [
    "EVOLVED_SKILL_PROVENANCE",
    "LIBRARY_DIRECTORY",
    "LIBRARY_VERSION_FORMAT",
    "PHASE_RECORD_FORMAT",
    "TRANSITION_RECORD_FORMAT",
    "LibraryVersionStore",
    "PhaseCarrier",
    "PhaseRecord",
    "committed_spec",
    "decode_phase_state",
    "encode_phase_state",
    "initial_library_version",
    "library_matches_runtime",
    "library_version_from_value",
    "library_version_to_value",
    "runtime_library_state",
    "same_skill_content",
    "skill_spec_document",
    "skill_spec_from_value",
    "skill_spec_to_value",
]
