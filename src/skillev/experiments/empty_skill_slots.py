from __future__ import annotations

from typing import TYPE_CHECKING, Final

from skillev.runtime.skill_md import (
    EMPTY_SLOT_BODY,
    EMPTY_SLOT_DESCRIPTION,
    TRANSFERABLE_FAMILIES,
    SkillMd,
)

if TYPE_CHECKING:
    from skillev.r2flow_evolution.types import SkillSpec
    from skillev.runtime import SkillDocument, SkillLibraryState

PROFILE: Final = "empty-skill-slots@1"
SLOTS_PER_FAMILY: Final = 2
EMPTY_SLOT_NAME: Final = "empty slot"
EMPTY_SLOT_VERSION: Final = 1
EMPTY_SLOT_PROVENANCE: Final = "empty-skill-slot@1"


def slot_id(family: str, ordinal: int) -> str:
    if family not in TRANSFERABLE_FAMILIES or ordinal not in range(1, SLOTS_PER_FAMILY + 1):
        raise ValueError("an empty slot names a transferable family and ordinal 1..2")
    return f"skill-{family}-slot-{ordinal}"


def empty_slot_families(domains: tuple[str, ...] | None) -> tuple[str, ...]:
    if domains is None:
        return tuple(sorted(TRANSFERABLE_FAMILIES))
    from skillev.evolution.task_features import transferable_family

    families = []
    for domain in domains:
        family = transferable_family(domain)
        if family is None:
            raise ValueError(f"domain {domain!r} has no transferable family for an empty slot")
        families.append(family)
    return tuple(sorted(set(families)))


def empty_slot_skills(domains: tuple[str, ...] | None) -> tuple[SkillMd, ...]:
    return tuple(
        SkillMd(
            name=slot_id(family, ordinal),
            description=EMPTY_SLOT_DESCRIPTION,
            version=str(EMPTY_SLOT_VERSION),
            families=(family,),
            body=EMPTY_SLOT_BODY,
        )
        for family in empty_slot_families(domains)
        for ordinal in range(1, SLOTS_PER_FAMILY + 1)
    )


def empty_slot_documents(domains: tuple[str, ...] | None) -> tuple[SkillDocument, ...]:
    from .skill_md_candidates import skill_md_document

    return tuple(
        skill_md_document(skill, provenance_kind=EMPTY_SLOT_PROVENANCE)
        for skill in empty_slot_skills(domains)
    )


def empty_slot_specs(domains: tuple[str, ...] | None) -> tuple[SkillSpec, ...]:
    from skillev.r2flow_evolution.types import SkillSpec

    return tuple(
        SkillSpec(
            skill_id=skill.name,
            name=EMPTY_SLOT_NAME,
            description=skill.description,
            body=skill.body,
            families=skill.families,
            version=EMPTY_SLOT_VERSION,
            parent_id=None,
        )
        for skill in empty_slot_skills(domains)
    )


def visible_skill_count(state: SkillLibraryState, skill_visibility: str | None) -> int:
    ids = state.active_skill_ids
    if skill_visibility is None:
        return len(ids)
    from skillev.contracts.skill_visibility import SKILL_VISIBILITY_RULES

    if skill_visibility not in SKILL_VISIBILITY_RULES:
        raise ValueError("unsupported skill visibility rule")
    return sum(1 for skill_id in ids if not state.documents[skill_id].skill_md().is_empty_slot)


def visible_skills_by_family(
    state: SkillLibraryState, skill_visibility: str | None
) -> dict[str, int]:
    from skillev.contracts.skill_visibility import SKILL_VISIBILITY_RULES

    if skill_visibility is not None and skill_visibility not in SKILL_VISIBILITY_RULES:
        raise ValueError("unsupported skill visibility rule")
    counts: dict[str, int] = {}
    for skill_id in state.active_skill_ids:
        skill = state.documents[skill_id].skill_md()
        if skill_visibility is not None and skill.is_empty_slot:
            continue
        for family in sorted(set(skill.families)):
            counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


__all__ = [
    "EMPTY_SLOT_NAME",
    "EMPTY_SLOT_PROVENANCE",
    "EMPTY_SLOT_VERSION",
    "PROFILE",
    "SLOTS_PER_FAMILY",
    "empty_slot_documents",
    "empty_slot_families",
    "empty_slot_skills",
    "empty_slot_specs",
    "slot_id",
    "visible_skill_count",
    "visible_skills_by_family",
]
