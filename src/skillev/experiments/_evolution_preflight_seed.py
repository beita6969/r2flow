from __future__ import annotations

from skillev.runtime import SkillDocument


def planned_seed_documents(
    profile: str = "empty-skill-slots@1", domains: tuple[str, ...] | None = None
) -> tuple[SkillDocument, ...]:
    if profile != "empty-skill-slots@1":
        raise ValueError("unknown initial skill library profile")
    from .empty_skill_slots import empty_slot_documents

    return empty_slot_documents(domains)


__all__ = ["planned_seed_documents"]
