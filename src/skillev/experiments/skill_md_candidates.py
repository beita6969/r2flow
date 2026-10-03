from __future__ import annotations

from collections.abc import Callable, Sequence

from skillev.contracts.canonical import stable_hash
from skillev.runtime import SkillDocument
from skillev.runtime.contracts import SkillManifest
from skillev.runtime.skill_md import SkillMd, render_skill_md

PROFILE = "public-skill-md@1"
SKILL_MD_BODY_TOKEN_CAP = 768
EXECUTOR_INPUT_SCHEMA = "skill-executor-input@1"
EXECUTOR_OUTPUT_SCHEMA = "skill-executor-output@1"


def skill_md_document(skill: SkillMd, *, provenance_kind: str) -> SkillDocument:
    text = render_skill_md(skill)
    manifest = SkillManifest(
        skill_id=skill.name,
        version=skill.version,
        content_hash=stable_hash({"skill_md": text}),
        input_schema_id=EXECUTOR_INPUT_SCHEMA,
        output_schema_id=EXECUTOR_OUTPUT_SCHEMA,
        license_id="CC0-1.0",
        provenance_hash=stable_hash(
            {"kind": provenance_kind, "skill_id": skill.name, "version": skill.version}
        ),
    )
    return SkillDocument.from_skill_md(manifest, skill)


def require_skill_md_token_cap(
    documents: tuple[SkillDocument, ...], encode: Callable[[str], Sequence[int]]
) -> None:
    for document in documents:
        count = len(encode(render_skill_md(document.skill_md())))
        if count > SKILL_MD_BODY_TOKEN_CAP:
            raise ValueError(
                f"{document.manifest.skill_id} has {count} tokens > {SKILL_MD_BODY_TOKEN_CAP}"
            )


__all__ = [
    "EXECUTOR_INPUT_SCHEMA",
    "EXECUTOR_OUTPUT_SCHEMA",
    "PROFILE",
    "SKILL_MD_BODY_TOKEN_CAP",
    "require_skill_md_token_cap",
    "skill_md_document",
]
