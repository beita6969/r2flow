from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass

import yaml

from skillev.contracts.identity import validate_identifier

SKILL_MD_FORMAT = "skill-md@1"
TRANSFERABLE_FAMILIES = frozenset(
    {
        "code-generation",
        "factual-qa",
        "health-dialogue",
        "interactive-decision",
        "mathematical-reasoning",
        "multi-hop-qa",
    }
)
_FENCE = "---"
EMPTY_SLOT_BODY = ""
EMPTY_SLOT_DESCRIPTION = "Empty skill slot: no procedure yet."


def normalise_body(text: str) -> str:
    if type(text) is not str:
        raise TypeError("SKILL.md body must be text")
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n" if lines else ""


@dataclass(frozen=True, slots=True)
class SkillMd:
    name: str
    description: str
    version: str
    families: tuple[str, ...]
    body: str

    def __post_init__(self) -> None:
        for field in ("name", "description", "version", "body"):
            if type(getattr(self, field)) is not str:
                raise TypeError(f"SKILL.md {field} must be text")
        validate_identifier(self.name)
        if not self.description.strip() or any(c in self.description for c in "\r\n"):
            raise ValueError("SKILL.md description must be one non-empty line")
        if self.description != self.description.strip():
            raise ValueError("SKILL.md description must not have outer whitespace")
        if not self.version.strip() or any(c in self.version for c in "\r\n"):
            raise ValueError("SKILL.md version must be one non-empty line")
        if not isinstance(self.families, tuple) or not self.families:
            raise ValueError("SKILL.md families must be a non-empty tuple")
        if tuple(sorted(set(self.families))) != self.families:
            raise ValueError("SKILL.md families must be sorted and unique")
        unknown = set(self.families) - TRANSFERABLE_FAMILIES
        if unknown:
            raise ValueError(f"SKILL.md families outside the declared set: {sorted(unknown)}")
        if self.is_empty_slot:
            return
        if not self.body.strip() or "\x00" in self.body:
            raise ValueError("SKILL.md body must be non-empty text without NUL")
        if normalise_body(self.body) != self.body:
            raise ValueError("SKILL.md body must be normalised")
        if self.body.split("\n", 1)[0] == _FENCE:
            raise ValueError("SKILL.md body cannot start with a frontmatter fence")

    @property
    def is_empty_slot(self) -> bool:
        return self.body == EMPTY_SLOT_BODY and self.description == EMPTY_SLOT_DESCRIPTION


def render_skill_md(skill: SkillMd) -> str:
    families = ", ".join(json.dumps(family) for family in skill.families)
    return (
        f"{_FENCE}\n"
        f"name: {skill.name}\n"
        f"description: {json.dumps(skill.description, ensure_ascii=False)}\n"
        f"version: {json.dumps(skill.version, ensure_ascii=False)}\n"
        "applicability:\n"
        f"  families: [{families}]\n"
        f"{_FENCE}\n\n" + skill.body
    )


def parse_skill_md(text: str) -> SkillMd:
    if type(text) is not str:
        raise TypeError("SKILL.md must be text")
    if "\r" in text:
        raise ValueError("SKILL.md must use LF line endings")
    lines = text.split("\n")
    if not lines or lines[0] != _FENCE:
        raise ValueError("SKILL.md must start with a frontmatter fence")
    try:
        end = lines.index(_FENCE, 1)
    except ValueError as error:
        raise ValueError("SKILL.md frontmatter is not closed") from error
    try:
        loaded = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError as error:
        raise ValueError("SKILL.md frontmatter is not valid YAML") from error
    if not isinstance(loaded, dict) or set(loaded) != {
        "name",
        "description",
        "version",
        "applicability",
    }:
        raise ValueError("SKILL.md frontmatter has an incompatible key set")
    applicability = loaded["applicability"]
    if not isinstance(applicability, dict) or set(applicability) != {"families"}:
        raise ValueError("SKILL.md applicability must contain exactly families")
    families = applicability["families"]
    if not isinstance(families, list) or any(type(item) is not str for item in families):
        raise ValueError("SKILL.md families must be a list of text")
    body = "\n".join(lines[end + 1 :])
    if not body.startswith("\n"):
        raise ValueError("SKILL.md frontmatter must be followed by one blank line")
    try:
        parsed = SkillMd(
            name=loaded["name"],
            description=loaded["description"],
            version=loaded["version"],
            families=tuple(families),
            body=body[1:],
        )
    except TypeError as error:
        raise ValueError(str(error)) from error
    if render_skill_md(parsed) != text:
        raise ValueError("SKILL.md is not in canonical form")
    return parsed


__all__ = [
    "EMPTY_SLOT_BODY",
    "EMPTY_SLOT_DESCRIPTION",
    "SKILL_MD_FORMAT",
    "TRANSFERABLE_FAMILIES",
    "SkillMd",
    "normalise_body",
    "parse_skill_md",
    "render_skill_md",
]
