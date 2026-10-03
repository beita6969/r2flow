from __future__ import annotations

import json
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from skillev.contracts import JsonValue

from .evidence import clip_middle

MEMORY_RULE: Final = "author-memory=previous-drafts@2"
MAX_PREVIOUS_DRAFTS: Final = 3
PREVIOUS_BODY_CHARS: Final = 1200
SECTION_TITLE: Final = "# Previous drafts for this family"


@dataclass(frozen=True, slots=True)
class PreviousDraft:
    phase: int
    order: int
    skill_id: str
    families: tuple[str, ...]
    body: str
    kind: str
    template: str | None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _dicts(value: object) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _phase_drafts(phase: int, record: Path) -> list[PreviousDraft]:
    authored = _read_json(record / "authored.json")
    if not isinstance(authored, dict) or not (record / "decision.json").is_file():
        return []
    drafts: list[PreviousDraft] = []
    for order, row in enumerate(_dicts(authored.get("drafts"))):
        candidate = row.get("candidate")
        kind = str(candidate.get("kind") or "") if isinstance(candidate, dict) else ""
        author_record = row.get("author_record")
        template = author_record.get("template") if isinstance(author_record, dict) else None
        for spec in _dicts(row.get("added")):
            body, families = spec.get("body"), spec.get("families")
            if not isinstance(spec.get("skill_id"), str) or not isinstance(families, list):
                continue
            if not isinstance(body, str) or not body.strip():
                continue
            drafts.append(
                PreviousDraft(
                    phase=phase,
                    order=order,
                    skill_id=str(spec["skill_id"]),
                    families=tuple(sorted({str(f) for f in families})),
                    body=body,
                    kind=kind,
                    template=template if isinstance(template, str) else None,
                )
            )
    return drafts


def load_previous_drafts(run_root: Path, phase: int) -> tuple[PreviousDraft, ...]:
    if type(phase) is not int or phase < 0:
        raise ValueError("phase must be a non-negative integer")
    evolution = Path(run_root) / "evolution"
    drafts: list[PreviousDraft] = []
    for earlier in range(phase - 1, -1, -1):
        record = evolution / f"phase-{earlier}"
        if record.is_dir():
            drafts.extend(_phase_drafts(earlier, record))
    return tuple(drafts)


def drafts_for_families(
    drafts: Sequence[PreviousDraft],
    families: Collection[str],
    *,
    limit: int = MAX_PREVIOUS_DRAFTS,
) -> tuple[PreviousDraft, ...]:
    wanted = set(families)
    ordered = sorted(drafts, key=lambda draft: (-draft.phase, draft.order, draft.skill_id))
    return tuple(draft for draft in ordered if wanted & set(draft.families))[:limit]


def render_previous_drafts(
    drafts: Sequence[PreviousDraft],
    redact: Callable[[str], str],
    *,
    current_ids: Collection[str] = (),
) -> str | None:
    if not drafts:
        return None
    lines = [
        SECTION_TITLE,
        "Earlier drafts written for these families in this run, most recent first.",
    ]
    for number, draft in enumerate(drafts, start=1):
        kind = f" ({draft.kind} edit)" if draft.kind else ""
        lines.append(f"\n## Draft {number}: phase {draft.phase}{kind}")
        if draft.skill_id in current_ids:
            lines.append("Draft body: the current skill shown above.")
        else:
            body = clip_middle(redact(draft.body.strip()), PREVIOUS_BODY_CHARS)
            lines.append(f"Draft body:\n<<<\n{body}\n>>>")
    lines.append(
        "\nHow to use these drafts: they show what has already been written for these families. "
        "Do not resubmit or lightly rephrase an earlier draft; write the procedure the current "
        "evidence below calls for, keeping what the evidence shows working and changing what it "
        "shows failing."
    )
    return "\n".join(lines)


def memory_record(drafts: Sequence[PreviousDraft]) -> dict[str, JsonValue]:
    shown: list[JsonValue] = [
        {"phase": draft.phase, "skill_id": draft.skill_id} for draft in drafts
    ]
    return {"rule": MEMORY_RULE, "shown": shown}


__all__ = [
    "MAX_PREVIOUS_DRAFTS",
    "MEMORY_RULE",
    "PREVIOUS_BODY_CHARS",
    "SECTION_TITLE",
    "PreviousDraft",
    "drafts_for_families",
    "load_previous_drafts",
    "memory_record",
    "render_previous_drafts",
]
