from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import Any, Final

from skillev.contracts import JsonValue

from .phi import BOOTSTRAP_DOMAIN_FAMILIES
from .types import (
    STRUCTURAL_EDITS,
    AuthoredEdit,
    EditKind,
    GateDecision,
    LibraryVersion,
    PairedOutcome,
    PhaseState,
    serialized_fields,
)

GATE_TRACE_FORMAT: Final = "r2flow-gate-trace@1"
VALIDATION_SCOPE: Final = "sequential-per-candidate@1"
RULE_AFFECTED_FAMILIES: Final = "validation-scope=affected-families@2"
COOLDOWN_KINDS: Final = frozenset({EditKind.SPLIT, EditKind.REFINE, EditKind.PRUNE})
EDIT_SHAPES: Final = {
    EditKind.REFINE: (1, 1),
    EditKind.SPLIT: (1, 2),
    EditKind.COMPRESS: (2, 1),
    EditKind.GENERATE: (0, 1),
    EditKind.PRUNE: (1, 0),
}


def edit_id(edit: AuthoredEdit) -> str:
    added = ",".join(spec.skill_id for spec in edit.added)
    return f"{edit.candidate.kind.value}:{','.join(edit.removed)}->{added}"


def apply_edit(library: LibraryVersion, edit: AuthoredEdit) -> LibraryVersion:
    removed = set(edit.removed)
    return LibraryVersion(
        version=library.version + 1,
        skills=tuple(spec for spec in library.skills if spec.skill_id not in removed)
        + tuple(edit.added),
    )


def _families(library: LibraryVersion) -> set[str]:
    return {family for spec in library.skills for family in spec.families}


def _ineligible(edit: AuthoredEdit, current: LibraryVersion, state: PhaseState) -> str | None:
    candidate = edit.candidate
    kind = candidate.kind
    if kind not in STRUCTURAL_EDITS:
        return f"not a structural edit ({kind.value})"
    if candidate.evidence.get("verifier_eligible") is not True:
        return "(i) no verifier-only structural eligibility"
    shape = EDIT_SHAPES[kind]
    if (len(edit.removed), len(edit.added)) != shape:
        return f"(iii) malformed {kind.value}: expected {shape[0]} removed and {shape[1]} added"
    if kind is not EditKind.GENERATE and tuple(edit.removed) != tuple(candidate.skill_ids):
        return "(iii) removed skills differ from the candidate's targets"
    if len(set(edit.removed)) != len(edit.removed):
        return "(iii) duplicate removed skill"
    present = set(current.skill_ids)
    missing = [skill_id for skill_id in edit.removed if skill_id not in present]
    if missing:
        return f"(iii) version: target {missing[0]} is no longer in library v{current.version}"
    added_ids = [spec.skill_id for spec in edit.added]
    if any(spec.is_empty_slot for spec in edit.added):
        return "(iii) an added skill is empty"
    if len(set(added_ids)) != len(added_ids):
        return "(iii) duplicate added skill id"
    clash = [skill_id for skill_id in added_ids if skill_id in present]
    if clash:
        return f"(iii) version: added skill {clash[0]} is already in library v{current.version}"
    retired = [skill_id for skill_id in added_ids if skill_id in state.retired]
    if retired:
        return f"(iii) version: added skill {retired[0]} was retired"
    if kind in COOLDOWN_KINDS:
        for skill_id in edit.removed:
            remaining = state.cooldown.get(skill_id, 0)
            if remaining > 0:
                return f"(iii) cooldown: {skill_id} has {remaining} phase(s) remaining"
    lost = sorted(_families(current) - _families(apply_edit(current, edit)))
    if lost:
        return f"(iii) coverage: family {lost[0]} would have no skill"
    return None


def _json(value: Any) -> JsonValue:
    if hasattr(value, "to_value") and callable(value.to_value):
        return _json(value.to_value())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {name: _json(item) for name, item in serialized_fields(value)}
    if isinstance(value, Enum):
        return _json(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        items = sorted(value, key=repr) if isinstance(value, set | frozenset) else value
        return [_json(item) for item in items]
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return repr(value)


_PASS_KEYS: Final = ("passed", "pass", "noninferior", "ok")


def _tost_passed(result: Any, summary: JsonValue) -> bool:
    if isinstance(result, bool):
        return result
    for key in _PASS_KEYS:
        attribute = getattr(result, key, None)
        if isinstance(attribute, bool):
            return attribute
        if isinstance(summary, dict) and isinstance(summary.get(key), bool):
            return bool(summary[key])
    raise TypeError("the TOST result exposes no boolean 'passed'")


def _pair_means(pairs: Sequence[PairedOutcome]) -> dict[str, JsonValue]:
    if not pairs:
        return {}
    fields = (
        "success_a",
        "success_b",
        "reward_eta_a",
        "reward_eta_b",
        "tokens_a",
        "tokens_b",
        "latency_a",
        "latency_b",
    )
    return {
        name: _json(math.fsum(float(getattr(pair, name)) for pair in pairs) / len(pairs))
        for name in fields
    }


def _affected_families(edit: AuthoredEdit, current: LibraryVersion) -> set[str]:
    families = {
        f for spec in current.skills if spec.skill_id in edit.removed for f in spec.families
    }
    families.update(f for spec in edit.added for f in spec.families)
    return families


def _scoped_pairs(pairs: Sequence[PairedOutcome], families: set[str]) -> tuple[PairedOutcome, ...]:
    scoped = tuple(pair for pair in pairs if BOOTSTRAP_DOMAIN_FAMILIES.get(pair.domain) in families)
    return scoped if len(scoped) >= 2 else ()


def _decision_row(order: int, edit: AuthoredEdit, outcome: str, reason: str | None) -> JsonValue:
    candidate = edit.candidate
    return {
        "order": order,
        "edit_id": edit_id(edit),
        "kind": candidate.kind.value,
        "skills": list(candidate.skill_ids),
        "context": list(candidate.context) if candidate.context is not None else None,
        "rank": _json(candidate.rank),
        "evidence": _json(candidate.evidence),
        "removed": list(edit.removed),
        "added": [spec.skill_id for spec in edit.added],
        "author_model": edit.author_model,
        "outcome": outcome,
        "reason": reason,
    }


def gate(
    authored: Sequence[AuthoredEdit],
    *,
    library: LibraryVersion,
    run_paired: Callable[[LibraryVersion, LibraryVersion], Sequence[PairedOutcome]],
    tost: Callable[..., Any],
    margins: Mapping[str, float],
    alpha: float,
    state: PhaseState,
    max_validations: int,
    phase: int,
    diagnostics: Mapping[str, JsonValue] | None = None,
) -> GateDecision:
    if type(max_validations) is not int or max_validations < 0:
        raise ValueError("max_validations must be a non-negative integer")
    current = library
    accepted: list[AuthoredEdit] = []
    rejected: list[tuple[AuthoredEdit, str]] = []
    decisions: list[JsonValue] = []
    validations: list[JsonValue] = []
    for order, edit in enumerate(authored):
        reason = _ineligible(edit, current, state)
        if reason is None and len(validations) >= max_validations:
            reason = f"not validated: max_validations={max_validations} reached"
        if reason is not None:
            rejected.append((edit, reason))
            decisions.append(_decision_row(order, edit, "rejected", reason))
            continue
        candidate_library = apply_edit(current, edit)
        pairs = tuple(run_paired(current, candidate_library))
        n_all = len(pairs)
        pairs = _scoped_pairs(pairs, _affected_families(edit, current))
        if pairs:
            result = tost(pairs, margins=margins, alpha=alpha)
            summary = _json(result)
            passed = _tost_passed(result, summary)
        else:
            summary, passed = None, False
        row: dict[str, JsonValue] = {
            "order": order,
            "edit_id": edit_id(edit),
            "v_current": current.version,
            "v_candidate": candidate_library.version,
            "n": len(pairs),
            "pair_means": _pair_means(pairs),
            "tost": summary,
            "passed": passed,
            "scope": "affected-transferable-families",
            "n_all_pairs": n_all,
        }
        validations.append(row)
        if passed:
            accepted.append(edit)
            decisions.append(_decision_row(order, edit, "accepted", None))
            current = candidate_library
        else:
            reason = (
                "(ii) paired non-inferiority failed: rolled back"
                if pairs
                else "(ii) no paired held-out outcomes: rolled back"
            )
            rejected.append((edit, reason))
            decisions.append(_decision_row(order, edit, "rejected", reason))
    library_after = library if not accepted else current
    trace_row: dict[str, JsonValue] = {
        "format": GATE_TRACE_FORMAT,
        "k": phase,
        "v_k": library.version,
        "v_k1": library_after.version,
        "delta": _json(dict(diagnostics or {})),
        "decisions": decisions,
        "validations": validations,
        "validation_scope": VALIDATION_SCOPE,
        "alpha": _json(alpha),
        "margins": {key: _json(margins[key]) for key in sorted(margins)},
        "max_validations": max_validations,
        "amendments": [RULE_AFFECTED_FAMILIES],
        "accepted": [edit_id(edit) for edit in accepted],
        "rejected": [{"edit_id": edit_id(edit), "reason": why} for edit, why in rejected],
        "library_skill_ids": list(library_after.skill_ids),
    }
    return GateDecision(
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        library_after=library_after,
        trace_row=trace_row,
    )


__all__ = [
    "COOLDOWN_KINDS",
    "EDIT_SHAPES",
    "GATE_TRACE_FORMAT",
    "VALIDATION_SCOPE",
    "apply_edit",
    "edit_id",
    "gate",
]
