from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Callable, Collection
from typing import Final

from skillev.contracts.identity import validate_identifier
from skillev.runtime.skill_md import (
    TRANSFERABLE_FAMILIES,
    SkillMd,
    normalise_body,
    parse_skill_md,
    render_skill_md,
)

from .types import SkillSpec

SKILL_MD_VALIDATION_VERSION: Final = "r2flow-skill-md-validation@2"
SUPERVISOR_FORMAT_RULE: Final = "no-supervisor-answer-format@2"
TOKEN_ESTIMATE_RULE: Final = "approx-tokens-words-punct-x1.3@1"
DEFAULT_TOKEN_CAP: Final = 768
NAME_MAX_CHARS: Final = 64
DESCRIPTION_MAX_CHARS: Final = 300
FORBIDDEN_MIN_CHARS: Final = 3
GENERIC_ANSWER_TOKENS: Final = frozenset(
    {"yes", "no", "true", "false", "none", "null", "unknown", "n/a"}
)
BENCHMARK_NAMES: Final = (
    "mbpp",
    "mbpp-plus",
    "mbpp+",
    "humaneval",
    "hotpotqa",
    "hotpot qa",
    "triviaqa",
    "trivia qa",
    "alfworld",
    "aime",
    "healthbench",
    "livemedbench",
    "scienceworld",
    "webshop",
    "swe-bench",
    "swebench",
    "livecodebench",
    "nq-open",
    "gsm8k",
    "math-hard",
    "omni-math",
    "apps-introductory",
    "mind2web",
    "appworld",
)
_TOKEN_RE: Final = re.compile(r"\w+|[^\w\s]")
_WS_RE: Final = re.compile(r"\s+")
_DIGITS_RE: Final = re.compile(r"[0-9]+")
_BENCHMARK_RES: Final = (
    re.compile(
        r"(?<![\w-])(?:" + "|".join(re.escape(name) for name in BENCHMARK_NAMES) + r")(?![\w])",
        re.IGNORECASE,
    ),
    re.compile(r"\b[A-Z][A-Za-z]{2,}(?:[+-][A-Za-z]+)?/\d+\b"),
    re.compile(r"\b[a-z][\w-]*:(?:train|test|dev|valid\w*|val|heldout|held-out)\b", re.I),
    re.compile(r"\bvalid_(?:un)?seen\b|\bjson_2\.1\.\d\b", re.IGNORECASE),
    re.compile(r"\b(?:https?|ftp)://|\bwww\.[a-z0-9-]+\.[a-z]", re.IGNORECASE),
)
_SUBMIT_FUNCTION_RE: Final = re.compile(
    r"\bsubmit_answer\b|\bopen_passage\b|\binvoke_skill\b|\bcorpus_search\b"
)
_ANSWER_WRAPPER_RES: Final = (
    re.compile(r"\\boxed\b|\bboxed\{", re.IGNORECASE),
    re.compile(r"^\s*####\s*(?:\d|<)", re.MULTILINE),
    re.compile(r"\b0{1,3}\s*(?:-|\u2013|to|through|and)\s*999\b", re.IGNORECASE),
)
_ADDRESSEE_RE: Final = re.compile(
    r"\b(?:supervisor|orchestrator|caller|calling agent)\b", re.IGNORECASE
)
_SUBMISSION_RE: Final = re.compile(
    r"\b(?:submit\w*|submission|final answer|answer format|format(?:ting)? (?:the |its |your )?"
    r"(?:final )?answer)\b",
    re.IGNORECASE,
)


def approx_token_count(text: str) -> int:
    return math.ceil(1.3 * len(_TOKEN_RE.findall(text)))


def normalise_for_match(text: str) -> str:
    return _WS_RE.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _library_skill_md(spec: SkillSpec) -> SkillMd:
    return SkillMd(
        name=spec.skill_id,
        description=spec.description,
        version=str(spec.version),
        families=tuple(sorted(set(spec.families))),
        body=spec.body,
    )


def render_library_skill(spec: SkillSpec) -> str:
    if spec.is_empty_slot:
        raise ValueError(f"{spec.skill_id} is an empty slot and has no SKILL.md document")
    return render_skill_md(_library_skill_md(spec))


def forbidden_patterns(forbidden_strings: Collection[str]) -> tuple[re.Pattern[str], ...]:
    patterns = forbidden_needle_patterns(forbidden_strings)
    return tuple(patterns[key] for key in sorted(patterns))


def forbidden_needle_patterns(forbidden_strings: Collection[str]) -> dict[str, re.Pattern[str]]:
    patterns: dict[str, re.Pattern[str]] = {}
    for raw in forbidden_strings:
        if type(raw) is not str:
            raise TypeError("forbidden strings must be text")
        needle = normalise_for_match(raw)
        numeric = _DIGITS_RE.fullmatch(needle) is not None
        if needle in GENERIC_ANSWER_TOKENS:
            continue
        if len(needle) < FORBIDDEN_MIN_CHARS and not (numeric and len(needle) >= 2):
            continue
        if numeric:
            pattern = r"(?<![\w.,])" + needle + r"(?!\w|[.,]\d)"
        else:
            head = r"(?<!\w)" if needle[0].isalnum() or needle[0] == "_" else ""
            tail = r"(?!\w)" if needle[-1].isalnum() or needle[-1] == "_" else ""
            pattern = head + r"\s+".join(re.escape(part) for part in needle.split(" ")) + tail
        patterns.setdefault(needle, re.compile(pattern, re.IGNORECASE))
    return patterns


def needle_sha256(needle: str) -> str:
    return hashlib.sha256(needle.encode("utf-8")).hexdigest()


def contains_forbidden(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    nfkc = unicodedata.normalize("NFKC", text)
    folded = nfkc.casefold()
    return any(pattern.search(nfkc) or pattern.search(folded) for pattern in patterns)


def redact_forbidden(
    text: str, patterns: tuple[re.Pattern[str], ...], *, marker: str = "[REMOVED]"
) -> tuple[str, int]:
    if not patterns:
        return text, 0
    redacted = unicodedata.normalize("NFKC", text)
    count = 0
    for pattern in patterns:
        redacted, hits = pattern.subn(marker, redacted)
        count += hits
    if contains_forbidden(redacted, patterns):
        redacted = normalise_for_match(redacted)
        for pattern in patterns:
            redacted, hits = pattern.subn(marker, redacted)
            count += hits
    return (text if count == 0 else redacted), count


def _structure_errors(body: str) -> list[str]:
    errors = []
    if not re.search(r"(?m)^INPUT:[ \t]*\S", body):
        errors.append("structure: the body needs an 'INPUT: <what the caller passes>' line")
    if not re.search(r"(?m)^OUTPUT:[ \t]*\S", body):
        errors.append("structure: the body needs an 'OUTPUT: <what the executor returns>' line")
    procedure = re.search(r"(?m)^Procedure:[ \t]*$", body)
    if procedure is None:
        errors.append("structure: the body needs a 'Procedure:' line followed by numbered steps")
    elif not re.search(r"(?m)^\d+\.[ \t]+\S", body[procedure.end() :]):
        errors.append("structure: 'Procedure:' must be followed by numbered steps ('1. ...')")
    return errors


def _benchmark_errors(text: str) -> list[str]:
    errors = []
    if _BENCHMARK_RES[0].search(text):
        errors.append(
            "benchmark identity: do not name benchmarks or datasets; describe the task class"
        )
    if any(pattern.search(text) for pattern in _BENCHMARK_RES[1:4]):
        errors.append("benchmark identity: remove task ids, dataset ids or split names")
    if _BENCHMARK_RES[4].search(text):
        errors.append("benchmark identity: remove URLs")
    return errors


def _supervisor_format_errors(text: str) -> list[str]:
    errors = []
    if _SUBMIT_FUNCTION_RE.search(text):
        errors.append(
            "supervisor instruction: do not name the Supervisor's functions; the skill "
            "instructs only the executor"
        )
    if any(pattern.search(text) for pattern in _ANSWER_WRAPPER_RES):
        errors.append(
            "supervisor instruction: remove benchmark answer-format conventions (answer "
            "wrappers, fixed answer ranges)"
        )
    if any(_ADDRESSEE_RE.search(line) and _SUBMISSION_RE.search(line) for line in text.split("\n")):
        errors.append(
            "supervisor instruction: do not tell the Supervisor/caller how to format or submit "
            "its final answer; state only what the executor returns"
        )
    return errors


def validate_skill_md(
    spec: SkillSpec,
    *,
    forbidden_strings: Collection[str],
    token_cap: int = DEFAULT_TOKEN_CAP,
    count_tokens: Callable[[str], int] | None = None,
) -> list[str]:
    errors: list[str] = []
    if type(spec.body) is not str or not spec.body.strip():
        return ["body: authored skills must have a non-empty body"]
    if normalise_body(spec.body) != spec.body:
        errors.append("body: not in normalised form (LF, NFC, no trailing spaces, one final LF)")
    if spec.body.split("\n", 1)[0] == "---":
        errors.append("body: must not start with a frontmatter fence")
    if type(spec.name) is not str or not spec.name:
        errors.append("frontmatter: name is missing")
    else:
        try:
            validate_identifier(spec.name)
        except ValueError:
            errors.append("frontmatter: name must be a lowercase slug (letters, digits, '-')")
        if len(spec.name) > NAME_MAX_CHARS:
            errors.append(f"frontmatter: name longer than {NAME_MAX_CHARS} characters")
    description = spec.description if type(spec.description) is str else ""
    if not description.strip():
        errors.append("frontmatter: description is missing")
    elif "\n" in description or "\r" in description or description != description.strip():
        errors.append("frontmatter: description must be one line without outer spaces")
    elif len(description) > DESCRIPTION_MAX_CHARS:
        errors.append(f"frontmatter: description longer than {DESCRIPTION_MAX_CHARS} characters")
    families = tuple(sorted(set(spec.families)))
    if not families or not set(families) <= TRANSFERABLE_FAMILIES:
        errors.append(f"families: must be non-empty and within {sorted(TRANSFERABLE_FAMILIES)}")
    errors.extend(_structure_errors(spec.body))
    document: str | None = None
    if not errors:
        try:
            document = render_library_skill(spec)
            parsed = parse_skill_md(document)
        except (TypeError, ValueError) as error:
            errors.append(f"skill-md: not a canonical SKILL.md document ({error})")
        else:
            if parsed != _library_skill_md(spec):
                errors.append("skill-md: the SKILL.md document does not round-trip")
    counted = document if document is not None else spec.body
    tokens = (count_tokens or approx_token_count)(counted)
    if tokens > token_cap:
        errors.append(
            f"size: about {tokens} tokens > cap {token_cap}; shorten the procedure "
            "(fewer, shorter steps)"
        )
    visible = "\n".join((spec.name or "", description, spec.body))
    if contains_forbidden(visible, forbidden_patterns(forbidden_strings)):
        errors.append(
            "answer-free: the skill reproduces task-specific text from the training material "
            "(an answer or solution); remove every task-specific value"
        )
    errors.extend(_benchmark_errors(visible))
    errors.extend(_supervisor_format_errors(visible))
    return errors


__all__ = [
    "BENCHMARK_NAMES",
    "DEFAULT_TOKEN_CAP",
    "DESCRIPTION_MAX_CHARS",
    "FORBIDDEN_MIN_CHARS",
    "GENERIC_ANSWER_TOKENS",
    "NAME_MAX_CHARS",
    "SKILL_MD_VALIDATION_VERSION",
    "SUPERVISOR_FORMAT_RULE",
    "TOKEN_ESTIMATE_RULE",
    "approx_token_count",
    "contains_forbidden",
    "forbidden_needle_patterns",
    "forbidden_patterns",
    "needle_sha256",
    "normalise_for_match",
    "redact_forbidden",
    "render_library_skill",
    "validate_skill_md",
]
