from __future__ import annotations

import re
import string
from collections.abc import Sequence
from typing import Final

CLAIM_PATTERN = re.compile(
    r"^\s*(?:final\s+)?answer\s*[:\N{FULLWIDTH COLON}]\s*(.+?)\s*$", re.IGNORECASE
)
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = frozenset(string.punctuation)


def claimed_answer(output: str) -> str | None:
    claim = None
    for line in output.split("\n"):
        match = CLAIM_PATTERN.match(line)
        if match is not None:
            claim = match.group(1)
    return claim


def squad_normalize(text: str) -> str:
    lowered = "".join(char for char in text.lower() if char not in _PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", lowered).split())


_H0_DOCUMENTS: Final = re.compile(r"\n\nDocuments \((\d+)\):\n")
YES_NO_ANSWERS: Final = frozenset({"yes", "no"})


def distractor_documents(query: str) -> str | None:
    match = _H0_DOCUMENTS.search(query)
    if match is None or int(match.group(1)) < 1:
        return None
    documents = query[match.end() :]
    return documents if documents.strip() else None


def submit_grounding_outcome(answer: str, contexts: Sequence[str]) -> tuple[bool | None, str]:
    normalized = squad_normalize(answer)
    if not normalized:
        return False, "empty-answer"
    if normalized in YES_NO_ANSWERS:
        return None, "yes-no-answer"
    if not contexts:
        return None, "no-context"
    padded = f" {normalized} "
    for text in contexts:
        if padded in f" {squad_normalize(text)} ":
            return True, "grounded"
    return False, "not-grounded"


def grounding_outcome(output: str, opened_passages: Sequence[str]) -> tuple[bool | None, str]:
    claim = claimed_answer(output)
    if claim is None:
        return None, "no-claimed-answer"
    normalized = squad_normalize(claim)
    if not normalized:
        return False, "empty-claim"
    if not opened_passages:
        return False, "no-opened-passage"
    padded = f" {normalized} "
    for passage in opened_passages:
        if padded in f" {squad_normalize(passage)} ":
            return True, "grounded"
    return False, "not-grounded"


__all__ = [
    "CLAIM_PATTERN",
    "YES_NO_ANSWERS",
    "claimed_answer",
    "distractor_documents",
    "grounding_outcome",
    "squad_normalize",
    "submit_grounding_outcome",
]
