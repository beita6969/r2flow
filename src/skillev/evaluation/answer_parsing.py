from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class ParseStatus(StrEnum):
    EXTRACTED = "extracted"
    EMPTY = "empty"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class ParsedResponse:
    value: str | None
    status: ParseStatus


_FINAL_MARKER = re.compile(r"(?im)^\s*(?:final answer|answer)\s*:\s*(.+?)\s*$")


def _strip_thinking(text: str) -> str:
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    return text.strip()


def _empty() -> ParsedResponse:
    return ParsedResponse(None, ParseStatus.EMPTY)


def _unique(candidates: list[str]) -> ParsedResponse:
    unique = tuple(dict.fromkeys(item.strip() for item in candidates if item.strip()))
    if not unique:
        return _empty()
    if len(unique) > 1:
        return ParsedResponse(None, ParseStatus.AMBIGUOUS)
    return ParsedResponse(unique[0], ParseStatus.EXTRACTED)


def parse_short_answer(text: str) -> ParsedResponse:
    visible = _strip_thinking(text)
    if not visible:
        return _empty()
    explicit = [match.group(1).strip() for match in _FINAL_MARKER.finditer(visible)]
    if explicit:
        return _unique(explicit)
    lines = [line.strip() for line in visible.splitlines() if line.strip()]
    if len(lines) == 1:
        return ParsedResponse(lines[0], ParseStatus.EXTRACTED)
    return _empty()
