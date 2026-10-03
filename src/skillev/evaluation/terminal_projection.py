from __future__ import annotations

import re
from enum import StrEnum

from .answer_parsing import (
    ParseStatus,
    parse_short_answer,
)
from .integer_payload import (
    is_integer_alternatives,
    is_integer_payload,
    parse_explicit_integer_payload,
)
from .python_payload import decode_python_module


class TerminalMode(StrEnum):
    SHORT_ANSWER = "short-answer"
    AIME_INTEGER = "integer-0-999"
    NATURAL_LANGUAGE = "natural-language"
    PYTHON_SOURCE = "python-source"


def _python_candidate(text: str) -> str | None:
    return decode_python_module(text)


def project_terminal_candidate(mode: TerminalMode, text: str) -> str | None:
    if type(text) is not str or not text.strip():
        return None
    if mode is TerminalMode.PYTHON_SOURCE:
        return _python_candidate(text)
    candidate = text.strip()
    if mode is TerminalMode.AIME_INTEGER:
        if is_integer_payload(candidate):
            value = parse_explicit_integer_payload(candidate)
            return rf"\boxed{{{value}}}" if value is not None else None
        if is_integer_alternatives(candidate):
            return None
        finals = re.findall(r"\\boxed\{([^{}]*)\}", candidate)
        last_line = candidate.splitlines()[-1].strip()
        if re.fullmatch(r"[0-9]{1,3}[ \t]*\.?", last_line):
            finals.append(last_line)
        finals.extend(
            re.findall(
                r"^(?:Final(?: answer)?|Answer|The answer is)\s*[:=]?\s*([0-9]+)\s*\.?\s*$",
                candidate,
                flags=re.I | re.M,
            )
        )
        values = {parse_explicit_integer_payload(value) for value in finals}
        if not values or None in values:
            return None
        return rf"\boxed{{{values.pop()}}}" if len(values) == 1 else None
    if mode is TerminalMode.SHORT_ANSWER:
        parsed = parse_short_answer(candidate)
        if parsed.status is ParseStatus.AMBIGUOUS:
            return None
        return parsed.value if parsed.value is not None else candidate
    return candidate


__all__ = ["TerminalMode", "project_terminal_candidate"]
