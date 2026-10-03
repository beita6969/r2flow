from __future__ import annotations

import re

from .integer_payload import INTEGER_SCALAR
from .integer_payload import parse_explicit_integer_payload as parse_explicit_integer_payload
from .native_tool_calls import native_tool_call
from .response_syntax import protocol_lines
from .terminal_projection import TerminalMode, project_terminal_candidate


_FINAL_LABEL = r"(?:Final(?: answer| response| code)?|Terminal payload)"


def _short_answer_header(
    line: str, label_pattern: str = _FINAL_LABEL
) -> tuple[int, int | None] | None:
    prefix = r"^[ \t]*(?:\#{1,6}[ \t]+)?"
    plain = re.match(prefix + label_pattern + r"[ \t]*:[ \t]*", line, re.I)
    if plain:
        return plain.end(), None
    for emphasis in (r"\*\*", "__", r"\*", "_"):
        label = re.match(
            prefix + emphasis + label_pattern + rf"[ \t]*(?::{emphasis}|{emphasis}:)[ \t]*",
            line,
            re.I,
        )
        if label:
            return label.end(), None
        whole = re.fullmatch(
            prefix
            + emphasis
            + label_pattern
            + r"[ \t]*:[ \t]*(?P<body>.*?)"
            + emphasis
            + r"[ \t]*(?:\r?\n)?",
            line,
            re.I,
        )
        if whole:
            return whole.start("body"), whole.end("body")
    return None


def _final_fields(
    text: str, *, markdown: bool = False, label_pattern: str = _FINAL_LABEL
) -> tuple[str, ...]:
    offsets: list[tuple[int, int, int | None]] = []
    for offset, line in protocol_lines(text):
        if markdown:
            bounds = _short_answer_header(line, label_pattern)
            if bounds is not None:
                start, end = bounds
                offsets.append((offset, offset + start, offset + end if end is not None else None))
            continue
        header = re.match(r"^" + label_pattern + r":[ \t]*", line, re.I)
        if header:
            offsets.append((offset, offset + header.end(), None))
    return tuple(
        text[
            start : end
            if end is not None
            else offsets[index + 1][0]
            if index + 1 < len(offsets)
            else len(text)
        ]
        for index, (_, start, end) in enumerate(offsets)
    )


def _short_answer_fields(text: str) -> tuple[str, ...]:
    for label in (_FINAL_LABEL, r"Short answer", r"Answer"):
        fields = _final_fields(text, markdown=True, label_pattern=label)
        if fields:
            return fields
    return ()


def _declared_short_answer(fields: tuple[str, ...]) -> str | None:
    values = {field.strip().splitlines()[0].strip() if field.strip() else None for field in fields}
    return next(iter(values)) if len(values) == 1 and None not in values else None


def _python_payload(payload: str) -> str | None:
    return project_terminal_candidate(TerminalMode.PYTHON_SOURCE, payload)


def _aime_field_value(payload: str) -> int | None:
    complete = parse_explicit_integer_payload(payload)
    if complete is not None:
        return complete
    values = set()
    while payload.strip():
        leading = re.match(rf"\s*({INTEGER_SCALAR}[ \t]*\.?)[ \t]*(?:\r?\n|$)", payload)
        if leading is None:
            if re.match(r"\s*(?:[0-9+-]|\\boxed|\b(?:or|and)\b)", payload, re.I):
                return None
            break
        values.add(parse_explicit_integer_payload(leading[1]))
        payload = payload[leading.end() :]
    return next(iter(values)) if len(values) == 1 and None not in values else None


def project_owner_final(mode: TerminalMode, text: str) -> str | None:
    if not text.strip():
        return None
    try:
        native = native_tool_call(text)
    except ValueError:
        return None
    if native is not None:
        if native.name != "submit_answer" or set(native.arguments) != {"answer"}:
            return None
        payload = native.arguments["answer"]
        if mode is TerminalMode.AIME_INTEGER:
            integer = parse_explicit_integer_payload(payload)
            return rf"\boxed{{{integer}}}" if integer is not None else None
        if mode is TerminalMode.PYTHON_SOURCE:
            return _python_payload(payload)
        if mode is TerminalMode.SHORT_ANSWER:
            fields = _short_answer_fields(payload)
            return _declared_short_answer(fields) if fields else payload.strip() or None
        return project_terminal_candidate(mode, payload)
    fields = (
        ()
        if mode is TerminalMode.NATURAL_LANGUAGE
        else _short_answer_fields(text)
        if mode is TerminalMode.SHORT_ANSWER
        else _final_fields(
            text,
            markdown=mode is TerminalMode.PYTHON_SOURCE,
        )
    )
    if fields and mode is TerminalMode.AIME_INTEGER:
        values = {_aime_field_value(payload) for payload in fields}
        integer = next(iter(values)) if len(values) == 1 and None not in values else None
        return rf"\boxed{{{integer}}}" if integer is not None else None
    if fields and mode is TerminalMode.SHORT_ANSWER:
        return _declared_short_answer(fields)
    if len(fields) > 1 and mode is not TerminalMode.PYTHON_SOURCE:
        return None
    if not fields:
        return (
            text.strip()
            if mode is TerminalMode.SHORT_ANSWER
            else project_terminal_candidate(mode, text)
        )
    payload = fields[-1]
    if mode is TerminalMode.PYTHON_SOURCE:
        return _python_payload(payload)
    return project_terminal_candidate(mode, payload)
