from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from typing import Final

REASONING_CALL_LINE: Final = "reasoning-call-line@1"
REASONING_CALL_LINE_RULES: Final = frozenset({REASONING_CALL_LINE})
REASONING_STOP_AT_TOOL_CALL: Final = "reasoning-stop-at-tool-call@1"
REASONING_STOP_VERSIONS: Final = frozenset({REASONING_STOP_AT_TOOL_CALL})
TOOL_CALL_OPEN: Final = "<tool_call>"
TOOL_CALL_CLOSE: Final = "</tool_call>"
FUNCTION_OPEN: Final = "<function="
FUNCTION_CLOSE: Final = "</function>"
PARAMETER_CLOSE: Final = "</parameter>"
REASONING_STOP_TEXT: Final = TOOL_CALL_OPEN
REASONING_CALL_LINE_TASK_SENTENCE: Final = (
    "Only the call written in the action phase is executed, in the format that phase gives; "
    "the reasoning names its chosen call in one plain final line. The action-phase reply is "
    "that one call and nothing else; JSON objects, code fences and described calls are not "
    "executed."
)
REASONING_CALL_LINE_INSTRUCTION: Final = (
    "The action phase writes the call itself; do not write the call in the action-phase "
    "format while reasoning. End your reasoning with one plain line that names the call you "
    "choose, "
)
ENVIRONMENT_ACT_FUNCTION: Final = "act"
ENVIRONMENT_COMMAND_FIELD: Final = "command"
NEXT_COMMAND: Final = "Next command:"
NEXT_CALL: Final = "Next call:"
CHOSEN_COMMAND: Final = "Chosen command:"
CHOSEN_CALL: Final = "Chosen call:"
UNFINISHED: Final = " (unfinished)"
EXACT_COMMAND_PLACEHOLDER: Final = "<exact admissible command>"

DraftedCall = tuple[str, tuple[tuple[str, str], ...]]

_OPEN_TAG = re.compile(re.escape(TOOL_CALL_OPEN))
_FUNCTION_START = re.compile(r"\s*<function=([^<>\s]+)>")
_FUNCTION_TAG = re.compile(r"<function=([^<>\s]*)>?")
_OPEN_PARAMETER = re.compile(r"<parameter=([^<>\s]+)>(.*?)(?=</parameter>|<parameter=|\Z)", re.S)
_CALL_TAG = re.compile(
    r"</?tool_call>|<function=[^<>\s]*>?|</function>|<parameter=[^<>\s]*>?|</parameter>"
)
_TAG_HEADS = (
    "<tool_call>",
    "</tool_call>",
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
)


def plain_call(name: str, values: Sequence[tuple[str, str]]) -> str:
    if name == ENVIRONMENT_ACT_FUNCTION and [key for key, _ in values] == [
        ENVIRONMENT_COMMAND_FIELD
    ]:
        return values[0][1]
    return f"{name}(" + ", ".join(f"{key}={value}" for key, value in values) + ")"


def opens_call(rest: str) -> bool:
    head = rest.lstrip()
    return head.startswith(FUNCTION_OPEN) or FUNCTION_OPEN.startswith(head)


def call_openers(text: str) -> tuple[int, ...]:
    return tuple(
        match.start() for match in _OPEN_TAG.finditer(text) if opens_call(text[match.end() :])
    )


def _label(name: str | None, *, unfinished: bool) -> str:
    base = CHOSEN_COMMAND if name == ENVIRONMENT_ACT_FUNCTION else CHOSEN_CALL
    return base[:-1] + UNFINISHED + ":" if unfinished else base


def _framed(value: str) -> str:
    value = value[1:] if value.startswith("\n") else value
    return value[:-1] if value.endswith("\n") else value


def _flatten(text: str) -> str:
    named = _FUNCTION_TAG.sub(lambda match: f" {match.group(1)} ", text)
    return " ".join(_CALL_TAG.sub(" ", named).split())


def _without_partial_tag(fragment: str) -> str:
    start = fragment.rfind("<")
    if start != -1 and any(tag.startswith(fragment[start:]) for tag in _TAG_HEADS):
        return fragment[:start]
    return fragment


def _segments(text: str) -> Iterator[tuple[str, bool | None]]:
    openers = call_openers(text)
    position = 0
    for index, start in enumerate(openers):
        yield text[position:start], None
        body = start + len(TOOL_CALL_OPEN)
        limit = openers[index + 1] if index + 1 < len(openers) else len(text)
        close = text.find(TOOL_CALL_CLOSE, body, limit)
        if close == -1:
            yield text[body:limit], False
            position = limit
        else:
            yield text[body:close], True
            position = close + len(TOOL_CALL_CLOSE)
    yield text[position:], None


def _parsed(
    body: str, *, closed: bool
) -> tuple[str | None, tuple[tuple[str, str], ...] | None, bool, str, str]:
    if not closed:
        body = _without_partial_tag(body)
    function = _FUNCTION_START.match(body)
    if function is None:
        return None, None, closed, _flatten(body), ""
    name, rest = function.group(1), body[function.end() :]
    end = rest.find(FUNCTION_CLOSE)
    last = rest.rfind(PARAMETER_CLOSE)
    after = rest[last + len(PARAMETER_CLOSE) :] if last != -1 else ""
    if end != -1:
        inner, tail, finished = rest[:end], rest[end + len(FUNCTION_CLOSE) :], True
    elif after.strip() and not after.lstrip().startswith("<"):
        inner, tail, finished = rest[: last + len(PARAMETER_CLOSE)], after, True
    else:
        inner, tail, finished = rest, "", closed
    tail = tail.rstrip() if closed else tail
    if _CALL_TAG.sub("", _OPEN_PARAMETER.sub("", inner)).strip():
        return name, None, finished, _flatten(body[: function.end()] + inner), tail
    values = tuple(
        (key, _framed(_CALL_TAG.sub("", value))) for key, value in _OPEN_PARAMETER.findall(inner)
    )
    return name, values, finished, "", tail


def _rendered(body: str, *, closed: bool) -> str:
    name, values, finished, words, tail = _parsed(body, closed=closed)
    if values is None or name is None:
        line = f"{_label(None, unfinished=not finished)} {words}".rstrip()
    else:
        line = f"{_label(name, unfinished=not finished)} {plain_call(name, values)}"
    return line + tail if tail.strip() else line


def plain_call_view(text: str) -> str:
    if TOOL_CALL_OPEN not in text and TOOL_CALL_CLOSE not in text:
        return text
    rendered = "".join(
        part if closed is None else _rendered(part, closed=closed)
        for part, closed in _segments(text)
    )
    while _CALL_TAG.search(rendered):
        rendered = _CALL_TAG.sub("", rendered)
    return rendered


def finished_calls(text: str) -> tuple[DraftedCall, ...]:
    calls: list[DraftedCall] = []
    for part, closed in _segments(text):
        if closed is None:
            continue
        name, values, finished, _, _ = _parsed(part, closed=closed)
        if finished and name is not None and values is not None:
            calls.append((name, values))
    return tuple(calls)


def draft_prose(text: str) -> str:
    parts = []
    for part, closed in _segments(text):
        if closed is None:
            parts.append(part)
        else:
            parts.append(_parsed(part, closed=closed)[4])
    prose = "".join(parts)
    while _CALL_TAG.search(prose):
        prose = _CALL_TAG.sub("", prose)
    return prose


__all__ = [
    "CHOSEN_CALL",
    "CHOSEN_COMMAND",
    "ENVIRONMENT_ACT_FUNCTION",
    "ENVIRONMENT_COMMAND_FIELD",
    "EXACT_COMMAND_PLACEHOLDER",
    "FUNCTION_OPEN",
    "NEXT_CALL",
    "NEXT_COMMAND",
    "REASONING_CALL_LINE",
    "REASONING_CALL_LINE_INSTRUCTION",
    "REASONING_CALL_LINE_RULES",
    "REASONING_CALL_LINE_TASK_SENTENCE",
    "REASONING_STOP_AT_TOOL_CALL",
    "REASONING_STOP_TEXT",
    "REASONING_STOP_VERSIONS",
    "TOOL_CALL_CLOSE",
    "TOOL_CALL_OPEN",
    "DraftedCall",
    "call_openers",
    "draft_prose",
    "finished_calls",
    "opens_call",
    "plain_call",
    "plain_call_view",
]
