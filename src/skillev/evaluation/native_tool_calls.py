from __future__ import annotations

import re
from dataclasses import dataclass

from .response_syntax import protocol_lines


_BRACKET_COMMAND = r"(click|search)\[([^\[\]\r\n<]+)\]"
_LITERAL_FUNCTION = r"(?:[A-Za-z_][A-Za-z0-9_]* [^<>\r\n]+|(?:click|search)\[[^\[\]\r\n<]+\])"
_ENVELOPE_END = r"</function>\s*</tool_call>\s*(?:<\|im_end\|>\s*)?"


@dataclass(frozen=True, slots=True)
class NativeToolCall:
    name: str
    arguments: dict[str, str]


def _unframe(value: str) -> str:
    value = value[2:] if value.startswith("\r\n") else value.removeprefix("\n")
    return value[:-2] if value.endswith("\r\n") else value.removesuffix("\n")


def _agrees_with_final(payload: str, repeated: list[str]) -> bool:
    declared = [
        match[1].strip()
        for _, line in protocol_lines(payload)
        if (match := re.fullmatch(r"Final(?: answer)?:[ \t]*(.+?)\s*", line, re.I))
    ]
    values = declared or [payload.strip()]
    return all(value == repeated[0] for value in (*values, *repeated))


def _tool_starts(text: str) -> list[int]:
    return [
        offset + len(line) - len(line.lstrip())
        for offset, line in protocol_lines(text)
        if line.lstrip().startswith("<tool_call>")
    ]


def native_tool_call(text: str) -> NativeToolCall | None:
    starts = _tool_starts(text)
    if not starts:
        return None
    if len(starts) != 1:
        raise ValueError("one native tool call is required per owner decision")
    prefix = text[: starts[0]].strip().splitlines()
    if prefix and re.match(r"(?:(?:for\s+)?examples?\b|do\s+not\b|don't\b)", prefix[-1], re.I):
        raise ValueError("a quoted or negated tool call is not a submission")
    repeated_finals: list[str] = []
    for _, line in protocol_lines(text[: starts[0]]):
        if final := re.fullmatch(r"Final(?: answer)?:[ \t]*(.+?)\s*", line, re.I):
            repeated_finals.append(final[1].strip())
            continue
        if re.match(
            r"(?:Action|Final(?: answer| response| code)?|Terminal payload|Message to \w+|Review):"
            r"|(?:\w+\.)?(?:click|search|act|点击|搜索)\s*[\[(]"
            r'|\{\s*"kind"\s*:\s*"(?:tool|message|history)"',
            line,
            re.I,
        ):
            raise ValueError("multiple explicit submission channels")
    envelope = re.fullmatch(
        rf"<tool_call>\s*<function=([A-Za-z_][A-Za-z0-9_]*|点击|搜索|{_LITERAL_FUNCTION})>"
        r"(.*?)" + _ENVELOPE_END,
        text[starts[0] :],
        re.S,
    )
    if envelope is None:
        signature = re.fullmatch(
            r"<tool_call>\s*<function=act\(command\)?>\r?\n"
            r"(.*?)</parameter>\s*" + _ENVELOPE_END,
            text[starts[0] :],
            re.S,
        )
        literal = re.fullmatch(
            rf"<tool_call>\s*<function=({_LITERAL_FUNCTION})\r?\n"
            r"\s*(?:</parameter>\s*)?" + _ENVELOPE_END,
            text[starts[0] :],
        )
        if signature is not None:
            function_name, rest = "act", "<parameter=command>" + signature[1] + "</parameter>"
        elif literal is None:
            raise ValueError("malformed native tool envelope")
        else:
            function_name, rest = literal[1], ""
    else:
        function_name, rest = envelope[1], envelope[2]
    arguments: dict[str, str] = {}
    while rest.strip():
        parameter = re.match(
            r"\s*<parameter=([A-Za-z_][A-Za-z0-9_]*)>(.*?)</parameter>", rest, re.S
        )
        if parameter is None or parameter[1] in arguments:
            raise ValueError("malformed or duplicate native parameter")
        value = _unframe(parameter[2])
        if re.search(r"</?(?:tool_call|function|parameter)(?:[=>])", value):
            raise ValueError("ambiguous nested native envelope")
        arguments[parameter[1]] = value
        rest = rest[parameter.end() :]
    if repeated_finals and not (
        function_name == "submit_answer"
        and set(arguments) == {"answer"}
        and _agrees_with_final(arguments["answer"], repeated_finals)
    ):
        raise ValueError("explicit final and native submission disagree")
    bracket_command = re.fullmatch(_BRACKET_COMMAND, function_name)
    if bracket_command is not None:
        if arguments:
            raise ValueError("a literal command cannot also supply parameters")
        name, argument = bracket_command.groups()
        return NativeToolCall(name, {"target" if name == "click" else "query": argument})
    name = {"点击": "click", "搜索": "search"}.get(function_name, function_name)
    return NativeToolCall(name, arguments)
