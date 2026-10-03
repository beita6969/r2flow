from __future__ import annotations

import io
import token
import tokenize
from collections.abc import Iterator


def python_literal_lines(text: str) -> set[int]:
    lines: set[int] = set()
    formatted_starts: list[int] = []
    try:
        for item in tokenize.generate_tokens(io.StringIO(text).readline):
            if item.type == token.STRING:
                lines.update(range(item.start[0] + 1, item.end[0] + 1))
            name = token.tok_name[item.type]
            if name in {"FSTRING_START", "TSTRING_START"}:
                formatted_starts.append(item.start[0])
            elif name in {"FSTRING_END", "TSTRING_END"} and formatted_starts:
                lines.update(range(formatted_starts.pop() + 1, item.end[0] + 1))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return lines


def protocol_lines(text: str) -> Iterator[tuple[int, str]]:
    literals = python_literal_lines(text)
    offset, fenced = 0, False
    for number, line in enumerate(text.splitlines(keepends=True), start=1):
        if number not in literals:
            if line.lstrip().startswith("```"):
                fenced = not fenced
            elif not fenced:
                yield offset, line
        offset += len(line)
