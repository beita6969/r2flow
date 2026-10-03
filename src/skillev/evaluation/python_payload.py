from __future__ import annotations

import re

from .response_syntax import python_literal_lines


def decode_python_module(text: str) -> str | None:
    payload = text.strip("\r\n")
    lines = payload.splitlines(keepends=True)
    literal_lines = python_literal_lines(payload)
    fences = []
    for index, line in enumerate(lines):
        if index + 1 in literal_lines:
            continue
        match = re.fullmatch(r"[ \t]*(`{3,})([^`\r\n]*)[ \t]*", line.rstrip("\r\n"))
        if match:
            label = match[2].strip().lower()
            fences.append((index, match[1], label))
    if len(fences) >= 2:
        if len(fences) % 2 or any(label for _, _, label in fences[1::2]):
            return None
        source = ""
        for opening, closing in zip(fences[::2], fences[1::2], strict=True):
            start, marker, _ = opening
            end, end_marker, _ = closing
            if marker != end_marker:
                return None
            source = "".join(lines[start + 1 : end]).strip("\r\n")
        return source if source.strip() else None
    if fences:
        index, _, label = fences[0]
        if index == 0:
            lines = lines[1:]
        elif index == len(lines) - 1 and not label:
            lines = lines[:-1]
        else:
            return None
    return "".join(lines).strip("\r\n") or None
