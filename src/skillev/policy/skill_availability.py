from __future__ import annotations

import json

from skillev.contracts import canonical_json

HEADER = "SKILLEV_SKILL_AVAILABILITY_V1 "


def wrap_skill_availability(*, before: str, after: str, from_turn: int) -> str:
    if type(from_turn) is not int or from_turn < 1:
        raise ValueError("skill availability requires a positive trajectory turn")
    return HEADER + canonical_json({"before": before, "after": after, "from_turn": from_turn})


def skill_available_from_turn(initial_text: str) -> int:
    if not initial_text.startswith(HEADER):
        return 1
    value = json.loads(initial_text[len(HEADER) :])
    turn = value["from_turn"]
    if type(turn) is not int or turn < 1:
        raise ValueError("invalid persisted skill availability turn")
    return turn


def context_for_turn(initial_text: str, turn: int) -> str:
    if not isinstance(initial_text, str):
        raise ValueError("initial context must be text")
    if not initial_text.startswith(HEADER):
        return initial_text
    if type(turn) is not int or turn < 1:
        raise ValueError("context selection requires a positive trajectory turn")
    value = json.loads(initial_text[len(HEADER) :])
    selected = value["after" if turn >= skill_available_from_turn(initial_text) else "before"]
    if not isinstance(selected, str) or not selected or selected.startswith(HEADER):
        raise ValueError("invalid persisted skill context")
    return selected
