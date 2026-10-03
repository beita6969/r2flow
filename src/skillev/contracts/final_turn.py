from __future__ import annotations

from typing import Final

FINAL_TURN_SUBMIT: Final = "final-turn-submit@1"
FINAL_TURN_COMPLETION_RULES: Final = frozenset({FINAL_TURN_SUBMIT})
COMPLETION_FUNCTION: Final = "submit_answer"
EXPLICIT_COMPLETION_TERMINAL_MODE: Final = "explicit-completion"

__all__ = [
    "COMPLETION_FUNCTION",
    "EXPLICIT_COMPLETION_TERMINAL_MODE",
    "FINAL_TURN_COMPLETION_RULES",
    "FINAL_TURN_SUBMIT",
]
