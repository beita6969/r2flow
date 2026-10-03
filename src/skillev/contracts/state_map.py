from __future__ import annotations

from collections.abc import Iterable
from typing import Final

SIGMA_TRACE_QUOTIENT: Final = "sigma-trace-quotient@3"
SKILL_CONSUMPTION_DEPENDENCY: Final = "environment-skill-consumption@1"
PB_IN_EDGE_SOFTMAX: Final = "pb-in-edge-softmax@2"

_ACT: Final = "act"
_SUBMIT: Final = "submit_answer"
_FREE_TEXT: Final = frozenset({"invoke_skill", "corpus_search"})
_PRODUCERS: Final = frozenset({"invoke_skill", "open_passage", "corpus_search"})


def always_dependent(first: str, second: str) -> bool:
    if _SUBMIT in (first, second) or first == second == _ACT:
        return True
    if (first in _FREE_TEXT and second in _PRODUCERS) or (
        second in _FREE_TEXT and first in _PRODUCERS
    ):
        return True
    return (first in _FREE_TEXT and second == _ACT) or (second in _FREE_TEXT and first == _ACT)


def commuting_free_text_pairs(functions: Iterable[str]) -> tuple[tuple[str, str], ...]:
    names = sorted(set(functions))
    return tuple(
        (first, second)
        for first in names
        if first in _FREE_TEXT
        for second in names
        if not always_dependent(first, second)
    )


def chain_functions(functions: Iterable[str]) -> bool:
    names = sorted(set(functions))
    return all(
        always_dependent(first, second)
        for index, first in enumerate(names)
        for second in names[index:]
    )


__all__ = [
    "PB_IN_EDGE_SOFTMAX",
    "SIGMA_TRACE_QUOTIENT",
    "SKILL_CONSUMPTION_DEPENDENCY",
    "always_dependent",
    "chain_functions",
    "commuting_free_text_pairs",
]
