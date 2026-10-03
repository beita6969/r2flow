from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from skillev.policy.event_grammar import EventGrammarSpec

QWEN35_EVENT_STOP_TOKEN_ID = 248044


class ReplayedActionMasks(Protocol):
    @property
    def forced(self) -> tuple[bool, ...]: ...

    @property
    def budget_forced_positions(self) -> tuple[int, ...]: ...

    @property
    def digest(self) -> str: ...


class EventGrammarRuntime(Protocol):
    @property
    def stop_token_id(self) -> int: ...

    def grammar_key(self, spec: EventGrammarSpec) -> str: ...

    def replay(
        self, key: str, action_ids: Sequence[int], stop_ids: Sequence[int]
    ) -> ReplayedActionMasks: ...


@dataclass(frozen=True, slots=True)
class XGrammarEventRuntime:
    encode: Callable[[str], Sequence[int]]
    tokenizer_info: Any
    stop_token_id: int = QWEN35_EVENT_STOP_TOKEN_ID

    def grammar_key(self, spec: EventGrammarSpec) -> str:
        from skillev.policy.event_closing import tokenizer_info_sha256
        from skillev.policy.event_grammar import build_token_plan, event_grammar_key

        return event_grammar_key(
            spec, build_token_plan(spec, self.encode), tokenizer_info_sha256(self.tokenizer_info)
        )

    def replay(
        self, key: str, action_ids: Sequence[int], stop_ids: Sequence[int]
    ) -> ReplayedActionMasks:
        from skillev.policy.event_closing import replay_action_masks

        return replay_action_masks(key, action_ids, stop_ids, self.tokenizer_info)


__all__ = [
    "QWEN35_EVENT_STOP_TOKEN_ID",
    "EventGrammarRuntime",
    "ReplayedActionMasks",
    "XGrammarEventRuntime",
]
