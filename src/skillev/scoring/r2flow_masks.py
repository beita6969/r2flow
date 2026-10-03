from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

from skillev.policy.event_grammar import (
    EventGrammarSpec,
    event_grammar_sha256,
    parse_event_grammar_key,
)
from skillev.policy.token_mask import TokenMask


class ReplayedPlan(Protocol):
    @property
    def vocab_size(self) -> int: ...

    @property
    def forced(self) -> tuple[bool, ...]: ...

    @property
    def budget_forced_positions(self) -> tuple[int, ...]: ...

    @property
    def digest(self) -> str: ...

    def packed_rows(self) -> Any: ...


class EventMaskRuntime(Protocol):
    def grammar_key(self, spec: EventGrammarSpec) -> str: ...

    def replay(self, key: str, action_ids: Sequence[int], stop_ids: Sequence[int]) -> Any: ...


@dataclass(frozen=True, slots=True)
class ReplayedTokenMask:
    mask: TokenMask
    digest: str
    forced_count: int
    budget_forced_positions: tuple[int, ...]


class ArtifactActionMasks:
    def __init__(self, action_grammars: Mapping[str, str], runtime: EventMaskRuntime) -> None:
        self.action_grammars = dict(action_grammars)
        self.runtime = runtime

    def _key(self, constraint_hash: str) -> str:
        key = self.action_grammars.get(constraint_hash)
        if key is None:
            raise ValueError("the artifact has no action grammar for this constraint hash")
        if event_grammar_sha256(key) != constraint_hash:
            raise ValueError("action grammar key text differs from its recorded sha256")
        return key

    def replay(self, *, constraint_hash: str, action_token_ids: Sequence[int]) -> ReplayedTokenMask:
        key = self._key(constraint_hash)
        ids = tuple(int(token) for token in action_token_ids)
        stop = parse_event_grammar_key(key).stop_token_id
        if len(ids) < 2 or ids[-1] != stop:
            raise ValueError("scored event ids must be the content ids plus the grammar stop")
        plan = self.runtime.replay(key, ids[:-1], ids[-1:])
        rows = np.ascontiguousarray(plan.packed_rows(), dtype=np.int32)
        if rows.shape[0] != len(ids):
            raise ValueError("replayed mask rows differ from the scored event length")
        mask = TokenMask(constraint_hash, int(plan.vocab_size), torch.from_numpy(rows.copy()))
        mask.require_targets(ids)
        return ReplayedTokenMask(
            mask,
            str(plan.digest),
            sum(1 for flag in plan.forced if flag),
            tuple(plan.budget_forced_positions),
        )

    def replay_action_mask(
        self, *, constraint_hash: str, action_token_ids: Sequence[int]
    ) -> TokenMask:
        return self.replay(constraint_hash=constraint_hash, action_token_ids=action_token_ids).mask

    def grammar_sha_for(self, spec: EventGrammarSpec) -> str:
        return event_grammar_sha256(self.runtime.grammar_key(spec))


_RUNTIMES: dict[str, EventMaskRuntime] = {}


def scorer_event_runtime(tokenizer: Any) -> EventMaskRuntime:
    identity = str(tokenizer.tokenizer_id)
    runtime = _RUNTIMES.get(identity)
    if runtime is None:
        from skillev.policy.event_closing import formal_tokenizer_info
        from skillev.rollout.event_grammar_runtime import XGrammarEventRuntime

        hf_tokenizer = getattr(tokenizer, "hf_tokenizer", None)
        if hf_tokenizer is None:
            raise TypeError("R2 Flow mask replay needs a Hugging Face tokenizer")
        runtime = XGrammarEventRuntime(tokenizer.encode, formal_tokenizer_info(hf_tokenizer))
        _RUNTIMES[identity] = runtime
    return runtime


def artifact_action_masks(
    backbone: Any, action_grammars: Mapping[str, str] | None
) -> ArtifactActionMasks:
    if action_grammars is None:
        raise ValueError("R2 Flow scoring requires a sigma-mode artifact with action_grammars")
    runtime = getattr(backbone, "event_mask_runtime", None) or scorer_event_runtime(
        backbone.tokenizer
    )
    return ArtifactActionMasks(action_grammars, runtime)


__all__ = [
    "ArtifactActionMasks",
    "EventMaskRuntime",
    "ReplayedTokenMask",
    "artifact_action_masks",
    "scorer_event_runtime",
]
