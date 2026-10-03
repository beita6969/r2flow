from __future__ import annotations

from collections.abc import Collection
from typing import Final

RSI_AMENDMENTS: Final = (
    "author-material=solver-evidence@1",
    "author-memory=previous-drafts@2",
    "author-prompt=answer-writer@2",
    "author-prompt=concrete-procedure@1",
    "empty-slot-bootstrap@1",
    "empty-slot-bootstrap@2",
    "gate-improvement-diagnostic=family-posterior@1",
    "generate-adequacy=context-class@1",
    "generate-failure-mass=verifier@1",
    "generate-support=cluster-neff@1",
    "pairwise-redundancy@2",
    "residual-weighted-rollout-share@1",
    "sole-server-of-a-context-class@1",
    "split-support-every-context-nmin@1",
    "tost-cost=paired-log-ratio@1",
    "validation-latency=agent-time@1",
    "validation-rollout=changed-families-only@1",
    "validation-scope=affected-families@2",
    "validation-tokens=policy-generated@1",
    "verification-budget=flow-ranked@1",
    "verifier-evidence-floor@1",
)
LOG_COST_MARGIN_KEYS: Final = (
    "latency_log_ratio",
    "success_abs",
    "tempered_reward_abs",
    "tokens_log_ratio",
)


def validate(rules: Collection[str]) -> tuple[str, ...]:
    ordered = tuple(sorted(set(rules)))
    if ordered != RSI_AMENDMENTS:
        raise ValueError(f"rsi_amendments must declare exactly {list(RSI_AMENDMENTS)}")
    return ordered


__all__ = ["LOG_COST_MARGIN_KEYS", "RSI_AMENDMENTS", "validate"]
