from __future__ import annotations

from skillev.scoring.r2flow_plan import R2FlowEdgePlan


def validate_sequence_limits(limits: tuple[int, ...]) -> None:
    if not isinstance(limits, tuple) or any(type(v) is not int or v < 0 for v in limits):
        raise ValueError("worker sequence limits must be nonnegative integer capacities")
    if limits and 0 not in limits:
        raise ValueError("at least one worker must accept unrestricted sequences")


def accepts_plan(limit: int, plan: R2FlowEdgePlan | None) -> bool:
    return limit == 0 or (plan is not None and plan.max_sequence_tokens <= limit)
