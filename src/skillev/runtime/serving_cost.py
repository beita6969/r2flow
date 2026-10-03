from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

FIXED_BENCHMARK = "fixed-benchmark"
PREFERRED_WORK_CONSERVING = "preferred-benchmark-work-conserving"
ACTOR_ROUTING_POLICIES = (FIXED_BENCHMARK, PREFERRED_WORK_CONSERVING)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


@dataclass(frozen=True, slots=True)
class EpisodeWork:
    benchmark: str | None
    turns: int
    input_tokens: int | None = None
    max_output_tokens: int | None = None
    phase: str | None = None


@dataclass(frozen=True, slots=True)
class PhaseObservation:
    input_tokens: int
    output_tokens: int
    response_seconds: float
    prefill_tokens: float | None
    prefill_span_seconds: float | None
    after_prefill_span_seconds: float | None
    queue_seconds: float | None


class ServingCostModel:
    def __init__(self) -> None:
        self._samples: dict[tuple[str, str, str, int], deque[PhaseObservation]] = defaultdict(
            lambda: deque(maxlen=32)
        )

    def observe(
        self,
        endpoint: str,
        benchmark: str | None,
        phase: str,
        *,
        input_tokens: int,
        output_tokens: int,
        response_seconds: float,
        metrics: Mapping[str, object],
    ) -> None:
        if not benchmark or _number(response_seconds) is None or response_seconds <= 0:
            return
        sample = PhaseObservation(
            input_tokens,
            output_tokens,
            response_seconds,
            _number(metrics.get("server_prefill_tokens")),
            _number(metrics.get("server_prefill_span_seconds")),
            _number(metrics.get("server_after_prefill_span_seconds")),
            _number(metrics.get("server_queue_seconds")),
        )
        self._samples[(endpoint, benchmark, phase, max(1, input_tokens).bit_length())].append(
            sample
        )

    def _phase_seconds(self, endpoint: str, work: EpisodeWork, phase: str) -> float | None:
        candidates = [
            (key, rows)
            for key, rows in self._samples.items()
            if key[1:3] == (work.benchmark, phase)
        ]
        local = [(key, rows) for key, rows in candidates if key[0] == endpoint]
        candidates = local or candidates
        if work.input_tokens is not None and candidates:
            bucket = max(1, work.input_tokens).bit_length()
            distance = min(abs(key[3] - bucket) for key, _ in candidates)
            candidates = [(k, rows) for k, rows in candidates if abs(k[3] - bucket) == distance]
        rows = [row for _, group in candidates for row in group]
        if not rows:
            return None
        measured = [
            row
            for row in rows
            if row.prefill_tokens is not None
            and row.prefill_tokens > 0
            and row.prefill_span_seconds is not None
            and row.after_prefill_span_seconds is not None
            and row.output_tokens > 0
            and row.queue_seconds is not None
        ]
        if not measured:
            return math.fsum(row.response_seconds for row in rows) / len(rows)
        uncached = math.fsum(cast(float, row.prefill_tokens) for row in measured)
        prompt = sum(row.input_tokens for row in measured)
        outputs = sum(row.output_tokens for row in measured)
        next_prefill = (
            work.input_tokens * min(1.0, uncached / prompt)
            if work.input_tokens is not None and prompt
            else uncached / len(measured)
        )
        next_output = outputs / len(measured)
        if work.phase == phase and work.max_output_tokens is not None:
            next_output = min(next_output, work.max_output_tokens)
        return (
            math.fsum(cast(float, row.queue_seconds) for row in measured) / len(measured)
            + next_prefill
            * math.fsum(cast(float, row.prefill_span_seconds) for row in measured)
            / uncached
            + next_output
            * math.fsum(cast(float, row.after_prefill_span_seconds) for row in measured)
            / outputs
        )

    def estimate(self, endpoint: str, work: EpisodeWork) -> float | None:
        reasoning = self._phase_seconds(endpoint, work, "reasoning")
        action = self._phase_seconds(endpoint, work, "action")
        if reasoning is None or action is None:
            return None
        return (reasoning + action) * work.turns - (reasoning if work.phase == "action" else 0.0)
