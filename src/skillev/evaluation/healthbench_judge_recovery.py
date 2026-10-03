from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

HEALTHBENCH_JUDGE_RECOVERY: Final = "healthbench-judge-recovery@3"
SPOOL_DEADLINE_ERROR: Final = "SpoolDeadlineExceeded"
FAILURE_TRANSPORT: Final = "transport"
FAILURE_UNKNOWN_OUTCOME: Final = "unknown-outcome"
FAILURE_JUDGE: Final = "judge"
NO_CALL_FAILURE_TYPES: Final = frozenset(
    {"HealthBenchCapacityTimeoutError", "HealthBenchGradingClosedError", "RateLimitError"}
)
EXHAUSTION_TYPES: Final = frozenset({"HealthBenchTransportExhausted"})
UNKNOWN_OUTCOME_TYPES: Final = frozenset(
    {
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "TimeoutError",
        "ConnectionError",
    }
)
RELAY_BACKOFF_BASE_SECONDS: Final = 5.0
RELAY_BACKOFF_CAP_SECONDS: Final = 60.0


class HealthBenchCapacityTimeoutError(TimeoutError):
    pass


class HealthBenchGradingClosedError(RuntimeError):
    pass


class CallGate:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._closed = False
        self._in_flight = 0

    def admit(self) -> None:
        with self._condition:
            if self._closed:
                raise HealthBenchGradingClosedError("the grading failed; no further judge call")
            self._in_flight += 1

    def check(self) -> None:
        with self._condition:
            if self._closed:
                raise HealthBenchGradingClosedError("the grading failed; no further judge call")

    def leave(self) -> None:
        with self._condition:
            self._in_flight -= 1
            self._condition.notify_all()

    def close_and_drain(self, timeout: float) -> bool:
        with self._condition:
            self._closed = True
            return self._condition.wait_for(lambda: self._in_flight == 0, max(0.0, timeout))

    def idle(self) -> bool:
        with self._condition:
            return self._in_flight == 0


@dataclass(frozen=True, slots=True)
class JudgeRecovery:
    rule: str = HEALTHBENCH_JUDGE_RECOVERY
    spool_deadline_seconds: float = 900.0
    capacity_timeout_seconds: float = 900.0
    supersede_transport_incomplete: bool = True
    vq_rescore_delays_seconds: tuple[float, ...] = (60.0, 300.0, 900.0)
    relay_max_attempts: int = 3
    relay_default_timeout_seconds: float = 180.0
    withdrawal_margin_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.rule != HEALTHBENCH_JUDGE_RECOVERY:
            raise ValueError(f"unknown HealthBench judge recovery rule {self.rule!r}")
        if self.spool_deadline_seconds <= 0 or self.capacity_timeout_seconds <= 0:
            raise ValueError("judge recovery deadlines must be positive")
        if any(delay <= 0 for delay in self.vq_rescore_delays_seconds):
            raise ValueError("V_q re-scoring delays must be positive")
        if (
            type(self.relay_max_attempts) is not int
            or self.relay_max_attempts < 1
            or self.relay_default_timeout_seconds <= 0
            or self.withdrawal_margin_seconds < 0
        ):
            raise ValueError("the relay schedule of the withdrawal grace must be positive")

    def withdrawal_grace_seconds(self, timeout: object) -> float:
        per_attempt = (
            float(timeout)
            if isinstance(timeout, int | float) and not isinstance(timeout, bool) and timeout > 0
            else self.relay_default_timeout_seconds
        )
        backoffs = sum(
            min(RELAY_BACKOFF_CAP_SECONDS, RELAY_BACKOFF_BASE_SECONDS * 2.0 ** (attempt - 1))
            for attempt in range(1, self.relay_max_attempts)
        )
        return self.relay_max_attempts * per_attempt + backoffs + self.withdrawal_margin_seconds

    def drain_seconds(self, timeout: object) -> float:
        return (
            self.spool_deadline_seconds
            + self.withdrawal_grace_seconds(timeout)
            + self.withdrawal_margin_seconds
        )

    def to_value(self) -> dict[str, object]:
        return {
            "rule": self.rule,
            "spool_deadline_seconds": self.spool_deadline_seconds,
            "capacity_timeout_seconds": self.capacity_timeout_seconds,
            "supersede_transport_incomplete": self.supersede_transport_incomplete,
            "vq_rescore_delays_seconds": list(self.vq_rescore_delays_seconds),
            "withdrawal_grace": {
                "relay_max_attempts": self.relay_max_attempts,
                "relay_default_timeout_seconds": self.relay_default_timeout_seconds,
                "relay_backoff_seconds": "min(60, 5 * 2 ** (k - 1)) after attempt k",
                "margin_seconds": self.withdrawal_margin_seconds,
            },
            "failure_classes": {
                FAILURE_TRANSPORT: sorted(
                    [f"JudgeSpoolError:{SPOOL_DEADLINE_ERROR}", *NO_CALL_FAILURE_TYPES]
                )
                + [f"{name} (no unknown-outcome cause)" for name in sorted(EXHAUSTION_TYPES)],
                FAILURE_UNKNOWN_OUTCOME: sorted(
                    [
                        "JudgeSpoolError (any other relay-recorded error_type)",
                        *UNKNOWN_OUTCOME_TYPES,
                    ]
                ),
                FAILURE_JUDGE: "any other failure",
            },
            "replay": "recorded criterion responses of superseded ledgers and late spool "
            "responses of withdrawn requests; request equality without timeout",
            "quiescence": "a failed grading closes its call gate and drains every admitted call "
            "(spool_deadline + withdrawal_grace + margin) before its ledger is written; only a "
            "quiescent ledger whose failed rows are all transport is superseded",
            "grading_class": "worst of the propagated error, every failed or unsettled request "
            "row and the quiescence (unknown-outcome > judge > transport)",
        }


def judge_recovery(rule: str | None) -> JudgeRecovery | None:
    if rule is None:
        return None
    return JudgeRecovery(rule=rule)


def _chain(error: BaseException) -> list[BaseException]:
    links: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen and len(links) < 16:
        seen.add(id(current))
        links.append(current)
        current = current.__cause__ or current.__context__
    return links


def failure_chain(error: BaseException) -> list[str]:
    return [type(link).__name__ for link in _chain(error)]


def _link_class(link: BaseException) -> str | None:
    names = {cls.__name__ for cls in type(link).__mro__}
    if "JudgeSpoolError" in names:
        return (
            FAILURE_TRANSPORT
            if getattr(link, "error_type", None) == SPOOL_DEADLINE_ERROR
            else FAILURE_UNKNOWN_OUTCOME
        )
    if names & NO_CALL_FAILURE_TYPES or names & EXHAUSTION_TYPES:
        return FAILURE_TRANSPORT
    if names & UNKNOWN_OUTCOME_TYPES:
        return FAILURE_UNKNOWN_OUTCOME
    return None


def failure_class(error: BaseException) -> str:
    links = _chain(error)
    for link in links:
        stamped = getattr(link, "ledger_failure_class", None)
        if isinstance(stamped, str) and stamped in _SEVERITY:
            return stamped
    classes = {_link_class(link) for link in links}
    if FAILURE_UNKNOWN_OUTCOME in classes:
        return FAILURE_UNKNOWN_OUTCOME
    if FAILURE_TRANSPORT in classes:
        return FAILURE_TRANSPORT
    return FAILURE_JUDGE


def is_transport_failure(error: BaseException) -> bool:
    return failure_class(error) == FAILURE_TRANSPORT


_SEVERITY: Final = {FAILURE_TRANSPORT: 0, FAILURE_JUDGE: 1, FAILURE_UNKNOWN_OUTCOME: 2}


def row_failure_class(row: Mapping[str, object]) -> str | None:
    if "error_type" in row:
        recorded = row.get("failure_class")
        if isinstance(recorded, str) and recorded in _SEVERITY:
            return recorded
        return FAILURE_UNKNOWN_OUTCOME
    if isinstance(row.get("response"), dict):
        return None
    return FAILURE_UNKNOWN_OUTCOME


def grading_failure_class(
    error: BaseException, rows: Iterable[Mapping[str, object]], *, quiescent: bool
) -> str:
    classes = [failure_class(error)]
    classes.extend(kind for row in rows if (kind := row_failure_class(row)) is not None)
    if not quiescent:
        classes.append(FAILURE_UNKNOWN_OUTCOME)
    return max(classes, key=_SEVERITY.__getitem__)


def ledger_supersedable(state: Mapping[str, object]) -> bool:
    if (
        state.get("status") != "incomplete-grading"
        or state.get("transport_failure") is not True
        or state.get("failure_class") != FAILURE_TRANSPORT
        or state.get("quiescent") is not True
    ):
        return False
    rows = state.get("requests")
    return isinstance(rows, list) and all(
        isinstance(row, dict) and row_failure_class(row) in {None, FAILURE_TRANSPORT}
        for row in rows
    )


__all__ = [
    "FAILURE_JUDGE",
    "FAILURE_TRANSPORT",
    "FAILURE_UNKNOWN_OUTCOME",
    "HEALTHBENCH_JUDGE_RECOVERY",
    "NO_CALL_FAILURE_TYPES",
    "SPOOL_DEADLINE_ERROR",
    "UNKNOWN_OUTCOME_TYPES",
    "CallGate",
    "HealthBenchCapacityTimeoutError",
    "HealthBenchGradingClosedError",
    "JudgeRecovery",
    "failure_chain",
    "failure_class",
    "grading_failure_class",
    "is_transport_failure",
    "judge_recovery",
    "ledger_supersedable",
    "row_failure_class",
]
