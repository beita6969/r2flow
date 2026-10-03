from __future__ import annotations

import asyncio
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from skillev.contracts import JsonValue
from skillev.evaluation.healthbench_judge_profile import VERIFIER, judge_profile
from skillev.training import AsyncResourceLimiter

from .healthbench_memo import (
    HealthBenchVerdictMemo,
    canonical_answer_text,
    memo_key,
    with_memo_reference,
)
from .native_backends import HealthBenchGrade

DEFAULT_API_CAPACITY: Final = 4
_CAPACITY_SLICE_SECONDS = 1.0


class JudgeCallCapacity:
    def __init__(self, limit: int = DEFAULT_API_CAPACITY) -> None:
        self._lock = threading.Lock()
        self._limit = _require_capacity(limit)
        self._sized = False
        self._in_flight = 0
        self._queue: deque[threading.Event] = deque()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def waiting(self) -> int:
        with self._lock:
            return len(self._queue)

    def size(self, limit: int) -> None:
        _require_capacity(limit)
        with self._lock:
            if self._sized and limit != self._limit:
                raise ValueError(
                    f"HealthBench judge capacity {limit} declared in a process already sized "
                    f"{self._limit}"
                )
            self._limit, self._sized = limit, True
            self._grant()

    def acquire(
        self,
        timeout: float | None = None,
        *,
        check: Callable[[], None] | None = None,
        check_every: float = _CAPACITY_SLICE_SECONDS,
    ) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        ticket = threading.Event()
        with self._lock:
            self._queue.append(ticket)
            self._grant()
        try:
            while not ticket.is_set():
                if check is not None:
                    check()
                wait = None if check is None else check_every
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return self._withdraw(ticket)
                    wait = remaining if wait is None else min(wait, remaining)
                ticket.wait(wait)
        except BaseException:
            if self._withdraw(ticket):
                self.release()
            raise
        return True

    def release(self) -> None:
        with self._lock:
            if self._in_flight == 0:
                raise ValueError("HealthBench judge capacity released more often than acquired")
            self._in_flight -= 1
            self._grant()

    def _grant(self) -> None:
        while self._queue and self._in_flight < self._limit:
            self._in_flight += 1
            self._queue.popleft().set()

    def _withdraw(self, ticket: threading.Event) -> bool:
        with self._lock:
            if ticket.is_set():
                return True
            self._queue.remove(ticket)
            return False


def _require_capacity(limit: int) -> int:
    if type(limit) is not int or limit < 1:
        raise ValueError("HealthBench judge capacity must be a positive integer")
    return limit


_API_CAPACITY = JudgeCallCapacity()


def configure_api_capacity(limit: int) -> None:
    _API_CAPACITY.size(limit)


def make_client(*, recovery: Any = None) -> Any:
    from r2flow.evaluation.external_judge_api import make_external_judge_client

    return make_external_judge_client(recovery=recovery)


def _spool_root() -> Path | None:
    root = os.environ.get("SKILLEV_JUDGE_SPOOL_DIR")
    return Path(root) if root else None


class IncompleteHealthBenchResponseError(RuntimeError):
    def __init__(self, response: Any) -> None:
        super().__init__("HealthBench judge response was incomplete")
        self.response = response


class BoundedAPICompletions:
    def __init__(
        self,
        delegate: Any,
        *,
        capacity_timeout_seconds: float | None = None,
        gate: Any = None,
    ) -> None:
        self.delegate = delegate
        self.capacity_timeout_seconds = capacity_timeout_seconds
        self.gate = gate

    def _acquire(self, capacity: JudgeCallCapacity, wait: float | None) -> bool:
        gate = self.gate
        if gate is None:
            return capacity.acquire() if wait is None else capacity.acquire(timeout=wait)
        return capacity.acquire(timeout=wait, check=gate.check, check_every=_CAPACITY_SLICE_SECONDS)

    def create(self, **kwargs: Any) -> Any:
        from skillev.evaluation.healthbench_judge_recovery import HealthBenchCapacityTimeoutError
        from skillev.evaluation.judge_spool import JudgeSpoolClient

        capacity = _API_CAPACITY
        started = time.monotonic()
        timeout = float(kwargs.get("timeout", 120.0))
        durable_delivery = isinstance(self.delegate, JudgeSpoolClient)
        no_call: type[TimeoutError] = (
            TimeoutError
            if self.capacity_timeout_seconds is None
            else HealthBenchCapacityTimeoutError
        )
        if not durable_delivery:
            acquired = self._acquire(capacity, timeout)
        elif self.capacity_timeout_seconds is None:
            acquired = self._acquire(capacity, None)
        else:
            acquired = self._acquire(capacity, self.capacity_timeout_seconds)
            if not acquired:
                raise no_call("HealthBench API capacity wait deadline reached")
        if not acquired:
            raise no_call("HealthBench API queue deadline reached")
        try:
            if self.gate is not None:
                self.gate.check()
            remaining = timeout if durable_delivery else timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise no_call("HealthBench API request deadline reached")
            response = self.delegate.create(**{**kwargs, "timeout": remaining, "store": False})
            if getattr(response.choices[0], "finish_reason", None) in {"length", "content_filter"}:
                raise IncompleteHealthBenchResponseError(response)
            return response
        finally:
            capacity.release()


@dataclass(frozen=True, slots=True)
class OpenAIHealthBenchGrader:
    private_cases: Mapping[str, JsonValue]
    official_source: Path
    case_limiter: AsyncResourceLimiter
    verifier_version: str = VERIFIER
    criterion_ledger_root: Path | None = None
    verdict_memo: HealthBenchVerdictMemo | None = None
    judge_recovery: str | None = None

    async def grade(self, task_id: str, candidate_answer: str) -> HealthBenchGrade:
        from r2flow.benchmarks.healthbench_grading import grade_health as _grade_health
        from skillev.evaluation.healthbench_judge_recovery import (
            FAILURE_TRANSPORT,
            failure_chain,
            grading_failure_class,
            judge_recovery,
        )

        recovery = judge_recovery(self.judge_recovery)
        case = self.private_cases[task_id]
        if not isinstance(case, dict):
            raise ValueError("HealthBench private rubric case is unavailable")
        settings = {
            "official_source": str(self.official_source),
            "effective_profile": asdict(judge_profile()),
        }
        memo = self.verdict_memo
        if memo is not None:
            candidate_answer = canonical_answer_text(candidate_answer)

        def grade_once() -> HealthBenchGrade:
            from .healthbench_ledger import HealthBenchCriterionLedger

            ledger = (
                None
                if self.criterion_ledger_root is None
                else HealthBenchCriterionLedger(
                    self.criterion_ledger_root,
                    task_id,
                    {
                        "candidate_answer": candidate_answer,
                        "private_case": case,
                        "settings": settings,
                    },
                    supersede_transport_incomplete=recovery is not None
                    and recovery.supersede_transport_incomplete,
                    spool_root=_spool_root() if recovery is not None else None,
                )
            )
            replay = None if ledger is None else ledger.replay
            if ledger is not None and ledger.completed:
                saved = ledger.state["result"]
                return HealthBenchGrade(
                    saved["native_raw_score"],
                    saved["negative_criterion_count"],
                    saved["grader_cost"],
                    ledger.reference(),
                    saved.get("refused_count"),
                )
            diagnostics: dict[str, Any] = {}
            try:
                score, negative, cost = _grade_health(
                    candidate_answer,
                    case,
                    settings,
                    diagnostics=diagnostics,
                    record_requests=None if ledger is None else ledger.record_requests,
                    **({} if recovery is None else {"recovery": recovery}),
                    **({} if replay is None else {"replay": replay}),
                )
            except BaseException as error:
                kind = None
                if recovery is not None:
                    kind = grading_failure_class(
                        error,
                        diagnostics.get("requests") or [],
                        quiescent=diagnostics.get("quiescent", True) is True,
                    )
                    error.ledger_failure_class = kind
                if ledger is not None:
                    ledger.update(
                        status="incomplete-grading",
                        diagnostics=diagnostics,
                        error_type=type(error).__name__,
                        **(
                            {
                                "transport_failure": kind == FAILURE_TRANSPORT,
                                "failure_class": kind,
                                "failure_chain": failure_chain(error),
                                "quiescent": diagnostics.get("quiescent", True) is True,
                                **(
                                    {"requests": diagnostics["requests"]}
                                    if "requests" in diagnostics
                                    else {}
                                ),
                            }
                            if recovery is not None
                            else {}
                        ),
                        **({} if replay is None else {"replayed_criteria": replay.taken}),
                    )
                raise
            refused = diagnostics.get("refused_count")
            if ledger is not None:
                ledger.update(
                    status="completed",
                    diagnostics=diagnostics,
                    result={
                        "native_raw_score": score,
                        "negative_criterion_count": negative,
                        "learning_reward": min(1.0, max(0.0, score)),
                        "binary_success": score >= 0.60 and negative == 0,
                        "grader_cost": cost,
                        **({"refused_count": refused} if refused is not None else {}),
                    },
                    **({} if replay is None else {"replayed_criteria": replay.taken}),
                )
            return HealthBenchGrade(
                score, negative, cost, None if ledger is None else ledger.reference(), refused
            )

        def grade_memoised() -> HealthBenchGrade:
            assert memo is not None
            key = memo_key(case, candidate_answer, settings)
            with memo.key_lock(key):
                saved = memo.lookup(key)
                if saved is not None:
                    return with_memo_reference(saved, key, hit=True)
                return with_memo_reference(memo.store(key, grade_once()), key, hit=False)

        async with self.case_limiter.lease():
            task = asyncio.create_task(
                asyncio.to_thread(grade_once if memo is None else grade_memoised)
            )
            try:
                grade = await asyncio.shield(task)
            except asyncio.CancelledError:
                await asyncio.gather(task, return_exceptions=True)
                raise
        return grade


@dataclass(frozen=True, slots=True)
class RescoringHealthBenchGrader:
    delegate: Any
    delays_seconds: tuple[float, ...]
    sleep: Any = asyncio.sleep

    @property
    def verifier_version(self) -> str:
        return str(self.delegate.verifier_version)

    async def grade(self, task_id: str, candidate_answer: str) -> HealthBenchGrade:
        from dataclasses import replace

        from skillev.evaluation.healthbench_judge_recovery import is_transport_failure

        attempts = 0
        while True:
            try:
                grade: HealthBenchGrade = await self.delegate.grade(task_id, candidate_answer)
            except Exception as error:
                if attempts >= len(self.delays_seconds) or not is_transport_failure(error):
                    raise
                await self.sleep(self.delays_seconds[attempts])
                attempts += 1
                continue
            if not attempts:
                return grade
            cost = dict(grade.grader_cost or {})
            cost["vq_rescore_attempts"] = float(attempts)
            return replace(grade, grader_cost=cost)
