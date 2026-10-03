from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Final, Generic, TypeVar

from skillev.contracts import JsonValue, normalize_json
from skillev.diagnostics.rollout_progress import current_progress

from .request_scheduling import FAIR_MODEL_REQUESTS, FairRequestGate

ROLLOUT_WORKFLOW_BINDING_FORMAT: Final = "skillev-rollout-workflow-binding@2"
LONG_HORIZON_FIRST: Final = "long-horizon-first-stable@1"


def _positive(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class RolloutWorkflowBinding:
    max_resident_trajectories: int = 1
    max_inflight_model_requests: int = 1
    max_inflight_environment_calls: int = 1
    max_inflight_terminal_evaluations: int = 1
    max_inflight_process_graders: int = 1
    transport_worker_threads: int = 1

    def __post_init__(self) -> None:
        for name in (
            "max_resident_trajectories",
            "max_inflight_model_requests",
            "max_inflight_environment_calls",
            "max_inflight_terminal_evaluations",
            "max_inflight_process_graders",
            "transport_worker_threads",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), field_name=name))

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": ROLLOUT_WORKFLOW_BINDING_FORMAT,
            "max_inflight_environment_calls": self.max_inflight_environment_calls,
            "max_inflight_model_requests": self.max_inflight_model_requests,
            "max_inflight_process_graders": self.max_inflight_process_graders,
            "max_inflight_terminal_evaluations": self.max_inflight_terminal_evaluations,
            "max_resident_trajectories": self.max_resident_trajectories,
            "scheduling_policy": LONG_HORIZON_FIRST,
            "transport_worker_threads": self.transport_worker_threads,
            "request_scheduling_policy": FAIR_MODEL_REQUESTS,
        }

    @classmethod
    def from_value(cls, value: object) -> RolloutWorkflowBinding:
        normalized = normalize_json(value)
        if not isinstance(normalized, dict):
            raise TypeError("rollout workflow binding must be an object")
        expected = {
            "format",
            "max_inflight_environment_calls",
            "max_inflight_model_requests",
            "max_inflight_process_graders",
            "max_inflight_terminal_evaluations",
            "max_resident_trajectories",
            "scheduling_policy",
            "transport_worker_threads",
            "request_scheduling_policy",
        }
        if set(normalized) != expected:
            raise ValueError("rollout workflow binding has an incompatible field set")
        if (
            normalized["format"] != ROLLOUT_WORKFLOW_BINDING_FORMAT
            or normalized["scheduling_policy"] != LONG_HORIZON_FIRST
            or normalized["request_scheduling_policy"] != FAIR_MODEL_REQUESTS
        ):
            raise ValueError("unsupported rollout workflow binding")
        return cls(
            max_resident_trajectories=_positive(
                normalized["max_resident_trajectories"],
                field_name="max_resident_trajectories",
            ),
            max_inflight_model_requests=_positive(
                normalized["max_inflight_model_requests"],
                field_name="max_inflight_model_requests",
            ),
            max_inflight_environment_calls=_positive(
                normalized["max_inflight_environment_calls"],
                field_name="max_inflight_environment_calls",
            ),
            max_inflight_terminal_evaluations=_positive(
                normalized["max_inflight_terminal_evaluations"],
                field_name="max_inflight_terminal_evaluations",
            ),
            max_inflight_process_graders=_positive(
                normalized["max_inflight_process_graders"],
                field_name="max_inflight_process_graders",
            ),
            transport_worker_threads=_positive(
                normalized["transport_worker_threads"],
                field_name="transport_worker_threads",
            ),
        )


@dataclass(frozen=True, slots=True)
class ResourceTiming:
    calls: int
    high_water_mark: int
    queue_seconds: float
    service_seconds: float

    def __post_init__(self) -> None:
        if self.calls < 0 or self.high_water_mark < 0:
            raise ValueError("resource counters cannot be negative")
        if not math.isfinite(self.queue_seconds) or self.queue_seconds < 0:
            raise ValueError("resource queue time must be finite and non-negative")
        if not math.isfinite(self.service_seconds) or self.service_seconds < 0:
            raise ValueError("resource service time must be finite and non-negative")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "calls": self.calls,
            "high_water_mark": self.high_water_mark,
            "queue_seconds": self.queue_seconds,
            "service_seconds": self.service_seconds,
        }

    def subtract(self, earlier: ResourceTiming) -> ResourceTiming:
        if not isinstance(earlier, ResourceTiming):
            raise TypeError("resource timing baseline is incompatible")
        return ResourceTiming(
            calls=self.calls - earlier.calls,
            high_water_mark=self.high_water_mark,
            queue_seconds=self.queue_seconds - earlier.queue_seconds,
            service_seconds=self.service_seconds - earlier.service_seconds,
        )


@dataclass(frozen=True, slots=True)
class RolloutBatchPerformanceReport:
    batch_id: str
    batch_size: int
    binding: RolloutWorkflowBinding
    collection_seconds: float
    sealing_seconds: float
    model: ResourceTiming
    environment: ResourceTiming
    terminal_evaluator: ResourceTiming
    process_grader: ResourceTiming
    session_setup: ResourceTiming
    session_cleanup: ResourceTiming
    prompt_tokens: int
    completion_tokens: int
    model_calls: int
    trajectory_p50_seconds: float
    trajectory_p95_seconds: float
    trajectory_p99_seconds: float
    maximum_trajectory_seconds: float
    straggler_ratio: float
    reward_mean: float
    success_rate: float

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "batch_id": self.batch_id,
            "batch_size": self.batch_size,
            "binding": self.binding.to_value(),
            "collection_seconds": self.collection_seconds,
            "completion_tokens": self.completion_tokens,
            "environment": self.environment.to_value(),
            "maximum_trajectory_seconds": self.maximum_trajectory_seconds,
            "model": self.model.to_value(),
            "model_calls": self.model_calls,
            "process_grader": self.process_grader.to_value(),
            "prompt_tokens": self.prompt_tokens,
            "reward_mean": self.reward_mean,
            "sealing_seconds": self.sealing_seconds,
            "session_cleanup": self.session_cleanup.to_value(),
            "session_setup": self.session_setup.to_value(),
            "straggler_ratio": self.straggler_ratio,
            "success_rate": self.success_rate,
            "terminal_evaluator": self.terminal_evaluator.to_value(),
            "trajectory_p50_seconds": self.trajectory_p50_seconds,
            "trajectory_p95_seconds": self.trajectory_p95_seconds,
            "trajectory_p99_seconds": self.trajectory_p99_seconds,
        }


@dataclass(slots=True)
class _ResourceLease:
    limiter: AsyncResourceLimiter
    token_cost: int = 0
    role: str = "unspecified"
    _queued_at: float = field(default=0.0, init=False)
    _started_at: float = field(default=0.0, init=False)

    async def __aenter__(self) -> None:
        self._queued_at = time.perf_counter()
        self.limiter._waiting[id(self)] = self._queued_at
        row = current_progress()
        if row is not None:
            row.stage(f"{row.phase or 'rollout'}-{self.limiter.name}-queue")
        try:
            if isinstance(self.limiter._semaphore, FairRequestGate):
                await self.limiter._semaphore.acquire(self.token_cost)
            else:
                await self.limiter._semaphore.__aenter__()
        finally:
            self.limiter._waiting.pop(id(self), None)
        self._started_at = time.perf_counter()
        self.limiter._serving[id(self)] = self._started_at
        self.limiter._entered(self._started_at - self._queued_at)
        self.limiter._roles[self.role] = self.limiter._roles.get(self.role, 0) + 1
        if row is not None:
            row.stage(f"{row.phase or 'rollout'}-{self.limiter.name}-service")
            if self.limiter.name == "model":
                row.phase_metrics(client_model_queue_seconds=self._started_at - self._queued_at)

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        finished_at = time.perf_counter()
        self.limiter._serving.pop(id(self), None)
        self.limiter._exited(finished_at - self._started_at)
        if isinstance(self.limiter._semaphore, FairRequestGate):
            self.limiter._semaphore.release(self.token_cost)
        else:
            await self.limiter._semaphore.__aexit__(None, None, None)


@dataclass(slots=True)
class AsyncResourceLimiter:
    limit: int
    name: str = "resource"
    fair: bool = False
    token_capacity: int | None = None
    aggregate: AsyncResourceLimiter | None = field(default=None, repr=False)
    _roles: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _waiting: dict[int, float] = field(default_factory=dict, init=False, repr=False)
    _serving: dict[int, float] = field(default_factory=dict, init=False, repr=False)
    _semaphore: asyncio.Semaphore | FairRequestGate = field(init=False, repr=False)
    _active: int = field(default=0, init=False, repr=False)
    _calls: int = field(default=0, init=False, repr=False)
    _high_water_mark: int = field(default=0, init=False, repr=False)
    _queue_seconds: float = field(default=0.0, init=False, repr=False)
    _service_seconds: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.limit = _positive(self.limit, field_name="resource limit")
        self._semaphore = (
            FairRequestGate(self.limit, token_capacity=self.token_capacity)
            if self.fair or self.token_capacity is not None
            else asyncio.Semaphore(self.limit)
        )

    def lease(self, *, token_cost: int | None = None, role: str = "unspecified") -> _ResourceLease:
        cost = (self.token_capacity or 0) if token_cost is None else token_cost
        return _ResourceLease(self, cost, role)

    def _entered(self, queue_seconds: float) -> None:
        if self.aggregate is not None:
            self.aggregate._entered(queue_seconds)
        self._active += 1
        self._calls += 1
        self._queue_seconds += queue_seconds
        self._high_water_mark = max(self._high_water_mark, self._active)

    def _exited(self, service_seconds: float) -> None:
        if self.aggregate is not None:
            self.aggregate._exited(service_seconds)
        self._active -= 1
        self._service_seconds += service_seconds

    @property
    def timing(self) -> ResourceTiming:
        return ResourceTiming(
            calls=self._calls,
            high_water_mark=self._high_water_mark,
            queue_seconds=self._queue_seconds,
            service_seconds=self._service_seconds,
        )

    def active_snapshot(self) -> dict[str, object]:
        now = time.perf_counter()
        waiting, serving = tuple(self._waiting.values()), tuple(self._serving.values())
        return {
            "token_capacity": self.token_capacity,
            "active_reserved_tokens": (
                self._semaphore.active_tokens
                if isinstance(self._semaphore, FairRequestGate)
                else None
            ),
            "calls_by_role": dict(self._roles),
            "waiting": len(waiting),
            "serving": len(serving),
            "queue_ages_seconds": [max(0.0, now - t) for t in waiting],
            "service_ages_seconds": [max(0.0, now - t) for t in serving],
        }

    def reset_timing(self) -> None:
        if self._active or self._waiting:
            raise RuntimeError("cannot reset resource timing while calls are active")
        self._calls = 0
        self._roles.clear()
        self._high_water_mark = 0
        self._queue_seconds = 0.0
        self._service_seconds = 0.0


@dataclass(slots=True)
class RolloutWorkflowResources:
    binding: RolloutWorkflowBinding
    model_requests: AsyncResourceLimiter = field(init=False)
    _model_endpoints: dict[str, AsyncResourceLimiter] = field(default_factory=dict, init=False)
    environment_calls: AsyncResourceLimiter = field(init=False)
    terminal_evaluations: AsyncResourceLimiter = field(init=False)
    process_graders: AsyncResourceLimiter = field(init=False)
    session_setups: AsyncResourceLimiter = field(init=False)
    session_cleanups: AsyncResourceLimiter = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.binding, RolloutWorkflowBinding):
            raise TypeError("workflow resources require a rollout workflow binding")
        self.model_requests = AsyncResourceLimiter(
            self.binding.max_inflight_model_requests,
            name="model",
            fair=True,
        )
        self.environment_calls = AsyncResourceLimiter(
            self.binding.max_inflight_environment_calls, name="environment"
        )
        self.terminal_evaluations = AsyncResourceLimiter(
            self.binding.max_inflight_terminal_evaluations, name="terminal-evaluator"
        )
        self.process_graders = AsyncResourceLimiter(
            self.binding.max_inflight_process_graders, name="process-grader"
        )
        self.session_setups = AsyncResourceLimiter(
            self.binding.max_inflight_environment_calls, name="session-setup"
        )
        self.session_cleanups = AsyncResourceLimiter(
            self.binding.max_inflight_environment_calls, name="session-cleanup"
        )

    def configure_model_endpoint(
        self, endpoint: str, *, capacity: int, token_capacity: int | None = None
    ) -> None:
        key = endpoint.rstrip("/").removesuffix("/v1")
        if key in self._model_endpoints:
            raise ValueError("service capacity may only be configured once before collection")
        self._model_endpoints[key] = AsyncResourceLimiter(
            capacity,
            name="model",
            fair=True,
            token_capacity=token_capacity,
            aggregate=self.model_requests,
        )

    def model_limiter(self, endpoint: str) -> AsyncResourceLimiter:
        key = endpoint.rstrip("/").removesuffix("/v1")
        if not self._model_endpoints:
            return self.model_requests
        if key not in self._model_endpoints:
            raise ValueError("model endpoint has no declared service capacity")
        return self._model_endpoints[key]

    def active_snapshot(self) -> dict[str, object]:
        result = {
            name: getattr(self, name).active_snapshot()
            for name in (
                "model_requests",
                "environment_calls",
                "terminal_evaluations",
                "process_graders",
                "session_setups",
                "session_cleanups",
            )
        }
        result["model_services"] = {
            k: v.active_snapshot() for k, v in self._model_endpoints.items()
        }
        return result

    def begin_batch_window(self) -> None:
        for limiter in (
            *self._model_endpoints.values(),
            self.model_requests,
            self.environment_calls,
            self.terminal_evaluations,
            self.process_graders,
            self.session_setups,
            self.session_cleanups,
        ):
            limiter.reset_timing()


ItemT = TypeVar("ItemT")
ResultT = TypeVar("ResultT")


class RolloutBatchWorkflow(Generic[ItemT, ResultT]):
    def __init__(self, binding: RolloutWorkflowBinding) -> None:
        if not isinstance(binding, RolloutWorkflowBinding):
            raise TypeError("rollout workflow requires a binding")
        self._binding = binding

    async def run(
        self,
        items: tuple[ItemT, ...],
        execute: Callable[[ItemT], Awaitable[ResultT]],
        *,
        declared_horizons: tuple[int, ...] | None = None,
    ) -> tuple[ResultT, ...]:
        if not items:
            return ()
        order = list(range(len(items)))
        if declared_horizons is None or len(declared_horizons) != len(items):
            raise ValueError("long-horizon scheduling requires every declared task horizon")
        if any(type(h) is not int or h < 1 for h in declared_horizons):
            raise ValueError("declared task horizons must be positive integers")
        order.sort(key=lambda i: (-declared_horizons[i], i))
        slots: list[ResultT | None] = [None] * len(items)
        next_index = 0
        stop_launching = False
        failures: list[tuple[int, Exception]] = []
        scheduling_lock = asyncio.Lock()

        async def worker() -> None:
            nonlocal next_index, stop_launching
            while True:
                async with scheduling_lock:
                    if stop_launching or next_index >= len(items):
                        return
                    index = order[next_index]
                    next_index += 1
                try:
                    slots[index] = await execute(items[index])
                except (OSError, RuntimeError, TypeError, ValueError) as error:
                    async with scheduling_lock:
                        stop_launching = True
                        failures.append((index + 1, error))
                    return

        worker_count = min(self._binding.max_resident_trajectories, len(items))
        workers = tuple(asyncio.create_task(worker()) for _ in range(worker_count))
        outcomes = await asyncio.gather(*workers, return_exceptions=True)
        cancellation = next(
            (outcome for outcome in outcomes if isinstance(outcome, asyncio.CancelledError)),
            None,
        )
        if cancellation is not None:
            raise cancellation
        unexpected = next((item for item in outcomes if isinstance(item, Exception)), None)
        if unexpected is not None:
            raise unexpected
        if failures:
            _, cause = min(failures, key=lambda item: item[0])
            raise cause
        if any(slot is None for slot in slots):
            raise RuntimeError("rollout workflow completed with an empty planned slot")
        return tuple(slot for slot in slots if slot is not None)


__all__ = [
    "ROLLOUT_WORKFLOW_BINDING_FORMAT",
    "AsyncResourceLimiter",
    "ResourceTiming",
    "RolloutBatchPerformanceReport",
    "RolloutBatchWorkflow",
    "RolloutWorkflowBinding",
    "RolloutWorkflowResources",
]
