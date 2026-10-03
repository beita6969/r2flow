from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


class RolloutProgress:
    def __init__(self, **identity: object) -> None:
        self._lock = threading.RLock()
        self._value: dict[str, Any] = {
            **identity,
            "stage": "waiting-resident-slot",
            "stage_started": time.perf_counter(),
            "turn_index": 0,
            "environment_command_count": 0,
            "repeated_command_count": 0,
            "unchanged_observation_count": 0,
            "environment_seconds": 0.0,
            "environment_errors": 0,
            "phases": [],
            "action_outcomes": {},
            "submission_outcomes": [],
            "stage_seconds": {},
        }
        self._phase: dict[str, Any] | None = None
        self._last_command: str | None = None
        self._last_observation: str | None = None

    def stage(self, stage: str, **values: object) -> None:
        with self._lock:
            now = time.perf_counter()
            old = self._value["stage"]
            totals = self._value["stage_seconds"]
            totals[old] = totals.get(old, 0.0) + now - self._value["stage_started"]
            self._value.update(stage=stage, stage_started=now, **values)

    @property
    def phase(self) -> str | None:
        with self._lock:
            return None if self._phase is None else str(self._phase["phase"])

    @property
    def task_domain(self) -> str | None:
        with self._lock:
            value = self._value.get("task_domain")
            return None if value is None else str(value)

    @property
    def request_priority(self) -> tuple[int, str | None]:
        with self._lock:
            return int(self._value.get("max_turns", 1)), self.phase

    @property
    def benchmark_episode_index(self) -> int | None:
        with self._lock:
            value = self._value.get("benchmark_episode_index")
            return value if type(value) is int else None

    def begin_phase(self, phase: str, input_tokens: int) -> None:
        with self._lock:
            self._phase = {
                "phase": phase,
                "turn_index": self._value["turn_index"],
                "started": time.perf_counter(),
                "input_tokens": input_tokens,
                "output_tokens": None,
                "finish_reason": None,
                "client_model_queue_seconds": None,
                "client_transport_queue_seconds": None,
                "client_response_seconds": None,
                "server_queue_seconds": None,
                "server_prefill_seconds": None,
                "server_decode_seconds": None,
                "server_decode_tokens_per_second": None,
                "server_e2e_seconds": None,
                "server_cached_tokens": None,
                "server_prefill_tokens": None,
                "serving_adapter_name": None,
                "serving_adapter_revision": None,
                "server_generated_tokens": None,
                "server_request_id": None,
                "server_prefill_span_seconds": None,
                "server_after_prefill_span_seconds": None,
                "client_first_token_seconds": None,
            }
            self._value["phases"].append(self._phase)
            self.stage(phase + "-request")

    def phase_metrics(self, **values: object) -> None:
        with self._lock:
            if self._phase is not None:
                self._phase.update(values)

    def finish_phase(self, output_tokens: int, finish_reason: str) -> None:
        with self._lock:
            if self._phase is not None:
                phase = self._phase["phase"]
                self._phase.update(
                    response_received=True,
                    reasoning_completed=None,
                    client_minus_server_e2e_seconds=(
                        self._phase["client_response_seconds"] - self._phase["server_e2e_seconds"]
                        if isinstance(self._phase.get("client_response_seconds"), int | float)
                        and isinstance(self._phase.get("server_e2e_seconds"), int | float)
                        else None
                    ),
                    client_minus_server_interpretation=(
                        "unattributed span difference; not network or decode time"
                    ),
                    output_tokens=output_tokens,
                    finish_reason=finish_reason,
                    finished=time.perf_counter(),
                    elapsed_seconds=time.perf_counter() - self._phase["started"],
                )
                self._phase = None
                self.stage(phase + "-complete")

    @contextmanager
    def environment_command(self, command: str) -> Iterator[None]:
        started = time.perf_counter()
        with self._lock:
            self._value["environment_command_count"] += 1
            self._value["repeated_command_count"] += int(command == self._last_command)
            self._last_command = command
            self.stage("environment-ipc")
        completed = False
        try:
            yield
            completed = True
        finally:
            with self._lock:
                self._value["environment_seconds"] += time.perf_counter() - started
                self._value["environment_errors"] += int(not completed)

    def observation(self, value: object) -> None:
        text = json.dumps(value, sort_keys=True, ensure_ascii=False)
        with self._lock:
            self._value["unchanged_observation_count"] += int(text == self._last_observation)
            self._last_observation = text

    def action_outcome(self, categories: tuple[str, ...]) -> None:
        with self._lock:
            counts = self._value["action_outcomes"]
            for category in categories:
                counts[category] = counts.get(category, 0) + 1

    def submission_outcome(self, value: dict[str, object]) -> None:
        with self._lock:
            self._value["submission_outcomes"].append(
                {"turn_index": self._value["turn_index"], **value}
            )

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                **self._value,
                "phases": [dict(p) for p in self._value["phases"]],
                "action_outcomes": dict(self._value["action_outcomes"]),
                "submission_outcomes": [dict(row) for row in self._value["submission_outcomes"]],
                "stage_seconds": dict(self._value["stage_seconds"]),
                "phase_summary": phase_summary(self._value["phases"]),
                "stage_age_seconds": max(0.0, time.perf_counter() - self._value["stage_started"]),
            }


_CURRENT: ContextVar[RolloutProgress | None] = ContextVar(
    "rollout_execution_progress", default=None
)


def current_progress() -> RolloutProgress | None:
    return _CURRENT.get()


@contextmanager
def bind_progress(progress: RolloutProgress) -> Iterator[None]:
    token = _CURRENT.set(progress)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def progress_stage(stage: str, **values: object) -> None:
    row = current_progress()
    if row is not None:
        row.stage(stage, **values)


def server_metrics(meta: object) -> dict[str, object]:
    values = meta if isinstance(meta, dict) else {}
    result: dict[str, object] = {}
    for target, source in (
        ("server_queue_seconds", "queue_time"),
        ("server_prefill_seconds", "prefill_time"),
        ("server_decode_seconds", "decode_time"),
        ("server_e2e_seconds", "e2e_latency"),
        ("server_cached_tokens", "cached_tokens"),
        ("server_prompt_tokens", "prompt_tokens"),
        ("server_generated_tokens", "completion_tokens"),
        ("server_decode_tokens_per_second", "decode_throughput"),
        ("server_first_forward_ts", "forward_entry_time"),
        ("server_prefill_finished_ts", "prefill_finished_time"),
        ("server_request_finished_ts", "request_finished_ts"),
    ):
        value = values.get(source)
        result[target] = (
            value
            if isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
            else None
        )
    result["server_request_id"] = values.get("id") if isinstance(values.get("id"), str) else None
    prompt, cached = result["server_prompt_tokens"], result["server_cached_tokens"]
    result["server_prefill_tokens"] = (
        prompt - cached
        if type(prompt) is int and type(cached) is int and 0 <= cached <= prompt
        else None
    )
    decode_seconds = result["server_decode_seconds"]
    if not isinstance(decode_seconds, int | float) or decode_seconds <= 0:
        result["server_decode_tokens_per_second"] = None
    if result["server_queue_seconds"] == 0 and not result["server_first_forward_ts"]:
        result["server_queue_seconds"] = None
    for target, start_key, end_key in (
        ("server_prefill_span_seconds", "server_first_forward_ts", "server_prefill_finished_ts"),
        (
            "server_after_prefill_span_seconds",
            "server_prefill_finished_ts",
            "server_request_finished_ts",
        ),
    ):
        start, end = result[start_key], result[end_key]
        result[target] = (
            end - start
            if isinstance(start, int | float) and isinstance(end, int | float) and 0 < start <= end
            else None
        )
    return result


def phase_summary(phases: list[dict[str, Any]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for phase in ("reasoning", "action"):
        rows = [p for p in phases if p["phase"] == phase]
        fields = (
            "input_tokens",
            "output_tokens",
            "elapsed_seconds",
            "client_model_queue_seconds",
            "client_transport_queue_seconds",
            "client_response_seconds",
            "server_queue_seconds",
            "server_prefill_seconds",
            "server_decode_seconds",
            "server_cached_tokens",
            "server_prefill_tokens",
            "server_prefill_span_seconds",
            "server_after_prefill_span_seconds",
        )
        totals = {}
        for name in fields:
            measured = [p[name] for p in rows if p.get(name) is not None]
            totals[name] = {
                "measured_requests": len(measured),
                "sum": sum(measured) if measured else None,
            }
        result[phase] = {"requests": len(rows), "measurements": totals}
    return result
