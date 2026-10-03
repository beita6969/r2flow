from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any


class IncompleteNativeGradingError(RuntimeError):
    def __init__(self, scorer_cost: dict[str, float]) -> None:
        super().__init__("Native rubric grading did not finish")
        self.scorer_cost = dict(scorer_cost)


RECORDED_REQUEST_KEYS = (
    "model",
    "messages",
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "reasoning_effort",
    "top_p",
    "extra_body",
    "response_format",
    "timeout",
    "stream",
    "store",
)


def recorded_request(kwargs: Mapping[str, object]) -> dict[str, Any]:
    return deepcopy({key: kwargs[key] for key in RECORDED_REQUEST_KEYS if key in kwargs})


def response_evidence(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    return {
        "content": getattr(getattr(response.choices[0], "message", None), "content", None),
        "finish_reason": getattr(response.choices[0], "finish_reason", None),
        "input_tokens": None if usage is None else usage.prompt_tokens,
        "output_tokens": None if usage is None else usage.completion_tokens,
        "request_id": getattr(response, "_request_id", None),
        "response_id": getattr(response, "id", None),
        "model": getattr(response, "model", None),
        "raw_response": response.model_dump(mode="json")
        if callable(getattr(response, "model_dump", None))
        else None,
    }


class MeteredCompletions:
    def __init__(
        self,
        delegate: Any,
        *,
        retain_evidence: bool = False,
        record_evidence: Callable[[list[dict[str, Any]]], None] | None = None,
        replay: Any = None,
        record_spool_request_ids: bool = False,
        gate: Any = None,
        failure_classifier: Callable[[BaseException], str] | None = None,
    ) -> None:
        self.delegate = delegate
        self._gate = gate
        self._classify = failure_classifier
        self._lock = threading.Lock()
        self._attempts = self._completed = self._missing_usage = 0
        self._input_tokens = self._output_tokens = 0
        self._replayed = 0
        self._replay = replay
        self._spool_ids = record_spool_request_ids
        self._retain_evidence = retain_evidence or record_evidence is not None
        self._record_evidence = record_evidence
        self._criterion_messages: list[list[dict[str, str]]] | None = None
        self._evidence: list[dict[str, Any]] = []

    def _criterion_indices(self, kwargs: Mapping[str, object]) -> list[int] | None:
        if self._criterion_messages is None:
            return None
        return [
            index
            for index, messages in enumerate(self._criterion_messages)
            if kwargs.get("messages") == messages
        ]

    def _replayed_response(self, kwargs: Mapping[str, object]) -> Any:
        hit = None if self._replay is None else self._replay.take(kwargs)
        if hit is None:
            return None
        response, recorded, source = hit
        with self._lock:
            self._replayed += 1
            if self._retain_evidence:
                self._evidence.append(
                    {
                        "attempt": None,
                        "replayed_from": deepcopy(source),
                        "criterion_indices": self._criterion_indices(kwargs),
                        "request": recorded_request(kwargs),
                        "response": deepcopy(recorded),
                    }
                )
                self._publish()
        return response

    def create(self, **kwargs: object) -> Any:
        gate = self._gate
        if gate is None:
            return self._create(kwargs)
        gate.admit()
        try:
            return self._create(kwargs)
        finally:
            gate.leave()

    def _create(self, kwargs: dict[str, object]) -> Any:
        replayed = self._replayed_response(kwargs)
        if replayed is not None:
            return replayed
        started = time.monotonic()
        with self._lock:
            self._attempts += 1
            evidence: dict[str, Any] = (
                {
                    "attempt": self._attempts,
                    "elapsed_scope": "grader-adapter-call-including-capacity-wait",
                    "criterion_indices": self._criterion_indices(kwargs),
                    "request": recorded_request(kwargs),
                }
                if self._retain_evidence
                else {}
            )
            if self._retain_evidence:
                self._evidence.append(evidence)
                self._publish()
        try:
            response = self.delegate.create(**kwargs)
        except Exception as error:
            if self._retain_evidence:
                with self._lock:
                    evidence["error_type"] = type(error).__name__
                    if self._classify is not None:
                        evidence["failure_class"] = self._classify(error)
                    if self._spool_ids:
                        from skillev.evaluation.judge_spool import JudgeSpoolError

                        if isinstance(error, JudgeSpoolError):
                            evidence["spool_request_id"] = error.request_id
                            evidence["spool_error_type"] = error.error_type
                    evidence["elapsed_seconds"] = time.monotonic() - started
                    rejected_response = getattr(error, "response", None)
                    if getattr(rejected_response, "choices", None):
                        self._response(evidence, rejected_response)
                    elif rejected_response is not None and callable(
                        getattr(rejected_response, "json", None)
                    ):
                        try:
                            body = rejected_response.json()
                        except ValueError:
                            body = getattr(rejected_response, "text", None)
                        evidence["error_response"] = {
                            "status_code": getattr(rejected_response, "status_code", None),
                            "body": body,
                        }
                    self._publish()
            raise
        with self._lock:
            self._response(evidence, response)
            if self._retain_evidence:
                evidence["elapsed_seconds"] = time.monotonic() - started
                self._publish()
        return response

    def bind_criterion_messages(self, messages: list[list[dict[str, str]]] | None) -> None:
        with self._lock:
            if self._attempts or self._replayed:
                raise ValueError("criterion binding must precede every grader request")
            self._criterion_messages = deepcopy(messages)

    def _publish(self) -> None:
        if self._record_evidence is not None:
            self._record_evidence(deepcopy(self._evidence))

    def _response(self, evidence: dict[str, Any], response: Any) -> None:
        usage = getattr(response, "usage", None)
        self._completed += 1
        if usage is None:
            self._missing_usage += 1
        else:
            self._input_tokens += usage.prompt_tokens
            self._output_tokens += usage.completion_tokens
        if self._retain_evidence:
            evidence["response"] = response_evidence(response)

    def evidence(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._evidence)

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return {
                "model_request_attempts": float(self._attempts),
                "model_responses": float(self._completed),
                "unknown_usage_calls": float(
                    self._attempts - self._completed + self._missing_usage
                ),
                "known_input_tokens": float(self._input_tokens),
                "known_output_tokens": float(self._output_tokens),
                **(
                    {"replayed_responses": float(self._replayed)}
                    if self._replay is not None
                    else {}
                ),
            }
