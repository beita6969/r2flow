from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .external_judge_policy import (
    EXTERNAL_JUDGE_MODEL,
    HEALTHBENCH_JUDGE_PROFILE,
    HEALTHBENCH_SEMANTIC_CALLS,
)
from .healthbench_transport import (
    ExternalJudgeTransport as ExternalJudgeTransport,
)
from .healthbench_transport import (
    HealthBenchTransportError as HealthBenchTransportError,
)
from .healthbench_transport import (
    HealthBenchTransportExhausted as HealthBenchTransportExhausted,
)
from .healthbench_transport import (
    TransportRetryPolicy as TransportRetryPolicy,
)

GRADER_INVOCATION = "official-grader-worker"
REFUSED_CRITERION_EXPLANATION = "skillev:rubric-item-refused-after-semantic-call-limit@1"


def semantic_messages_key(messages: Sequence[Mapping[str, str]]) -> str:
    return json.dumps(
        [{"role": m["role"], "content": m["content"]} for m in messages],
        ensure_ascii=False,
        sort_keys=True,
    )


@dataclass(frozen=True, slots=True)
class HealthBenchExternalJudgeProfile:
    profile_id: str
    backend: str
    model: str
    rubric_source_repository: str
    rubric_source_revision: str
    rubric_source_path: str
    endpoint_environment: str
    api_key_environment: str
    call_mode: str
    response_format: str
    max_completion_tokens: int
    reasoning_effort: str
    temperature: None
    top_p: None
    request_timeout_seconds: float
    maximum_attempts: int

    def __post_init__(self) -> None:
        required = (
            self.profile_id,
            self.model,
            self.rubric_source_repository,
            self.rubric_source_revision,
            self.rubric_source_path,
            self.endpoint_environment,
            self.api_key_environment,
        )
        if any(not value.strip() for value in required):
            raise ValueError("external HealthBench judge identity is incomplete")
        if (
            self.profile_id != HEALTHBENCH_JUDGE_PROFILE
            or self.backend != "openai-chat-completions"
            or self.call_mode != "per-rubric"
            or self.response_format != "json-object"
            or self.max_completion_tokens != 8000
            or self.reasoning_effort != "medium"
            or self.model != EXTERNAL_JUDGE_MODEL
            or self.maximum_attempts != 1
            or self.temperature is not None
            or self.top_p is not None
        ):
            raise ValueError("external HealthBench judge differs from its declared profile")
        if self.request_timeout_seconds <= 0:
            raise ValueError("external HealthBench judge retry policy is invalid")


@dataclass(slots=True)
class RubricAttemptRegistry:
    _attempts: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _refused: set[str] = field(default_factory=set)

    def calls(self, invocation_id: str, messages: list[dict[str, str]]) -> int:
        key = invocation_id + "\n" + json.dumps(messages, ensure_ascii=False, sort_keys=True)
        with self._lock:
            return self._attempts.get(key, 0)

    def mark_refused(self, messages: list[dict[str, str]]) -> None:
        with self._lock:
            self._refused.add(semantic_messages_key(messages))

    @property
    def refused_message_keys(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._refused)

    def begin_semantic_call(
        self, invocation_id: str, messages: list[dict[str, str]], *, maximum_calls: int
    ) -> int:
        if not invocation_id.strip():
            raise ValueError("HealthBench semantic invocation ID must be non-empty")
        key = invocation_id + "\n" + json.dumps(messages, ensure_ascii=False, sort_keys=True)
        with self._lock:
            attempt = self._attempts.get(key, 0) + 1
            if attempt > maximum_calls:
                raise RuntimeError("HealthBench rubric exceeded its declared request allowance")
            self._attempts[key] = attempt
        return attempt

    @property
    def semantic_repair_count(self) -> int:
        with self._lock:
            return sum(max(0, value - 1) for value in self._attempts.values())


@dataclass(slots=True)
class ExternalHealthBenchRubricSampler:
    transport: ExternalJudgeTransport
    response_type: type[Any]
    semantic_attempts: RubricAttemptRegistry = field(default_factory=RubricAttemptRegistry)

    def __call__(self, message_list: Sequence[Mapping[str, str]]) -> Any:
        messages = [{"role": item["role"], "content": item["content"]} for item in message_list]
        if self.semantic_attempts.calls(GRADER_INVOCATION, messages) >= HEALTHBENCH_SEMANTIC_CALLS:
            self.semantic_attempts.mark_refused(messages)
            return self.response_type(
                response_text=json.dumps(
                    {"criteria_met": False, "explanation": REFUSED_CRITERION_EXPLANATION}
                ),
                response_metadata={
                    "usage": None,
                    "judge_profile": HEALTHBENCH_JUDGE_PROFILE,
                    "refused_after_semantic_calls": HEALTHBENCH_SEMANTIC_CALLS,
                },
                actual_queried_message_list=messages,
            )
        self.semantic_attempts.begin_semantic_call(
            GRADER_INVOCATION, messages, maximum_calls=HEALTHBENCH_SEMANTIC_CALLS
        )
        response = self.transport.create(messages=messages)
        content = response.choices[0].message.content
        return self.response_type(
            response_text=content,
            response_metadata={
                "usage": response.usage,
                "judge_profile": HEALTHBENCH_JUDGE_PROFILE,
                "judge_model": self.transport.model,
                "reasoning_effort": self.transport.reasoning_effort,
                "returned_model": getattr(response, "model", None),
            },
            actual_queried_message_list=messages,
        )


__all__ = [
    "ExternalHealthBenchRubricSampler",
    "ExternalJudgeTransport",
    "HealthBenchExternalJudgeProfile",
    "HealthBenchTransportError",
    "HealthBenchTransportExhausted",
    "RubricAttemptRegistry",
    "TransportRetryPolicy",
]
