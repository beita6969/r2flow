from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from skillev.contracts.canonical import JsonValue, normalize_json

from .contracts import BudgetVector


@dataclass(frozen=True, slots=True)
class TraceContext:
    run_id: str
    attempt_id: str
    invocation_id: str
    turn: int

    def __post_init__(self) -> None:
        if not all((self.run_id, self.attempt_id, self.invocation_id)):
            raise ValueError("Trace context identity fields cannot be empty")
        if type(self.turn) is not int or self.turn < 1:
            raise ValueError("Trace context turn must be positive")


@dataclass(frozen=True, slots=True)
class ModelRequest:
    prompt: str
    response_schema: JsonValue
    tools: JsonValue = None

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("Model prompt cannot be empty")
        if normalize_json(self.response_schema) != self.response_schema:
            raise ValueError("Response schema must be normalized JSON")
        if normalize_json(self.tools) != self.tools:
            raise ValueError("Tool declarations must be normalized JSON")


@dataclass(frozen=True, slots=True)
class ModelResponse:
    content: str
    input_tokens: int
    output_tokens: int
    action_token_ids: tuple[int, ...] = ()
    action_token_logprobs: tuple[float, ...] = ()
    finish_reason: str = "complete"
    provider_metadata: JsonValue = None

    def __post_init__(self) -> None:
        if type(self.input_tokens) is not int or type(self.output_tokens) is not int:
            raise TypeError("Token usage counters must be integers")
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("Token usage counters cannot be negative")
        if len(self.action_token_ids) != len(self.action_token_logprobs):
            raise ValueError("Action token IDs and log probabilities must align")
        if len(self.action_token_ids) > self.output_tokens:
            raise ValueError("Action-token span cannot exceed aggregate output usage")
        if any(type(token_id) is not int or token_id < 0 for token_id in self.action_token_ids):
            raise ValueError("Action token IDs must be non-negative integers")
        for value in self.action_token_logprobs:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError("Action token log probabilities must be numeric")
            if not math.isfinite(value) or value > 0:
                raise ValueError("Action token log probabilities must be finite and non-positive")
        if not self.finish_reason:
            raise ValueError("Model response requires a finish reason")
        if normalize_json(self.provider_metadata) != self.provider_metadata:
            raise ValueError("Provider metadata must be normalized JSON")


class ModelProvider(Protocol):
    async def generate(
        self,
        request: ModelRequest,
        *,
        budget: BudgetVector,
        trace_context: TraceContext,
    ) -> ModelResponse: ...


class ModelCallRejectedError(RuntimeError):
    pass


@dataclass(slots=True)
class ScriptedModel:
    responses: tuple[ModelResponse | BaseException, ...]
    _cursor: int = 0
    calls: list[tuple[ModelRequest, TraceContext]] = field(default_factory=list)

    async def generate(
        self,
        request: ModelRequest,
        *,
        budget: BudgetVector,
        trace_context: TraceContext,
    ) -> ModelResponse:
        if self._cursor >= len(self.responses):
            raise RuntimeError("Scripted model response sequence is exhausted")
        item = self.responses[self._cursor]
        self._cursor += 1
        self.calls.append((request, trace_context))
        if isinstance(item, BaseException):
            raise item
        actual = BudgetVector(
            input_tokens=item.input_tokens,
            output_tokens=item.output_tokens,
            model_calls=1,
            agent_turns=1,
        )
        if not actual.fits_within(budget):
            raise ModelCallRejectedError("Scripted response does not fit the call budget")
        return item
