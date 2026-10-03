from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, cast
from urllib.parse import urlsplit

from skillev.contracts.canonical import canonical_json_bytes, normalize_json

from .contracts import BudgetVector
from .models import (
    ModelCallRejectedError,
    ModelRequest,
    ModelResponse,
    TraceContext,
)


class HTTPTransportError(RuntimeError):
    pass


class ProviderResponseError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class HTTPRequest:
    url: str
    body: bytes = field(repr=False)
    headers: Mapping[str, str] = field(repr=False)
    timeout_seconds: float
    max_response_bytes: int


@dataclass(frozen=True, slots=True)
class HTTPResponse:
    status: int
    body: bytes = field(repr=False)
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if type(self.status) is not int or not 100 <= self.status <= 599:
            raise ValueError("HTTP status must be an integer in the standard range")
        if not isinstance(self.body, bytes):
            raise TypeError("HTTP response body must be bytes")
        if not all(
            isinstance(key, str) and isinstance(value, str) for key, value in self.headers.items()
        ):
            raise TypeError("HTTP response headers must map strings to strings")


class HTTPTransport(Protocol):
    def send(self, request: HTTPRequest) -> HTTPResponse: ...


class UrllibHTTPTransport:
    def send(self, request: HTTPRequest) -> HTTPResponse:
        http_request = urllib.request.Request(
            request.url,
            data=request.body,
            headers=dict(request.headers),
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                http_request,
                timeout=request.timeout_seconds,
            ) as response:
                status = getattr(response, "status", None)
                if type(status) is not int:
                    raise HTTPTransportError("HTTP response did not expose a status")
                body = response.read(request.max_response_bytes + 1)
                headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
                return HTTPResponse(status=status, body=body, headers=headers)
        except urllib.error.HTTPError as error:
            try:
                body = error.read(request.max_response_bytes + 1)
            except OSError:
                body = b""
            headers = (
                {}
                if error.headers is None
                else {str(key).lower(): str(value) for key, value in error.headers.items()}
            )
            return HTTPResponse(status=error.code, body=body, headers=headers)
        except (TimeoutError, urllib.error.URLError, OSError) as error:
            raise HTTPTransportError("OpenAI-compatible HTTP transport failed") from error


@dataclass(frozen=True, slots=True)
class OpenAIProviderConfig:
    endpoint_base: str
    model: str
    seed: int = 0
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens: int = 2048
    request_timeout_seconds: float = 300.0
    max_response_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        parsed = urlsplit(self.endpoint_base)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Endpoint must be an HTTP(S) base URL without embedded credentials")
        if not self.model:
            raise ValueError("Provider model cannot be empty")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("Provider seed must be one unsigned 64-bit integer")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, int | float)
            or not math.isfinite(self.temperature)
            or self.temperature < 0
        ):
            raise ValueError("Temperature must be finite and non-negative")
        if (
            isinstance(self.top_p, bool)
            or not isinstance(self.top_p, int | float)
            or not math.isfinite(self.top_p)
            or not 0 < self.top_p <= 1
        ):
            raise ValueError("Top-p must lie in (0, 1]")
        if type(self.max_output_tokens) is not int or self.max_output_tokens < 1:
            raise ValueError("Maximum output tokens must be positive")
        if (
            isinstance(self.request_timeout_seconds, bool)
            or not isinstance(self.request_timeout_seconds, int | float)
            or not math.isfinite(self.request_timeout_seconds)
            or self.request_timeout_seconds <= 0
        ):
            raise ValueError("Request timeout must be finite and positive")
        if type(self.max_response_bytes) is not int or self.max_response_bytes < 1:
            raise ValueError("Maximum response bytes must be positive")

    @property
    def chat_completions_url(self) -> str:
        return self.endpoint_base.rstrip("/") + "/chat/completions"


BearerTokenProvider = Callable[[], str | None]


class OpenAICompatibleProvider:
    def __init__(
        self,
        config: OpenAIProviderConfig,
        *,
        transport: HTTPTransport | None = None,
        bearer_token_provider: BearerTokenProvider | None = None,
    ) -> None:
        self.config = config
        self._transport = transport if transport is not None else UrllibHTTPTransport()
        self._bearer_token_provider = bearer_token_provider

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(endpoint_base={self.config.endpoint_base!r}, "
            f"model={self.config.model!r}, seed={self.config.seed!r})"
        )

    async def generate(
        self,
        request: ModelRequest,
        *,
        budget: BudgetVector,
        trace_context: TraceContext,
    ) -> ModelResponse:
        return self._generate_sync(request, budget, trace_context)

    def _generate_sync(
        self,
        request: ModelRequest,
        budget: BudgetVector,
        trace_context: TraceContext,
    ) -> ModelResponse:
        del trace_context
        if budget.model_calls < 1 or budget.output_tokens < 1:
            raise ModelCallRejectedError("The call budget does not permit model generation")
        if not isinstance(request.response_schema, dict):
            raise ModelCallRejectedError("Structured output requires a JSON Schema object")

        maximum_output = min(self.config.max_output_tokens, budget.output_tokens)
        body: dict[str, object] = {
            "logprobs": True,
            "max_tokens": maximum_output,
            "messages": [{"content": request.prompt, "role": "user"}],
            "model": self.config.model,
            "response_format": {
                "json_schema": {
                    "name": "skillev_action",
                    "schema": request.response_schema,
                    "strict": True,
                },
                "type": "json_schema",
            },
            "seed": self.config.seed,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        if request.tools is not None:
            body["tools"] = request.tools

        headers = {"Content-Type": "application/json"}
        if self._bearer_token_provider is not None:
            token = self._bearer_token_provider()
            if token is not None:
                if not token or "\r" in token or "\n" in token:
                    raise ModelCallRejectedError("Bearer token is invalid")
                endpoint = urlsplit(self.config.endpoint_base)
                if endpoint.scheme != "https" and endpoint.hostname not in {
                    "127.0.0.1",
                    "::1",
                    "localhost",
                }:
                    raise ModelCallRejectedError(
                        "Bearer tokens require HTTPS or a loopback endpoint"
                    )
                headers["Authorization"] = f"Bearer {token}"

        transport_request = HTTPRequest(
            url=self.config.chat_completions_url,
            body=canonical_json_bytes(body),
            headers=headers,
            timeout_seconds=float(self.config.request_timeout_seconds),
            max_response_bytes=self.config.max_response_bytes,
        )
        response = self._transport.send(transport_request)
        if len(response.body) > self.config.max_response_bytes:
            raise ProviderResponseError("Provider response exceeded its byte limit")
        if response.status != 200:
            raise ProviderResponseError("Provider returned a non-success HTTP status")
        return self._parse_response(
            response.body,
            budget=budget,
            maximum_output=maximum_output,
        )

    def _parse_response(
        self,
        payload: bytes,
        *,
        budget: BudgetVector,
        maximum_output: int,
    ) -> ModelResponse:
        value = _json_object(payload)
        choices = value.get("choices")
        usage = value.get("usage")
        if (
            not isinstance(choices, list)
            or len(choices) != 1
            or not isinstance(choices[0], dict)
            or not isinstance(usage, dict)
        ):
            raise ProviderResponseError("Provider response is missing choices or usage")

        choice = cast(dict[str, object], choices[0])
        message = choice.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ProviderResponseError("Provider response is missing structured content")
        content = cast(str, message["content"])
        try:
            json.loads(content)
        except json.JSONDecodeError as error:
            raise ProviderResponseError("Structured model content is not JSON") from error

        prompt_tokens = _usage_counter(usage, "prompt_tokens")
        completion_tokens = _usage_counter(usage, "completion_tokens")
        total_tokens = usage.get("total_tokens")
        if total_tokens is not None and (
            type(total_tokens) is not int or total_tokens != prompt_tokens + completion_tokens
        ):
            raise ProviderResponseError("Provider total-token usage is inconsistent")
        if prompt_tokens > budget.input_tokens or completion_tokens > maximum_output:
            raise ProviderResponseError("Provider usage exceeded the reserved token budget")

        token_ids, token_logprobs, span_source = _action_token_span(
            choice,
            message=cast(dict[str, object], message),
            completion_tokens=completion_tokens,
        )
        finish_reason = choice.get("finish_reason")
        if not isinstance(finish_reason, str) or not finish_reason:
            raise ProviderResponseError("Provider response is missing a finish reason")

        metadata = normalize_json(
            {
                "action_span_source": span_source,
                "model": value.get("model"),
                "response_id": value.get("id"),
                "system_fingerprint": value.get("system_fingerprint"),
                "transport": "openai-compatible-http",
            }
        )
        return ModelResponse(
            content=content,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            action_token_ids=token_ids,
            action_token_logprobs=token_logprobs,
            finish_reason=finish_reason,
            provider_metadata=metadata,
        )


def _json_object(payload: bytes) -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ProviderResponseError("Provider JSON contains a duplicate object key")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ProviderResponseError(f"Provider JSON contains a non-finite value: {value}")

    try:
        parsed = json.loads(
            payload,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProviderResponseError("Provider response is not valid JSON") from error
    normalized = normalize_json(parsed)
    if not isinstance(normalized, dict):
        raise ProviderResponseError("Provider response must be a JSON object")
    return cast(dict[str, object], normalized)


def _usage_counter(usage: Mapping[str, object], name: str) -> int:
    value = usage.get(name)
    if type(value) is not int or value < 0:
        raise ProviderResponseError("Provider usage counters must be non-negative integers")
    return value


def _action_token_span(
    choice: Mapping[str, object],
    *,
    message: Mapping[str, object],
    completion_tokens: int,
) -> tuple[tuple[int, ...], tuple[float, ...], str]:
    span = choice.get("action_token_span")
    if span is None:
        span = message.get("action_token_span")
    if span is not None:
        if not isinstance(span, dict):
            raise ProviderResponseError("Explicit action-token span must be an object")
        token_ids = span.get("token_ids")
        token_logprobs = span.get("token_logprobs", span.get("logprobs"))
        ids, probabilities = _validate_span(
            token_ids,
            token_logprobs,
            completion_tokens=completion_tokens,
        )
        return ids, probabilities, "action_token_span"
    return (), (), "missing"


def _validate_span(
    token_ids: object,
    token_logprobs: object,
    *,
    completion_tokens: int,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if not isinstance(token_ids, list) or not isinstance(token_logprobs, list):
        raise ProviderResponseError("Action-token IDs and log probabilities must be lists")
    if len(token_ids) != len(token_logprobs) or len(token_ids) > completion_tokens:
        raise ProviderResponseError("Action-token span is not aligned with completion usage")

    ids: list[int] = []
    probabilities: list[float] = []
    for token_id, logprob in zip(token_ids, token_logprobs, strict=True):
        if type(token_id) is not int or token_id < 0:
            raise ProviderResponseError("Action-token IDs must be non-negative integers")
        if (
            isinstance(logprob, bool)
            or not isinstance(logprob, int | float)
            or not math.isfinite(logprob)
            or logprob > 0
        ):
            raise ProviderResponseError(
                "Action-token log probabilities must be finite and non-positive"
            )
        ids.append(token_id)
        probabilities.append(float(logprob))
    return tuple(ids), tuple(probabilities)
