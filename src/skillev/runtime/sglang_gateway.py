from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, cast
from urllib.parse import urlsplit

from skillev.contracts import JsonValue, normalize_json

from .contracts import BudgetVector
from .models import ModelProvider, ModelRequest, ModelResponse, TraceContext
from .openai_provider import (
    HTTPTransport,
    OpenAICompatibleProvider,
    OpenAIProviderConfig,
)
from .serving_profile import require_same_profile, serving_profile


class SGLangRole(StrEnum):
    EXECUTOR = "executor"
    HEALTH_GRADER = "health_grader"
    SUPERVISOR = "supervisor"
    SKILL_CREATOR = "skill_creator"


class SGLangGatewayError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SGLangGatewayConfig:
    endpoint_base: str
    base_model: str
    supervisor_adapter: str
    seed: int = 0
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens: int = 2048
    request_timeout_seconds: float = 300.0
    control_timeout_seconds: float = 30.0
    control_retries: int = 2
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
            raise ValueError("endpoint_base must be an HTTP(S) URL without credentials")
        if not self.base_model.strip() or not self.supervisor_adapter.strip():
            raise ValueError("base and supervisor model names must be non-empty")
        if self.base_model == self.supervisor_adapter:
            raise ValueError("base and supervisor model names must differ")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        for field_name in ("temperature", "top_p"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{field_name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
        if self.temperature < 0 or not 0 < self.top_p <= 1:
            raise ValueError("temperature/top_p are outside their supported ranges")
        for field_name in (
            "max_output_tokens",
            "control_retries",
            "max_response_bytes",
        ):
            value = getattr(self, field_name)
            minimum = 0 if field_name == "control_retries" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{field_name} is outside its supported range")
        for field_name in ("request_timeout_seconds", "control_timeout_seconds"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{field_name} must be finite and positive")

    @property
    def api_root(self) -> str:
        base = self.endpoint_base.rstrip("/")
        return base.removesuffix("/v1")

    @property
    def openai_base(self) -> str:
        return self.api_root + "/v1"

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "base_model": self.base_model,
            "control_retries": self.control_retries,
            "control_timeout_seconds": float(self.control_timeout_seconds),
            "endpoint_base": self.endpoint_base,
            "max_output_tokens": self.max_output_tokens,
            "max_response_bytes": self.max_response_bytes,
            "request_timeout_seconds": float(self.request_timeout_seconds),
            "seed": self.seed,
            "supervisor_adapter": self.supervisor_adapter,
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
        }

    @classmethod
    def from_value(cls, value: object) -> SGLangGatewayConfig:
        normalized = normalize_json(value)
        fields = {
            "base_model",
            "control_retries",
            "control_timeout_seconds",
            "endpoint_base",
            "max_output_tokens",
            "max_response_bytes",
            "request_timeout_seconds",
            "seed",
            "supervisor_adapter",
            "temperature",
            "top_p",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("SGLang gateway binding has incompatible fields")
        text_fields = ("base_model", "endpoint_base", "supervisor_adapter")
        if any(type(normalized[field]) is not str for field in text_fields):
            raise TypeError("SGLang gateway text fields are invalid")
        integer_fields = (
            "control_retries",
            "max_output_tokens",
            "max_response_bytes",
            "seed",
        )
        if any(type(normalized[field]) is not int for field in integer_fields):
            raise TypeError("SGLang gateway integer fields are invalid")
        numeric_fields = (
            "control_timeout_seconds",
            "request_timeout_seconds",
            "temperature",
            "top_p",
        )
        if any(
            isinstance(normalized[field], bool) or not isinstance(normalized[field], int | float)
            for field in numeric_fields
        ):
            raise TypeError("SGLang gateway numeric fields are invalid")
        return cls(
            endpoint_base=normalized["endpoint_base"],
            base_model=normalized["base_model"],
            supervisor_adapter=normalized["supervisor_adapter"],
            seed=normalized["seed"],
            temperature=float(normalized["temperature"]),
            top_p=float(normalized["top_p"]),
            max_output_tokens=normalized["max_output_tokens"],
            request_timeout_seconds=float(normalized["request_timeout_seconds"]),
            control_timeout_seconds=float(normalized["control_timeout_seconds"]),
            control_retries=normalized["control_retries"],
            max_response_bytes=normalized["max_response_bytes"],
        )


class SGLangControlTransport(Protocol):
    def request(
        self,
        *,
        method: str,
        url: str,
        payload: Mapping[str, JsonValue] | None,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> tuple[int, JsonValue]: ...


class UrllibSGLangControlTransport:
    def request(
        self,
        *,
        method: str,
        url: str,
        payload: Mapping[str, JsonValue] | None,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> tuple[int, JsonValue]:
        body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout_seconds,
            ) as response:
                raw = response.read(max_response_bytes + 1)
                status = response.status
        except urllib.error.HTTPError as error:
            status = error.code
            raw = error.read(max_response_bytes + 1)
        except (OSError, TimeoutError, urllib.error.URLError) as error:
            raise SGLangGatewayError("SGLang control request failed") from error
        if len(raw) > max_response_bytes:
            raise SGLangGatewayError("SGLang control response exceeded its byte limit")
        if not raw:
            return status, None
        try:
            return status, normalize_json(json.loads(raw))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            if not 200 <= status < 300:
                return status, None
            raise SGLangGatewayError("SGLang control response was not valid JSON") from error


@dataclass(frozen=True, slots=True)
class AdapterGeneration:
    generation: int
    adapter_name: str
    adapter_revision: str

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("adapter generation must be non-negative")
        if not self.adapter_name.strip() or not self.adapter_revision.strip():
            raise ValueError("adapter name and revision must be non-empty")


@dataclass(frozen=True, slots=True)
class PreparedAdapterSwap:
    previous: AdapterGeneration
    previous_path: str | None
    candidate: AdapterGeneration
    candidate_path: str


@dataclass(slots=True)
class _RoleAwareProvider:
    gateway: SGLangGateway
    role: SGLangRole

    async def generate(
        self,
        request: ModelRequest,
        *,
        budget: BudgetVector,
        trace_context: TraceContext,
    ) -> ModelResponse:
        self.gateway._begin_request(self.role)
        try:
            result = await self.gateway._delegate(self.role).generate(
                request,
                budget=budget,
                trace_context=trace_context,
            )
        finally:
            self.gateway._end_request(self.role)
        metadata = result.provider_metadata
        merged: dict[str, JsonValue] = {}
        if isinstance(metadata, dict):
            merged.update(metadata)
        merged.update(
            {
                "adapter_generation": self.gateway.adapter_generation.generation,
                "sglang_role": self.role.value,
            }
        )
        return ModelResponse(
            content=result.content,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            action_token_ids=result.action_token_ids,
            action_token_logprobs=result.action_token_logprobs,
            finish_reason=result.finish_reason,
            provider_metadata=normalize_json(merged),
        )


@dataclass(slots=True)
class SGLangGateway:
    config: SGLangGatewayConfig
    _control_transport: SGLangControlTransport = field(
        default_factory=UrllibSGLangControlTransport,
        repr=False,
    )
    _model_transport: HTTPTransport | None = field(default=None, repr=False)
    _condition: threading.Condition = field(
        default_factory=threading.Condition,
        init=False,
        repr=False,
    )
    _supervisor_inflight: int = field(default=0, init=False, repr=False)
    _swap_pending: bool = field(default=False, init=False, repr=False)
    _generation: AdapterGeneration = field(init=False, repr=False)
    _active_adapter_path: str | None = field(default=None, init=False, repr=False)
    _prepared_swap: PreparedAdapterSwap | None = field(default=None, init=False, repr=False)
    _runtime_profile: dict[str, JsonValue] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config, SGLangGatewayConfig):
            raise TypeError("config must be SGLangGatewayConfig")
        self._generation = AdapterGeneration(
            generation=0,
            adapter_name=self.config.supervisor_adapter,
            adapter_revision="not-loaded",
        )

    @property
    def adapter_generation(self) -> AdapterGeneration:
        return self._generation

    @property
    def supervisor_inflight(self) -> int:
        with self._condition:
            return self._supervisor_inflight

    def provider(self, role: SGLangRole) -> ModelProvider:
        if not isinstance(role, SGLangRole):
            raise TypeError("role must be SGLangRole")
        return _RoleAwareProvider(self, role)

    def begin_supervisor_rollout(self) -> AdapterGeneration:
        self._begin_request(SGLangRole.SUPERVISOR)
        return self._generation

    def end_supervisor_rollout(self) -> None:
        self._end_request(SGLangRole.SUPERVISOR)

    def begin_skill_creator_request(self) -> None:
        self._begin_request(SGLangRole.SKILL_CREATOR)

    def end_skill_creator_request(self) -> None:
        self._end_request(SGLangRole.SKILL_CREATOR)

    def begin_executor_request(self) -> None:
        self._begin_request(SGLangRole.EXECUTOR)

    def end_executor_request(self) -> None:
        self._end_request(SGLangRole.EXECUTOR)

    def _delegate(self, role: SGLangRole) -> ModelProvider:
        model = (
            self._generation.adapter_name
            if role is SGLangRole.SUPERVISOR
            else self.config.base_model
        )
        return OpenAICompatibleProvider(
            OpenAIProviderConfig(
                endpoint_base=self.config.openai_base,
                model=model,
                seed=self.config.seed,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_output_tokens=self.config.max_output_tokens,
                request_timeout_seconds=self.config.request_timeout_seconds,
                max_response_bytes=self.config.max_response_bytes,
            ),
            transport=self._model_transport,
        )

    async def health(self) -> tuple[str, ...]:
        return self._health_sync()

    def read_serving_profile(self, *, event_grammar: bool = False) -> dict[str, JsonValue]:
        status, value = self._request_control("GET", "/get_server_info", None)
        if status != 200:
            raise SGLangGatewayError("SGLang execution settings are unavailable")
        return serving_profile(value, event_grammar=event_grammar)

    def bind_serving_profile(self, profile: dict[str, JsonValue]) -> None:
        with self._condition:
            if self._supervisor_inflight or self._swap_pending:
                raise RuntimeError("cannot replace serving expectations during active work")
            self._runtime_profile = dict(profile)

    def _require_serving_profile(self) -> None:
        if self._runtime_profile is not None:
            actual = (
                self.read_serving_profile(event_grammar=True)
                if "event_grammar_patch" in self._runtime_profile
                else self.read_serving_profile()
            )
            require_same_profile(self._runtime_profile, actual)

    def bind_existing_supervisor_adapter(self, *, adapter_revision: str) -> AdapterGeneration:
        if not adapter_revision.strip():
            raise ValueError("adapter revision must be non-empty")
        self._require_serving_profile()
        models = self._health_sync()
        if self.config.supervisor_adapter not in models:
            raise SGLangGatewayError("configured supervisor adapter is absent from SGLang")
        with self._condition:
            if self._swap_pending or self._prepared_swap is not None or self._supervisor_inflight:
                raise RuntimeError("cannot bind an adapter while supervisor work is active")
            self._generation = AdapterGeneration(
                generation=0,
                adapter_name=self.config.supervisor_adapter,
                adapter_revision=adapter_revision,
            )
            self._active_adapter_path = None
            return self._generation

    def restore_supervisor_adapter(
        self,
        *,
        adapter_path: str,
        adapter_revision: str,
    ) -> AdapterGeneration:
        if not adapter_path.strip() or not adapter_revision.strip():
            raise ValueError("adapter path and revision must be non-empty")
        self._require_serving_profile()
        candidate_name = self._adapter_name(adapter_revision)
        models = self._health_sync()
        if candidate_name not in models:
            prepared = self.prepare_supervisor_adapter(
                adapter_path=adapter_path,
                adapter_revision=adapter_revision,
            )
            return self.commit_supervisor_adapter(prepared)
        self._validate_adapter(candidate_name)
        with self._condition:
            if self._swap_pending or self._prepared_swap is not None or self._supervisor_inflight:
                raise RuntimeError("cannot restore an adapter while supervisor work is active")
            self._generation = AdapterGeneration(
                generation=self._generation.generation + 1,
                adapter_name=candidate_name,
                adapter_revision=adapter_revision,
            )
            self._active_adapter_path = adapter_path
            return self._generation

    def _health_sync(self) -> tuple[str, ...]:
        health_status, _ = self._request_control("GET", "/health", None)
        if health_status != 200:
            raise SGLangGatewayError("SGLang health endpoint is not ready")
        model_status, value = self._request_control("GET", "/v1/models", None)
        if model_status != 200 or not isinstance(value, dict):
            raise SGLangGatewayError("SGLang model endpoint is not ready")
        data = value.get("data")
        if not isinstance(data, list):
            raise SGLangGatewayError("SGLang model list is malformed")
        model_ids: list[str] = []
        for item in data:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise SGLangGatewayError("SGLang model list contains a malformed entry")
            model_ids.append(cast(str, item["id"]))
        if self.config.base_model not in model_ids:
            raise SGLangGatewayError("configured base model is absent from SGLang")
        return tuple(model_ids)

    async def swap_supervisor_adapter(
        self,
        *,
        adapter_path: str,
        adapter_revision: str,
    ) -> AdapterGeneration:
        prepared = self.prepare_supervisor_adapter(
            adapter_path=adapter_path,
            adapter_revision=adapter_revision,
        )
        try:
            return self.commit_supervisor_adapter(prepared)
        except SGLangGatewayError:
            if self._prepared_swap == prepared:
                self.rollback_supervisor_adapter(prepared)
            raise

    def prepare_supervisor_adapter(
        self,
        *,
        adapter_path: str,
        adapter_revision: str,
    ) -> PreparedAdapterSwap:
        if not adapter_path.strip() or not adapter_revision.strip():
            raise ValueError("adapter path and revision must be non-empty")
        self._require_serving_profile()
        with self._condition:
            if self._swap_pending or self._prepared_swap is not None:
                raise RuntimeError("another supervisor adapter transaction is active")
            self._swap_pending = True
            while self._supervisor_inflight:
                self._condition.wait()
        previous = self._generation
        previous_path = self._active_adapter_path
        candidate_name = self._adapter_name(adapter_revision)
        try:
            self._expect_success(
                *self._request_control(
                    "POST",
                    "/load_lora_adapter",
                    {
                        "lora_name": candidate_name,
                        "lora_path": adapter_path,
                    },
                ),
                operation="load supervisor adapter",
            )
            models = self._health_sync()
            if candidate_name not in models:
                raise SGLangGatewayError("loaded supervisor adapter is absent from SGLang")
            self._validate_adapter(candidate_name)
            generation = AdapterGeneration(
                generation=previous.generation + 1,
                adapter_name=candidate_name,
                adapter_revision=adapter_revision,
            )
            self._generation = generation
            self._active_adapter_path = adapter_path
            prepared = PreparedAdapterSwap(
                previous=previous,
                previous_path=previous_path,
                candidate=generation,
                candidate_path=adapter_path,
            )
            self._prepared_swap = prepared
            return prepared
        except SGLangGatewayError as error:
            self._generation = previous
            self._active_adapter_path = previous_path
            if candidate_name != previous.adapter_name:
                try:
                    self._expect_success(
                        *self._request_control(
                            "POST",
                            "/unload_lora_adapter",
                            {"lora_name": candidate_name},
                        ),
                        operation="remove failed supervisor adapter",
                    )
                except SGLangGatewayError as rollback_error:
                    raise SGLangGatewayError(
                        "supervisor adapter swap and rollback both failed"
                    ) from rollback_error
            self._release_swap()
            raise error

    def commit_supervisor_adapter(self, prepared: PreparedAdapterSwap) -> AdapterGeneration:
        self._require_prepared_swap(prepared)
        try:
            if prepared.previous.generation:
                self._expect_success(
                    *self._request_control(
                        "POST",
                        "/unload_lora_adapter",
                        {"lora_name": prepared.previous.adapter_name},
                    ),
                    operation="unload old supervisor adapter",
                )
        except SGLangGatewayError:
            raise
        self._prepared_swap = None
        self._release_swap()
        return prepared.candidate

    def rollback_supervisor_adapter(self, prepared: PreparedAdapterSwap) -> None:
        self._require_prepared_swap(prepared)
        self._expect_success(
            *self._request_control(
                "POST",
                "/unload_lora_adapter",
                {"lora_name": prepared.candidate.adapter_name},
            ),
            operation="rollback supervisor adapter",
        )
        self._generation = prepared.previous
        self._active_adapter_path = prepared.previous_path
        self._prepared_swap = None
        self._release_swap()

    def _require_prepared_swap(self, prepared: PreparedAdapterSwap) -> None:
        if not isinstance(prepared, PreparedAdapterSwap) or prepared != self._prepared_swap:
            raise ValueError("supervisor adapter transaction identity differs")

    def _release_swap(self) -> None:
        with self._condition:
            self._swap_pending = False
            self._condition.notify_all()

    def _adapter_name(self, revision: str) -> str:
        safe = "".join(
            character if character.isalnum() or character in "_-" else "_" for character in revision
        )
        if not safe:
            raise ValueError("adapter revision has no usable name")
        return f"{self.config.supervisor_adapter}{safe}"

    def _validate_adapter(self, adapter_name: str) -> None:
        status, value = self._request_control(
            "POST",
            "/v1/chat/completions",
            cast(
                dict[str, JsonValue],
                normalize_json(
                    {
                        "max_tokens": 1,
                        "messages": [{"content": "Reply with OK.", "role": "user"}],
                        "model": adapter_name,
                        "temperature": 0,
                    }
                ),
            ),
        )
        if status != 200 or not isinstance(value, dict) or not value.get("choices"):
            raise SGLangGatewayError("loaded supervisor adapter failed its validation request")

    def _request_control(
        self,
        method: str,
        path: str,
        payload: Mapping[str, JsonValue] | None,
    ) -> tuple[int, JsonValue]:
        last_error: SGLangGatewayError | None = None
        for attempt in range(self.config.control_retries + 1):
            try:
                return self._control_transport.request(
                    method=method,
                    url=self.config.api_root + path,
                    payload=payload,
                    timeout_seconds=self.config.control_timeout_seconds,
                    max_response_bytes=self.config.max_response_bytes,
                )
            except SGLangGatewayError as error:
                last_error = error
                if attempt < self.config.control_retries:
                    time.sleep(min(0.25 * (2**attempt), 1.0))
        raise SGLangGatewayError("SGLang control retries were exhausted") from last_error

    @staticmethod
    def _expect_success(status: int, value: JsonValue, *, operation: str) -> None:
        del value
        if not 200 <= status < 300:
            raise SGLangGatewayError(f"SGLang failed to {operation}")

    def _begin_request(self, role: SGLangRole) -> None:
        if role is not SGLangRole.SUPERVISOR:
            return
        with self._condition:
            while self._swap_pending:
                self._condition.wait()
            self._supervisor_inflight += 1

    def _end_request(self, role: SGLangRole) -> None:
        if role is not SGLangRole.SUPERVISOR:
            return
        with self._condition:
            self._supervisor_inflight -= 1
            if self._supervisor_inflight < 0:
                raise RuntimeError("supervisor in-flight count became negative")
            self._condition.notify_all()


__all__ = [
    "AdapterGeneration",
    "PreparedAdapterSwap",
    "SGLangControlTransport",
    "SGLangGateway",
    "SGLangGatewayConfig",
    "SGLangGatewayError",
    "SGLangRole",
    "UrllibSGLangControlTransport",
]
