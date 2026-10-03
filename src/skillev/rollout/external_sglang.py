from __future__ import annotations

import math
import time
from asyncio import CancelledError, sleep
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit

from skillev.contracts import JsonValue, normalize_json
from skillev.contracts.action_decoding import ACTION_DECODING_RULES, ACTION_GREEDY_UNSEEDED
from skillev.diagnostics.rollout_progress import current_progress, server_metrics
from skillev.runtime import BudgetVector
from skillev.runtime.request_journal import DurableRequestJournal
from skillev.runtime.sglang_gateway import (
    SGLangControlTransport,
    SGLangGateway,
    SGLangGatewayError,
    UrllibSGLangControlTransport,
)

from .generator import (
    PolicySnapshotMismatchError,
    RolloutGenerationRequest,
    RolloutGenerationResult,
    RolloutTokenizerProtocol,
)
from .types import GenerationPhase, PolicySnapshot

if TYPE_CHECKING:
    from skillev.runtime.frozen_executor import ExecutorOutput, FrozenExecutorSpec


class ExternalSGLangGenerationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ExternalSGLangRolloutConfig:
    endpoint_base: str
    request_timeout_seconds: float = 300.0
    max_response_bytes: int = 16 * 1024 * 1024
    transport_worker_threads: int = 1
    transport_isolation: str = "shared"

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
            raise ValueError("SGLang rollout endpoint must be an HTTP(S) URL without credentials")
        if (
            isinstance(self.request_timeout_seconds, bool)
            or not isinstance(self.request_timeout_seconds, int | float)
            or not math.isfinite(float(self.request_timeout_seconds))
            or self.request_timeout_seconds <= 0
        ):
            raise ValueError("SGLang rollout timeout must be finite and positive")
        if type(self.max_response_bytes) is not int or self.max_response_bytes <= 0:
            raise ValueError("SGLang rollout response limit must be positive")
        if type(self.transport_worker_threads) is not int or self.transport_worker_threads < 1:
            raise ValueError("SGLang transport worker count must be positive")
        if self.transport_isolation not in ("shared", "endpoint-partitioned"):
            raise ValueError("unknown SGLang transport isolation")

    @property
    def generate_url(self) -> str:
        return self.endpoint_base.rstrip("/").removesuffix("/v1") + "/generate"

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "endpoint_base": self.endpoint_base,
            "max_response_bytes": self.max_response_bytes,
            "request_timeout_seconds": float(self.request_timeout_seconds),
            "transport_worker_threads": self.transport_worker_threads,
            "server_action_boundary": True,
            **(
                {"transport_isolation": self.transport_isolation}
                if self.transport_isolation != "shared"
                else {}
            ),
        }

    @classmethod
    def from_value(cls, value: object) -> ExternalSGLangRolloutConfig:
        normalized = normalize_json(value)
        fields = {
            "endpoint_base",
            "max_response_bytes",
            "request_timeout_seconds",
            "transport_worker_threads",
            "server_action_boundary",
        }
        if not isinstance(normalized, dict) or (
            set(normalized) - {"transport_isolation"} != fields
        ):
            raise ValueError("SGLang rollout binding has incompatible fields")
        endpoint = normalized["endpoint_base"]
        timeout = normalized["request_timeout_seconds"]
        response_limit = normalized["max_response_bytes"]
        worker_threads = normalized["transport_worker_threads"]
        if normalized["server_action_boundary"] is not True:
            raise ValueError("SGLang rollout binding has an unsupported action boundary")
        if type(endpoint) is not str:
            raise TypeError("SGLang rollout endpoint must be text")
        if isinstance(timeout, bool) or not isinstance(timeout, int | float):
            raise TypeError("SGLang rollout timeout must be numeric")
        if type(response_limit) is not int or type(worker_threads) is not int:
            raise TypeError("SGLang rollout integer fields are invalid")
        return cls(
            endpoint_base=endpoint,
            request_timeout_seconds=float(timeout),
            max_response_bytes=response_limit,
            transport_worker_threads=worker_threads,
            transport_isolation=cast(str, normalized.get("transport_isolation", "shared")),
        )


@dataclass(slots=True)
class ExternalSGLangRolloutGenerator:
    config: ExternalSGLangRolloutConfig
    tokenizer: RolloutTokenizerProtocol
    gateway: SGLangGateway | None
    snapshot_provider: Callable[[], PolicySnapshot]
    transport: SGLangControlTransport = field(
        default_factory=UrllibSGLangControlTransport,
        repr=False,
    )
    request_journal: DurableRequestJournal | None = None
    action_decoding: str | None = None
    physical_usage: dict[str, int] = field(
        default_factory=lambda: {
            "server_generated_tokens": 0,
            "admitted_content_tokens": 0,
            "admitted_stop_tokens": 0,
            "discarded_suffix_tokens": 0,
        },
        init=False,
    )
    _episodes: set[str] = field(default_factory=set, init=False, repr=False)
    _episode_gateways: dict[str, SGLangGateway] = field(
        default_factory=dict, init=False, repr=False
    )
    _executors: dict[str, ThreadPoolExecutor] = field(init=False, repr=False)
    transport_capacity: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        from skillev.runtime.sglang_pool import SGLangActorPool

        if self.action_decoding is not None and self.action_decoding not in ACTION_DECODING_RULES:
            raise ValueError(f"action_decoding declares {ACTION_GREEDY_UNSEEDED} or nothing")
        endpoints = (
            [member.config.api_root + "/generate" for member in self.gateway.members]
            if isinstance(self.gateway, SGLangActorPool)
            else [self.config.generate_url]
        )
        total = self.config.transport_worker_threads
        if self.config.transport_isolation == "shared":
            self.transport_capacity = {"shared": total}
        else:
            if total < len(endpoints):
                raise ValueError("isolated transport needs at least one thread per endpoint")
            base, extra = divmod(total, len(endpoints))
            self.transport_capacity = {
                endpoint: base + int(index < extra) for index, endpoint in enumerate(endpoints)
            }
        self._executors = {
            endpoint: ThreadPoolExecutor(
                max_workers=capacity,
                thread_name_prefix=f"skillev-sglang-http-{index}",
            )
            for index, (endpoint, capacity) in enumerate(self.transport_capacity.items())
        }

    def snapshot(self) -> PolicySnapshot:
        snapshot = self.snapshot_provider()
        if not isinstance(snapshot, PolicySnapshot):
            raise TypeError("external SGLang snapshot provider returned an incompatible value")
        return snapshot

    def begin_episode(self, episode_id: str, expected_policy_snapshot_id: str) -> None:
        if not episode_id.strip() or episode_id in self._episodes:
            raise ValueError("rollout episode identity is empty or already active")
        if self.snapshot().snapshot_id != expected_policy_snapshot_id:
            raise PolicySnapshotMismatchError("remote rollout policy changed before episode")
        from skillev.runtime.sglang_pool import SGLangActorPool

        if isinstance(self.gateway, SGLangActorPool):
            endpoint = (
                self.request_journal.episode_route(episode_id, expected_policy_snapshot_id)
                if self.request_journal is not None
                else None
            )
            row = current_progress()
            horizon = 1 if row is None else row.request_priority[0]
            member = self.gateway.acquire_episode(
                episode_id,
                expected_policy_snapshot_id,
                endpoint=endpoint,
                estimated_work=horizon * self.gateway.config.max_output_tokens,
                horizon=horizon,
                benchmark_episode_index=None if row is None else row.benchmark_episode_index,
                benchmark=(
                    None
                    if row is None or row.task_domain is None
                    else row.task_domain.partition("/")[0]
                ),
            )
            try:
                if self.request_journal is not None:
                    self.request_journal.save_episode_route(
                        episode_id, expected_policy_snapshot_id, member.config.api_root
                    )
            except Exception:
                self.gateway.release_episode(episode_id)
                raise
            self._episode_gateways[episode_id] = member
            if row is not None:
                row.stage(
                    "actor-routed",
                    actor_endpoint=member.config.api_root,
                    actor_transport_capacity=dict(self.transport_capacity),
                    **self.gateway.routing_observation(episode_id),
                )
        self._episodes.add(episode_id)

    def end_episode(self, episode_id: str) -> None:
        if episode_id not in self._episodes:
            raise ValueError("rollout episode is not active")
        from skillev.runtime.sglang_pool import SGLangActorPool

        if isinstance(self.gateway, SGLangActorPool):
            self.gateway.release_episode(episode_id)
            del self._episode_gateways[episode_id]
        self._episodes.remove(episode_id)

    def execution_endpoint(self, request: RolloutGenerationRequest) -> str:
        return self.episode_endpoint(request.episode_id or "")

    def episode_endpoint(self, episode_id: str) -> str:
        gateway = self._episode_gateways.get(episode_id, self.gateway)
        return self.config.endpoint_base if gateway is None else gateway.config.api_root

    async def generate(self, request: RolloutGenerationRequest) -> RolloutGenerationResult:
        from skillev.runtime.sglang_pool import SGLangActorPool

        if isinstance(self.gateway, SGLangActorPool) and request.episode_id is not None:
            from skillev.runtime.serving_cost import EpisodeWork

            row = current_progress()
            horizon = 1 if row is None else row.request_priority[0]
            remaining = max(1, horizon - (request.turn_index or 1) + 1)
            self.gateway.update_episode_work(
                request.episode_id,
                remaining * (len(request.input_ids) + request.max_new_tokens),
                context=EpisodeWork(
                    None
                    if row is None or row.task_domain is None
                    else row.task_domain.partition("/")[0],
                    remaining,
                    len(request.input_ids),
                    request.max_new_tokens,
                    request.phase.value,
                ),
            )
        before = self.snapshot()
        if before.snapshot_id != request.expected_policy_snapshot_id:
            raise PolicySnapshotMismatchError("remote rollout policy changed before generation")
        gateway = self._episode_gateways.get(request.episode_id or "", self.gateway)
        generation = None if gateway is None else gateway.begin_supervisor_rollout()
        endpoint = (
            self.config.generate_url if gateway is None else gateway.config.api_root + "/generate"
        )
        payload: dict[str, JsonValue] = cast(
            dict[str, JsonValue],
            normalize_json(
                {
                    "input_ids": list(request.input_ids),
                    "sampling_params": {
                        "max_new_tokens": request.max_new_tokens,
                        "sampling_seed": request.seed,
                        "temperature": 1.0,
                        "top_k": -1,
                        "top_p": 1.0,
                    },
                    "stream": False,
                }
            ),
        )
        if generation is not None:
            payload["lora_path"] = generation.adapter_name
        if request.extra_stop_token_ids:
            sampling = cast(dict[str, JsonValue], payload["sampling_params"])
            sampling["stop_token_ids"] = list(request.extra_stop_token_ids)
        constrained = request.sampling_constraint is not None
        if constrained:
            sampling = cast(dict[str, JsonValue], payload["sampling_params"])
            sampling["structural_tag"] = request.sampling_constraint
            payload["return_logprob"] = True
        if (
            self.action_decoding == ACTION_GREEDY_UNSEEDED
            and request.phase is GenerationPhase.ACTION
        ):
            sampling = cast(dict[str, JsonValue], payload["sampling_params"])
            del sampling["sampling_seed"]
            sampling.update(temperature=0.0, top_k=1, top_p=1.0)
        request_started = time.perf_counter()
        try:
            status, raw = await self._request_without_early_lease_release(
                payload=payload, endpoint=endpoint, request=request
            )
        except SGLangGatewayError as error:
            raise ExternalSGLangGenerationError("SGLang native rollout request failed") from error
        finally:
            if generation is not None:
                assert gateway is not None
                gateway.end_supervisor_rollout()
        if status != 200:
            raise ExternalSGLangGenerationError("SGLang native rollout returned failure")
        content_ids, stop_ids, finish_reason, prompt_tokens = self._parse(raw)
        server_logq = None
        if constrained:
            self._require_event_finish(request, content_ids, stop_ids, finish_reason)
            server_logq = self._server_logq(raw, content_ids + stop_ids)
        physical_tokens = len(content_ids) + len(stop_ids)
        restored = isinstance(raw, dict) and raw.get("skillev_restored_response") is True
        meta = cast(dict[str, JsonValue], cast(dict[str, JsonValue], raw)["meta_info"])
        measured = {} if restored else server_metrics(meta)
        if (
            not restored
            and isinstance(self.gateway, SGLangActorPool)
            and request.episode_id is not None
        ):
            self.gateway.observe_episode_phase(
                request.episode_id,
                request.phase.value,
                input_tokens=prompt_tokens,
                output_tokens=physical_tokens,
                response_seconds=time.perf_counter() - request_started,
                metrics=measured,
            )
        row = current_progress()
        if row is not None:
            row.phase_metrics(
                **measured,
                restored_response=restored,
                serving_endpoint=endpoint,
                serving_adapter_name=None if generation is None else generation.adapter_name,
                serving_adapter_revision=None
                if generation is None
                else generation.adapter_revision,
            )
        if not restored:
            self.physical_usage["server_generated_tokens"] += physical_tokens
            self.physical_usage["admitted_content_tokens"] += len(content_ids)
            self.physical_usage["admitted_stop_tokens"] += len(stop_ids)
            self.physical_usage["discarded_suffix_tokens"] += (
                physical_tokens - len(content_ids) - len(stop_ids)
            )
        after = self.snapshot()
        if after != before:
            raise PolicySnapshotMismatchError("remote rollout policy changed during generation")
        return RolloutGenerationResult(
            content_token_ids=content_ids,
            stop_token_ids=stop_ids,
            finish_reason=finish_reason,
            policy_snapshot_id=after.snapshot_id,
            backend_id="sglang-native-exact-token",
            usage=BudgetVector(
                input_tokens=prompt_tokens,
                output_tokens=len(content_ids) + len(stop_ids),
                model_calls=1,
            ),
            server_token_logq=server_logq,
        )

    async def generate_frozen_base(
        self,
        *,
        episode_id: str,
        input_ids: tuple[int, ...],
        spec: FrozenExecutorSpec,
        regex: str | None = None,
    ) -> ExecutorOutput:
        from skillev.runtime.frozen_executor import ExecutorOutput, canonical_executor_output

        gateway = self._episode_gateways.get(episode_id, self.gateway)
        endpoint = (
            self.config.generate_url if gateway is None else gateway.config.api_root + "/generate"
        )
        payload: dict[str, JsonValue] = cast(
            dict[str, JsonValue],
            normalize_json(
                {
                    "input_ids": list(input_ids),
                    "sampling_params": {
                        "max_new_tokens": spec.max_output_tokens,
                        "sampling_seed": spec.seed,
                        "temperature": 0.0,
                        "top_k": 1,
                        "top_p": 1.0,
                        **({} if regex is None else {"regex": regex}),
                    },
                    "stream": False,
                }
            ),
        )
        started = time.perf_counter()
        if gateway is not None:
            gateway.begin_executor_request()
        try:
            key = "shared" if self.config.transport_isolation == "shared" else endpoint

            def send() -> tuple[int, JsonValue]:
                return self.transport.request(
                    method="POST",
                    url=endpoint,
                    payload=payload,
                    timeout_seconds=float(self.config.request_timeout_seconds),
                    max_response_bytes=self.config.max_response_bytes,
                )

            future: Future[tuple[int, JsonValue]] = self._executors[key].submit(send)
            try:
                status, raw = await _await_thread_result(future)
            except CancelledError:
                while not future.done():
                    await sleep(0.001)
                raise
        except SGLangGatewayError as error:
            raise ExternalSGLangGenerationError("SGLang frozen-executor request failed") from error
        finally:
            if gateway is not None:
                gateway.end_executor_request()
        if status != 200:
            raise ExternalSGLangGenerationError("SGLang frozen-executor request returned failure")
        content_ids, stop_ids, finish_reason, prompt_tokens = self._parse(raw)
        if prompt_tokens != len(input_ids):
            raise ExternalSGLangGenerationError("executor prompt usage differs from input IDs")
        latency_ms = round((time.perf_counter() - started) * 1000)
        row = current_progress()
        if row is not None:
            row.phase_metrics(
                executor_endpoint=endpoint,
                executor_latency_ms=latency_ms,
                executor_output_tokens=len(content_ids) + len(stop_ids),
            )
        return ExecutorOutput(
            token_ids=content_ids + stop_ids,
            text=canonical_executor_output(self.tokenizer.decode(content_ids)),
            finish=finish_reason,
            prompt_tokens=prompt_tokens,
            latency_ms=latency_ms,
            endpoint=endpoint,
        )

    @staticmethod
    def _require_event_finish(
        request: RolloutGenerationRequest,
        content_ids: tuple[int, ...],
        stop_ids: tuple[int, ...],
        finish_reason: str,
    ) -> None:
        from skillev.policy.event_grammar import parse_event_grammar_key

        key = parse_event_grammar_key(cast(str, request.sampling_constraint))
        if not (
            finish_reason == "stop"
            and stop_ids == (key.stop_token_id,)
            and len(content_ids) + 1 <= key.budget
        ):
            raise ExternalSGLangGenerationError("event grammar did not terminate within budget")

    @staticmethod
    def _server_logq(raw: JsonValue, generated: tuple[int, ...]) -> tuple[float, ...] | None:
        meta = cast(dict[str, JsonValue], cast(dict[str, JsonValue], raw)["meta_info"])
        entries = meta.get("output_token_logprobs")
        if not isinstance(entries, list) or len(entries) != len(generated):
            return None
        values: list[float] = []
        for entry, token in zip(entries, generated, strict=True):
            if (
                not isinstance(entry, list)
                or len(entry) < 2
                or isinstance(entry[0], bool)
                or not isinstance(entry[0], int | float)
                or entry[1] != token
            ):
                return None
            values.append(float(entry[0]))
        return tuple(values)

    async def _request_without_early_lease_release(
        self,
        *,
        payload: dict[str, JsonValue],
        endpoint: str | None = None,
        request: RolloutGenerationRequest | None = None,
    ) -> tuple[int, JsonValue]:
        row = current_progress()
        queued_at = time.perf_counter()
        if row is not None:
            row.stage(f"{row.phase or 'model'}-transport-queue")

        def send() -> tuple[int, JsonValue]:
            started = time.perf_counter()
            if row is not None:
                row.phase_metrics(client_transport_queue_seconds=started - queued_at)
                row.stage(f"{row.phase or 'model'}-awaiting-response")
            try:

                def dispatch() -> tuple[int, JsonValue]:
                    return self.transport.request(
                        method="POST",
                        url=endpoint or self.config.generate_url,
                        payload=payload,
                        timeout_seconds=float(self.config.request_timeout_seconds),
                        max_response_bytes=self.config.max_response_bytes,
                    )

                if self.request_journal is None:
                    return dispatch()
                if request is None or request.episode_id is None or request.turn_index is None:
                    raise ValueError("durable rollout requests need episode and turn coordinates")
                return self.request_journal.request(
                    identity=(
                        request.episode_id,
                        str(request.turn_index),
                        request.phase.value,
                        request.expected_policy_snapshot_id,
                        request.library_version or "unbound-library",
                        request.decoding_snapshot_id,
                    ),
                    endpoint=endpoint or self.config.generate_url,
                    payload=payload,
                    send=dispatch,
                )
            finally:
                if row is not None:
                    row.phase_metrics(client_response_seconds=time.perf_counter() - started)

        key = (
            "shared"
            if self.config.transport_isolation == "shared"
            else endpoint or self.config.generate_url
        )
        future: Future[tuple[int, JsonValue]] = self._executors[key].submit(send)
        try:
            return await _await_thread_result(future)
        except CancelledError:
            while not future.done():
                await sleep(0.001)
            raise

    def close(self) -> None:
        for executor in self._executors.values():
            executor.shutdown(wait=True, cancel_futures=False)

    def _parse(
        self,
        raw: JsonValue,
    ) -> tuple[tuple[int, ...], tuple[int, ...], str, int]:
        if not isinstance(raw, dict):
            raise ExternalSGLangGenerationError("SGLang native response must be an object")
        output_ids = raw.get("output_ids")
        meta = raw.get("meta_info")
        if not isinstance(output_ids, list) or not isinstance(meta, dict):
            raise ExternalSGLangGenerationError("SGLang native response fields are incomplete")
        completion_tokens = meta.get("completion_tokens")
        prompt_tokens = meta.get("prompt_tokens")
        finish = meta.get("finish_reason")
        if (
            type(completion_tokens) is not int
            or completion_tokens < 0
            or type(prompt_tokens) is not int
            or prompt_tokens < 0
            or not isinstance(finish, dict)
        ):
            raise ExternalSGLangGenerationError("SGLang native usage or finish reason is invalid")
        if completion_tokens > len(output_ids):
            raise ExternalSGLangGenerationError("SGLang completion count exceeds output IDs")
        generated = output_ids[-completion_tokens:] if completion_tokens else []
        if any(type(token_id) is not int or token_id < 0 for token_id in generated):
            raise ExternalSGLangGenerationError("SGLang output IDs are invalid")
        generated_ids = tuple(cast(list[int], generated))
        finish_type = finish.get("type")
        if not isinstance(finish_type, str) or not finish_type:
            raise ExternalSGLangGenerationError("SGLang finish type is invalid")
        matched = finish.get("matched")
        stop_ids: tuple[int, ...] = ()
        content_ids = generated_ids
        if type(matched) is int and generated_ids and generated_ids[-1] == matched:
            content_ids = generated_ids[:-1]
            stop_ids = (matched,)
        return content_ids, stop_ids, finish_type, prompt_tokens


async def _await_thread_result(
    future: Future[tuple[int, JsonValue]],
) -> tuple[int, JsonValue]:
    while not future.done():
        await sleep(0.001)
    return future.result()


__all__ = [
    "ExternalSGLangGenerationError",
    "ExternalSGLangRolloutConfig",
    "ExternalSGLangRolloutGenerator",
]
