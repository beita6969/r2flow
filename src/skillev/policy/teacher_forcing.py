from __future__ import annotations

import math
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Final

import torch

from .action_logprobs import action_token_logprobs, hidden_token_logprobs
from .interface import AdapterRole, PolicyScoringMemoryError
from .scoring_execution import ActivationOffload, TeacherForcingConfig
from .token_mask import TokenMask

SEQUENCE_SCORING_IMPLEMENTATION: Final = "chunked-hidden@1"


def detached_scalar_values(values: list[torch.Tensor]) -> list[float]:
    return [float(value) for value in torch.stack(values).cpu().tolist()]


def backward_action_logprob_means(
    scores: tuple[torch.Tensor, ...], lengths: tuple[int, ...], *, forward: bool
) -> tuple[float, ...]:
    means = torch.stack([score.sum() / k for score, k in zip(scores, lengths, strict=True)])
    signed = means.sum() if forward else -means.sum()
    signed.backward()
    return tuple(float(value) for value in means.detach().cpu().tolist())


@dataclass(frozen=True, slots=True)
class ScoredSpan:
    start: int
    stop: int
    mask: TokenMask | None = None

    def __post_init__(self) -> None:
        if type(self.start) is not int or type(self.stop) is not int or not 1 <= self.start:
            raise ValueError("scored spans need a non-empty conditioning prefix")
        if self.stop <= self.start:
            raise ValueError("scored spans must be non-empty")
        if self.mask is not None and self.mask.rows != self.stop - self.start:
            raise ValueError("span mask rows must equal the span length")


@dataclass(frozen=True, slots=True)
class ScoringSequence:
    token_ids: tuple[int, ...]
    spans: tuple[ScoredSpan, ...]
    feature_position: int | None = None

    def __post_init__(self) -> None:
        if not self.token_ids or (not self.spans and self.feature_position is None):
            raise ValueError("a scoring sequence needs tokens and a span or a feature position")
        if any(type(token) is not int or token < 0 for token in self.token_ids):
            raise ValueError("scoring sequence token ids must be non-negative integers")
        if any(span.stop > len(self.token_ids) for span in self.spans):
            raise ValueError("scored span exceeds the sequence")
        if self.feature_position is not None and not (
            type(self.feature_position) is int and 0 <= self.feature_position < len(self.token_ids)
        ):
            raise ValueError("feature position outside the sequence")


@dataclass(frozen=True, slots=True)
class SequenceScores:
    spans: tuple[torch.Tensor, ...]
    feature: torch.Tensor | None
    unmasked: tuple[torch.Tensor | None, ...] = ()


def backward_signed_span_sums(
    scores: SequenceScores, signs: tuple[float, ...]
) -> tuple[float, ...]:
    if len(signs) != len(scores.spans) or not all(math.isfinite(sign) for sign in signs):
        raise ValueError("one finite sign is required per scored span")
    sums = [span.double().sum() for span in scores.spans]
    torch.stack([sign * value for sign, value in zip(signs, sums, strict=True)]).sum().backward()
    return tuple(float(value) for value in torch.stack(sums).detach().cpu().tolist())


def _decoder_and_head(model: Any) -> tuple[Any, torch.Tensor]:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    head = base.get_output_embeddings()
    if head is None or getattr(head, "bias", None) is not None:
        raise RuntimeError("sequence scoring requires a bias-free output head")
    return base.get_decoder(), head.weight


class TeacherForcingExecutor:
    def __init__(self, config: TeacherForcingConfig) -> None:
        self.config = config
        self.offload = ActivationOffload(config.pinned_memory_bytes)
        self.metrics: list[dict[str, Any]] = []
        self._profile_start: torch.cuda.Event | None = None
        self._session = False
        self._session_checkpointed: bool | None = None
        self._resident_model: int | None = None
        self._offload_before = (0, 0, 0.0, 0.0, 0)

    @contextmanager
    def session(self, model: Any) -> Iterator[None]:
        if self._session:
            raise RuntimeError("teacher-forcing sessions cannot overlap")
        previous_training = model.training
        self._session = True
        self._session_checkpointed = None
        try:
            yield
        finally:
            self._session = False
            self._session_checkpointed = None
            if model.training != previous_training:
                model.train(previous_training)

    def finish_backward_profile(self) -> None:
        if self.metrics:
            saved, restored, packed, unpacked, resident = self._offload_before
            self.metrics[-1].update(
                offload_saved_bytes=self.offload.saved_bytes - saved,
                offload_restored_bytes=self.offload.restored_bytes - restored,
                offload_pack_host_seconds=self.offload.pack_host_seconds - packed,
                offload_unpack_host_seconds=self.offload.unpack_host_seconds - unpacked,
                offload_resident_parameter_bytes=self.offload.resident_parameter_bytes - resident,
                offload_pinned_pool={
                    "allocated_bytes": self.offload.pool.allocated_bytes,
                    "live_bytes": self.offload.pool.live_bytes,
                    "peak_live_bytes": self.offload.pool.peak_live_bytes,
                    "allocations_cumulative": self.offload.pool.allocations,
                    "reuses_cumulative": self.offload.pool.reuses,
                    "fallbacks_cumulative": self.offload.pool.fallbacks,
                },
            )
        if self._profile_start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        end.synchronize()
        self.metrics[-1]["cuda_forward_backward_ms"] = self._profile_start.elapsed_time(end)
        self.metrics[-1]["cuda_owner_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        self._profile_start = None

    def score(
        self,
        *,
        model: Any,
        device: torch.device,
        edges: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
        role: AdapterRole,
        checkpoint_enabled: bool,
    ) -> tuple[torch.Tensor, ...]:
        if not edges or any(not p or not a for p, a in edges):
            raise ValueError("teacher forcing requires non-empty prefix/action pairs")
        lengths = [len(p) + len(a) for p, a in edges]
        width = max(lengths)
        if len(edges) > 1 and (
            len(edges) > self.config.microbatch_size
            or width * len(edges) > self.config.microbatch_max_tokens
        ):
            raise ValueError("edge batch exceeds the configured padded-token capacity")
        checkpointed = checkpoint_enabled and width >= self.config.checkpoint_min_tokens
        offloaded = checkpointed and width >= self.config.offload_min_tokens
        if offloaded and self._resident_model != id(model):
            self.offload.bind_resident_parameters(model.parameters())
            self._resident_model = id(model)
        if not self._session or self._session_checkpointed != checkpointed:
            model.train(checkpointed)
            self._session_checkpointed = checkpointed
        pad_id = getattr(model.config.get_text_config(), "pad_token_id", None)
        if pad_id is None:
            pad_id = 0
        inputs = torch.full((len(edges), width), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(inputs)
        for row, ((prefix, action), length) in enumerate(zip(edges, lengths, strict=True)):
            inputs[row, :length] = torch.tensor(prefix + action, dtype=torch.long, device=device)
            mask[row, :length] = 1
        context = self.offload.context() if offloaded else nullcontext()
        start = time.perf_counter()
        if self.config.profile_cuda and device.type == "cuda":
            self._profile_start = torch.cuda.Event(enable_timing=True)
            self._profile_start.record()
        saved_before = self.offload.saved_bytes
        resident_before = self.offload.resident_parameter_bytes
        pack_before = self.offload.pack_host_seconds
        self._offload_before = (
            saved_before,
            self.offload.restored_bytes,
            pack_before,
            self.offload.unpack_host_seconds,
            resident_before,
        )
        metric: dict[str, Any] = {
            "role": role.value,
            "edges": len(edges),
            "lengths": lengths,
            "prefix_tokens": sum(len(p) for p, _ in edges),
            "action_tokens": sum(len(a) for _, a in edges),
            "padded_tokens": width * len(edges),
            "checkpointed": checkpointed,
            "offloaded": offloaded,
        }
        try:
            with context:
                kwargs: dict[str, Any] = {
                    "input_ids": inputs,
                    "attention_mask": mask,
                    "use_cache": False,
                    "return_dict": True,
                }
                model_type = model.config.model_type
                if model_type in {"qwen3_5", "qwen3_5_text"}:
                    if len(edges) == 1:
                        kwargs["logits_to_keep"] = len(edges[0][1]) + 1
                        logits = model(**kwargs).logits
                        action_logits = (logits[0, :-1, :],)
                    else:
                        positions = sorted(
                            {j for p, a in edges for j in range(len(p) - 1, len(p) + len(a) - 1)}
                        )
                        kwargs["logits_to_keep"] = torch.tensor(positions, device=device)
                        logits = model(**kwargs).logits
                        lookup = {position: i for i, position in enumerate(positions)}
                        action_logits = tuple(
                            logits[
                                row, [lookup[j] for j in range(len(p) - 1, len(p) + len(a) - 1)], :
                            ]
                            for row, (p, a) in enumerate(edges)
                        )
                elif model_type == "gpt2" and device.type == "cpu":
                    logits = model(**kwargs).logits
                    action_logits = tuple(
                        logits[row, len(p) - 1 : len(p) + len(a) - 1, :]
                        for row, (p, a) in enumerate(edges)
                    )
                else:
                    raise RuntimeError("teacher forcing requires the fixed Qwen3.5 backend")
                return tuple(
                    action_token_logprobs(
                        values,
                        torch.tensor(action, dtype=torch.long, device=device),
                        implementation=self.config.action_logprobs,
                    )
                    for values, (_, action) in zip(action_logits, edges, strict=True)
                )
        except torch.OutOfMemoryError as error:
            raise PolicyScoringMemoryError(
                prefix_token_count=max(len(p) for p, _ in edges),
                action_token_count=max(len(a) for _, a in edges),
                role=role,
            ) from error
        finally:
            metric["forward_host_seconds"] = time.perf_counter() - start
            metric["offload_saved_bytes"] = self.offload.saved_bytes - saved_before
            metric["offload_resident_parameter_bytes"] = (
                self.offload.resident_parameter_bytes - resident_before
            )
            metric["offload_pack_host_seconds"] = self.offload.pack_host_seconds - pack_before
            self.metrics.append(metric)
            if not self._session:
                model.eval()

    def score_sequence(
        self,
        *,
        model: Any,
        device: torch.device,
        sequence: ScoringSequence,
        role: AdapterRole,
        checkpoint_enabled: bool,
    ) -> SequenceScores:
        width = len(sequence.token_ids)
        checkpointed = checkpoint_enabled and width >= self.config.checkpoint_min_tokens
        offloaded = checkpointed and width >= self.config.offload_min_tokens
        if offloaded and self._resident_model != id(model):
            self.offload.bind_resident_parameters(model.parameters())
            self._resident_model = id(model)
        if not self._session or self._session_checkpointed != checkpointed:
            model.train(checkpointed)
            self._session_checkpointed = checkpointed
        model_type = model.config.model_type
        if not (
            model_type in {"qwen3_5", "qwen3_5_text"}
            or (model_type == "gpt2" and device.type == "cpu")
        ):
            raise RuntimeError("teacher forcing requires the fixed Qwen3.5 backend")
        decoder, head_weight = _decoder_and_head(model)
        inputs = torch.tensor([sequence.token_ids], dtype=torch.long, device=device)
        context = self.offload.context() if offloaded else nullcontext()
        start = time.perf_counter()
        saved_before = self.offload.saved_bytes
        resident_before = self.offload.resident_parameter_bytes
        pack_before = self.offload.pack_host_seconds
        self._offload_before = (
            saved_before,
            self.offload.restored_bytes,
            pack_before,
            self.offload.unpack_host_seconds,
            resident_before,
        )
        scored = sum(span.stop - span.start for span in sequence.spans)
        metric: dict[str, Any] = {
            "role": role.value,
            "edges": 1,
            "lengths": [width],
            "prefix_tokens": width - scored,
            "action_tokens": scored,
            "padded_tokens": width,
            "checkpointed": checkpointed,
            "offloaded": offloaded,
            "implementation": SEQUENCE_SCORING_IMPLEMENTATION,
            "spans": len(sequence.spans),
            "scored_tokens": scored,
            "masked_rows": sum(0 if s.mask is None else s.mask.rows for s in sequence.spans),
        }
        try:
            with context:
                hidden = decoder(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state[0]
                spans = tuple(
                    hidden_token_logprobs(
                        hidden[span.start - 1 : span.stop - 1],
                        head_weight,
                        inputs[0, span.start : span.stop],
                        mask=span.mask,
                    )
                    for span in sequence.spans
                )
                with torch.no_grad():
                    unmasked = tuple(
                        None
                        if span.mask is None
                        else hidden_token_logprobs(
                            hidden[span.start - 1 : span.stop - 1].detach(),
                            head_weight,
                            inputs[0, span.start : span.stop],
                        )
                        for span in sequence.spans
                    )
                feature = (
                    None
                    if sequence.feature_position is None
                    else hidden[sequence.feature_position].detach().float().clone()
                )
                return SequenceScores(spans, feature, unmasked)
        except torch.OutOfMemoryError as error:
            raise PolicyScoringMemoryError(
                prefix_token_count=width - scored, action_token_count=scored, role=role
            ) from error
        finally:
            metric["forward_host_seconds"] = time.perf_counter() - start
            metric["offload_saved_bytes"] = self.offload.saved_bytes - saved_before
            metric["offload_resident_parameter_bytes"] = (
                self.offload.resident_parameter_bytes - resident_before
            )
            metric["offload_pack_host_seconds"] = self.offload.pack_host_seconds - pack_before
            self.metrics.append(metric)
            if not self._session:
                model.eval()
