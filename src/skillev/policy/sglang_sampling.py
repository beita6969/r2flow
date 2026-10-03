from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from torch import Tensor


class _SamplingInfo(Protocol):
    sampling_seed: Tensor | None
    top_ks: Tensor
    top_ps: Tensor
    min_ps: Tensor


def _safe_seeded_raw_sample(
    upstream: Any,
    probs: Tensor,
    *,
    sampling_seed: Tensor,
    positions: Tensor,
) -> Tensor:
    import torch

    return _safe_seeded_logprob_sample(
        upstream,
        probs.to(torch.float64).log(),
        sampling_seed=sampling_seed,
        positions=positions,
    )


def _safe_seeded_logprob_sample(
    upstream: Any,
    logprobs: Tensor,
    *,
    sampling_seed: Tensor,
    positions: Tensor,
) -> Tensor:
    import torch

    finite = torch.isfinite(logprobs)
    torch._assert_async(finite.any(dim=-1).all(), "No finite sampling log probability")
    seeds = sampling_seed.to(torch.uint64)
    columns = torch.arange(logprobs.shape[-1], device=logprobs.device)
    hashed = upstream.murmur_hash32(seeds, positions, columns)
    uniform = hashed.to(torch.float64) / torch.iinfo(torch.uint32).max
    below_one = torch.nextafter(
        torch.ones((), dtype=torch.float64, device=logprobs.device),
        torch.zeros((), dtype=torch.float64, device=logprobs.device),
    )
    torch.minimum(uniform, below_one, out=uniform)
    uniform.log_().clamp_(min=torch.finfo(uniform.dtype).min).neg_()
    uniform.log_().neg_()
    scores = torch.where(
        finite,
        logprobs.to(torch.float64) + uniform,
        torch.full_like(uniform, -torch.inf),
    )
    return torch.argmax(scores, dim=1).to(torch.int32)


def install_raw_sampling_compatibility() -> None:
    import torch

    upstream = importlib.import_module("sglang.srt.layers.sampler")
    sampler = upstream.Sampler
    _install_logprob_sampling(upstream)
    original = sampler._sample_from_probs
    if getattr(original, "_skillev_raw_request_seeds", False):
        return

    @wraps(original)
    def sample(
        self: object,
        probs: Tensor,
        sampling_info: _SamplingInfo,
        positions: Tensor,
        simple_sampling_case: bool,
    ) -> Tensor:
        filtered = cast(
            "Tensor", original(self, probs, sampling_info, positions, simple_sampling_case)
        )
        if (
            sampling_info.sampling_seed is None
            or upstream.get_flags().sampling_backend != "pytorch"
        ):
            return filtered
        unfiltered_rows = (
            (sampling_info.top_ks >= probs.shape[-1])
            & (sampling_info.top_ps >= 1)
            & (sampling_info.min_ps <= 0)
        ).view(-1)
        raw = _safe_seeded_raw_sample(
            upstream,
            probs,
            sampling_seed=sampling_info.sampling_seed,
            positions=positions,
        )
        return torch.where(unfiltered_rows, raw, filtered)

    sample._skillev_raw_request_seeds = True
    sampler._sample_from_probs = sample


def _install_logprob_sampling(upstream: Any) -> None:
    sampler = upstream.Sampler
    original = sampler._sample_from_logprobs
    if getattr(original, "_skillev_safe_logprob_seeds", False):
        return

    @wraps(original)
    def sample_logprobs(
        self: object,
        logprobs: Tensor,
        sampling_info: _SamplingInfo,
        positions: Tensor,
    ) -> Tensor:
        if (
            sampling_info.sampling_seed is None
            or upstream.get_flags().sampling_backend != "pytorch"
        ):
            return cast("Tensor", original(self, logprobs, sampling_info, positions))
        return _safe_seeded_logprob_sample(
            upstream,
            logprobs,
            sampling_seed=sampling_info.sampling_seed,
            positions=positions,
        )

    sample_logprobs._skillev_safe_logprob_seeds = True
    sampler._sample_from_logprobs = sample_logprobs


class _RequestParameters(Protocol):
    sampling_seed: int | None


class _ScheduledRequest(Protocol):
    sampling_params: _RequestParameters


class _ScheduledBatch(Protocol):
    reqs: Sequence[_ScheduledRequest]


def install_unsigned_seed_compatibility() -> None:
    upstream = importlib.import_module("sglang.srt.sampling.sampling_batch_info")
    batch_info = upstream.SamplingBatchInfo
    original = batch_info.from_schedule_batch.__func__
    if getattr(original, "_skillev_unsigned_request_seeds", False):
        return

    @wraps(original)
    def from_schedule_batch(cls: type[object], batch: _ScheduledBatch, vocab_size: int) -> object:
        restored: list[tuple[_RequestParameters, int]] = []
        try:
            for request in batch.reqs:
                parameters = request.sampling_params
                seed = parameters.sampling_seed
                if isinstance(seed, int) and (1 << 63) <= seed < (1 << 64):
                    restored.append((parameters, seed))
                    parameters.sampling_seed = seed - (1 << 64)
            return original(cls, batch, vocab_size)
        finally:
            for parameters, seed in restored:
                parameters.sampling_seed = seed

    from_schedule_batch._skillev_unsigned_request_seeds = True
    batch_info.from_schedule_batch = classmethod(from_schedule_batch)
