from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

if TYPE_CHECKING:
    from .token_mask import TokenMask


class _ChunkedTargetLogprobs(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(logits, targets)
        result = torch.empty(targets.shape, dtype=torch.float32, device=logits.device)
        for start in range(0, len(targets), 16):
            stop = min(start + 16, len(targets))
            values = logits[start:stop].float()
            result[start:stop] = (
                torch.log_softmax(values, -1).gather(-1, targets[start:stop, None]).squeeze(-1)
            )
        return result

    @staticmethod
    def backward(ctx: Any, upstream: torch.Tensor) -> tuple[torch.Tensor, None]:
        logits, targets = ctx.saved_tensors
        gradient = torch.empty_like(logits)
        for start in range(0, len(targets), 16):
            stop = min(start + 16, len(targets))
            values = -torch.softmax(logits[start:stop].float(), -1)
            values.scatter_add_(
                -1,
                targets[start:stop, None],
                torch.ones_like(targets[start:stop, None], dtype=values.dtype),
            )
            values.mul_(upstream[start:stop, None])
            gradient[start:stop] = values.to(logits.dtype)
        return gradient, None


def _masked(values: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    return values.masked_fill(~allowed, float("-inf"))


class _MaskedChunkedTargetLogprobs(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, logits: torch.Tensor, targets: torch.Tensor, mask: TokenMask
    ) -> torch.Tensor:
        ctx.save_for_backward(logits, targets)
        ctx.mask = mask
        result = torch.empty(targets.shape, dtype=torch.float32, device=logits.device)
        for start in range(0, len(targets), 16):
            stop = min(start + 16, len(targets))
            values = _masked(logits[start:stop].float(), mask.allowed(start, stop, logits.device))
            result[start:stop] = (
                torch.log_softmax(values, -1).gather(-1, targets[start:stop, None]).squeeze(-1)
            )
        return result

    @staticmethod
    def backward(ctx: Any, upstream: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        logits, targets = ctx.saved_tensors
        mask = ctx.mask
        gradient = torch.empty_like(logits)
        for start in range(0, len(targets), 16):
            stop = min(start + 16, len(targets))
            allowed = mask.allowed(start, stop, logits.device)
            values = -torch.softmax(_masked(logits[start:stop].float(), allowed), -1)
            values.scatter_add_(
                -1,
                targets[start:stop, None],
                torch.ones_like(targets[start:stop, None], dtype=values.dtype),
            )
            values.mul_(upstream[start:stop, None])
            gradient[start:stop] = values.to(logits.dtype)
        return gradient, None, None


def _require_finite_targets(values: torch.Tensor) -> torch.Tensor:
    if not bool(torch.isfinite(values).all()):
        raise ValueError("masked action log-probability is not finite at a scored target")
    return values


def action_token_logprobs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    implementation: str = "reference",
    mask: TokenMask | None = None,
) -> torch.Tensor:
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("action probabilities require one target per original action position")
    if mask is not None:
        return _masked_action_token_logprobs(logits, targets, implementation, mask)
    if implementation == "reference":
        return torch.log_softmax(logits.float(), -1).gather(-1, targets[:, None]).squeeze(-1)
    if implementation == "chunked-target@1":
        return cast(torch.Tensor, _ChunkedTargetLogprobs.apply(logits, targets))
    raise ValueError("unsupported action log-probability execution implementation")


def _masked_action_token_logprobs(
    logits: torch.Tensor, targets: torch.Tensor, implementation: str, mask: TokenMask
) -> torch.Tensor:
    if logits.shape[1] < mask.vocab_size:
        raise ValueError("logits do not cover the token mask vocabulary")
    mask.require_targets(tuple(int(token) for token in targets.tolist()))
    if logits.shape[1] > mask.vocab_size:
        logits = logits[:, : mask.vocab_size]
    if implementation == "reference":
        values = _masked(logits.float(), mask.allowed(0, len(targets), logits.device))
        result = torch.log_softmax(values, -1).gather(-1, targets[:, None]).squeeze(-1)
    elif implementation == "chunked-target@1":
        result = cast(
            torch.Tensor,
            _MaskedChunkedTargetLogprobs.apply(logits, targets, mask),
        )
    else:
        raise ValueError("unsupported action log-probability execution implementation")
    return _require_finite_targets(result)


class _ChunkedHiddenLogprobs(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        mask: TokenMask | None,
        chunk: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(hidden, weight, targets)
        ctx.mask = mask
        ctx.chunk = chunk
        result = torch.empty(targets.shape, dtype=torch.float32, device=hidden.device)
        for start in range(0, len(targets), chunk):
            stop = min(start + chunk, len(targets))
            values = _hidden_chunk_logits(hidden, weight, mask, start, stop)
            result[start:stop] = (
                torch.log_softmax(values, -1).gather(-1, targets[start:stop, None]).squeeze(-1)
            )
        return result

    @staticmethod
    def backward(ctx: Any, upstream: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None]:
        hidden, weight, targets = ctx.saved_tensors
        gradient = torch.empty_like(hidden)
        for start in range(0, len(targets), ctx.chunk):
            stop = min(start + ctx.chunk, len(targets))
            values = -torch.softmax(_hidden_chunk_logits(hidden, weight, ctx.mask, start, stop), -1)
            values.scatter_add_(
                -1,
                targets[start:stop, None],
                torch.ones_like(targets[start:stop, None], dtype=values.dtype),
            )
            values.mul_(upstream[start:stop, None])
            gradient[start:stop] = (values.to(weight.dtype) @ weight).to(hidden.dtype)
        return gradient, None, None, None, None


def _hidden_chunk_logits(
    hidden: torch.Tensor, weight: torch.Tensor, mask: TokenMask | None, start: int, stop: int
) -> torch.Tensor:
    values = (hidden[start:stop].to(weight.dtype) @ weight.T).float()
    if mask is None:
        return values
    return _masked(values, mask.allowed(start, stop, values.device))


def hidden_token_logprobs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: TokenMask | None = None,
    chunk: int = 64,
) -> torch.Tensor:
    if (
        hidden.ndim != 2
        or weight.ndim != 2
        or targets.ndim != 1
        or hidden.shape[0] != targets.shape[0]
        or hidden.shape[1] != weight.shape[1]
    ):
        raise ValueError("hidden log-probabilities require one hidden row per scored target")
    if type(chunk) is not int or chunk < 1:
        raise ValueError("chunk size must be a positive integer")
    if weight.requires_grad:
        raise ValueError("chunked-hidden@1 requires a frozen output head")
    if mask is not None:
        if weight.shape[0] < mask.vocab_size:
            raise ValueError("output head does not cover the token mask vocabulary")
        mask.require_targets(tuple(int(token) for token in targets.tolist()))
        weight = weight[: mask.vocab_size]
    result = cast(
        torch.Tensor,
        _ChunkedHiddenLogprobs.apply(hidden, weight, targets, mask, chunk),
    )
    return _require_finite_targets(result)
