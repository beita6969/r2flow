from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class PackedGradients:
    names: tuple[tuple[str, ...], ...]
    buffers: tuple[torch.Tensor, ...]

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.buffers)

    def on_device(self, parameters: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result = {}
        for names, buffer in zip(self.names, self.buffers, strict=True):
            flat = buffer.to(parameters[names[0]].device)
            result.update(unpack_bucket(names, flat, parameters))
        return result


def pack_gradients(tensors: Mapping[str, torch.Tensor]) -> PackedGradients:
    names = tuple(tensor_buckets(tensors))
    return PackedGradients(
        names,
        tuple(flatten_bucket(bucket, tensors).to("cpu") for bucket in names),
    )


def tensor_buckets(
    tensors: Mapping[str, torch.Tensor],
    *,
    maximum_bytes: int = 8 * 1024 * 1024,
) -> Iterator[tuple[str, ...]]:
    if maximum_bytes < 1:
        raise ValueError("bucket capacity must be positive")
    names: list[str] = []
    size = 0
    identity = None
    for name in sorted(tensors):
        value = tensors[name]
        next_identity = (value.device, value.dtype)
        count = value.numel() * value.element_size()
        if names and (next_identity != identity or size + count > maximum_bytes):
            yield tuple(names)
            names, size = [], 0
        names.append(name)
        size += count
        identity = next_identity
    if names:
        yield tuple(names)


def flatten_bucket(names: tuple[str, ...], tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensors[name].detach().reshape(-1) for name in names])


def unpack_bucket(
    names: tuple[str, ...],
    flat: torch.Tensor,
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    chunks = flat.split([tensors[name].numel() for name in names])
    return {name: chunk.view_as(tensors[name]) for name, chunk in zip(names, chunks, strict=True)}
