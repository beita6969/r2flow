from __future__ import annotations

import time
import weakref
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

from .activation_pool import PinnedActivation, PinnedActivationPool


@dataclass(frozen=True, slots=True)
class TeacherForcingConfig:
    microbatch_size: int = 1
    microbatch_max_tokens: int = 8192
    checkpoint_min_tokens: int = 1
    offload_min_tokens: int = 32768
    pinned_memory_bytes: int = 0
    profile_cuda: bool = False
    action_logprobs: str = "reference"

    def __post_init__(self) -> None:
        if type(self.microbatch_size) is not int or self.microbatch_size not in (1, 2, 4):
            raise ValueError("edge microbatch size must be 1, 2 or 4")
        for value in (
            self.microbatch_max_tokens,
            self.checkpoint_min_tokens,
            self.offload_min_tokens,
        ):
            if type(value) is not int or value < 1:
                raise ValueError("teacher-forcing token thresholds must be positive")
        if self.offload_min_tokens < self.checkpoint_min_tokens:
            raise ValueError("activation offload must belong to the checkpointed tier")
        if type(self.pinned_memory_bytes) is not int or self.pinned_memory_bytes < 0:
            raise ValueError("pinned activation budget must be non-negative")
        if type(self.profile_cuda) is not bool:
            raise TypeError("CUDA profiling switch must be boolean")
        if self.action_logprobs not in {"reference", "chunked-target@1", "chunked-hidden@1"}:
            raise ValueError("unsupported target log-probability execution mode")


class ActivationOffload:
    def __init__(self, pinned_limit: int) -> None:
        self.limit = pinned_limit
        self.pool = PinnedActivationPool(pinned_limit)
        self._copy_streams: dict[torch.device, torch.cuda.Stream] = {}
        self.saved_bytes = 0
        self.restored_bytes = 0
        self.resident_parameter_bytes = 0
        self.pack_host_seconds = 0.0
        self.unpack_host_seconds = 0.0
        self._resident_storages: set[tuple[torch.device, int]] = set()

    def bind_resident_parameters(self, parameters: Iterable[torch.Tensor]) -> None:
        self._resident_storages = {
            (p.device, p.untyped_storage().data_ptr())
            for p in parameters
            if not p.requires_grad and p.device.type == "cuda"
        }

    @property
    def live_pinned_bytes(self) -> int:
        return self.pool.live_bytes

    @property
    def peak_pinned_bytes(self) -> int:
        return self.pool.peak_live_bytes

    def _copy_stream(self, device: torch.device) -> torch.cuda.Stream:
        if device not in self._copy_streams:
            self._copy_streams[device] = torch.cuda.Stream(device=device)
        return self._copy_streams[device]

    def pack(
        self, tensor: torch.Tensor
    ) -> tuple[torch.device, torch.Tensor, bool] | PinnedActivation:
        if tensor.device.type != "cuda":
            return tensor.device, tensor.detach(), False
        size = tensor.numel() * tensor.element_size()
        if (tensor.device, tensor.untyped_storage().data_ptr()) in self._resident_storages:
            self.resident_parameter_bytes += size
            return tensor.device, tensor.detach(), False
        started = time.perf_counter()
        saved = self.pool.acquire(tensor) if self.limit else None
        if saved is not None:
            copy = self._copy_stream(tensor.device)
            copy.wait_stream(torch.cuda.current_stream(tensor.device))
            with torch.no_grad(), torch.cuda.stream(copy):
                saved.copy_(tensor, non_blocking=True)
                ready = torch.cuda.Event()
                ready.record(copy)
            tensor.record_stream(copy)
            torch.cuda.current_stream(tensor.device).wait_event(ready)
            payload = PinnedActivation(tensor.device, saved, ready)
            weakref.finalize(payload, self.pool.release, saved, ready, payload.restores)
            self.saved_bytes += size
            self.pack_host_seconds += time.perf_counter() - started
            return payload
        saved = torch.empty_like(tensor, device="cpu", pin_memory=False)
        with torch.no_grad():
            saved.copy_(tensor)
        self.saved_bytes += size
        self.pack_host_seconds += time.perf_counter() - started
        return tensor.device, saved, False

    def unpack(
        self, payload: tuple[torch.device, torch.Tensor, bool] | PinnedActivation
    ) -> torch.Tensor:
        if isinstance(payload, PinnedActivation):
            started = time.perf_counter()
            copy = self._copy_stream(payload.device)
            copy.wait_event(payload.ready)
            with torch.cuda.stream(copy):
                result = payload.saved.to(payload.device, non_blocking=True)
                restored = torch.cuda.Event()
                restored.record(copy)
            current = torch.cuda.current_stream(payload.device)
            current.wait_event(restored)
            result.record_stream(current)
            payload.restores.append(restored)
            self.restored_bytes += payload.saved.numel() * payload.saved.element_size()
            self.unpack_host_seconds += time.perf_counter() - started
            return result
        device, saved, pinned = payload
        if saved.device == device:
            return saved
        started = time.perf_counter()
        if device.type == "cuda":
            self.restored_bytes += saved.numel() * saved.element_size()
        result = saved.to(device, non_blocking=pinned)
        self.unpack_host_seconds += time.perf_counter() - started
        return result

    def context(self) -> Any:
        return torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)
