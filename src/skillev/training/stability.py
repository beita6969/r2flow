from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from skillev.contracts import JsonValue

if TYPE_CHECKING:
    from skillev.policy import PolicyParameterGroups


@dataclass(frozen=True, slots=True)
class PolicyStabilityConfig:
    reference_id: str | None = None
    coefficient: float = 0.0
    forward_max_norm: float | None = None
    backward_max_norm: float | None = None
    z_max_norm: float | None = None
    format: str = "reference-categorical-kl-group-clip@1"

    def __post_init__(self) -> None:
        if self.format != "reference-categorical-kl-group-clip@1":
            raise ValueError("unsupported stability condition")
        if self.coefficient != 0 or self.reference_id is not None:
            raise ValueError("the stability condition declares gradient-group clipping only")
        for value in (self.forward_max_norm, self.backward_max_norm, self.z_max_norm):
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError("clip threshold must be positive or disabled")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "reference_id": self.reference_id,
            "coefficient": self.coefficient,
            "forward_max_norm": self.forward_max_norm,
            "backward_max_norm": self.backward_max_norm,
            "z_max_norm": self.z_max_norm,
        }

    @classmethod
    def from_value(cls, value: object) -> PolicyStabilityConfig:
        if not isinstance(value, dict):
            raise TypeError("stability config must be an object")
        return cls(**value)


def clip_full_batch_groups(
    parameters: PolicyParameterGroups, config: PolicyStabilityConfig
) -> dict[str, JsonValue]:
    import torch

    report: dict[str, JsonValue] = {}
    for name, values, maximum in (
        ("forward", parameters.forward, config.forward_max_norm),
        ("backward", parameters.backward, config.backward_max_norm),
        ("z", parameters.z_head, config.z_max_norm),
    ):
        norm = math.sqrt(
            math.fsum(
                float(p.grad.detach().double().square().sum()) for p in values if p.grad is not None
            )
        )
        if not math.isfinite(norm):
            raise ValueError("nonfinite full-batch gradient norm")
        scale = 1.0 if maximum is None or norm <= maximum else maximum / norm
        if scale < 1:
            with torch.no_grad():
                for parameter in values:
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)
        report[name] = {"pre_clip_norm": norm, "scale": scale, "threshold": maximum}
    return report
