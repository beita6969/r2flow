from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from skillev.contracts import JsonValue

if TYPE_CHECKING:
    from torch import nn


@dataclass(frozen=True, slots=True)
class ZInitializationSpec:
    mode: str
    epsilon: float
    initial_seed: int = 0
    reset_on_mutation: bool = True

    def __post_init__(self) -> None:
        if self.mode != "output-bias-log-epsilon@1":
            raise ValueError("unsupported Z initialization")
        if type(self.initial_seed) is not int or not 0 <= self.initial_seed < 2**64:
            raise ValueError("invalid Z initialization seed")
        if self.reset_on_mutation is not True:
            raise ValueError("Z initialization must also apply at mutation resets")
        if (
            isinstance(self.epsilon, bool)
            or not isinstance(self.epsilon, int | float)
            or not math.isfinite(self.epsilon)
            or self.epsilon <= 0
        ):
            raise ValueError("log-epsilon initialization requires positive epsilon")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "mode": self.mode,
            "epsilon": self.epsilon,
            "initial_seed": self.initial_seed,
            "reset_on_mutation": self.reset_on_mutation,
        }

    @classmethod
    def from_value(cls, value: object) -> ZInitializationSpec:
        if not isinstance(value, dict):
            raise ValueError("Z initialization must be an object")
        return cls(**value)

    def initialize(self, module: nn.Module, *, seed: int) -> None:
        import torch
        from torch import nn

        layers = [child for child in module.modules() if isinstance(child, nn.Linear)]
        if not layers or layers[-1].out_features != 1 or layers[-1].bias is None:
            raise ValueError("Z initialization requires a scalar final affine head")
        generator = torch.Generator(device=layers[0].weight.device).manual_seed(seed)
        for child in layers:
            nn.init.kaiming_uniform_(child.weight, a=math.sqrt(5), generator=generator)
            if child.bias is not None:
                bound = 1 / math.sqrt(child.in_features)
                nn.init.uniform_(child.bias, -bound, bound, generator=generator)
        with torch.no_grad():
            layers[-1].weight.zero_()
            layers[-1].bias.fill_(math.log(self.epsilon))
