from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from skillev.contracts import JsonValue, normalize_json

if TYPE_CHECKING:
    from torch import nn

FLOW_HEAD_FEATURE: Final = "forward-reasoning-prompt-last-hidden@1"
FLOW_HEAD_INITIALIZATION: Final = "output-bias-eta-log-epsilon@1"
FLOW_HEAD_DOMAIN_OFFSET_ID: Final = "psi-mlp-forward-reasoning-prompt-last-hidden+domain-offset@5"
FLOW_OFFSET_INITIALIZATION: Final = "negative-median-delta0T-first-batch-no-grad@1"
FLOW_OFFSET_TRACKING_DAMPING: Final = 0.3
FLOW_OFFSET_STEP_CAP: Final = 5.0
FLOW_OFFSET_CAP_HORIZON_UNIT: Final = 5
FLOW_OFFSET_LEARNING_RATE: Final = 0.1
FLOW_OFFSET_FORMAT: Final = "flow-domain-offsets@1"
FLOW_OFFSET_FILE: Final = "flow-domain-offsets.json"
FLOW_OFFSET_PARAMETER: Final = "flow_offsets.domain_offset"
PSI_SEED_SALT: Final = 0x5053495F48454144


@dataclass(frozen=True, slots=True)
class FlowHeadSpec:
    hidden_width: int
    eta: float
    epsilon: float
    initial_seed: int = 0
    feature: str = FLOW_HEAD_FEATURE
    initialization: str = FLOW_HEAD_INITIALIZATION

    def __post_init__(self) -> None:
        if self.feature != FLOW_HEAD_FEATURE:
            raise ValueError("unsupported flow-head feature")
        if self.initialization != FLOW_HEAD_INITIALIZATION:
            raise ValueError("unsupported flow-head initialization")
        if type(self.hidden_width) is not int or self.hidden_width <= 0:
            raise ValueError("flow-head width must be a positive integer")
        for name in ("eta", "epsilon"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"flow-head {name} must be finite and positive")
            object.__setattr__(self, name, float(value))
        if type(self.initial_seed) is not int or not 0 <= self.initial_seed < 2**64:
            raise ValueError("invalid flow-head seed")

    @property
    def initial_log_flow(self) -> float:
        return self.eta * math.log(self.epsilon)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "epsilon": self.epsilon,
            "eta": self.eta,
            "feature": self.feature,
            "hidden_width": self.hidden_width,
            "initial_seed": self.initial_seed,
            "initialization": self.initialization,
        }

    @classmethod
    def from_value(cls, value: object) -> FlowHeadSpec:
        normalized = normalize_json(value)
        fields = {"epsilon", "eta", "feature", "hidden_width", "initial_seed", "initialization"}
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("flow-head spec has incompatible fields")
        if any(type(normalized[name]) is not str for name in ("feature", "initialization")):
            raise TypeError("flow-head identifiers must be text")
        if any(type(normalized[name]) is not int for name in ("hidden_width", "initial_seed")):
            raise TypeError("flow-head integer fields must be integers")
        eta, epsilon = normalized["eta"], normalized["epsilon"]
        if isinstance(eta, bool) or not isinstance(eta, int | float):
            raise TypeError("flow-head eta must be numeric")
        if isinstance(epsilon, bool) or not isinstance(epsilon, int | float):
            raise TypeError("flow-head epsilon must be numeric")
        return cls(
            hidden_width=int(normalized["hidden_width"]),
            eta=float(eta),
            epsilon=float(epsilon),
            initial_seed=int(normalized["initial_seed"]),
            feature=str(normalized["feature"]),
            initialization=str(normalized["initialization"]),
        )

    def build(self, hidden_size: int) -> nn.Sequential:
        from torch import nn

        return nn.Sequential(
            nn.Linear(hidden_size, self.hidden_width),
            nn.GELU(),
            nn.Linear(self.hidden_width, 1),
        )

    def initialize(self, module: nn.Module, *, seed: int) -> None:
        import torch
        from torch import nn

        layers = [child for child in module.modules() if isinstance(child, nn.Linear)]
        if not layers or layers[-1].out_features != 1 or layers[-1].bias is None:
            raise ValueError("flow-head initialization requires a scalar final affine head")
        generator = torch.Generator(device=layers[0].weight.device).manual_seed(seed)
        for child in layers:
            nn.init.kaiming_uniform_(child.weight, a=math.sqrt(5), generator=generator)
            if child.bias is not None:
                bound = 1 / math.sqrt(child.in_features)
                nn.init.uniform_(child.bias, -bound, bound, generator=generator)
        with torch.no_grad():
            layers[-1].weight.zero_()
            layers[-1].bias.fill_(self.initial_log_flow)


def psi_reset_seed(seed: int) -> int:
    return seed ^ PSI_SEED_SALT


__all__ = [
    "FLOW_HEAD_DOMAIN_OFFSET_ID",
    "FLOW_HEAD_FEATURE",
    "FLOW_HEAD_INITIALIZATION",
    "FLOW_OFFSET_CAP_HORIZON_UNIT",
    "FLOW_OFFSET_FILE",
    "FLOW_OFFSET_FORMAT",
    "FLOW_OFFSET_INITIALIZATION",
    "FLOW_OFFSET_LEARNING_RATE",
    "FLOW_OFFSET_PARAMETER",
    "FLOW_OFFSET_STEP_CAP",
    "FLOW_OFFSET_TRACKING_DAMPING",
    "PSI_SEED_SALT",
    "FlowHeadSpec",
    "psi_reset_seed",
]
