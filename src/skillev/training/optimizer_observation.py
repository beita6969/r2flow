from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from skillev.contracts import JsonValue
from skillev.policy import PolicyParameterGroups


def _norm(tensors: tuple[torch.Tensor, ...]) -> float | None:
    value = math.hypot(*(float(torch.linalg.vector_norm(t)) for t in tensors))
    return value if math.isfinite(value) else None


def _nonfinite(tensors: tuple[torch.Tensor, ...]) -> int:
    return sum(int((~torch.isfinite(t)).sum()) for t in tensors)


def _copies(parameters: tuple[torch.nn.Parameter, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(p.detach().to(device="cpu", dtype=torch.float64, copy=True) for p in parameters)


def _adam_state(
    optimizer: torch.optim.Optimizer, parameters: tuple[torch.nn.Parameter, ...]
) -> dict[str, JsonValue] | None:
    if not isinstance(optimizer, torch.optim.Adam | torch.optim.AdamW):
        return None
    states = [optimizer.state.get(p, {}) for p in parameters]
    moments: dict[str, JsonValue] = {}
    for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        values = [
            state[name].detach() for state in states if isinstance(state.get(name), torch.Tensor)
        ]
        moments[name] = {
            "tensor_count": len(values),
            "element_count": sum(t.numel() for t in values),
            "nonfinite_count": _nonfinite(tuple(values)) if values else None,
        }
    steps: list[float] = []
    invalid = 0
    missing = 0
    for state in states:
        step = state.get("step")
        if step is None:
            missing += 1
            continue
        if isinstance(step, torch.Tensor) and step.numel() == 1:
            step = step.item()
        if not isinstance(step, int | float) or not math.isfinite(step):
            invalid += 1
        else:
            steps.append(float(step))
    return {
        "state_present_count": sum(bool(state) for state in states),
        "missing_state_count": sum(not state for state in states),
        "moments": moments,
        "step_min": min(steps) if steps else None,
        "step_max": max(steps) if steps else None,
        "step_invalid_count": invalid,
        "step_missing_count": missing,
    }


@dataclass(frozen=True)
class _ComponentBefore:
    parameters: tuple[torch.nn.Parameter, ...]
    values: tuple[torch.Tensor, ...]
    adam: dict[str, JsonValue] | None
    no_grad_count: int


@dataclass(frozen=True)
class OptimizerTransitionObservation:
    optimizer: torch.optim.Optimizer
    components: dict[str, _ComponentBefore]

    @classmethod
    def capture(
        cls, optimizer: torch.optim.Optimizer, groups: PolicyParameterGroups
    ) -> OptimizerTransitionObservation:
        components = {}
        named = [("forward", groups.forward), ("backward", groups.backward), ("z", groups.z_head)]
        if groups.psi_head:
            named.append(("psi", groups.psi_head))
        for name, group in named:
            parameters = tuple(p for p in group if p.requires_grad)
            components[name] = _ComponentBefore(
                parameters,
                _copies(parameters),
                _adam_state(optimizer, parameters),
                sum(p.grad is None for p in parameters),
            )
        return cls(optimizer, components)

    def finish(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {}
        for name, before in self.components.items():
            after = _copies(before.parameters)
            differences = tuple(a - b for a, b in zip(after, before.values, strict=True))
            pre_norm, post_norm, update = _norm(before.values), _norm(after), _norm(differences)
            relative = update / pre_norm if update is not None and pre_norm else None
            if relative is not None and not math.isfinite(relative):
                relative = None
            result[name] = {
                "parameters": {
                    "tensor_count": len(before.parameters),
                    "element_count": sum(p.numel() for p in before.parameters),
                    "no_gradient_tensor_count": before.no_grad_count,
                    "before_l2": pre_norm,
                    "after_l2": post_norm,
                    "update_l2": update,
                    "relative_update_l2": relative,
                    "before_nonfinite_count": _nonfinite(before.values),
                    "after_nonfinite_count": _nonfinite(after),
                    "update_nonfinite_count": _nonfinite(differences),
                },
                "adam": {
                    "before": before.adam,
                    "after": _adam_state(self.optimizer, before.parameters),
                },
            }
        return {
            "format": "optimizer-transition-observation@1",
            "scope": "same-single-optimizer-step;trainable-F/B/Z"
            + ("/psi" if "psi" in self.components else "")
            + ";CPU-float64-differences",
            "optimizer_type": type(self.optimizer).__name__,
            "components": result,
        }
