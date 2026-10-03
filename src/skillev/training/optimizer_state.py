from __future__ import annotations

from typing import Any

import torch

from skillev.policy import PolicyBackbone

_LAYOUT = "skillev_parameter_layout"


def optimizer_parameter_layout(
    backbone: PolicyBackbone, optimizer: torch.optim.Optimizer
) -> list[dict[str, object]]:
    named = backbone.named_trainable_parameters()
    by_id = {id(parameter): name for name, parameter in named.items()}
    if len(by_id) != len(named):
        raise ValueError("trainable parameter names alias one tensor")
    seen: set[int] = set()
    layout: list[dict[str, object]] = []
    saved_groups = optimizer.state_dict()["param_groups"]
    for group, saved in zip(optimizer.param_groups, saved_groups, strict=True):
        parameters = []
        for parameter in group["params"]:
            if id(parameter) not in by_id or id(parameter) in seen:
                raise ValueError("optimizer parameters are unknown or repeated")
            seen.add(id(parameter))
            parameters.append(
                {
                    "name": by_id[id(parameter)],
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                }
            )
        layout.append(
            {
                "group_name": group.get("name"),
                "parameters": parameters,
                "parameter_ids": saved["params"],
            }
        )
    return layout


def checkpoint_optimizer_state(
    backbone: PolicyBackbone, optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    state = optimizer.state_dict()
    state[_LAYOUT] = optimizer_parameter_layout(backbone, optimizer)
    return state


def require_optimizer_state_layout(
    state: dict[str, Any], backbone: PolicyBackbone, optimizer: torch.optim.Optimizer
) -> None:
    expected = optimizer_parameter_layout(backbone, optimizer)
    if state.get(_LAYOUT) != expected:
        raise ValueError("optimizer checkpoint lacks the matching named parameter layout")
    groups = state["param_groups"]
    if len(groups) != len(optimizer.param_groups):
        raise ValueError("optimizer checkpoint group count differs")
    seen: set[int] = set()
    for saved, current, layout in zip(groups, optimizer.param_groups, expected, strict=True):
        if saved.get("stability_condition") != current.get("stability_condition"):
            raise ValueError("optimizer checkpoint stability condition differs")
        if saved.get("name") != current.get("name") or len(saved["params"]) != len(
            current["params"]
        ):
            raise ValueError("optimizer checkpoint group membership differs")
        if saved["params"] != layout["parameter_ids"]:
            raise ValueError("optimizer state IDs differ from the named parameter layout")
        for saved_id, parameter in zip(saved["params"], current["params"], strict=True):
            if saved_id in seen:
                raise ValueError("optimizer checkpoint repeats a parameter")
            seen.add(saved_id)
            moments = state["state"].get(saved_id, {})
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if name not in moments:
                    continue
                tensor = moments[name]
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.shape != parameter.shape
                    or tensor.dtype != parameter.dtype
                    or not torch.isfinite(tensor).all()
                ):
                    raise ValueError("optimizer checkpoint moment differs from its parameter")
    if set(state["state"]) - seen:
        raise ValueError("optimizer checkpoint has state outside its parameter groups")
