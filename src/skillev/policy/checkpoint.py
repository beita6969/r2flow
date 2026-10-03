from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import torch
from peft import PeftModel
from peft.utils.save_and_load import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from safetensors.torch import load_file, save_file

from skillev.contracts import JsonValue, canonical_json, parse_canonical_json, stable_hash

from .interface import AdapterRole
from .trainable_state import TrainableStateIdentity

CHECKPOINT_FORMAT: Final = "skillev-policy-checkpoint@5"
POLICY_STATE_FILE: Final = "policy_state.json"
ADAPTER_FILE: Final = "adapter_model.safetensors"
Z_HEAD_FILE: Final = "z_head.pt"
PSI_HEAD_FILE: Final = "psi_head.pt"


def _adapter_directory(directory: Path, role: AdapterRole) -> Path:
    return directory / (
        "forward_adapter" if role is AdapterRole.FORWARD_POLICY else "backward_adapter"
    )


def _cpu_contiguous(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().to(device="cpu").contiguous()
        for name, tensor in sorted(state.items())
    }


def adapter_state(model: PeftModel, role: AdapterRole) -> dict[str, torch.Tensor]:
    raw = get_peft_model_state_dict(
        model,
        adapter_name=role.value,
        save_embedding_layers=False,
    )
    if not isinstance(raw, dict) or not raw:
        raise RuntimeError(f"adapter {role.value!r} produced no checkpoint state")
    state: dict[str, torch.Tensor] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise RuntimeError(f"adapter {role.value!r} produced invalid checkpoint state")
        state[name] = value
    return state


def _tensor_mapping_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TypeError("trainable state must map text names to tensors")
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return f"sha256:{digest.hexdigest()}"


def trainable_state_identity(
    *,
    model: PeftModel,
    z_head: torch.nn.Module,
    backbone_deployment_hash: str,
    psi_head: torch.nn.Module,
) -> TrainableStateIdentity:
    forward = _tensor_mapping_hash(adapter_state(model, AdapterRole.FORWARD_POLICY))
    backward = _tensor_mapping_hash(adapter_state(model, AdapterRole.BACKWARD_POLICY))
    z_state = z_head.state_dict()
    if not isinstance(z_state, dict) or not z_state:
        raise RuntimeError("Z head produced no checkpoint state")
    z = _tensor_mapping_hash(cast(dict[str, torch.Tensor], z_state))
    psi_state = psi_head.state_dict()
    if not isinstance(psi_state, dict) or not psi_state:
        raise RuntimeError("flow head produced no checkpoint state")
    psi = _tensor_mapping_hash(cast(dict[str, torch.Tensor], psi_state))
    return TrainableStateIdentity.create(
        backbone_deployment_hash=backbone_deployment_hash,
        forward_adapter_hash=forward,
        backward_adapter_hash=backward,
        z_head_hash=z,
        psi_head_hash=psi,
    )


@dataclass(frozen=True, slots=True)
class PolicyCheckpointState:
    backbone_id: str
    forward_version: str
    backward_version: str
    z_version: str
    optimizer_step: int
    trainable_state: TrainableStateIdentity
    format: str = CHECKPOINT_FORMAT

    def __post_init__(self) -> None:
        for field_name, value in (
            ("backbone_id", self.backbone_id),
            ("forward_version", self.forward_version),
            ("backward_version", self.backward_version),
            ("z_version", self.z_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty text")
        if type(self.optimizer_step) is not int or self.optimizer_step < 0:
            raise ValueError("optimizer_step must be a non-negative integer")
        if not isinstance(self.trainable_state, TrainableStateIdentity):
            raise TypeError("policy checkpoint requires a trainable state identity")
        if self.format != CHECKPOINT_FORMAT:
            raise ValueError("unsupported policy checkpoint format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "backbone_id": self.backbone_id,
            "backward_version": self.backward_version,
            "format": self.format,
            "forward_version": self.forward_version,
            "optimizer_step": self.optimizer_step,
            "trainable_state": self.trainable_state.to_value(),
            "z_version": self.z_version,
        }

    @classmethod
    def from_value(cls, value: object) -> PolicyCheckpointState:
        if not isinstance(value, dict):
            raise ValueError("policy checkpoint state must be an object")
        expected = {
            "backbone_id",
            "backward_version",
            "format",
            "forward_version",
            "optimizer_step",
            "trainable_state",
            "z_version",
        }
        if set(value) != expected:
            raise ValueError("policy checkpoint state has incompatible fields")
        if any(
            not isinstance(value[field], str)
            for field in (
                "backbone_id",
                "backward_version",
                "format",
                "forward_version",
                "z_version",
            )
        ):
            raise ValueError("policy checkpoint state text fields are invalid")
        if type(value["optimizer_step"]) is not int:
            raise ValueError("policy checkpoint optimizer_step is invalid")
        return cls(
            backbone_id=cast(str, value["backbone_id"]),
            forward_version=cast(str, value["forward_version"]),
            backward_version=cast(str, value["backward_version"]),
            z_version=cast(str, value["z_version"]),
            optimizer_step=value["optimizer_step"],
            trainable_state=TrainableStateIdentity.from_value(value["trainable_state"]),
            format=cast(str, value["format"]),
        )


@dataclass(frozen=True, slots=True)
class PolicyCheckpointInspection:
    state: PolicyCheckpointState
    forward_tensor_names: tuple[str, ...]
    backward_tensor_names: tuple[str, ...]
    z_tensor_names: tuple[str, ...]

    @property
    def content_hash(self) -> str:
        return stable_hash(
            {
                "backward_tensor_names": list(self.backward_tensor_names),
                "format": "skillev-policy-checkpoint-inspection@1",
                "forward_tensor_names": list(self.forward_tensor_names),
                "state": self.state.to_value(),
                "z_tensor_names": list(self.z_tensor_names),
            }
        )


def _load_head_state(path: Path, *, label: str) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{label} checkpoint must contain a non-empty tensor mapping")
    if any(
        type(name) is not str or not isinstance(tensor, torch.Tensor)
        for name, tensor in raw.items()
    ):
        raise ValueError(f"{label} checkpoint contains an invalid tensor mapping")
    return cast(dict[str, torch.Tensor], raw)


def read_policy_checkpoint_state(directory: Path) -> PolicyCheckpointState:
    parsed = parse_canonical_json((directory / POLICY_STATE_FILE).read_text(encoding="utf-8"))
    return PolicyCheckpointState.from_value(parsed)


def inspect_policy_checkpoint(directory: Path) -> PolicyCheckpointInspection:
    state = read_policy_checkpoint_state(directory)
    adapters: dict[AdapterRole, dict[str, torch.Tensor]] = {}
    for role in AdapterRole:
        raw = load_file(_adapter_directory(directory, role) / ADAPTER_FILE, device="cpu")
        if not raw or any(type(name) is not str for name in raw):
            raise ValueError(f"{role.value} checkpoint has no valid tensor names")
        adapters[role] = raw
    z_raw = torch.load(directory / Z_HEAD_FILE, map_location="cpu", weights_only=True)
    if not isinstance(z_raw, dict) or not z_raw:
        raise ValueError("Z checkpoint must contain a non-empty tensor mapping")
    if any(
        type(name) is not str or not isinstance(tensor, torch.Tensor)
        for name, tensor in z_raw.items()
    ):
        raise ValueError("Z checkpoint contains an invalid tensor mapping")
    psi_hash = _tensor_mapping_hash(_load_head_state(directory / PSI_HEAD_FILE, label="flow head"))
    measured = TrainableStateIdentity.create(
        backbone_deployment_hash=state.trainable_state.backbone_deployment_hash,
        forward_adapter_hash=_tensor_mapping_hash(adapters[AdapterRole.FORWARD_POLICY]),
        backward_adapter_hash=_tensor_mapping_hash(adapters[AdapterRole.BACKWARD_POLICY]),
        z_head_hash=_tensor_mapping_hash(cast(dict[str, torch.Tensor], z_raw)),
        psi_head_hash=psi_hash,
    )
    if measured != state.trainable_state:
        raise ValueError("checkpoint tensor bytes differ from policy metadata")
    return PolicyCheckpointInspection(
        state=state,
        forward_tensor_names=tuple(sorted(adapters[AdapterRole.FORWARD_POLICY])),
        backward_tensor_names=tuple(sorted(adapters[AdapterRole.BACKWARD_POLICY])),
        z_tensor_names=tuple(sorted(z_raw)),
    )


def save_policy_checkpoint(
    *,
    directory: Path,
    model: PeftModel,
    z_head: torch.nn.Module,
    state: PolicyCheckpointState,
    psi_head: torch.nn.Module,
) -> None:
    actual = trainable_state_identity(
        model=model,
        z_head=z_head,
        backbone_deployment_hash=state.trainable_state.backbone_deployment_hash,
        psi_head=psi_head,
    )
    if actual != state.trainable_state:
        raise ValueError("checkpoint metadata differs from trainable tensor bytes")
    directory.mkdir(parents=True, exist_ok=False)
    for role in AdapterRole:
        adapter_directory = _adapter_directory(directory, role)
        adapter_directory.mkdir()
        config = model.peft_config.get(role.value)
        if config is None:
            raise RuntimeError(f"adapter {role.value!r} has no PEFT configuration")
        config.save_pretrained(str(adapter_directory))
        save_file(
            _cpu_contiguous(adapter_state(model, role)),
            adapter_directory / ADAPTER_FILE,
        )
    torch.save(_cpu_contiguous(z_head.state_dict()), directory / Z_HEAD_FILE)
    torch.save(_cpu_contiguous(psi_head.state_dict()), directory / PSI_HEAD_FILE)
    (directory / POLICY_STATE_FILE).write_text(
        canonical_json(state.to_value()),
        encoding="utf-8",
    )


def load_policy_checkpoint(
    *,
    directory: Path,
    model: PeftModel,
    z_head: torch.nn.Module,
    device: torch.device,
    psi_head: torch.nn.Module,
) -> PolicyCheckpointState:
    state = read_policy_checkpoint_state(directory)
    for role in AdapterRole:
        adapter_path = _adapter_directory(directory, role) / ADAPTER_FILE
        adapter_tensors = load_file(adapter_path, device=str(device))
        result: Any = set_peft_model_state_dict(
            model,
            cast(dict[str, Any], adapter_tensors),
            adapter_name=role.value,
        )
        if result.unexpected_keys:
            raise ValueError(f"adapter checkpoint at {adapter_path} has unexpected tensors")
    z_state = torch.load(
        directory / Z_HEAD_FILE,
        map_location=device,
        weights_only=True,
    )
    z_head.load_state_dict(z_state, strict=True)
    psi_state = torch.load(directory / PSI_HEAD_FILE, map_location=device, weights_only=True)
    psi_head.load_state_dict(psi_state, strict=True)
    actual = trainable_state_identity(
        model=model,
        z_head=z_head,
        backbone_deployment_hash=state.trainable_state.backbone_deployment_hash,
        psi_head=psi_head,
    )
    if actual != state.trainable_state:
        raise ValueError("checkpoint trainable tensor bytes differ from metadata")
    return state


__all__ = [
    "ADAPTER_FILE",
    "CHECKPOINT_FORMAT",
    "POLICY_STATE_FILE",
    "PSI_HEAD_FILE",
    "Z_HEAD_FILE",
    "PolicyCheckpointInspection",
    "PolicyCheckpointState",
    "adapter_state",
    "inspect_policy_checkpoint",
    "load_policy_checkpoint",
    "read_policy_checkpoint_state",
    "save_policy_checkpoint",
    "trainable_state_identity",
]
