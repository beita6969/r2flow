from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Self, TypeAlias, cast

from skillev.contracts import JsonValue, normalize_json, stable_hash
from skillev.contracts.identity import validate_sha256

from .artifact_identity import BaseModelArtifactIdentity
from .flow_head import FlowHeadSpec
from .z_initialization import ZInitializationSpec

DEFAULT_LORA_TARGET_MODULES: Final = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

TEACHER_FORCED_ACTIVATION_OFFLOAD_MIN_TOKENS: Final = 32_768
QWEN35_GATED_DELTA_KERNEL_PACKAGE: Final = "flash-linear-attention"
QWEN35_GATED_DELTA_KERNEL_VERSION: Final = "0.5.2"


@dataclass(frozen=True, slots=True)
class QwenBackboneConfig:
    base_model_path: str
    revision: str
    tokenizer_id: str
    tokenizer_content_hash: str
    hidden_size: int
    device: str
    torch_dtype: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: tuple[str, ...]
    z_hidden_width: int
    eos_token_ids: tuple[int, ...]
    z_initialization: ZInitializationSpec
    flow_head: FlowHeadSpec
    tokenizer_path: str | None = None
    teacher_forced_gradient_checkpointing: bool = False
    attention_implementation: str = "sdpa"
    base_model_artifact: BaseModelArtifactIdentity | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.z_initialization, ZInitializationSpec):
            raise TypeError("Z initialization must be typed")
        if not isinstance(self.flow_head, FlowHeadSpec):
            raise TypeError("flow head must be a FlowHeadSpec")
        for field_name, value in (
            ("base_model_path", self.base_model_path),
            ("revision", self.revision),
            ("tokenizer_id", self.tokenizer_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty text")
        if self.tokenizer_path is not None and (
            not isinstance(self.tokenizer_path, str) or not self.tokenizer_path.strip()
        ):
            raise ValueError("tokenizer_path must be non-empty text or None")
        validate_sha256(self.tokenizer_content_hash)
        if type(self.hidden_size) is not int or self.hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if self.torch_dtype not in {"float32", "bfloat16"}:
            raise ValueError("torch_dtype must be 'float32' or 'bfloat16'")
        if type(self.lora_rank) is not int or self.lora_rank <= 0:
            raise ValueError("lora_rank must be a positive integer")
        if type(self.lora_alpha) is not int or self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be a positive integer")
        if (
            isinstance(self.lora_dropout, bool)
            or not isinstance(self.lora_dropout, int | float)
            or not math.isfinite(float(self.lora_dropout))
            or not 0.0 <= float(self.lora_dropout) < 1.0
        ):
            raise ValueError("lora_dropout must be finite and in [0, 1)")
        object.__setattr__(self, "lora_dropout", float(self.lora_dropout))
        if not isinstance(self.lora_target_modules, tuple) or not self.lora_target_modules:
            raise ValueError("lora_target_modules must be a non-empty tuple")
        if any(
            not isinstance(module, str) or not module.strip() for module in self.lora_target_modules
        ):
            raise ValueError("lora_target_modules must contain non-empty text")
        if len(set(self.lora_target_modules)) != len(self.lora_target_modules):
            raise ValueError("lora_target_modules must be unique")
        if type(self.z_hidden_width) is not int or self.z_hidden_width <= 0:
            raise ValueError("z_hidden_width must be a positive integer")
        if not isinstance(self.eos_token_ids, tuple) or not self.eos_token_ids:
            raise ValueError("eos_token_ids must be a non-empty tuple")
        if any(type(token_id) is not int or token_id < 0 for token_id in self.eos_token_ids):
            raise ValueError("eos_token_ids must contain non-negative integers")
        if len(set(self.eos_token_ids)) != len(self.eos_token_ids):
            raise ValueError("eos_token_ids must be unique")
        if type(self.teacher_forced_gradient_checkpointing) is not bool:
            raise TypeError("teacher_forced_gradient_checkpointing must be boolean")
        if self.teacher_forced_gradient_checkpointing and self.lora_dropout != 0.0:
            raise ValueError("gradient-checkpointed teacher forcing requires zero LoRA dropout")
        if self.attention_implementation != "sdpa":
            raise ValueError("attention_implementation must be the fixed 'sdpa' backend")
        if self.base_model_artifact is not None and not isinstance(
            self.base_model_artifact,
            BaseModelArtifactIdentity,
        ):
            raise TypeError("base_model_artifact must be BaseModelArtifactIdentity or None")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "attention_implementation": self.attention_implementation,
            "base_model_artifact": (
                self.base_model_artifact.to_value() if self.base_model_artifact else None
            ),
            "base_model_path": self.base_model_path,
            "device": self.device,
            "eos_token_ids": list(self.eos_token_ids),
            "hidden_size": self.hidden_size,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_rank": self.lora_rank,
            "lora_target_modules": list(self.lora_target_modules),
            "revision": self.revision,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_path": self.tokenizer_path,
            "tokenizer_content_hash": self.tokenizer_content_hash,
            "torch_dtype": self.torch_dtype,
            "teacher_forced_gradient_checkpointing": self.teacher_forced_gradient_checkpointing,
            "z_hidden_width": self.z_hidden_width,
            "z_initialization": self.z_initialization.to_value(),
            "flow_head": self.flow_head.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        normalized = normalize_json(value)
        fields = {
            "attention_implementation",
            "base_model_artifact",
            "base_model_path",
            "device",
            "eos_token_ids",
            "hidden_size",
            "lora_alpha",
            "lora_dropout",
            "lora_rank",
            "lora_target_modules",
            "revision",
            "tokenizer_id",
            "tokenizer_path",
            "tokenizer_content_hash",
            "torch_dtype",
            "teacher_forced_gradient_checkpointing",
            "z_hidden_width",
            "z_initialization",
            "flow_head",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("QwenBackboneConfig has incompatible fields")
        text_fields = (
            "attention_implementation",
            "base_model_path",
            "device",
            "revision",
            "tokenizer_id",
            "tokenizer_content_hash",
            "torch_dtype",
        )
        if any(type(normalized[field]) is not str for field in text_fields):
            raise TypeError("QwenBackboneConfig text fields must be strings")
        integer_fields = (
            "hidden_size",
            "lora_alpha",
            "lora_rank",
            "z_hidden_width",
        )
        if any(type(normalized[field]) is not int for field in integer_fields):
            raise TypeError("QwenBackboneConfig integer fields must be integers")
        modules = normalized["lora_target_modules"]
        eos_ids = normalized["eos_token_ids"]
        if not isinstance(modules, list) or any(type(item) is not str for item in modules):
            raise TypeError("lora_target_modules must be a text array")
        if not isinstance(eos_ids, list) or any(type(item) is not int for item in eos_ids):
            raise TypeError("eos_token_ids must be an integer array")
        dropout = normalized["lora_dropout"]
        if isinstance(dropout, bool) or not isinstance(dropout, int | float):
            raise TypeError("lora_dropout must be numeric")
        checkpointing = normalized["teacher_forced_gradient_checkpointing"]
        if type(checkpointing) is not bool:
            raise TypeError("teacher_forced_gradient_checkpointing must be boolean")
        tokenizer_path = normalized["tokenizer_path"]
        if tokenizer_path is not None and type(tokenizer_path) is not str:
            raise TypeError("tokenizer_path must be a string or null")
        return cls(
            flow_head=FlowHeadSpec.from_value(normalized["flow_head"]),
            z_initialization=ZInitializationSpec.from_value(normalized["z_initialization"]),
            base_model_path=cast(str, normalized["base_model_path"]),
            revision=cast(str, normalized["revision"]),
            tokenizer_id=cast(str, normalized["tokenizer_id"]),
            tokenizer_path=tokenizer_path,
            tokenizer_content_hash=cast(str, normalized["tokenizer_content_hash"]),
            hidden_size=cast(int, normalized["hidden_size"]),
            device=cast(str, normalized["device"]),
            torch_dtype=cast(str, normalized["torch_dtype"]),
            teacher_forced_gradient_checkpointing=checkpointing,
            attention_implementation=cast(str, normalized["attention_implementation"]),
            lora_rank=cast(int, normalized["lora_rank"]),
            lora_alpha=cast(int, normalized["lora_alpha"]),
            lora_dropout=float(dropout),
            lora_target_modules=tuple(cast(list[str], modules)),
            z_hidden_width=cast(int, normalized["z_hidden_width"]),
            eos_token_ids=tuple(cast(list[int], eos_ids)),
            base_model_artifact=(
                BaseModelArtifactIdentity.from_value(normalized["base_model_artifact"])
                if normalized["base_model_artifact"] is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class QwenMultimodalBackboneConfig(QwenBackboneConfig):
    pass


QwenDeploymentConfig: TypeAlias = QwenBackboneConfig | QwenMultimodalBackboneConfig


def qwen_backend_class(config: QwenDeploymentConfig) -> str:
    if type(config) is QwenBackboneConfig:
        return "transformers.AutoModelForCausalLM"
    if type(config) is QwenMultimodalBackboneConfig:
        return "transformers.AutoModelForMultimodalLM"
    raise TypeError("backend class requires a Qwen deployment config")


def qwen_dtype_conversion_policy(config: QwenDeploymentConfig) -> str:
    if not isinstance(config, QwenBackboneConfig | QwenMultimodalBackboneConfig):
        raise TypeError("dtype conversion policy requires a Qwen deployment config")
    return f"from_pretrained:{config.torch_dtype};module_to:{config.torch_dtype}"


def public_qwen_deployment_hash(
    config: QwenDeploymentConfig,
    *,
    backend_kind: str,
) -> str:
    if not isinstance(config, QwenBackboneConfig | QwenMultimodalBackboneConfig):
        raise TypeError("public deployment hash requires a Qwen deployment config")
    if type(backend_kind) is not str or not backend_kind:
        raise ValueError("backend_kind must be non-empty text")
    return stable_hash(
        {
            "backend_kind": backend_kind,
            "attention_implementation": config.attention_implementation,
            "base_model_artifact": (
                config.base_model_artifact.to_value()
                if config.base_model_artifact is not None
                else None
            ),
            "eos_token_ids": list(config.eos_token_ids),
            "hidden_size": config.hidden_size,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "lora_rank": config.lora_rank,
            "lora_target_modules": list(config.lora_target_modules),
            "revision": config.revision,
            "tokenizer_content_hash": config.tokenizer_content_hash,
            "tokenizer_id": config.tokenizer_id,
            "torch_dtype": config.torch_dtype,
            "teacher_forced_gradient_checkpointing": config.teacher_forced_gradient_checkpointing,
            "teacher_forced_activation_offload_min_tokens": (
                TEACHER_FORCED_ACTIVATION_OFFLOAD_MIN_TOKENS
                if config.teacher_forced_gradient_checkpointing
                else None
            ),
            "qwen35_gated_delta_kernel": (
                {
                    "package": QWEN35_GATED_DELTA_KERNEL_PACKAGE,
                    "version": QWEN35_GATED_DELTA_KERNEL_VERSION,
                }
                if config.teacher_forced_gradient_checkpointing
                else None
            ),
            "z_hidden_width": config.z_hidden_width,
            "z_initialization": config.z_initialization.to_value(),
            "flow_head": config.flow_head.to_value(),
        }
    )


__all__ = [
    "DEFAULT_LORA_TARGET_MODULES",
    "QWEN35_GATED_DELTA_KERNEL_PACKAGE",
    "QWEN35_GATED_DELTA_KERNEL_VERSION",
    "TEACHER_FORCED_ACTIVATION_OFFLOAD_MIN_TOKENS",
    "QwenBackboneConfig",
    "QwenDeploymentConfig",
    "QwenMultimodalBackboneConfig",
    "public_qwen_deployment_hash",
    "qwen_backend_class",
    "qwen_dtype_conversion_policy",
]
