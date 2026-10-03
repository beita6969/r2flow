from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Final

from skillev.contracts import JsonValue, canonical_json, normalize_json
from skillev.contracts.identity import validate_sha256

from .attempt_protocol import AttemptBuilderKind

RUNTIME_SNAPSHOT_IDENTITY_FORMAT: Final = "skillev-runtime-snapshot-identity@7"


@dataclass(frozen=True, slots=True)
class RuntimeSnapshotIdentity:
    builder_kind: AttemptBuilderKind
    public_identity_content_hash: str
    application_config_hash: str
    method_identity_hash: str
    protocol_hash: str
    protocol_freeze_id: str
    run_plan_hash: str
    initial_library_version: str
    initial_skill_library_state_hash: str
    initial_trainable_state_hash: str
    ordered_task_sequence_hash: str
    sampling_schedule_algorithm: str
    sampling_schedule_hash: str
    formal_execution_hash: str | None = None
    base_model_artifact_hash: str | None = None
    tokenizer_artifact_hash: str | None = None
    implementation_build_hash: str | None = None
    terminal_evaluation_conditions_json: str | None = None
    task_feature_mapping_version: str | None = None
    format: str = RUNTIME_SNAPSHOT_IDENTITY_FORMAT

    def __post_init__(self) -> None:
        if not isinstance(self.builder_kind, AttemptBuilderKind):
            raise TypeError("snapshot builder_kind must be AttemptBuilderKind")
        for field in (
            "public_identity_content_hash",
            "application_config_hash",
            "method_identity_hash",
            "protocol_hash",
            "protocol_freeze_id",
            "run_plan_hash",
            "initial_trainable_state_hash",
            "ordered_task_sequence_hash",
            "sampling_schedule_hash",
        ):
            validate_sha256(getattr(self, field))
        validate_sha256(self.initial_skill_library_state_hash)
        if (
            type(self.sampling_schedule_algorithm) is not str
            or not self.sampling_schedule_algorithm.strip()
        ):
            raise ValueError("sampling_schedule_algorithm must be non-empty text")
        for field in (
            "formal_execution_hash",
            "base_model_artifact_hash",
            "tokenizer_artifact_hash",
            "implementation_build_hash",
        ):
            value = getattr(self, field)
            if value is not None:
                validate_sha256(value)
        validate_sha256(self.initial_library_version)
        if self.format != RUNTIME_SNAPSHOT_IDENTITY_FORMAT:
            raise ValueError("unsupported runtime snapshot identity format")
        if self.task_feature_mapping_version is not None:
            if not self.task_feature_mapping_version.strip():
                raise ValueError("task feature mapping requires an explicit current identity")
        if self.terminal_evaluation_conditions_json is not None:
            value = normalize_json(json.loads(self.terminal_evaluation_conditions_json))
            if not isinstance(value, dict):
                raise TypeError("terminal evaluation conditions must be an object")
            object.__setattr__(self, "terminal_evaluation_conditions_json", canonical_json(value))

    def to_value(self) -> dict[str, JsonValue]:
        value: dict[str, JsonValue] = {
            "application_config_hash": self.application_config_hash,
            "base_model_artifact_hash": self.base_model_artifact_hash,
            "builder_kind": self.builder_kind.value,
            "format": self.format,
            "formal_execution_hash": self.formal_execution_hash,
            "initial_library_version": self.initial_library_version,
            "initial_skill_library_state_hash": self.initial_skill_library_state_hash,
            "initial_trainable_state_hash": self.initial_trainable_state_hash,
            "implementation_build_hash": self.implementation_build_hash,
            "method_identity_hash": self.method_identity_hash,
            "ordered_task_sequence_hash": self.ordered_task_sequence_hash,
            "sampling_schedule_algorithm": self.sampling_schedule_algorithm,
            "sampling_schedule_hash": self.sampling_schedule_hash,
            "protocol_freeze_id": self.protocol_freeze_id,
            "protocol_hash": self.protocol_hash,
            "public_identity_content_hash": self.public_identity_content_hash,
            "run_plan_hash": self.run_plan_hash,
            "tokenizer_artifact_hash": self.tokenizer_artifact_hash,
            "task_feature_mapping_version": self.task_feature_mapping_version,
            "terminal_evaluation_conditions": (
                None
                if self.terminal_evaluation_conditions_json is None
                else normalize_json(json.loads(self.terminal_evaluation_conditions_json))
            ),
        }
        return value

    @classmethod
    def from_value(cls, value: object) -> RuntimeSnapshotIdentity:
        normalized = normalize_json(value)
        if not isinstance(normalized, dict):
            raise TypeError("RuntimeSnapshotIdentity must be an object")
        conditions = normalized.get("terminal_evaluation_conditions")
        feature_mapping = normalized.get("task_feature_mapping_version")
        if feature_mapping is not None and not isinstance(feature_mapping, str):
            raise TypeError("task feature mapping must be text or null")
        if "terminal_evaluation_conditions" not in normalized:
            raise ValueError("snapshot lacks its declared terminal scoring condition")
        normalized = {
            key: item
            for key, item in normalized.items()
            if key not in {"terminal_evaluation_conditions", "task_feature_mapping_version"}
        }
        fields = {
            "application_config_hash",
            "base_model_artifact_hash",
            "builder_kind",
            "format",
            "formal_execution_hash",
            "initial_library_version",
            "initial_skill_library_state_hash",
            "initial_trainable_state_hash",
            "implementation_build_hash",
            "method_identity_hash",
            "ordered_task_sequence_hash",
            "sampling_schedule_algorithm",
            "sampling_schedule_hash",
            "protocol_freeze_id",
            "protocol_hash",
            "public_identity_content_hash",
            "run_plan_hash",
            "tokenizer_artifact_hash",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("RuntimeSnapshotIdentity has incompatible fields")
        text_fields = fields - {
            "base_model_artifact_hash",
            "formal_execution_hash",
            "implementation_build_hash",
            "tokenizer_artifact_hash",
        }
        if any(type(normalized[field]) is not str for field in text_fields):
            raise TypeError("RuntimeSnapshotIdentity fields must be text")
        if any(
            normalized[field] is not None and type(normalized[field]) is not str
            for field in fields - text_fields
        ):
            raise TypeError("RuntimeSnapshotIdentity formal hashes must be text or null")
        return cls(
            builder_kind=AttemptBuilderKind(normalized["builder_kind"]),
            public_identity_content_hash=normalized["public_identity_content_hash"],
            application_config_hash=normalized["application_config_hash"],
            method_identity_hash=normalized["method_identity_hash"],
            protocol_hash=normalized["protocol_hash"],
            protocol_freeze_id=normalized["protocol_freeze_id"],
            run_plan_hash=normalized["run_plan_hash"],
            initial_library_version=normalized["initial_library_version"],
            initial_skill_library_state_hash=normalized["initial_skill_library_state_hash"],
            initial_trainable_state_hash=normalized["initial_trainable_state_hash"],
            ordered_task_sequence_hash=normalized["ordered_task_sequence_hash"],
            sampling_schedule_algorithm=normalized["sampling_schedule_algorithm"],
            sampling_schedule_hash=normalized["sampling_schedule_hash"],
            formal_execution_hash=normalized["formal_execution_hash"],
            base_model_artifact_hash=normalized["base_model_artifact_hash"],
            tokenizer_artifact_hash=normalized["tokenizer_artifact_hash"],
            implementation_build_hash=normalized["implementation_build_hash"],
            terminal_evaluation_conditions_json=None
            if conditions is None
            else canonical_json(conditions),
            task_feature_mapping_version=feature_mapping,
            format=normalized["format"],
        )


__all__ = ["RUNTIME_SNAPSHOT_IDENTITY_FORMAT", "RuntimeSnapshotIdentity"]
