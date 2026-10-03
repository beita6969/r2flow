from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import cast

import yaml

from skillev.contracts import JsonValue
from skillev.policy.scoring_execution import TeacherForcingConfig

from .gradient_worker_limits import validate_sequence_limits
from .request_scheduling import FAIR_MODEL_REQUESTS
from .rollout_workflow import LONG_HORIZON_FIRST, RolloutWorkflowBinding

DEFAULT_EDGE_GRADIENT_DEVICE_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_HEALTHBENCH_JUDGE_CAPACITY = 4


@dataclass(frozen=True, slots=True)
class TrainingPerformanceConfig:
    resident_trajectories: int = 16
    actor_requests: int = 8
    environment_calls: int = 8
    terminal_evaluations: int = 8
    process_graders: int = 4
    transport_threads: int = 16
    coordinator_participates: bool = True
    max_running_requests: int = 12
    max_loras_per_batch: int = 2
    deterministic_gradients: bool = True
    disable_radix_cache: bool = True
    pipeline_mode: str = "within-step"
    z_feature_cache_entries: int = 256
    gradient_buffer_bytes: int = 512 * 1024 * 1024
    gradient_worker_weights: tuple[int, ...] = ()
    gradient_worker_max_sequence_tokens: tuple[int, ...] = ()
    edge_gradient_device_bytes: int = DEFAULT_EDGE_GRADIENT_DEVICE_BYTES
    healthbench_judge_capacity: int = DEFAULT_HEALTHBENCH_JUDGE_CAPACITY
    teacher_forcing: TeacherForcingConfig = field(default_factory=TeacherForcingConfig)
    fla_profile: dict[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        for entry in fields(self):
            value = getattr(self, entry.name)
            if entry.name not in {
                "teacher_forcing",
                "fla_profile",
                "coordinator_participates",
                "pipeline_mode",
                "disable_radix_cache",
                "deterministic_gradients",
                "gradient_worker_weights",
                "gradient_worker_max_sequence_tokens",
            }:
                if type(value) is not int or value < 1:
                    raise ValueError("execution capacities must be positive integers")
        if type(self.deterministic_gradients) is not bool:
            raise TypeError("gradient determinism switch must be boolean")
        if type(self.disable_radix_cache) is not bool:
            raise TypeError("prefix cache switch must be boolean")
        if type(self.coordinator_participates) is not bool:
            raise TypeError("coordinator participation must be boolean")
        if not isinstance(self.gradient_worker_weights, tuple) or any(
            type(weight) is not int or weight < 1 for weight in self.gradient_worker_weights
        ):
            raise ValueError("gradient worker weights must be positive integer capacities")
        validate_sequence_limits(self.gradient_worker_max_sequence_tokens)
        if self.max_loras_per_batch < 2:
            raise ValueError("base and current LoRA require two serving slots")
        if self.pipeline_mode not in {"sealed-batch", "within-step"}:
            raise ValueError("unsupported gradient execution mode")
        self.workflow()

    def configure_process(self) -> None:
        import torch

        if self.deterministic_gradients:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(self.deterministic_gradients)

    def workflow(self) -> RolloutWorkflowBinding:
        return RolloutWorkflowBinding(
            max_resident_trajectories=self.resident_trajectories,
            max_inflight_model_requests=self.actor_requests,
            max_inflight_environment_calls=self.environment_calls,
            max_inflight_terminal_evaluations=self.terminal_evaluations,
            max_inflight_process_graders=self.process_graders,
            transport_worker_threads=self.transport_threads,
        )

    def to_value(self) -> dict[str, JsonValue]:
        value = cast(dict[str, JsonValue], asdict(self))
        value["rollout_scheduling_policy"] = LONG_HORIZON_FIRST
        value["request_scheduling_policy"] = FAIR_MODEL_REQUESTS
        if self.gradient_worker_weights:
            value["gradient_worker_weights"] = list(self.gradient_worker_weights)
        else:
            value.pop("gradient_worker_weights")
        if self.gradient_worker_max_sequence_tokens:
            value["gradient_worker_max_sequence_tokens"] = list(
                self.gradient_worker_max_sequence_tokens
            )
        else:
            value.pop("gradient_worker_max_sequence_tokens")
        if self.edge_gradient_device_bytes == DEFAULT_EDGE_GRADIENT_DEVICE_BYTES:
            value.pop("edge_gradient_device_bytes")
        if self.healthbench_judge_capacity == DEFAULT_HEALTHBENCH_JUDGE_CAPACITY:
            value.pop("healthbench_judge_capacity")
        return value

    @classmethod
    def from_value(cls, value: object) -> TrainingPerformanceConfig:
        names = {f.name for f in fields(cls)}
        optional = {
            "gradient_buffer_bytes",
            "gradient_worker_weights",
            "gradient_worker_max_sequence_tokens",
            "teacher_forcing",
            "fla_profile",
            "edge_gradient_device_bytes",
            "healthbench_judge_capacity",
        }
        policies = {
            "rollout_scheduling_policy": LONG_HORIZON_FIRST,
            "request_scheduling_policy": FAIR_MODEL_REQUESTS,
        }
        if (
            not isinstance(value, dict)
            or not names - optional <= set(value) - set(policies) <= names
            or any(value[name] != policy for name, policy in policies.items() if name in value)
        ):
            raise ValueError("execution profile has incompatible fields")
        values = {k: v for k, v in value.items() if k not in policies}
        if "gradient_worker_weights" in values:
            if not isinstance(values["gradient_worker_weights"], list | tuple):
                raise TypeError("gradient worker weights must be a sequence")
            values["gradient_worker_weights"] = tuple(values["gradient_worker_weights"])
        if "gradient_worker_max_sequence_tokens" in values:
            if not isinstance(values["gradient_worker_max_sequence_tokens"], list | tuple):
                raise TypeError("worker sequence limits must be a sequence")
            values["gradient_worker_max_sequence_tokens"] = tuple(
                values["gradient_worker_max_sequence_tokens"]
            )
        if "teacher_forcing" in values:
            if not isinstance(values["teacher_forcing"], dict):
                raise TypeError("teacher-forcing profile must be an object")
            values["teacher_forcing"] = TeacherForcingConfig(**values["teacher_forcing"])
        return cls(**values)

    @classmethod
    def load(cls, path: str | Path) -> TrainingPerformanceConfig:
        return cls.from_value(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
