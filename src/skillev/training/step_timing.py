from __future__ import annotations

from dataclasses import dataclass

from skillev.contracts import JsonValue

from .rollout_workflow import RolloutBatchPerformanceReport


@dataclass(frozen=True, slots=True)
class StepTiming:
    batch_id: str
    optimizer_step: int
    started: float
    rollout_finished: float
    gradient_started: float
    gradient_finished: float
    committed: float
    rollout: RolloutBatchPerformanceReport | None
    gradients_completed_before_last_rollout: int = 0
    rank_metrics: tuple[dict[str, JsonValue], ...] = ()

    process_instance_id: str | None = None
    process_step_index: int | None = None
    service_instance_id: str | None = None
    rollout_detail: dict[str, JsonValue] | None = None
    gradient_detail: dict[str, JsonValue] | None = None

    @property
    def is_process_warmup(self) -> bool:
        return (self.process_step_index or self.optimizer_step) <= 2

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": "r2flow-training-performance@3",
            "batch_id": self.batch_id,
            "optimizer_step": self.optimizer_step,
            "committed": True,
            "warmup": self.is_process_warmup,
            "global_step_initial_warmup": self.optimizer_step <= 2,
            "process_instance_id": self.process_instance_id,
            "process_step_index": self.process_step_index,
            "service_instance_id": self.service_instance_id,
            "service_cache_warmth": "use-request-cache-measurements-not-global-step",
            "step_wall_seconds": self.committed - self.started,
            "rollout_span_seconds": self.rollout_finished - self.started,
            "gradient_span_seconds": self.gradient_finished - self.gradient_started,
            "overlap_seconds": max(
                0.0,
                min(self.rollout_finished, self.gradient_finished)
                - max(self.started, self.gradient_started),
            ),
            "gradient_tail_seconds": max(0.0, self.gradient_finished - self.rollout_finished),
            "durability_seconds": self.committed
            - max(self.rollout_finished, self.gradient_finished),
            "first_gradient_started_seconds": self.gradient_started - self.started,
            "gradients_completed_before_last_rollout": self.gradients_completed_before_last_rollout,
            "gradient_ranks": list(self.rank_metrics),
            "rollout": None if self.rollout is None else self.rollout.to_value(),
        }
