from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import cast

from skillev.contracts import JsonValue


@dataclass(frozen=True, slots=True)
class DeadlineForecast:
    completed_steps: int
    remaining_steps: int
    steady_seconds_per_step: float | None
    projected_total_seconds: float | None
    remaining_margin_seconds: float | None
    admissible: bool
    status: str

    def to_value(self) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], asdict(self))


def forecast_deadline(
    *,
    completed_steps: int,
    elapsed_seconds: float,
    measured_step_seconds: tuple[float, ...],
    future_extra_seconds: float,
    growth_factor: float = 1.0,
    total_steps: int = 250,
    deadline_seconds: float = 72 * 3600.0,
) -> DeadlineForecast:
    if type(total_steps) is not int or total_steps < 1:
        raise ValueError("total steps must be positive")
    if type(completed_steps) is not int or not 0 <= completed_steps <= total_steps:
        raise ValueError("completed steps are outside the run")
    values = (elapsed_seconds, future_extra_seconds, growth_factor, deadline_seconds)
    if any(not math.isfinite(v) for v in values):
        raise ValueError("forecast inputs must be finite")
    if min(elapsed_seconds, future_extra_seconds) < 0 or growth_factor < 1 or deadline_seconds <= 0:
        raise ValueError("forecast time or growth inputs are invalid")
    if any(not math.isfinite(v) or v <= 0 for v in measured_step_seconds):
        raise ValueError("committed step durations must be finite and positive")
    remaining = total_steps - completed_steps
    if remaining and len(measured_step_seconds) < 3:
        return DeadlineForecast(
            completed_steps,
            remaining,
            None,
            None,
            None,
            False,
            "insufficient-steady-state-evidence",
        )
    seconds = 0.0
    if remaining:
        recent = measured_step_seconds[-10:]
        seconds = (
            max(
                math.fsum(measured_step_seconds) / len(measured_step_seconds),
                math.fsum(recent) / len(recent),
            )
            * growth_factor
        )
    projected = elapsed_seconds + remaining * seconds + future_extra_seconds
    admissible = projected <= deadline_seconds
    status = "within-budget-estimate" if admissible else "capacity-shortfall"
    if not remaining:
        status = "finalizing" if future_extra_seconds else "complete"
    return DeadlineForecast(
        completed_steps,
        remaining,
        seconds,
        projected,
        deadline_seconds - projected,
        admissible,
        status,
    )


class TrainingDeadlineExceededError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TrainingDeadline:
    started: float
    duration_seconds: float = 72 * 3600.0
    drain_margin_seconds: float = 0.0
    active_step_seconds: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.started) or not math.isfinite(self.duration_seconds):
            raise ValueError("deadline must be finite")
        if (
            not math.isfinite(self.drain_margin_seconds)
            or not 0 <= self.drain_margin_seconds < self.duration_seconds
        ):
            raise ValueError("drain margin must fit inside the hard duration")
        if self.active_step_seconds is not None and (
            not math.isfinite(self.active_step_seconds) or self.active_step_seconds <= 0
        ):
            raise ValueError("active step timeout must be finite and positive")
        if self.duration_seconds <= 0:
            raise ValueError("deadline duration must be positive")

    def require_next_step(self, now: float) -> None:
        if now - self.started >= self.duration_seconds - self.drain_margin_seconds:
            raise TrainingDeadlineExceededError("wall-time deadline reached at a durable boundary")
