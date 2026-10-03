from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from skillev.runtime.attempt_run_plan import ExactAttemptRunPlan

from .run_protocol import (
    RUN_FORMAT,
    RUN_SEED,
    ActiveBenchmarkProtocol,
    RunProtocolError,
    FormalMethod,
)

RUN_FORMAL_EXPERIMENT_FORMAT: Final = "r2flow-formal-experiment@1"
RUN_BATCH_SIZE: Final = 16
RUN_PHASE_SEARCH_STEPS: Final = 272
RUN_CLOSURE_STEPS: Final = 16
RUN_MAXIMUM_CYCLES: Final = 2
RUN_TOTAL_STEPS: Final = 288
RUN_TOTAL_EPISODES: Final = 4_608


class FormalApplication(StrEnum):
    EXACT_SKILLFLOW_BASELINE = "exact-skillflow-baseline"
    BAYESIAN_IMPROVE_FULL = "bayesian-improve-full"
    BAYESIAN_IMPROVE_NO_CALIBRATION = "bayesian-improve-no-calibration"


@dataclass(frozen=True, slots=True)
class FormalMethodBinding:
    method: FormalMethod
    application: FormalApplication

    def __post_init__(self) -> None:
        expected = {
            FormalMethod.SKILLFLOW_BASELINE: FormalApplication.EXACT_SKILLFLOW_BASELINE,
            FormalMethod.BAYESIAN_IMPROVE_FULL: FormalApplication.BAYESIAN_IMPROVE_FULL,
            FormalMethod.BAYESIAN_IMPROVE_NO_CALIBRATION: (
                FormalApplication.BAYESIAN_IMPROVE_NO_CALIBRATION
            ),
        }
        if self.application is not expected[self.method]:
            raise RunProtocolError("formal method is bound to another application semantics")


@dataclass(frozen=True, slots=True)
class FormalExperimentSpec:
    protocol_format: str
    seed: int
    batch_size: int
    phase_search_steps: int
    closure_steps: int
    maximum_cycles: int
    total_steps: int
    total_episodes: int
    methods: tuple[FormalMethodBinding, ...]
    state_flow_estimator: str
    skill_outcome_estimator: str
    skill_flow_weight: str
    closure_semantics: str
    authoring_backend: str
    executable: bool
    format: str = RUN_FORMAL_EXPERIMENT_FORMAT

    def __post_init__(self) -> None:
        if self.format != RUN_FORMAL_EXPERIMENT_FORMAT:
            raise RunProtocolError("formal experiment format is unsupported")
        if self.protocol_format != RUN_FORMAT or self.seed != RUN_SEED:
            raise RunProtocolError("formal experiment differs from Protocol 10")
        if (
            self.batch_size != RUN_BATCH_SIZE
            or self.phase_search_steps != RUN_PHASE_SEARCH_STEPS
            or self.closure_steps != RUN_CLOSURE_STEPS
            or self.maximum_cycles != RUN_MAXIMUM_CYCLES
            or self.total_steps != RUN_TOTAL_STEPS
            or self.total_episodes != RUN_TOTAL_EPISODES
        ):
            raise RunProtocolError("formal experiment training shape is not frozen Protocol 10")
        if self.phase_search_steps + self.closure_steps != self.total_steps:
            raise RunProtocolError("formal experiment step partitions are inconsistent")
        if self.batch_size * self.total_steps != self.total_episodes:
            raise RunProtocolError("formal experiment does not consume all 4,608 episodes")
        if tuple(binding.method for binding in self.methods) != tuple(FormalMethod):
            raise RunProtocolError("formal method bindings differ from Protocol 10")
        if len({binding.application for binding in self.methods}) != len(self.methods):
            raise RunProtocolError("formal methods must have distinct executable semantics")
        expected_semantics = (
            "single-observed-history-prefix@1",
            "trajectory-terminal-success-shared-by-invoked-skills@1",
            "mean-normalized-state-flow@1",
            "post-treatment-closure-tail-no-new-phase@1",
            "external-sglang-base-model-authoring@1",
        )
        actual_semantics = (
            self.state_flow_estimator,
            self.skill_outcome_estimator,
            self.skill_flow_weight,
            self.closure_semantics,
            self.authoring_backend,
        )
        if actual_semantics != expected_semantics:
            raise RunProtocolError("formal method semantics differ from the frozen interpretation")

    @property
    def run_plan(self) -> ExactAttemptRunPlan:
        return ExactAttemptRunPlan(
            phase_search_steps=self.phase_search_steps,
            closure_steps=self.closure_steps,
            maximum_cycles=self.maximum_cycles,
        )

    def binding(self, method: FormalMethod) -> FormalMethodBinding:
        if not isinstance(method, FormalMethod):
            raise TypeError("formal method must be a Protocol 10 method")
        return next(binding for binding in self.methods if binding.method is method)

    def require_execution_ready(self, protocol: ActiveBenchmarkProtocol) -> None:
        if protocol.format != self.protocol_format or protocol.seed != self.seed:
            raise RunProtocolError("formal experiment and benchmark protocol disagree")
        protocol.require_execution_ready()
        if not self.executable:
            raise RunProtocolError("Protocol 10 formal experiment runtime gate is not open")


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise RunProtocolError(f"{label} must be a mapping")
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunProtocolError(f"{label} must be non-empty text")
    return value


def _integer(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise RunProtocolError(f"{label} must be an integer")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise RunProtocolError(f"{label} must be boolean")
    return value


def parse_formal_experiment(value: object) -> FormalExperimentSpec:
    root = _mapping(value, label="formal experiment")
    training = _mapping(root.get("training"), label="formal training")
    semantics = _mapping(root.get("method_semantics"), label="method semantics")
    gate = _mapping(root.get("execution_gate"), label="formal execution gate")
    method_rows = root.get("methods")
    if not isinstance(method_rows, list):
        raise RunProtocolError("formal methods must be a sequence")
    methods = tuple(
        FormalMethodBinding(
            method=FormalMethod(_text(row.get("method"), label="formal method")),
            application=FormalApplication(
                _text(row.get("application"), label="formal application")
            ),
        )
        for row in (_mapping(item, label="formal method binding") for item in method_rows)
    )
    return FormalExperimentSpec(
        format=_text(root.get("format"), label="formal experiment format"),
        protocol_format=_text(root.get("protocol_format"), label="benchmark protocol format"),
        seed=_integer(root.get("seed"), label="formal seed"),
        batch_size=_integer(training.get("batch_size"), label="batch size"),
        phase_search_steps=_integer(training.get("phase_search_steps"), label="phase-search steps"),
        closure_steps=_integer(training.get("closure_steps"), label="closure steps"),
        maximum_cycles=_integer(training.get("maximum_cycles"), label="maximum cycles"),
        total_steps=_integer(training.get("total_steps"), label="total steps"),
        total_episodes=_integer(training.get("total_episodes"), label="total episodes"),
        methods=methods,
        state_flow_estimator=_text(
            semantics.get("state_flow_estimator"), label="state-flow estimator"
        ),
        skill_outcome_estimator=_text(
            semantics.get("skill_outcome_estimator"), label="skill outcome estimator"
        ),
        skill_flow_weight=_text(semantics.get("skill_flow_weight"), label="skill flow weight"),
        closure_semantics=_text(semantics.get("closure_semantics"), label="closure semantics"),
        authoring_backend=_text(semantics.get("authoring_backend"), label="authoring backend"),
        executable=_boolean(gate.get("executable"), label="formal execution gate"),
    )


def load_formal_experiment(path: Path) -> FormalExperimentSpec:
    import yaml

    return parse_formal_experiment(yaml.safe_load(path.read_text(encoding="utf-8")))


__all__ = [
    "RUN_BATCH_SIZE",
    "RUN_CLOSURE_STEPS",
    "RUN_FORMAL_EXPERIMENT_FORMAT",
    "RUN_MAXIMUM_CYCLES",
    "RUN_PHASE_SEARCH_STEPS",
    "RUN_TOTAL_EPISODES",
    "RUN_TOTAL_STEPS",
    "FormalApplication",
    "FormalMethodBinding",
    "FormalExperimentSpec",
    "load_formal_experiment",
    "parse_formal_experiment",
]
