from __future__ import annotations

from dataclasses import dataclass, field

from skillev.calibration import CalibrationConfig
from skillev.contracts import JsonValue, normalize_json, stable_hash
from skillev.diagnostics import DiagnosticsConfig
from skillev.evolution import AuthoringSamplingConfig, EvolutionConfig
from skillev.training.config import TrainerConfig
from skillev.training.r2flow_config import R2FlowRunConfig

METHOD_SEMANTICS_FORMAT = "skillev-method-semantics@5"


@dataclass(frozen=True, slots=True)
class MethodSemanticsConfig:
    context_feature: str = "namespaced-task-family@1"
    calibration_outcome: str = "trajectory-terminal-success@1"
    flow_weight_normalization: str = "per-batch-invoking-edge-mean@1"
    skill_invocation_credit: str = "explicit-skill-action-in-h0@1"
    phase_detection_scope: str = "phase-search-slots-only@1"
    initial_library: str = "non-empty-seeded-library@1"
    state_flow_estimator: str = "single-observed-history-prefix@1"
    posterior_state_owner: str = "committed-projection-snapshot@1"
    posterior_skill_inheritance: str = "unchanged-ids-only-new-ids-start-at-prior@1"
    phase_no_op: str = "recheck-after-complete-fresh-comparison@2"
    feature_timing: str = "post-hoc-task-family-status-h0-tokens-final-horizon@1"
    confidence_authority: str = "evolution-k-and-explicit-split-confidence-k@1"
    source_requirements: str = "immutable-constraints-evidence-revisable-strategies@2"
    authoring_failure: str = "persist-raw-response-before-validation-stop-on-unknown@3"
    retain_infeasible: str = "abort-complete-attempt-before-authoring@1"
    generate_coverage: str = "no-explicit-invocation-public-action-including-complete@2"
    authoring_material: str = "bounded-public-execution-diversity@1"
    format: str = METHOD_SEMANTICS_FORMAT

    def __post_init__(self) -> None:
        expected = {
            "retain_infeasible": "abort-complete-attempt-before-authoring@1",
            "generate_coverage": "no-explicit-invocation-public-action-including-complete@2",
            "authoring_material": "bounded-public-execution-diversity@1",
            "source_requirements": "immutable-constraints-evidence-revisable-strategies@2",
            "authoring_failure": "persist-raw-response-before-validation-stop-on-unknown@3",
            "confidence_authority": "evolution-k-and-explicit-split-confidence-k@1",
            "feature_timing": "post-hoc-task-family-status-h0-tokens-final-horizon@1",
            "phase_no_op": "recheck-after-complete-fresh-comparison@2",
            "posterior_skill_inheritance": "unchanged-ids-only-new-ids-start-at-prior@1",
            "posterior_state_owner": "committed-projection-snapshot@1",
            "context_feature": "namespaced-task-family@1",
            "calibration_outcome": "trajectory-terminal-success@1",
            "flow_weight_normalization": "per-batch-invoking-edge-mean@1",
            "skill_invocation_credit": "explicit-skill-action-in-h0@1",
            "phase_detection_scope": "phase-search-slots-only@1",
            "initial_library": "non-empty-seeded-library@1",
            "state_flow_estimator": "single-observed-history-prefix@1",
        }
        if self.format != METHOD_SEMANTICS_FORMAT:
            raise ValueError("unsupported method semantics format")
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(f"unsupported method semantics: {field_name}")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "retain_infeasible": self.retain_infeasible,
            "generate_coverage": self.generate_coverage,
            "authoring_material": self.authoring_material,
            "source_requirements": self.source_requirements,
            "authoring_failure": self.authoring_failure,
            "confidence_authority": self.confidence_authority,
            "feature_timing": self.feature_timing,
            "phase_no_op": self.phase_no_op,
            "posterior_skill_inheritance": self.posterior_skill_inheritance,
            "posterior_state_owner": self.posterior_state_owner,
            "calibration_outcome": self.calibration_outcome,
            "context_feature": self.context_feature,
            "flow_weight_normalization": self.flow_weight_normalization,
            "format": self.format,
            "initial_library": self.initial_library,
            "phase_detection_scope": self.phase_detection_scope,
            "skill_invocation_credit": self.skill_invocation_credit,
            "state_flow_estimator": self.state_flow_estimator,
        }

    @classmethod
    def from_value(cls, value: object) -> MethodSemanticsConfig:
        expected_fields = {
            "retain_infeasible",
            "generate_coverage",
            "authoring_material",
            "source_requirements",
            "authoring_failure",
            "confidence_authority",
            "feature_timing",
            "phase_no_op",
            "posterior_skill_inheritance",
            "posterior_state_owner",
            "calibration_outcome",
            "context_feature",
            "flow_weight_normalization",
            "format",
            "initial_library",
            "phase_detection_scope",
            "skill_invocation_credit",
            "state_flow_estimator",
        }
        if not isinstance(value, dict) or set(value) != expected_fields:
            raise ValueError("MethodSemanticsConfig has incompatible fields")
        if any(type(item) is not str for item in value.values()):
            raise TypeError("method semantics values must be text")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ApplicationConfig:
    trainer: TrainerConfig
    diagnostics: DiagnosticsConfig
    calibration: CalibrationConfig
    evolution: EvolutionConfig
    authoring_sampling: AuthoringSamplingConfig
    maximum_h0_tokens: int
    semantics: MethodSemanticsConfig = field(default_factory=MethodSemanticsConfig)
    r2flow: R2FlowRunConfig | None = None

    def __post_init__(self) -> None:
        if type(self.maximum_h0_tokens) is not int or self.maximum_h0_tokens < 1:
            raise ValueError("maximum_h0_tokens must be positive")
        if not isinstance(self.semantics, MethodSemanticsConfig):
            raise TypeError("semantics must be MethodSemanticsConfig")
        if self.r2flow is not None:
            if not isinstance(self.r2flow, R2FlowRunConfig):
                raise TypeError("r2flow must be R2FlowRunConfig")
            if self.trainer.method != self.r2flow.method:
                raise ValueError("the trainer method differs from the declared R2 Flow method")

    @property
    def legacy_phase_detection(self) -> bool:
        return self.r2flow is None

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "authoring_sampling": self.authoring_sampling.to_value(),
            "calibration": self.calibration.to_value(),
            "diagnostics": self.diagnostics.to_value(),
            "evolution": self.evolution.to_value(),
            "maximum_h0_tokens": self.maximum_h0_tokens,
            **({"r2flow": self.r2flow.to_value()} if self.r2flow is not None else {}),
            "semantics": self.semantics.to_value(),
            "trainer": self.trainer.to_value(),
        }

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    @classmethod
    def from_value(cls, value: object) -> ApplicationConfig:
        normalized = normalize_json(value)
        fields = {
            "authoring_sampling",
            "calibration",
            "diagnostics",
            "evolution",
            "maximum_h0_tokens",
            "semantics",
            "trainer",
        }
        if not isinstance(normalized, dict) or set(normalized) - {"r2flow"} != fields:
            raise ValueError("ApplicationConfig has incompatible fields")
        maximum_h0_tokens = normalized["maximum_h0_tokens"]
        if type(maximum_h0_tokens) is not int:
            raise TypeError("maximum_h0_tokens must be an integer")
        return cls(
            trainer=TrainerConfig.from_value(normalized["trainer"]),
            diagnostics=DiagnosticsConfig.from_value(normalized["diagnostics"]),
            calibration=CalibrationConfig.from_value(normalized["calibration"]),
            evolution=EvolutionConfig.from_value(normalized["evolution"]),
            authoring_sampling=AuthoringSamplingConfig.from_value(normalized["authoring_sampling"]),
            maximum_h0_tokens=maximum_h0_tokens,
            semantics=MethodSemanticsConfig.from_value(normalized["semantics"]),
            r2flow=R2FlowRunConfig.from_value(normalized["r2flow"])
            if "r2flow" in normalized
            else None,
        )


__all__ = ["METHOD_SEMANTICS_FORMAT", "ApplicationConfig", "MethodSemanticsConfig"]
