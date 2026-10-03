from __future__ import annotations

import math
from dataclasses import dataclass, field

from skillev.contracts import JsonValue

EVOLUTION_CONFIG_FORMAT = "skillev-evolution-config@7"
GENERATE_IMPORTANCE_SEMANTICS = "absolute-log-density-ratio@1"
AUTHORING_COMPLETION_BOUNDARY = "authoring-json-root@1"


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _positive_integer(value: object, *, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _sorted_unique_text(values: tuple[str, ...], *, field: str) -> None:
    if any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError(f"{field} must contain non-empty text")
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"{field} must be sorted and unique")


@dataclass(frozen=True, slots=True)
class AuthoringSamplingConfig:
    temperature: float
    top_p: float

    def __post_init__(self) -> None:
        for name in ("temperature", "top_p"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{name} must be finite")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, number)
        if self.temperature <= 0.0 or not 0.0 < self.top_p <= 1.0:
            raise ValueError("authoring sampling controls are out of range")

    def to_value(self) -> dict[str, JsonValue]:
        return {"temperature": self.temperature, "top_p": self.top_p}

    @classmethod
    def from_value(cls, value: object) -> AuthoringSamplingConfig:
        if not isinstance(value, dict) or set(value) != {"temperature", "top_p"}:
            raise ValueError("AuthoringSamplingConfig has incompatible fields")
        temperature = value["temperature"]
        top_p = value["top_p"]
        if any(
            isinstance(item, bool) or not isinstance(item, int | float)
            for item in (temperature, top_p)
        ):
            raise TypeError("authoring sampling controls must be numeric")
        return cls(temperature=float(temperature), top_p=float(top_p))


@dataclass(frozen=True, slots=True)
class SkillAuthoringAuthority:
    input_schema_id: str
    output_schema_id: str
    license_id: str
    allowed_task_families: tuple[str, ...]
    allowed_tools: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("input_schema_id", "output_schema_id", "license_id"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty text")
        _sorted_unique_text(
            self.allowed_task_families,
            field="allowed_task_families",
        )
        _sorted_unique_text(self.allowed_tools, field="allowed_tools")
        if not self.allowed_task_families:
            raise ValueError("authoring authority requires task families")
        if "*" in self.allowed_task_families:
            raise ValueError("authoring authority task-family universe must be finite")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "allowed_task_families": list(self.allowed_task_families),
            "allowed_tools": list(self.allowed_tools),
            "input_schema_id": self.input_schema_id,
            "license_id": self.license_id,
            "output_schema_id": self.output_schema_id,
        }

    @classmethod
    def from_value(cls, value: object) -> SkillAuthoringAuthority:
        fields = {
            "allowed_task_families",
            "allowed_tools",
            "input_schema_id",
            "license_id",
            "output_schema_id",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("SkillAuthoringAuthority has incompatible fields")
        families = value["allowed_task_families"]
        tools = value["allowed_tools"]
        if not isinstance(families, list) or any(type(item) is not str for item in families):
            raise TypeError("allowed_task_families must be a text array")
        if not isinstance(tools, list) or any(type(item) is not str for item in tools):
            raise TypeError("allowed_tools must be a text array")
        for name in ("input_schema_id", "license_id", "output_schema_id"):
            if type(value[name]) is not str:
                raise TypeError(f"{name} must be text")
        return cls(
            input_schema_id=value["input_schema_id"],
            output_schema_id=value["output_schema_id"],
            license_id=value["license_id"],
            allowed_task_families=tuple(families),
            allowed_tools=tuple(tools),
        )


@dataclass(frozen=True, slots=True)
class SplitCriterionConfig:
    min_context_evidence_mass: float = 4.0
    max_within_context_mean_span: float = 0.20
    min_between_context_mean_gap: float = 0.40
    confidence_k: float = 1.0

    def __post_init__(self) -> None:
        minimum_mass = _finite(
            self.min_context_evidence_mass,
            field="min_context_evidence_mass",
        )
        maximum_span = _finite(
            self.max_within_context_mean_span,
            field="max_within_context_mean_span",
        )
        minimum_gap = _finite(
            self.min_between_context_mean_gap,
            field="min_between_context_mean_gap",
        )
        confidence_k = _finite(self.confidence_k, field="confidence_k")
        if min(minimum_mass, maximum_span, minimum_gap, confidence_k) < 0.0:
            raise ValueError("split thresholds must be non-negative")
        object.__setattr__(self, "min_context_evidence_mass", minimum_mass)
        object.__setattr__(self, "max_within_context_mean_span", maximum_span)
        object.__setattr__(self, "min_between_context_mean_gap", minimum_gap)
        object.__setattr__(self, "confidence_k", confidence_k)
        if self.max_within_context_mean_span > 1.0:
            raise ValueError("max_within_context_mean_span must not exceed one")
        if self.min_between_context_mean_gap > 1.0:
            raise ValueError("min_between_context_mean_gap must not exceed one")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "confidence_k": self.confidence_k,
            "max_within_context_mean_span": self.max_within_context_mean_span,
            "min_between_context_mean_gap": self.min_between_context_mean_gap,
            "min_context_evidence_mass": self.min_context_evidence_mass,
        }

    @classmethod
    def from_value(cls, value: object) -> SplitCriterionConfig:
        if not isinstance(value, dict) or set(value) != {
            "confidence_k",
            "max_within_context_mean_span",
            "min_between_context_mean_gap",
            "min_context_evidence_mass",
        }:
            raise ValueError("SplitCriterionConfig has incompatible fields")
        return cls(
            confidence_k=_finite(value["confidence_k"], field="confidence_k"),
            max_within_context_mean_span=_finite(
                value["max_within_context_mean_span"],
                field="max_within_context_mean_span",
            ),
            min_between_context_mean_gap=_finite(
                value["min_between_context_mean_gap"],
                field="min_between_context_mean_gap",
            ),
            min_context_evidence_mass=_finite(
                value["min_context_evidence_mass"],
                field="min_context_evidence_mass",
            ),
        )


@dataclass(frozen=True, slots=True)
class EvolutionConfig:
    generate_min_absolute_log_importance: float
    entropy_window: int = 50
    required_consecutive_drops: int = 2
    high_flow_quantile: float = 0.75
    low_flow_quantile: float = 0.25
    lcb_high: float = 0.6
    lcb_low: float = 0.3
    ucb_low: float = 0.4
    importance_quantile: float = 0.9
    generate_importance_semantics: str = GENERATE_IMPORTANCE_SEMANTICS
    max_skill_instruction_tokens_per_draft: int = 1024
    max_authoring_completion_tokens: int = 4096
    max_authoring_prompt_tokens: int = 8192
    authoring_completion_boundary_version: str = AUTHORING_COMPLETION_BOUNDARY
    k: float = 1.0
    split: SplitCriterionConfig = field(default_factory=SplitCriterionConfig)
    format: str = EVOLUTION_CONFIG_FORMAT

    def __post_init__(self) -> None:
        if self.format != EVOLUTION_CONFIG_FORMAT:
            raise ValueError(f"format must be {EVOLUTION_CONFIG_FORMAT!r}")
        minimum = _finite(
            self.generate_min_absolute_log_importance,
            field="generate_min_absolute_log_importance",
        )
        if minimum <= 0.0:
            raise ValueError("Generate absolute importance floor must be positive")
        object.__setattr__(self, "generate_min_absolute_log_importance", minimum)
        if self.generate_importance_semantics != GENERATE_IMPORTANCE_SEMANTICS:
            raise ValueError("unsupported Generate importance semantics")
        object.__setattr__(
            self,
            "entropy_window",
            _positive_integer(self.entropy_window, field="entropy_window"),
        )
        object.__setattr__(
            self,
            "required_consecutive_drops",
            _positive_integer(
                self.required_consecutive_drops,
                field="required_consecutive_drops",
            ),
        )
        object.__setattr__(
            self,
            "max_skill_instruction_tokens_per_draft",
            _positive_integer(
                self.max_skill_instruction_tokens_per_draft,
                field="max_skill_instruction_tokens_per_draft",
            ),
        )
        object.__setattr__(
            self,
            "max_authoring_completion_tokens",
            _positive_integer(
                self.max_authoring_completion_tokens,
                field="max_authoring_completion_tokens",
            ),
        )
        object.__setattr__(
            self,
            "max_authoring_prompt_tokens",
            _positive_integer(
                self.max_authoring_prompt_tokens,
                field="max_authoring_prompt_tokens",
            ),
        )
        if (
            not isinstance(self.authoring_completion_boundary_version, str)
            or self.authoring_completion_boundary_version != AUTHORING_COMPLETION_BOUNDARY
        ):
            raise ValueError("authoring completion boundary version is unsupported")
        high_flow = _finite(self.high_flow_quantile, field="high_flow_quantile")
        low_flow = _finite(self.low_flow_quantile, field="low_flow_quantile")
        lcb_high = _finite(self.lcb_high, field="lcb_high")
        lcb_low = _finite(self.lcb_low, field="lcb_low")
        ucb_low = _finite(self.ucb_low, field="ucb_low")
        importance = _finite(self.importance_quantile, field="importance_quantile")
        for name, value in (
            ("high_flow_quantile", high_flow),
            ("low_flow_quantile", low_flow),
            ("lcb_high", lcb_high),
            ("lcb_low", lcb_low),
            ("ucb_low", ucb_low),
            ("importance_quantile", importance),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, value)
        if self.low_flow_quantile > self.high_flow_quantile:
            raise ValueError("low_flow_quantile cannot exceed high_flow_quantile")
        if self.lcb_low > self.lcb_high:
            raise ValueError("lcb_low cannot exceed lcb_high")
        k = _finite(self.k, field="k")
        if k < 0.0:
            raise ValueError("k must be non-negative")
        object.__setattr__(self, "k", k)
        if not isinstance(self.split, SplitCriterionConfig):
            raise ValueError("split must be SplitCriterionConfig")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "authoring_completion_boundary_version": self.authoring_completion_boundary_version,
            "entropy_window": self.entropy_window,
            "format": self.format,
            "generate_importance_semantics": self.generate_importance_semantics,
            "generate_min_absolute_log_importance": self.generate_min_absolute_log_importance,
            "high_flow_quantile": self.high_flow_quantile,
            "importance_quantile": self.importance_quantile,
            "k": self.k,
            "lcb_high": self.lcb_high,
            "lcb_low": self.lcb_low,
            "low_flow_quantile": self.low_flow_quantile,
            "max_authoring_completion_tokens": self.max_authoring_completion_tokens,
            "max_authoring_prompt_tokens": self.max_authoring_prompt_tokens,
            "max_skill_instruction_tokens_per_draft": self.max_skill_instruction_tokens_per_draft,
            "required_consecutive_drops": self.required_consecutive_drops,
            "split": self.split.to_value(),
            "ucb_low": self.ucb_low,
        }

    @classmethod
    def from_value(cls, value: object) -> EvolutionConfig:
        if not isinstance(value, dict) or set(value) != {
            "authoring_completion_boundary_version",
            "entropy_window",
            "format",
            "generate_importance_semantics",
            "generate_min_absolute_log_importance",
            "high_flow_quantile",
            "importance_quantile",
            "k",
            "lcb_high",
            "lcb_low",
            "low_flow_quantile",
            "max_authoring_completion_tokens",
            "max_authoring_prompt_tokens",
            "max_skill_instruction_tokens_per_draft",
            "required_consecutive_drops",
            "split",
            "ucb_low",
        }:
            raise ValueError("EvolutionConfig has incompatible fields")
        if value["format"] != EVOLUTION_CONFIG_FORMAT:
            raise ValueError("EvolutionConfig has an incompatible format")
        if type(value["authoring_completion_boundary_version"]) is not str:
            raise ValueError("authoring_completion_boundary_version must be text")
        if type(value["generate_importance_semantics"]) is not str:
            raise ValueError("generate_importance_semantics must be text")
        for name in (
            "entropy_window",
            "max_authoring_completion_tokens",
            "max_authoring_prompt_tokens",
            "max_skill_instruction_tokens_per_draft",
            "required_consecutive_drops",
        ):
            if type(value[name]) is not int:
                raise ValueError(f"{name} must be an integer")
        return cls(
            generate_min_absolute_log_importance=_finite(
                value["generate_min_absolute_log_importance"],
                field="generate_min_absolute_log_importance",
            ),
            generate_importance_semantics=value["generate_importance_semantics"],
            authoring_completion_boundary_version=value["authoring_completion_boundary_version"],
            entropy_window=value["entropy_window"],
            required_consecutive_drops=value["required_consecutive_drops"],
            high_flow_quantile=_finite(value["high_flow_quantile"], field="high_flow_quantile"),
            low_flow_quantile=_finite(value["low_flow_quantile"], field="low_flow_quantile"),
            lcb_high=_finite(value["lcb_high"], field="lcb_high"),
            lcb_low=_finite(value["lcb_low"], field="lcb_low"),
            ucb_low=_finite(value["ucb_low"], field="ucb_low"),
            importance_quantile=_finite(value["importance_quantile"], field="importance_quantile"),
            max_skill_instruction_tokens_per_draft=value["max_skill_instruction_tokens_per_draft"],
            max_authoring_completion_tokens=value["max_authoring_completion_tokens"],
            max_authoring_prompt_tokens=value["max_authoring_prompt_tokens"],
            k=_finite(value["k"], field="k"),
            split=SplitCriterionConfig.from_value(value["split"]),
        )


__all__ = [
    "EVOLUTION_CONFIG_FORMAT",
    "GENERATE_IMPORTANCE_SEMANTICS",
    "AuthoringSamplingConfig",
    "EvolutionConfig",
    "SkillAuthoringAuthority",
    "SplitCriterionConfig",
]
