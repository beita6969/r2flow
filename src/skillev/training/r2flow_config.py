from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Final, Self

from skillev.contracts import JsonValue, normalize_json, stable_hash
from skillev.r2flow_evolution.amendments import RSI_AMENDMENTS
from skillev.evolution.vq_trigger import (
    VQ_ENTROPY_UNDEFINED_RULES,
    VQ_ENTROPY_VACUOUS,
    VQ_SLOPE,
)

from .config import TTBMethodConfig
from .r2flow_evolution_config import (
    DEFAULT_AUTHOR_BASE_FILE,
    DEFAULT_AUTHOR_KEY_FILE,
    DEFAULT_AUTHOR_MODEL,
    DEDICATED_VALIDATION_POOL,
    REFERENCE_VERIFIER_DOMAINS,
    REFERENCE_VERIFIER_FIELDS,
    TOST_MARGINS,
    EvolutionConfig,
    reference_verifier_domains,
    rsi_amendments,
)

VQ_TRIGGER_FORMAT: Final = "vq-plateau-trigger@1"
VQ_PLATEAU_TWO_CONSECUTIVE: Final = "vq-plateau-two-consecutive-firings@1"
VQ_PLATEAU_CONFIRMATIONS: Final = frozenset({VQ_PLATEAU_TWO_CONSECUTIVE})
R2FLOW_EVOLUTION_PROTOCOL: Final = "r2flow-evolution-protocol@2"
R2FLOW_RUN_FORMAT: Final = "r2flow-run@1"
R2FLOW_VERIFIER_SUITE: Final = "r2flow-verifier-suite@2"
R2FLOW_VERIFICATION_BUDGET: Final = "verify-all@1"
R2FLOW_FLOW_RECORDING: Final = "r2flow-flow-record@1"
R2FLOW_HEALTHBENCH_REWARD: Final = "healthbench-verdict-memo@1"
R2FLOW_HELDOUT_SPLIT: Final = "canonical-source-seed0-holdout4plus0-quality-only@1"
R2FLOW_HELDOUT_QUERIES: Final = 4
R2FLOW_BOUNDARY_ACTION: Final = "evolve@1"
VQ_ENTROPY_VERIFIER_Z: Final = "call-weighted-conditional-normalized-verifier-z@1"


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _count(value: object, *, field: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _exact(value: object, *, cls: type, label: str) -> dict[str, Any]:
    normalized = normalize_json(value)
    names = {item.name for item in fields(cls)}
    if not isinstance(normalized, dict) or set(normalized) != names:
        raise ValueError(f"{label} has an incompatible field set")
    return dict(normalized)


@dataclass(frozen=True, slots=True)
class VqTriggerConfig:
    cadence_steps: int = 5
    rollouts_per_query: int = 4
    queries_per_domain: int = 8
    window_points: int = 4
    alpha: float = 0.05
    epsilon_b_fraction: float = 0.02
    gamma_var: float = 0.05
    v_min: float = 1e-3
    h0: float = 0.02
    baseline_step: int = 0
    pooling: str = "random-effects-dl-bias-corrected-logvar@1"
    slope: str = VQ_SLOPE
    entropy: str = VQ_ENTROPY_VERIFIER_Z
    entropy_undefined: str = VQ_ENTROPY_VACUOUS
    plateau_confirmation: str | None = None
    format: str = VQ_TRIGGER_FORMAT

    _OPTIONAL: ClassVar[tuple[str, ...]] = ("plateau_confirmation",)

    def __post_init__(self) -> None:
        if self.format != VQ_TRIGGER_FORMAT:
            raise ValueError("unsupported V_q trigger format")
        if (
            self.plateau_confirmation is not None
            and self.plateau_confirmation not in VQ_PLATEAU_CONFIRMATIONS
        ):
            raise ValueError(f"plateau_confirmation declares {VQ_PLATEAU_TWO_CONSECUTIVE}")
        if self.entropy_undefined not in VQ_ENTROPY_UNDEFINED_RULES:
            raise ValueError(f"entropy_undefined declares {VQ_ENTROPY_VACUOUS}")
        if self.entropy != VQ_ENTROPY_VERIFIER_Z:
            raise ValueError("the V_q entropy is the verifier-context (C.1 z) entropy")
        _count(self.cadence_steps, field="cadence_steps")
        _count(self.rollouts_per_query, field="rollouts_per_query", minimum=2)
        _count(self.queries_per_domain, field="queries_per_domain")
        _count(self.window_points, field="window_points", minimum=2)
        _count(self.baseline_step, field="baseline_step", minimum=0)
        for name in ("alpha", "epsilon_b_fraction", "gamma_var", "v_min", "h0"):
            value = _number(getattr(self, name), field=name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if not self.alpha < 1.0:
            raise ValueError("alpha must satisfy 0 < alpha < 1")
        for name in ("pooling", "slope", "entropy"):
            _text(getattr(self, name), field=name)
        if self.slope != VQ_SLOPE:
            raise ValueError(f"unsupported V_q slope rule {self.slope!r}")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name not in self._OPTIONAL or getattr(self, item.name) is not None
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        normalized = normalize_json(value)
        if isinstance(normalized, dict):
            normalized = {**dict.fromkeys(cls._OPTIONAL), **normalized}
        return cls(**_exact(normalized, cls=cls, label="V_q trigger config"))


@dataclass(frozen=True, slots=True)
class R2FlowEvolutionProtocol:
    trigger: VqTriggerConfig = VqTriggerConfig(queries_per_domain=R2FLOW_HELDOUT_QUERIES)
    boundary_action: str = R2FLOW_BOUNDARY_ACTION
    phi_enabled: bool = True
    acceptance_gate_enabled: bool = True
    tost_validation_enabled: bool = True
    tau_c: float = 1.0
    nu: str = "uniform-legal-skill-ids-policy-arguments@1"
    pi_eval: str = "forward-snapshot-at-phase-boundary@1"
    delta: float = 0.05
    theta_low: float = 0.3
    theta_mid: float = 0.5
    theta_high: float = 0.7
    n_min: int = 8
    theta_h: float = 0.2
    kappa_u: float = 8.0
    mu_u: str = "skill-verifier-posterior-mean-empirical-bayes-per-phase@1"
    cooldown_phases: int = 2
    gamma_carry: float = 0.5
    tost_alpha: float = 0.05
    tost_margins: tuple[tuple[str, float], ...] = TOST_MARGINS
    mu_rule: str = "eb-skill-mean-beta11@1"
    lcb_glob_rule: str = "skill-pooled-beta11-quantile@1"
    cell_domain_rule: str = "observed-cells-only@1"
    split_context_rule: str = "context-class-projection@1"
    unique_coverage_rule: str = "sole-server-of-a-context-class@1"
    compress_rule: str = "pairwise-redundancy@2"
    generate_rule: str = "context-class-failure-mass@1"
    a_tilde_veto: str = "prune-requires-a-tilde-le-0@1"
    rank_rule: str = "class-order-then-psi0-desc-then-atilde@1"
    carry_rule: str = "scale-sufficient-stats@1"
    unchanged_version_rule: str = "accumulate-undiscounted@1"
    tost_mode: str = "tost-derived-noninferiority@1"
    validation_scope: str = "sequential-per-candidate@1"
    post_edit_retraining: str = "continue-all-parameters@1"
    verification_budget: str = "flow-ranked-budget@1"
    generate_min_support: int = 8
    generate_min_failure_rate: float = 0.5
    compress_min_context_overlap: float = 0.8
    compress_max_reliability_gap: float = 0.1
    max_edits_per_phase: int = 4
    max_validations_per_phase: int = 3
    validation_queries_per_domain: int = 8
    validation_rollouts_per_query: int = 2
    validation_query_selection: str = DEDICATED_VALIDATION_POOL
    author_model: str = DEFAULT_AUTHOR_MODEL
    author_base_file: str = DEFAULT_AUTHOR_BASE_FILE
    author_key_file: str = DEFAULT_AUTHOR_KEY_FILE
    max_phase_steps_no_calls: int | None = 30
    degenerate_entropy_phase_cap: str | None = None
    max_phase_steps: int | None = None
    rsi_amendments: tuple[str, ...] = RSI_AMENDMENTS
    verifier_evidence_floor: int = 2
    verification_budget_per_family: int = 64
    reference_verifier_model: str | None = None
    reference_verifier_domains: tuple[str, ...] = REFERENCE_VERIFIER_DOMAINS
    reference_verifier_timeout_s: float = 600.0
    reference_verifier_max_concurrency: int = 4
    format: str = R2FLOW_EVOLUTION_PROTOCOL

    _REFERENCE_FIELDS: ClassVar[tuple[str, ...]] = REFERENCE_VERIFIER_FIELDS
    _OPTIONAL_FIELDS: ClassVar[tuple[str, ...]] = (
        *REFERENCE_VERIFIER_FIELDS,
        "max_phase_steps",
        "degenerate_entropy_phase_cap",
    )
    _EVOLUTION_FIELDS: ClassVar[tuple[str, ...]] = (
        "generate_min_support",
        "generate_min_failure_rate",
        "compress_min_context_overlap",
        "compress_max_reliability_gap",
        "max_edits_per_phase",
        "max_validations_per_phase",
        "validation_queries_per_domain",
        "validation_rollouts_per_query",
        "validation_query_selection",
        "author_model",
        "author_base_file",
        "author_key_file",
        "max_phase_steps_no_calls",
        "degenerate_entropy_phase_cap",
        "max_phase_steps",
        "rsi_amendments",
        "verifier_evidence_floor",
        "verification_budget_per_family",
        *REFERENCE_VERIFIER_FIELDS,
    )

    def __post_init__(self) -> None:
        if self.format != R2FLOW_EVOLUTION_PROTOCOL:
            raise ValueError("unsupported R2 Flow evolution protocol")
        if isinstance(self.trigger, dict):
            object.__setattr__(self, "trigger", VqTriggerConfig.from_value(self.trigger))
        if not isinstance(self.trigger, VqTriggerConfig):
            raise TypeError("trigger must be VqTriggerConfig")
        operators = ("phi_enabled", "acceptance_gate_enabled", "tost_validation_enabled")
        for name in operators:
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be boolean")
        for name in (
            "tau_c",
            "delta",
            "theta_low",
            "theta_mid",
            "theta_high",
            "theta_h",
            "kappa_u",
            "gamma_carry",
            "tost_alpha",
        ):
            object.__setattr__(self, name, _number(getattr(self, name), field=name))
        if not 0.0 <= self.theta_low < self.theta_mid < self.theta_high <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= theta_low < theta_mid < theta_high <= 1")
        if not 0.0 < self.delta < 0.5 or not 0.0 < self.tost_alpha < 0.5:
            raise ValueError("delta and TOST alpha must lie in (0, 0.5)")
        if self.tau_c <= 0 or self.kappa_u <= 0 or not 0.0 < self.theta_h <= 1.0:
            raise ValueError("tau_c, kappa_u and theta_h must be positive")
        if not 0.0 <= self.gamma_carry <= 1.0:
            raise ValueError("gamma_carry must lie in [0, 1]")
        _count(self.n_min, field="n_min")
        _count(self.cooldown_phases, field="cooldown_phases")
        object.__setattr__(self, "rsi_amendments", rsi_amendments(self.rsi_amendments))
        margins = tuple((str(k), _number(v, field=str(k))) for k, v in self.tost_margins)
        object.__setattr__(self, "tost_margins", margins)
        for name in self._RULES:
            _text(getattr(self, name), field=name)
        if self.boundary_action != R2FLOW_BOUNDARY_ACTION or not all(
            getattr(self, name) for name in operators
        ):
            raise ValueError(
                "r2flow-evolution-protocol@2 declares evolve@1 with Phi, Acc_k and TOST"
            )
        config = self.evolution_config()
        for name in self._EVOLUTION_FIELDS:
            object.__setattr__(self, name, getattr(config, name))
        if (
            config.max_phase_steps_no_calls is not None
            and config.max_phase_steps_no_calls < self.trigger.cadence_steps
        ):
            raise ValueError("the no-call phase cap must span at least one V_q cadence")
        if (
            config.max_phase_steps is not None
            and config.max_phase_steps < self.trigger.cadence_steps
        ):
            raise ValueError("the per-phase step cap must span at least one V_q cadence")
        if self.reference_verifier_model is None and any(
            getattr(self, name) != _EVOLUTION_DEFAULTS[name] for name in self._REFERENCE_FIELDS
        ):
            raise ValueError("reference verifier settings require reference_verifier_model")

    _RULES: ClassVar[tuple[str, ...]] = (
        "nu",
        "pi_eval",
        "mu_u",
        "mu_rule",
        "lcb_glob_rule",
        "cell_domain_rule",
        "split_context_rule",
        "unique_coverage_rule",
        "compress_rule",
        "generate_rule",
        "a_tilde_veto",
        "rank_rule",
        "carry_rule",
        "unchanged_version_rule",
        "tost_mode",
        "validation_scope",
        "post_edit_retraining",
        "verification_budget",
    )

    def evolution_config(self) -> EvolutionConfig:
        names = {item.name for item in fields(EvolutionConfig)} - {"format"}
        return EvolutionConfig(**{name: getattr(self, name) for name in sorted(names)})

    def to_value(self) -> dict[str, JsonValue]:
        value: dict[str, JsonValue] = {}
        for item in fields(self):
            current = getattr(self, item.name)
            if item.name in self._REFERENCE_FIELDS and self.reference_verifier_model is None:
                continue
            if item.name in ("max_phase_steps", "degenerate_entropy_phase_cap") and current is None:
                continue
            if item.name == "trigger":
                value[item.name] = current.to_value()
            elif item.name == "tost_margins":
                value[item.name] = [[k, v] for k, v in current]
            elif item.name in ("reference_verifier_domains", "rsi_amendments"):
                value[item.name] = list(current)
            else:
                value[item.name] = current
        return value

    @classmethod
    def from_value(cls, value: object) -> Self:
        normalized = normalize_json(value)
        names = {item.name for item in fields(cls)}
        if isinstance(normalized, dict):
            names -= {name for name in cls._OPTIONAL_FIELDS if name not in normalized}
        if not isinstance(normalized, dict) or set(normalized) != names:
            raise ValueError("R2 Flow evolution protocol has an incompatible field set")
        data = dict(normalized)
        if "reference_verifier_domains" in data:
            data["reference_verifier_domains"] = reference_verifier_domains(
                data["reference_verifier_domains"]
            )
        data["rsi_amendments"] = rsi_amendments(data["rsi_amendments"])
        margins = data["tost_margins"]
        if not isinstance(margins, list) or any(
            not isinstance(item, list) or len(item) != 2 for item in margins
        ):
            raise ValueError("TOST margins must be [name, margin] pairs")
        data["tost_margins"] = tuple((item[0], item[1]) for item in margins)
        data["trigger"] = VqTriggerConfig.from_value(data["trigger"])
        return cls(**data)


_EVOLUTION_DEFAULTS: Final = {
    item.name: item.default
    for item in fields(R2FlowEvolutionProtocol)
    if item.name in R2FlowEvolutionProtocol._REFERENCE_FIELDS
}


@dataclass(frozen=True, slots=True)
class R2FlowRunConfig:
    method: TTBMethodConfig
    evolution: R2FlowEvolutionProtocol = R2FlowEvolutionProtocol()
    verifier_suite: str = R2FLOW_VERIFIER_SUITE
    verification_budget: str = R2FLOW_VERIFICATION_BUDGET
    flow_recording: str = R2FLOW_FLOW_RECORDING
    healthbench_reward: str | None = R2FLOW_HEALTHBENCH_REWARD
    heldout_split: str = R2FLOW_HELDOUT_SPLIT
    legacy_evolution_enabled: bool = False
    format: str = R2FLOW_RUN_FORMAT

    def __post_init__(self) -> None:
        if self.format != R2FLOW_RUN_FORMAT:
            raise ValueError("unsupported R2 Flow run format")
        if isinstance(self.method, dict):
            object.__setattr__(self, "method", TTBMethodConfig.from_value(self.method))
        if isinstance(self.evolution, dict):
            object.__setattr__(
                self, "evolution", R2FlowEvolutionProtocol.from_value(self.evolution)
            )
        if not isinstance(self.method, TTBMethodConfig):
            raise ValueError("an R2 Flow run requires method@4")
        if not isinstance(self.evolution, R2FlowEvolutionProtocol):
            raise TypeError("evolution must be R2FlowEvolutionProtocol")
        if self.legacy_evolution_enabled is not False:
            raise ValueError("the superseded Phi, calibration and detector stay disabled")
        expected = (
            ("verifier_suite", R2FLOW_VERIFIER_SUITE),
            ("verification_budget", R2FLOW_VERIFICATION_BUDGET),
            ("flow_recording", R2FLOW_FLOW_RECORDING),
            ("heldout_split", R2FLOW_HELDOUT_SPLIT),
        )
        for name, declared in expected:
            if getattr(self, name) != declared:
                raise ValueError(f"unsupported R2 Flow {name}")
        if self.evolution.trigger.queries_per_domain != R2FLOW_HELDOUT_QUERIES:
            raise ValueError("unsupported R2 Flow heldout_split for the V_q queries per domain")
        if self.healthbench_reward not in {R2FLOW_HEALTHBENCH_REWARD, None}:
            raise ValueError("unsupported R2 Flow healthbench_reward")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "evolution": self.evolution.to_value(),
            "flow_recording": self.flow_recording,
            "format": self.format,
            "healthbench_reward": self.healthbench_reward,
            "heldout_split": self.heldout_split,
            "legacy_evolution_enabled": self.legacy_evolution_enabled,
            "method": self.method.to_value(),
            "verification_budget": self.verification_budget,
            "verifier_suite": self.verifier_suite,
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        data = _exact(value, cls=cls, label="R2 Flow run config")
        return cls(
            method=TTBMethodConfig.from_value(data.pop("method")),
            evolution=R2FlowEvolutionProtocol.from_value(data.pop("evolution")),
            **data,
        )

    def condition_segment(self) -> str:
        def h(value: dict[str, JsonValue]) -> str:
            return stable_hash(value).removeprefix("sha256:")[:16]

        m, e, t = self.method, self.evolution, self.evolution.trigger
        return (
            f"{self.format}[method={m.format}#{h(m.to_value())};"
            f"evolution={e.format}#{h(e.to_value())}:{e.boundary_action};"
            f"trigger={t.format}#{h(t.to_value())};"
            f"verifiers={self.verifier_suite}:{self.verification_budget};"
            f"record={self.flow_recording};hb={self.healthbench_reward or 'none'};"
            f"heldout={self.heldout_split}]"
        )


__all__ = [
    "R2FLOW_BOUNDARY_ACTION",
    "R2FLOW_EVOLUTION_PROTOCOL",
    "R2FLOW_HELDOUT_QUERIES",
    "R2FLOW_HELDOUT_SPLIT",
    "R2FLOW_RUN_FORMAT",
    "R2FLOW_VERIFIER_SUITE",
    "VQ_ENTROPY_VACUOUS",
    "VQ_ENTROPY_VERIFIER_Z",
    "VQ_PLATEAU_TWO_CONSECUTIVE",
    "VQ_TRIGGER_FORMAT",
    "EvolutionConfig",
    "R2FlowEvolutionProtocol",
    "R2FlowRunConfig",
    "VqTriggerConfig",
]
