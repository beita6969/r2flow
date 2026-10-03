from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import PurePosixPath
from typing import Any, Final, Self

from skillev.r2flow_evolution.amendments import LOG_COST_MARGIN_KEYS, RSI_AMENDMENTS, validate

EVOLUTION_CONFIG_FORMAT: Final = "r2flow-evolution-config@1"
DEFAULT_AUTHOR_MODEL: Final = "author-model"
DEFAULT_AUTHOR_BASE_FILE: Final = "/path/to/author_api_base"
DEFAULT_AUTHOR_KEY_FILE: Final = "/path/to/author_api_key"
TOST_MARGINS: Final = (
    ("latency_log_ratio", math.log(2.0)),
    ("success_abs", 0.10),
    ("tempered_reward_abs", 0.10),
    ("tokens_log_ratio", math.log(2.0)),
)
DEDICATED_VALIDATION_POOL: Final = "dedicated-validation-pool@1"
UNIQUE_COVERAGE_RULE: Final = "sole-server-of-a-context-class@1"
COMPRESS_RULE: Final = "pairwise-redundancy@2"
FLOW_RANKED_BUDGET_RULE: Final = "flow-ranked-budget@1"
NO_CALL_PHASE_CAP: Final = "no-call-phase-cap@1"
DEGENERATE_ENTROPY_PHASE_CAP: Final = "degenerate-entropy-phase-cap@1"
PHASE_STEP_CAP: Final = "phase-step-cap@1"
REFERENCE_VERIFIER_DOMAINS: Final = ("aime-2026", "triviaqa")
REFERENCE_VERIFIER_FIELDS: Final = (
    "reference_verifier_model",
    "reference_verifier_domains",
    "reference_verifier_timeout_s",
    "reference_verifier_max_concurrency",
)


def reference_verifier_domains(value: object) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, list | tuple | frozenset | set):
        raise ValueError("reference_verifier_domains must be a list of domains")
    domains = tuple(sorted(set(value)))
    if (
        not domains
        or len(domains) != len(value)
        or any(domain not in REFERENCE_VERIFIER_DOMAINS for domain in domains)
    ):
        raise ValueError(
            "reference_verifier_domains must be a non-repeating subset of "
            f"{REFERENCE_VERIFIER_DOMAINS}"
        )
    return domains


def validate_reference_verifier(
    model: object, domains: object, timeout_s: object, max_concurrency: object
) -> tuple[str | None, tuple[str, ...], float, int]:
    if model is not None and (type(model) is not str or not model.strip()):
        raise ValueError("reference_verifier_model must be absent or non-empty text")
    normalized_domains = reference_verifier_domains(domains)
    timeout = _number(timeout_s, name="reference_verifier_timeout_s")
    if timeout <= 0.0:
        raise ValueError("reference_verifier_timeout_s must be positive")
    concurrency = _count(max_concurrency, name="reference_verifier_max_concurrency")
    return model, normalized_domains, timeout, concurrency


def rsi_amendments(value: object) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, list | tuple):
        raise ValueError("rsi_amendments must be a list of rule ids")
    if any(type(item) is not str for item in value):
        raise ValueError("rsi_amendments must be a list of rule ids")
    return validate(value)


def validate_verifier_evidence_floor(floor: object, *, n_min: object) -> int:
    value = _count(floor, name="verifier_evidence_floor")
    if type(n_min) is int and value > n_min:
        raise ValueError("verifier_evidence_floor must not exceed n_min")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _count(value: object, *, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _unit(value: object, *, name: str) -> float:
    number = _number(value, name=name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return number


def _secret_path(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a path")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"{name} must be a normalised absolute path")
    return value


@dataclass(frozen=True, slots=True)
class EvolutionConfig:
    tau_c: float = 1.0
    delta: float = 0.05
    theta_low: float = 0.3
    theta_mid: float = 0.5
    theta_high: float = 0.7
    n_min: int = 8
    theta_h: float = 0.2
    kappa_u: float = 8.0
    cooldown_phases: int = 2
    gamma_carry: float = 0.5
    generate_min_support: int = 8
    generate_min_failure_rate: float = 0.5
    compress_min_context_overlap: float = 0.8
    compress_max_reliability_gap: float = 0.1
    tost_alpha: float = 0.05
    tost_margins: tuple[tuple[str, float], ...] = TOST_MARGINS
    max_edits_per_phase: int = 4
    max_validations_per_phase: int = 3
    validation_queries_per_domain: int = 8
    validation_rollouts_per_query: int = 2
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
    nu: str = "uniform-legal-skill-ids-policy-arguments@1"
    pi_eval: str = "forward-snapshot-at-phase-boundary@1"
    mu_u: str = "skill-verifier-posterior-mean-empirical-bayes-per-phase@1"
    mu_rule: str = "eb-skill-mean-beta11@1"
    lcb_glob_rule: str = "skill-pooled-beta11-quantile@1"
    cell_domain_rule: str = "observed-cells-only@1"
    split_context_rule: str = "context-class-projection@1"
    unique_coverage_rule: str = UNIQUE_COVERAGE_RULE
    compress_rule: str = COMPRESS_RULE
    generate_rule: str = "context-class-failure-mass@1"
    a_tilde_veto: str = "prune-requires-a-tilde-le-0@1"
    rank_rule: str = "class-order-then-psi0-desc-then-atilde@1"
    carry_rule: str = "scale-sufficient-stats@1"
    unchanged_version_rule: str = "accumulate-undiscounted@1"
    tost_mode: str = "tost-derived-noninferiority@1"
    validation_scope: str = "sequential-per-candidate@1"
    post_edit_retraining: str = "continue-all-parameters@1"
    verification_budget: str = FLOW_RANKED_BUDGET_RULE
    validation_query_selection: str = DEDICATED_VALIDATION_POOL
    format: str = EVOLUTION_CONFIG_FORMAT

    def __post_init__(self) -> None:
        if self.format != EVOLUTION_CONFIG_FORMAT:
            raise ValueError("unsupported evolution config format")
        for name in ("tau_c", "kappa_u", "theta_h"):
            value = _number(getattr(self, name), name=name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in (
            "theta_low",
            "theta_mid",
            "theta_high",
            "gamma_carry",
            "generate_min_failure_rate",
            "compress_min_context_overlap",
            "compress_max_reliability_gap",
        ):
            object.__setattr__(self, name, _unit(getattr(self, name), name=name))
        if not self.theta_low < self.theta_mid < self.theta_high:
            raise ValueError("thresholds must satisfy theta_low < theta_mid < theta_high")
        if self.theta_h > 1.0:
            raise ValueError("theta_h must lie in (0, 1]")
        for name in ("delta", "tost_alpha"):
            value = _number(getattr(self, name), name=name)
            if not 0.0 < value < 0.5:
                raise ValueError(f"{name} must lie in (0, 0.5)")
            object.__setattr__(self, name, value)
        for name in (
            "n_min",
            "cooldown_phases",
            "generate_min_support",
            "max_edits_per_phase",
            "max_validations_per_phase",
            "validation_queries_per_domain",
        ):
            _count(getattr(self, name), name=name)
        _count(self.validation_rollouts_per_query, name="validation_rollouts_per_query")
        if self.max_phase_steps_no_calls is not None:
            _count(self.max_phase_steps_no_calls, name="max_phase_steps_no_calls")
        if self.max_phase_steps is not None:
            _count(self.max_phase_steps, name="max_phase_steps")
        amendments = rsi_amendments(self.rsi_amendments)
        object.__setattr__(self, "rsi_amendments", amendments)
        margins = tuple(
            (str(key), _number(value, name=str(key))) for key, value in self.tost_margins
        )
        if tuple(key for key, _ in margins) != LOG_COST_MARGIN_KEYS or any(
            value <= 0 for _, value in margins
        ):
            raise ValueError(
                "TOST margins must declare the four positive paper margins"
                f" {list(LOG_COST_MARGIN_KEYS)}"
            )
        object.__setattr__(self, "tost_margins", margins)
        validate_verifier_evidence_floor(self.verifier_evidence_floor, n_min=self.n_min)
        _count(self.verification_budget_per_family, name="verification_budget_per_family")
        for name, fixed in (
            ("unique_coverage_rule", UNIQUE_COVERAGE_RULE),
            ("compress_rule", COMPRESS_RULE),
            ("verification_budget", FLOW_RANKED_BUDGET_RULE),
            ("validation_query_selection", DEDICATED_VALIDATION_POOL),
        ):
            if getattr(self, name) != fixed:
                raise ValueError(f"{name} must be {fixed}")
        if type(self.author_model) is not str or not self.author_model.strip():
            raise ValueError("author_model must be non-empty text")
        _secret_path(self.author_base_file, name="author_base_file")
        _secret_path(self.author_key_file, name="author_key_file")
        if self.author_base_file == self.author_key_file:
            raise ValueError("the author base URL and key must be separate files")
        for item in fields(self):
            if item.type == "str" and (
                type(getattr(self, item.name)) is not str or not getattr(self, item.name).strip()
            ):
                raise ValueError(f"{item.name} must be non-empty text")
        model, domains, timeout, concurrency = validate_reference_verifier(
            self.reference_verifier_model,
            self.reference_verifier_domains,
            self.reference_verifier_timeout_s,
            self.reference_verifier_max_concurrency,
        )
        object.__setattr__(self, "reference_verifier_domains", domains)
        object.__setattr__(self, "reference_verifier_timeout_s", timeout)
        if self.degenerate_entropy_phase_cap not in (None, DEGENERATE_ENTROPY_PHASE_CAP):
            raise ValueError(
                f"degenerate_entropy_phase_cap must be absent or {DEGENERATE_ENTROPY_PHASE_CAP}"
            )
        if self.degenerate_entropy_phase_cap is not None and self.max_phase_steps_no_calls is None:
            raise ValueError(
                f"{DEGENERATE_ENTROPY_PHASE_CAP} requires max_phase_steps_no_calls (its constant)"
            )

    @property
    def tost_margin_map(self) -> dict[str, float]:
        return dict(self.tost_margins)

    def to_value(self) -> dict[str, Any]:
        value: dict[str, Any] = {item.name: getattr(self, item.name) for item in fields(self)}
        value["tost_margins"] = [[key, margin] for key, margin in self.tost_margins]
        value["reference_verifier_domains"] = list(self.reference_verifier_domains)
        if self.max_phase_steps is None:
            del value["max_phase_steps"]
        value["rsi_amendments"] = list(self.rsi_amendments)
        if self.degenerate_entropy_phase_cap is None:
            del value["degenerate_entropy_phase_cap"]
        return value

    @classmethod
    def from_value(cls, value: object) -> Self:
        names = {item.name for item in fields(cls)}
        if not isinstance(value, dict) or not (
            names
            - {
                *REFERENCE_VERIFIER_FIELDS,
                "max_phase_steps",
                "rsi_amendments",
                "degenerate_entropy_phase_cap",
            }
            <= set(value)
            <= names
        ):
            raise ValueError("evolution config has an incompatible field set")
        data = dict(value)
        if "rsi_amendments" in data:
            data["rsi_amendments"] = rsi_amendments(data["rsi_amendments"])
        if "reference_verifier_domains" in data:
            data["reference_verifier_domains"] = reference_verifier_domains(
                data["reference_verifier_domains"]
            )
        margins = data["tost_margins"]
        if not isinstance(margins, list | tuple) or any(
            not isinstance(item, list | tuple) or len(item) != 2 for item in margins
        ):
            raise ValueError("TOST margins must be [name, margin] pairs")
        data["tost_margins"] = tuple((item[0], item[1]) for item in margins)
        return cls(**data)


__all__ = [
    "DEDICATED_VALIDATION_POOL",
    "DEFAULT_AUTHOR_BASE_FILE",
    "DEFAULT_AUTHOR_KEY_FILE",
    "DEFAULT_AUTHOR_MODEL",
    "DEGENERATE_ENTROPY_PHASE_CAP",
    "EVOLUTION_CONFIG_FORMAT",
    "NO_CALL_PHASE_CAP",
    "PHASE_STEP_CAP",
    "REFERENCE_VERIFIER_DOMAINS",
    "REFERENCE_VERIFIER_FIELDS",
    "TOST_MARGINS",
    "EvolutionConfig",
    "reference_verifier_domains",
    "rsi_amendments",
    "validate_reference_verifier",
    "validate_verifier_evidence_floor",
]
