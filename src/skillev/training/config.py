from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Final, Protocol, Self

from skillev.contracts import JsonValue, normalize_json
from skillev.contracts.action_decoding import ACTION_DECODING_RULES, ACTION_GREEDY_UNSEEDED
from skillev.contracts.action_wire import NATIVE_EVENT_CALL_WIRE
from skillev.contracts.final_turn import FINAL_TURN_COMPLETION_RULES
from skillev.contracts.reasoning_call_line import REASONING_CALL_LINE, REASONING_STOP_AT_TOOL_CALL
from skillev.contracts.skill_call_budget import SKILL_CALL_BUDGET
from skillev.contracts.skill_exposure import SKILL_EXPOSURE
from skillev.contracts.skill_visibility import SKILL_VISIBILITY_RULES
from skillev.contracts.state_map import PB_IN_EDGE_SOFTMAX, SIGMA_TRACE_QUOTIENT
from skillev.contracts.subtb import SUBTB_HUBER_GRAD_DELTA, SUBTB_HUBER_GRAD_ID
from skillev.policy.interface import ModelInputWindow
from skillev.runtime.contracts import BudgetVector
from skillev.runtime.frozen_executor import FrozenExecutorSpec
from skillev.task_semantic_guidance import TASK_SEMANTIC_GUIDANCE

from .stability import PolicyStabilityConfig

if TYPE_CHECKING:
    from skillev.rollout.context import CanonicalInitialContextAssembler

METHOD_FORMAT: Final = "skillev-method@4"
R2FLOW_OBJECTIVE_FORMAT: Final = "r2flow-objective@1"
POLICY_ROLLOUT_CONFIG_FORMAT: Final = "skillev-policy-rollout@10"
OPTIMIZER_CONFIG_FORMAT: Final = "skillev-optimizer-config@5"
TRAINING_EXECUTION_CONFIG_FORMAT: Final = "skillev-training-execution@3"
CHECKPOINT_CONFIG_FORMAT: Final = "skillev-checkpoint-config@4"
PRIVATE_CHECKPOINT_STORAGE_BINDING_FORMAT: Final = "skillev-checkpoint-storage@1"
TRAINER_CONFIG_FORMAT: Final = "skillev-trainer-config@4"

_UINT64_LIMIT = 2**64


def _bool(value: object) -> bool:
    if type(value) is not bool:
        raise TypeError("expected boolean rollout setting")
    return value


def _skill_call_budgets(value: object) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, list) or len(item) != 2 for item in value
    ):
        raise ValueError("skill_call_budget_by_domain must be [domain, budget] pairs")
    return tuple((item[0], item[1]) for item in value)


def _domain_list(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(type(item) is not str for item in value):
        raise ValueError(f"{field} must be a list of domain names")
    return tuple(value)


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return unicodedata.normalize("NFC", value)


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _uint64(value: object, *, field: str) -> int:
    if type(value) is not int or not 0 <= value < _UINT64_LIMIT:
        raise ValueError(f"{field} must be an unsigned 64-bit integer")
    return value


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite number")
    try:
        normalized = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be a finite number")
    return normalized


def _positive_float(value: object, *, field: str) -> float:
    normalized = _finite_float(value, field=field)
    if normalized <= 0.0:
        raise ValueError(f"{field} must be positive")
    return normalized


def _object(
    value: object,
    *,
    label: str,
    fields: frozenset[str],
    format_value: str,
) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict):
        raise ValueError(f"{label} must be a JSON object")
    if set(normalized) != fields:
        raise ValueError(f"{label} has an incompatible field set")
    if normalized["format"] != format_value:
        raise ValueError(f"unsupported {label} format")
    return normalized


@dataclass(frozen=True, slots=True)
class R2FlowObjectiveConfig:
    residual: str = SUBTB_HUBER_GRAD_ID
    subtb_lambda: float = 0.9
    weight_normalization: str = "per-trajectory-included-pairs@1"
    event_scoring: str = "grammar-masked-sum-free-text-conditioned@1"
    reasoning_scoring: str = "sampled-reasoning-conditioned@1"
    backward_policy: str = PB_IN_EDGE_SOFTMAX
    flow_head: str = "psi-mlp-forward-reasoning-prompt-last-hidden+domain-offset@5"
    gradient_accumulation: str = "per-edge-gradient-bank@1"
    target_law: str = "execution-trace@1"
    state_map: str = SIGMA_TRACE_QUOTIENT
    format: str = R2FLOW_OBJECTIVE_FORMAT

    _FIXED: ClassVar[tuple[tuple[str, str], ...]] = (
        ("residual", SUBTB_HUBER_GRAD_ID),
        ("weight_normalization", "per-trajectory-included-pairs@1"),
        ("event_scoring", "grammar-masked-sum-free-text-conditioned@1"),
        ("reasoning_scoring", "sampled-reasoning-conditioned@1"),
        ("backward_policy", PB_IN_EDGE_SOFTMAX),
        ("flow_head", "psi-mlp-forward-reasoning-prompt-last-hidden+domain-offset@5"),
        ("gradient_accumulation", "per-edge-gradient-bank@1"),
        ("target_law", "execution-trace@1"),
        ("state_map", SIGMA_TRACE_QUOTIENT),
        ("format", R2FLOW_OBJECTIVE_FORMAT),
    )
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "backward_policy",
            "event_scoring",
            "flow_head",
            "format",
            "gradient_accumulation",
            "reasoning_scoring",
            "residual",
            "state_map",
            "subtb_lambda",
            "target_law",
            "weight_normalization",
        }
    )

    def __post_init__(self) -> None:
        for field, expected in self._FIXED:
            if getattr(self, field) != expected:
                raise ValueError(f"unsupported R2 Flow objective {field}")
        lam = _finite_float(self.subtb_lambda, field="subtb_lambda")
        if not 0.0 < lam <= 1.0:
            raise ValueError("subtb_lambda must satisfy 0 < lambda <= 1")
        object.__setattr__(self, "subtb_lambda", lam)

    def to_value(self) -> dict[str, JsonValue]:
        return {field: getattr(self, field) for field in sorted(self._FIELDS)}

    @classmethod
    def from_value(cls, value: object) -> Self:
        data = _object(
            value,
            label="R2 Flow objective config",
            fields=cls._FIELDS,
            format_value=R2FLOW_OBJECTIVE_FORMAT,
        )
        if any(type(data[f]) is not str for f in cls._FIELDS - {"subtb_lambda"}):
            raise TypeError("R2 Flow objective identifiers must be text")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class TTBMethodConfig:
    epsilon_min: float
    temperature_beta: float
    objective: R2FlowObjectiveConfig
    format: str = METHOD_FORMAT

    def __post_init__(self) -> None:
        if self.format != METHOD_FORMAT:
            raise ValueError("unsupported TTB method format")
        if not isinstance(self.objective, R2FlowObjectiveConfig):
            raise TypeError("method objective must be R2FlowObjectiveConfig")
        object.__setattr__(
            self,
            "epsilon_min",
            _positive_float(self.epsilon_min, field="epsilon_min"),
        )
        object.__setattr__(
            self,
            "temperature_beta",
            _positive_float(self.temperature_beta, field="temperature_beta"),
        )

    @property
    def residual_gradient_clip(self) -> float:
        return SUBTB_HUBER_GRAD_DELTA

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "epsilon_min": self.epsilon_min,
            "format": self.format,
            "objective": self.objective.to_value(),
            "temperature_beta": self.temperature_beta,
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        data = _object(
            value,
            label="TTB method config",
            fields=frozenset({"epsilon_min", "format", "objective", "temperature_beta"}),
            format_value=METHOD_FORMAT,
        )
        return cls(
            epsilon_min=_positive_float(data["epsilon_min"], field="epsilon_min"),
            temperature_beta=_positive_float(data["temperature_beta"], field="temperature_beta"),
            objective=R2FlowObjectiveConfig.from_value(data["objective"]),
        )


@dataclass(frozen=True, slots=True)
class PolicyRolloutConfig:
    base_seed: int
    max_turns: int
    max_reasoning_tokens: int
    max_action_tokens: int
    per_rollout_maximum: BudgetVector
    reasoning_native_thinking: bool
    input_window: ModelInputWindow
    reasoning_by_domain: tuple[tuple[str, bool], ...]
    executor: FrozenExecutorSpec
    final_turn_completion: str
    skill_visibility: str
    action_decoding: str
    skill_call_budget_by_domain: tuple[tuple[str, int], ...] = ()
    reasoning_call_line_domains: tuple[str, ...] = ()
    reasoning_stop_domains: tuple[str, ...] = ()

    _CONSTANTS: ClassVar[dict[str, JsonValue]] = {
        "format": POLICY_ROLLOUT_CONFIG_FORMAT,
        "phase_context": True,
        "action_wire": NATIVE_EVENT_CALL_WIRE,
        "skill_exposure": SKILL_EXPOSURE,
        "public_action_semantics": True,
        "token_budget_notice": True,
        "reasoning_tool_catalog": True,
        "task_semantic_guidance": TASK_SEMANTIC_GUIDANCE,
        "hotpot_deliberation": False,
    }
    _REQUIRED: ClassVar[frozenset[str]] = frozenset(
        {
            "action_decoding",
            "base_seed",
            "executor",
            "final_turn_completion",
            "input_window",
            "max_action_tokens",
            "max_reasoning_tokens",
            "max_turns",
            "per_rollout_maximum",
            "reasoning_by_domain",
            "reasoning_native_thinking",
            "skill_visibility",
        }
        | frozenset(_CONSTANTS)
    )
    _OPTIONAL: ClassVar[frozenset[str]] = frozenset(
        {"skill_call_budget_by_domain", "reasoning_call_line_domains", "reasoning_stop_domains"}
    )

    def __post_init__(self) -> None:
        if self.action_decoding not in ACTION_DECODING_RULES:
            raise ValueError(f"action_decoding declares {ACTION_GREEDY_UNSEEDED}")
        call_line = self.reasoning_call_line_domains
        if call_line and (
            not isinstance(call_line, tuple)
            or any(type(domain) is not str or not domain.strip() for domain in call_line)
            or list(call_line) != sorted(set(call_line))
        ):
            raise ValueError(
                f"reasoning_call_line_domains declares {REASONING_CALL_LINE} "
                "(unique sorted domains)"
            )
        stops = self.reasoning_stop_domains
        if stops and (
            not isinstance(stops, tuple)
            or any(type(domain) is not str for domain in stops)
            or list(stops) != sorted(set(stops))
            or not set(stops) <= set(call_line)
        ):
            raise ValueError(
                f"reasoning_stop_domains declares {REASONING_STOP_AT_TOOL_CALL} for unique sorted "
                "domains of reasoning_call_line_domains"
            )
        if self.skill_visibility not in SKILL_VISIBILITY_RULES:
            raise ValueError("skill_visibility declares nonempty-only@1")
        budgets = self.skill_call_budget_by_domain
        if budgets and (
            not isinstance(budgets, tuple)
            or any(
                not isinstance(item, tuple)
                or len(item) != 2
                or type(item[0]) is not str
                or type(item[1]) is not int
                or item[1] < 1
                for item in budgets
            )
            or [item[0] for item in budgets] != sorted({item[0] for item in budgets})
        ):
            raise ValueError(
                "skill_call_budget_by_domain declares skill-call-budget@1 "
                "(unique sorted domains, positive budgets)"
            )
        if self.final_turn_completion not in FINAL_TURN_COMPLETION_RULES:
            raise ValueError("final_turn_completion declares final-turn-submit@1")
        if not isinstance(self.executor, FrozenExecutorSpec):
            raise TypeError("executor must be a FrozenExecutorSpec")
        if not isinstance(self.input_window, ModelInputWindow):
            raise TypeError("input_window must be a ModelInputWindow")
        if type(self.reasoning_native_thinking) is not bool:
            raise TypeError("reasoning_native_thinking must be boolean")
        if not isinstance(self.reasoning_by_domain, tuple) or any(
            not isinstance(pair, tuple)
            or len(pair) != 2
            or type(pair[0]) is not str
            or not pair[0].strip()
            or type(pair[1]) is not bool
            for pair in self.reasoning_by_domain
        ):
            raise TypeError("domain reasoning modes must be immutable named booleans")
        domains = tuple(pair[0] for pair in self.reasoning_by_domain)
        if domains != tuple(sorted(set(domains))):
            raise ValueError("domain reasoning modes must have unique sorted domains")
        object.__setattr__(self, "base_seed", _uint64(self.base_seed, field="base_seed"))
        for field in (
            "max_turns",
            "max_reasoning_tokens",
            "max_action_tokens",
        ):
            object.__setattr__(
                self,
                field,
                _positive_int(getattr(self, field), field=field),
            )
        if not isinstance(self.per_rollout_maximum, BudgetVector):
            raise TypeError("per_rollout_maximum must be a BudgetVector")
        model_calls = 2 * self.max_turns
        expected_output_tokens = self.max_turns * (
            self.max_reasoning_tokens + self.max_action_tokens
        )
        maximum = self.policy_call_maximum
        if (
            maximum.model_calls != model_calls
            or maximum.agent_turns != self.max_turns
            or maximum.tool_calls != self.max_turns
            or maximum.output_tokens != expected_output_tokens
            or maximum.input_tokens < model_calls
            or maximum.input_tokens % model_calls
            or maximum.wall_time_milliseconds < self.max_turns
            or maximum.wall_time_milliseconds % self.max_turns
        ):
            raise ValueError("per_rollout_maximum must exactly match rollout turns and call caps")
        if self.input_window.max_tokens != maximum.input_tokens // model_calls:
            raise ValueError("input window differs from the reserved per-request input allowance")

    @property
    def executor_call_maximum(self) -> BudgetVector:
        return BudgetVector(
            input_tokens=self.executor.max_input_tokens,
            output_tokens=self.executor.max_output_tokens,
            model_calls=1,
        )

    @property
    def policy_call_maximum(self) -> BudgetVector:
        total = self.executor_call_maximum.scale(self.max_turns)
        if not total.fits_within(self.per_rollout_maximum):
            raise ValueError("per_rollout_maximum does not cover the executor term")
        return self.per_rollout_maximum.subtract(total)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            **self._CONSTANTS,
            "executor": self.executor.to_value(),
            "final_turn_completion": self.final_turn_completion,
            **(
                {
                    "skill_call_budget_by_domain": normalize_json(
                        [list(item) for item in self.skill_call_budget_by_domain]
                    )
                }
                if self.skill_call_budget_by_domain
                else {}
            ),
            "skill_visibility": self.skill_visibility,
            **(
                {
                    "reasoning_call_line_domains": normalize_json(
                        list(self.reasoning_call_line_domains)
                    )
                }
                if self.reasoning_call_line_domains
                else {}
            ),
            **(
                {"reasoning_stop_domains": normalize_json(list(self.reasoning_stop_domains))}
                if self.reasoning_stop_domains
                else {}
            ),
            "action_decoding": self.action_decoding,
            "input_window": self.input_window.to_value(),
            "reasoning_by_domain": normalize_json(dict(self.reasoning_by_domain)),
            "base_seed": self.base_seed,
            "max_action_tokens": self.max_action_tokens,
            "max_reasoning_tokens": self.max_reasoning_tokens,
            "max_turns": self.max_turns,
            "per_rollout_maximum": normalize_json(self.per_rollout_maximum.to_value()),
            "reasoning_native_thinking": self.reasoning_native_thinking,
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        normalized = normalize_json(value)
        if not isinstance(normalized, dict):
            raise ValueError("policy rollout config must be a JSON object")
        fields = set(normalized)
        if not cls._REQUIRED <= fields <= cls._REQUIRED | cls._OPTIONAL:
            raise ValueError("policy rollout config has an incompatible field set")
        if any(normalized[name] != expected for name, expected in cls._CONSTANTS.items()):
            raise ValueError("unsupported policy rollout config")
        modes = normalized["reasoning_by_domain"]
        if not isinstance(modes, dict):
            raise TypeError("domain reasoning modes must be an object")
        return cls(
            executor=FrozenExecutorSpec.from_value(normalized["executor"]),
            final_turn_completion=_text(
                normalized["final_turn_completion"], field="final_turn_completion"
            ),
            skill_call_budget_by_domain=_skill_call_budgets(
                normalized["skill_call_budget_by_domain"]
            )
            if "skill_call_budget_by_domain" in normalized
            else (),
            skill_visibility=_text(normalized["skill_visibility"], field="skill_visibility"),
            reasoning_call_line_domains=_domain_list(
                normalized["reasoning_call_line_domains"], field="reasoning_call_line_domains"
            )
            if "reasoning_call_line_domains" in normalized
            else (),
            reasoning_stop_domains=_domain_list(
                normalized["reasoning_stop_domains"], field="reasoning_stop_domains"
            )
            if "reasoning_stop_domains" in normalized
            else (),
            action_decoding=_text(normalized["action_decoding"], field="action_decoding"),
            reasoning_by_domain=tuple((k, _bool(v)) for k, v in sorted(modes.items())),
            base_seed=_uint64(normalized["base_seed"], field="base_seed"),
            max_turns=_positive_int(normalized["max_turns"], field="max_turns"),
            max_reasoning_tokens=_positive_int(
                normalized["max_reasoning_tokens"],
                field="max_reasoning_tokens",
            ),
            max_action_tokens=_positive_int(
                normalized["max_action_tokens"],
                field="max_action_tokens",
            ),
            per_rollout_maximum=BudgetVector.from_value(normalized["per_rollout_maximum"]),
            reasoning_native_thinking=_bool(normalized["reasoning_native_thinking"]),
            input_window=ModelInputWindow.from_value(normalized["input_window"]),
        )

    @property
    def condition_id(self) -> str:
        return (
            (
                f"trained-skillev-interface@1/phase=1/wire={NATIVE_EVENT_CALL_WIRE}"
                f"/skills={SKILL_EXPOSURE}"
            )
            + "/public-action-semantics@1"
            + "/token-budget-notice@1"
            + "/reasoning-tool-catalog@1"
            + f"/{TASK_SEMANTIC_GUIDANCE}/hotpot=0"
            + f"/executor={self.executor.format}#{self.executor.identity()[7:19]}"
            + f"/{self.final_turn_completion}"
            + (
                f"/{SKILL_CALL_BUDGET}="
                + ",".join(f"{d}:{k}" for d, k in self.skill_call_budget_by_domain)
                if self.skill_call_budget_by_domain
                else ""
            )
            + f"/skill-visibility={self.skill_visibility}"
            + (
                f"/{REASONING_CALL_LINE}=" + ",".join(self.reasoning_call_line_domains)
                if self.reasoning_call_line_domains
                else ""
            )
            + (
                f"/{REASONING_STOP_AT_TOOL_CALL}=" + ",".join(self.reasoning_stop_domains)
                if self.reasoning_stop_domains
                else ""
            )
            + f"/{self.action_decoding}"
        )

    def context_assembler(
        self, *, maximum_h0_tokens: int, state_map: str
    ) -> CanonicalInitialContextAssembler:
        from skillev.rollout.context import CanonicalInitialContextAssembler

        return CanonicalInitialContextAssembler(
            maximum_h0_tokens=maximum_h0_tokens,
            input_window=self.input_window,
            state_map=state_map,
            final_turn_completion=self.final_turn_completion,
            skill_call_budget_by_domain=self.skill_call_budget_by_domain,
            skill_visibility=self.skill_visibility,
            reasoning_call_line_domains=self.reasoning_call_line_domains,
        )


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    adapter_learning_rate: float
    z_learning_rate: float
    psi_learning_rate: float
    weight_decay: float = 0.0
    format: str = OPTIMIZER_CONFIG_FORMAT
    backward_learning_rate: float | None = None
    stability: PolicyStabilityConfig | None = None

    def __post_init__(self) -> None:
        if self.format != OPTIMIZER_CONFIG_FORMAT:
            raise ValueError("unsupported optimizer config format")
        if self.backward_learning_rate is not None:
            _positive_float(self.backward_learning_rate, field="backward_learning_rate")
        if self.stability is not None and not isinstance(self.stability, PolicyStabilityConfig):
            raise TypeError("stability configuration must be typed")
        object.__setattr__(
            self,
            "adapter_learning_rate",
            _positive_float(
                self.adapter_learning_rate,
                field="adapter_learning_rate",
            ),
        )
        object.__setattr__(
            self,
            "z_learning_rate",
            _positive_float(self.z_learning_rate, field="z_learning_rate"),
        )
        object.__setattr__(
            self,
            "psi_learning_rate",
            _positive_float(self.psi_learning_rate, field="psi_learning_rate"),
        )
        weight_decay = _finite_float(self.weight_decay, field="weight_decay")
        if weight_decay != 0.0:
            raise ValueError("full method requires zero AdamW weight decay")
        object.__setattr__(self, "weight_decay", weight_decay)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "adapter_learning_rate": self.adapter_learning_rate,
            "backward_learning_rate": self.backward_learning_rate,
            "stability": None if self.stability is None else self.stability.to_value(),
            "format": self.format,
            "psi_learning_rate": self.psi_learning_rate,
            "weight_decay": self.weight_decay,
            "z_learning_rate": self.z_learning_rate,
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        data = _object(
            value,
            label="optimizer config",
            fields=frozenset(
                {
                    "adapter_learning_rate",
                    "backward_learning_rate",
                    "format",
                    "psi_learning_rate",
                    "stability",
                    "weight_decay",
                    "z_learning_rate",
                }
            ),
            format_value=OPTIMIZER_CONFIG_FORMAT,
        )
        return cls(
            backward_learning_rate=None
            if data["backward_learning_rate"] is None
            else _positive_float(data["backward_learning_rate"], field="backward_learning_rate"),
            stability=None
            if data["stability"] is None
            else PolicyStabilityConfig.from_value(data["stability"]),
            adapter_learning_rate=_positive_float(
                data["adapter_learning_rate"], field="adapter_learning_rate"
            ),
            z_learning_rate=_positive_float(data["z_learning_rate"], field="z_learning_rate"),
            psi_learning_rate=_positive_float(data["psi_learning_rate"], field="psi_learning_rate"),
            weight_decay=_finite_float(data["weight_decay"], field="weight_decay"),
        )


@dataclass(frozen=True, slots=True)
class TrainingExecutionConfig:
    experiment_id: str
    batch_size: int
    format: str = TRAINING_EXECUTION_CONFIG_FORMAT

    def __post_init__(self) -> None:
        if self.format != TRAINING_EXECUTION_CONFIG_FORMAT:
            raise ValueError("unsupported training execution config format")
        object.__setattr__(
            self,
            "experiment_id",
            _text(self.experiment_id, field="experiment_id"),
        )
        object.__setattr__(
            self,
            "batch_size",
            _positive_int(self.batch_size, field="batch_size"),
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "batch_size": self.batch_size,
            "experiment_id": self.experiment_id,
            "format": self.format,
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        data = _object(
            value,
            label="training execution config",
            fields=frozenset({"batch_size", "experiment_id", "format"}),
            format_value=TRAINING_EXECUTION_CONFIG_FORMAT,
        )
        return cls(
            experiment_id=_text(data["experiment_id"], field="experiment_id"),
            batch_size=_positive_int(data["batch_size"], field="batch_size"),
        )


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    every_n_steps: int
    format: str = CHECKPOINT_CONFIG_FORMAT

    def __post_init__(self) -> None:
        if self.format != CHECKPOINT_CONFIG_FORMAT:
            raise ValueError("unsupported checkpoint config format")
        object.__setattr__(
            self,
            "every_n_steps",
            _positive_int(self.every_n_steps, field="every_n_steps"),
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "every_n_steps": self.every_n_steps,
            "format": self.format,
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        data = _object(
            value,
            label="checkpoint config",
            fields=frozenset({"every_n_steps", "format"}),
            format_value=CHECKPOINT_CONFIG_FORMAT,
        )
        return cls(
            every_n_steps=_positive_int(data["every_n_steps"], field="every_n_steps"),
        )


@dataclass(frozen=True, slots=True)
class PrivateCheckpointStorageBinding:
    directory: str
    format: str = PRIVATE_CHECKPOINT_STORAGE_BINDING_FORMAT

    def __post_init__(self) -> None:
        if self.format != PRIVATE_CHECKPOINT_STORAGE_BINDING_FORMAT:
            raise ValueError("unsupported private checkpoint storage binding format")
        object.__setattr__(self, "directory", _text(self.directory, field="directory"))

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "directory": self.directory,
            "format": self.format,
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        data = _object(
            value,
            label="private checkpoint storage binding",
            fields=frozenset({"directory", "format"}),
            format_value=PRIVATE_CHECKPOINT_STORAGE_BINDING_FORMAT,
        )
        return cls(directory=_text(data["directory"], field="directory"))


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    method: TTBMethodConfig
    rollout: PolicyRolloutConfig
    optimizer: OptimizerConfig
    execution: TrainingExecutionConfig
    checkpoint: CheckpointConfig
    format: str = TRAINER_CONFIG_FORMAT

    def __post_init__(self) -> None:
        if self.format != TRAINER_CONFIG_FORMAT:
            raise ValueError("unsupported trainer config format")
        expected = (
            (self.method, TTBMethodConfig, "method"),
            (self.rollout, PolicyRolloutConfig, "rollout"),
            (self.optimizer, OptimizerConfig, "optimizer"),
            (self.execution, TrainingExecutionConfig, "execution"),
            (self.checkpoint, CheckpointConfig, "checkpoint"),
        )
        for value, expected_type, field in expected:
            if not isinstance(value, expected_type):
                raise TypeError(f"{field} must be {expected_type.__name__}")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "checkpoint": normalize_json(self.checkpoint.to_value()),
            "execution": normalize_json(self.execution.to_value()),
            "format": self.format,
            "method": normalize_json(self.method.to_value()),
            "optimizer": normalize_json(self.optimizer.to_value()),
            "rollout": normalize_json(self.rollout.to_value()),
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        data = _object(
            value,
            label="trainer config",
            fields=frozenset(
                {"checkpoint", "execution", "format", "method", "optimizer", "rollout"}
            ),
            format_value=TRAINER_CONFIG_FORMAT,
        )
        return cls(
            method=TTBMethodConfig.from_value(data["method"]),
            rollout=PolicyRolloutConfig.from_value(data["rollout"]),
            optimizer=OptimizerConfig.from_value(data["optimizer"]),
            execution=TrainingExecutionConfig.from_value(data["execution"]),
            checkpoint=CheckpointConfig.from_value(data["checkpoint"]),
        )


class _FlowHeadLike(Protocol):
    @property
    def eta(self) -> float: ...

    @property
    def epsilon(self) -> float: ...


def require_r2flow_coupling(
    *,
    method: TTBMethodConfig,
    flow_head: _FlowHeadLike | None,
    microbatch_size: int = 1,
) -> None:
    if flow_head is None:
        raise ValueError("method@4 requires a backbone F_psi head")
    if flow_head.eta != method.temperature_beta or flow_head.epsilon != method.epsilon_min:
        raise ValueError("flow-head eta/epsilon differ from the method's eta/epsilon")
    if microbatch_size != 1:
        raise ValueError("method@4 requires edge microbatch size 1")


def conservative_rollout_maximum(
    *,
    max_turns: int,
    max_reasoning_tokens: int,
    max_action_tokens: int,
    max_model_input_tokens: int,
    max_tool_wall_time_milliseconds: int,
) -> BudgetVector:
    turns = _positive_int(max_turns, field="max_turns")
    reasoning = _positive_int(max_reasoning_tokens, field="max_reasoning_tokens")
    action = _positive_int(max_action_tokens, field="max_action_tokens")
    model_input = _positive_int(max_model_input_tokens, field="max_model_input_tokens")
    tool_wall_time = _positive_int(
        max_tool_wall_time_milliseconds,
        field="max_tool_wall_time_milliseconds",
    )
    model_calls = 2 * turns
    return BudgetVector(
        input_tokens=model_calls * model_input,
        output_tokens=turns * (reasoning + action),
        model_calls=model_calls,
        agent_turns=turns,
        tool_calls=turns,
        wall_time_milliseconds=turns * tool_wall_time,
    )


__all__ = [
    "CHECKPOINT_CONFIG_FORMAT",
    "METHOD_FORMAT",
    "OPTIMIZER_CONFIG_FORMAT",
    "POLICY_ROLLOUT_CONFIG_FORMAT",
    "PRIVATE_CHECKPOINT_STORAGE_BINDING_FORMAT",
    "R2FLOW_OBJECTIVE_FORMAT",
    "TRAINER_CONFIG_FORMAT",
    "TRAINING_EXECUTION_CONFIG_FORMAT",
    "CheckpointConfig",
    "OptimizerConfig",
    "PolicyRolloutConfig",
    "PrivateCheckpointStorageBinding",
    "R2FlowObjectiveConfig",
    "TTBMethodConfig",
    "TrainerConfig",
    "TrainingExecutionConfig",
    "conservative_rollout_maximum",
    "require_r2flow_coupling",
]
