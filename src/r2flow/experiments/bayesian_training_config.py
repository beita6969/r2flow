from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import cast

import yaml

from r2flow.benchmarks.training_schedule import (
    TRAINING_DOMAINS,
    TRAJECTORIES_PER_QUESTION,
    domain_schedule_condition,
)
from skillev.application_config import ApplicationConfig
from skillev.calibration import CalibrationConfig
from skillev.contracts import JsonValue, normalize_json
from skillev.contracts.action_decoding import ACTION_DECODING_RULES, ACTION_GREEDY_UNSEEDED
from skillev.contracts.action_wire import NATIVE_EVENT_CALL_WIRE
from skillev.contracts.final_turn import FINAL_TURN_COMPLETION_RULES
from skillev.contracts.reasoning_call_line import REASONING_CALL_LINE, REASONING_STOP_AT_TOOL_CALL
from skillev.contracts.skill_call_budget import SKILL_CALL_BUDGET
from skillev.contracts.skill_exposure import SKILL_EXPOSURE
from skillev.contracts.skill_visibility import SKILL_VISIBILITY_RULES
from skillev.diagnostics import DiagnosticsConfig
from skillev.evaluation.external_judge_policy import HEALTHBENCH_JUDGE_PROFILE
from skillev.evaluation.training_domains.catalog import TrainingBenchmark
from skillev.policy import QwenMultimodalBackboneConfig
from skillev.policy.interface import INPUT_WINDOW_VERSION, ModelInputWindow
from skillev.rollout import RolloutBudgetProfile
from skillev.runtime.attempt_run_plan import ExactAttemptRunPlan
from skillev.runtime.contracts import BudgetVector
from skillev.runtime.frozen_executor import FrozenExecutorSpec
from skillev.task_semantic_guidance import TASK_SEMANTIC_GUIDANCE, validate_task_semantic_guidance
from skillev.training import (
    OptimizerConfig,
    PolicyRolloutConfig,
    TTBMethodConfig,
    conservative_rollout_maximum,
)
from skillev.training.r2flow_config import R2FLOW_EVOLUTION_PROTOCOL, R2FlowRunConfig
from skillev.training.stability import PolicyStabilityConfig

from .bayesian_training_setup import build_application_config

R2FLOW_FORMAL = "skillev-bayesian-formal-training@7"
R2FLOW_EVOLVE_FORMAL = "skillev-bayesian-formal-training@8"
R2FLOW_FORMATS = (R2FLOW_FORMAL, R2FLOW_EVOLVE_FORMAL)
R2FLOW_INITIAL_PROFILES = {
    R2FLOW_FORMAL: "public-skill-md@1",
    R2FLOW_EVOLVE_FORMAL: "empty-skill-slots@1",
}
R2FLOW_EVOLUTION_FORMATS = {R2FLOW_EVOLVE_FORMAL: R2FLOW_EVOLUTION_PROTOCOL}
CURRENT_FORMATS = ("@6", "@7", "@8")
R2FLOW_ONLY_FIELDS = (
    "r2flow",
    "executor",
    "tool_set",
    "psi_learning_rate",
    "turns_by_domain",
    "final_turn_completion",
    "skill_call_budget_by_domain",
    "gradient_group_clip",
    "skill_visibility",
    "reasoning_call_line_domains",
    "reasoning_stop_domains",
    "healthbench_judge_recovery",
    "thinking_on_domains",
    "training_source_exclusions",
    "action_decoding",
)
TRAINING_SOURCE_EXCLUSIONS = "training-source-exclusions@1"
GRADIENT_GROUP_CLIP = "gradient-group-clip@1"
GRADIENT_CLIP_GROUPS = frozenset({"forward", "backward", "z"})
R2FLOW_FIXED_CONTROLS = (
    ("prior_alpha", 1.0),
    ("prior_beta", 1.0),
    ("k", 1.0),
    ("window", 50),
    ("rho", 0.05),
    ("maximum_cycles", 1),
    ("phi_calls_per_cycle", 64),
)


def is_current_candidate(fmt: str) -> bool:
    return fmt.endswith(CURRENT_FORMATS)


def is_r2flow_format(fmt: str) -> bool:
    return fmt in R2FLOW_FORMATS


@dataclass(frozen=True, slots=True)
class BayesianFormalConfig:
    domains: tuple[str, ...] = tuple(d.value for d in TRAINING_DOMAINS)
    format: str = "skillev-bayesian-formal-training@3"
    phase_context: bool = False
    reasoning_tool_catalog: bool = False
    token_budget_notice: bool = False
    action_wire: str = NATIVE_EVENT_CALL_WIRE
    skill_exposure: str = SKILL_EXPOSURE
    initial_skill_profile: str = "empty-skill-slots@1"
    learning_protocol: str | None = None
    healthbench_judge: str | None = None
    public_action_semantics: bool = False
    task_semantic_guidance: str = TASK_SEMANTIC_GUIDANCE
    model: str = "Qwen3.5-9B"
    base_dtype: str = "bfloat16"
    lora_rank: int = 4
    lora_alpha: int = 8
    seed: int = 0
    steps: int = 250
    closure_steps: int = 1
    maximum_cycles: int = 2
    checkpoint_every: int = 10
    max_turns: int = 20
    static_max_turns: int = 8
    max_reasoning_tokens: int = 1024
    reasoning_tokens_by_domain: tuple[tuple[str, int], ...] = ()
    max_action_tokens: int = 2048
    action_tokens_by_domain: tuple[tuple[str, int], ...] = ()
    max_input_tokens: int = 65_536
    input_window: str = INPUT_WINDOW_VERSION
    adapter_learning_rate: float = 0.0001
    z_learning_rate: float = 0.0001
    weight_decay: float = 0.0
    gradient_clipping: bool = False
    extra_kl: float = 0.0
    ttb_beta: float = 1.0
    epsilon: float = 0.1
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    window: int = 50
    rho: float = 0.05
    k: float = 1.0
    reasoning_native_thinking: bool = True
    thinking_off_domains: tuple[str, ...] = ()
    hotpot_deliberation: bool = True
    phi_calls_per_cycle: int = 64
    performance_profile: str = "configs/r2flow/execution_2gpu.yaml"
    planning_hours: float = 72.0
    target_steps_per_hour: float = 4.2
    r2flow: R2FlowRunConfig | None = None
    executor: FrozenExecutorSpec | None = None
    tool_set: str | None = None
    psi_learning_rate: float | None = None
    turns_by_domain: tuple[tuple[str, int], ...] = ()
    final_turn_completion: str | None = None
    skill_call_budget_by_domain: tuple[tuple[str, int], ...] = ()
    gradient_group_clip: tuple[tuple[str, float], ...] = ()
    skill_visibility: str | None = None
    reasoning_call_line_domains: tuple[str, ...] = ()
    reasoning_stop_domains: tuple[str, ...] = ()
    healthbench_judge_recovery: str | None = None
    thinking_on_domains: tuple[str, ...] = ()
    training_source_exclusions: tuple[tuple[str, str], ...] = ()
    action_decoding: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.r2flow, dict):
            object.__setattr__(self, "r2flow", R2FlowRunConfig.from_value(self.r2flow))
        if isinstance(self.executor, dict):
            object.__setattr__(self, "executor", FrozenExecutorSpec.from_value(self.executor))
        if self.format not in R2FLOW_FORMATS and any(
            getattr(self, name) is not None and getattr(self, name) != ()
            for name in R2FLOW_ONLY_FIELDS
        ):
            raise ValueError("R2 Flow controls require formal condition @7")
        available = tuple(d.value for d in TRAINING_DOMAINS)
        domains = tuple(self.domains)
        if not domains or tuple(d for d in available if d in domains) != domains:
            raise ValueError("active domains must be a nonempty canonical subset")
        object.__setattr__(self, "domains", domains)
        if self.initial_skill_profile != "empty-skill-slots@1":
            raise ValueError("unknown initial skill profile")
        if self.skill_exposure != SKILL_EXPOSURE or self.format != R2FLOW_EVOLVE_FORMAL:
            raise ValueError(
                "empty-skill-slots@1 requires formal @8 with the skill-md-invoke@2 exposure"
            )
        if self.learning_protocol is not None:
            from .autonomous_ttb import require_autonomous_config

            require_autonomous_config(self)
        if self.healthbench_judge not in {None, HEALTHBENCH_JUDGE_PROFILE}:
            raise ValueError("unsupported HealthBench scoring condition")
        if type(self.token_budget_notice) is not bool or (
            self.token_budget_notice and not self.phase_context
        ):
            raise ValueError("token budget notice requires phase context")
        if type(self.reasoning_tool_catalog) is not bool or (
            self.reasoning_tool_catalog and not self.phase_context
        ):
            raise ValueError("reasoning tool catalog requires native phase context")
        if self.format not in {
            "skillev-bayesian-formal-training@2",
            "skillev-bayesian-formal-training@3",
            "skillev-bayesian-formal-training@4",
            "skillev-bayesian-formal-training@5",
            "skillev-bayesian-formal-training@6",
            R2FLOW_FORMAL,
            R2FLOW_EVOLVE_FORMAL,
        }:
            raise ValueError("unsupported formal training condition")
        if not self.format.endswith(("@4", "@5", "@6", "@7", "@8")) and self.phase_context:
            raise ValueError("interface candidates require formal condition version 4")
        validate_task_semantic_guidance(self.task_semantic_guidance)
        if is_current_candidate(self.format) and not self.public_action_semantics:
            raise ValueError("shared native task guidance requires public action semantics")
        if self.public_action_semantics and not self.format.endswith(("@5", "@6", "@7", "@8")):
            raise ValueError("public semantics require formal candidate version 5")
        if not isinstance(self.thinking_off_domains, tuple | list) or any(
            type(v) is not str for v in self.thinking_off_domains
        ):
            raise TypeError("thinking-off domains must be a sequence of domain names")
        domains = tuple(self.thinking_off_domains)
        if len(set(domains)) != len(domains) or not set(domains) <= {
            d.value for d in TRAINING_DOMAINS
        }:
            raise ValueError(
                "thinking-off domains must be unique members of the six-domain schedule"
            )
        object.__setattr__(self, "thinking_off_domains", tuple(sorted(domains)))
        if not isinstance(self.thinking_on_domains, tuple | list) or any(
            type(v) is not str for v in self.thinking_on_domains
        ):
            raise TypeError("thinking-on domains must be a sequence of domain names")
        thinking_on = tuple(self.thinking_on_domains)
        if (
            len(set(thinking_on)) != len(thinking_on)
            or not set(thinking_on) <= set(self.domains)
            or set(thinking_on) & set(domains)
        ):
            raise ValueError(
                "thinking-on domains are unique active domains outside thinking_off_domains"
            )
        object.__setattr__(self, "thinking_on_domains", tuple(sorted(thinking_on)))
        for field in ("reasoning_tokens_by_domain", "action_tokens_by_domain", "turns_by_domain"):
            limits = tuple(tuple(item) for item in getattr(self, field))
            if any(
                len(item) != 2
                or item[0] not in {d.value for d in TRAINING_DOMAINS}
                or type(item[1]) is not int
                or item[1] < 1
                for item in limits
            ) or len({item[0] for item in limits}) != len(limits):
                raise ValueError("domain budgets need unique domains and positive token caps")
            if limits and not is_current_candidate(self.format):
                raise ValueError("domain budgets require an explicit current candidate")
            object.__setattr__(self, field, tuple(sorted(limits)))
        if self.format.endswith("@2") and domains:
            raise ValueError("legacy formal condition cannot declare mixed thinking")
        self._validate_owner_model()
        if self.seed != 0 or type(self.seed) is not int:
            raise ValueError("this formal schedule uses the single declared seed zero")
        for name in (
            "lora_rank",
            "lora_alpha",
            "steps",
            "closure_steps",
            "maximum_cycles",
            "checkpoint_every",
            "max_turns",
            "static_max_turns",
            "max_reasoning_tokens",
            "max_action_tokens",
            "max_input_tokens",
            "window",
            "phi_calls_per_cycle",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.steps > 250 or not 1 <= self.closure_steps < self.steps:
            raise ValueError("formal schedule needs search steps and a positive closure tail")
        if self.static_max_turns > self.max_turns:
            raise ValueError("static domain horizon exceeds the global cap")
        if self.weight_decay != 0 or self.gradient_clipping is not False or self.extra_kl != 0:
            raise ValueError("declared TTB has no decay, clipping or added KL objective")
        if self.reasoning_native_thinking is not True or type(self.hotpot_deliberation) is not bool:
            raise ValueError("formal owner reasoning must actually enable native thinking")
        if self.planning_hours <= 0 or self.target_steps_per_hour <= 0:
            raise ValueError("planning estimates must be positive, not hard deadlines")
        budgets = tuple(tuple(item) for item in self.skill_call_budget_by_domain)
        if any(
            len(item) != 2 or item[0] not in self.domains or type(item[1]) is not int or item[1] < 1
            for item in budgets
        ) or len({item[0] for item in budgets}) != len(budgets):
            raise ValueError(
                "skill-call-budget@1 needs unique active domains with positive integer budgets"
            )
        object.__setattr__(self, "skill_call_budget_by_domain", tuple(sorted(budgets)))
        clips = tuple(tuple(item) for item in self.gradient_group_clip)
        if any(
            len(item) != 2
            or item[0] not in GRADIENT_CLIP_GROUPS
            or isinstance(item[1], bool)
            or not isinstance(item[1], int | float)
            or not math.isfinite(item[1])
            or item[1] <= 0
            for item in clips
        ) or len({item[0] for item in clips}) != len(clips):
            raise ValueError(
                f"{GRADIENT_GROUP_CLIP} needs unique groups of {sorted(GRADIENT_CLIP_GROUPS)} "
                "with positive finite norms"
            )
        if clips and self.format not in R2FLOW_FORMATS:
            raise ValueError(f"{GRADIENT_GROUP_CLIP} is declared for R2 Flow formats only")
        if self.skill_visibility is not None and (
            self.skill_visibility not in SKILL_VISIBILITY_RULES
            or self.skill_exposure != SKILL_EXPOSURE
        ):
            raise ValueError("skill_visibility declares nonempty-only@1 (skill-md-invoke@2 only)")
        call_line = tuple(self.reasoning_call_line_domains)
        if any(domain not in self.domains for domain in call_line) or len(set(call_line)) != len(
            call_line
        ):
            raise ValueError(f"{REASONING_CALL_LINE} needs unique active domains")
        object.__setattr__(self, "reasoning_call_line_domains", tuple(sorted(call_line)))
        stops = tuple(self.reasoning_stop_domains)
        if any(domain not in call_line for domain in stops) or len(set(stops)) != len(stops):
            raise ValueError(
                f"{REASONING_STOP_AT_TOOL_CALL} needs unique domains of reasoning_call_line_domains"
            )
        object.__setattr__(self, "reasoning_stop_domains", tuple(sorted(stops)))
        object.__setattr__(
            self,
            "gradient_group_clip",
            tuple(sorted((str(group), float(cast(float, norm))) for group, norm in clips)),
        )
        exclusions = tuple(tuple(item) for item in self.training_source_exclusions)
        if any(
            len(item) != 2
            or item[0] not in self.domains
            or type(item[1]) is not str
            or not item[1].strip()
            for item in exclusions
        ) or len(set(exclusions)) != len(exclusions):
            raise ValueError(
                f"{TRAINING_SOURCE_EXCLUSIONS} needs unique (active domain, source id) pairs"
            )
        object.__setattr__(
            self,
            "training_source_exclusions",
            tuple(sorted((str(domain), str(source)) for domain, source in exclusions)),
        )
        _ = self.sampling_config, self.run_plan
        if self.format in R2FLOW_FORMATS:
            self._validate_r2flow()
        _ = self.optimizer_config
        CalibrationConfig(self.prior_alpha, self.prior_beta, self.k)
        DiagnosticsConfig(window_size=self.window, stagnation_rho=self.rho)

    def _validate_r2flow(self) -> None:
        from skillev.benchmarks.r2flow_tool_set import R2FLOW_TOOL_SET

        r2flow = self.r2flow
        if not isinstance(r2flow, R2FlowRunConfig):
            raise ValueError("formal condition @7 requires the r2flow block")
        if r2flow.evolution.format != R2FLOW_EVOLUTION_FORMATS[self.format]:
            raise ValueError(f"{self.format} declares {R2FLOW_EVOLUTION_FORMATS[self.format]}")
        if "healthbench" in self.domains:
            if self.healthbench_judge != HEALTHBENCH_JUDGE_PROFILE:
                raise ValueError(
                    f"an R2 Flow run pins the HealthBench judge {HEALTHBENCH_JUDGE_PROFILE}"
                )
        elif self.healthbench_judge is not None:
            raise ValueError("an R2 Flow run without HealthBench declares no HealthBench judge")
        if (r2flow.healthbench_reward is None) == ("healthbench" in self.domains):
            raise ValueError(
                "the R2 Flow HealthBench reward rule is declared iff HealthBench trains"
            )
        from skillev.evaluation.healthbench_judge_recovery import HEALTHBENCH_JUDGE_RECOVERY

        if self.healthbench_judge_recovery is not None and (
            self.healthbench_judge_recovery != HEALTHBENCH_JUDGE_RECOVERY
            or "healthbench" not in self.domains
        ):
            raise ValueError(
                "healthbench_judge_recovery declares healthbench-judge-recovery@3 for a run "
                "that trains HealthBench"
            )
        if self.hotpot_deliberation:
            raise ValueError("an R2 Flow run declares no Hotpot deliberation (task semantics @15)")
        if self.checkpoint_every != r2flow.evolution.trigger.cadence_steps:
            raise ValueError("checkpoint_every must equal the V_q trigger cadence")
        method = r2flow.method
        if self.ttb_beta != method.temperature_beta or self.epsilon != method.epsilon_min:
            raise ValueError("formal eta/epsilon differ from the declared R2 Flow method")
        for name, inert in R2FLOW_FIXED_CONTROLS:
            if getattr(self, name) != inert:
                raise ValueError(f"R2 Flow control {name} must stay at its fixed value")
        if (
            self.action_wire != NATIVE_EVENT_CALL_WIRE
            or self.skill_exposure != SKILL_EXPOSURE
            or self.initial_skill_profile != R2FLOW_INITIAL_PROFILES[self.format]
            or self.tool_set != R2FLOW_TOOL_SET
            or self.task_semantic_guidance != TASK_SEMANTIC_GUIDANCE
            or self.input_window != INPUT_WINDOW_VERSION
            or self.executor is None
            or self.psi_learning_rate is None
            or self.hotpot_deliberation
        ):
            raise ValueError(
                "R2 Flow couples wire @7, the event boundary, method@4 with a state map, "
                "rollout @10 with the executor, skill-md-invoke@2, the R2 Flow tool set, "
                "the format's initial profile, task semantics @17 and input window @2"
            )
        self._validate_answer_writer()
        if any(
            domain not in self.domains or turns > self.max_turns
            for domain, turns in self.turns_by_domain
        ):
            raise ValueError("domain horizons must name active domains within the global cap")
        if any(
            domain not in self.domains
            for domain, _ in (*self.reasoning_tokens_by_domain, *self.action_tokens_by_domain)
        ):
            raise ValueError("R2 Flow domain token budgets must name active domains")
        if (
            self.final_turn_completion is not None
            and self.final_turn_completion not in FINAL_TURN_COMPLETION_RULES
        ):
            raise ValueError(
                "final_turn_completion declares the environment rule final-turn-submit@1"
            )
        if self.action_decoding is not None and self.action_decoding not in ACTION_DECODING_RULES:
            raise ValueError(f"action_decoding declares the decoding rule {ACTION_GREEDY_UNSEEDED}")
        self._validate_changed_families_rollout()

    def _validate_answer_writer(self) -> None:
        assert self.executor is not None
        self._validate_writer_backward_passes()
        if (
            self.executor.max_input_tokens < self.max_input_tokens
            or self.executor.max_output_tokens < self.maximum_action_tokens
        ):
            raise ValueError(
                f"{self.tool_set} (executor-answer@1): the executor caps must cover "
                "max_input_tokens and every domain's action budget (the writer's output cap)"
            )

    def _validate_writer_backward_passes(self) -> None:
        from skillev.benchmarks.r2flow_tool_set import TOOL_REGIMES
        from skillev.contracts.state_map import (
            PB_IN_EDGE_SOFTMAX,
            SIGMA_TRACE_QUOTIENT,
            commuting_free_text_pairs,
        )

        commuting = {
            domain: pairs
            for domain in self.domains
            if (pairs := commuting_free_text_pairs(cast(list[str], TOOL_REGIMES[domain]["tools"])))
        }
        if commuting:
            raise ValueError(
                f"{self.tool_set} (executor-answer@1) with {PB_IN_EDGE_SOFTMAX}: under "
                f"{SIGMA_TRACE_QUOTIENT} a free-text event can be one of several in-edges, whose "
                f"text P_B scores ({commuting})"
            )

    def _validate_changed_families_rollout(self) -> None:
        from skillev.evolution.task_features import transferable_family
        from skillev.r2flow_evolution.phi import BOOTSTRAP_DOMAIN_FAMILIES

        if self.skill_visibility is None:
            raise ValueError(
                "changed-families validation requires skill_visibility nonempty-only@1"
            )
        mismatched = sorted(
            domain
            for domain in self.domains
            if transferable_family(domain) != BOOTSTRAP_DOMAIN_FAMILIES.get(domain)
        )
        if mismatched:
            raise ValueError(
                f"changed-families validation: the retrieval family of {mismatched} is not the "
                "gate's scope family"
            )

    @property
    def optimizer_config(self) -> OptimizerConfig:
        assert self.psi_learning_rate is not None
        clips = dict(self.gradient_group_clip)
        return OptimizerConfig(
            self.adapter_learning_rate,
            self.z_learning_rate,
            self.psi_learning_rate,
            self.weight_decay,
            stability=PolicyStabilityConfig(
                forward_max_norm=clips.get("forward"),
                backward_max_norm=clips.get("backward"),
                z_max_norm=clips.get("z"),
            )
            if clips
            else None,
        )

    @property
    def method_config(self) -> TTBMethodConfig:
        assert self.r2flow is not None
        return self.r2flow.method

    def _validate_owner_model(self) -> None:
        if self.model != "Qwen3.5-9B" or self.base_dtype != "bfloat16":
            raise ValueError("formal training requires the declared single Qwen BF16 owner")

    @property
    def run_plan(self) -> ExactAttemptRunPlan:
        return ExactAttemptRunPlan(
            self.steps - self.closure_steps, self.closure_steps, self.maximum_cycles
        )

    @property
    def batch_size(self) -> int:
        return len(self.domains) * TRAJECTORIES_PER_QUESTION

    @property
    def scheduled_domains(self) -> tuple[TrainingBenchmark, ...]:
        return tuple(d for d in TRAINING_DOMAINS if d.value in self.domains)

    @property
    def condition(self) -> str:
        suffix = "+hotpot-evidence-deliberation@1" if self.hotpot_deliberation else ""
        if self.healthbench_judge is not None:
            suffix += "+" + self.healthbench_judge
        if not self.format.endswith("@2"):
            suffix += "+domain-thinking@1-off=" + ",".join(self.thinking_off_domains)
        if self.format.endswith(("@4", "@5", "@6", "@7", "@8")):
            suffix += (
                f"+interface@1-phase={int(self.phase_context)}"
                f"-wire={self.action_wire}-skills={self.skill_exposure}"
            )
        if self.public_action_semantics:
            suffix += "+public-action-semantics@1"
        if self.token_budget_notice:
            suffix += "+token-budget-notice@1"
        if self.reasoning_tool_catalog:
            suffix += "+reasoning-tool-catalog@1"
        if is_current_candidate(self.format):
            suffix += "+" + self.task_semantic_guidance
        if self.reasoning_tokens_by_domain:
            suffix += "+domain-reasoning-budgets@1=" + ",".join(
                f"{domain}:{tokens}" for domain, tokens in self.reasoning_tokens_by_domain
            )
            suffix += f"-static{self.static_max_turns}-interactive{self.max_turns}"
        if self.initial_skill_profile != "public-advisory@2":
            suffix += "+initial-skills=" + self.initial_skill_profile
        if self.learning_protocol is not None:
            suffix += "+" + self.learning_protocol
        if self.action_tokens_by_domain:
            suffix += "+domain-action-budgets@1=" + ",".join(
                f"{domain}:{tokens}" for domain, tokens in self.action_tokens_by_domain
            )
        if self.turns_by_domain:
            suffix += "+domain-horizons@1=" + ",".join(
                f"{domain}:{turns}" for domain, turns in self.turns_by_domain
            )
        if self.final_turn_completion is not None:
            suffix += "+" + self.final_turn_completion
        if self.skill_call_budget_by_domain:
            suffix += f"+{SKILL_CALL_BUDGET}=" + ",".join(
                f"{domain}:{budget}" for domain, budget in self.skill_call_budget_by_domain
            )
        if self.skill_visibility is not None:
            suffix += "+skill-visibility=" + self.skill_visibility
        if self.reasoning_call_line_domains:
            suffix += f"+{REASONING_CALL_LINE}=" + ",".join(self.reasoning_call_line_domains)
        if self.reasoning_stop_domains:
            suffix += f"+{REASONING_STOP_AT_TOOL_CALL}=" + ",".join(self.reasoning_stop_domains)
        if self.healthbench_judge_recovery is not None:
            suffix += "+" + self.healthbench_judge_recovery
        if self.thinking_on_domains:
            suffix += "+domain-thinking-declared-on=" + ",".join(self.thinking_on_domains)
        if self.training_source_exclusions:
            suffix += f"+{TRAINING_SOURCE_EXCLUSIONS}=" + ",".join(
                f"{domain}:{source}" for domain, source in self.training_source_exclusions
            )
        if self.gradient_group_clip:
            suffix += f"+{GRADIENT_GROUP_CLIP}=" + ",".join(
                f"{group}:{norm:g}" for group, norm in self.gradient_group_clip
            )
        if self.action_decoding is not None:
            suffix += "+" + self.action_decoding
        if self.r2flow is not None:
            assert self.executor is not None
            assert self.tool_set is not None
            suffix += (
                f"+tool-set={self.tool_set}"
                f"+executor={self.executor.format}"
                f"#{self.executor.identity().removeprefix('sha256:')[:16]}"
                f"+psi-lr={self.psi_learning_rate!r}"
                "+legacy-evolution=disabled+" + self.r2flow.condition_segment()
            )
        return (
            domain_schedule_condition(self.scheduled_domains)
            + "+domain-horizons-native-reasoning@2"
            + "+completion-wire@2+"
            + self.input_window
            + suffix
        )

    @property
    def task_budget(self) -> RolloutBudgetProfile:
        return RolloutBudgetProfile(
            "formal-interactive-budget@2",
            self.max_turns,
            self.max_reasoning_tokens,
            self.max_action_tokens,
        )

    @property
    def static_task_budget(self) -> RolloutBudgetProfile:
        return RolloutBudgetProfile(
            "formal-static-budget@2",
            self.static_max_turns,
            self.max_reasoning_tokens,
            self.max_action_tokens,
        )

    @property
    def maximum_reasoning_tokens(self) -> int:
        return max((self.max_reasoning_tokens, *(v for _, v in self.reasoning_tokens_by_domain)))

    @property
    def maximum_action_tokens(self) -> int:
        return max((self.max_action_tokens, *(v for _, v in self.action_tokens_by_domain)))

    @property
    def domain_task_budgets(self) -> dict[str, RolloutBudgetProfile]:
        reasoning, action = (
            dict(self.reasoning_tokens_by_domain),
            dict(self.action_tokens_by_domain),
        )
        turns = dict(self.turns_by_domain)
        return {
            domain: replace(
                self.task_budget if domain == "alfworld" else self.static_task_budget,
                profile_id=(
                    "formal-domain-phase-budget@2:"
                    if self.action_tokens_by_domain
                    else "formal-domain-reasoning-budget@1:"
                )
                + domain,
                max_reasoning_tokens=reasoning.get(domain, self.max_reasoning_tokens),
                max_action_tokens=action.get(domain, self.max_action_tokens),
                max_turns=turns.get(
                    domain,
                    self.max_turns if domain == "alfworld" else self.static_max_turns,
                ),
            )
            for domain in sorted(reasoning.keys() | action.keys() | turns.keys())
        }

    @property
    def sampling_config(self) -> PolicyRolloutConfig:
        per_rollout_maximum = conservative_rollout_maximum(
            max_turns=self.max_turns,
            max_reasoning_tokens=self.maximum_reasoning_tokens,
            max_action_tokens=self.maximum_action_tokens,
            max_model_input_tokens=self.max_input_tokens,
            max_tool_wall_time_milliseconds=120_000,
        )
        executor = self.executor
        final_turn_completion = self.final_turn_completion
        skill_visibility = self.skill_visibility
        action_decoding = self.action_decoding
        if (
            executor is None
            or final_turn_completion is None
            or skill_visibility is None
            or action_decoding is None
        ):
            raise ValueError(
                "the rollout declares executor, final_turn_completion, skill_visibility "
                "and action_decoding"
            )
        per_rollout_maximum = per_rollout_maximum.add(
            BudgetVector(
                input_tokens=executor.max_input_tokens,
                output_tokens=executor.max_output_tokens,
                model_calls=1,
            ).scale(self.max_turns)
        )
        return PolicyRolloutConfig(
            base_seed=self.seed,
            max_turns=self.max_turns,
            max_reasoning_tokens=self.maximum_reasoning_tokens,
            max_action_tokens=self.maximum_action_tokens,
            reasoning_native_thinking=self.reasoning_native_thinking,
            per_rollout_maximum=per_rollout_maximum,
            reasoning_by_domain=tuple(sorted(self.reasoning_modes.items())),
            input_window=ModelInputWindow(self.max_input_tokens, self.input_window),
            executor=executor,
            final_turn_completion=final_turn_completion,
            skill_call_budget_by_domain=self.skill_call_budget_by_domain,
            skill_visibility=skill_visibility,
            reasoning_call_line_domains=self.reasoning_call_line_domains,
            reasoning_stop_domains=self.reasoning_stop_domains,
            action_decoding=action_decoding,
        )

    def application_config(self, run_id: str) -> ApplicationConfig:
        base, _ = build_application_config(
            run_id=run_id,
            steps=self.steps,
            run_plan=self.run_plan,
            batch_size=self.batch_size,
            checkpoint_every=self.checkpoint_every,
            method=self.method_config,
            rollout=self.sampling_config,
            optimizer=self.optimizer_config,
            max_input_tokens=self.max_input_tokens,
        )
        return replace(
            base,
            calibration=CalibrationConfig(self.prior_alpha, self.prior_beta, self.k),
            diagnostics=DiagnosticsConfig(window_size=self.window, stagnation_rho=self.rho),
            evolution=replace(base.evolution, k=self.k, entropy_window=self.window),
            r2flow=self.r2flow,
        )

    def require_backbone(self, backbone: QwenMultimodalBackboneConfig) -> None:
        if (
            backbone.lora_rank != self.lora_rank
            or backbone.lora_alpha != self.lora_alpha
            or backbone.lora_dropout != 0
            or backbone.torch_dtype != self.base_dtype
        ):
            raise ValueError("initial model binding differs from the declared LoRA/BF16 condition")
        if self.r2flow is not None:
            method = self.r2flow.method
            head = backbone.flow_head
            z_init = backbone.z_initialization
            if (
                head is None
                or head.eta != method.temperature_beta
                or head.epsilon != method.epsilon_min
                or z_init.mode != "output-bias-log-epsilon@1"
                or z_init.epsilon != method.epsilon_min
            ):
                raise ValueError(
                    "an R2 Flow run needs a preparation whose backbone declares the F_psi "
                    "head and the log-epsilon Z initialization of the declared method"
                )

    @property
    def reasoning_modes(self) -> dict[str, bool]:
        return {d: d not in self.thinking_off_domains for d in self.domains}

    def to_value(self) -> dict[str, JsonValue]:
        value = cast(dict[str, JsonValue], normalize_json(asdict(self)))
        if self.domains == tuple(d.value for d in TRAINING_DOMAINS):
            value.pop("domains")
        if self.healthbench_judge is None:
            value.pop("healthbench_judge")
        if self.initial_skill_profile == "public-advisory@2":
            value.pop("initial_skill_profile")
        if self.learning_protocol is None:
            value.pop("learning_protocol")
        if not self.token_budget_notice:
            value.pop("token_budget_notice")
        if not self.reasoning_tool_catalog:
            value.pop("reasoning_tool_catalog")
        if not self.reasoning_tokens_by_domain:
            value.pop("reasoning_tokens_by_domain")
        if not self.action_tokens_by_domain:
            value.pop("action_tokens_by_domain")
        if self.format.endswith("@2"):
            value.pop("thinking_off_domains")
        if not self.format.endswith(("@4", "@5", "@6", "@7", "@8")):
            for field in ("phase_context", "action_wire", "skill_exposure"):
                value.pop(field)
        if not self.format.endswith(("@5", "@6", "@7", "@8")):
            value.pop("public_action_semantics")
        if not is_current_candidate(self.format):
            value.pop("task_semantic_guidance")
        return self._r2flow_value(value)

    def _r2flow_value(self, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        if self.format not in R2FLOW_FORMATS:
            for name in R2FLOW_ONLY_FIELDS:
                value.pop(name)
            return value
        assert self.r2flow is not None
        assert self.executor is not None
        value["r2flow"] = self.r2flow.to_value()
        value["executor"] = self.executor.to_value()
        if self.final_turn_completion is None:
            value.pop("final_turn_completion")
        if not self.skill_call_budget_by_domain:
            value.pop("skill_call_budget_by_domain")
        if not self.gradient_group_clip:
            value.pop("gradient_group_clip")
        else:
            value["gradient_group_clip"] = [list(item) for item in self.gradient_group_clip]
        if self.skill_visibility is None:
            value.pop("skill_visibility")
        if not self.reasoning_call_line_domains:
            value.pop("reasoning_call_line_domains")
        if not self.reasoning_stop_domains:
            value.pop("reasoning_stop_domains")
        if self.healthbench_judge_recovery is None:
            value.pop("healthbench_judge_recovery")
        if not self.thinking_on_domains:
            value.pop("thinking_on_domains")
        if not self.training_source_exclusions:
            value.pop("training_source_exclusions")
        else:
            value["training_source_exclusions"] = [
                list(item) for item in self.training_source_exclusions
            ]
        if self.action_decoding is None:
            value.pop("action_decoding")
        return value

    def expanded_value(self) -> dict[str, JsonValue]:
        value = cast(dict[str, JsonValue], normalize_json(asdict(self)))
        if self.domains == tuple(d.value for d in TRAINING_DOMAINS):
            value.pop("domains")
        if self.learning_protocol is None:
            value.pop("learning_protocol")
        if self._r2flow_without_judge:
            value.pop("healthbench_judge")
        return self._r2flow_value(value)

    @property
    def _r2flow_without_judge(self) -> bool:
        return self.format in R2FLOW_FORMATS and "healthbench" not in self.domains

    def schedule_summary(self) -> dict[str, JsonValue]:
        return {
            "domains": list(self.domains),
            "questions_per_step": len(self.domains),
            "trajectories_per_question": TRAJECTORIES_PER_QUESTION,
            "batch_size": self.batch_size,
            "question_occurrences": self.steps * len(self.domains),
            "trajectories": self.steps * self.batch_size,
            "condition": self.condition,
            "healthbench_judge": None if self._r2flow_without_judge else self.healthbench_judge,
            **({"learning_protocol": self.learning_protocol} if self.learning_protocol else {}),
            "static_max_turns": self.static_max_turns,
            "interactive_max_turns": self.max_turns,
            "input_window": self.input_window,
            "reasoning_native_thinking_by_domain": normalize_json(self.reasoning_modes),
            "action_tokens_by_domain": normalize_json(
                {
                    d.value: dict(self.action_tokens_by_domain).get(d.value, self.max_action_tokens)
                    for d in self.scheduled_domains
                }
            ),
            "reasoning_tokens_by_domain": normalize_json(
                {
                    d.value: dict(self.reasoning_tokens_by_domain).get(
                        d.value, self.max_reasoning_tokens
                    )
                    for d in self.scheduled_domains
                }
            ),
        }

    @classmethod
    def load(cls, path: Path) -> BayesianFormalConfig:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        expected = {field.name for field in fields(cls)}
        if isinstance(value, dict) and "domains" not in value:
            expected.remove("domains")
        if isinstance(value, dict) and "initial_skill_profile" not in value:
            expected.remove("initial_skill_profile")
        if isinstance(value, dict) and "learning_protocol" not in value:
            expected.remove("learning_protocol")
        if isinstance(value, dict) and "healthbench_judge" not in value:
            expected.remove("healthbench_judge")
        if isinstance(value, dict) and "token_budget_notice" not in value:
            expected.remove("token_budget_notice")
        if isinstance(value, dict) and "reasoning_tool_catalog" not in value:
            expected.remove("reasoning_tool_catalog")
        if isinstance(value, dict) and "action_tokens_by_domain" not in value:
            expected.remove("action_tokens_by_domain")
        if isinstance(value, dict) and "reasoning_tokens_by_domain" not in value:
            expected.remove("reasoning_tokens_by_domain")
        if isinstance(value, dict) and value.get("format") == "skillev-bayesian-formal-training@2":
            expected.remove("thinking_off_domains")
        if isinstance(value, dict) and value.get("format") not in {
            "skillev-bayesian-formal-training@4",
            "skillev-bayesian-formal-training@5",
            "skillev-bayesian-formal-training@6",
            R2FLOW_FORMAL,
            R2FLOW_EVOLVE_FORMAL,
        }:
            expected -= {"phase_context", "action_wire", "skill_exposure"}
        if isinstance(value, dict) and value.get("format") not in {
            "skillev-bayesian-formal-training@5",
            "skillev-bayesian-formal-training@6",
            R2FLOW_FORMAL,
            R2FLOW_EVOLVE_FORMAL,
        }:
            expected.remove("public_action_semantics")
        if isinstance(value, dict) and value.get("format") not in {
            "skillev-bayesian-formal-training@6",
            R2FLOW_FORMAL,
            R2FLOW_EVOLVE_FORMAL,
        }:
            expected.remove("task_semantic_guidance")
        if isinstance(value, dict) and value.get("format") not in R2FLOW_FORMATS:
            expected -= set(R2FLOW_ONLY_FIELDS)
        else:
            for optional in (
                "turns_by_domain",
                "final_turn_completion",
                "skill_call_budget_by_domain",
                "gradient_group_clip",
                "skill_visibility",
                "reasoning_call_line_domains",
                "reasoning_stop_domains",
                "healthbench_judge_recovery",
                "thinking_on_domains",
                "training_source_exclusions",
                "action_decoding",
            ):
                if isinstance(value, dict) and optional not in value:
                    expected.remove(optional)
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("formal configuration must explicitly declare every control")
        return cls(**value)
