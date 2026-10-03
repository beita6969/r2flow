from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

RUN_FORMAT: Final = "skillev-benchmark-protocol@10"
RUN_SEED: Final = 0
TRAINING_EPISODES_PER_BENCHMARK: Final = 512


class RunProtocolError(ValueError):
    pass


class Benchmark(StrEnum):
    HOTPOT_QA = "hotpotqa"
    TRIVIA_QA = "triviaqa"
    AIME_2026 = "aime-2026"
    HEALTHBENCH = "healthbench"
    ALFWORLD = "alfworld"
    MBPP_PLUS_FIXED_100 = "mbpp-plus-fixed-100"


ACTIVE_BENCHMARKS_: Final = tuple(Benchmark)


class FormalMethod(StrEnum):
    SKILLFLOW_BASELINE = "skillflow-baseline"
    BAYESIAN_IMPROVE_FULL = "bayesian-improve-full"
    BAYESIAN_IMPROVE_NO_CALIBRATION = "bayesian-improve-no-calibration"


class PopulationRole(StrEnum):
    TRAINING = "training"
    VALIDATION = "validation"
    FINAL_EVALUATION = "final-evaluation"


class PopulationSampling(StrEnum):
    FIXED_TRAINING_MIX = "fixed-training-mix-512"
    HELD_OUT_VALIDATION = "held-out-validation"
    FULL = "full"


class RewardProjectionRule(StrEnum):
    IDENTITY = "identity"
    CLIP_TO_ZERO_ONE = "clip-to-zero-one"


class PosteriorSuccessRule(StrEnum):
    EQUALS_ONE = "equals-one"
    SCORE_AND_NO_NEGATIVE_RUBRIC = "score-at-least-and-no-negative-rubric-triggered"


@dataclass(frozen=True, slots=True)
class BenchmarkPopulation:
    benchmark: Benchmark
    population_id: str
    role: PopulationRole
    source_version: str
    sampling: PopulationSampling
    evaluator_identity: str

    def __post_init__(self) -> None:
        for value in (self.population_id, self.source_version, self.evaluator_identity):
            if not value.strip():
                raise RunProtocolError("population text fields must be non-empty")
        expected = {
            PopulationRole.TRAINING: PopulationSampling.FIXED_TRAINING_MIX,
            PopulationRole.VALIDATION: PopulationSampling.HELD_OUT_VALIDATION,
            PopulationRole.FINAL_EVALUATION: PopulationSampling.FULL,
        }[self.role]
        if self.sampling is not expected:
            raise RunProtocolError("population role and sampling mode disagree")


@dataclass(frozen=True, slots=True)
class TTBRewardProjection:
    source: str
    rule: RewardProjectionRule

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise RunProtocolError("reward projection source must be non-empty")

    def project(self, trusted_native_fields: Mapping[str, object]) -> float:
        value = _number(trusted_native_fields.get(self.source), label=self.source)
        if self.rule is RewardProjectionRule.CLIP_TO_ZERO_ONE:
            return min(1.0, max(0.0, value))
        if not 0.0 <= value <= 1.0:
            raise RunProtocolError("identity reward source must lie in [0, 1]")
        return value


@dataclass(frozen=True, slots=True)
class PosteriorSuccessProjection:
    sources: tuple[str, ...]
    rule: PosteriorSuccessRule
    threshold: float | None = None

    def __post_init__(self) -> None:
        if not self.sources or any(not source.strip() for source in self.sources):
            raise RunProtocolError("success projection requires non-empty sources")
        if len(set(self.sources)) != len(self.sources):
            raise RunProtocolError("success projection sources must be unique")
        if self.rule is PosteriorSuccessRule.EQUALS_ONE:
            if len(self.sources) != 1 or self.threshold is not None:
                raise RunProtocolError("equals-one requires one source and no threshold")
        elif self.rule is PosteriorSuccessRule.SCORE_AND_NO_NEGATIVE_RUBRIC:
            if len(self.sources) != 2 or self.threshold is None:
                raise RunProtocolError("rubric conjunction requires two sources and a threshold")
            if not 0.0 < self.threshold <= 1.0:
                raise RunProtocolError("success threshold must lie in (0, 1]")

    def project(self, trusted_native_fields: Mapping[str, object]) -> bool:
        if self.rule is PosteriorSuccessRule.EQUALS_ONE:
            return _number(trusted_native_fields.get(self.sources[0]), label=self.sources[0]) == 1.0
        score = _number(trusted_native_fields.get(self.sources[0]), label=self.sources[0])
        negative_count = _number(trusted_native_fields.get(self.sources[1]), label=self.sources[1])
        if negative_count < 0.0 or not negative_count.is_integer():
            raise RunProtocolError("negative-rubric count must be a non-negative integer")
        if self.threshold is None:
            raise RunProtocolError("rubric projection threshold is absent")
        return score >= self.threshold and negative_count == 0.0


@dataclass(frozen=True, slots=True)
class BenchmarkProtocol:
    benchmark: Benchmark
    source_version: str
    populations: tuple[BenchmarkPopulation, ...]
    native_metrics: tuple[str, ...]
    reward_projection: TTBRewardProjection
    success_projection: PosteriorSuccessProjection
    evaluator_identity: str
    aggregation_constraint: str | None = None
    excluded_primary_variants: tuple[str, ...] = ()
    final_selection: str | None = None
    forbidden_label: str | None = None

    def __post_init__(self) -> None:
        if not self.source_version.strip() or not self.evaluator_identity.strip():
            raise RunProtocolError("benchmark identity fields must be non-empty")
        if not self.native_metrics or any(not item.strip() for item in self.native_metrics):
            raise RunProtocolError("benchmark native metrics must be non-empty")
        if len(set(self.native_metrics)) != len(self.native_metrics):
            raise RunProtocolError("benchmark native metrics must be unique")
        if any(item.benchmark is not self.benchmark for item in self.populations):
            raise RunProtocolError("population belongs to another benchmark")
        roles = tuple(item.role for item in self.populations)
        if roles.count(PopulationRole.TRAINING) != 1:
            raise RunProtocolError("benchmark requires exactly one training population")
        if roles.count(PopulationRole.VALIDATION) != 1:
            raise RunProtocolError("benchmark requires exactly one validation population")
        if roles.count(PopulationRole.FINAL_EVALUATION) < 1:
            raise RunProtocolError("benchmark requires a final-evaluation population")

    def population(self, role: PopulationRole) -> tuple[BenchmarkPopulation, ...]:
        return tuple(item for item in self.populations if item.role is role)


@dataclass(frozen=True, slots=True)
class TrainingMixPolicy:
    episodes_per_benchmark: int
    total_episodes: int
    benchmark_order: tuple[Benchmark, ...]
    within_block_order: str
    undersized_population_rule: str
    final_mix_order: str
    adaptive_sampling: bool
    require_disjoint_source_ids: bool
    require_disjoint_normalized_content: bool
    final_evaluation_is_read_only: bool

    def __post_init__(self) -> None:
        if self.episodes_per_benchmark != TRAINING_EPISODES_PER_BENCHMARK:
            raise RunProtocolError("training block size must be 512")
        if self.benchmark_order != ACTIVE_BENCHMARKS_:
            raise RunProtocolError("training benchmark order differs from Protocol 10")
        if self.total_episodes != len(self.benchmark_order) * self.episodes_per_benchmark:
            raise RunProtocolError("training mixture total is inconsistent")
        if self.within_block_order != "deterministic-shuffle-seed-0":
            raise RunProtocolError("training mixture has another within-block order")
        if self.undersized_population_rule != (
            "repeat-deterministically-shuffled-cycles-with-distinct-episode-ids"
        ):
            raise RunProtocolError("training mixture has another repeat rule")
        if self.final_mix_order != "deterministic-global-shuffle-seed-0":
            raise RunProtocolError("training mixture must use the SkillFlow-style global shuffle")
        if self.adaptive_sampling:
            raise RunProtocolError("adaptive training sampling is forbidden")
        if not (
            self.require_disjoint_source_ids
            and self.require_disjoint_normalized_content
            and self.final_evaluation_is_read_only
        ):
            raise RunProtocolError("Protocol 10 population isolation cannot be weakened")


@dataclass(frozen=True, slots=True)
class TerminalContract:
    reward_range: tuple[float, float]
    posterior_success_values: tuple[int, int]
    infrastructure_failure: str
    completed_candidate_failure: str
    private_evaluator_payload_model_visible: bool

    def __post_init__(self) -> None:
        if self.reward_range != (0.0, 1.0):
            raise RunProtocolError("TTB reward range must be [0, 1]")
        if self.posterior_success_values != (0, 1):
            raise RunProtocolError("posterior observations must be Bernoulli")
        if self.infrastructure_failure != "abort-uncommitted-step":
            raise RunProtocolError("infrastructure failure must abort the uncommitted step")
        if self.completed_candidate_failure != "valid-zero-reward":
            raise RunProtocolError("completed candidate failures must remain valid zero rewards")
        if self.private_evaluator_payload_model_visible:
            raise RunProtocolError("private evaluator payload cannot be model-visible")


@dataclass(frozen=True, slots=True)
class ProtocolAmendment:
    amendment_id: str
    reason: str
    adopted_before_first_formal_run: bool
    prior_gpt_4_1_profile_used_for_formal_run: bool

    def __post_init__(self) -> None:
        if not self.amendment_id.strip() or not self.reason.strip():
            raise RunProtocolError("Protocol 10 amendment identity is incomplete")
        if not self.adopted_before_first_formal_run:
            raise RunProtocolError("the active amendment must precede the first formal run")
        if self.prior_gpt_4_1_profile_used_for_formal_run:
            raise RunProtocolError("the superseded HealthBench profile cannot have formal results")


@dataclass(frozen=True, slots=True)
class ActiveBenchmarkProtocol:
    seed: int
    methods: tuple[FormalMethod, ...]
    benchmarks: tuple[BenchmarkProtocol, ...]
    training_mix: TrainingMixPolicy
    terminal_contract: TerminalContract
    amendment: ProtocolAmendment
    executable: bool
    format: str = RUN_FORMAT

    def __post_init__(self) -> None:
        if self.format != RUN_FORMAT:
            raise RunProtocolError("active protocol must use Protocol 10")
        if self.seed != RUN_SEED:
            raise RunProtocolError("Protocol 10 uses seed 0")
        if self.methods != tuple(FormalMethod):
            raise RunProtocolError("formal method order differs from Protocol 10")
        if tuple(item.benchmark for item in self.benchmarks) != ACTIVE_BENCHMARKS_:
            raise RunProtocolError("active benchmark suite differs from Protocol 10")
        population_ids = tuple(
            population.population_id
            for benchmark in self.benchmarks
            for population in benchmark.populations
        )
        if len(population_ids) != len(set(population_ids)):
            raise RunProtocolError("population identities must be globally unique")

    def benchmark(self, benchmark: Benchmark) -> BenchmarkProtocol:
        return next(item for item in self.benchmarks if item.benchmark is benchmark)

    def require_execution_ready(self) -> None:
        if not self.executable:
            raise RunProtocolError("Protocol 10 execution gate is not open")


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise RunProtocolError(f"{label} must be a mapping")
    return value


def _sequence(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise RunProtocolError(f"{label} must be a sequence")
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunProtocolError(f"{label} must be non-empty text")
    return value


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RunProtocolError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RunProtocolError(f"{label} must be finite")
    return result


def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise RunProtocolError(f"{label} must be boolean")
    return value


def _number_pair(value: object, *, label: str) -> tuple[float, float]:
    items = _sequence(value, label=label)
    if len(items) != 2:
        raise RunProtocolError(f"{label} must contain exactly two values")
    return (_number(items[0], label=label), _number(items[1], label=label))


def _integer_pair(value: object, *, label: str) -> tuple[int, int]:
    first, second = _number_pair(value, label=label)
    if not first.is_integer() or not second.is_integer():
        raise RunProtocolError(f"{label} must contain integers")
    return (int(first), int(second))


def _text_tuple(value: object, *, label: str) -> tuple[str, ...]:
    return tuple(_text(item, label=label) for item in _sequence(value, label=label))


def _optional_text(value: object, *, label: str) -> str | None:
    return None if value is None else _text(value, label=label)


def _projection_sources(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (_text(value, label="success projection source"),)
    return _text_tuple(value, label="success projection source")


def _population_sampling(role: PopulationRole, value: object) -> PopulationSampling:
    if role is PopulationRole.TRAINING:
        if value is not None:
            raise RunProtocolError("training population sampling is fixed by the mix policy")
        return PopulationSampling.FIXED_TRAINING_MIX
    if role is PopulationRole.VALIDATION:
        if value is not None:
            raise RunProtocolError("validation population sampling is held out")
        return PopulationSampling.HELD_OUT_VALIDATION
    if value != "full":
        raise RunProtocolError("final evaluation populations must use full sampling")
    return PopulationSampling.FULL


def _parse_benchmark(value: object) -> BenchmarkProtocol:
    row = _mapping(value, label="benchmark")
    benchmark = Benchmark(_text(row.get("id"), label="benchmark id"))
    version = _text(row.get("version"), label="benchmark version")
    evaluator = _text(row.get("evaluator"), label="benchmark evaluator")
    populations = tuple(
        BenchmarkPopulation(
            benchmark=benchmark,
            population_id=_text(population.get("id"), label="population id"),
            role=(role := PopulationRole(_text(population.get("role"), label="population role"))),
            source_version=version,
            sampling=_population_sampling(role, population.get("sampling")),
            evaluator_identity=evaluator,
        )
        for population in (
            _mapping(item, label="population")
            for item in _sequence(row.get("populations"), label="populations")
        )
    )
    reward = _mapping(row.get("ttb_reward_projection"), label="reward projection")
    success = _mapping(row.get("posterior_success_projection"), label="success projection")
    threshold_value = success.get("threshold")
    if threshold_value is not None and (
        isinstance(threshold_value, bool) or not isinstance(threshold_value, int | float)
    ):
        raise RunProtocolError("success threshold must be numeric")
    return BenchmarkProtocol(
        benchmark=benchmark,
        source_version=version,
        populations=populations,
        native_metrics=_text_tuple(row.get("native_metrics"), label="native metrics"),
        reward_projection=TTBRewardProjection(
            source=_text(reward.get("source"), label="reward projection source"),
            rule=RewardProjectionRule(_text(reward.get("rule"), label="reward projection rule")),
        ),
        success_projection=PosteriorSuccessProjection(
            sources=_projection_sources(success.get("source")),
            rule=PosteriorSuccessRule(_text(success.get("rule"), label="success projection rule")),
            threshold=None if threshold_value is None else float(threshold_value),
        ),
        evaluator_identity=evaluator,
        aggregation_constraint=_optional_text(
            row.get("aggregation_constraint"), label="aggregation constraint"
        ),
        excluded_primary_variants=(
            ()
            if row.get("excluded_primary_variants") is None
            else _text_tuple(row.get("excluded_primary_variants"), label="excluded primary variant")
        ),
        final_selection=_optional_text(row.get("final_selection"), label="final selection"),
        forbidden_label=_optional_text(row.get("forbidden_label"), label="forbidden label"),
    )


def parse_active_protocol(value: object) -> ActiveBenchmarkProtocol:
    root = _mapping(value, label="active protocol")
    if root.get("format") != RUN_FORMAT:
        raise RunProtocolError("new formal runs accept only Protocol 10")
    if (
        root.get("status") != "frozen-scientific-specification"
        or root.get("method_authority") != "idea.tex"
        or root.get("evaluation_authority") != "evaluation.tex"
    ):
        raise RunProtocolError("Protocol 10 scientific authorities are not frozen")
    policy = _mapping(root.get("population_policy"), label="population policy")
    gate = _mapping(root.get("execution_gate"), label="execution gate")
    terminal = _mapping(root.get("terminal_contract"), label="terminal contract")
    amendment = _mapping(root.get("amendment"), label="protocol amendment")
    executable = _boolean(gate.get("executable"), label="execution gate")
    adaptive = policy.get("adaptive_sampling")
    if adaptive not in {"forbidden", "allowed"}:
        raise RunProtocolError("protocol gate or sampling policy is invalid")
    if _text_tuple(policy.get("roles"), label="population roles") != tuple(
        role.value for role in PopulationRole
    ):
        raise RunProtocolError("population roles differ from Protocol 10")
    if policy.get("training_count_rule_applies_to") != "training-only":
        raise RunProtocolError("the 512-count rule must apply only to training")
    episodes = policy.get("training_episodes_per_benchmark")
    total = policy.get("training_episode_total")
    if type(episodes) is not int or type(total) is not int:
        raise RunProtocolError("training mixture counts must be integers")
    seed = root.get("seed")
    if type(seed) is not int:
        raise RunProtocolError("protocol seed must be an integer")
    return ActiveBenchmarkProtocol(
        seed=seed,
        methods=tuple(
            FormalMethod(_text(item, label="formal method"))
            for item in _sequence(root.get("formal_methods"), label="formal methods")
        ),
        benchmarks=tuple(
            _parse_benchmark(item) for item in _sequence(root.get("benchmarks"), label="benchmarks")
        ),
        training_mix=TrainingMixPolicy(
            episodes_per_benchmark=episodes,
            total_episodes=total,
            benchmark_order=tuple(
                Benchmark(_text(item, label="training benchmark"))
                for item in _sequence(
                    policy.get("training_block_order"), label="training benchmark order"
                )
            ),
            within_block_order=_text(policy.get("within_block_order"), label="within-block order"),
            undersized_population_rule=_text(
                policy.get("undersized_population"), label="undersized population rule"
            ),
            final_mix_order=_text(policy.get("final_mix_order"), label="final mix order"),
            adaptive_sampling=adaptive == "allowed",
            require_disjoint_source_ids=_boolean(
                policy.get("require_disjoint_source_ids"), label="source-ID isolation"
            ),
            require_disjoint_normalized_content=_boolean(
                policy.get("require_disjoint_normalized_content"), label="content isolation"
            ),
            final_evaluation_is_read_only=_boolean(
                policy.get("final_evaluation_is_read_only"), label="final evaluation read-only"
            ),
        ),
        terminal_contract=TerminalContract(
            reward_range=_number_pair(terminal.get("reward_range"), label="reward range"),
            posterior_success_values=_integer_pair(
                terminal.get("posterior_success_values"), label="posterior success values"
            ),
            infrastructure_failure=_text(
                terminal.get("infrastructure_failure"), label="infrastructure failure policy"
            ),
            completed_candidate_failure=_text(
                terminal.get("completed_candidate_failure"),
                label="candidate failure policy",
            ),
            private_evaluator_payload_model_visible=_boolean(
                terminal.get("private_evaluator_payload_model_visible"),
                label="private evaluator visibility",
            ),
        ),
        amendment=ProtocolAmendment(
            amendment_id=_text(amendment.get("id"), label="amendment id"),
            reason=_text(amendment.get("reason"), label="amendment reason"),
            adopted_before_first_formal_run=_boolean(
                amendment.get("adopted_before_first_formal_run"),
                label="pre-run amendment adoption",
            ),
            prior_gpt_4_1_profile_used_for_formal_run=_boolean(
                amendment.get("prior_gpt_4_1_profile_used_for_formal_run"),
                label="prior grader formal-use state",
            ),
        ),
        executable=executable,
    )


def load_active_protocol(path: Path) -> ActiveBenchmarkProtocol:
    import yaml

    return parse_active_protocol(yaml.safe_load(path.read_text(encoding="utf-8")))


__all__ = [
    "ACTIVE_BENCHMARKS_",
    "RUN_FORMAT",
    "RUN_SEED",
    "TRAINING_EPISODES_PER_BENCHMARK",
    "ActiveBenchmarkProtocol",
    "BenchmarkPopulation",
    "BenchmarkProtocol",
    "Benchmark",
    "FormalMethod",
    "PopulationRole",
    "PopulationSampling",
    "PosteriorSuccessProjection",
    "PosteriorSuccessRule",
    "ProtocolAmendment",
    "RunProtocolError",
    "RewardProjectionRule",
    "TTBRewardProjection",
    "TerminalContract",
    "TrainingMixPolicy",
    "load_active_protocol",
    "parse_active_protocol",
]
