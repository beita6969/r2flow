from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import cast

from skillev.benchmarks import (
    ALFWorldPublicItem,
    BenchmarkPublicItem,
    CompletionBenchmarkEnvironment,
)
from skillev.benchmarks.alfworld import ALFWorldEnvironment
from skillev.benchmarks.passage_corpus import PassageCorpus, PassageCorpusEnvironment
from skillev.benchmarks.r2flow_tool_set import (
    R2FLOW_TOOL_SET,
    r2flow_action_contract,
    r2flow_hotpot_task,
    r2flow_trivia_task,
    retrieval_corpus_hash,
)
from skillev.benchmarks.wikipedia_search import WikipediaSearchBackend, WikipediaSearchEnvironment
from skillev.contracts import JsonValue, SuccessRule, TerminalReward, canonical_json, normalize_json
from skillev.contracts.observed_reset import (
    ALFWORLD_RESET_KIND,
    RESET_BINDING_FORMAT,
    STATIC_RESET_KIND,
)
from skillev.evaluation.external_judge_policy import HEALTHBENCH_JUDGE_PROFILE
from skillev.evaluation.owner_final import parse_explicit_integer_payload, project_owner_final
from skillev.evaluation.terminal_projection import TerminalMode
from skillev.evaluation.training_domains.catalog import TrainingBenchmark
from skillev.rollout import (
    NoTerminalSubmission,
    RolloutBudgetProfile,
    RolloutTask,
    TerminalEvaluationRequest,
    TerminalEvaluator,
    TerminalEvaluatorError,
    UnskilledRolloutSessionBundle,
)
from skillev.rollout.errors import EpisodeInfrastructureError
from skillev.runtime import EnvironmentObservation, RolloutEnvironmentSession, StructuredAction
from skillev.training import AsyncResourceLimiter, RolloutWorkflowResources
from skillev.verification import ReferenceAnswerBackend, VerifierSuite

from .alfworld import PrivateALFWorldCase, PrivateALFWorldTerminalEvaluator
from .alfworld_official import (
    OfficialALFWorldEpisodeFactory,
    OfficialALFWorldTask,
    bind_reset_public_item,
)
from .alfworld_public_goal import GOAL_BINDING
from .evaluation_episode import EvaluationEpisodeRecord
from .healthbench_memo import HealthBenchVerdictMemo
from .mbpp_scoring import (
    MBPPScorerProfile,
    decode_mbpp_verdict,
    mbpp_failure_kind,
    mbpp_request,
    resolve_mbpp_profile,
)
from .native_backends import HealthBenchOfficialGrader
from .official_process import (
    ALFWorldGameDeployment,
    OfficialALFWorldProcessFactory,
    PinnedOfficialProcess,
)
from .private_workers import PrivateJSONWorker
from .qa_diagnostics import qa_answer_diagnostics
from .qa_metrics import (
    best_alias_metrics,
    normalize_triviaqa_answer,
    score_hotpotqa_answers,
)
from .session_deployments import TrainingDeployments
from .static import parse_aime_answer
from .terminal_inputs import submitted_value
from .training_records import TrainingRecord

NativeEpisodeRecord = TrainingRecord | EvaluationEpisodeRecord


_STATIC_VERIFIER = "r2flow-training-static@3"


def native_scorer_contracts(
    mbpp_profile: MBPPScorerProfile, *, domains: Iterable[str]
) -> dict[str, JsonValue]:
    from skillev.evaluation.healthbench_judge_profile import healthbench_condition

    from .alfworld import ALFWORLD_VERIFIER

    catalog: dict[str, JsonValue] = {
        "hotpotqa": {
            "verifier": _STATIC_VERIFIER,
            "metric": "answer-f1",
            "success": "answer-exact-match",
            "projection": TerminalMode.SHORT_ANSWER.value,
        },
        "triviaqa": {
            "verifier": _STATIC_VERIFIER,
            "metric": "answer-f1",
            "success": "answer-exact-match",
            "projection": TerminalMode.SHORT_ANSWER.value,
        },
        "aime-2026": {
            "verifier": _STATIC_VERIFIER,
            "metric": "accuracy",
            "success": "exact-integer",
            "projection": TerminalMode.AIME_INTEGER.value,
        },
        "alfworld": {
            "verifier": ALFWORLD_VERIFIER,
            "metric": "alfworld-success",
            "success": "official-environment-terminal",
        },
        "mbpp-plus": mbpp_profile.to_value(),
    }
    domains = tuple(domains)
    if "healthbench" in domains:
        catalog["healthbench"] = healthbench_condition()
    return {domain: catalog[domain] for domain in domains}


def _target(record: NativeEpisodeRecord) -> dict[str, JsonValue]:
    return record.output.target


def _answer(request: TerminalEvaluationRequest) -> str | None:
    if isinstance(request.evaluation_input, NoTerminalSubmission):
        return None
    value = normalize_json(submitted_value(request))
    if not isinstance(value, dict) or set(value) != {"answer"}:
        raise TerminalEvaluatorError("completion has incompatible fields")
    answer = value["answer"]
    if type(answer) is not str or not answer.strip():
        raise TerminalEvaluatorError("completion answer is empty")
    return answer


def _projected_answer(request: TerminalEvaluationRequest, mode: TerminalMode) -> str | None:
    answer = _answer(request)
    if answer is None:
        return None
    return project_owner_final(mode, answer)


def _reward(
    *,
    task: RolloutTask,
    value: float,
    success: bool,
    metric: str,
    verifier: str,
    fields: dict[str, JsonValue],
) -> TerminalReward:
    return TerminalReward(
        value=value,
        success=success,
        success_rule=SuccessRule.TRUSTED_NATIVE_PROJECTION,
        success_threshold=None,
        native_metric_name=metric,
        native_payload={
            "benchmark_id": cast(dict[str, JsonValue], task.public_context)["benchmark_id"],
            **fields,
        },
        environment_id=task.environment_id,
        verifier_version=verifier,
    )


@dataclass(slots=True)
class _ExactCompletionEnvironment:
    task: RolloutTask
    delegate: RolloutEnvironmentSession

    @property
    def environment_id(self) -> str:
        return self.task.environment_id

    @property
    def task_family(self) -> str:
        return self.task.task_family

    async def execute(self, action: StructuredAction, *, step_index: int) -> EnvironmentObservation:
        return await self.delegate.execute(action, step_index=step_index)

    def validate_completion(self, submission: JsonValue) -> bool:
        return self.delegate.validate_completion(submission)


def _completion_environment(task: RolloutTask) -> _ExactCompletionEnvironment:
    context = cast(dict[str, JsonValue], task.public_context)
    public = BenchmarkPublicItem(
        benchmark_id=cast(str, context["benchmark_id"]),
        dataset_revision=cast(str, context["dataset_revision"]),
        split=cast(str, context["split"]),
        task_id=task.task_id,
        task_family=task.task_family,
        query=task.query,
        public_context=context.get("payload"),
    )
    return _ExactCompletionEnvironment(task, CompletionBenchmarkEnvironment(public))


@dataclass(frozen=True, slots=True)
class _StaticEvaluator:
    record: NativeEpisodeRecord
    task: RolloutTask

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        if request.task_id != self.task.task_id:
            raise TerminalEvaluatorError("static request reached another task")
        benchmark = self.record.episode.benchmark
        answer = _projected_answer(
            request,
            TerminalMode.AIME_INTEGER
            if benchmark is TrainingBenchmark.AIME_2026
            else TerminalMode.SHORT_ANSWER,
        )
        accepted = _target(self.record).get("accepted_answers")
        if not isinstance(accepted, list) or any(type(item) is not str for item in accepted):
            raise TerminalEvaluatorError("static private answers are incompatible")
        aliases = tuple(cast(list[str], accepted))
        if answer is None:
            value = exact = 0.0
        elif benchmark is TrainingBenchmark.HOTPOT_QA:
            result = score_hotpotqa_answers(answer, aliases)
            value, exact = result.f1, result.em
        elif benchmark is TrainingBenchmark.TRIVIA_QA:
            result = best_alias_metrics(answer, aliases, normalize=normalize_triviaqa_answer)
            value, exact = result.f1, result.em
        elif benchmark is TrainingBenchmark.AIME_2026:
            parsed = parse_explicit_integer_payload(answer)
            expected = {parse_aime_answer(item) for item in aliases}
            value = exact = float(parsed is not None and str(parsed) in expected)
        else:
            raise TerminalEvaluatorError("static evaluator received another benchmark")
        return _reward(
            task=self.task,
            value=value,
            success=bool(exact),
            metric="answer-f1" if benchmark is not TrainingBenchmark.AIME_2026 else "accuracy",
            verifier=_STATIC_VERIFIER,
            fields={
                "answer-exact-match": exact,
                "qa_diagnostics": None
                if benchmark is TrainingBenchmark.AIME_2026
                else qa_answer_diagnostics(
                    benchmark.value,
                    original_submission=_answer(request),
                    projected_answer=answer,
                    accepted_aliases=aliases,
                ),
                "public_metrics": (
                    {"accuracy": value}
                    if benchmark is TrainingBenchmark.AIME_2026
                    else {"answer-exact-match": exact, "answer-f1": value}
                ),
            },
        )


class HealthBenchJudgeInfrastructureError(TerminalEvaluatorError, EpisodeInfrastructureError):
    pass


@dataclass(frozen=True, slots=True)
class _HealthEvaluator:
    task: RolloutTask
    grader: HealthBenchOfficialGrader
    judge_failure_as_infrastructure: bool = False

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        if request.task_id != self.task.task_id:
            raise TerminalEvaluatorError("HealthBench request reached another task")
        answer = _answer(request)
        if answer is None:
            raw, negative = 0.0, 0
        else:
            try:
                grade = await self.grader.grade(self.task.task_id, answer)
            except Exception as error:
                failure = (
                    HealthBenchJudgeInfrastructureError
                    if self.judge_failure_as_infrastructure
                    else TerminalEvaluatorError
                )
                raise failure("HealthBench judge infrastructure failed") from error
            raw = float(grade.official_rubric_score)
            negative = grade.triggered_negative_rubric_count
        ledger = None if answer is None else grade.criterion_ledger
        memo_reference = ledger.get("verdict_memo") if isinstance(ledger, dict) else None
        clipped = min(1.0, max(0.0, raw))
        from skillev.evaluation.healthbench_judge_profile import METRIC, healthbench_condition

        return _reward(
            task=self.task,
            value=clipped,
            success=raw >= 0.60 and negative == 0,
            metric=METRIC,
            verifier=self.grader.verifier_version,
            fields={
                "triggered-negative-rubric-count": negative,
                "native_raw_score": None if answer is None else raw,
                "learning_reward": clipped,
                "binary_success": raw >= 0.60 and negative == 0,
                "negative_criterion_count": None if answer is None else negative,
                "criterion_ledger": ledger,
                **({"verdict_memo": memo_reference} if memo_reference is not None else {}),
                **(
                    {"refused_criterion_count": grade.refused_criterion_count}
                    if answer is not None and grade.refused_criterion_count is not None
                    else {}
                ),
                "public_metrics": {METRIC: raw},
                "grader_profile": healthbench_condition(),
                "grader_cost": normalize_json(grade.grader_cost) if answer is not None else None,
            },
        )


@dataclass(frozen=True, slots=True)
class _MBPPPlusEvaluator:
    record: NativeEpisodeRecord
    task: RolloutTask
    worker: PrivateJSONWorker
    scorer_profile: MBPPScorerProfile

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        if request.task_id != self.task.task_id:
            raise TerminalEvaluatorError("MBPP+ request reached another task")
        answer = _projected_answer(request, TerminalMode.PYTHON_SOURCE)
        passed = False
        base_passed = False
        plus_passed = False
        scorer_attempts = 0
        result = None
        if answer is not None:
            payload = mbpp_request(
                profile=self.scorer_profile,
                private_target=_target(self.record),
                prompt=self.task.query,
                source_task_id=self.record.episode.source_id,
                submission=answer,
                task_id=self.task.task_id,
            )
            infrastructure: tuple[str, str, str | None] | None = None
            for scorer_attempts in range(1, 3):
                try:
                    candidate_result = await self.worker.request(payload)
                except Exception as error:
                    if scorer_attempts == 1:
                        await asyncio.sleep(1.0)
                        continue
                    raise TerminalEvaluatorError(
                        "EvalPlus worker failed after two attempts"
                    ) from error
                if candidate_result.get("infrastructure_error"):
                    stage = str(candidate_result.get("error_stage"))
                    error_type = str(candidate_result.get("error_type"))
                    module = candidate_result.get("error_module")
                    infrastructure = stage, error_type, str(module) if module else None
                    if scorer_attempts == 1:
                        await asyncio.sleep(1.0)
                    continue
                try:
                    result = decode_mbpp_verdict(candidate_result, self.scorer_profile)
                except RuntimeError as error:
                    raise TerminalEvaluatorError("EvalPlus worker response differs") from error
                break
            if result is None:
                assert infrastructure is not None
                stage, error_type, error_module = infrastructure
                raise TerminalEvaluatorError(
                    f"EvalPlus worker infrastructure failure after two attempts at {stage}: "
                    f"{error_type} ({error_module})"
                )
            base_passed = result["base_passed"] is True
            plus_passed = result["plus_passed"] is True
            passed = base_passed and plus_passed
        return _reward(
            task=self.task,
            value=float(passed),
            success=passed,
            metric="base-plus-pass@1",
            verifier=self.scorer_profile.profile_id,
            fields={
                "base-passed": base_passed,
                "plus-passed": plus_passed,
                "scorer-attempts": scorer_attempts,
                "scorer_profile": self.scorer_profile.to_value(),
                "native_verdict": result,
                "failure_kind": None if result is None else mbpp_failure_kind(result),
                "public_metrics": {
                    "base-pass": float(base_passed),
                    "plus-pass": float(plus_passed),
                    "base-plus-pass@1": float(passed),
                },
            },
        )


@dataclass(frozen=True, slots=True)
class _SourceBoundEvaluator:
    delegate: TerminalEvaluator
    record: NativeEpisodeRecord
    public_task: RolloutTask | None = None
    _reset_binding_json: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        binding = self._reset_binding()
        if binding is not None:
            object.__setattr__(self, "_reset_binding_json", canonical_json(binding))

    def _reset_binding(self) -> dict[str, JsonValue] | None:
        if self.public_task is None:
            return None
        observed: JsonValue = None
        if self.record.episode.benchmark is TrainingBenchmark.ALF_WORLD:
            if not isinstance(self.delegate, PrivateALFWorldTerminalEvaluator):
                return None
            captured = self.delegate.observed_reset_json
            if captured is None:
                return None
            observed = normalize_json(json.loads(captured))
            kind = ALFWORLD_RESET_KIND
        else:
            kind = STATIC_RESET_KIND
        public = self.public_task.to_value()
        public.pop("task_id")
        return {
            "format": RESET_BINDING_FORMAT,
            "kind": kind,
            "public_task": public,
            "observed_reset": observed,
        }

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        reward = await self.delegate.evaluate(request)
        coordinates: dict[str, JsonValue] = {
            "benchmark_id": self.record.episode.benchmark.value,
            "population_id": self.record.episode.population_id,
            "source_question_id": self.record.episode.source_id,
            "occurrence_id": self.record.episode.episode_id,
        }
        if self._reset_binding_json is not None:
            coordinates["reset_binding"] = normalize_json(json.loads(self._reset_binding_json))
        return replace(
            reward,
            native_payload={
                **reward.native_payload,
                (
                    "evaluation_evidence_source"
                    if isinstance(self.record, EvaluationEpisodeRecord)
                    else "training_evidence_source"
                ): coordinates,
            },
        )


@dataclass(frozen=True, slots=True)
class _Route:
    task: RolloutTask
    create: Callable[[], UnskilledRolloutSessionBundle]


@dataclass(frozen=True, slots=True)
class TrainingSessionFactory:
    routes: dict[str, _Route]
    mbpp_profile: MBPPScorerProfile
    prepare_pending: (
        Callable[[tuple[RolloutTask, ...]], Awaitable[tuple[RolloutTask, ...]]] | None
    ) = None
    healthbench_judge: str | None = None
    retrieval_corpus_hash: str | None = None
    wikipedia_search: WikipediaSearchBackend | None = None

    def corpus_search_stats(self) -> dict[str, JsonValue] | None:
        stats = getattr(self.wikipedia_search, "stats", None)
        value = stats() if callable(stats) else None
        return {"corpus": "wikipedia", **value} if isinstance(value, dict) else None

    async def prepare_tasks(self, tasks: tuple[RolloutTask, ...]) -> tuple[RolloutTask, ...]:
        return tasks if self.prepare_pending is None else await self.prepare_pending(tasks)

    @property
    def task_feature_mapping_version(self) -> str:
        from skillev.evolution.task_features import TASK_FEATURE_MAPPING_VERSION

        return TASK_FEATURE_MAPPING_VERSION

    @property
    def terminal_evaluation_conditions_json(self) -> str:
        from skillev.evaluation.healthbench_judge_profile import healthbench_condition

        conditions: dict[str, JsonValue] = {"mbpp-plus": self.mbpp_profile.to_value()}
        if self.healthbench_judge is not None:
            conditions["healthbench"] = healthbench_condition()
        conditions["alfworld_goal_binding"] = GOAL_BINDING
        conditions["tool_set"] = R2FLOW_TOOL_SET
        conditions["retrieval_corpus_hash"] = self.retrieval_corpus_hash
        return canonical_json(conditions)

    def create(self, task: RolloutTask) -> UnskilledRolloutSessionBundle:
        route = self.routes.get(task.task_id)
        if route is None or route.task != task or not callable(route.create):
            raise ValueError("Protocol 13 task has no exact trusted session route")
        bundle = route.create()
        if not isinstance(bundle, UnskilledRolloutSessionBundle):
            raise TypeError("Protocol 13 session route returned an incompatible bundle")
        return bundle


async def _alfworld_route(
    record: NativeEpisodeRecord,
    deployments: TrainingDeployments,
) -> tuple[RolloutTask, Callable[[], UnskilledRolloutSessionBundle]]:
    target = _target(record)
    raw_route = target.get("environment_route")
    if not isinstance(raw_route, dict):
        raise ValueError("Protocol 13 ALFWorld route is absent")
    raw_game_file = raw_route.get("game_file")
    raw_config_file = raw_route.get("config_file")
    if type(raw_game_file) is not str or type(raw_config_file) is not str:
        raise ValueError("Protocol 13 ALFWorld path route is incompatible")
    source_game_file = Path(raw_game_file)
    try:
        marker = source_game_file.parts.index("json_2.1.1")
    except ValueError as error:
        raise ValueError("Protocol 13 ALFWorld game has no canonical dataset route") from error
    relative_game_file = Path(*source_game_file.parts[marker + 1 :])
    dataset_root = (deployments.alfworld.dataset_root / "json_2.1.1").resolve()
    game_file = (dataset_root / relative_game_file).resolve()
    if not game_file.is_relative_to(dataset_root) or not game_file.is_file():
        raise ValueError("Protocol 13 ALFWorld game is absent from the active deployment")
    config_file = deployments.alfworld.config_path
    seed = cast(int, raw_route["seed"])
    max_steps = cast(int, raw_route["max_steps"])
    mode = cast(str, raw_route["mode"])
    source = record.input
    context = cast(dict[str, JsonValue], source.public_context)
    revision = cast(str, context["dataset_revision"])
    snapshot = record.episode.population_id
    public = ALFWorldPublicItem(
        dataset_revision=revision,
        environment_snapshot_id=snapshot,
        split=cast(str, context["split"]),
        task_id=source.task_id,
        task_family=source.task_family,
        query=source.query,
        public_context={"admissible_commands": ["look"], "initial_observation": "pending"},
        seed=seed,
        max_steps=max_steps,
    )
    official_task = OfficialALFWorldTask(
        task_id=source.task_id,
        environment_id=public.environment_id,
        game_id=record.episode.source_id,
        seed=seed,
        max_steps=max_steps,
        payload={"game_file": str(game_file)},
    )
    deployment = deployments.alfworld
    process_factory = OfficialALFWorldProcessFactory(
        PinnedOfficialProcess(
            deployment.interpreter,
            deployment.source_root,
            deployment.source_revision,
            deployment.timeout_seconds,
        ),
        config_file,
        {record.episode.source_id: ALFWorldGameDeployment(game_file.parent, mode)},
        seed,
        simulator_max_steps=max_steps,
    )
    env = await asyncio.to_thread(process_factory.create, official_task)
    try:
        reset = await asyncio.to_thread(env.reset, seed)
    finally:
        await env.close()
    public = bind_reset_public_item(public, reset)
    surface, profile = r2flow_action_contract("alfworld", max_steps=max_steps)
    task = replace(public.to_rollout_task(), action_surface=surface, budget_profile=profile)
    case = PrivateALFWorldCase(public, official_task)
    episode_factory = OfficialALFWorldEpisodeFactory(process_factory)

    def create() -> UnskilledRolloutSessionBundle:
        session = episode_factory.create(case)
        return UnskilledRolloutSessionBundle(
            ALFWorldEnvironment(public, session.episode),
            PrivateALFWorldTerminalEvaluator(
                public, session.outcome_view, session.observed_reset_json
            ),
            session.cleanup,
        )

    return task, create


def r2flow_verifier_suites(
    resources: RolloutWorkflowResources,
    *,
    reference_backend: ReferenceAnswerBackend | None = None,
) -> dict[str, VerifierSuite]:
    from skillev.verification.reference_agreement import REFERENCE_DOMAINS
    from skillev.verification.suite import SUITE_DOMAINS

    from .mbpp_public_asserts import IsolatedPublicAssertBackend

    served = frozenset(getattr(reference_backend, "domains", REFERENCE_DOMAINS))
    return {
        domain: VerifierSuite(
            domain=domain,
            code_backend=IsolatedPublicAssertBackend(limiter=resources.process_graders)
            if domain == "mbpp-plus"
            else None,
            reference_backend=reference_backend
            if domain in REFERENCE_DOMAINS and domain in served
            else None,
        )
        for domain in sorted(SUITE_DOMAINS)
    }


async def build_training_sessions(
    records: tuple[NativeEpisodeRecord, ...],
    *,
    deployments_path: Path,
    resources: RolloutWorkflowResources,
    mbpp_interpreter: Path,
    mbpp_source_root: Path,
    mbpp_profile: MBPPScorerProfile,
    rollout_budget: RolloutBudgetProfile | None = None,
    static_rollout_budget: RolloutBudgetProfile | None = None,
    domain_rollout_budgets: Mapping[str, RolloutBudgetProfile] | None = None,
    lazy_environments: bool = False,
    request_journal_path: Path | None = None,
    healthbench_judge: str | None = None,
    healthbench_verdict_memo_root: Path | None = None,
    reference_backend: ReferenceAnswerBackend | None = None,
    wikipedia_backend: WikipediaSearchBackend | None = None,
    healthbench_judge_recovery: str | None = None,
    healthbench_rescore_collected: bool = False,
    healthbench_grader_override: HealthBenchOfficialGrader | None = None,
    healthbench_judge_failure_as_infrastructure: bool = False,
    healthbench_grader_wrapper: Callable[[HealthBenchOfficialGrader], HealthBenchOfficialGrader]
    | None = None,
) -> tuple[tuple[RolloutTask, ...], TrainingSessionFactory]:
    if not records or len({record.input.task_id for record in records}) != len(records):
        raise ValueError("Protocol 13 mini records must be non-empty and task-unique")
    resolved = resolve_mbpp_profile(
        {"source_root": str(mbpp_source_root), "profile": mbpp_profile.to_value()}
    )
    if resolved != mbpp_profile or not mbpp_source_root.is_absolute():
        raise ValueError("training MBPP source differs from the frozen scoring condition")
    deployments = TrainingDeployments.read(deployments_path)
    wikipedia: WikipediaSearchBackend | None = None
    wikipedia_identity: dict[str, JsonValue] | None = None
    wikipedia_pin = deployments.triviaqa_wikipedia
    if wikipedia_pin is None:
        if any(r.episode.benchmark is TrainingBenchmark.TRIVIA_QA for r in records):
            raise ValueError(
                f"{R2FLOW_TOOL_SET} TriviaQA requires the pinned Wikipedia corpus "
                "(deployments triviaqa_wikipedia)"
            )
    else:
        from .triviaqa_wikipedia_search import WikipediaSearchBackend as SqliteWikipedia

        wikipedia_pin.verify()
        wikipedia_identity = wikipedia_pin.identity()
        wikipedia = (
            SqliteWikipedia(wikipedia_pin) if wikipedia_backend is None else wikipedia_backend
        )
    health_cases = {
        record.input.task_id: record.output.target
        for record in records
        if record.episode.benchmark is TrainingBenchmark.HEALTHBENCH
    }
    health_config = deployments.healthbench

    if healthbench_judge not in (None, HEALTHBENCH_JUDGE_PROFILE):
        raise ValueError("unknown HealthBench judge profile")
    from skillev.evaluation.healthbench_judge_recovery import judge_recovery

    from .healthbench_api import OpenAIHealthBenchGrader, RescoringHealthBenchGrader

    recovery = judge_recovery(healthbench_judge_recovery)
    if healthbench_rescore_collected and recovery is None:
        raise ValueError("V_q re-scoring belongs to healthbench-judge-recovery@3")

    def _judge_grader() -> HealthBenchOfficialGrader:
        if recovery is not None and request_journal_path is None:
            raise ValueError("healthbench-judge-recovery@3 needs a durable request journal")
        grader = OpenAIHealthBenchGrader(
            health_cases,
            health_config.source_root,
            AsyncResourceLimiter(4),
            verdict_memo=None
            if healthbench_verdict_memo_root is None
            else HealthBenchVerdictMemo(healthbench_verdict_memo_root),
            criterion_ledger_root=None
            if request_journal_path is None
            else request_journal_path.with_name(
                request_journal_path.stem + "-healthbench-criteria"
            ),
            judge_recovery=None if recovery is None else recovery.rule,
        )
        if healthbench_rescore_collected and recovery is not None:
            return RescoringHealthBenchGrader(grader, recovery.vq_rescore_delays_seconds)
        return grader

    def _health_graders() -> tuple[HealthBenchOfficialGrader, ...]:
        graders = (
            (healthbench_grader_override,)
            if healthbench_grader_override is not None
            else (_judge_grader(),)
        )
        if healthbench_grader_wrapper is None:
            return graders
        return tuple(healthbench_grader_wrapper(grader) for grader in graders)

    health_graders: tuple[HealthBenchOfficialGrader, ...]
    if healthbench_judge is None:
        if health_cases:
            raise ValueError("HealthBench cases require a declared judge")
        health_graders = ()
    else:
        health_graders = _health_graders()
    health_routes = {
        task_id: health_graders[index % len(health_graders)]
        for index, task_id in enumerate(health_cases)
    }
    mbpp_worker = PrivateJSONWorker(
        command=(
            str(mbpp_interpreter),
            str(Path(__file__).with_name("mbpp_worker.py")),
            "--official-source-root",
            str(mbpp_source_root),
        ),
        working_directory=deployments_path.parent,
        timeout_seconds=mbpp_profile.outer_timeout_seconds,
        process_limiter=resources.process_graders,
    )
    verifiers = r2flow_verifier_suites(resources, reference_backend=reference_backend)

    async def hydrate(record: NativeEpisodeRecord) -> _Route:
        benchmark = record.episode.benchmark
        from skillev.evolution.task_features import public_task_features

        features = public_task_features(benchmark.value, task_family=record.input.task_family)
        record = replace(
            record,
            input=replace(
                record.input, task_family=features.task_family, context_id=features.context_id
            ),
        )
        corpus: PassageCorpus | None = None
        if benchmark is TrainingBenchmark.ALF_WORLD:
            async with resources.process_graders.lease():
                task, create = await _alfworld_route(record, deployments)
        else:
            if benchmark is TrainingBenchmark.HOTPOT_QA:
                task, corpus = r2flow_hotpot_task(record.input)
                corpora[record.episode.source_id] = corpus
            elif benchmark is TrainingBenchmark.TRIVIA_QA:
                task = r2flow_trivia_task(record.input)
            else:
                surface, profile = r2flow_action_contract(benchmark.value)
                task = replace(record.input, action_surface=surface, budget_profile=profile)
            evaluator: TerminalEvaluator
            if benchmark in {
                TrainingBenchmark.HOTPOT_QA,
                TrainingBenchmark.TRIVIA_QA,
                TrainingBenchmark.AIME_2026,
            }:
                evaluator = _StaticEvaluator(record, task)
            elif benchmark is TrainingBenchmark.HEALTHBENCH:
                evaluator = _HealthEvaluator(
                    task,
                    health_routes[task.task_id],
                    judge_failure_as_infrastructure=healthbench_judge_failure_as_infrastructure,
                )
            elif benchmark is TrainingBenchmark.MBPP_PLUS:
                evaluator = _MBPPPlusEvaluator(record, task, mbpp_worker, mbpp_profile)
            else:
                raise ValueError("unsupported Protocol 13 training benchmark")

            wiki = wikipedia if benchmark is TrainingBenchmark.TRIVIA_QA else None
            if benchmark is TrainingBenchmark.TRIVIA_QA:
                assert wiki is not None

            def create(
                task: RolloutTask = task,
                evaluator: TerminalEvaluator = evaluator,
                corpus: PassageCorpus | None = corpus,
                wiki: WikipediaSearchBackend | None = wiki,
            ) -> UnskilledRolloutSessionBundle:
                environment: RolloutEnvironmentSession = _completion_environment(task)
                if corpus is not None:
                    environment = PassageCorpusEnvironment(environment, corpus)
                if wiki is not None:
                    environment = WikipediaSearchEnvironment(environment, wiki)
                return UnskilledRolloutSessionBundle(environment, evaluator)

        def source_bound_create(
            create: Callable[[], UnskilledRolloutSessionBundle] = create,
            record: NativeEpisodeRecord = record,
        ) -> UnskilledRolloutSessionBundle:
            bundle = create()
            return replace(
                bundle,
                evaluator=_SourceBoundEvaluator(bundle.evaluator, record, task),
                verifier=verifiers.get(record.episode.benchmark.value),
            )

        override = (
            static_rollout_budget
            if benchmark is not TrainingBenchmark.ALF_WORLD and static_rollout_budget is not None
            else rollout_budget
        )
        override = (domain_rollout_budgets or {}).get(benchmark.value, override)
        task = replace(
            task,
            context_id=features.context_id,
            budget_profile=override if override is not None else task.budget_profile,
        )
        return _Route(task, source_bound_create)

    from .session_hydration import hydrate_ordered

    pending: dict[str, NativeEpisodeRecord] = {}
    corpora: dict[str, PassageCorpus] = {}

    async def describe(record: NativeEpisodeRecord) -> _Route:
        if not lazy_environments or record.episode.benchmark is not TrainingBenchmark.ALF_WORLD:
            return await hydrate(record)
        from skillev.evolution.task_features import public_task_features

        features = public_task_features(
            record.episode.benchmark.value, task_family=record.input.task_family
        )
        raw = _target(record)["environment_route"]
        assert isinstance(raw, dict)
        surface, profile = r2flow_action_contract("alfworld", max_steps=cast(int, raw["max_steps"]))
        task = replace(
            record.input,
            task_family=features.task_family,
            context_id=features.context_id,
            action_surface=surface,
            budget_profile=(domain_rollout_budgets or {}).get(
                "alfworld", rollout_budget or profile
            ),
        )
        pending[task.task_id] = record

        def not_prepared() -> UnskilledRolloutSessionBundle:
            raise RuntimeError("environment description must be hydrated before rollout")

        return _Route(task, not_prepared)

    hydrated = await hydrate_ordered(records, describe, limiter=resources.session_setups)
    routes = {route.task.task_id: route for route in hydrated}
    tasks = tuple(route.task for route in hydrated)

    async def prepare(selected: tuple[RolloutTask, ...]) -> tuple[RolloutTask, ...]:
        wanted = tuple(pending[task.task_id] for task in selected if task.task_id in pending)
        ready = await hydrate_ordered(wanted, hydrate, limiter=resources.session_setups)
        for route in ready:
            routes[route.task.task_id] = route
            del pending[route.task.task_id]
        return tuple(routes[task.task_id].task for task in selected)

    return tasks, TrainingSessionFactory(
        routes,
        mbpp_profile,
        prepare if lazy_environments else None,
        healthbench_judge,
        retrieval_corpus_hash(corpora, wikipedia=wikipedia_identity),
        wikipedia,
    )


CORPUS_SEARCH_RECORD_FORMAT = "r2flow-corpus-search-counts@1"


def write_corpus_search_record(
    path: Path, sessions: object, **context: JsonValue
) -> dict[str, JsonValue] | None:
    stats = getattr(sessions, "corpus_search_stats", None)
    value = stats() if callable(stats) else None
    if value is None:
        return None
    record: dict[str, JsonValue] = {
        "format": CORPUS_SEARCH_RECORD_FORMAT,
        **context,
        "corpus_search": dict(value),
    }
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return record


__all__ = [
    "CORPUS_SEARCH_RECORD_FORMAT",
    "TrainingSessionFactory",
    "build_training_sessions",
    "r2flow_verifier_suites",
    "write_corpus_search_record",
]
