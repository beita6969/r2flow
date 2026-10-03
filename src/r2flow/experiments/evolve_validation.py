from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from r2flow.benchmarks.training_records import TrainingRecord
from r2flow.benchmarks.training_sessions import (
    build_training_sessions,
    native_scorer_contracts,
    write_corpus_search_record,
)
from skillev.contracts import canonical_json, normalize_json
from skillev.r2flow_evolution.paired import RULE_CHANGED_FAMILIES_ROLLOUT
from skillev.r2flow_evolution.types import LibraryVersion, TrajectoryObs
from skillev.rollout.external_sglang import ExternalSGLangRolloutGenerator
from skillev.runtime.request_journal import DurableRequestJournal
from skillev.training.r2flow_library_versions import runtime_library_state
from skillev.training.r2flow_phase_loop import (
    VALIDATION_EDGE_RULE,
    VALIDATION_LATENCY_RULE,
    VALIDATION_TASK_PREFIX,
    VALIDATION_TOKENS_RULE,
    heldout_trajectory_obs,
    last_step_commits,
    trajectory_obs_from_value,
    trajectory_obs_to_value,
    validation_query_key,
)
from skillev.training.rollout_workflow import RolloutWorkflowResources
from skillev.verification.reference_agreement import reference_backend_from_config

from .vq_collection import HEALTHBENCH_MEMO_DIRECTORY
from .zero_update_bridge import collect_training_condition

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from r2flow.benchmarks.mbpp_scoring import MBPPScorerProfile
    from skillev.application import SKILLEVApplication
    from skillev.runtime.executor_ledger import ExecutorCallRecord
    from skillev.runtime.formal_sglang_runtime import BoundFormalSGLangRuntime
    from skillev.training.performance_config import TrainingPerformanceConfig
    from skillev.training.r2flow_rollout import R2FlowRolloutBinding
    from skillev.training.run_condition import EffectiveRunCondition

    from .bayesian_improve_training import FormalTrainingBindings
    from .bayesian_training_config import BayesianFormalConfig

VALIDATION_FORMAT: Final = "r2flow-heldout-validation-arm@1"
VALIDATION_PURPOSE: Final = "r2flow-evolve-validation@1"
VALIDATION_DIRECTORY: Final = "evolution/validation"


def validation_task_id(record: TrainingRecord, k: int) -> str:
    return (
        f"{VALIDATION_TASK_PREFIX}/{record.episode.benchmark.value}/{record.episode.source_id}/r{k}"
    )


def select_validation_queries(
    heldout: Sequence[TrainingRecord],
    *,
    domains: Sequence[str],
    per_domain: int,
    source: str = "the V_q held-out set",
) -> tuple[TrainingRecord, ...]:
    selected: list[TrainingRecord] = []
    for domain in domains:
        rows = [r for r in heldout if r.episode.benchmark.value == domain]
        if len(rows) < per_domain:
            raise ValueError(f"{domain}: {source} has fewer than {per_domain} queries")
        selected.extend(rows[:per_domain])
    return tuple(selected)


def replicate_for_validation(
    records: Sequence[TrainingRecord], m: int
) -> tuple[TrainingRecord, ...]:
    if type(m) is not int or m < 1:
        raise ValueError("validation needs at least one rollout per query")
    return tuple(
        replace(
            r,
            episode=replace(r.episode, episode_id=validation_task_id(r, k)),
            input=replace(r.input, task_id=validation_task_id(r, k)),
        )
        for r in records
        for k in range(m)
    )


def _durable_write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(canonical_json(normalize_json(value)) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@dataclass
class FormalHeldoutValidator:
    heldout: tuple[TrainingRecord, ...]
    run_root: Path
    application: SKILLEVApplication
    runtime: BoundFormalSGLangRuntime
    bindings: FormalTrainingBindings
    config: BayesianFormalConfig
    profile: TrainingPerformanceConfig
    condition: EffectiveRunCondition
    mbpp_interpreter: Path
    mbpp_profile: MBPPScorerProfile
    r2flow_rollout: R2FlowRolloutBinding | None = None

    @property
    def queries(self) -> tuple[TrainingRecord, ...]:
        assert self.config.r2flow is not None
        evolution = self.config.r2flow.evolution
        return select_validation_queries(
            self.heldout,
            domains=self.config.domains,
            per_domain=evolution.validation_queries_per_domain,
            source="the dedicated validation pool",
        )

    def for_boundary(
        self, *, snapshot: Any, phase: int, optimizer_step: int
    ) -> Callable[[LibraryVersion, int], Awaitable[Sequence[TrajectoryObs]]]:
        base = self.application.library.state
        root = self.run_root / VALIDATION_DIRECTORY / f"phase-{phase:04d}"
        snap = snapshot.snapshot_id.removeprefix("sha256:")[:16]

        async def rollout_heldout(
            library: LibraryVersion, seed: int, *, domains: frozenset[str] | None = None
        ) -> list[TrajectoryObs]:
            if domains is not None and (not domains or not domains <= set(self.config.domains)):
                raise ValueError(f"rolled-out domains {sorted(domains)} are not run domains")
            state = runtime_library_state(library, base)
            digest = state.current_version.removeprefix("sha256:")[:16]
            name = f"v{library.version:04d}-{digest}-seed-{seed}"
            if domains is not None:
                name += "-domains-" + "+".join(sorted(domains))
            directory = root / f"pi-{snap}" / name
            result = directory / "result.json"
            if result.is_file():
                value = json.loads(result.read_text(encoding="utf-8"))
                return [trajectory_obs_from_value(item) for item in value["trajectories"]]
            if directory.exists():
                ordinal = 1
                while directory.with_name(f"{directory.name}-interrupted-{ordinal}").exists():
                    ordinal += 1
                directory.rename(directory.with_name(f"{directory.name}-interrupted-{ordinal}"))
            directory.mkdir(parents=True, mode=0o700)
            calls: list[ExecutorCallRecord] = []
            artifacts = await self._collect(
                directory / "collection",
                library_state=state,
                snapshot=snapshot,
                seed=seed,
                optimizer_step=optimizer_step,
                observer=calls.append,
                **({} if domains is None else {"domains": domains}),
            )
            assert self.config.r2flow is not None
            method = self.config.r2flow.method
            by_trajectory: dict[str, list[ExecutorCallRecord]] = {}
            for call in calls:
                by_trajectory.setdefault(call.trajectory_id, []).append(call)
            namespace = f"p{phase:04d}-v{library.version:04d}-{digest[:8]}-s{seed}-"
            commits = last_step_commits(directory / "collection" / "events.jsonl")
            observations = sorted(
                (
                    heldout_trajectory_obs(
                        artifact,
                        eta=method.temperature_beta,
                        epsilon=method.epsilon_min,
                        executor_calls=by_trajectory.get(artifact.record.trajectory_id, ()),
                        namespace=namespace,
                        agent_step_commit=commits.get(artifact.record.trajectory_id),
                    )
                    for artifact in artifacts
                ),
                key=lambda obs: (obs.query_id, obs.trajectory_id),
            )
            arm: dict[str, Any] = {
                "format": VALIDATION_FORMAT,
                "phase": phase,
                "optimizer_step": optimizer_step,
                "policy_snapshot_id": snapshot.snapshot_id,
                "library_version": library.version,
                "library_digest": state.current_version,
                "seed": seed,
                "rules": [VALIDATION_TOKENS_RULE, VALIDATION_LATENCY_RULE, VALIDATION_EDGE_RULE],
                "trajectories": [trajectory_obs_to_value(obs) for obs in observations],
            }
            if domains is not None:
                arm["rollout"] = {
                    "rule": RULE_CHANGED_FAMILIES_ROLLOUT,
                    "domains": sorted(domains),
                    "rolled_out": sorted({obs.query_id for obs in observations}),
                }
            _durable_write(result, arm)
            return observations

        return rollout_heldout

    async def _collect(
        self,
        root: Path,
        *,
        library_state: Any,
        snapshot: Any,
        seed: int,
        optimizer_step: int,
        observer: Callable[[ExecutorCallRecord], None],
        domains: frozenset[str] | None = None,
    ) -> tuple[Any, ...]:
        from r2flow.benchmarks.evaluation_episode import (
            EvaluationEpisodeRecord,
            EvaluationSource,
            EvaluationTarget,
        )

        config = self.config
        assert config.r2flow is not None
        if self.r2flow_rollout is None:
            raise ValueError("held-out validation needs the run's R2FlowRolloutBinding")
        evolution = config.r2flow.evolution
        resources = RolloutWorkflowResources(self.profile.workflow())
        if self.bindings.topology is not None:
            for service in self.bindings.roles.services:
                resources.configure_model_endpoint(
                    service.endpoint,
                    capacity=service.request_capacity,
                    token_capacity=service.token_capacity,
                )
        journal = root.parent / (root.name + "-requests.sqlite3")
        replicas = replicate_for_validation(self.queries, evolution.validation_rollouts_per_query)
        full_positions = {r.input.task_id: position for position, r in enumerate(replicas)}
        if domains is not None:
            replicas = tuple(r for r in replicas if r.episode.benchmark.value in domains)
            if not replicas:
                raise ValueError(f"no held-out validation query in the domains {sorted(domains)}")
        records = tuple(
            EvaluationEpisodeRecord(
                EvaluationSource(
                    r.episode.benchmark,
                    r.episode.population_id,
                    r.episode.source_id,
                    r.input.task_id,
                ),
                r.input,
                EvaluationTarget(r.output.target),
            )
            for r in replicas
        )
        tasks, sessions = await build_training_sessions(
            records,
            deployments_path=self.bindings.deployments,
            request_journal_path=journal,
            resources=resources,
            mbpp_interpreter=self.mbpp_interpreter,
            mbpp_source_root=self.bindings.evalplus_source_root,
            mbpp_profile=self.mbpp_profile,
            rollout_budget=config.task_budget,
            static_rollout_budget=config.static_task_budget,
            domain_rollout_budgets=config.domain_task_budgets,
            lazy_environments=True,
            healthbench_judge=config.healthbench_judge,
            reference_backend=None
            if config.r2flow is None
            else reference_backend_from_config(config.r2flow.evolution, self.run_root),
            healthbench_verdict_memo_root=self.run_root / HEALTHBENCH_MEMO_DIRECTORY
            if "healthbench" in config.domains
            else None,
            healthbench_judge_recovery=getattr(config, "healthbench_judge_recovery", None),
        )
        generator = ExternalSGLangRolloutGenerator(
            config=self.runtime.binding.rollout,
            tokenizer=self.application.backbone.tokenizer,
            gateway=self.runtime.gateway,
            snapshot_provider=lambda: snapshot,
            request_journal=DurableRequestJournal(journal),
            action_decoding=config.action_decoding,
        )
        app_config = config.application_config("evolve-validation-collect-only")
        schedule = f"{config.r2flow.heldout_split}/{VALIDATION_PURPOSE}/seed-{seed}"
        positions = (
            None if domains is None else tuple(full_positions[task.task_id] for task in tasks)
        )
        subpanel: dict[str, Any] = {} if positions is None else {"sequence_positions": positions}
        controls = normalize_json(
            {
                "execution_machine": "skillev.rollout.episode_executor.execute_episode@1",
                "purpose": VALIDATION_PURPOSE,
                "rollout": config.sampling_config.to_value(),
                "maximum_h0_tokens": app_config.maximum_h0_tokens,
                "native_scorers": native_scorer_contracts(
                    self.mbpp_profile, domains=config.domains
                ),
                "public_tasks": [task.to_value() for task in tasks],
                "model_tokenizer": self.application.backbone.tokenizer.tokenizer_id,
                "rollouts_per_query": evolution.validation_rollouts_per_query,
                "seed": seed,
                **(
                    {}
                    if positions is None
                    else {
                        "rollout_rule": RULE_CHANGED_FAMILIES_ROLLOUT,
                        "domains": sorted(domains or ()),
                        "sequence_positions": list(positions),
                    }
                ),
            }
        )
        assert isinstance(controls, dict)
        try:
            result = await collect_training_condition(
                root=root,
                condition=self.condition,
                tasks=tasks,
                generator=generator,
                base_sessions=sessions,
                library_state=library_state,
                trainer=app_config.trainer,
                maximum_h0_tokens=app_config.maximum_h0_tokens,
                workflow=resources.binding,
                workflow_resources=resources,
                sampling_schedule_id=schedule,
                ordered_task_sequence_id=schedule + "/replicas",
                sampled_policy_step=optimizer_step,
                execution_controls=controls,
                r2flow_rollout=self.r2flow_rollout,
                schedule_purpose=VALIDATION_PURPOSE,
                executor_record_observer=observer,
                **subpanel,
            )
        finally:
            generator.close()
            write_corpus_search_record(
                root.parent / (root.name + "-corpus-search.json"),
                sessions,
                purpose=VALIDATION_PURPOSE,
                optimizer_step=optimizer_step,
                seed=seed,
            )
        artifacts = result.diagnostic_artifacts
        planned = {task.task_id for task in tasks}
        if {a.manifest.task_id for a in artifacts} != planned:
            raise RuntimeError("held-out validation arm is incomplete (infrastructure failure)")
        for artifact in artifacts:
            validation_query_key(artifact.manifest.task_id)
        return artifacts


__all__ = [
    "VALIDATION_DIRECTORY",
    "VALIDATION_FORMAT",
    "VALIDATION_PURPOSE",
    "FormalHeldoutValidator",
    "replicate_for_validation",
    "select_validation_queries",
    "validation_task_id",
]
