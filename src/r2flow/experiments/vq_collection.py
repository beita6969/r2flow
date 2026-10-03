from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from r2flow.benchmarks.mbpp_scoring import MBPPScorerProfile
from r2flow.benchmarks.training_records import TrainingRecord
from r2flow.benchmarks.training_sessions import (
    build_training_sessions,
    native_scorer_contracts,
    write_corpus_search_record,
)
from skillev.contracts import normalize_json
from skillev.rollout.external_sglang import ExternalSGLangRolloutGenerator
from skillev.runtime.request_journal import DurableRequestJournal
from skillev.training.rollout_workflow import RolloutWorkflowResources
from skillev.training.run_condition import EffectiveRunCondition
from skillev.verification.reference_agreement import reference_backend_from_config

from .vq_heldout import replicate_for_vq
from .zero_update_bridge import collect_training_condition

if TYPE_CHECKING:
    from skillev.application import SKILLEVApplication
    from skillev.rollout import RolloutArtifact
    from skillev.runtime.formal_sglang_runtime import BoundFormalSGLangRuntime
    from skillev.training.performance_config import TrainingPerformanceConfig
    from skillev.training.r2flow_rollout import R2FlowRolloutBinding

    from .bayesian_improve_training import FormalTrainingBindings
    from .bayesian_training_config import BayesianFormalConfig

HEALTHBENCH_MEMO_DIRECTORY = "healthbench-verdict-memo"


@dataclass
class FormalVqCollector:
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
    def rollouts_per_query(self) -> int:
        assert self.config.r2flow is not None
        return self.config.r2flow.evolution.trigger.rollouts_per_query

    @property
    def expected_queries(self) -> frozenset[str]:
        return frozenset(f"{r.episode.benchmark.value}:{r.episode.source_id}" for r in self.heldout)

    async def __call__(
        self, root: Path, step: int, snapshot_id: str
    ) -> tuple[RolloutArtifact, ...]:
        application = self.application
        snapshot = application.generator.snapshot()
        if snapshot.snapshot_id != snapshot_id:
            raise ValueError("V_q collection must use the committed forward policy")
        if self.config.r2flow is not None and self.r2flow_rollout is None:
            raise ValueError("R2 Flow V_q collection needs the run's R2FlowRolloutBinding")
        resources = RolloutWorkflowResources(self.profile.workflow())
        if self.bindings.topology is not None:
            for service in self.bindings.roles.services:
                resources.configure_model_endpoint(
                    service.endpoint,
                    capacity=service.request_capacity,
                    token_capacity=service.token_capacity,
                )
        journal = root.parent / (root.name + "-requests.sqlite3")
        from r2flow.benchmarks.evaluation_episode import (
            EvaluationEpisodeRecord,
            EvaluationSource,
            EvaluationTarget,
        )

        replicas = replicate_for_vq(self.heldout, self.rollouts_per_query)
        diagnostic_records = tuple(
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
        recovery = getattr(self.config, "healthbench_judge_recovery", None)
        tasks, sessions = await build_training_sessions(
            diagnostic_records,
            deployments_path=self.bindings.deployments,
            request_journal_path=journal,
            resources=resources,
            mbpp_interpreter=self.mbpp_interpreter,
            mbpp_source_root=self.bindings.evalplus_source_root,
            mbpp_profile=self.mbpp_profile,
            rollout_budget=self.config.task_budget,
            static_rollout_budget=self.config.static_task_budget,
            domain_rollout_budgets=self.config.domain_task_budgets,
            lazy_environments=True,
            healthbench_judge=self.config.healthbench_judge,
            reference_backend=None
            if self.config.r2flow is None
            else reference_backend_from_config(self.config.r2flow.evolution, self.run_root),
            healthbench_verdict_memo_root=self.run_root / HEALTHBENCH_MEMO_DIRECTORY
            if "healthbench" in self.config.domains
            else None,
            healthbench_judge_recovery=recovery,
            healthbench_rescore_collected=recovery is not None,
        )
        generator = ExternalSGLangRolloutGenerator(
            config=self.runtime.binding.rollout,
            tokenizer=application.backbone.tokenizer,
            gateway=self.runtime.gateway,
            snapshot_provider=lambda: snapshot,
            request_journal=DurableRequestJournal(journal),
            action_decoding=self.config.action_decoding,
        )
        app_config = self.config.application_config("vq-collect-only")
        controls = normalize_json(
            {
                "execution_machine": "skillev.rollout.episode_executor.execute_episode@1",
                "purpose": "r2flow-vq-heldout@1",
                "rollout": self.config.sampling_config.to_value(),
                "maximum_h0_tokens": app_config.maximum_h0_tokens,
                "native_scorers": native_scorer_contracts(
                    self.mbpp_profile, domains=self.config.domains
                ),
                "public_tasks": [task.to_value() for task in tasks],
                "model_tokenizer": application.backbone.tokenizer.tokenizer_id,
                "rollouts_per_query": self.rollouts_per_query,
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
                library_state=application.library.state,
                trainer=app_config.trainer,
                maximum_h0_tokens=app_config.maximum_h0_tokens,
                workflow=resources.binding,
                workflow_resources=resources,
                sampling_schedule_id=self.config.r2flow.heldout_split,
                ordered_task_sequence_id=self.config.r2flow.heldout_split + "/replicas",
                sampled_policy_step=step,
                execution_controls=controls,
                r2flow_rollout=self.r2flow_rollout,
            )
        finally:
            generator.close()
            write_corpus_search_record(
                root.parent / (root.name + "-corpus-search.json"),
                sessions,
                purpose="r2flow-vq-heldout@1",
                optimizer_step=step,
            )
        return result.diagnostic_artifacts


__all__ = ["HEALTHBENCH_MEMO_DIRECTORY", "FormalVqCollector"]
