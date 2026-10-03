from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch.distributed as dist

from r2flow.benchmarks.mbpp_scoring import MBPPScorerProfile, resolve_mbpp_profile
from r2flow.benchmarks.training_schedule import (
    load_training_sources,
)
from r2flow.benchmarks.training_sessions import (
    build_training_sessions,
    native_scorer_contracts,
)
from skillev.application import SKILLEVApplication, TerminalComponents
from skillev.application_reporting import resolved_method_state
from skillev.contracts import JsonValue, canonical_json, normalize_json
from skillev.contracts.action_wire import NATIVE_EVENT_CALL_WIRE
from skillev.evolution import PhiBudgetAuthority
from skillev.experiments._evolution_preflight_seed import planned_seed_documents
from skillev.experiments.empty_skill_slots import visible_skill_count, visible_skills_by_family
from skillev.policy import build_qwen_policy_backbone as build_qwen_policy_backbone
from skillev.runtime import BudgetLedger, LiveAttemptEventLog, SkillLibraryState
from skillev.runtime.formal_sglang_runtime import (
    BoundFormalSGLangRuntime,
    FormalSGLangRuntimeBinding,
)
from skillev.runtime.service_topology import InferenceService, ServiceTopology
from skillev.runtime.sglang_gateway import SGLangGatewayConfig
from skillev.training import FixedAttemptBudgetPlan, PrivateCheckpointStorageBinding
from skillev.training.distributed_ttb import (
    DistributedTTBGradientCoordinator,
    DistributedTTBTopology,
)
from skillev.training.distributed_ttb import (
    initialize_distributed_ttb as initialize_distributed_ttb,
)
from skillev.training.distributed_ttb import (
    serve_distributed_ttb_worker as serve_distributed_ttb_worker,
)
from skillev.training.performance_config import TrainingPerformanceConfig
from skillev.training.run_clock import RunWallClock
from skillev.training.run_condition import EffectiveRunCondition
from skillev.training.stopping import StopAfterCheckpoint, TrainingPausedError
from skillev.training.vq_monitor import (
    VqPlateauMonitor,
    mirror_directory_additive,
)
from skillev.verification.reference_agreement import reference_backend_from_config

from .bayesian_condition_transition import (
    _bind_run_directory as _bind_run_directory,
)
from .bayesian_condition_transition import (
    condition_configs,
    observer_batch_size_starts,
    observer_condition_starts,
    phase_step_cap_start,
    sampling_condition,
)
from .bayesian_serving import register_serving
from .bayesian_training_config import BayesianFormalConfig, is_current_candidate
from .bayesian_training_setup import (
    _authoring_authority,
    _clock,
    _phi_per_cycle,
    _public_identity,
    _read_preparation,
    _start_progress_monitor,
    _validated_worker_interpreter,
)
from .task_provider import (
    OrderedTaskProvider,
    TaskProviderFactory,
)
from .training_data_condition import (
    data_condition_scientific,
    inline_data_condition,
    require_same_run_data_condition,
)
from .training_domain_schedule import schedule_summary, training_schedule

__all__ = [
    "BayesianFormalConfig",
    "TrainingPerformanceConfig",
    "_read_preparation",
    "dist",
    "load_training_sources",
]


def _write(path: Path, value: object) -> None:
    path.write_text(canonical_json(normalize_json(value)) + "\n", encoding="utf-8")


def _emit(status: str, **fields: JsonValue) -> None:
    print(canonical_json({"time": _clock(), "status": status, **fields}), flush=True)


@dataclass(frozen=True, slots=True)
class FormalTrainingBindings:
    preparation: Path
    dataset: Path
    deployments: Path
    evalplus_python: Path
    evalplus_source_root: Path
    endpoint: str
    base_model: str
    adapter_namespace: str
    training_gpu_uuids: tuple[str, ...]
    serving_gpu_uuid: str
    topology: ServiceTopology | None = None
    data_condition: dict[str, JsonValue] | None = None
    evidence_mirror_root: Path | None = None
    evidence_max_pending_steps: int = 2
    evidence_minimum_free_bytes: int = 1_073_741_824

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_condition", inline_data_condition(self.data_condition))
        if self.evidence_max_pending_steps < 1 or self.evidence_minimum_free_bytes < 0:
            raise ValueError("declare positive mirror backlog and nonnegative storage headroom")

    @property
    def roles(self) -> ServiceTopology:
        if self.topology is not None:
            return self.topology
        return ServiceTopology(
            (InferenceService("shared", self.endpoint, self.serving_gpu_uuid),),
            ("shared",),
            ("shared",),
            ("shared",),
            self.training_gpu_uuids,
        )

    def require_device_mapping(self, visible: str, world_size: int) -> None:
        self.roles.require_device_mapping(visible, world_size)
        if self.topology is not None and (
            self.training_gpu_uuids != self.topology.gradient_workers
            or self.endpoint != self.topology.members("actor")[0].endpoint
            or self.serving_gpu_uuid != self.topology.members("actor")[0].gpu_uuid
        ):
            raise ValueError("legacy aliases disagree with explicit role topology")

    @classmethod
    def load(cls, path: Path) -> FormalTrainingBindings:
        value = json.loads(path.read_text(encoding="utf-8"))
        topology = ServiceTopology.from_value(value["topology"]) if "topology" in value else None
        actor = topology.members("actor")[0] if topology is not None else None
        return cls(
            topology=topology,
            data_condition=inline_data_condition(value.get("data_condition")),
            evidence_mirror_root=Path(value["evidence_mirror_root"])
            if value.get("evidence_mirror_root") is not None
            else None,
            evidence_max_pending_steps=value.get("evidence_max_pending_steps", 2),
            evidence_minimum_free_bytes=value.get("evidence_minimum_free_bytes", 1_073_741_824),
            preparation=Path(value["preparation"]),
            dataset=Path(value["dataset"]),
            deployments=Path(value["deployments"]),
            evalplus_python=Path(value["evalplus_python"]),
            evalplus_source_root=Path(value["evalplus_source_root"]),
            endpoint=value["endpoint"] if actor is None else actor.endpoint,
            base_model=value["base_model"],
            adapter_namespace=value["adapter_namespace"],
            training_gpu_uuids=tuple(value["training_gpu_uuids"])
            if topology is None
            else topology.gradient_workers,
            serving_gpu_uuid=value["serving_gpu_uuid"] if actor is None else actor.gpu_uuid,
        )

    def runtime(self, root: Path, profile: TrainingPerformanceConfig) -> FormalSGLangRuntimeBinding:
        from .formal_episode_config import formal_actor_transport

        binding = FormalSGLangRuntimeBinding(
            gateway=SGLangGatewayConfig(
                endpoint_base=self.endpoint,
                base_model=self.base_model,
                supervisor_adapter=f"{self.adapter_namespace}-supervisor",
                seed=0,
                temperature=0.0,
                top_p=1.0,
                max_output_tokens=4096,
                request_timeout_seconds=600.0,
                control_timeout_seconds=300.0,
            ),
            rollout=formal_actor_transport(self.endpoint, worker_threads=profile.transport_threads),
            workflow=profile.workflow(),
            adapter_export_root=root / "adapters",
            adapter_namespace=self.adapter_namespace,
            adapter_keep_recent=3,
            performance=profile,
            request_journal_path=root / "requests.sqlite3",
        )
        if self.topology is None:
            return binding
        return replace(
            binding,
            actor_routing_policy=self.roles.actor_routing_policy,
            actor_balanced_benchmarks=self.roles.actor_balanced_benchmarks,
            rollout=replace(
                binding.rollout,
                transport_isolation=self.roles.actor_transport_isolation,
            ),
            actor_replicas=tuple(
                replace(binding.gateway, endpoint_base=v.endpoint)
                for v in self.roles.members("actor")
            ),
            actor_benchmark_routes=tuple(
                (domain, next(v.endpoint for v in self.roles.services if v.service_id == service))
                for domain, service in self.roles.actor_benchmark_routes
            ),
            actor_benchmark_pools=tuple(
                (
                    domain,
                    tuple(
                        next(v.endpoint for v in self.roles.services if v.service_id == service)
                        for service in services
                    ),
                )
                for domain, services in self.roles.actor_benchmark_pools
            ),
        )


def require_formal_execution(profile: TrainingPerformanceConfig) -> None:
    if profile.pipeline_mode != "within-step":
        raise ValueError("formal execution requires within-step rollout overlap")


async def _stop_at_committed_boundary(
    request: StopAfterCheckpoint,
    *,
    optimizer_step: int,
    policy_snapshot_id: str,
    pause_at_step: int | None,
    vq: VqPlateauMonitor | None = None,
) -> bool:
    if request():
        return True
    if vq is not None:
        await vq.check(policy_step=optimizer_step, policy_snapshot_id=policy_snapshot_id)
    return request() or (pause_at_step is not None and optimizer_step >= pause_at_step)


def _vq_monitor(
    *,
    root: Path,
    heldout: Path,
    config: BayesianFormalConfig,
    application: Any,
    runtime: Any,
    bindings: FormalTrainingBindings,
    profile: TrainingPerformanceConfig,
    condition: Any,
    interpreter: Path,
    mbpp_profile: Any,
    coordinator: DistributedTTBGradientCoordinator,
    phase_loop: Any,
    r2flow_rollout: Any = None,
) -> VqPlateauMonitor:
    from .vq_collection import FormalVqCollector
    from .vq_heldout import load_vq_heldout_records

    assert config.r2flow is not None
    method = config.r2flow.method
    trigger = config.r2flow.evolution.trigger
    collector = FormalVqCollector(
        load_vq_heldout_records(heldout, config.domains),
        root,
        application,
        runtime,
        bindings,
        config,
        profile,
        condition,
        interpreter,
        mbpp_profile,
        r2flow_rollout,
    )
    if len(collector.heldout) != trigger.queries_per_domain * len(config.domains):
        raise ValueError("the V_q held-out set must have queries_per_domain per domain")

    async def score(artifacts: Sequence[Any]) -> Sequence[Any]:
        return await asyncio.to_thread(
            coordinator.score_flow,
            backbone=application.backbone,
            artifacts=tuple(artifacts),
            method=method,
        )

    mirror_root = bindings.evidence_mirror_root

    def mirror(source: Path) -> None:
        assert mirror_root is not None
        mirror_directory_additive(source, mirror_root / "vq")
        memo = root / "healthbench-verdict-memo"
        if memo.is_dir():
            mirror_directory_additive(memo, mirror_root / "healthbench-verdict-memo")

    evolution = config.r2flow.evolution
    return VqPlateauMonitor(
        root / "vq",
        trigger,
        expected_queries=collector.expected_queries,
        collect=collector,
        score=score,
        active_skills=lambda: visible_skill_count(
            application.library.state, config.skill_visibility
        ),
        library_version=phase_loop.segment_label,
        mirror=None if mirror_root is None else mirror,
        **_vq_phase_kwargs(
            evolution,
            phase_loop,
            library_state=lambda: application.library.state,
            skill_visibility=config.skill_visibility,
            phase_cap_from_step=lambda: phase_step_cap_start(root, config),
        ),
    )


def _vq_phase_kwargs(
    evolution: Any,
    phase_loop: Any,
    *,
    library_state: Callable[[], Any],
    skill_visibility: str | None,
    phase_cap_from_step: Callable[[], int],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "on_boundary": phase_loop.on_boundary,
        "no_call_cap_steps": evolution.max_phase_steps_no_calls,
        "visible_skills_by_family": lambda: visible_skills_by_family(
            library_state(), skill_visibility
        ),
    }
    if evolution.max_phase_steps is not None:
        kwargs.update(
            phase_cap_steps=evolution.max_phase_steps,
            phase_cap_from_step=phase_cap_from_step(),
        )
    if evolution.degenerate_entropy_phase_cap is not None:
        kwargs["degenerate_entropy_cap"] = evolution.degenerate_entropy_phase_cap
    if evolution.max_phase_steps is not None or evolution.degenerate_entropy_phase_cap is not None:
        kwargs["phase_start"] = phase_loop.phase_start
    return kwargs


def _evolve_phase_loop(
    *,
    root: Path,
    heldout: Path,
    validation_pool: Path | None = None,
    config: BayesianFormalConfig,
    application: Any,
    runtime: Any,
    bindings: FormalTrainingBindings,
    profile: TrainingPerformanceConfig,
    condition: Any,
    interpreter: Path,
    mbpp_profile: Any,
    r2flow_rollout: Any,
    set_state: Any = None,
) -> Any:
    from skillev.evolution.task_features import transferable_family
    from skillev.experiments.empty_skill_slots import PROFILE as EMPTY_SLOTS
    from skillev.experiments.empty_skill_slots import empty_slot_specs
    from skillev.training.r2flow_phase_loop import controller_for_application
    from skillev.training.vq_monitor import VQ_IMPROVEMENT_TRACE

    from .evolve_validation import FormalHeldoutValidator
    from .validation_pool import load_validation_pool_records, require_validation_pool_argument
    from .vq_heldout import load_vq_heldout_records

    assert config.r2flow is not None
    require_validation_pool_argument(config, validation_pool)
    method = config.r2flow.method
    validator = FormalHeldoutValidator(
        load_vq_heldout_records(heldout, config.domains)
        if validation_pool is None
        else load_validation_pool_records(validation_pool, config.domains),
        root,
        application,
        runtime,
        bindings,
        config,
        profile,
        condition,
        interpreter,
        mbpp_profile,
        r2flow_rollout,
    )
    _ = validator.queries
    phase_loop = controller_for_application(
        application,
        run_root=root,
        config=config.r2flow.evolution.evolution_config(),
        eta=method.temperature_beta,
        epsilon=method.epsilon_min,
        family_universe=frozenset(
            family
            for family in (transferable_family(domain) for domain in config.domains)
            if family is not None
        ),
        heldout=validator.for_boundary,
        trace_path=root / "vq" / VQ_IMPROVEMENT_TRACE,
        set_state=set_state,
        log=_emit,
    )
    phase_loop.start(
        optimizer_step=application.training_loop.optimizer_step,
        initial_specs=empty_slot_specs(config.domains)
        if config.initial_skill_profile == EMPTY_SLOTS
        else None,
    )
    return phase_loop


async def run_coordinator(
    *,
    config: BayesianFormalConfig,
    bindings: FormalTrainingBindings,
    profile: TrainingPerformanceConfig,
    root: Path,
    resume: Path | None,
    topology: DistributedTTBTopology,
    training_sources: Path,
    pause_at_step: int | None = None,
    vq_heldout: Path | None = None,
    validation_pool: Path | None = None,
) -> None:
    from .autonomous_ttb import (
        autonomous_training_sources,
        require_autonomous_initialization,
        source_coverage_report,
    )
    from .training_sources import require_training_sources, training_source_condition

    source_condition = training_source_condition(
        config, sources=training_sources, root=root, resume=resume
    )
    records = autonomous_training_sources(
        config, load_training_sources(bindings.dataset), bindings.data_condition
    )
    authorized_schedule = training_schedule(config, records, root=root, resume=resume)
    require_training_sources(
        training_sources,
        authorized_schedule,
        expected_trajectories=len(authorized_schedule),
    )
    require_formal_execution(profile)
    bindings.require_device_mapping(os.environ.get("CUDA_VISIBLE_DEVICES", ""), topology.world_size)
    if topology.rank != 0:
        raise ValueError("the training coordinator must be the participating rank zero")
    if pause_at_step is not None and (
        type(pause_at_step) is not int or not 1 <= pause_at_step < config.steps
    ):
        raise ValueError("scheduled pause must be an intermediate committed optimizer step")
    if (config.r2flow is None) != (vq_heldout is None):
        raise ValueError("an R2 Flow run requires (only it accepts) the V_q held-out set")
    from .validation_pool import require_validation_pool_argument

    require_validation_pool_argument(config, validation_pool)
    if is_current_candidate(config.format) and bindings.evidence_mirror_root is None:
        raise ValueError("fresh architecture runs require an explicit persistent evidence mirror")
    if is_current_candidate(config.format) and bindings.data_condition is None:
        raise ValueError("fresh architecture runs require explicit training sources")
    started = time.monotonic()
    _bind_run_directory(root, config, resume)
    if resume is None:
        _write(root / "training-condition.json", source_condition)
        (root / "training-sources.json").write_bytes(training_sources.read_bytes())
    if validation_pool is not None:
        from .validation_pool import bind_validation_pool

        bind_validation_pool(root, validation_pool)
    run_clock = RunWallClock(root)
    backbone_config, checkpoint = _read_preparation(bindings.preparation)
    require_autonomous_initialization(config, bindings.preparation)
    config.require_backbone(backbone_config)
    interpreter = _validated_worker_interpreter(bindings.evalplus_python)
    mbpp_profile = resolve_mbpp_profile(
        {
            "source_root": str(bindings.evalplus_source_root),
            "profile": MBPPScorerProfile().to_value(),
        }
    )
    coordinator = DistributedTTBGradientCoordinator(
        topology,
        coordinator_participates=profile.coordinator_participates,
        pipeline_mode=profile.pipeline_mode,
    )
    runtime = BoundFormalSGLangRuntime.build(
        binding=bindings.runtime(root, profile),
        gradient_preparer=coordinator,
        action_decoding=config.action_decoding,
    )
    from .training_controller import TrainingController

    evidence = None
    controller = TrainingController(
        root,
        resource_roles=bindings.roles.to_value(),
        gpu_uuids=tuple(v.gpu_uuid for v in bindings.roles.services) + bindings.training_gpu_uuids,
        evidence=evidence,
    )
    stop = thread = None
    try:
        await asyncio.to_thread(
            register_serving,
            runtime.gateway,
            bindings.roles,
            root=root,
            model_path=backbone_config.base_model_path,
            tokenizer_path=backbone_config.tokenizer_path or backbone_config.base_model_path,
            minimum_context=config.max_input_tokens
            + max(config.maximum_reasoning_tokens, config.max_action_tokens),
            event_grammar=config.action_wire == NATIVE_EVENT_CALL_WIRE,
        )
        if bindings.topology is not None:
            for service in bindings.roles.services:
                runtime.resources.configure_model_endpoint(
                    service.endpoint,
                    capacity=service.request_capacity,
                    token_capacity=service.token_capacity,
                )
        _emit(
            "preparing-domain-sessions", domains=list(config.domains), batch_size=config.batch_size
        )
        selected = training_schedule(config, records, root=root, resume=resume)
        if vq_heldout is not None:
            from .vq_heldout import load_vq_heldout_records

            if {
                (r.episode.benchmark.value, r.episode.source_id)
                for r in load_vq_heldout_records(vq_heldout)
            } & {(r.episode.benchmark.value, r.episode.source_id) for r in selected}:
                raise ValueError("V_q held-out sources must be excluded from training")
        if validation_pool is not None:
            from .validation_pool import require_pool_outside

            require_pool_outside(
                validation_pool,
                training=selected,
                vq_heldout=vq_heldout,
                data_condition=bindings.data_condition,
            )
        if config.learning_protocol is not None:
            assert bindings.data_condition is not None
            _write(
                root / "source-coverage-private.json",
                source_coverage_report(
                    bindings.data_condition,
                    getattr(config, "training_source_exclusions", ()),
                ),
            )
        tasks, sessions = await build_training_sessions(
            selected,
            deployments_path=bindings.deployments,
            request_journal_path=root / "requests.sqlite3",
            resources=runtime.resources,
            mbpp_interpreter=interpreter,
            mbpp_source_root=bindings.evalplus_source_root,
            mbpp_profile=mbpp_profile,
            rollout_budget=config.task_budget,
            static_rollout_budget=config.static_task_budget,
            domain_rollout_budgets=config.domain_task_budgets,
            lazy_environments=True,
            healthbench_judge=config.healthbench_judge,
            reference_backend=None
            if config.r2flow is None
            else reference_backend_from_config(config.r2flow.evolution, root),
            healthbench_verdict_memo_root=None
            if config.r2flow is None or "healthbench" not in config.domains
            else root / "healthbench-verdict-memo",
            healthbench_judge_recovery=getattr(config, "healthbench_judge_recovery", None),
        )
        hydration_seconds = time.monotonic() - started
        application_config = config.application_config(root.name)
        plan = config.run_plan
        curriculum = sampling_condition(root, config.condition)
        identity = _public_identity(
            application=application_config,
            run_plan=plan,
            task_ids=tuple(task.task_id for task in tasks),
            checkpoint=checkpoint,
            mbpp_profile=mbpp_profile,
            sampling_condition=curriculum,
            entry_kind="formal",
            healthbench_judge=config.healthbench_judge,
            initial_skill_profile=config.initial_skill_profile,
            domains=config.domains,
            retrieval_corpus_hash=sessions.retrieval_corpus_hash,
        )
        effective = EffectiveRunCondition.create(
            condition_id=config.condition,
            scientific={
                **data_condition_scientific(bindings.data_condition, selected),
                **source_condition,
                "formal": {
                    key: value
                    for key, value in config.to_value().items()
                    if key
                    not in {
                        "performance_profile",
                        "planning_hours",
                        "target_steps_per_hour",
                        "checkpoint_every",
                    }
                },
                "batch_size": config.batch_size,
                "maximum_h0_tokens": application_config.maximum_h0_tokens,
                "rollout": application_config.trainer.rollout.to_value(),
                "evolution": application_config.evolution.to_value(),
                "initial_partition": checkpoint.trainable_state.to_value(),
                "tokenizer_id": backbone_config.tokenizer_id,
                "base_revision": backbone_config.revision,
                "model_artifact": (
                    backbone_config.base_model_artifact.to_value()
                    if backbone_config.base_model_artifact is not None
                    else None
                ),
                "mbpp_scorer": mbpp_profile.to_value(),
            },
            execution={
                "performance": profile.to_value(),
                "roles": bindings.roles.to_value(),
                "checkpoint_every": config.checkpoint_every,
                "planning_hours": config.planning_hours,
                "target_steps_per_hour": config.target_steps_per_hour,
                **(
                    {"optimizer_transition_observation": "actual-trainable-adam@1"}
                    if is_current_candidate(config.format)
                    else {}
                ),
                **({"pause_at_optimizer_step": pause_at_step} if pause_at_step is not None else {}),
            },
        )
        if is_current_candidate(config.format):
            from .fresh_restart import resolve_effective_run_condition, resolved_input_profiles

            expanded = resolve_effective_run_condition(
                config=config,
                backbone=backbone_config,
                initial_library=SkillLibraryState.from_seed_documents(
                    planned_seed_documents(config.initial_skill_profile, config.domains)
                ),
                data_condition=cast(
                    dict[str, JsonValue],
                    data_condition_scientific(bindings.data_condition, selected)["data_condition"],
                ),
                input_profiles=resolved_input_profiles(tasks),
                scorer_contracts=native_scorer_contracts(mbpp_profile, domains=config.domains),
                execution=effective.execution,
                condition_id=config.condition,
            )
            effective = EffectiveRunCondition.create(
                condition_id=config.condition,
                scientific={
                    **expanded.scientific,
                    **source_condition,
                    "initial_partition": checkpoint.trainable_state.to_value(),
                },
                execution=expanded.execution,
            )
        if resume is not None:
            history = condition_configs(root)
            declared_data = None
            if any(value.domains != config.domains for value in history.values()):
                declared_data = {
                    value.condition: data_condition_scientific(
                        bindings.data_condition, training_schedule(value, records, root=root)
                    ).get("data_condition")
                    for value in history.values()
                }
                declared_data[config.condition] = effective.scientific.get("data_condition")
            require_same_run_data_condition(
                root, effective, declared_data_by_condition=declared_data
            )
        _write(
            root / f"effective-condition-process-{os.getpid()}.json",
            {
                **effective.to_value(),
                "source_snapshot_identity": identity.snapshot_identity.to_value(),
                "initial_adapter_base_equivalence": "requires-I0-model-comparison",
            },
        )
        phi = _phi_per_cycle(application_config, config.phi_calls_per_cycle)
        cap = FixedAttemptBudgetPlan.from_trainer_and_run_plan(
            trainer=application_config.trainer,
            run_plan=plan,
            phi_per_cycle_maximum=phi,
        ).required()
        attempt_id = "formal-seed0"
        terminal = TerminalComponents(
            ledger=BudgetLedger(run_id=root.name, attempt_id=attempt_id, cap=cap),
            authoring_authority=_authoring_authority(
                tasks,
                transferable_domains=None if config.r2flow is None else tuple(config.domains),
            ),
            phi_budget=PhiBudgetAuthority(phi),
        )
        from .bayesian_branch import checkpoint_only_evidence_start

        evidence_after = checkpoint_only_evidence_start(root, resume)
        fresh_segment = evidence_after > 0 and not (root / "events.jsonl").exists()
        event_factory = (
            LiveAttemptEventLog if resume is None or fresh_segment else LiveAttemptEventLog.resume
        )
        event_log = event_factory(root / "events.jsonl", run_id=root.name, attempt_id=attempt_id)
        storage = PrivateCheckpointStorageBinding(directory=str(root / "checkpoints"))
        _write(
            root / "resolved-run-plan.json",
            {
                "mode": "training",
                **source_condition,
                "schedule": schedule_summary(config, selected),
                "application": application_config.to_value(),
                "run_plan": plan.to_value(),
                "terminal_evaluation_conditions": json.loads(
                    sessions.terminal_evaluation_conditions_json
                ),
                "task_feature_mapping_version": sessions.task_feature_mapping_version,
                "phi_per_cycle_maximum": phi.to_value(),
                "total_attempt_budget": cap.to_value(),
                "cadence_steps": list(
                    range(config.checkpoint_every, config.steps + 1, config.checkpoint_every)
                ),
                "recovery_snapshots": "every-transaction-rolling-three-plus-cadence-phase-final",
                "performance": profile.to_value(),
                "training_gpu_uuids": list(bindings.training_gpu_uuids),
                "serving_gpu_uuid": bindings.serving_gpu_uuid,
                "role_topology": bindings.roles.to_value(),
                "planning_hours_not_hard_deadline": config.planning_hours,
            },
        )
        formal_dependencies = runtime.dependencies()
        if config.r2flow is not None:
            from dataclasses import replace as _replace

            from skillev.training.r2flow_rollout import R2FlowRolloutBinding

            formal_dependencies = _replace(
                formal_dependencies, r2flow_rollout=R2FlowRolloutBinding(root)
            )
        _emit("building-formal-application", resume=resume is not None)
        model_started = time.monotonic()
        if resume is None:
            application = SKILLEVApplication.build_formal(
                backbone_config=backbone_config,
                task_provider=OrderedTaskProvider(tasks, curriculum),
                base_session_factory=sessions,
                seed_documents=planned_seed_documents(config.initial_skill_profile, config.domains),
                terminal_components=terminal,
                checkpoint_storage=storage,
                initial_checkpoint=checkpoint,
                public_identity=identity,
                event_log=event_log,
                clock=_clock,
                runtime=formal_dependencies,
            )
            from .fresh_restart import require_clean_initial_application

            require_clean_initial_application(
                application,
                preparation=bindings.preparation,
                checkpoint_directory=Path(checkpoint.directory),
                root=root,
                initial_skill_profile=config.initial_skill_profile,
                domains=config.domains,
            )
        else:
            application = SKILLEVApplication.resume_formal(
                checkpoint_only_recovery_step=evidence_after or None,
                snapshot_directory=resume,
                backbone_config=backbone_config,
                task_provider_factory=TaskProviderFactory(tasks, curriculum),
                base_session_factory=sessions,
                terminal_components=terminal,
                checkpoint_storage=storage,
                initial_checkpoint=checkpoint,
                public_identity=identity,
                event_log=event_log,
                clock=_clock,
                runtime=formal_dependencies,
            )
        if bindings.evidence_mirror_root is not None:
            from skillev.training.run_observer import CommittedRunObserver

            evidence = CommittedRunObserver(
                root,
                run_id=root.name,
                condition_id=config.condition,
                mirror_root=bindings.evidence_mirror_root,
                condition_starts=observer_condition_starts(root, config),
                code_revision=os.environ.get("SKILLEV_CODE_REVISION"),
                expected_batch_size=config.batch_size,
                batch_size_starts=observer_batch_size_starts(root, config),
                max_pending_steps=bindings.evidence_max_pending_steps,
                minimum_free_bytes=bindings.evidence_minimum_free_bytes,
                committed_after=evidence_after,
            )
        controller.evidence = evidence
        application.training_loop.configure_inflight(
            root / "inflight",
            condition={
                "formal": config.to_value(),
                "performance": profile.to_value(),
                **source_condition,
            },
        )
        if is_current_candidate(config.format):
            application.training_loop.enable_update_observation()
        controller.attach(application.training_loop)
        initial_step = application.training_loop.optimizer_step
        process = f"{os.getpid()}-from-step-{initial_step:08d}"
        _write(
            root / f"preparation-{process}.json",
            {
                "environment_evaluator_preparation_seconds": hydration_seconds,
                "model_or_resume_seconds": time.monotonic() - model_started,
                "total_preparation_seconds": time.monotonic() - started,
                "process_id": os.getpid(),
                "initial_optimizer_step": initial_step,
                "service_instance_id": os.environ.get("SKILLEV_SERVICE_INSTANCE_ID"),
                "warmup": "first-two-steps-per-process-all-times-retained",
            },
        )
        _write(root / f"initial-method-state-{process}.json", resolved_method_state(application))
        stop, thread = _start_progress_monitor(
            application,
            total_steps=config.steps,
            performance_path=root / "performance.jsonl",
            run_started=started,
            planning_hours=config.planning_hours,
            target_steps_per_hour=config.target_steps_per_hour,
            elapsed_since_run_start=run_clock.elapsed,
            metrics_condition_id=config.condition,
            metrics_condition_starts=observer_condition_starts(root, config),
            controller_observe=controller.publish,
            corpus_search_stats=getattr(sessions, "corpus_search_stats", None),
        )
        _emit(
            "formal-training-started",
            initial_optimizer_step=initial_step,
            total_steps=config.steps,
            batch_size=config.batch_size,
        )
        request = StopAfterCheckpoint(root / "STOP_AFTER_CHECKPOINT")
        vq = None
        phase_loop = None
        if config.r2flow is not None:
            assert vq_heldout is not None
            phase_loop = _evolve_phase_loop(
                root=root,
                heldout=vq_heldout,
                validation_pool=validation_pool,
                config=config,
                application=application,
                runtime=runtime,
                bindings=bindings,
                profile=profile,
                condition=effective,
                interpreter=interpreter,
                mbpp_profile=mbpp_profile,
                r2flow_rollout=formal_dependencies.r2flow_rollout,
                set_state=controller.set_state,
            )
            vq = _vq_monitor(
                root=root,
                heldout=vq_heldout,
                config=config,
                application=application,
                runtime=runtime,
                bindings=bindings,
                profile=profile,
                condition=effective,
                interpreter=interpreter,
                mbpp_profile=mbpp_profile,
                coordinator=coordinator,
                r2flow_rollout=formal_dependencies.r2flow_rollout,
                phase_loop=phase_loop,
            )

        async def stop_at_boundary() -> bool:
            if await asyncio.to_thread(controller.checkpoint_boundary):
                (root / "STOP_AFTER_CHECKPOINT").touch()
            controller.set_state(None)
            try:
                return await _stop_at_committed_boundary(
                    request,
                    optimizer_step=application.training_loop.optimizer_step,
                    policy_snapshot_id=application.training_loop.policy_snapshot_id,
                    pause_at_step=pause_at_step,
                    vq=vq,
                )
            finally:
                controller.set_state(None)

        with request.signals():
            try:
                summary = await application.evolution_loop.run(
                    plan, stop_requested=stop_at_boundary
                )
            except TrainingPausedError as paused:
                application.training_loop.ledger.assert_fully_settled()
                evidence_status = await asyncio.to_thread(controller.finish_evidence)
                _write(
                    root / "paused.json",
                    {
                        "status": "paused-after-complete-checkpoint",
                        "optimizer_step": paused.optimizer_step,
                        "checkpoint": str(paused.checkpoint),
                        "planned_steps": config.steps,
                        "process_id": os.getpid(),
                        "evidence": evidence_status,
                    },
                )
                _emit("formal-training-paused", optimizer_step=paused.optimizer_step)
                controller.set_state("paused")
                return
        application.training_loop.ledger.assert_fully_settled()
        if summary.final_optimizer_step != config.steps:
            raise RuntimeError("formal run did not finish the declared number of complete steps")
        await asyncio.to_thread(controller.checkpoint_boundary)
        controller.set_state(None)
        _write(root / "final-method-state.json", resolved_method_state(application))
        evidence_status = await asyncio.to_thread(controller.finish_evidence)
        final_paused = evidence_status["pause_required"]
        _write(
            root / "summary.json",
            {
                "mode": "training",
                **source_condition,
                "status": "paused-after-complete-checkpoint" if final_paused else "finished",
                "summary": summary.to_value(),
                "evidence": evidence_status,
                "schedule": schedule_summary(config, selected),
                "current_process_initial_step": initial_step,
                "current_process_elapsed_seconds": time.monotonic() - started,
                "natural_evolution": "read-actual-mutations-not-assumed-from-step-count",
                "final_quality": "not-configured",
            },
        )
        _emit(
            "formal-training-paused" if final_paused else "formal-training-complete",
            optimizer_step=summary.final_optimizer_step,
        )
        controller.set_state("paused" if final_paused else "finished")
    except BaseException as exc:
        controller.set_state("failed")
        _emit("formal-training-failed-no-partial-batch-commit", error_type=type(exc).__name__)
        raise
    finally:
        if stop is not None:
            stop.set()
        if thread is not None:
            thread.join(timeout=10.0)
        try:
            if evidence is not None:
                evidence.close()
        finally:
            try:
                coordinator.close()
            finally:
                controller.close_resources()


def main() -> None:
    from .bayesian_training_cli import main as run_cli

    run_cli()


if __name__ == "__main__":
    main()
