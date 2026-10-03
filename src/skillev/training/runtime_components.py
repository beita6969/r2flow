from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import torch

from skillev.contracts import (
    FLOAT_TOLERANCE,
    JsonValue,
    ScientificSamplingCoordinate,
    TrainingStepReportValue,
)
from skillev.contracts.action_decoding import require_action_decoding_coupling
from skillev.diagnostics.rollout_progress import RolloutProgress, bind_progress
from skillev.diagnostics.rollout_trace import RolloutTraceSink
from skillev.policy import PolicyBackbone
from skillev.policy.interface import (
    ROLLOUT_PROMPT_ENCODER_VERSION,
    THINKING_ROLLOUT_PROMPT_ENCODER_VERSION,
)
from skillev.rollout import (
    CanonicalInitialContextAssembler,
    DecodingSnapshot,
    PolicySnapshot,
    RolloutArtifact,
    RolloutGenerator,
    RolloutSessionBundle,
    RolloutTask,
)
from skillev.rollout.episode_executor import episode_decoding, execute_episode
from skillev.runtime import (
    BudgetLedger,
    BudgetVector,
    OrderedTaskCursorState,
    RuntimeEventEmitter,
    RuntimeSnapshot,
    RuntimeSnapshotIdentity,
    SkillLibraryState,
)
from skillev.runtime.attempt_run_plan import RemainingAttemptCapacity

from .checkpoint import TrainingCheckpointStore
from .config import TrainerConfig, require_r2flow_coupling
from .inflight import InFlightBatchStore
from .planning import (
    CollectedTrainingBatch,
    FixedAttemptBudgetPlan,
    PlannedRollout,
    TrainingBatchPlan,
)
from .rollout_workflow import (
    ResourceTiming,
    RolloutBatchPerformanceReport,
    RolloutBatchWorkflow,
    RolloutWorkflowBinding,
    RolloutWorkflowResources,
)
from .stability import clip_full_batch_groups
from .step_math import (
    PreparedTTBStep,
    TTBStepPreparer,
    apply_optimizer_step,
    create_ttb_optimizer,
    prepare_ttb_step,
)

if TYPE_CHECKING:
    from skillev.rollout.event_grammar_runtime import EventGrammarRuntime
    from skillev.runtime.frozen_executor import FrozenSkillExecutor

    from .streaming_step import GradientStepStream


class TaskProvider(Protocol):
    def next_task(self) -> RolloutTask: ...

    @property
    def runtime_state(self) -> OrderedTaskCursorState: ...


class RolloutSessionFactory(Protocol):
    def create(self, task: RolloutTask) -> RolloutSessionBundle: ...


class SkillLibraryView(Protocol):
    @property
    def current_version(self) -> str: ...

    @property
    def state(self) -> SkillLibraryState: ...


class RolloutBatchCollector:
    def __init__(
        self,
        *,
        generator: RolloutGenerator,
        task_provider: TaskProvider,
        session_factory: RolloutSessionFactory,
        context_assembler: CanonicalInitialContextAssembler,
        library: SkillLibraryView,
        config: TrainerConfig,
        ledger: BudgetLedger,
        emitter: RuntimeEventEmitter,
        clock: Callable[[], str],
        sampling_schedule_hash: str,
        ordered_task_sequence_hash: str,
        condition_id: str,
        trace_sink_factory: Callable[[PlannedRollout], RolloutTraceSink] | None = None,
        workflow_binding: RolloutWorkflowBinding | None = None,
        workflow_resources: RolloutWorkflowResources | None = None,
        skill_executor_factory: Callable[[str], FrozenSkillExecutor] | None = None,
        event_grammar: EventGrammarRuntime | None = None,
    ) -> None:
        self._skill_executor_factory = skill_executor_factory
        self._event_grammar = event_grammar
        self._generator = generator
        self._task_provider = task_provider
        self._session_factory = session_factory
        self._context_assembler = context_assembler
        self._library = library
        self._config = config
        self._ledger = ledger
        self._emitter = emitter
        self._clock = clock
        self.inflight_store: InFlightBatchStore | None = None
        self._sampling_schedule_hash = sampling_schedule_hash
        self._ordered_task_sequence_hash = ordered_task_sequence_hash
        if not condition_id.strip():
            raise ValueError("collector condition_id must be explicit")
        self._condition_id = condition_id
        self._trace_sink_factory = trace_sink_factory
        self._live_rollouts: dict[str, RolloutProgress] = {}
        self._workflow_binding = (
            workflow_resources.binding
            if workflow_binding is None and workflow_resources is not None
            else workflow_binding or RolloutWorkflowBinding()
        )
        if workflow_resources is not None and workflow_resources.binding != self._workflow_binding:
            raise ValueError("rollout workflow resources differ from the binding")
        self._workflow_resources = (
            workflow_resources
            if workflow_resources is not None
            else RolloutWorkflowResources(self._workflow_binding)
        )
        self._workflow = RolloutBatchWorkflow[PlannedRollout, RolloutArtifact](
            self._workflow_binding
        )
        self._last_performance_report: RolloutBatchPerformanceReport | None = None
        self._decoding = DecodingSnapshot.create(
            max_reasoning_tokens=config.rollout.max_reasoning_tokens,
            max_action_tokens=config.rollout.max_action_tokens,
            base_seed=config.rollout.base_seed,
            prompt_encoder_version=(
                THINKING_ROLLOUT_PROMPT_ENCODER_VERSION
                if config.rollout.reasoning_native_thinking
                else ROLLOUT_PROMPT_ENCODER_VERSION
            ),
            action_decoding_version=config.rollout.action_decoding,
        )
        require_action_decoding_coupling(config.rollout.action_decoding, generator)
        per_rollout = config.rollout.policy_call_maximum
        calls = 2 * config.rollout.max_turns
        input_tokens_per_call = per_rollout.input_tokens // calls
        self._reasoning_call_maximum = BudgetVector(
            input_tokens=input_tokens_per_call,
            output_tokens=config.rollout.max_reasoning_tokens,
            model_calls=1,
        )
        self._action_call_maximum = BudgetVector(
            input_tokens=input_tokens_per_call,
            output_tokens=config.rollout.max_action_tokens,
            model_calls=1,
            agent_turns=1,
        )
        self._tool_call_maximum = BudgetVector(
            tool_calls=1,
            wall_time_milliseconds=(per_rollout.wall_time_milliseconds // config.rollout.max_turns),
        )

    @property
    def decoding_snapshot(self) -> DecodingSnapshot:
        return self._decoding

    @property
    def generator(self) -> RolloutGenerator:
        return self._generator

    @property
    def task_provider(self) -> TaskProvider:
        return self._task_provider

    @property
    def ledger(self) -> BudgetLedger:
        return self._ledger

    @property
    def library(self) -> SkillLibraryView:
        return self._library

    @property
    def workflow_binding(self) -> RolloutWorkflowBinding:
        return self._workflow_binding

    @property
    def workflow_resources(self) -> RolloutWorkflowResources:
        return self._workflow_resources

    @property
    def last_performance_report(self) -> RolloutBatchPerformanceReport | None:
        return self._last_performance_report

    async def collect(
        self, *, optimizer_step: int, gradient_stream: GradientStepStream | None = None
    ) -> CollectedTrainingBatch:
        self._workflow_resources.begin_batch_window()
        pinned = self._generator.snapshot()
        batch_id = f"{self._config.execution.experiment_id}-step-{optimizer_step:06d}"
        library_state = self._library.state
        library_version = library_state.current_version
        sequence_start = self._task_provider.runtime_state.cursor
        tasks = tuple(
            self._task_provider.next_task() for _ in range(self._config.execution.batch_size)
        )
        prepare_tasks = getattr(self._session_factory, "prepare_tasks", None)
        if prepare_tasks is not None:
            prepared_tasks = cast(tuple[RolloutTask, ...], await prepare_tasks(tasks))
            if tuple(task.task_id for task in prepared_tasks) != tuple(
                task.task_id for task in tasks
            ):
                raise ValueError("environment preparation changed the planned task order")
            tasks = prepared_tasks
        plan = TrainingBatchPlan(
            batch_id=batch_id,
            optimizer_step=optimizer_step,
            policy_snapshot_id=pinned.snapshot_id,
            library_version=library_version,
            rollouts=tuple(
                PlannedRollout(
                    position=position,
                    task=task,
                    trajectory_id=f"{batch_id}-t{position:03d}",
                    decoding=self._decoding_for_task(task),
                )
                for position, task in enumerate(tasks, start=1)
            ),
        )
        if gradient_stream is not None:
            await gradient_stream.begin(plan, pinned)
        entry_ids_before = {entry.reservation.reservation_id for entry in self._ledger.entries}
        saved_batch = (
            None
            if self.inflight_store is None
            else await asyncio.to_thread(self.inflight_store.begin, plan)
        )
        trajectory_seconds = [0.0] * len(plan.rollouts)

        self._live_rollouts = {
            item.trajectory_id: RolloutProgress(
                canonical_position=item.position - 1,
                benchmark_episode_index=sum(
                    previous.position < item.position
                    and previous.task.task_family.partition("/")[0]
                    == item.task.task_family.partition("/")[0]
                    for previous in plan.rollouts
                ),
                trajectory_id=item.trajectory_id,
                task_domain=item.task.task_family,
                batch_id=batch_id,
                policy_snapshot_id=pinned.snapshot_id,
                library_version=library_version,
                max_turns=self._config.rollout.max_turns
                if item.task.budget_profile is None
                else item.task.budget_profile.max_turns,
            )
            for item in plan.rollouts
        }

        async def execute(item: PlannedRollout) -> RolloutArtifact:
            row = self._live_rollouts[item.trajectory_id]
            complete = False
            with bind_progress(row):
                try:
                    restored = (
                        None
                        if saved_batch is None
                        else await asyncio.to_thread(
                            saved_batch.load,
                            item,
                            tokenizer=self._generator.tokenizer,
                            ledger=self._ledger,
                        )
                    )
                    if restored is not None:
                        row.stage("restored-complete-artifact")
                        if gradient_stream is not None:
                            await gradient_stream.accept(item.position - 1, restored)
                        result = restored
                    else:
                        result = await execute_active(item)
                        if saved_batch is not None:
                            await asyncio.to_thread(saved_batch.save, item, result, self._ledger)
                    complete = True
                    return result
                finally:
                    row.stage("artifact-ready" if complete else "failed")

        async def execute_active(item: PlannedRollout) -> RolloutArtifact:
            trajectory_started = time.perf_counter()
            trajectory_emitter = self._emitter.child(f"rollout:{batch_id}:{item.position:06d}")

            async def send_artifact(artifact: RolloutArtifact) -> None:
                assert gradient_stream is not None
                await gradient_stream.accept(item.position - 1, artifact)

            try:
                return await execute_episode(
                    generator=self._generator,
                    sessions=self._session_factory,
                    assembler=self._context_assembler,
                    rollout=self._config.rollout,
                    task=item.task,
                    trajectory_id=item.trajectory_id,
                    library=library_state,
                    coordinate=ScientificSamplingCoordinate(
                        sampling_schedule_hash=self._sampling_schedule_hash,
                        schedule_purpose="iid-training",
                        ordered_sequence_hash=self._ordered_task_sequence_hash,
                        sequence_position=sequence_start + item.position - 1,
                        task_id=item.task.task_id,
                        optimizer_step_or_anchor_ordinal=optimizer_step,
                    ),
                    decoding=item.decoding,
                    epsilon_min=self._config.method.epsilon_min,
                    condition_id=self._condition_id,
                    ledger=self._ledger,
                    emitter=trajectory_emitter,
                    clock=self._clock,
                    resources=self._workflow_resources,
                    artifact_sink=send_artifact if gradient_stream is not None else None,
                    trace_sink=None
                    if self._trace_sink_factory is None
                    else self._trace_sink_factory(item),
                    skill_executor_factory=self._skill_executor_factory,
                    event_grammar=self._event_grammar,
                )
            finally:
                trajectory_seconds[item.position - 1] = time.perf_counter() - trajectory_started

        collection_started = time.perf_counter()
        artifacts = await self._workflow.run(
            plan.rollouts,
            execute,
            declared_horizons=tuple(
                self._config.rollout.max_turns
                if item.task.budget_profile is None
                else item.task.budget_profile.max_turns
                for item in plan.rollouts
            ),
        )
        collection_seconds = time.perf_counter() - collection_started
        sealing_started = time.perf_counter()
        batch = CollectedTrainingBatch(
            batch_id=batch_id,
            optimizer_step=optimizer_step,
            policy_snapshot_id=pinned.snapshot_id,
            library_version=library_version,
            artifacts=artifacts,
        )
        self._validate(batch, plan)
        if saved_batch is not None:
            await asyncio.to_thread(
                saved_batch.require_complete, batch.artifacts, tokenizer=self._generator.tokenizer
            )
        if any(
            artifact.record.initial_context.active_skill_ids != library_state.active_skill_ids
            for artifact in artifacts
        ):
            raise ValueError("rollout invocation domain differs from the pinned library")
        sealing_seconds = time.perf_counter() - sealing_started
        actual_usage = BudgetVector()
        for entry in self._ledger.entries:
            if (
                entry.reservation.reservation_id not in entry_ids_before
                and entry.settlement is not None
            ):
                actual_usage = actual_usage.add(entry.settlement.actual)
        resource_after = self._resource_timings()
        ordered_seconds = tuple(sorted(trajectory_seconds))
        p50 = _percentile(ordered_seconds, 0.50)
        self._last_performance_report = RolloutBatchPerformanceReport(
            batch_id=batch_id,
            batch_size=len(artifacts),
            binding=self._workflow_binding,
            collection_seconds=collection_seconds,
            sealing_seconds=sealing_seconds,
            model=resource_after[0],
            environment=resource_after[1],
            terminal_evaluator=resource_after[2],
            process_grader=resource_after[3],
            session_setup=resource_after[4],
            session_cleanup=resource_after[5],
            prompt_tokens=actual_usage.input_tokens,
            completion_tokens=actual_usage.output_tokens,
            model_calls=actual_usage.model_calls,
            trajectory_p50_seconds=p50,
            trajectory_p95_seconds=_percentile(ordered_seconds, 0.95),
            trajectory_p99_seconds=_percentile(ordered_seconds, 0.99),
            maximum_trajectory_seconds=max(ordered_seconds),
            straggler_ratio=(max(ordered_seconds) / p50 if p50 > 0 else 1.0),
            reward_mean=sum(item.record.reward.value for item in artifacts) / len(artifacts),
            success_rate=(
                sum(float(item.record.reward.success) for item in artifacts) / len(artifacts)
            ),
        )
        return batch

    def _resource_timings(
        self,
    ) -> tuple[
        ResourceTiming,
        ResourceTiming,
        ResourceTiming,
        ResourceTiming,
        ResourceTiming,
        ResourceTiming,
    ]:
        resources = self._workflow_resources
        return (
            resources.model_requests.timing,
            resources.environment_calls.timing,
            resources.terminal_evaluations.timing,
            resources.process_graders.timing,
            resources.session_setups.timing,
            resources.session_cleanups.timing,
        )

    def validate_attempt_budget(
        self,
        *,
        capacity: RemainingAttemptCapacity,
        phi_per_cycle_maximum: BudgetVector,
    ) -> None:
        FixedAttemptBudgetPlan(
            batch_count=capacity.training_steps,
            batch_size=self._config.execution.batch_size,
            max_turns=self._config.rollout.max_turns,
            reasoning_call_maximum=self._reasoning_call_maximum,
            action_call_maximum=self._action_call_maximum,
            tool_call_maximum=self._tool_call_maximum,
            maximum_cycles=capacity.possible_cycles,
            phi_per_cycle_maximum=phi_per_cycle_maximum,
            executor_call_maximum=self._config.rollout.executor_call_maximum or BudgetVector(),
        ).validate_against(self._ledger.available)

    def _validate(
        self,
        batch: CollectedTrainingBatch,
        plan: TrainingBatchPlan,
    ) -> None:
        if len(batch.artifacts) != len(plan.rollouts):
            raise ValueError("training batch length differs from fixed plan")
        for artifact, planned in zip(batch.artifacts, plan.rollouts, strict=True):
            if artifact.record.trajectory_id != planned.trajectory_id:
                raise ValueError("artifact trajectory differs from planned position")
            if artifact.manifest.task_id != planned.task.task_id:
                raise ValueError("artifact task differs from planned position")
            if artifact.manifest.policy_snapshot.snapshot_id != batch.policy_snapshot_id:
                raise ValueError("training batch policy snapshot differs")
            if artifact.manifest.library_version != batch.library_version:
                raise ValueError("training batch library version differs")
            if artifact.manifest.decoding_snapshot_id != planned.decoding.snapshot_id:
                raise ValueError("training batch decoding snapshot differs")
            if not math.isclose(
                artifact.record.epsilon_min,
                self._config.method.epsilon_min,
                rel_tol=0.0,
                abs_tol=FLOAT_TOLERANCE,
            ):
                raise ValueError("training batch epsilon_min differs")

    def _decoding_for_task(self, task: RolloutTask) -> DecodingSnapshot:
        return episode_decoding(self._config.rollout, task)


class TTBOptimizerKernel:
    def __init__(
        self,
        *,
        backbone: PolicyBackbone,
        generator: RolloutGenerator,
        config: TrainerConfig,
        clock: Callable[[], str],
        checkpoint_store: TrainingCheckpointStore,
        gradient_preparer: TTBStepPreparer | None = None,
        flow_offset_record: Path | None = None,
    ) -> None:
        self._backbone = backbone
        self._generator = generator
        self._flow_offset_record = flow_offset_record
        self.flow_offset_initialization: dict[str, JsonValue] | None = None
        self._config = config
        self._clock = clock
        self._checkpoint_store = checkpoint_store
        self._gradient_preparer = gradient_preparer
        z_spec = getattr(backbone, "z_initialization_spec", None)
        flow_head = getattr(backbone, "flow_head_spec", None)
        performance = getattr(backbone, "performance_config", None)
        require_r2flow_coupling(
            method=config.method,
            flow_head=flow_head,
            microbatch_size=1
            if performance is None
            else performance.teacher_forcing.microbatch_size,
        )
        if z_spec is None or z_spec.mode != "output-bias-log-epsilon@1":
            raise ValueError("method@4 requires the log-epsilon Z initialization")
        if (
            z_spec is not None
            and z_spec.epsilon is not None
            and z_spec.epsilon != config.method.epsilon_min
        ):
            raise ValueError("Z initialization epsilon differs from the declared reward shift")
        self.last_stability_report: dict[str, JsonValue] | None = None
        self._observe_update_transition = False
        from .flow_offsets import flow_offset_domains

        enable = getattr(backbone, "enable_flow_offsets", None)
        if enable is None:
            raise ValueError("per-domain flow offsets need a backbone that supports them")
        enable(flow_offset_domains(config.rollout))
        self._optimizer, self._parameters = create_ttb_optimizer(backbone, config.optimizer)
        self._optimizer_step = 0

    @property
    def optimizer_step(self) -> int:
        return self._optimizer_step

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def backbone(self) -> PolicyBackbone:
        return self._backbone

    @property
    def config(self) -> TrainerConfig:
        return self._config

    @property
    def checkpoint_due(self) -> bool:
        return self._optimizer_step % self._config.checkpoint.every_n_steps == 0

    @property
    def gradient_preparer(self) -> TTBStepPreparer | None:
        return self._gradient_preparer

    def policy_snapshot(self) -> PolicySnapshot:
        return self._generator.snapshot()

    def create_gradient_stream(self) -> GradientStepStream | None:
        from .distributed_ttb import DistributedTTBGradientCoordinator
        from .streaming_step import GradientStepStream, LocalGradientStepStream

        preparer = self._gradient_preparer
        if self.flow_offsets_pending:
            return None
        performance = getattr(self._backbone, "performance_config", None)
        coordinator = preparer if isinstance(preparer, DistributedTTBGradientCoordinator) else None
        mode = (
            coordinator.pipeline_mode
            if coordinator is not None
            else ("sealed-batch" if performance is None else performance.pipeline_mode)
        )
        if mode != "within-step" or (preparer is not None and coordinator is None):
            return None
        if coordinator is None:
            from skillev.rollout.external_sglang import ExternalSGLangRolloutGenerator

            if not isinstance(self._generator, ExternalSGLangRolloutGenerator):
                raise ValueError("local gradient overlap requires an external rollout model owner")
        stream_type = LocalGradientStepStream if coordinator is None else GradientStepStream
        return stream_type(
            coordinator=coordinator,
            backbone=self._backbone,
            parameters=self._parameters,
            optimizer=self._optimizer,
            method=self._config.method,
            clock=self._clock,
            gradient_worker_weights=()
            if performance is None
            else performance.gradient_worker_weights,
            gradient_worker_max_sequence_tokens=()
            if performance is None
            else performance.gradient_worker_max_sequence_tokens,
            max_buffer_bytes=(
                512 * 1024 * 1024 if performance is None else performance.gradient_buffer_bytes
            ),
        )

    @property
    def flow_offsets_pending(self) -> bool:
        return getattr(self._backbone, "flow_offsets_initialization", None) is None

    def _initialize_flow_offsets(self, batch: CollectedTrainingBatch) -> None:
        from .distributed_ttb import DistributedTTBGradientCoordinator
        from .flow_offsets import initialize_flow_offsets
        from .vq_scoring import score_flow_no_grad

        if self._optimizer_step != 0 or batch.optimizer_step != 1:
            raise RuntimeError("flow offsets are initialised only at the first training step")
        method = self._config.method
        preparer = self._gradient_preparer

        def score(artifacts: tuple[object, ...]) -> list[float]:
            scores = (
                preparer.score_flow(backbone=self._backbone, artifacts=artifacts, method=method)
                if isinstance(preparer, DistributedTTBGradientCoordinator)
                else score_flow_no_grad(self._backbone, artifacts, method)
            )
            return [s.delta_0T for s in scores]

        self.flow_offset_initialization = initialize_flow_offsets(
            backbone=self._backbone,
            batch=batch,
            score=score,
            record_path=self._flow_offset_record,
        )

    def prepare(self, batch: CollectedTrainingBatch) -> PreparedTTBStep:
        snapshot_before = self._generator.snapshot()
        if snapshot_before.snapshot_id != batch.policy_snapshot_id:
            raise ValueError("live policy differs from collected batch")
        if self.flow_offsets_pending:
            self._initialize_flow_offsets(batch)
        prepare = (
            prepare_ttb_step if self._gradient_preparer is None else self._gradient_preparer.prepare
        )
        return prepare(
            backbone=self._backbone,
            optimizer=self._optimizer,
            parameters=self._parameters,
            batch=batch,
            snapshot_before=snapshot_before,
            clock=self._clock,
            method=self._config.method,
        )

    def enable_update_observation(self) -> None:
        self._observe_update_transition = True

    def apply(self, prepared: PreparedTTBStep) -> TrainingStepReportValue:
        stability = self._config.optimizer.stability
        if stability is not None:
            clipping = clip_full_batch_groups(self._parameters, stability)
            self.last_stability_report = {
                "condition": stability.to_value(),
                "stability_loss": 0.0,
                "ttb_loss": prepared.stats.batch_loss,
                "total_loss": prepared.stats.batch_loss + 0.0,
                "gradient_groups": clipping,
            }
        offset = getattr(self._backbone, "flow_offset_parameter", None)
        offset_before = None if offset is None else offset.detach().clone()
        report = apply_optimizer_step(
            optimizer=self._optimizer,
            backbone=self._backbone,
            prepared=prepared,
            clock=self._clock,
            observe_transition=self._observe_update_transition,
        )
        if offset is not None and offset_before is not None:
            from .flow_offsets import tracked_offset_update

            domains = self._backbone.flow_offset_domains
            update = tracked_offset_update(prepared, domains)
            with torch.no_grad():
                values = offset_before.clone()
                for index, domain in enumerate(domains):
                    values[index] += update.get(domain, 0.0)
                offset.copy_(values)
            self._optimizer.state.pop(offset, None)
        self._optimizer_step = prepared.batch.optimizer_step
        diagnostics = dict(self.last_stability_report or {})
        if self._observe_update_transition:
            from .ttb_reporting import ttb_update_diagnostics

            diagnostics["ttb_decomposition"] = ttb_update_diagnostics(prepared)
        if diagnostics:
            report = replace(report, optimization_diagnostics=diagnostics)
        return report

    def reset_partition(self, seed: int) -> str:
        new_version = self._backbone.reset_z(seed)
        for parameter in (*self._parameters.z_head, *self._parameters.psi_head):
            self._optimizer.state.pop(parameter, None)
            parameter.grad = None
        return new_version

    def restore_policy_optimizer_exact(
        self,
        directory: str | Path,
        *,
        expected_identity: RuntimeSnapshotIdentity,
    ) -> RuntimeSnapshot:
        if self._optimizer_step != 0:
            raise RuntimeError("restore requires a newly constructed optimizer kernel")
        restored = self._checkpoint_store.restore(
            Path(directory).resolve(),
            backbone=self._backbone,
            optimizer=self._optimizer,
            expected_experiment_id=self._config.execution.experiment_id,
            expected_identity=expected_identity,
        )
        self._optimizer_step = restored.optimizer_step
        return restored


def _percentile(ordered: tuple[float, ...], quantile: float) -> float:
    if not ordered:
        raise ValueError("trajectory latency population cannot be empty")
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


__all__ = [
    "RolloutBatchCollector",
    "RolloutSessionFactory",
    "SkillLibraryView",
    "TTBOptimizerKernel",
    "TaskProvider",
]
