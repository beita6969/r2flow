from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from skillev.application_config import ApplicationConfig
from skillev.application_snapshot import (
    ApplicationSnapshotFactory,
    require_full_execution_coherence,
)
from skillev.contracts import LibraryInitialized
from skillev.evolution import (
    BaseRolloutSessionFactory,
    EvolutionLoop,
    LibrarySegmentDetector,
    PhiBudgetAuthority,
    RetrievingRolloutSessionFactory,
    SkillAuthoringAuthority,
    TaskConditionedSkillRetriever,
)
from skillev.policy import (
    PolicyBackbone,
    PrivateInitialCheckpointBinding,
    QwenDeploymentConfig,
)
from skillev.policy.hf_backbone import build_qwen_policy_backbone
from skillev.rollout import (
    LocalPolicyGenerator,
    RolloutGenerator,
)
from skillev.runtime import (
    AttemptRunCursorState,
    AttemptRunProgress,
    BudgetLedger,
    EventType,
    FullRuntimeExecutionState,
    LiveAttemptEventLog,
    OrderedTaskCursorState,
    RuntimeEventEmitter,
    RuntimeSnapshotIdentity,
    RuntimeSnapshotStore,
    SkillDocument,
    SkillLibrary,
    SkillLibraryState,
    StepAdapterPublisher,
    StepTransactionJournal,
    require_seed_library,
)
from skillev.runtime.attempt_run_plan import ExactAttemptRunPlan
from skillev.training import (
    FilesystemTrainingCheckpointStore,
    MethodProjectionPipeline,
    PosteriorEventProvenance,
    PrivateCheckpointStorageBinding,
    RolloutWorkflowBinding,
    RolloutWorkflowResources,
    TaskProvider,
    TrainingLoop,
    TrainingProjectionPipeline,
    TTBStepPreparer,
)
from skillev.training.flow_offsets import FLOW_OFFSET_RECORD_FILE
from skillev.training.r2flow_library_versions import PhaseCarrier
from skillev.training.r2flow_rollout import R2FlowRolloutBinding


class ApplicationProjectionPipeline(TrainingProjectionPipeline, Protocol):
    @property
    def posterior_provenance(self) -> PosteriorEventProvenance: ...


class TaskProviderFactory(Protocol):
    def from_exact_state(self, state: OrderedTaskCursorState) -> TaskProvider: ...


class RolloutGeneratorFactory(Protocol):
    def __call__(
        self,
        backbone: PolicyBackbone,
        resources: RolloutWorkflowResources,
    ) -> RolloutGenerator: ...


@dataclass(frozen=True, slots=True)
class FormalSharedRuntimeDependencies:
    rollout_generator_factory: RolloutGeneratorFactory
    workflow_resources: RolloutWorkflowResources
    step_adapter_publisher_factory: Callable[[PolicyBackbone], StepAdapterPublisher]

    def __post_init__(self) -> None:
        if not callable(self.rollout_generator_factory):
            raise TypeError("formal shared runtime requires a rollout generator factory")
        if not isinstance(self.workflow_resources, RolloutWorkflowResources):
            raise TypeError("formal shared runtime requires workflow resources")
        if not callable(self.step_adapter_publisher_factory):
            raise TypeError("formal shared runtime requires an adapter publisher factory")


@dataclass(frozen=True, slots=True)
class FormalRuntimeDependencies:
    rollout_generator_factory: RolloutGeneratorFactory
    gradient_preparer: TTBStepPreparer
    workflow_resources: RolloutWorkflowResources
    step_adapter_publisher_factory: Callable[[PolicyBackbone], StepAdapterPublisher]
    r2flow_rollout: R2FlowRolloutBinding | None = None

    @property
    def shared(self) -> FormalSharedRuntimeDependencies:
        return FormalSharedRuntimeDependencies(
            rollout_generator_factory=self.rollout_generator_factory,
            workflow_resources=self.workflow_resources,
            step_adapter_publisher_factory=self.step_adapter_publisher_factory,
        )

    def __post_init__(self) -> None:
        from skillev.training.distributed_ttb import DistributedTTBGradientCoordinator

        if not callable(self.rollout_generator_factory):
            raise TypeError("formal runtime requires a rollout generator factory")
        if not isinstance(self.gradient_preparer, DistributedTTBGradientCoordinator):
            raise TypeError("formal runtime requires the distributed TTB coordinator")
        if not isinstance(self.workflow_resources, RolloutWorkflowResources):
            raise TypeError("formal runtime requires shared workflow resources")
        if not callable(self.step_adapter_publisher_factory):
            raise TypeError("formal runtime requires an adapter publisher factory")

    def require_bound(self, application: SKILLEVApplication) -> None:
        from skillev.rollout.external_sglang import ExternalSGLangRolloutGenerator
        from skillev.runtime.sglang_step_publisher import SGLangStepAdapterPublisher

        if not isinstance(application.generator, ExternalSGLangRolloutGenerator):
            raise TypeError("formal application did not bind external exact-token SGLang")
        if application.training_loop.gradient_preparer is not self.gradient_preparer:
            raise TypeError("formal application did not bind the distributed TTB coordinator")
        if application.training_loop.workflow_resources is not self.workflow_resources:
            raise TypeError("formal application did not bind the shared workflow resources")
        if not isinstance(
            application.evolution_loop.step_adapter_publisher,
            SGLangStepAdapterPublisher,
        ):
            raise TypeError("formal application did not bind the SGLang adapter publisher")
        if application.evolution_loop.step_transaction_journal is None:
            raise TypeError("formal application did not bind the step transaction journal")


class ApplicationPublicIdentity(Protocol):
    @property
    def application_config(self) -> ApplicationConfig: ...

    @property
    def run_plan(self) -> ExactAttemptRunPlan: ...

    @property
    def initial_run_cursor(self) -> AttemptRunCursorState: ...

    @property
    def method_identity_hash(self) -> str: ...

    @property
    def sampling_schedule_hash(self) -> str: ...

    @property
    def ordered_task_sequence_hash(self) -> str: ...

    @property
    def phase_checkpoint_cycle_ordinals(self) -> tuple[int, ...]: ...

    def runtime_snapshot_identity(self) -> RuntimeSnapshotIdentity: ...


@dataclass(frozen=True, slots=True)
class TerminalComponents:
    ledger: BudgetLedger
    authoring_authority: SkillAuthoringAuthority
    phi_budget: PhiBudgetAuthority


@dataclass(slots=True)
class SKILLEVApplication:
    backbone: PolicyBackbone
    generator: RolloutGenerator
    library: SkillLibrary
    retriever: TaskConditionedSkillRetriever
    projections: ApplicationProjectionPipeline
    training_loop: TrainingLoop
    detector: LibrarySegmentDetector
    evolution_loop: EvolutionLoop
    snapshot_store: RuntimeSnapshotStore
    emitter: RuntimeEventEmitter
    public_identity: ApplicationPublicIdentity
    run_progress: AttemptRunProgress
    snapshot_identity: RuntimeSnapshotIdentity
    phase_carrier: PhaseCarrier = field(default_factory=PhaseCarrier)

    @property
    def final_training_snapshot_directory(self) -> Path:
        return self.evolution_loop.final_training_snapshot_directory

    @classmethod
    def build(
        cls,
        *,
        backbone_config: QwenDeploymentConfig,
        task_provider: TaskProvider,
        base_session_factory: BaseRolloutSessionFactory,
        seed_documents: tuple[SkillDocument, ...],
        terminal_components: TerminalComponents,
        checkpoint_storage: PrivateCheckpointStorageBinding,
        initial_checkpoint: PrivateInitialCheckpointBinding,
        public_identity: ApplicationPublicIdentity,
        event_log: LiveAttemptEventLog,
        clock: Callable[[], str],
        generator: RolloutGenerator | None = None,
        generator_factory: RolloutGeneratorFactory | None = None,
        gradient_preparer: TTBStepPreparer | None = None,
        workflow_binding: RolloutWorkflowBinding | None = None,
        workflow_resources: RolloutWorkflowResources | None = None,
        step_adapter_publisher_factory: Callable[[PolicyBackbone], StepAdapterPublisher]
        | None = None,
        step_transaction_journal: StepTransactionJournal | None = None,
        r2flow_rollout: R2FlowRolloutBinding | None = None,
    ) -> SKILLEVApplication:
        require_seed_library(seed_documents)
        config = _require_application_config(public_identity)
        run_plan = _require_run_plan(public_identity)
        backbone = build_qwen_policy_backbone(backbone_config)
        backbone.load_checkpoint(initial_checkpoint.directory)
        backbone.bind_initial_trainable_state(initial_checkpoint.trainable_state)
        library = SkillLibrary(SkillLibraryState.from_seed_documents(seed_documents))
        projections = MethodProjectionPipeline.fresh(
            diagnostics_config=config.diagnostics,
            library_version=library.current_version,
        )
        detector = LibrarySegmentDetector.fresh(library_version=library.current_version)
        return cls.build_from_components(
            backbone=backbone,
            task_provider=task_provider,
            base_session_factory=base_session_factory,
            terminal_components=terminal_components,
            checkpoint_storage=checkpoint_storage,
            event_log=event_log,
            clock=clock,
            library=library,
            projections=projections,
            detector=detector,
            public_identity=public_identity,
            run_progress=AttemptRunProgress.from_state(
                run_plan,
                _require_initial_cursor(public_identity, run_plan),
            ),
            generator=generator,
            generator_factory=generator_factory,
            gradient_preparer=gradient_preparer,
            workflow_binding=workflow_binding,
            workflow_resources=workflow_resources,
            step_adapter_publisher_factory=step_adapter_publisher_factory,
            step_transaction_journal=step_transaction_journal,
            r2flow_rollout=r2flow_rollout,
        )

    @classmethod
    def build_formal(
        cls,
        *,
        backbone_config: QwenDeploymentConfig,
        task_provider: TaskProvider,
        base_session_factory: BaseRolloutSessionFactory,
        seed_documents: tuple[SkillDocument, ...],
        terminal_components: TerminalComponents,
        checkpoint_storage: PrivateCheckpointStorageBinding,
        initial_checkpoint: PrivateInitialCheckpointBinding,
        public_identity: ApplicationPublicIdentity,
        event_log: LiveAttemptEventLog,
        clock: Callable[[], str],
        runtime: FormalRuntimeDependencies,
    ) -> SKILLEVApplication:
        application = cls.build(
            backbone_config=backbone_config,
            task_provider=task_provider,
            base_session_factory=base_session_factory,
            seed_documents=seed_documents,
            terminal_components=terminal_components,
            checkpoint_storage=checkpoint_storage,
            initial_checkpoint=initial_checkpoint,
            public_identity=public_identity,
            event_log=event_log,
            clock=clock,
            generator_factory=runtime.rollout_generator_factory,
            gradient_preparer=runtime.gradient_preparer,
            workflow_binding=runtime.workflow_resources.binding,
            workflow_resources=runtime.workflow_resources,
            step_adapter_publisher_factory=runtime.step_adapter_publisher_factory,
            r2flow_rollout=getattr(runtime, "r2flow_rollout", None),
            step_transaction_journal=StepTransactionJournal(
                (Path(checkpoint_storage.directory) / "step-transactions").resolve()
            ),
        )
        runtime.require_bound(application)
        return application

    @classmethod
    def resume(
        cls,
        *,
        snapshot_directory: Path,
        backbone_config: QwenDeploymentConfig,
        task_provider_factory: TaskProviderFactory,
        base_session_factory: BaseRolloutSessionFactory,
        terminal_components: TerminalComponents,
        checkpoint_storage: PrivateCheckpointStorageBinding,
        initial_checkpoint: PrivateInitialCheckpointBinding,
        public_identity: ApplicationPublicIdentity,
        event_log: LiveAttemptEventLog,
        clock: Callable[[], str],
        generator: RolloutGenerator | None = None,
        generator_factory: RolloutGeneratorFactory | None = None,
        gradient_preparer: TTBStepPreparer | None = None,
        workflow_binding: RolloutWorkflowBinding | None = None,
        workflow_resources: RolloutWorkflowResources | None = None,
        step_adapter_publisher_factory: Callable[[PolicyBackbone], StepAdapterPublisher]
        | None = None,
        step_transaction_journal: StepTransactionJournal | None = None,
        r2flow_rollout: R2FlowRolloutBinding | None = None,
        checkpoint_only_recovery_step: int | None = None,
    ) -> SKILLEVApplication:
        config = _require_application_config(public_identity)
        run_plan = _require_run_plan(public_identity)
        restore_identity = public_identity.runtime_snapshot_identity()

        artifact_store = FilesystemTrainingCheckpointStore(root=Path(checkpoint_storage.directory))
        metadata = artifact_store.load_metadata(snapshot_directory.resolve())
        if metadata.identity != restore_identity:
            raise ValueError("snapshot identity differs from public attempt identity")
        state = metadata.execution_state
        if not isinstance(state, FullRuntimeExecutionState):
            raise TypeError("full application requires a full runtime snapshot")
        require_full_execution_coherence(state, metadata.optimizer_step)
        backbone = build_qwen_policy_backbone(backbone_config)
        backbone.load_checkpoint(initial_checkpoint.directory)
        backbone.bind_initial_trainable_state(initial_checkpoint.trainable_state)
        task_provider = task_provider_factory.from_exact_state(state.task_cursor)
        library = SkillLibrary(state.library)
        projections = MethodProjectionPipeline.from_runtime_state(
            diagnostics_config=config.diagnostics,
            state=state.projections,
        )
        detector = LibrarySegmentDetector.from_runtime_state(
            expected_library_version=library.current_version,
            state=state.detector,
        )
        application = cls._assemble(
            backbone=backbone,
            task_provider=task_provider,
            base_session_factory=base_session_factory,
            terminal_components=terminal_components,
            checkpoint_storage=checkpoint_storage,
            event_log=event_log,
            clock=clock,
            library=library,
            projections=projections,
            detector=detector,
            public_identity=public_identity,
            run_progress=AttemptRunProgress.from_state(run_plan, state.run_cursor),
            generator=generator,
            generator_factory=generator_factory,
            gradient_preparer=gradient_preparer,
            workflow_binding=workflow_binding,
            workflow_resources=workflow_resources,
            step_adapter_publisher_factory=step_adapter_publisher_factory,
            step_transaction_journal=step_transaction_journal,
            r2flow_rollout=r2flow_rollout,
        )
        application.training_loop.restore_policy_optimizer_exact(
            snapshot_directory,
            expected_identity=restore_identity,
        )
        application.phase_carrier.value = state.r2flow_phase
        from skillev.application_recovery import reconcile_application_step

        journal = application.evolution_loop.step_transaction_journal
        if journal is None:
            raise RuntimeError("application resume requires its durable step journal")
        reconcile_application_step(
            application,
            journal,
            snapshot_directory,
            checkpoint_only_recovery_step=checkpoint_only_recovery_step,
        )
        application.record_method_state()
        return application

    @classmethod
    def resume_formal(
        cls,
        *,
        snapshot_directory: Path,
        backbone_config: QwenDeploymentConfig,
        task_provider_factory: TaskProviderFactory,
        base_session_factory: BaseRolloutSessionFactory,
        terminal_components: TerminalComponents,
        checkpoint_storage: PrivateCheckpointStorageBinding,
        initial_checkpoint: PrivateInitialCheckpointBinding,
        public_identity: ApplicationPublicIdentity,
        event_log: LiveAttemptEventLog,
        clock: Callable[[], str],
        runtime: FormalRuntimeDependencies,
        checkpoint_only_recovery_step: int | None = None,
    ) -> SKILLEVApplication:
        journal = StepTransactionJournal(
            (Path(checkpoint_storage.directory) / "step-transactions").resolve()
        )
        application = cls.resume(
            snapshot_directory=snapshot_directory,
            backbone_config=backbone_config,
            task_provider_factory=task_provider_factory,
            base_session_factory=base_session_factory,
            terminal_components=terminal_components,
            checkpoint_storage=checkpoint_storage,
            initial_checkpoint=initial_checkpoint,
            public_identity=public_identity,
            event_log=event_log,
            clock=clock,
            generator_factory=runtime.rollout_generator_factory,
            gradient_preparer=runtime.gradient_preparer,
            workflow_binding=runtime.workflow_resources.binding,
            workflow_resources=runtime.workflow_resources,
            step_adapter_publisher_factory=runtime.step_adapter_publisher_factory,
            r2flow_rollout=getattr(runtime, "r2flow_rollout", None),
            step_transaction_journal=journal,
            checkpoint_only_recovery_step=checkpoint_only_recovery_step,
        )
        runtime.require_bound(application)
        return application

    @classmethod
    def build_from_components(
        cls,
        *,
        backbone: PolicyBackbone,
        task_provider: TaskProvider,
        base_session_factory: BaseRolloutSessionFactory,
        terminal_components: TerminalComponents,
        checkpoint_storage: PrivateCheckpointStorageBinding,
        public_identity: ApplicationPublicIdentity,
        event_log: LiveAttemptEventLog,
        clock: Callable[[], str],
        library: SkillLibrary,
        projections: ApplicationProjectionPipeline,
        detector: LibrarySegmentDetector,
        run_progress: AttemptRunProgress,
        generator: RolloutGenerator | None = None,
        generator_factory: RolloutGeneratorFactory | None = None,
        gradient_preparer: TTBStepPreparer | None = None,
        workflow_binding: RolloutWorkflowBinding | None = None,
        workflow_resources: RolloutWorkflowResources | None = None,
        step_adapter_publisher_factory: Callable[[PolicyBackbone], StepAdapterPublisher]
        | None = None,
        step_transaction_journal: StepTransactionJournal | None = None,
        r2flow_rollout: R2FlowRolloutBinding | None = None,
    ) -> SKILLEVApplication:
        run_plan = _require_run_plan(public_identity)
        if run_progress.plan != run_plan:
            raise ValueError("application run progress differs from public run plan")
        application = cls._assemble(
            backbone=backbone,
            task_provider=task_provider,
            base_session_factory=base_session_factory,
            terminal_components=terminal_components,
            checkpoint_storage=checkpoint_storage,
            event_log=event_log,
            clock=clock,
            library=library,
            projections=projections,
            detector=detector,
            public_identity=public_identity,
            run_progress=run_progress,
            generator=generator,
            generator_factory=generator_factory,
            gradient_preparer=gradient_preparer,
            workflow_binding=workflow_binding,
            workflow_resources=workflow_resources,
            step_adapter_publisher_factory=step_adapter_publisher_factory,
            step_transaction_journal=step_transaction_journal,
            r2flow_rollout=r2flow_rollout,
        )
        application._emit_library_initialized()
        application.evolution_loop.save_initial_snapshot()
        return application

    @classmethod
    def _assemble(
        cls,
        *,
        backbone: PolicyBackbone,
        task_provider: TaskProvider,
        base_session_factory: BaseRolloutSessionFactory,
        terminal_components: TerminalComponents,
        checkpoint_storage: PrivateCheckpointStorageBinding,
        public_identity: ApplicationPublicIdentity,
        event_log: LiveAttemptEventLog,
        clock: Callable[[], str],
        library: SkillLibrary,
        projections: ApplicationProjectionPipeline,
        detector: LibrarySegmentDetector,
        run_progress: AttemptRunProgress,
        generator: RolloutGenerator | None = None,
        generator_factory: RolloutGeneratorFactory | None = None,
        gradient_preparer: TTBStepPreparer | None = None,
        workflow_binding: RolloutWorkflowBinding | None = None,
        workflow_resources: RolloutWorkflowResources | None = None,
        step_adapter_publisher_factory: Callable[[PolicyBackbone], StepAdapterPublisher]
        | None = None,
        step_transaction_journal: StepTransactionJournal | None = None,
        r2flow_rollout: R2FlowRolloutBinding | None = None,
    ) -> SKILLEVApplication:
        if step_transaction_journal is None:
            step_transaction_journal = StepTransactionJournal(
                (Path(checkpoint_storage.directory) / "step-transactions").resolve()
            )
        config = _require_application_config(public_identity)
        run_plan = _require_run_plan(public_identity)
        if run_progress.plan != run_plan:
            raise ValueError("application run progress differs from public run plan")
        _validate_library_task_family_universe(
            library,
            terminal_components.authoring_authority.allowed_task_families,
        )
        snapshot_identity = public_identity.runtime_snapshot_identity()
        scoring = getattr(base_session_factory, "terminal_evaluation_conditions_json", None)
        if scoring != snapshot_identity.terminal_evaluation_conditions_json:
            raise ValueError("terminal evaluator configuration differs from the training identity")
        if (
            getattr(base_session_factory, "task_feature_mapping_version", None)
            != snapshot_identity.task_feature_mapping_version
        ):
            raise ValueError("task feature mapping differs from the training identity")

        emitter = RuntimeEventEmitter(
            log=event_log,
            producer_id="skillev-application",
            clock=clock,
        )
        if generator is not None and generator_factory is not None:
            raise ValueError("provide either a rollout generator or a generator factory")
        active_workflow_binding = (
            workflow_resources.binding
            if workflow_binding is None and workflow_resources is not None
            else workflow_binding or RolloutWorkflowBinding()
        )
        if workflow_resources is not None and workflow_resources.binding != active_workflow_binding:
            raise ValueError("rollout workflow resources differ from the binding")
        active_workflow_resources = (
            workflow_resources
            if workflow_resources is not None
            else RolloutWorkflowResources(active_workflow_binding)
        )
        active_generator = (
            generator
            if generator is not None
            else (
                generator_factory(backbone, active_workflow_resources)
                if generator_factory is not None
                else LocalPolicyGenerator(
                    backbone,
                    episode_cache_enabled=(active_workflow_binding.max_resident_trajectories == 1),
                )
            )
        )
        if active_generator.tokenizer.tokenizer_id != backbone.tokenizer.tokenizer_id:
            raise ValueError("rollout generator and training backbone use different tokenizers")
        r2flow_runtime = (
            None
            if r2flow_rollout is None
            else r2flow_rollout.bind(
                rollout=config.trainer.rollout,
                generator=active_generator,
                ledger=terminal_components.ledger,
                resources=active_workflow_resources,
                emitter=emitter,
            )
        )
        objective = config.trainer.method.objective
        if objective is None:
            raise ValueError("the rollout state map is declared by the method objective")
        retriever = TaskConditionedSkillRetriever(
            library=library,
        )
        session_factory = RetrievingRolloutSessionFactory(
            base_factory=base_session_factory,
            retriever=retriever,
        )
        artifact_store = FilesystemTrainingCheckpointStore(root=Path(checkpoint_storage.directory))
        snapshot_store = RuntimeSnapshotStore(artifact_store)
        training_loop = TrainingLoop(
            backbone=backbone,
            generator=active_generator,
            task_provider=task_provider,
            session_factory=session_factory,
            context_assembler=config.trainer.rollout.context_assembler(
                maximum_h0_tokens=config.maximum_h0_tokens,
                state_map=objective.state_map,
            ),
            library=library,
            config=config.trainer,
            ledger=terminal_components.ledger,
            emitter=emitter,
            clock=clock,
            projections=projections,
            checkpoint_store=artifact_store,
            sampling_schedule_hash=public_identity.sampling_schedule_hash,
            ordered_task_sequence_hash=public_identity.ordered_task_sequence_hash,
            gradient_preparer=gradient_preparer,
            workflow_binding=active_workflow_binding,
            workflow_resources=active_workflow_resources,
            skill_executor_factory=None
            if r2flow_runtime is None
            else r2flow_runtime.skill_executor_factory,
            event_grammar=None if r2flow_runtime is None else r2flow_runtime.event_grammar,
            flow_offset_record=None
            if r2flow_rollout is None
            else r2flow_rollout.run_root / FLOW_OFFSET_RECORD_FILE,
        )
        phase_carrier = PhaseCarrier()
        snapshot_factory = ApplicationSnapshotFactory(
            training_loop=training_loop,
            task_provider=task_provider,
            run_progress=run_progress,
            snapshot_identity=snapshot_identity,
            library=library,
            projections=projections,
            detector=detector,
            phase_carrier=phase_carrier,
        )
        step_adapter_publisher = (
            None
            if step_adapter_publisher_factory is None
            else step_adapter_publisher_factory(backbone)
        )
        evolution_loop = EvolutionLoop(
            training_loop=training_loop,
            library=library,
            phi_budget=terminal_components.phi_budget,
            emitter=emitter,
            snapshot_store=snapshot_store,
            snapshot_factory=snapshot_factory,
            run_progress=run_progress,
            step_adapter_publisher=step_adapter_publisher,
            step_transaction_journal=step_transaction_journal,
        )
        return cls(
            backbone=backbone,
            generator=active_generator,
            library=library,
            retriever=retriever,
            projections=projections,
            training_loop=training_loop,
            detector=detector,
            evolution_loop=evolution_loop,
            snapshot_store=snapshot_store,
            emitter=emitter,
            public_identity=public_identity,
            run_progress=run_progress,
            snapshot_identity=snapshot_identity,
            phase_carrier=phase_carrier,
        )

    def _emit_library_initialized(self) -> None:
        state = self.library.state
        self.emitter.emit(
            EventType.LIBRARY_INITIALIZED,
            LibraryInitialized(
                documents=tuple(document.to_value() for document in self.library.all_documents()),
                active_skill_ids=state.active_skill_ids,
                library_version=state.current_version,
                initial_optimizer_step=self.training_loop.optimizer_step,
                method_identity_hash=self.public_identity.method_identity_hash,
                run_cursor=self.run_progress.state.to_source_value(),
            ).to_value(),
        )
        self.record_method_state()

    def record_method_state(self) -> None:
        from skillev.application_reporting import resolved_method_state

        self.emitter.emit(EventType.METHOD_STATE_RECORDED, resolved_method_state(self))


def _validate_library_task_family_universe(
    library: SkillLibrary,
    task_family_universe: tuple[str, ...],
) -> None:
    if not task_family_universe or "*" in task_family_universe:
        raise ValueError("authoring authority requires a finite task-family universe")
    universe = set(task_family_universe)
    for document in library.active_documents():
        families = document.applicability.task_families
        if families != ("*",) and not set(families) <= universe:
            raise ValueError("active skill targets a family outside the formal universe")


def _require_application_config(identity: ApplicationPublicIdentity) -> ApplicationConfig:
    config = identity.application_config
    if not isinstance(config, ApplicationConfig):
        raise TypeError("public identity application config is incompatible")
    return config


def _require_run_plan(identity: ApplicationPublicIdentity) -> ExactAttemptRunPlan:
    plan = identity.run_plan
    if not isinstance(plan, ExactAttemptRunPlan):
        raise TypeError("public identity run plan is incompatible")
    return plan


def _require_initial_cursor(
    identity: ApplicationPublicIdentity,
    plan: ExactAttemptRunPlan,
) -> AttemptRunCursorState:
    cursor = identity.initial_run_cursor
    if not isinstance(cursor, AttemptRunCursorState):
        raise TypeError("public identity initial run cursor is incompatible")
    cursor.require_plan(plan)
    return cursor


__all__ = [
    "ApplicationConfig",
    "ApplicationPublicIdentity",
    "RolloutGeneratorFactory",
    "SKILLEVApplication",
    "TaskProviderFactory",
    "TerminalComponents",
]
