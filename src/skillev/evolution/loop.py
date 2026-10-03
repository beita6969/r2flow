from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import ExitStack
from dataclasses import dataclass
from inspect import isawaitable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

from skillev.contracts import (
    TrainingStepCommit,
    TrainingStepReportValue,
)
from skillev.runtime import (
    AdapterGeneration,
    BudgetLedger,
    BudgetVector,
    EventEnvelope,
    EventType,
    FullAttemptSummary,
    RuntimeEventEmitter,
    RuntimeSnapshotStore,
    SkillLibrary,
    StepAdapterPublisher,
    StepTransactionJournal,
    StepTransactionRecord,
    StepTransactionState,
)
from skillev.runtime.attempt_failures import (
    EventAppendFailedError,
)
from skillev.runtime.attempt_run_plan import (
    AttemptRunProgress,
    ExactAttemptRunPlan,
    RemainingAttemptCapacity,
)


if TYPE_CHECKING:
    from skillev.training.checkpoint import TrainingCheckpointSnapshot
    from skillev.training.planning import TrainingStepExecutionContext


class EvolutionTrainingLoop(Protocol):
    @property
    def optimizer_step(self) -> int: ...

    @property
    def checkpoint_due(self) -> bool: ...

    @property
    def policy_snapshot_id(self) -> str: ...

    @property
    def ledger(self) -> BudgetLedger: ...

    @property
    def scientific_base_seed(self) -> int: ...

    async def collect_batch(self) -> object: ...

    def apply_step(
        self,
        context: TrainingStepExecutionContext,
    ) -> TrainingStepReportValue: ...

    def set_execution_stage(self, stage: str) -> None: ...

    def install_applied_projection(self) -> None: ...

    def finalize_applied_step(self) -> TrainingStepReportValue: ...

    @property
    def pending_commit(self) -> TrainingStepCommit: ...

    def finalize_applied_step_event(self, event: EventEnvelope) -> TrainingStepReportValue: ...

    def validate_attempt_budget(
        self,
        *,
        capacity: RemainingAttemptCapacity,
        phi_per_cycle_maximum: BudgetVector,
    ) -> None: ...


class EvolutionSnapshotFactory(Protocol):
    def snapshot(self) -> TrainingCheckpointSnapshot: ...


@dataclass(frozen=True, slots=True)
class PhiBudgetAuthority:
    cap: BudgetVector

    @property
    def available(self) -> BudgetVector:
        return self.cap


EvolutionRunSummary = FullAttemptSummary


@dataclass(frozen=True, slots=True)
class CommittedStepDurability:
    checkpoint_name: str
    adapter_revision: str

    def __post_init__(self) -> None:
        if Path(self.checkpoint_name).name != self.checkpoint_name:
            raise ValueError("committed checkpoint name must be one path component")
        if not self.adapter_revision.strip():
            raise ValueError("committed adapter revision must be non-empty")


@dataclass(frozen=True, slots=True)
class PreparedStepDurability:
    checkpoint_name: str
    prepared_adapter: object | None

    def __post_init__(self) -> None:
        if Path(self.checkpoint_name).name != self.checkpoint_name:
            raise ValueError("prepared checkpoint name must be one path component")


class EvolutionLoop:
    def __init__(
        self,
        *,
        training_loop: EvolutionTrainingLoop,
        library: SkillLibrary,
        phi_budget: PhiBudgetAuthority,
        emitter: RuntimeEventEmitter,
        snapshot_store: RuntimeSnapshotStore,
        snapshot_factory: EvolutionSnapshotFactory,
        run_progress: AttemptRunProgress,
        step_adapter_publisher: StepAdapterPublisher | None = None,
        step_transaction_journal: StepTransactionJournal | None = None,
    ) -> None:
        self._training_loop = training_loop
        self._library = library
        self._phi_budget = phi_budget
        self._emitter = emitter
        self._snapshot_store = snapshot_store
        self._snapshot_factory = snapshot_factory
        self._run_progress = run_progress
        self._step_adapter_publisher = step_adapter_publisher
        self._step_transaction_journal = step_transaction_journal
        self._failed = False
        self._current_adapter_published = False
        self._final_snapshot_directory: Path | None = None

    def save_initial_snapshot(self) -> Path:
        if self._training_loop.optimizer_step != 0:
            raise ValueError("initial snapshot requires an untrained application")
        return Path(
            self._snapshot_store.save(
                self._snapshot_factory.snapshot(), name="initial-step-00000000"
            )
        )

    def save_condition_boundary(self, name: str) -> Path:
        if self._step_transaction_journal is None or self._step_transaction_journal.pending():
            raise RuntimeError("condition boundary requires settled step transactions")
        return Path(self._snapshot_store.save(self._snapshot_factory.snapshot(), name=name))

    @property
    def final_training_snapshot_directory(self) -> Path:
        if self._final_snapshot_directory is None:
            raise RuntimeError("final training snapshot is unavailable before run completion")
        return self._final_snapshot_directory

    @property
    def step_adapter_publisher(self) -> StepAdapterPublisher | None:
        return self._step_adapter_publisher

    @property
    def step_transaction_journal(self) -> StepTransactionJournal | None:
        return self._step_transaction_journal

    async def run(
        self,
        plan: ExactAttemptRunPlan,
        *,
        maximum_steps_this_attempt: int | None = None,
        stop_requested: Callable[[], bool | Awaitable[bool]] | None = None,
    ) -> EvolutionRunSummary:
        if self._failed:
            raise RuntimeError("failed training transaction requires a fresh application restore")
        completed = False
        try:
            from skillev.training.stopping import TrainingPausedError

            try:
                summary = await self._run(
                    plan,
                    maximum_steps_this_attempt=maximum_steps_this_attempt,
                    stop_requested=stop_requested,
                )
            except TrainingPausedError:
                completed = True
                raise
            completed = True
            return summary
        finally:
            self._failed = not completed

    async def _run(
        self,
        plan: ExactAttemptRunPlan,
        *,
        maximum_steps_this_attempt: int | None = None,
        stop_requested: Callable[[], bool | Awaitable[bool]] | None = None,
    ) -> EvolutionRunSummary:
        if not isinstance(plan, ExactAttemptRunPlan):
            raise TypeError("EvolutionLoop.run requires ExactAttemptRunPlan")
        if plan.content_hash != self._run_progress.plan.content_hash:
            raise ValueError("run plan differs from application construction")
        initial_cursor = self._run_progress.state
        initial_cursor.require_plan(plan)
        if initial_cursor.committed_cycles != 0:
            raise ValueError("phase detection is disabled but cycles were committed")
        initial_optimizer_step = self._training_loop.optimizer_step
        if maximum_steps_this_attempt is not None and (
            type(maximum_steps_this_attempt) is not int or maximum_steps_this_attempt < 1
        ):
            raise ValueError("bounded run steps must be a positive integer")
        target_steps = plan.total_training_steps
        if maximum_steps_this_attempt is not None:
            target_steps = min(
                target_steps,
                initial_cursor.completed_training_steps + maximum_steps_this_attempt,
            )
        remaining_steps = target_steps - initial_cursor.completed_training_steps
        if remaining_steps < 0:
            raise ValueError("run cursor is beyond run plan")
        self._training_loop.validate_attempt_budget(
            capacity=RemainingAttemptCapacity(
                training_steps=remaining_steps,
                possible_cycles=plan.maximum_cycles - initial_cursor.committed_cycles,
            ),
            phi_per_cycle_maximum=self._phi_budget.available,
        )
        if (
            remaining_steps
            and self._step_adapter_publisher is not None
            and not self._current_adapter_published
        ):
            self._publish_current_adapter()
        reports: list[TrainingStepReportValue] = []
        cycles_before = initial_cursor.committed_cycles
        actions_before = initial_cursor.committed_actions

        while self._run_progress.state.completed_training_steps < target_steps:
            requested = False if stop_requested is None else stop_requested()
            should_stop = await requested if isawaitable(requested) else requested
            if should_stop:
                from skillev.training.stopping import TrainingPausedError

                self._training_loop.ledger.assert_fully_settled()
                step = self._training_loop.optimizer_step
                checkpoint = Path(
                    self._snapshot_store.save(
                        self._snapshot_factory.snapshot(),
                        name=f"paused-step-{step:08d}-{uuid4().hex[:12]}",
                    )
                )
                self._training_loop.set_execution_stage("paused-after-checkpoint")
                raise TrainingPausedError(step, checkpoint)
            position = self._run_progress.state.completed_training_steps + 1
            from skillev.training.planning import TrainingStepExecutionContext

            next_step_cursor = self._run_progress.preview_training_step()
            batch = await self._training_loop.collect_batch()
            transaction = self._begin_step_transaction(batch)
            applied_report = self._training_loop.apply_step(
                TrainingStepExecutionContext(next_step_cursor.to_source_value())
            )
            transaction = self._advance_step_transaction(
                transaction,
                StepTransactionState.OPTIMIZER_APPLIED,
                policy_snapshot_after=self._training_loop.policy_snapshot_id,
            )
            self._training_loop.install_applied_projection()
            transaction = self._advance_step_transaction(
                transaction,
                StepTransactionState.PROJECTION_INSTALLED,
            )
            self._run_progress.commit_training_step_state(next_step_cursor)

            self._training_loop.set_execution_stage("phase-detection-and-evolution")
            evolution_events: tuple[tuple[EventType, object], ...] = (
                (
                    EventType.PHASE_DETECTION_RECORDED,
                    {
                        "optimizer_step": position,
                        "batch_id": applied_report.batch_id,
                        "library_version": self._library.current_version,
                        "reason": "phase-detection-disabled",
                    },
                ),
            )
            transaction = self._advance_step_transaction(
                transaction,
                StepTransactionState.EVOLUTION_RESOLVED,
            )
            evolution_events = with_flow_step_event(
                evolution_events,
                getattr(self._training_loop, "pending_flow_payload", None),
            )
            self._training_loop.set_execution_stage("checkpoint-publication")
            prepared_durability = self._prepare_step_durability()
            prepared_events = (
                ()
                if transaction is None
                else self._emitter.prepare_many(
                    (
                        (
                            EventType.TRAINING_STEP_COMMITTED,
                            self._training_loop.pending_commit.to_value(),
                        ),
                        *evolution_events,
                    )
                )
            )
            transaction = self._advance_step_transaction(
                transaction,
                StepTransactionState.CHECKPOINT_PUBLISHED,
                checkpoint_name=prepared_durability.checkpoint_name,
                source_events=prepared_events,
            )
            self._training_loop.set_execution_stage("adapter-publication")
            durability = self._commit_step_adapter(prepared_durability)
            self._training_loop.set_execution_stage("source-event-publication")
            if transaction is None:
                reports.append(self._training_loop.finalize_applied_step())
                self._publish_evolution_events(evolution_events)
            else:
                transaction = self._advance_step_transaction(
                    transaction,
                    StepTransactionState.ADAPTER_COMMITTED,
                    adapter_revision=durability.adapter_revision,
                )
                reports.append(self._training_loop.finalize_applied_step_event(prepared_events[0]))
                self._publish_prepared_evolution_events(prepared_events[1:])
                transaction = self._advance_step_transaction(
                    transaction,
                    StepTransactionState.SOURCE_EVENTS_PUBLISHED,
                )
                self._advance_step_transaction(transaction, StepTransactionState.COMMITTED)
            self._training_loop.set_execution_stage("committed")

        final_cursor = self._run_progress.state
        if final_cursor.completed_training_steps != target_steps:
            raise RuntimeError("successful run ended before its requested target")
        self._final_snapshot_directory = Path(
            self._snapshot_store.save(
                self._snapshot_factory.snapshot(),
                name=f"final-step-{self._training_loop.optimizer_step:08d}",
            )
        )

        return EvolutionRunSummary(
            reports=tuple(reports),
            planned_training_steps_this_attempt=remaining_steps,
            completed_training_steps_this_attempt=len(reports),
            actions_committed_this_attempt=final_cursor.committed_actions - actions_before,
            cycles_committed_this_attempt=final_cursor.committed_cycles - cycles_before,
            cycles_committed_in_run=final_cursor.committed_cycles,
            initial_optimizer_step=initial_optimizer_step,
            final_optimizer_step=self._training_loop.optimizer_step,
            final_library_version=self._library.current_version,
            final_policy_snapshot_id=self._training_loop.policy_snapshot_id,
        )

    def _prepare_step_durability(self) -> PreparedStepDurability:
        prepared: object | None = None
        if self._step_adapter_publisher is not None:
            prepared = self._step_adapter_publisher.prepare(
                optimizer_step=self._training_loop.optimizer_step,
                policy_snapshot_id=self._training_loop.policy_snapshot_id,
            )
        with ExitStack() as rollback:
            if self._step_adapter_publisher is not None and prepared is not None:
                rollback.callback(self._step_adapter_publisher.rollback, prepared)
            snapshot = self._snapshot_factory.snapshot()
            checkpoint = Path(
                self._snapshot_store.save(
                    snapshot,
                    name=f"step-{self._training_loop.optimizer_step:08d}",
                )
            )
            if self._training_loop.checkpoint_due:
                self._snapshot_store.save(
                    snapshot,
                    name=f"cadence-step-{self._training_loop.optimizer_step:08d}",
                )
            self._snapshot_store.retain_recent(keep_recent=3)
            rollback.pop_all()
        return PreparedStepDurability(checkpoint.name, prepared)

    def _commit_step_adapter(
        self,
        prepared: PreparedStepDurability,
    ) -> CommittedStepDurability:
        adapter_revision = "no-external-adapter"
        if self._step_adapter_publisher is not None:
            if prepared.prepared_adapter is None:
                raise RuntimeError("step adapter preparation is absent")
            with ExitStack() as rollback:
                rollback.callback(
                    self._step_adapter_publisher.rollback,
                    prepared.prepared_adapter,
                )
                generation = self._step_adapter_publisher.commit(prepared.prepared_adapter)
                rollback.pop_all()
            self._current_adapter_published = True
            if self._step_transaction_journal is not None:
                if not isinstance(generation, AdapterGeneration):
                    raise TypeError("formal adapter publisher returned no generation")
                adapter_revision = generation.adapter_revision
            elif isinstance(generation, AdapterGeneration):
                adapter_revision = generation.adapter_revision
        return CommittedStepDurability(prepared.checkpoint_name, adapter_revision)

    def publish_restored_adapter(self) -> AdapterGeneration | None:
        if self._step_adapter_publisher is None:
            return None
        generation = self._publish_current_adapter()
        if not isinstance(generation, AdapterGeneration):
            raise TypeError("formal restored adapter publisher returned no generation")
        return generation

    def _publish_current_adapter(self) -> object | None:
        if self._step_adapter_publisher is None:
            return None
        generation = self._step_adapter_publisher.restore(
            optimizer_step=self._training_loop.optimizer_step,
            policy_snapshot_id=self._training_loop.policy_snapshot_id,
        )
        self._current_adapter_published = True
        return generation

    def _begin_step_transaction(self, batch: object) -> StepTransactionRecord | None:
        if self._step_transaction_journal is None:
            return None
        from skillev.training.planning import CollectedTrainingBatch

        if not isinstance(batch, CollectedTrainingBatch):
            raise TypeError("formal transaction requires a collected training batch")
        return self._step_transaction_journal.begin(
            optimizer_step=batch.optimizer_step,
            batch_id=batch.batch_id,
            policy_snapshot_before=batch.policy_snapshot_id,
        )

    def _publish_evolution_events(
        self,
        events: tuple[tuple[EventType, object], ...],
    ) -> None:
        try:
            for event_type, payload in events:
                self._emitter.emit(event_type, payload)
        except EventAppendFailedError:
            raise
        except OSError as error:
            raise EventAppendFailedError("full-method source event append failed") from error

    def _publish_prepared_evolution_events(
        self,
        events: tuple[EventEnvelope, ...],
    ) -> None:
        try:
            for event in events:
                self._emitter.publish_prepared(event)
        except EventAppendFailedError:
            raise
        except OSError as error:
            raise EventAppendFailedError("full-method source event append failed") from error

    def _advance_step_transaction(
        self,
        record: StepTransactionRecord | None,
        state: StepTransactionState,
        *,
        policy_snapshot_after: str | None = None,
        checkpoint_name: str | None = None,
        adapter_revision: str | None = None,
        source_events: tuple[EventEnvelope, ...] | None = None,
    ) -> StepTransactionRecord | None:
        if record is None:
            return None
        if self._step_transaction_journal is None:
            raise RuntimeError("step transaction record has no journal")
        return self._step_transaction_journal.advance(
            record,
            state,
            policy_snapshot_after=policy_snapshot_after,
            checkpoint_name=checkpoint_name,
            adapter_revision=adapter_revision,
            source_events=source_events,
        )


def with_flow_step_event(
    evolution_events: tuple[tuple[EventType, object], ...],
    flow_payload: object | None,
) -> tuple[tuple[EventType, object], ...]:
    if flow_payload is None:
        return evolution_events
    return ((EventType.FLOW_STEP_RECORDED, flow_payload), *evolution_events)


__all__ = [
    "EvolutionLoop",
    "EvolutionRunSummary",
    "EvolutionSnapshotFactory",
    "EvolutionTrainingLoop",
    "PhiBudgetAuthority",
]
