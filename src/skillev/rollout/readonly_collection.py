from __future__ import annotations

import asyncio
import json
import shutil
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from skillev.contracts import JsonValue, ScientificSamplingCoordinate, normalize_json
from skillev.contracts.action_decoding import require_action_decoding_coupling
from skillev.diagnostics.rollout_progress import RolloutProgress, bind_progress
from skillev.diagnostics.rollout_trace import RolloutTraceSink
from skillev.runtime import BudgetLedger, BudgetVector, RuntimeEventEmitter, SkillLibraryState
from skillev.runtime.budget_ledger import LedgerEntry
from skillev.training.config import PolicyRolloutConfig
from skillev.training.inflight import durable_json
from skillev.training.rollout_workflow import RolloutBatchWorkflow, RolloutWorkflowResources

from . import CanonicalInitialContextAssembler, RolloutArtifact, RolloutGenerator, RolloutTask
from .episode_executor import EpisodeSessionFactory, episode_decoding, execute_episode
from .errors import (
    EpisodeInfrastructureError,
    GenerationInfrastructureError,
    InitialContextInfrastructureError,
)

if TYPE_CHECKING:
    from skillev.runtime.frozen_executor import FrozenSkillExecutor

    from .event_grammar_runtime import EventGrammarRuntime

_INFRASTRUCTURE_ERRORS = (
    EpisodeInfrastructureError,
    GenerationInfrastructureError,
    InitialContextInfrastructureError,
    TimeoutError,
    ConnectionError,
)


def evaluation_isolation() -> dict[str, JsonValue]:
    return {
        "mode": "architecture-matched-read-only-episodes",
        "training_updates": 0,
        "posterior_updates": 0,
        "skill_evolution": 0,
        "training_evidence_writes": 0,
        "training_batch_membership": False,
        "ttb_loss_contribution": False,
        "backward_or_z_execution": False,
    }


@dataclass(frozen=True, slots=True)
class EvaluationEpisodeOutcome:
    position: int
    task_id: str
    artifact: RolloutArtifact | None
    infrastructure_error: str | None = None
    execution_status: str | None = None
    originating_failure_position: int | None = None
    preparation_chunk_start: int | None = None

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "purpose": "evaluation-only",
            "position": self.position,
            "task_id": self.task_id,
            "artifact": None if self.artifact is None else self.artifact.to_value(),
            "infrastructure_error": self.infrastructure_error,
            "execution_status": self.execution_status
            or ("completed" if self.artifact else "unknown"),
            "originating_failure_position": self.originating_failure_position,
            "preparation_chunk_start": self.preparation_chunk_start,
        }


@dataclass(frozen=True, slots=True)
class ReadOnlyPanelContinuation:
    source_root: Path
    outcomes: tuple[EvaluationEpisodeOutcome, ...]
    entries: tuple[LedgerEntry, ...]
    plan: dict[str, JsonValue]
    elapsed_seconds: float


class IncompletePanelError(ValueError, RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ReadOnlyPanelResult:
    outcomes: tuple[EvaluationEpisodeOutcome, ...]
    consumed_budget: BudgetVector
    elapsed_seconds: float

    @property
    def artifacts(self) -> tuple[RolloutArtifact, ...]:
        if any(row.artifact is None for row in self.outcomes):
            raise IncompletePanelError(
                "evaluation has infrastructure failures; no complete panel exists"
            )
        return tuple(cast(RolloutArtifact, row.artifact) for row in self.outcomes)


async def collect_readonly_panel(
    *,
    root: Path,
    tasks: tuple[RolloutTask, ...],
    generator: RolloutGenerator,
    sessions: EpisodeSessionFactory,
    library: SkillLibraryState,
    rollout: PolicyRolloutConfig,
    assembler: CanonicalInitialContextAssembler,
    epsilon_min: float,
    condition_id: str,
    sampling_schedule_id: str,
    ordered_sequence_id: str,
    resources: RolloutWorkflowResources,
    ledger: BudgetLedger,
    emitter: RuntimeEventEmitter,
    clock: Callable[[], str],
    chunk_size: int,
    schedule_purpose: str = "architecture-matched-iid-evaluation",
    anchor_ordinal: int = 0,
    sequence_offset: int = 0,
    live_progress: dict[str, RolloutProgress] | None = None,
    trace_sink: RolloutTraceSink | None = None,
    continuation: ReadOnlyPanelContinuation | None = None,
    skill_executor_factory: Callable[[str], FrozenSkillExecutor] | None = None,
    event_grammar: EventGrammarRuntime | None = None,
    sequence_positions: tuple[int, ...] | None = None,
) -> ReadOnlyPanelResult:
    if not tasks or len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("evaluation panel needs unique planned task identities")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("evaluation chunk size must be positive")
    require_action_decoding_coupling(rollout.action_decoding, generator)
    if sequence_positions is not None and (
        sequence_offset != 0
        or len(sequence_positions) != len(tasks)
        or any(type(item) is not int or item < 0 for item in sequence_positions)
        or list(sequence_positions) != sorted(set(sequence_positions))
    ):
        raise ValueError("sequence positions must be one strictly increasing position per task")

    def sequence_position(position: int) -> int:
        if sequence_positions is None:
            return sequence_offset + position
        return sequence_positions[position]

    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    policy = generator.snapshot()
    if live_progress is None:
        live_progress = {}
    progress = live_progress
    started = time.perf_counter()
    durable_json(root / "isolation.json", evaluation_isolation())
    plan: dict[str, JsonValue] = {
        "condition_id": condition_id,
        "policy_snapshot_id": policy.snapshot_id,
        "library_version": library.current_version,
        "tasks": [task.to_value() for task in tasks],
        "sampling_schedule_id": sampling_schedule_id,
        "ordered_sequence_id": ordered_sequence_id,
        "anchor_ordinal": anchor_ordinal,
        "sequence_offset": sequence_offset,
        "schedule_purpose": schedule_purpose,
        "chunk_size": chunk_size,
        "rollout": rollout.to_value(),
        "epsilon_min": epsilon_min,
        "trajectory_namespace": root.name
        if continuation is None
        else continuation.plan.get("trajectory_namespace", continuation.source_root.name),
    }
    if sequence_positions is not None:
        plan["sequence_positions"] = list(sequence_positions)
    if continuation is not None:
        if any(plan.get(k) != v for k, v in continuation.plan.items()):
            raise ValueError("continuation changed the original collection plan")
        if len(continuation.outcomes) % chunk_size or len(continuation.outcomes) >= len(tasks):
            raise ValueError("continuation must stop at a failed preparation chunk boundary")
        for position, outcome in enumerate(continuation.outcomes):
            artifact = outcome.artifact
            if (
                outcome.position != position
                or outcome.task_id != tasks[position].task_id
                or artifact is None
                or artifact.manifest.policy_snapshot != policy
                or artifact.manifest.library_version != library.current_version
                or artifact.manifest.sampling_coordinate is None
                or artifact.manifest.sampling_coordinate.sequence_position
                != sequence_position(position)
            ):
                raise ValueError("saved complete prefix differs from the original plan")
        ledger.restore_completed(continuation.entries)
        for outcome in continuation.outcomes:
            name = f"episode-{outcome.position:06d}-private.json"
            shutil.copyfile(continuation.source_root / name, root / name)
    durable_json(root / "plan-private.json", plan)

    def save_failure(error: BaseException, *, stage: str, position: int) -> None:
        location: dict[str, JsonValue] = (
            {"position": position, "task_id": tasks[position].task_id}
            if stage == "episode"
            else {"chunk_start": position, "originating_task_id": None}
        )
        durable_json(
            root / f"failure-{stage}-{position:06d}-private.json",
            {
                **location,
                "stage": stage,
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(traceback.format_exception(error)),
                "status": "cancelled"
                if isinstance(error, asyncio.CancelledError)
                else "infrastructure-error"
                if isinstance(error, _INFRASTRUCTURE_ERRORS)
                else "unexpected-error",
                "native_label": None,
                "planned_population_count": len(tasks),
            },
        )

    async def execute(item: tuple[int, RolloutTask]) -> EvaluationEpisodeOutcome:
        position, task = item
        trajectory_id = f"{plan['trajectory_namespace']}-eval-{position:06d}"
        durable_json(
            root / f"started-episode-{position:06d}.json",
            {
                "position": position,
                "trajectory_id": trajectory_id,
            },
        )
        domain = task.task_family.partition("/")[0]
        row = RolloutProgress(
            canonical_position=position,
            benchmark_episode_index=sum(
                earlier.task_family.partition("/")[0] == domain for earlier in tasks[:position]
            ),
            trajectory_id=trajectory_id,
            task_domain=task.task_family,
            batch_id=root.name,
            policy_snapshot_id=policy.snapshot_id,
            library_version=library.current_version,
            max_turns=rollout.max_turns
            if task.budget_profile is None
            else task.budget_profile.max_turns,
        )
        progress[trajectory_id] = row
        with bind_progress(row):
            try:
                artifact = await execute_episode(
                    generator=generator,
                    sessions=sessions,
                    assembler=assembler,
                    rollout=rollout,
                    task=task,
                    trajectory_id=trajectory_id,
                    library=library,
                    coordinate=ScientificSamplingCoordinate(
                        sampling_schedule_hash=sampling_schedule_id,
                        schedule_purpose=schedule_purpose,
                        ordered_sequence_hash=ordered_sequence_id,
                        sequence_position=sequence_position(position),
                        task_id=task.task_id,
                        optimizer_step_or_anchor_ordinal=anchor_ordinal,
                    ),
                    decoding=episode_decoding(rollout, task),
                    epsilon_min=epsilon_min,
                    condition_id=condition_id,
                    ledger=ledger,
                    emitter=emitter.child(f"evaluation:{position:06d}"),
                    clock=clock,
                    resources=resources,
                    trace_sink=trace_sink,
                    skill_executor_factory=skill_executor_factory,
                    event_grammar=event_grammar,
                )
                if artifact.manifest.policy_snapshot != policy:
                    raise ValueError("evaluation episode used another policy snapshot")
                result = EvaluationEpisodeOutcome(position, task.task_id, artifact)
                row.stage("artifact-ready")
            except _INFRASTRUCTURE_ERRORS as error:
                save_failure(error, stage="episode", position=position)
                result = EvaluationEpisodeOutcome(
                    position,
                    task.task_id,
                    None,
                    type(error).__name__,
                    "episode-infrastructure-failure",
                    position,
                )
                row.stage("failed")
            except BaseException as error:
                row.stage("failed")
                save_failure(error, stage="episode", position=position)
                raise
        durable_json(root / f"episode-{position:06d}-private.json", result.to_value())
        return result

    prior_progress: list[JsonValue] = []
    if continuation is not None:
        prior_progress = json.loads(
            (continuation.source_root / "rollout-progress.json").read_text()
        )["trajectories"]

    async def save_progress(status: str) -> None:
        durable_json(
            root / "rollout-progress.json",
            {
                "status": status,
                "elapsed_seconds": time.perf_counter() - started,
                "trajectories": [
                    *prior_progress,
                    *[normalize_json(row.snapshot()) for row in progress.values()],
                ],
                "scope": "evaluation-only",
            },
        )

    async def monitor() -> None:
        while True:
            await save_progress("collecting")
            await asyncio.sleep(20)

    workflow = RolloutBatchWorkflow[tuple[int, RolloutTask], EvaluationEpisodeOutcome](
        resources.binding
    )
    monitor_task = asyncio.create_task(monitor())
    outcomes: list[EvaluationEpisodeOutcome] = list(continuation.outcomes) if continuation else []
    completed = False
    try:
        for start in range(len(outcomes), len(tasks), chunk_size):
            durable_json(root / f"chunk-{start:06d}.json", {"start": start, "status": "preparing"})
            selected = tasks[start : start + chunk_size]
            prepare = getattr(sessions, "prepare_tasks", None)
            try:
                if prepare is not None:
                    hydrated = await prepare(selected)
                    if tuple(task.task_id for task in hydrated) != tuple(
                        task.task_id for task in selected
                    ):
                        raise ValueError("evaluation hydration changed the frozen population")
                    selected = hydrated
            except _INFRASTRUCTURE_ERRORS as error:
                save_failure(error, stage="preparation", position=start)
                durable_json(
                    root / f"chunk-{start:06d}.json",
                    {
                        "start": start,
                        "status": "preparation-failed-before-episodes",
                        "episode_started": False,
                    },
                )
                for position, task in enumerate(tasks[start:], start=start):
                    outcome = EvaluationEpisodeOutcome(
                        position,
                        task.task_id,
                        None,
                        "not-started-after-preparation-failure",
                        "not-started",
                        None,
                        start,
                    )
                    durable_json(root / f"episode-{position:06d}-private.json", outcome.to_value())
                    outcomes.append(outcome)
                break
            except BaseException as error:
                save_failure(error, stage="preparation", position=start)
                raise
            durable_json(
                root / f"chunk-{start:06d}.json", {"start": start, "status": "episodes-started"}
            )
            outcomes.extend(
                await workflow.run(
                    tuple(enumerate(selected, start=start)),
                    execute,
                    declared_horizons=tuple(
                        rollout.max_turns
                        if task.budget_profile is None
                        else task.budget_profile.max_turns
                        for task in selected
                    ),
                )
            )
            if any(row.artifact is None for row in outcomes):
                for position in range(len(outcomes), len(tasks)):
                    outcome = EvaluationEpisodeOutcome(
                        position,
                        tasks[position].task_id,
                        None,
                        "not-started-after-infrastructure-failure",
                        "not-started",
                    )
                    durable_json(root / f"episode-{position:06d}-private.json", outcome.to_value())
                    outcomes.append(outcome)
                break
        if generator.snapshot() != policy:
            raise ValueError("read-only evaluation changed its selected snapshot")
        ledger.assert_fully_settled()
        completed = True
    finally:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        durable_json(
            root / "ledger-private.json",
            {
                "run_id": ledger.run_id,
                "attempt_id": ledger.attempt_id,
                "settled": ledger.settled.to_value(),
                "entries": [
                    {
                        "reservation_id": entry.reservation.reservation_id,
                        "run_id": entry.reservation.run_id,
                        "attempt_id": entry.reservation.attempt_id,
                        "invocation_id": entry.reservation.invocation_id,
                        "maximum": entry.reservation.maximum.to_value(),
                        "actual": None
                        if entry.settlement is None
                        else entry.settlement.actual.to_value(),
                    }
                    for entry in ledger.entries
                ],
            },
        )
        await save_progress(
            "complete"
            if completed and all(row.artifact is not None for row in outcomes)
            else "failed"
        )
    return ReadOnlyPanelResult(
        tuple(outcomes),
        ledger.settled,
        time.perf_counter() - started + (continuation.elapsed_seconds if continuation else 0.0),
    )
