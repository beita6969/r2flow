from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from skillev.contracts import JsonValue, canonical_json, normalize_json
from skillev.evolution import (
    BaseRolloutSessionFactory,
    RetrievingRolloutSessionFactory,
    TaskConditionedSkillRetriever,
)
from skillev.rollout import (
    RolloutArtifact,
    RolloutGenerator,
    RolloutTask,
)
from skillev.rollout.readonly_collection import collect_readonly_panel, evaluation_isolation
from skillev.runtime import (
    BudgetLedger,
    BudgetVector,
    LiveAttemptEventLog,
    RuntimeEventEmitter,
    SkillLibrary,
    SkillLibraryState,
)
from skillev.training.config import TrainerConfig
from skillev.training.rollout_workflow import RolloutWorkflowBinding, RolloutWorkflowResources
from skillev.training.run_condition import EffectiveRunCondition

from .bayesian_training_setup import _clock
from .quality_panel import FixedQualityPanel, collection_probe

if TYPE_CHECKING:
    from collections.abc import Callable

    from skillev.runtime.executor_ledger import ExecutorCallRecord
    from skillev.training.r2flow_rollout import R2FlowRolloutBinding


@dataclass(frozen=True)
class ReadOnlyCollectionResult:
    condition: EffectiveRunCondition
    policy_snapshot_id: str
    diagnostic_artifacts: tuple[RolloutArtifact, ...]
    consumed_budget: BudgetVector
    library_snapshot_id: str | None = None
    architecture_id: str | None = None
    execution_controls: dict[str, JsonValue] | None = None

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "purpose": "diagnostic-collect-only",
            "format": "skillev-read-only-collection@1",
            "condition": self.condition.to_value(),
            "policy_snapshot_id": self.policy_snapshot_id,
            "diagnostic_artifacts": [artifact.to_value() for artifact in self.diagnostic_artifacts],
            "consumed_budget": self.consumed_budget.to_value(),
            "library_snapshot_id": self.library_snapshot_id,
            "architecture_id": self.architecture_id,
            "execution_controls": self.execution_controls,
            **evaluation_isolation(),
        }


async def collect_training_condition(
    *,
    root: Path,
    condition: EffectiveRunCondition,
    tasks: tuple[RolloutTask, ...],
    generator: RolloutGenerator,
    base_sessions: BaseRolloutSessionFactory,
    library_state: SkillLibraryState,
    trainer: TrainerConfig,
    maximum_h0_tokens: int,
    workflow: RolloutWorkflowBinding,
    sampling_schedule_id: str,
    ordered_task_sequence_id: str,
    sampled_policy_step: int = 0,
    workflow_resources: RolloutWorkflowResources | None = None,
    quality_panel: FixedQualityPanel | None = None,
    excluded_quality_sources: frozenset[tuple[str, str, str]] | None = None,
    chunk_size: int | None = None,
    architecture_id: str | None = None,
    execution_controls: dict[str, JsonValue] | None = None,
    r2flow_rollout: R2FlowRolloutBinding | None = None,
    schedule_purpose: str = "diagnostic-collect-only",
    executor_record_observer: Callable[[ExecutorCallRecord], None] | None = None,
    sequence_positions: tuple[int, ...] | None = None,
) -> ReadOnlyCollectionResult:
    if sampled_policy_step < 0 or not tasks:
        raise ValueError("diagnostic requires the complete declared panel")
    if quality_panel is not None:
        if excluded_quality_sources is None:
            raise ValueError("declare training and final-evaluation exclusions before probing")
        quality_panel.require_disjoint(excluded_quality_sources)
        if quality_panel.condition_id != condition.condition_id or tuple(
            task.task_id for task in tasks
        ) != tuple(slot.task_id for slot in quality_panel.slots):
            raise ValueError("quality collection differs from the frozen panel")
    if workflow_resources is not None and workflow_resources.binding != workflow:
        raise ValueError("collector and evaluator resources must use the same workflow")
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    if quality_panel is not None:
        (root / "panel.json").write_text(
            canonical_json(normalize_json(asdict(quality_panel))) + "\n"
        )
    trainer = replace(trainer, execution=replace(trainer.execution, experiment_id=root.name))
    (root / "condition.json").write_text(canonical_json(condition.to_value()) + "\n")
    library = SkillLibrary(library_state)
    ledger = BudgetLedger(
        run_id=root.name,
        attempt_id="collect-only",
        cap=trainer.rollout.per_rollout_maximum.scale(len(tasks)),
    )
    log = LiveAttemptEventLog(root / "events.jsonl", run_id=root.name, attempt_id="collect-only")
    emitter = RuntimeEventEmitter(log, "diagnostic-collector")
    resources = workflow_resources or RolloutWorkflowResources(workflow)
    objective = trainer.method.objective
    if objective is None:
        raise ValueError("the rollout state map is declared by the method objective")
    r2flow_runtime = (
        None
        if r2flow_rollout is None
        else r2flow_rollout.bind(
            rollout=trainer.rollout,
            generator=generator,
            ledger=ledger,
            resources=resources,
            emitter=emitter,
            **(
                {"executor_record_observer": executor_record_observer}
                if executor_record_observer is not None
                else {}
            ),
        )
    )
    subpanel: dict[str, Any] = (
        {} if sequence_positions is None else {"sequence_positions": sequence_positions}
    )
    before = generator.snapshot()
    collected = await collect_readonly_panel(
        root=root / "episodes",
        tasks=tasks,
        generator=generator,
        sessions=RetrievingRolloutSessionFactory(
            base_factory=base_sessions, retriever=TaskConditionedSkillRetriever(library=library)
        ),
        library=library_state,
        rollout=trainer.rollout,
        assembler=trainer.rollout.context_assembler(
            maximum_h0_tokens=maximum_h0_tokens,
            state_map=objective.state_map,
        ),
        epsilon_min=trainer.method.epsilon_min,
        condition_id=condition.condition_id,
        sampling_schedule_id=sampling_schedule_id,
        ordered_sequence_id=ordered_task_sequence_id,
        resources=resources,
        ledger=ledger,
        emitter=emitter,
        clock=_clock,
        chunk_size=chunk_size or len(tasks),
        schedule_purpose=schedule_purpose,
        anchor_ordinal=sampled_policy_step + 1,
        skill_executor_factory=None
        if r2flow_runtime is None
        else r2flow_runtime.skill_executor_factory,
        event_grammar=None if r2flow_runtime is None else r2flow_runtime.event_grammar,
        **subpanel,
    )
    progress = json.loads((root / "episodes" / "rollout-progress.json").read_text())
    progress["performance"] = {
        "collection_seconds": collected.elapsed_seconds,
        "panel_size": len(collected.outcomes),
    }
    (root / "rollout-progress.json").write_text(canonical_json(normalize_json(progress)) + "\n")
    if generator.snapshot() != before or library.state != library_state:
        raise ValueError("read-only collection changed its policy or skill library")
    result = ReadOnlyCollectionResult(
        condition,
        before.snapshot_id,
        collected.artifacts,
        ledger.settled,
        library_state.current_version,
        architecture_id,
        execution_controls,
    )
    (root / "collection.json").write_text(canonical_json(result.to_value()) + "\n")
    if quality_panel is not None:
        probe = collection_probe(
            result,
            panel=quality_panel,
            policy_step=sampled_policy_step,
            evidence_id=root.name,
            events_path=root / "events.jsonl",
            event_run_id=root.name,
        )
        path = root / f"probe-{sampled_policy_step:08d}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(canonical_json(normalize_json(asdict(probe))) + "\n")
        temporary.replace(path)
    return result
