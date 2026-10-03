from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from r2flow.benchmarks.training_records import TrainingRecord
from r2flow.benchmarks.training_schedule import (
    training_trajectories,
)
from skillev.training.checkpoint import FilesystemTrainingCheckpointStore

from .bayesian_condition_transition import condition_configs
from .bayesian_training_config import BayesianFormalConfig


def training_schedule(
    config: BayesianFormalConfig,
    records: tuple[TrainingRecord, ...],
    *,
    root: Path,
    resume: Path | None = None,
) -> tuple[TrainingRecord, ...]:
    starts = {1: config.scheduled_domains}
    if (root / "formal-config.json").exists():
        history = condition_configs(root)
        matching = [
            step
            for step, value in history.items()
            if replace(value, performance_profile=config.performance_profile) == config
        ]
        if matching:
            history = {step: value for step, value in history.items() if step <= max(matching)}
        starts = {step: value.scheduled_domains for step, value in history.items()}
        if starts[max(starts)] != config.scheduled_domains:
            if resume is None:
                raise ValueError("domain continuation needs a complete checkpoint")
            saved = FilesystemTrainingCheckpointStore(root=root / "checkpoints").load_metadata(
                resume
            )
            starts[saved.optimizer_step + 1] = config.scheduled_domains
    return training_trajectories(records, steps=config.steps, domain_starts=starts)


def schedule_summary(
    config: BayesianFormalConfig, selected: tuple[TrainingRecord, ...]
) -> dict[str, Any]:
    sizes = Counter(r.episode.optimizer_step for r in selected)
    segments: list[dict[str, int]] = []
    for step, size in sorted(sizes.items()):
        if not segments or segments[-1]["batch_size"] != size:
            segments.append({"first_step": step, "last_step": step, "batch_size": size})
        else:
            segments[-1]["last_step"] = step
    return {
        **config.schedule_summary(),
        "trajectories": len(selected),
        "question_occurrences": len(selected) // 4,
        "batch_size_segments": segments,
    }
