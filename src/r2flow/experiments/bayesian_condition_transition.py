from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from skillev.contracts.identity import validate_identifier
from skillev.training.inflight import durable_json

from .bayesian_training_config import R2FLOW_FORMATS, BayesianFormalConfig


def resume_condition(root: Path, target: BayesianFormalConfig) -> None:
    current = root / "condition-current.json"
    value = (
        json.loads(current.read_text())["config"]
        if current.exists()
        else json.loads((root / "formal-config.json").read_text())
    )
    if value == target.to_value():
        return
    source = BayesianFormalConfig(**value)
    if replace(source, performance_profile=target.performance_profile) == target:
        return
    raise ValueError("resume changes an undeclared condition field")


def _max_phase_steps(config: BayesianFormalConfig) -> int | None:
    return None if config.r2flow is None else config.r2flow.evolution.max_phase_steps


def phase_step_cap_start(root: Path, config: BayesianFormalConfig) -> int:
    history = condition_configs(root)
    current = history[max(history)]
    if replace(current, performance_profile=config.performance_profile) != config:
        raise ValueError("the per-phase step cap condition has not been published")
    cap = _max_phase_steps(config)
    start = max(history)
    for step in sorted(history, reverse=True):
        if _max_phase_steps(history[step]) != cap:
            break
        start = step
    return start


def _root_is_r2flow(root: Path) -> bool:
    path = root / "formal-config.json"
    return path.exists() and json.loads(path.read_text()).get("format") in R2FLOW_FORMATS


def _require_r2flow_resume(root: Path, config: BayesianFormalConfig) -> None:
    if config.format not in R2FLOW_FORMATS or not _root_is_r2flow(root):
        raise ValueError("R2 Flow resumes only an R2 Flow run root under the same condition")
    original = json.loads((root / "formal-config.json").read_text()).get("format")
    if original != config.format:
        raise ValueError("an R2 Flow run resumes only under its own formal condition")
    if (root / "declared-library-supplement.json").exists():
        raise ValueError("an R2 Flow run refuses an owner-declared library supplement")
    for pattern in ("*.retry-authorization.json", "*.revalidated.json"):
        if next(root.rglob(pattern), None) is not None:
            raise ValueError("an R2 Flow run refuses authoring retries and revalidations")


def sampling_condition(root: Path, fallback: str) -> str:
    for name in ("condition-current.json", "branch-source.json"):
        path = root / name
        if path.exists():
            value = json.loads(path.read_text()).get("sampling_condition")
            if isinstance(value, str) and value:
                return value
    return fallback


def condition_configs(root: Path) -> dict[int, BayesianFormalConfig]:
    source = BayesianFormalConfig(**json.loads((root / "formal-config.json").read_text()))
    starts = {1: source}
    current = root / "condition-current.json"
    if current.exists():
        published = json.loads(current.read_text())
        through = published["effective_from_optimizer_step"]
        declarations = [
            json.loads(path.read_text()) for path in root.glob("*condition-step-*.json")
        ]
        for row in sorted(declarations, key=lambda value: value["effective_from_optimizer_step"]):
            step = row["effective_from_optimizer_step"]
            if step > through:
                continue
            if row["source_config"] != source.to_value() or step != row["saved_optimizer_step"] + 1:
                raise ValueError("condition history differs from the saved continuation chain")
            source = BayesianFormalConfig(**row["config"])
            starts[step] = source
        if published["config"] != source.to_value():
            raise ValueError("published condition is missing its history")
    return starts


def observer_condition_starts(root: Path, target: BayesianFormalConfig) -> dict[int, str]:
    starts = condition_configs(root)
    if replace(starts[max(starts)], performance_profile=target.performance_profile) != target:
        raise ValueError("observer condition has not been published")
    return {step: config.condition for step, config in starts.items()}


def observer_batch_size_starts(root: Path, target: BayesianFormalConfig) -> dict[int, int]:
    starts = condition_configs(root)
    if replace(starts[max(starts)], performance_profile=target.performance_profile) != target:
        raise ValueError("observer batch-size condition has not been published")
    return {step: config.batch_size for step, config in starts.items()}


def _bind_run_directory(root: Path, config: BayesianFormalConfig, resume: Path | None) -> None:
    validate_identifier(root.name)
    if resume is None:
        root.mkdir(parents=True, mode=0o700, exist_ok=False)
        durable_json(root / "formal-config.json", config.to_value())
        if config.r2flow is not None:
            durable_json(root / "r2flow-run.json", config.r2flow.to_value())
        return
    if _root_is_r2flow(root) or config.format in R2FLOW_FORMATS:
        _require_r2flow_resume(root, config)
    resume_condition(root, config)
