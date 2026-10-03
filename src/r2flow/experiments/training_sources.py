from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from r2flow.benchmarks.training_records import TrainingRecord
from r2flow.source_keys import canonical_source_key
from skillev.contracts import JsonValue

from .bayesian_training_config import BayesianFormalConfig
from .fresh_restart import require_fresh_interface


def _coordinates(rows: object, aliases: Any) -> set[tuple[str, str]]:
    if not isinstance(rows, list) or any(
        not isinstance(row, list)
        or len(row) != 2
        or any(not isinstance(part, str) or not part.strip() for part in row)
        for row in rows
    ):
        raise ValueError("source coordinates must be benchmark/source pairs")
    return {canonical_source_key((row[0], row[1]), aliases) for row in rows}


def require_training_sources(
    path: Path,
    selected: tuple[TrainingRecord, ...],
    *,
    expected_trajectories: int,
) -> dict[str, JsonValue]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("training source declaration must be an object")
    exclusions = value.get("excluded_sources")
    if not isinstance(exclusions, dict) or any(
        name not in exclusions for name in ("iid", "development", "quality")
    ):
        raise ValueError("declare IID, development and quality exclusions explicitly")
    aliases = value.get("source_aliases", {})

    allowed = _coordinates(value.get("training"), aliases)
    excluded = set().union(*(_coordinates(rows, aliases) for rows in exclusions.values()))
    actual = {
        canonical_source_key((r.episode.benchmark.value, r.episode.source_id), aliases)
        for r in selected
    }
    if not actual or not actual <= allowed or actual & excluded or allowed & excluded:
        raise ValueError("training sources overlap excluded material or lack training provenance")
    if len(selected) != expected_trajectories:
        raise ValueError("source declaration requires the complete declared schedule")
    return cast(dict[str, JsonValue], value)


def training_source_condition(
    config: BayesianFormalConfig,
    *,
    sources: Path,
    root: Path,
    resume: Path | None,
) -> dict[str, JsonValue]:
    saved = root / "training-condition.json"
    require_fresh_interface(config)
    from .bayesian_training_config import is_r2flow_format

    r2flow = is_r2flow_format(config.format)
    if (
        config.steps,
        config.closure_steps,
        config.checkpoint_every,
        config.maximum_cycles,
    ) != (
        (
            250,
            1,
            config.r2flow.evolution.trigger.cadence_steps,
            1,
        )
        if r2flow and config.r2flow is not None
        else (250, 1, 10, 2)
    ):
        raise ValueError("training retains the full 250-step declared plan")
    value = json.loads(sources.read_text())
    if not isinstance(value, dict) or value.get("format") != "r2flow-training-sources@1":
        raise ValueError("explicit training source declaration required")
    aliases = value.get("source_aliases", {})
    exclusions = value.get("excluded_sources")
    if not isinstance(exclusions, dict) or any(
        k not in exclusions for k in ("iid", "development", "quality")
    ):
        raise ValueError("declare IID, development and quality exclusions explicitly")
    canonical = sorted(_coordinates(value.get("training"), aliases))
    excluded = {k: sorted(_coordinates(rows, aliases)) for k, rows in exclusions.items()}
    condition: dict[str, JsonValue] = {
        "training_sources": {
            "format": "r2flow-training-sources@1",
            "optimizer_steps": config.steps,
            "phase_search_steps": config.steps - config.closure_steps,
            "closure_steps": 1,
            "batch_size": config.batch_size,
            "checkpoint_every": config.checkpoint_every,
            "source_declaration": value,
            "canonical_training_allowlist": [list(row) for row in canonical],
            "canonical_excluded_sources": {
                k: [list(row) for row in rows] for k, rows in excluded.items()
            },
        }
    }
    if resume is not None:
        original_condition = json.loads(saved.read_text()) if saved.is_file() else None
        if original_condition != json.loads(json.dumps(condition)):
            raise ValueError("resume cannot add/remove/change the training source condition")
        original = root / "training-sources.json"
        if not original.is_file() or json.loads(original.read_text()) != value:
            raise ValueError("resume lacks the unchanged original training source declaration")
    return condition
