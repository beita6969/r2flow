from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from r2flow.benchmarks.training_records import TrainingRecord
from r2flow.benchmarks.training_schedule import TRAINING_DOMAINS
from r2flow.source_keys import canonical_source_key
from skillev.contracts.action_wire import NATIVE_EVENT_CALL_WIRE
from skillev.contracts.skill_exposure import SKILL_EXPOSURE

if TYPE_CHECKING:
    from .bayesian_training_config import BayesianFormalConfig

PROTOCOL = "autonomous-ttb@1"
SOURCE_FORMAT = "public-task-needs@1"
SOURCE_ROLES = ("direct-control", "procedure-applicable", "exploration")


def require_autonomous_config(config: BayesianFormalConfig) -> None:
    if config.learning_protocol != PROTOCOL:
        raise ValueError("unknown declared learning protocol")
    from .bayesian_training_config import is_current_candidate

    if (
        not is_current_candidate(config.format)
        or not config.phase_context
        or not config.reasoning_tool_catalog
        or config.skill_exposure != SKILL_EXPOSURE
        or config.action_wire != NATIVE_EVENT_CALL_WIRE
    ):
        raise ValueError("autonomous TTB requires the declared visible native catalog interface")
    if config.steps - config.closure_steps < 2 * config.window:
        raise ValueError("declare a bounded real TTB budget covering both diagnostic windows")


def require_autonomous_initialization(config: BayesianFormalConfig, preparation: Path) -> None:
    if config.learning_protocol != PROTOCOL:
        return
    raw = json.loads(preparation.read_text(encoding="utf-8"))
    if raw.get("format") != "r2flow-initial-preparation@1" or "initialization" in raw:
        raise ValueError("autonomous TTB starts from the saved pre-skill-SFT initialization")


def _source(row: Any, aliases: Any) -> tuple[str, str]:
    if (
        not isinstance(row, dict)
        or not isinstance(row.get("benchmark"), str)
        or not isinstance(row.get("source_id"), str)
        or not row["source_id"].strip()
    ):
        raise ValueError("source rows require benchmark and canonical source_id")
    key = row["benchmark"], row["source_id"]
    if canonical_source_key(key, aliases) != key:
        raise ValueError("freeze canonical source coordinates, not population aliases")
    return key


def autonomous_training_sources(
    config: BayesianFormalConfig,
    records: tuple[TrainingRecord, ...],
    data_condition: dict[str, Any] | None,
) -> tuple[TrainingRecord, ...]:
    if config.learning_protocol != PROTOCOL:
        return records
    require_autonomous_config(config)
    declaration = (data_condition or {}).get("autonomous_ttb_sources")
    if not isinstance(declaration, dict) or declaration.get("format") != SOURCE_FORMAT:
        raise ValueError("declare the public-needs source schedule before autonomous sampling")
    aliases = declaration.get("source_aliases", {})
    if not isinstance(aliases, dict) or any(
        not isinstance(mapping, dict)
        or any(
            not isinstance(k, str) or not isinstance(v, str) or mapping.get(v, v) != v
            for k, v in mapping.items()
        )
        for mapping in aliases.values()
    ):
        raise ValueError("source aliases must resolve directly to canonical identities")
    exclusions = declaration.get("excluded_sources")
    if not isinstance(exclusions, dict) or not {"iid", "development", "quality"} <= set(exclusions):
        raise ValueError("keep IID, development and quality source exclusions explicit")
    excluded = set()
    for rows in exclusions.values():
        if not isinstance(rows, list):
            raise ValueError("excluded source groups must be arrays")
        for row in rows:
            if (
                not isinstance(row, list | tuple)
                or len(row) != 2
                or any(not isinstance(part, str) or not part.strip() for part in row)
            ):
                raise ValueError("excluded coordinates must be benchmark/source pairs")
            excluded.add(canonical_source_key((row[0], row[1]), aliases))
    rows = declaration.get("ordered_sources")
    if not isinstance(rows, list) or not rows:
        raise ValueError("declare the complete nonempty source order before collecting outcomes")
    dropped = {
        canonical_source_key((domain, source), aliases)
        for domain, source in getattr(config, "training_source_exclusions", ())
    }
    if dropped:
        declared = {_source(row, aliases) for row in rows}
        if not dropped <= declared:
            raise ValueError(
                "a declared training-source exclusion is not a declared training source: "
                f"{sorted(dropped - declared)}"
            )
        rows = [row for row in rows if _source(row, aliases) not in dropped]
    available: dict[tuple[str, str], TrainingRecord] = {}
    for record in records:
        key = canonical_source_key(
            (record.episode.benchmark.value, record.episode.source_id), aliases
        )
        if key in available:
            previous = available[key]
            if (
                replace(record.input, task_id=previous.input.task_id) != previous.input
                or record.output != previous.output
            ):
                raise ValueError("one canonical source has conflicting public/scoring inputs")
            continue
        available[key] = record
    selected = []
    seen = set()
    roles: Counter[str] = Counter()
    for row in rows:
        key = _source(row, aliases)
        if key in seen or key in excluded or key not in available:
            raise ValueError("declared source is duplicate, held out, or unavailable")
        if (
            row.get("role") not in SOURCE_ROLES
            or not isinstance(row.get("public_basis"), str)
            or not row["public_basis"].strip()
        ):
            raise ValueError("each source needs a pre-outcome public applicability basis")
        if row["role"] == "procedure-applicable" and (
            not isinstance(row.get("method_family"), str) or not row["method_family"].strip()
        ):
            raise ValueError("applicable tasks must name the reusable method, not a success label")
        selected.append(available[key])
        seen.add(key)
        roles[row["role"]] += 1
    if not set(config.scheduled_domains) <= {r.episode.benchmark for r in selected}:
        raise ValueError("the fixed source plan must contain every active training domain")
    if not roles["direct-control"] or not roles["procedure-applicable"]:
        raise ValueError("retain both reasonable direct tasks and publicly applicable tasks")
    return tuple(selected)


def source_coverage_report(
    data_condition: dict[str, Any],
    training_source_exclusions: Sequence[Sequence[str]] = (),
) -> dict[str, Any]:
    declaration = data_condition["autonomous_ttb_sources"]
    rows = declaration["ordered_sources"]
    aliases = declaration.get("source_aliases", {})
    dropped = {
        canonical_source_key((domain, source), aliases)
        for domain, source in training_source_exclusions
    }
    if dropped:
        rows = [row for row in rows if _source(row, aliases) not in dropped]
    return {
        "selection_basis": SOURCE_FORMAT,
        "source_count": len(rows),
        "roles": dict(Counter(row["role"] for row in rows)),
        "domains": {
            domain.value: dict(
                Counter(row["role"] for row in rows if row["benchmark"] == domain.value)
            )
            for domain in TRAINING_DOMAINS
        },
        **(
            {"training_source_exclusions": [list(pair) for pair in sorted(dropped)]}
            if dropped
            else {}
        ),
        "skill_benefit_established": False,
        "outcome_filtering": False,
        "extra_training_objective": False,
    }
