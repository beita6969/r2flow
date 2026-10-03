from __future__ import annotations

import json
import math
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from skillev.diagnostics.rollout_progress import phase_summary, server_metrics
from skillev.runtime.request_journal import read_connection


def _number(value: object) -> int | float | None:
    return (
        value
        if isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
        else None
    )


def _decode(value: bytes | None) -> Any:
    if value is None:
        return None
    result = json.loads(zlib.decompress(value))
    return json.loads(result) if isinstance(result, str) else result


def _journal_rows(
    path: Path | None, episodes: list[str], tasks: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if path is None or not path.is_file():
        return [], []
    actors, judges = [], []
    with read_connection(path) as db:
        placeholders = ",".join("?" for _ in episodes)
        task_places = ",".join("?" for _ in tasks) or "NULL"
        cursor = db.execute(
            f"SELECT rowid,identity,payload,state,response FROM requests WHERE "
            f"json_extract(identity,'$[0]') IN ({placeholders}) OR "
            "(json_extract(identity,'$[0]')='judge' AND "
            f"json_extract(identity,'$[1]') IN ({task_places}))",
            [*episodes, *tasks],
        )
        seen: dict[str, tuple[Any, ...]] = {}
        for rowid, identity, payload, state, response in cursor:
            original = (payload, state, response)
            if identity in seen:
                if seen[identity] != original:
                    raise ValueError("conflicting original journal rows for one request")
                continue
            seen[identity] = original
            coordinate = json.loads(identity)
            received = _decode(response)
            row = {"rowid": rowid, "coordinate": coordinate, "state": state}
            if len(coordinate) == 6 and coordinate[2] in {"reasoning", "action"}:
                request = _decode(payload)
                inputs = request.get("input_ids") if isinstance(request, dict) else None
                outputs = received.get("output_ids") if isinstance(received, dict) else None
                row.update(
                    input_tokens=len(inputs) if isinstance(inputs, list) else None,
                    output_tokens=len(outputs) if isinstance(outputs, list) else None,
                    measurements=server_metrics(received.get("meta_info"))
                    if isinstance(received, dict)
                    else {},
                )
                actors.append(row)
            elif len(coordinate) == 3 and coordinate[0] == "judge":
                row["usage"] = received.get("usage", {}) if isinstance(received, dict) else {}
                judges.append(row)
    return actors, judges


def committed_request_accounting(
    event: dict[str, Any],
    *,
    rollout_detail: dict[str, Any] | None = None,
    journal: Path | None = None,
) -> dict[str, Any]:
    if event.get("event_type") != "training_step_committed":
        raise ValueError("request accounting requires a committed training source")
    payload = event["payload"]
    records = {r["trajectory_id"]: r for r in payload["records"]}
    if not records or len(records) != len(payload["records"]):
        raise ValueError("committed trajectory population must be nonempty and unique")
    tasks = {r.get("initial_context", {}).get("meta", {}).get("task_id") for r in records.values()}
    tasks.discard(None)
    actor_rows, judge_rows = _journal_rows(journal, list(records), sorted(tasks))
    phases: dict[tuple[str, int, str], dict[str, Any]] = {}
    fields = tuple(
        cast(dict[str, Any], phase_summary([{"phase": "reasoning"}])["reasoning"])["measurements"]
    )
    for episode in (rollout_detail or {}).get("trajectories", []):
        identity = episode.get("trajectory_id")
        if identity not in records:
            raise ValueError("phase detail includes a nonmember trajectory")
        if (
            episode.get("batch_id"),
            episode.get("policy_snapshot_id"),
            episode.get("library_version"),
        ) != (payload["batch_id"], payload["policy_snapshot_before"], payload["library_version"]):
            raise ValueError("phase detail differs from committed batch/policy/library")
        for phase in episode.get("phases", []):
            name, turn = phase.get("phase"), phase.get("turn_index")
            if name not in {"reasoning", "action"} or type(turn) is not int:
                continue
            key = identity, turn, name
            row: dict[str, Any] = {field: _number(phase.get(field)) for field in fields}
            row.update(
                trajectory_id=identity,
                turn=turn,
                phase=name,
                measured_phase=True,
                server_request_id=phase.get("server_request_id")
                if isinstance(phase.get("server_request_id"), str)
                else None,
            )
            if key in phases and phases[key] != row:
                raise ValueError("conflicting phase measurements for one logical request")
            phases[key] = row
    for saved in actor_rows:
        episode, turn, phase, policy, library, decoding = saved["coordinate"]
        record = records[episode]
        if [policy, library, decoding] != [
            payload["policy_snapshot_before"],
            payload["library_version"],
            record["decoding_snapshot_id"],
        ]:
            raise ValueError("actor journal crosses committed policy/library/decoding")
        key = episode, int(turn), phase
        row = phases.setdefault(
            key,
            {
                "trajectory_id": episode,
                "turn": int(turn),
                "phase": phase,
                "measured_phase": False,
                **dict.fromkeys(fields),
            },
        )
        row["journal_reference"] = {"file": str(journal), "source_rowid": saved["rowid"]}
        row["journal_state"] = saved["state"]
        if saved["state"] == "COMPLETED":
            measured = saved["measurements"]
            for field in fields:
                if row.get(field) is None and field in measured:
                    row[field] = measured[field]
            for target in ("input_tokens", "output_tokens"):
                if row.get(target) is None:
                    row[target] = saved[target]
            row["server_request_id"] = measured.get("server_request_id")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trajectory_counts: dict[str, int] = defaultdict(int)
    for record in records.values():
        benchmark = (
            record.get("reward", {}).get("native_payload", {}).get("benchmark_id") or "unknown"
        )
        trajectory_counts[benchmark] += 1
    for row in phases.values():
        benchmark = (
            records[row["trajectory_id"]]
            .get("reward", {})
            .get("native_payload", {})
            .get("benchmark_id")
            or "unknown"
        )
        row["benchmark_id"] = benchmark
        groups[benchmark].append(row)
    domains = []
    for benchmark, count in trajectory_counts.items():
        rows = groups[benchmark]
        domains.append(
            {
                "benchmark_id": benchmark,
                "role": "actor",
                "trajectory_count": count,
                "logical_phase_count": sum(r["measured_phase"] for r in rows)
                if rollout_detail is not None
                else None,
                "distinct_journal_request_count": sum("journal_reference" in r for r in rows)
                if journal is not None and journal.is_file()
                else None,
                "physical_http_attempt_count": None,
                "phases": phase_summary(rows) if rows else None,
                "summary_scope": "observed logical phases and journal rows, not unknown requests",
            }
        )
        for role in ("terminal-judge", "skill-author"):
            domains.append(
                {
                    "benchmark_id": benchmark,
                    "role": role,
                    "request_count": None,
                    "seconds": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "cached_tokens": None,
                    "scope": "no exact committed role-request binding",
                }
            )
    candidates = []
    for row in judge_rows:
        usage: dict[str, Any] = row["usage"]
        if not isinstance(usage, dict):
            usage = {}
        details = usage.get("prompt_tokens_details", {})
        candidates.append(
            {
                "role": "terminal-judge",
                "reference": {"file": str(journal), "source_rowid": row["rowid"]},
                "state": row["state"],
                "input_tokens": _number(usage.get("prompt_tokens")),
                "output_tokens": _number(usage.get("completion_tokens")),
                "cached_tokens": _number(details.get("cached_tokens"))
                if isinstance(details, dict)
                else None,
                "seconds": None,
                "assigned_optimizer_step": None,
            }
        )
    return {
        "format": "committed-request-accounting@1",
        "run_id": event["run_id"],
        "source_commit_id": event["event_id"],
        "optimizer_step": payload["optimizer_step"],
        "policy_snapshot_id": payload["policy_snapshot_before"],
        "library_version": payload["library_version"],
        "domains": domains,
        "actor_requests": list(phases.values()),
        "unassigned_same_task_judge_rows": candidates,
        "uncovered": [
            "judge identity contains task plus private payload, not trajectory/policy/library; "
            "same-task rows are not step costs",
            "author rows need committed evolution reservation linkage, "
            "not an adjacent training step",
            "nonstreaming HTTP duration is not TTFT or pure decode latency",
            "logical phases and durable request rows do not count "
            "replayed/retried physical HTTP attempts",
        ],
    }
