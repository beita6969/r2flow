from __future__ import annotations

import argparse
import json
import zlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from skillev.contracts import JsonValue, canonical_json
from skillev.runtime.request_journal import read_connection

from .inflight import durable_json


def _requests(path: Path | None, episode: str) -> list[dict[str, Any]] | None:
    if path is None or not path.is_file():
        return None
    with read_connection(path) as db:
        found = db.execute(
            "SELECT identity,payload,state,status,response FROM requests "
            "WHERE json_extract(identity,'$[0]')=? ORDER BY identity",
            (episode,),
        ).fetchall()
        rows = []
        for identity, payload, state, status, response in found:
            coordinate = json.loads(identity)
            if len(coordinate) != 6 or coordinate[2] not in ("reasoning", "action"):
                continue
            request = json.loads(zlib.decompress(payload))
            received = json.loads(zlib.decompress(response)) if response is not None else None
            rows.append(
                {
                    "identity": coordinate,
                    "request": request,
                    "state": state,
                    "status": status,
                    "response": received,
                }
            )
        return rows


def read_event_rows(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        return
    with path.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line or not line.endswith(b"\n"):
                break
            yield offset, json.loads(line)


def _action_references(
    events: Path,
    *,
    run_id: str,
    records: list[dict[str, Any]],
    event_rows: Iterable[tuple[int, dict[str, Any]]] | None = None,
) -> dict[tuple[str, int], dict[str, Any]]:
    expected = {
        (r["trajectory_id"], turn): edge for r in records for turn, edge in enumerate(r["steps"], 1)
    }
    found: dict[tuple[str, int], dict[str, Any]] = {}
    for offset, event in read_event_rows(events) if event_rows is None else event_rows:
        if event.get("run_id") != run_id or event.get("event_type") != "agent_step_recorded":
            continue
        payload = event["payload"]
        key = payload.get("trajectory_id"), payload.get("turn")
        edge = expected.get(key)
        if edge is None or "assessment" not in payload:
            continue
        if not isinstance(edge.get("action_token_ids"), list) or not isinstance(
            edge.get("observation_text"), str
        ):
            continue
        if payload.get("action_token_ids") != edge.get("action_token_ids") or payload.get(
            "observation_text"
        ) != edge.get("observation_text"):
            continue
        if key in found:
            if found[key]["assessment"] != payload["assessment"]:
                raise ValueError("conflicting original execution assessments")
            continue
        found[key] = {
            "file": str(events),
            "byte_offset": offset,
            "event_id": event["event_id"],
            "assessment": payload["assessment"],
        }
    return found


def index_committed_event(
    event: dict[str, Any],
    *,
    events: Path,
    event_offset: int | None,
    inflight: Path,
    requests: Path | None,
    condition_id: str,
    code_revision: str | None = None,
    expected_batch_size: int | None = None,
    related_event_rows: Iterable[tuple[int, dict[str, Any]]] | None = None,
) -> dict[str, JsonValue]:
    if event.get("event_type") != "training_step_committed":
        raise ValueError("evidence index requires an authoritative committed event")
    payload = event["payload"]
    step = payload["optimizer_step"]
    records = payload["records"]
    ids = [record["trajectory_id"] for record in records]
    if len(set(ids)) != len(ids) or not records:
        raise ValueError("committed evidence contains duplicate or empty trajectory population")
    if expected_batch_size is not None and len(records) != expected_batch_size:
        raise ValueError("committed evidence differs from declared batch size")
    directory = inflight / f"step-{step:08d}"
    action_refs = _action_references(
        events, run_id=event["run_id"], records=records, event_rows=related_event_rows
    )
    issues: list[JsonValue] = []
    trajectories: list[JsonValue] = []
    for position, record in enumerate(records, 1):
        path = directory / f"trajectory-{position:06d}.json"
        missing: list[JsonValue] = []
        artifact = None
        try:
            artifact = json.loads(path.read_text())["artifact"]
        except FileNotFoundError:
            missing.append({"status": "RAW_MISSING", "object": str(path)})
        if artifact is not None:
            manifest = artifact["manifest"]
            if (
                artifact["record"] != record
                or manifest["policy_snapshot"]["snapshot_id"] != payload["policy_snapshot_before"]
                or manifest["library_version"] != payload["library_version"]
            ):
                raise ValueError("original artifact and committed source identity/content differ")
        horizon = len(record["steps"])
        raw_requests = _requests(requests, record["trajectory_id"])
        request_rows: list[JsonValue] = []
        by_phase: dict[tuple[str, str], dict[str, Any]] = {}
        for saved_row in raw_requests or []:
            key = saved_row["identity"][1], saved_row["identity"][2]
            if key in by_phase:
                raise ValueError("multiple request responses target one committed phase")
            by_phase[key] = saved_row
        for turn in range(1, horizon + 1):
            for phase in ("reasoning", "action"):
                row = by_phase.get((str(turn), phase))
                location = {
                    "file": str(requests) if requests is not None else None,
                    "trajectory_id": record["trajectory_id"],
                    "turn": turn,
                    "phase": phase,
                }
                status = "RAW_MISSING"
                input_count = output_count = finish = None
                request_identity = output_cap = None
                if row is not None:
                    identity = row["identity"]
                    request_identity = identity
                    output_cap = row["request"].get("sampling_params", {}).get("max_new_tokens")
                    if identity[3:] != [
                        payload["policy_snapshot_before"],
                        payload["library_version"],
                        record["decoding_snapshot_id"],
                    ]:
                        raise ValueError(
                            "request journal snapshot/library/decoding differs from commit"
                        )
                    response = row["response"]
                    inputs = row["request"].get("input_ids")
                    outputs = response.get("output_ids") if isinstance(response, dict) else None
                    if (
                        row["state"] == "COMPLETED"
                        and isinstance(inputs, list)
                        and isinstance(outputs, list)
                    ):
                        status = "RAW_AVAILABLE"
                        input_count, output_count = len(inputs), len(outputs)
                        meta = response.get("meta_info", {})
                        finish = meta.get("finish_reason") if isinstance(meta, dict) else None
                    elif row["state"] == "DISPATCHED":
                        status = "OUTCOME_UNKNOWN"
                request_rows.append(
                    {
                        **location,
                        "status": status,
                        "input_token_count": input_count,
                        "raw_output_token_count": output_count,
                        "finish_reason": finish,
                        "request_identity": request_identity,
                        "max_new_tokens": output_cap,
                    }
                )
                if status != "RAW_AVAILABLE":
                    missing.append({"status": status, "object": location})
        action_rows: list[JsonValue] = []
        for turn in range(1, horizon + 1):
            found = action_refs.get((record["trajectory_id"], turn))
            reference = {k: v for k, v in found.items() if k != "assessment"} if found else None
            action_rows.append(
                {
                    "turn": turn,
                    "assessment_reference": reference,
                    "record_step_reference": f"record.steps[{turn - 1}]",
                }
            )
            if found is None:
                missing.append(
                    {
                        "status": "RAW_MISSING",
                        "object": {
                            "file": str(events),
                            "trajectory_id": record["trajectory_id"],
                            "turn": turn,
                            "record": "agent_step_recorded.assessment",
                        },
                    }
                )
        source = record["reward"].get("native_payload", {}).get("training_evidence_source")
        coordinates = (
            {
                name: source.get(name)
                for name in ("benchmark_id", "population_id", "source_question_id", "occurrence_id")
            }
            if isinstance(source, dict)
            else None
        )
        trajectories.append(
            {
                "position": position,
                "trajectory_id": record["trajectory_id"],
                "source": coordinates,
                "sampling_coordinate": artifact["manifest"].get("sampling_coordinate")
                if artifact
                else None,
                "artifact": str(path),
                "action_count": horizon,
                "terminal_result_count": 1,
                "terminal_verifier_version": record["reward"].get("verifier_version"),
                "record_reference": {
                    "file": str(events),
                    "byte_offset": event_offset,
                    "record_position": position,
                },
                "request_records": request_rows,
                "action_records": action_rows,
                "issues": missing,
            }
        )
        issues.extend(missing)
    return {
        "format": "committed-batch-evidence-index@1",
        "run_id": event["run_id"],
        "condition_id": condition_id,
        "code_revision": code_revision,
        "source_commit_id": event["event_id"],
        "optimizer_step": step,
        "sampled_policy_step": step - 1,
        "policy_snapshot_id": payload["policy_snapshot_before"],
        "library_version": payload["library_version"],
        "committed_at": event["occurred_at"],
        "source_event": {"file": str(events), "byte_offset": event_offset},
        "score_projection_reference": "source_event.payload.edge_records/stats/posterior_batch",
        "trajectory_count": len(records),
        "action_count": sum(len(r["steps"]) for r in records),
        "trajectories": trajectories,
        "status": "RAW_AVAILABLE" if not issues else "RAW_MISSING",
        "issue_count": len(issues),
        "mirror_status": "not-observed",
    }


def write_committed_index(index: dict[str, JsonValue], root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    step = index["optimizer_step"]
    if type(step) is not int:
        raise ValueError("index requires a committed optimizer step")
    path = root / f"step-{step:08d}.json"
    if path.exists():
        if json.loads(path.read_text()) != index:
            raise ValueError("existing committed evidence index differs")
    else:
        durable_json(path, index)
    queue = root / "mirror-queue"
    queue.mkdir(exist_ok=True, mode=0o700)
    if not (queue / path.name).exists():
        durable_json(
            queue / path.name,
            {
                "format": "evidence-mirror-request@1",
                "index_file": str(path),
                "source_commit_id": index["source_commit_id"],
                "state": "pending",
            },
        )
    return path


def mirror_backlog_status(
    root: Path,
    *,
    max_pending_steps: int | None = None,
    minimum_free_bytes: int | None = None,
    observed_free_bytes: int | None = None,
) -> dict[str, JsonValue]:
    for bound in (max_pending_steps, minimum_free_bytes, observed_free_bytes):
        if bound is not None and (type(bound) is not int or bound < 0):
            raise ValueError("mirror risk bounds and measured bytes must be nonnegative")
    pending: list[JsonValue] = []
    for path in sorted((root / "mirror-queue").glob("step-*.json")):
        row = json.loads(path.read_text())
        if row.get("format") != "evidence-mirror-request@1" or row.get("state") not in {
            "pending",
            "copying",
            "failed",
            "mirrored",
        }:
            raise ValueError("unknown mirror receipt")
        if row["state"] != "mirrored":
            pending.append(path.name)
    reasons: list[JsonValue] = []
    if max_pending_steps is not None and len(pending) > max_pending_steps:
        reasons.append("declared-pending-step-limit")
    if minimum_free_bytes is not None:
        if observed_free_bytes is None:
            reasons.append("free-space-not-observed")
        elif observed_free_bytes < minimum_free_bytes:
            reasons.append("declared-free-space-limit")
    return {
        "format": "evidence-mirror-backlog@1",
        "pending_step_count": len(pending),
        "pending_indexes": pending,
        "observed_free_bytes": observed_free_bytes,
        "pause_before_next_batch": bool(reasons),
        "reasons": reasons,
        "scope": "asynchronous-mirror-not-local-commit",
    }


def export_committed_step_evidence(
    *,
    events: Path,
    inflight: Path,
    requests: Path | None,
    condition_id: str,
    first_step: int,
    last_step: int,
    code_revision: str | None = None,
    expected_batch_size: int | None = None,
) -> dict[str, JsonValue]:
    if first_step < 1 or last_step < first_step:
        raise ValueError("evidence export requires a positive committed step range")
    indexed = {}
    if not events.is_file():
        return {
            "format": "committed-evidence-export@1",
            "steps": [
                {
                    "optimizer_step": step,
                    "status": "RAW_MISSING",
                    "missing_object": str(events),
                    "reason": "source-events-not-readable",
                }
                for step in range(first_step, last_step + 1)
            ],
        }
    runs = set()
    with events.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line or not line.endswith(b"\n"):
                break
            event = json.loads(line)
            if event.get("event_type") != "training_step_committed":
                continue
            runs.add(event["run_id"])
            if len(runs) > 1:
                raise ValueError("evidence export cannot combine source runs")
            step = event["payload"]["optimizer_step"]
            if not first_step <= step <= last_step:
                continue
            if step in indexed:
                raise ValueError("multiple source commits target the exported step")
            indexed[step] = index_committed_event(
                event,
                events=events,
                event_offset=offset,
                inflight=inflight,
                requests=requests,
                condition_id=condition_id,
                code_revision=code_revision,
                expected_batch_size=expected_batch_size,
            )
    return {
        "format": "committed-evidence-export@1",
        "steps": [
            indexed.get(
                step,
                {
                    "optimizer_step": step,
                    "status": "RAW_MISSING",
                    "missing_object": str(events),
                    "reason": "no-complete-source-commit-observed",
                },
            )
            for step in range(first_step, last_step + 1)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Private locators for original committed evidence, never representative cases."
    )
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--inflight", required=True, type=Path)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--code-revision")
    parser.add_argument("--first-step", required=True, type=int)
    parser.add_argument("--last-step", required=True, type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = export_committed_step_evidence(
        events=args.events,
        inflight=args.inflight,
        requests=args.requests,
        condition_id=args.condition_id,
        code_revision=args.code_revision,
        first_step=args.first_step,
        last_step=args.last_step,
        expected_batch_size=args.batch_size,
    )
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(result) + "\n")


if __name__ == "__main__":
    main()
