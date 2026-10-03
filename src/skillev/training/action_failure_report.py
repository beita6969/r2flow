from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from skillev.diagnostics.action_failures import ACTION_FAILURE_FORMAT, classify_action_outcome
from skillev.rollout.codec import codec_for_initial_meta
from skillev.rollout.native_wire import NativeToolWire


def committed_action_outcomes(
    event: dict[str, Any],
    *,
    source_events: Iterable[dict[str, Any]] = (),
    evidence_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if event.get("event_type") != "training_step_committed":
        raise ValueError("action outcome report requires committed records")
    assessments: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for original in source_events:
        if (
            original.get("run_id") != event["run_id"]
            or original.get("event_type") != "agent_step_recorded"
        ):
            continue
        p = original["payload"]
        if "assessment" in p:
            assessments.setdefault((p["trajectory_id"], p["turn"]), []).append(p)
    requests = {}
    if evidence_index is not None:
        if (
            evidence_index.get("source_commit_id") != event["event_id"]
            or evidence_index.get("run_id") != event["run_id"]
        ):
            raise ValueError("request index belongs to another committed batch")
        requests = {
            (t["trajectory_id"], r["turn"]): r
            for t in evidence_index["trajectories"]
            for r in t["request_records"]
            if r["phase"] == "action"
        }
    rows = []
    for record in event["payload"]["records"]:
        meta = record.get("initial_context", {}).get("meta")
        codec = codec_for_initial_meta(meta) if isinstance(meta, dict) else None
        for turn, step in enumerate(record["steps"], 1):
            key = record["trajectory_id"], turn
            matched = [
                p["assessment"]
                for p in assessments.get(key, [])
                if isinstance(step.get("action_token_ids"), list)
                and isinstance(step.get("observation_text"), str)
                and p.get("action_token_ids") == step["action_token_ids"]
                and p.get("observation_text") == step["observation_text"]
            ]
            if matched and any(m != matched[0] for m in matched):
                raise ValueError("conflicting committed action assessments")
            assessment = matched[0] if matched else {}
            raw = step.get("action_text")
            parsed = codec.parse(raw) if codec is not None and isinstance(raw, str) else None
            request = requests.get(key, {})
            finish = request.get("finish_reason")
            if isinstance(finish, dict):
                finish = finish.get("type")
            completed = None
            if all(
                type(assessment.get(k)) is bool
                for k in ("accepted_submission", "environment_terminal")
            ):
                completed = assessment["accepted_submission"] or assessment["environment_terminal"]
            labels = classify_action_outcome(
                raw,
                parse_status=parsed.status.value if parsed else assessment.get("parse_status"),
                finish_reason=finish,
                action_kind=parsed.action.kind.value if parsed and parsed.action else None,
                completed=completed,
                observation_status=step.get("observation_status"),
                action_wire=codec.format_version if codec else None,
                public_error_code=parsed.public_error_code if parsed else None,
                available_native_names=tuple(b.name for b in codec.bindings)
                if isinstance(codec, NativeToolWire)
                else None,
                admitted=assessment.get("admitted"),
                executed=assessment.get("executed"),
                terminal_success=record.get("reward", {}).get("success"),
                output_token_count=request.get("raw_output_token_count"),
            )
            rows.append(
                {
                    "trajectory_id": key[0],
                    "turn": turn,
                    "labels": list(labels),
                    "action_wire": codec.format_version if codec else None,
                    "admitted": assessment.get("admitted"),
                    "executed": assessment.get("executed"),
                    "finish_reason": finish,
                    "terminal_success": record.get("reward", {}).get("success"),
                }
            )
    return {
        "format": ACTION_FAILURE_FORMAT,
        "source_commit_id": event["event_id"],
        "run_id": event["run_id"],
        "optimizer_step": event["payload"]["optimizer_step"],
        "actions": rows,
        "label_counts": {
            label: sum(label in r["labels"] for r in rows)
            for label in sorted({label for r in rows for label in r["labels"]})
        },
        "scope": "nonexclusive observations; terminal failure not per-action causal attribution",
    }
