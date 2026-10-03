from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from typing import cast

from skillev.contracts import JsonValue

from .metrics_contract import TrainingMetricsSnapshot, _object, _objects, _text


class ActionMetrics:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        connection.execute(
            "CREATE TABLE IF NOT EXISTS action_assessments "
            "(run TEXT, trajectory TEXT, turn INTEGER, payload TEXT, "
            "PRIMARY KEY(run, trajectory, turn, payload))"
        )

    def observe(self, event: dict[str, JsonValue]) -> None:
        payload = _object(event["payload"])
        if "assessment" not in payload:
            return
        row = {
            name: payload[name] for name in ("action_token_ids", "observation_text", "assessment")
        }
        turn = payload["turn"]
        if type(turn) is not int or turn < 1:
            raise ValueError("action assessment requires a positive turn")
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO action_assessments VALUES (?, ?, ?, ?)",
                (
                    _text(event["run_id"]),
                    _text(payload["trajectory_id"]),
                    turn,
                    json.dumps(row, sort_keys=True, allow_nan=False),
                ),
            )

    def enrich(
        self, snapshot: TrainingMetricsSnapshot, event: dict[str, JsonValue]
    ) -> TrainingMetricsSnapshot:
        metrics = dict(snapshot.metrics)
        metrics.update(
            self.for_records(snapshot.run_id, _objects(_object(event["payload"])["records"]))
        )
        return replace(snapshot, metrics=metrics)

    def for_records(self, run_id: str, records: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
        assessments: list[dict[str, JsonValue]] = []
        terminal_records: list[bool | None] = []
        action_count = sum(len(_objects(record["steps"])) for record in records)
        for record in records:
            record_assessments: list[dict[str, JsonValue]] = []
            for index, step in enumerate(_objects(record["steps"]), start=1):
                matches = []
                for (raw,) in self.connection.execute(
                    "SELECT payload FROM action_assessments "
                    "WHERE run=? AND trajectory=? AND turn=?",
                    (run_id, _text(record["trajectory_id"]), index),
                ):
                    row = json.loads(raw)
                    if row["action_token_ids"] == step.get("action_token_ids") and row[
                        "observation_text"
                    ] == step.get("observation_text"):
                        matches.append(_object(row["assessment"]))
                if len(matches) > 1:
                    raise ValueError(
                        "the same committed edge has conflicting execution assessments"
                    )
                if matches:
                    assessments.append(matches[0])
                    record_assessments.append(matches[0])
            terminal_records.append(
                any(
                    row.get("accepted_submission") is True
                    or row.get("environment_terminal") is True
                    for row in record_assessments
                )
                if record_assessments and len(record_assessments) == len(_objects(record["steps"]))
                else None
            )
        complete = bool(action_count) and len(assessments) == action_count
        metrics: dict[str, JsonValue] = {}
        metrics["assessed_action_count"] = len(assessments)
        counts = {
            "admitted_count": "admitted",
            "executed_count": "executed",
            "accepted_submission_count": "accepted_submission",
            "environment_terminal_count": "environment_terminal",
        }
        for metric, field in counts.items():
            if any(type(row.get(field)) is not bool for row in assessments):
                raise ValueError("action assessment flags must be explicit booleans")
            metrics[metric] = (
                sum(cast(bool, row[field]) for row in assessments) if complete else None
            )
        metrics["execution_returned_success_count"] = (
            sum(
                row["executed"] is True and row["execution_status"] == "success"
                for row in assessments
            )
            if complete
            else None
        )
        metrics["terminal_evidence_record_count"] = sum(v is not None for v in terminal_records)
        metrics["valid_terminal_record_count"] = (
            sum(v is True for v in terminal_records)
            if terminal_records and all(v is not None for v in terminal_records)
            else None
        )
        return metrics
