from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from collections.abc import Mapping
from pathlib import Path

from skillev.contracts import JsonValue

from .action_metrics import ActionMetrics
from .metrics_contract import FORMAT, TrainingMetricsSnapshot


class MetricsStore:
    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.actions = ActionMetrics(self.connection)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metrics (run TEXT, batch TEXT, commit_id TEXT, "
            "step INTEGER, payload TEXT, PRIMARY KEY(run, batch), UNIQUE(run, step))"
        )
        self.connection.execute("CREATE TABLE IF NOT EXISTS metrics_format (version INTEGER)")
        version = 1
        saved = self.connection.execute("SELECT version FROM metrics_format").fetchone()
        formats = {
            json.loads(row[0]).get("format")
            for row in self.connection.execute("SELECT payload FROM metrics")
        }
        if (saved is not None and saved[0] != version) or formats - {FORMAT} or len(formats) > 1:
            self.connection.close()
            raise ValueError("metrics format changed; use a new output/store")
        if saved is None:
            self.connection.execute("INSERT INTO metrics_format VALUES (?)", (version,))
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def add(self, snapshot: TrainingMetricsSnapshot) -> bool:
        payload = json.dumps(snapshot.to_value(), sort_keys=True, allow_nan=False)
        row = self.connection.execute(
            "SELECT commit_id, payload FROM metrics WHERE run=? AND batch=?",
            (snapshot.run_id, snapshot.batch_id),
        ).fetchone()
        if row is not None:
            if row != (snapshot.commit_id, payload):
                raise ValueError("a committed batch already has different metrics")
            return False
        with self.connection:
            self.connection.execute(
                "INSERT INTO metrics VALUES (?, ?, ?, ?, ?)",
                (*snapshot.identity, snapshot.optimizer_step, payload),
            )
        return True

    def values(self) -> list[dict[str, JsonValue]]:
        return [
            json.loads(row[0])
            for row in self.connection.execute("SELECT payload FROM metrics ORDER BY run, step")
        ]

    def publish(self, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in self.values():
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)


def export_available(
    events: Path,
    store: MetricsStore,
    *,
    condition_id: str,
    condition_starts: Mapping[int, str] | None = None,
    offset: int = 0,
    committed_through: int | None = None,
    committed_after: int = 0,
) -> tuple[int, int]:
    count = 0
    with events.open("rb") as stream:
        if stream.seek(0, os.SEEK_END) < offset:
            raise ValueError("source event log shrank; do not silently switch runs")
        stream.seek(offset)
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line or not line.endswith(b"\n"):
                return start, count
            event = json.loads(line)
            if event.get("event_type") == "agent_step_recorded":
                store.actions.observe(event)
            if event.get("event_type") == "training_step_committed":
                step = event["payload"]["optimizer_step"]
                if (
                    committed_through is not None
                    and event["payload"]["optimizer_step"] > committed_through
                ):
                    return start, count
                if event["payload"]["optimizer_step"] <= committed_after:
                    continue
                original_condition = (
                    condition_id
                    if condition_starts is None
                    else condition_starts[max(start for start in condition_starts if start <= step)]
                )
                count += store.add(
                    store.actions.enrich(
                        TrainingMetricsSnapshot.from_event(event, condition_id=original_condition),
                        event,
                    )
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incremental aggregate export for committed training events."
    )
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--committed-after", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        with args.output.open() as stream:
            if any(json.loads(line).get("format") != FORMAT for line in stream if line.strip()):
                parser.error("output has another format; choose a new output/store")
    store = MetricsStore(args.output.with_suffix(".sqlite3"))
    offset = 0
    try:
        while True:
            offset, _ = export_available(
                args.events,
                store,
                condition_id=args.condition_id,
                offset=offset,
                committed_after=args.committed_after,
            )
            store.publish(args.output)
            if not args.follow:
                break
            time.sleep(5)
    finally:
        store.close()


if __name__ == "__main__":
    main()
