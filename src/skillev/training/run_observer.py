from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .action_failure_report import committed_action_outcomes
from .committed_evidence import index_committed_event, write_committed_index
from .evidence_mirror import EvidenceMirror
from .inflight import durable_json
from .learning_behavior_report import committed_learning_report
from .request_accounting import committed_request_accounting
from .step_timing import StepTiming


def _save(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("original observer evidence differs from its saved projection")
    else:
        durable_json(path, value)


class CommittedRunObserver:
    def __init__(
        self,
        run_root: Path,
        *,
        run_id: str,
        condition_id: str,
        mirror_root: Path,
        condition_starts: Mapping[int, str] | None = None,
        code_revision: str | None = None,
        expected_batch_size: int | None = None,
        batch_size_starts: Mapping[int, int] | None = None,
        max_pending_steps: int | None = None,
        minimum_free_bytes: int | None = None,
        events: Path | None = None,
        inflight: Path | None = None,
        requests: Path | None = None,
        committed_after: int = 0,
    ) -> None:
        if type(committed_after) is not int or committed_after < 0:
            raise ValueError("evidence starting boundary must be a nonnegative step")
        self.committed_after = committed_after
        if max_pending_steps is not None and max_pending_steps < 0:
            raise ValueError("pending bound must be nonnegative")
        self.run_id, self.condition_id, self.code_revision = run_id, condition_id, code_revision
        self.condition_starts = dict(condition_starts or {1: condition_id})
        if (
            any(type(step) is not int or step < 1 for step in self.condition_starts)
            or 1 not in self.condition_starts
            or any(not value for value in self.condition_starts.values())
            or self.condition_starts[max(self.condition_starts)] != condition_id
        ):
            raise ValueError(
                "condition history must start at step one and end at current condition"
            )
        self.batch_size, self.pending_limit = expected_batch_size, max_pending_steps
        self.batch_size_starts = dict(batch_size_starts or {})
        if self.batch_size_starts and (
            1 not in self.batch_size_starts
            or any(
                type(k) is not int or k < 1 or type(v) is not int or v < 1
                for k, v in self.batch_size_starts.items()
            )
            or self.batch_size_starts[max(self.batch_size_starts)] != expected_batch_size
        ):
            raise ValueError("batch-size history must resolve to the current complete batch")
        self.minimum_free_bytes = minimum_free_bytes
        self.events = events or run_root / "events.jsonl"
        self.inflight = inflight or run_root / "inflight"
        self.requests = requests or run_root / "requests.sqlite3"
        self.root = run_root / "evidence"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.RLock()
        self._indexes: dict[int, dict[str, Any]] = {}
        for path in sorted(self.root.glob("step-*.json")):
            row = json.loads(path.read_text())
            if row["run_id"] != run_id or row["condition_id"] != self._condition_at(
                row["optimizer_step"]
            ):
                raise ValueError("observer indexes belong to another run/condition")
            self._indexes[row["optimizer_step"]] = row
        self._offset = 0
        self._buffer: list[tuple[int, bytes, dict[str, Any]]] = []
        self._through = 0
        self._missing_steps: list[int] = []
        self._failure: dict[str, Any] | None = None
        self.mirror = EvidenceMirror(self.root, mirror_root, minimum_free_bytes=minimum_free_bytes)

    def _condition_at(self, step: int) -> str:
        start = max(start for start in self.condition_starts if start <= step)
        return self.condition_starts[start]

    def record_timings(self, timings: tuple[StepTiming, ...]) -> None:
        with self._lock:
            for timing in timings:
                prepared = self.root / "prepared" / f"step-{timing.optimizer_step:08d}"
                if timing.optimizer_step in self._indexes:
                    continue
                prepared.mkdir(parents=True, exist_ok=True, mode=0o700)
                _save(prepared / "finalized-timing.json", timing.to_value())
                if timing.rollout_detail is not None:
                    _save(prepared / "rollout-phases.json", timing.rollout_detail)

    def observe_commits(self, committed_through: int) -> dict[str, Any]:
        if type(committed_through) is not int or committed_through < 0:
            raise ValueError("a complete committed application step is required")
        with self._lock:
            if committed_through < max(self._through, max(self._indexes, default=0)):
                raise ValueError("observer cannot roll back completed source history")
            self._through = committed_through
            try:
                if self.events.exists():
                    if self.events.stat().st_size < self._offset:
                        raise ValueError("original event stream was truncated")
                    with self.events.open("rb") as stream:
                        stream.seek(self._offset)
                        while True:
                            offset = stream.tell()
                            raw = stream.readline()
                            if not raw or not raw.endswith(b"\n"):
                                break
                            event = json.loads(raw)
                            if event.get("run_id") != self.run_id:
                                raise ValueError("observer source stream belongs to another run")
                            if event.get("event_type") != "training_step_committed":
                                self._buffer.append((offset, raw, event))
                                self._offset = stream.tell()
                                continue
                            step = event["payload"]["optimizer_step"]
                            if step > committed_through:
                                break
                            if step in self._indexes:
                                if self._indexes[step]["source_commit_id"] != event["event_id"]:
                                    raise ValueError("same step has a different committed identity")
                                write_committed_index(self._indexes[step], self.root)
                            else:
                                if step != max(self._indexes, default=self.committed_after) + 1:
                                    break
                                self._observe(event, offset, raw)
                            self._buffer.clear()
                            self._offset = stream.tell()
                self._failure = None
            except (OSError, ValueError, sqlite3.Error, KeyError, TypeError) as error:
                self._failure = {
                    "state": "observer-failed",
                    "error_type": type(error).__name__,
                    "source_events": str(self.events),
                    "committed_through": committed_through,
                }
                durable_json(self.root / "observer-failure.json", self._failure)
            self._missing_steps = [
                step
                for step in range(self.committed_after + 1, committed_through + 1)
                if step not in self._indexes
            ]
            self.mirror.notify()
            return self.status()

    def _observe(self, event: dict[str, Any], offset: int, raw: bytes) -> None:
        step = event["payload"]["optimizer_step"]
        ids = {r["trajectory_id"] for r in event["payload"]["records"]}
        batch_id = event["payload"]["batch_id"]
        related = [
            (p, line, original)
            for p, line, original in self._buffer
            if any(
                original.get("payload", {}).get(key) in ids
                for key in ("trajectory_id", "invocation_id")
            )
            or original.get("payload", {}).get("batch_id") == batch_id
        ]
        related.append((offset, raw, event))
        index = index_committed_event(
            event,
            events=self.events,
            event_offset=offset,
            inflight=self.inflight,
            requests=self.requests,
            condition_id=self._condition_at(step),
            code_revision=self.code_revision,
            expected_batch_size=(
                self.batch_size_starts[max(k for k in self.batch_size_starts if k <= step)]
                if self.batch_size_starts
                else self.batch_size
            ),
            related_event_rows=[(p, original) for p, _, original in related],
        )
        prepared = self.root / "prepared" / f"step-{step:08d}"
        prepared.mkdir(parents=True, exist_ok=True, mode=0o700)
        _save(prepared / "commit.json", event)
        timing_path = prepared / "finalized-timing.json"
        if timing_path.exists():
            timing = json.loads(timing_path.read_text())
            if (timing.get("optimizer_step"), timing.get("batch_id")) != (step, batch_id):
                raise ValueError("finalized timing belongs to a different committed batch")
        detail_path = prepared / "rollout-phases.json"
        _save(
            prepared / "request-accounting.json",
            committed_request_accounting(
                event,
                rollout_detail=json.loads(detail_path.read_text())
                if detail_path.exists()
                else None,
                journal=self.requests,
            ),
        )
        _save(
            prepared / "learning.json",
            committed_learning_report(event, condition_id=self._condition_at(step)),
        )
        _save(
            prepared / "actions.json",
            committed_action_outcomes(
                event, source_events=[original for _, _, original in related], evidence_index=index
            ),
        )
        data = b"".join(line for _, line, _ in related)
        path = prepared / "events.jsonl"
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError("saved raw event slice differs")
        else:
            with path.open("xb") as out:
                out.write(data)
                out.flush()
                os.fsync(out.fileno())
        mirror_offset = 0
        locations = []
        for original_offset, line, original in related:
            locations.append(
                {
                    "event_id": original["event_id"],
                    "source_byte_offset": original_offset,
                    "mirror_byte_offset": mirror_offset,
                }
            )
            mirror_offset += len(line)
        _save(
            prepared / "event-locations.json",
            {"source_file": str(self.events), "events": locations},
        )
        write_committed_index(index, self.root)
        self._indexes[step] = index

    def status(self) -> dict[str, Any]:
        with self._lock:
            counts = dict.fromkeys(("pending", "copying", "failed", "mirrored"), 0)
            rows = []
            for path in sorted((self.root / "mirror-queue").glob("step-*.json")):
                row = json.loads(path.read_text())
                counts[row["state"]] += 1
                rows.append(row)
            free = None
            try:
                free = shutil.disk_usage(self.mirror.destination).free
            except OSError:
                pass
            backlog = counts["pending"] + counts["copying"] + counts["failed"]
            initially_missing = [
                step for step, row in self._indexes.items() if row["status"] != "RAW_AVAILABLE"
            ]
            verified_commits = {
                row["source_commit_id"]
                for row in rows
                if row["state"] == "mirrored" and row.get("verified_raw_status") == "RAW_AVAILABLE"
            }
            missing = [
                step
                for step in initially_missing
                if self._indexes[step]["source_commit_id"] not in verified_commits
            ]
            reasons = []
            if missing or self._missing_steps:
                reasons.append("RAW_MISSING")
            if self._failure:
                reasons.append("observer-failed")
            if counts["failed"]:
                reasons.append("mirror-failed")
            if self.mirror.worker_error is not None:
                reasons.append("mirror-worker-failed")
            if self.pending_limit is not None and backlog > self.pending_limit:
                reasons.append("declared-backlog-limit")
            if self.minimum_free_bytes is not None and (
                free is None or free < self.minimum_free_bytes
            ):
                reasons.append("low-space" if free is not None else "free-space-unknown")
            latest = self._indexes.get(max(self._indexes, default=0), {})
            state = (
                "RAW_MISSING"
                if missing or self._missing_steps
                else "failed"
                if self._failure or counts["failed"] or self.mirror.worker_error
                else "low-space"
                if "low-space" in reasons or "free-space-unknown" in reasons
                else "pending"
                if backlog
                else "mirrored"
                if counts["mirrored"]
                else "not-observed"
            )
            return {
                "format": "committed-run-observer@1",
                "state": state,
                "run_id": self.run_id,
                "condition_id": self.condition_id,
                "last_indexed_step": max(self._indexes, default=0),
                "source_commit_id": latest.get("source_commit_id"),
                "committed_at": latest.get("committed_at"),
                "committed_through": self._through,
                "evidence_starts_after_step": self.committed_after,
                "historical_raw_not_recovered_through": self.committed_after,
                "raw_missing_steps": sorted(set(missing + self._missing_steps)),
                "initial_index_raw_missing_steps": initially_missing,
                "mirror_counts": counts,
                "backlog_steps": backlog,
                "observed_free_bytes": free,
                "pause_required": bool(reasons),
                "pause_reasons": reasons,
                "observer_failure": self._failure,
                "mirror_worker_error": self.mirror.worker_error,
                "mirrors": rows,
                "scope": "observer receipts, not uploader liveness or new training updates",
            }

    def retry_mirrors(self) -> None:
        self.mirror.retry_failed()

    def drain(self) -> dict[str, Any]:
        self.mirror.drain()
        return self.status()

    def close(self) -> dict[str, Any]:
        self.mirror.close()
        return self.status()

    def __enter__(self) -> CommittedRunObserver:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
