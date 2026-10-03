from __future__ import annotations

import fcntl
import json
import os
import shutil
import sqlite3
import tempfile
import threading
from contextlib import closing
from pathlib import Path
from typing import Any, cast

from skillev.runtime.request_journal import read_connection

from .committed_evidence import index_committed_event
from .inflight import durable_json


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def copy_file(source: Path, target: Path) -> None:
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        outgoing.flush()
        os.fsync(outgoing.fileno())


def copy_request_rows(
    source: Path,
    target: Path,
    episodes: list[str],
    *,
    additional_request_rowids: tuple[int, ...] = (),
) -> dict[str, int]:
    counts = {}
    with read_connection(source) as incoming:
        incoming.execute("BEGIN")
        with closing(sqlite3.connect(target)) as outgoing, outgoing:
            outgoing.execute("PRAGMA synchronous=FULL")
            outgoing.execute("PRAGMA journal_mode=DELETE")
            tables = {
                r[0] for r in incoming.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            for name, scope in (
                ("requests", "json_extract(identity,'$[0]')"),
                ("episode_routes", "episode"),
                ("authorized_retries", "json_extract(identity,'$[0]')"),
            ):
                if name not in tables:
                    if name == "requests":
                        raise ValueError("original request table is absent")
                    continue
                schema = incoming.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (name,)
                ).fetchone()[0]
                outgoing.execute(schema)
                placeholders = ",".join("?" for _ in episodes)
                selection = f"{scope} IN ({placeholders})"
                bindings: list[str | int] = list(episodes)
                if name == "requests" and additional_request_rowids:
                    extra = ",".join("?" for _ in additional_request_rowids)
                    selection += f" OR rowid IN ({extra})"
                    bindings.extend(additional_request_rowids)
                cursor = incoming.execute(
                    f"SELECT * FROM {name} WHERE {selection}",
                    bindings,
                )
                columns = ",".join("?" for _ in cursor.description)
                count = 0
                while values := cursor.fetchmany(16):
                    outgoing.executemany(f"INSERT INTO {name} VALUES ({columns})", values)
                    count += len(values)
                counts[name] = count
    with target.open("rb") as stream:
        os.fsync(stream.fileno())
    return counts


def criterion_ledger_paths(commit: dict[str, Any]) -> tuple[str, ...]:
    paths: dict[str, None] = {}
    for record in commit["payload"]["records"]:
        reference = record["reward"].get("native_payload", {}).get("criterion_ledger")
        if reference is None:
            continue
        if (
            not isinstance(reference, dict)
            or reference.get("format") != "healthbench-criterion-ledger@1"
        ):
            raise ValueError("unsupported committed criterion ledger reference")
        path = reference.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("committed criterion ledger lacks its original path")
        paths[path] = None
    return tuple(paths)


def copy_criterion_ledgers(paths: tuple[str, ...], stage: Path) -> dict[str, str]:
    if not paths:
        return {}
    directory = stage / "criterion-ledgers"
    directory.mkdir(mode=0o700)
    relocations = {}
    for position, source in enumerate(paths, 1):
        relative = f"criterion-ledgers/ledger-{position:06d}.json"
        copy_file(Path(source), stage / relative)
        relocations[source] = relative
    fsync_directory(directory)
    return relocations


def mirror_step(index_path: Path, destination: Path) -> dict[str, Any]:
    index = json.loads(index_path.read_text())
    step = index["optimizer_step"]
    prepared = index_path.parent / "prepared" / f"step-{step:08d}"
    final = destination / f"step-{step:08d}"
    commit = json.loads((prepared / "commit.json").read_text())
    ledger_paths = criterion_ledger_paths(commit)
    if final.exists():
        manifest = json.loads((final / "mirror.json").read_text())
        if (
            manifest.get("source_commit_id") != index["source_commit_id"]
            or json.loads((final / "index.json").read_text()) != index
        ):
            raise ValueError("destination belongs to different committed evidence")
        if manifest.get("state") != "complete":
            raise ValueError("destination has no complete mirror")
        relocations = manifest.get("criterion_ledger_relocations", {})
        for source in ledger_paths:
            relative = relocations.get(source)
            if not isinstance(relative, str) or not (final / relative).is_file():
                raise FileNotFoundError("published mirror lacks referenced criterion evidence")
        return cast(dict[str, Any], manifest)
    stage = Path(tempfile.mkdtemp(prefix=f".step-{step:08d}.partial-", dir=destination))
    copy_file(index_path, stage / "index.json")
    for name in ("events.jsonl", "event-locations.json", "learning.json", "actions.json"):
        copy_file(prepared / name, stage / name)
    for name in ("request-accounting.json", "finalized-timing.json", "rollout-phases.json"):
        if (prepared / name).exists():
            copy_file(prepared / name, stage / name)
    copy_file(prepared / "commit.json", stage / "commit.json")
    trajectory_dir = stage / "inflight" / f"step-{step:08d}"
    trajectory_dir.mkdir(parents=True, mode=0o700)
    for row in index["trajectories"]:
        copy_file(Path(row["artifact"]), trajectory_dir / f"trajectory-{row['position']:06d}.json")
    source_dir = Path(index["trajectories"][0]["artifact"]).parent
    for optional in ("batch.json", "complete-evidence.json"):
        if (source_dir / optional).exists():
            copy_file(source_dir / optional, trajectory_dir / optional)
    request_files = {r["file"] for row in index["trajectories"] for r in row["request_records"]}
    if len(request_files) != 1 or None in request_files:
        raise FileNotFoundError("original per-step request journal unavailable")
    source_requests = Path(next(iter(request_files)))
    accounting_path = prepared / "request-accounting.json"
    unassigned = (
        json.loads(accounting_path.read_text())["unassigned_same_task_judge_rows"]
        if accounting_path.exists()
        else []
    )
    counts = copy_request_rows(
        source_requests,
        stage / "requests.sqlite3",
        [r["trajectory_id"] for r in index["trajectories"]],
        additional_request_rowids=tuple(row["reference"]["source_rowid"] for row in unassigned),
    )
    checked = index_committed_event(
        commit,
        events=stage / "events.jsonl",
        event_offset=None,
        inflight=stage / "inflight",
        requests=stage / "requests.sqlite3",
        condition_id=index["condition_id"],
        expected_batch_size=index["trajectory_count"],
    )
    if checked["status"] != "RAW_AVAILABLE":
        raise ValueError("RAW_MISSING in copied original evidence")
    relocations = copy_criterion_ledgers(ledger_paths, stage)
    manifest = {
        "format": "committed-evidence-mirror@2",
        "criterion_ledger_relocations": relocations,
        "state": "complete",
        "raw_status": checked["status"],
        "source_commit_id": index["source_commit_id"],
        "run_id": index["run_id"],
        "condition_id": index["condition_id"],
        "optimizer_step": step,
        "trajectory_count": index["trajectory_count"],
        "request_table_counts": counts,
        "pre_optimizer_completeness_receipt": "copied"
        if (trajectory_dir / "complete-evidence.json").exists()
        else None,
        "coverage": (
            "exact trajectory-linked journal rows + related events + artifacts + "
            "explicitly referenced private criterion ledgers; "
            "same-task judge candidates, when present, remain unassigned to the step; "
            "unrelated author journals not inferred"
        ),
    }
    durable_json(stage / "mirror.json", manifest)
    fsync_directory(trajectory_dir)
    fsync_directory(trajectory_dir.parent)
    fsync_directory(stage)
    os.rename(stage, final)
    fsync_directory(destination)
    return manifest


class EvidenceMirror:
    def __init__(
        self, evidence_root: Path, destination: Path, *, minimum_free_bytes: int | None = None
    ) -> None:
        if minimum_free_bytes is not None and minimum_free_bytes < 0:
            raise ValueError("minimum free bytes must be nonnegative")
        if evidence_root.resolve() == destination.resolve():
            raise ValueError("mirror destination must differ from its source")
        self.root, self.destination = evidence_root, destination
        self.minimum_free_bytes = minimum_free_bytes
        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lockfile = (evidence_root / "mirror.lock").open("a")
        try:
            fcntl.flock(self._lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lockfile.close()
            raise
        self._condition = threading.Condition()
        self._fatal_error: str | None = None
        self._running = False
        self._closing = False
        self._wake = False
        self._thread = threading.Thread(
            target=self._work, name="committed-evidence-mirror", daemon=False
        )
        self._thread.start()

    def notify(self) -> None:
        with self._condition:
            if self._closing:
                raise RuntimeError("mirror worker is closed")
            self._wake = True
            self._condition.notify_all()

    def retry_failed(self) -> None:
        with self._condition:
            if self._running:
                raise RuntimeError("drain the mirror before retrying failed copies")
            for path in (self.root / "mirror-queue").glob("step-*.json"):
                row = json.loads(path.read_text())
                if row["state"] == "failed":
                    durable_json(path, {**row, "state": "pending"})
        self.notify()

    def _work(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._wake or self._closing)
                if self._closing and not self._wake:
                    return
                self._wake, self._running = False, True
            try:
                for path in sorted((self.root / "mirror-queue").glob("step-*.json")):
                    row = json.loads(path.read_text())
                    if row["state"] not in ("pending", "copying"):
                        continue
                    try:
                        free = shutil.disk_usage(self.destination).free
                        if self.minimum_free_bytes is not None and free < self.minimum_free_bytes:
                            durable_json(
                                path,
                                {
                                    **row,
                                    "state": "pending",
                                    "blocked_reason": "low-space",
                                    "observed_free_bytes": free,
                                },
                            )
                            continue
                        durable_json(path, {**row, "state": "copying"})
                        manifest = mirror_step(Path(row["index_file"]), self.destination)
                        durable_json(
                            path,
                            {
                                **row,
                                "state": "mirrored",
                                "verified_raw_status": manifest.get("raw_status"),
                                "destination": str(
                                    self.destination / f"step-{manifest['optimizer_step']:08d}"
                                ),
                                "blocked_reason": None,
                                "error_type": None,
                            },
                        )
                    except (OSError, ValueError, sqlite3.Error, KeyError, TypeError) as error:
                        durable_json(
                            path,
                            {
                                **row,
                                "state": "failed",
                                "error_type": type(error).__name__,
                                "blocked_reason": "copy-failed",
                            },
                        )
            except (OSError, ValueError, sqlite3.Error, KeyError, TypeError) as error:
                with self._condition:
                    self._fatal_error = type(error).__name__
            finally:
                with self._condition:
                    self._running = False
                    self._condition.notify_all()

    @property
    def worker_error(self) -> str | None:
        with self._condition:
            return self._fatal_error

    def drain(self) -> None:
        with self._condition:
            self._condition.wait_for(lambda: not self._running and not self._wake)

    def close(self) -> None:
        self.drain()
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._thread.join()
        self._lockfile.close()
