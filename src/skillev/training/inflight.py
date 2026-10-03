from __future__ import annotations

import json
import os
from pathlib import Path

from skillev.contracts import JsonValue, canonical_json
from skillev.rollout import RolloutArtifact, RolloutTokenizerProtocol
from skillev.runtime import BudgetLedger, BudgetReservation, BudgetSettlement, BudgetVector
from skillev.runtime.budget_ledger import LedgerEntry

from .planning import PlannedRollout, TrainingBatchPlan


def durable_json(path: Path, value: dict[str, JsonValue]) -> None:
    temporary = path.with_suffix(".pending")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(canonical_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def value_differences(
    stored: JsonValue, memory: JsonValue, path: str = "", limit: int = 12
) -> list[str]:
    found: list[str] = []

    def walk(a: JsonValue, b: JsonValue, where: str) -> None:
        if len(found) >= limit:
            return
        if type(a) is not type(b):
            found.append(f"{where or '$'} (type)")
        elif isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    found.append(f"{where}.{key} (only {'memory' if key not in a else 'durable'})")
                else:
                    walk(a[key], b[key], f"{where}.{key}")
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                found.append(f"{where} (length {len(a)} vs {len(b)})")
            for index, (x, y) in enumerate(zip(a, b, strict=False)):
                walk(x, y, f"{where}[{index}]")
        elif a != b:
            found.append(where or "$")

    walk(stored, memory, path)
    return found


def _entry_value(entry: LedgerEntry) -> dict[str, JsonValue]:
    reservation, settlement = entry.reservation, entry.settlement
    if settlement is None:
        raise ValueError("a completed artifact still has an unsettled call")
    return {
        "reservation_id": reservation.reservation_id,
        "run_id": reservation.run_id,
        "attempt_id": reservation.attempt_id,
        "invocation_id": reservation.invocation_id,
        "maximum": reservation.maximum.to_value(),
        "actual": settlement.actual.to_value(),
    }


class InFlightBatchStore:
    def __init__(self, root: Path, *, condition: dict[str, JsonValue]) -> None:
        self.root = root
        self.condition = condition

    def authorize_worker_reconfiguration(
        self, optimizer_step: int, *, authorization: str, reason: str
    ) -> None:
        if not authorization.strip() or not reason.strip():
            raise ValueError("worker reconfiguration requires explicit authorization")
        directory = self.root / f"step-{optimizer_step:08d}"
        original = json.loads((directory / "batch.json").read_text())["condition"]
        before = json.loads(canonical_json(original))
        after = json.loads(canonical_json(self.condition))
        for value in (before, after):
            value["formal"].pop("performance_profile", None)
            for key in ("gradient_worker_weights", "gradient_worker_max_sequence_tokens"):
                value["performance"].pop(key, None)
        if before != after:
            raise ValueError("worker transition changes non-placement conditions")
        transition: dict[str, JsonValue] = {
            "source": original,
            "target": self.condition,
            "authorization": authorization,
            "reason": reason,
        }
        path = directory / "worker-reconfiguration.json"
        if path.exists():
            if json.loads(path.read_text()) != transition:
                raise ValueError("cannot replace a recorded worker reconfiguration")
        else:
            durable_json(path, transition)

    def begin(self, plan: TrainingBatchPlan) -> InFlightBatch:
        directory = self.root / f"step-{plan.optimizer_step:08d}"
        directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        value: dict[str, JsonValue] = {
            "format": "uncommitted-batch@1",
            "condition": self.condition,
            "batch_id": plan.batch_id,
            "optimizer_step": plan.optimizer_step,
            "policy_snapshot_id": plan.policy_snapshot_id,
            "library_version": plan.library_version,
            "rollouts": [
                {
                    "position": item.position,
                    "trajectory_id": item.trajectory_id,
                    "task": item.task.to_value(),
                    "decoding": item.decoding.to_value(),
                }
                for item in plan.rollouts
            ],
        }
        path = directory / "batch.json"
        if path.exists():
            original = json.loads(path.read_text())
            transition_path = directory / "worker-reconfiguration.json"
            if original != value and transition_path.exists():
                transition = json.loads(transition_path.read_text())
                if (
                    transition["source"] == original["condition"]
                    and transition["target"] == self.condition
                ):
                    original = {**original, "condition": self.condition}
            if original != value:
                raise ValueError("in-flight evidence belongs to a different batch or condition")
        else:
            durable_json(path, value)
        return InFlightBatch(directory, plan)


class InFlightBatch:
    def __init__(self, directory: Path, plan: TrainingBatchPlan) -> None:
        self.directory, self.plan = directory, plan

    def _path(self, item: PlannedRollout) -> Path:
        return self.directory / f"trajectory-{item.position:06d}.json"

    def save(self, item: PlannedRollout, artifact: RolloutArtifact, ledger: BudgetLedger) -> None:
        self._require_artifact(item, artifact)
        entries: list[JsonValue] = [
            _entry_value(entry)
            for entry in ledger.entries
            if entry.reservation.invocation_id == item.trajectory_id
        ]
        value: dict[str, JsonValue] = {"artifact": artifact.to_value(), "budget_entries": entries}
        path = self._path(item)
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError("cannot replace the completed evidence of a planned rollout")
            return
        durable_json(path, value)

    def load(
        self, item: PlannedRollout, *, tokenizer: RolloutTokenizerProtocol, ledger: BudgetLedger
    ) -> RolloutArtifact | None:
        path = self._path(item)
        if not path.exists():
            return None
        value = json.loads(path.read_text())
        artifact = RolloutArtifact.from_value(value["artifact"], tokenizer=tokenizer)
        self._require_artifact(item, artifact)
        entries = []
        for raw in value["budget_entries"]:
            if raw["invocation_id"] != item.trajectory_id:
                raise ValueError("saved call charges belong to another trajectory")
            reservation = BudgetReservation(
                reservation_id=raw["reservation_id"],
                run_id=raw["run_id"],
                attempt_id=raw["attempt_id"],
                invocation_id=raw["invocation_id"],
                maximum=BudgetVector.from_value(raw["maximum"]),
            )
            settlement = BudgetSettlement(
                reservation.reservation_id, BudgetVector.from_value(raw["actual"])
            )
            entries.append(LedgerEntry.reserved(reservation).settled(settlement))
        ledger.restore_completed(entries)
        return artifact

    def require_complete(
        self, artifacts: tuple[RolloutArtifact, ...], *, tokenizer: RolloutTokenizerProtocol
    ) -> dict[str, JsonValue]:
        if len(artifacts) != len(self.plan.rollouts):
            raise ValueError("durable population differs from the complete planned batch")
        rows: list[JsonValue] = []
        for item, expected in zip(self.plan.rollouts, artifacts, strict=True):
            path = self._path(item)
            raw = json.loads(path.read_text(encoding="utf-8"))
            stored = RolloutArtifact.from_value(raw["artifact"], tokenizer=tokenizer)
            self._require_artifact(item, stored)
            stored_value, expected_value = stored.to_value(), expected.to_value()
            if stored_value != expected_value:
                paths = value_differences(stored_value, expected_value)
                durable_json(
                    path.with_name(path.stem + ".mismatch.json"),
                    {
                        "format": "artifact-mismatch-diagnostic@1",
                        "position": item.position,
                        "trajectory_id": item.trajectory_id,
                        "differing_paths": list[JsonValue](paths),
                        "in_memory_artifact": expected_value,
                    },
                )
                raise ValueError(
                    "durable artifact differs from the optimizer batch at position "
                    f"{item.position}: {', '.join(paths[:4])}"
                )
            rows.append(
                {
                    "position": item.position,
                    "trajectory_id": stored.record.trajectory_id,
                    "artifact_file": path.name,
                    "action_count": stored.record.horizon,
                    "terminal_result_count": 1,
                }
            )
        value: dict[str, JsonValue] = {
            "format": "readable-complete-batch-evidence@1",
            "batch_id": self.plan.batch_id,
            "optimizer_step": self.plan.optimizer_step,
            "policy_snapshot_id": self.plan.policy_snapshot_id,
            "library_version": self.plan.library_version,
            "trajectory_count": len(rows),
            "artifacts": rows,
            "state": "complete-artifacts-before-optimizer-not-a-commit",
        }
        path = self.directory / "complete-evidence.json"
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError("cannot replace the complete batch evidence identity")
        if not path.exists():
            durable_json(path, value)
        return value

    def _require_artifact(self, item: PlannedRollout, artifact: RolloutArtifact) -> None:
        manifest = artifact.manifest
        if (
            manifest.trajectory_id != item.trajectory_id
            or manifest.task_id != item.task.task_id
            or manifest.policy_snapshot.snapshot_id != self.plan.policy_snapshot_id
            or manifest.library_version != self.plan.library_version
            or manifest.decoding_snapshot_id != item.decoding.snapshot_id
        ):
            raise ValueError("saved artifact differs from its original rollout coordinates")
