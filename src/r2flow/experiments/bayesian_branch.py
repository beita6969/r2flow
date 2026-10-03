from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from skillev.runtime import StepTransactionJournal, StepTransactionState
from skillev.training.checkpoint import FilesystemTrainingCheckpointStore
from skillev.training.inflight import durable_json


def branch_checkpoint(*, source_root: Path, snapshot: Path, target_root: Path) -> Path:
    source_root, snapshot, target_root = (p.resolve() for p in (source_root, snapshot, target_root))
    if source_root == target_root or source_root.name != target_root.name:
        raise ValueError("branch needs a new parent, retaining the source experiment name")
    metadata = FilesystemTrainingCheckpointStore(root=snapshot.parent).load_metadata(snapshot)
    step = metadata.optimizer_step
    if step < 1 or metadata.experiment_id != source_root.name:
        raise ValueError("branch requires a committed trained snapshot of this experiment")
    evidence_start = checkpoint_only_evidence_start(source_root, snapshot)
    journal = StepTransactionJournal(source_root / "checkpoints" / "step-transactions")
    records = tuple(journal.load(i) for i in range(evidence_start + 1, step + 1))
    if any(r.state is not StepTransactionState.COMMITTED for r in records):
        raise ValueError("branch prefix contains an unresolved transaction")
    event_ids = {e.event_id for r in records for e in r.source_events}
    if not event_ids and step != evidence_start:
        raise ValueError("branch prefix has no committed source events")
    target_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    checkpoints = target_root / "checkpoints"
    checkpoints.mkdir()
    target = checkpoints / snapshot.name
    shutil.copytree(snapshot, target)
    if evidence_start:
        anchor = f"checkpoint-only-origin-step-{evidence_start:08d}"
        shutil.copytree(source_root / "checkpoints" / anchor, checkpoints / anchor)
        shutil.copy2(source_root / "checkpoint-only-recovery.json", target_root)
    destination = checkpoints / "step-transactions"
    destination.mkdir()
    for record in records:
        name = f"step-{record.optimizer_step:08d}.json"
        shutil.copy2(journal.directory / name, destination / name)
    with (
        (source_root / "events.jsonl").open("rb") as inp,
        (target_root / "events.jsonl").open("xb") as out,
    ):
        for line in inp:
            if not event_ids:
                break
            if not line.endswith(b"\n"):
                raise ValueError("branch prefix contains an incomplete source event")
            event = json.loads(line)
            if event["event_type"] == "training_step_committed":
                if event["payload"]["optimizer_step"] > step:
                    raise ValueError("required branch events not found before later commit")
            out.write(line)
            event_ids.discard(event["event_id"])
            if not event_ids:
                break
        if event_ids:
            raise ValueError("branch is missing original committed source events")
    for name in ("formal-config.json", "run-clock.json", "resolved-run-plan.json"):
        shutil.copy2(source_root / name, target_root / name)
    for path in source_root.glob("effective-condition-process-*.json"):
        shutil.copy2(path, target_root / path.name)
    durable_json(
        target_root / "branch-source.json",
        {
            "format": "committed-condition-branch@1",
            "source_root": str(source_root),
            "source_checkpoint": str(snapshot),
            "optimizer_step": step,
            "metrics_start_step": step + 1,
            **({"raw_evidence_unavailable_through_step": evidence_start} if evidence_start else {}),
            "sampling_condition": metadata.identity.sampling_schedule_algorithm,
            "history": "original source prefix; later source commits retained only in parent run",
            "quality": "new-condition baseline must be measured at the restored policy step",
        },
    )
    return target


def checkpoint_only_evidence_start(root: Path, resume: Path | None) -> int:
    declaration = root / "checkpoint-only-recovery.json"
    if not declaration.exists():
        return 0
    if resume is None:
        raise ValueError("checkpoint-only history requires a full checkpoint resume")
    value = json.loads(declaration.read_text())
    step = value.get("optimizer_step")
    if type(step) is not int or step < 1 or value.get("raw_history_available") is not False:
        raise ValueError("declare the missing original evidence boundary explicitly")
    store = FilesystemTrainingCheckpointStore(root=root / "checkpoints")
    anchor = root / "checkpoints" / f"checkpoint-only-origin-step-{step:08d}"
    original_path = anchor if anchor.exists() else root / "checkpoints" / f"step-{step:08d}"
    original = store.load_metadata(original_path)
    current = store.load_metadata(resume)
    if original.experiment_id != root.name or current.experiment_id != root.name:
        raise ValueError("recovery evidence belongs to another experiment")
    if original.optimizer_step != step or current.optimizer_step < step:
        raise ValueError("recovery evidence boundary exceeds the resumed checkpoint")
    if not anchor.exists():
        with tempfile.TemporaryDirectory(prefix=".recovery-origin-", dir=anchor.parent) as temp:
            staged = Path(temp) / "snapshot"
            shutil.copytree(original_path, staged)
            staged.rename(anchor)
    return step
