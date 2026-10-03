from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Protocol

from skillev.contracts import JsonValue
from skillev.training.controller_status import ControllerStatusWriter
from skillev.training.inflight import durable_json
from skillev.training.loop import TrainingLoop
from skillev.training.step_timing import StepTiming


class CommittedEvidenceObserver(Protocol):
    def record_timings(self, timings: tuple[StepTiming, ...]) -> None: ...

    def observe_commits(self, committed_through: int) -> Any: ...

    def status(self) -> dict[str, Any]: ...

    def drain(self) -> dict[str, Any]: ...


class TrainingController:
    def __init__(
        self,
        root: Path,
        *,
        resource_roles: dict[str, JsonValue],
        gpu_uuids: tuple[str, ...],
        evidence: CommittedEvidenceObserver | None = None,
        initial_committed_step: int = 0,
    ) -> None:
        self.root, self.evidence = root, evidence
        self._lock = threading.RLock()
        self.started = time.monotonic()
        self.process = f"{os.getpid()}:{time.monotonic_ns()}"
        self.gpus = tuple(dict.fromkeys(gpu_uuids))
        self.writer = ControllerStatusWriter(
            root / "controller-status.json",
            run_id=root.name,
            process_instance_id=self.process,
            resource_roles=resource_roles,
        )
        self.loop: TrainingLoop | None = None
        self._state: str | None = "starting"
        if type(initial_committed_step) is not int or initial_committed_step < 0:
            raise ValueError("initial committed coordinate must be a nonnegative integer")
        self._committed_step = initial_committed_step
        self._ended: float | None = None
        self.publish()

    def attach(self, loop: TrainingLoop) -> None:
        with self._lock:
            self.loop = loop
            self._committed_step = loop.optimizer_step

    def set_state(self, state: str | None) -> None:
        with self._lock:
            self._state = state
            self.publish()

    def finish_evidence(self) -> dict[str, Any]:
        if self.evidence is None:
            return {"state": "not-configured", "pause_required": False}
        value = self.evidence.drain()
        self.publish()
        return value

    def close_resources(self) -> None:
        with self._lock:
            if self._ended is None:
                self._ended = time.monotonic()
            self.publish()

    def checkpoint_boundary(self) -> bool:
        with self._lock:
            if self.loop is None:
                raise RuntimeError("a complete boundary requires the attached training loop")
            self._committed_step = self.loop.optimizer_step
            if self.evidence is not None:
                self.evidence.record_timings(self.loop.finalized_timings)
                self.evidence.observe_commits(self._committed_step)
            value = self.publish()
            return bool(value["evidence"].get("pause_required", False))

    def publish(self) -> dict[str, Any]:
        with self._lock:
            loop = self.loop
            rollout = {} if loop is None else loop.rollout_progress
            gradient = (
                {}
                if loop is None
                else {**loop.execution_progress, **(loop.gradient_progress or {})}
            )
            stage = gradient.get("transaction_stage")
            state = self._state or (
                "collecting"
                if stage == "collecting"
                else "starting"
                if stage in {None, "idle"}
                else "updating"
            )
            evidence = (
                {"state": "not-configured"} if self.evidence is None else self.evidence.status()
            )
            value = self.writer.write(
                state=state,
                last_committed_step=self._committed_step,
                commit_id=evidence.get("source_commit_id"),
                committed_at=evidence.get("committed_at"),
                rollout=rollout,
                gradient=gradient,
                evidence_status=str(evidence.get("state", "unknown")),
            )
            elapsed = (self._ended or time.monotonic()) - self.started
            resource: dict[str, JsonValue] = {
                "format": "controller-resource-time@1",
                "scope": "process-reservation-including-preparation-quality-and-waits",
                "accounting": "overlaps-step-reservation; never-add-the-two-scopes",
                "clock": "process-monotonic",
                "process_instance_id": self.process,
                "gpu_uuids": list(self.gpus),
                "reserved_gpu_count": len(self.gpus),
                "process_elapsed_seconds": elapsed,
                "process_reserved_gpu_hours": len(self.gpus) * elapsed / 3600,
                "controller_scope_closed": self._ended is not None,
                "state": state,
            }
            durable_json(self.root / "resource-usage.json", resource)
            return {"controller": value, "evidence": evidence, "resources": resource}
