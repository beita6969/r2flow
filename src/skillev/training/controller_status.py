from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from skillev.contracts import JsonValue

from .inflight import durable_json

STATES = frozenset(
    {"starting", "collecting", "updating", "validating", "paused", "finished", "failed", "unknown"}
)
TERMINAL_STATES = frozenset({"paused", "finished", "failed"})


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class ControllerStatusWriter:
    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        process_instance_id: str,
        resource_roles: dict[str, JsonValue],
    ) -> None:
        self.path, self.run_id, self.process = path, run_id, process_instance_id
        self.roles, self.pid = resource_roles, os.getpid()
        self._marker: object = None
        self._progress_monotonic: float | None = None
        self._progress_utc: str | None = None

    def write(
        self,
        *,
        state: str,
        last_committed_step: int,
        commit_id: str | None,
        committed_at: str | None,
        rollout: dict[str, Any] | None = None,
        gradient: dict[str, Any] | None = None,
        evidence_status: str | None = None,
        monotonic_now: float | None = None,
        observed_at: str | None = None,
    ) -> dict[str, JsonValue]:
        if os.getpid() != self.pid or state not in STATES or last_committed_step < 0:
            raise ValueError("controller state must be written by its owning process")
        now = time.monotonic() if monotonic_now is None else monotonic_now
        utc = utc_now() if observed_at is None else observed_at
        rollout, gradient = rollout or {}, gradient or {}
        trajectories = rollout.get("trajectories", [])
        marker = (
            state,
            last_committed_step,
            commit_id,
            tuple(
                (
                    r.get("trajectory_id"),
                    r.get("stage"),
                    r.get("stage_started"),
                    r.get("turn_index"),
                    len(r.get("phases", [])),
                )
                for r in trajectories
            ),
            tuple(
                (
                    r.get("trajectory_id"),
                    r.get("stage"),
                    r.get("updated"),
                )
                for r in gradient.get("trajectories", [])
            ),
            gradient.get("gradient_contributions_completed"),
            gradient.get("canonical_contributions_merged"),
            gradient.get("artifacts_ready"),
            gradient.get("edge_plans_ready"),
            gradient.get("stage"),
            gradient.get("transaction_stage"),
        )
        if marker != self._marker:
            self._marker, self._progress_monotonic, self._progress_utc = marker, now, utc
        value: dict[str, JsonValue] = {
            "format": "training-controller-status@1",
            "writer_scope": "training-controller",
            "run_id": self.run_id,
            "process_instance_id": self.process,
            "pid": self.pid,
            "resource_roles": self.roles,
            "state": state,
            "observed_at": utc,
            "heartbeat_monotonic": now,
            "phase_progress_monotonic": self._progress_monotonic,
            "phase_progress_at": self._progress_utc,
            "last_committed_step": last_committed_step,
            "source_commit_id": commit_id,
            "committed_at": committed_at,
            "evidence_status": evidence_status,
        }
        durable_json(self.path, value)
        return value


def controller_observation(
    value: dict[str, Any] | None,
    *,
    now: datetime,
    heartbeat_stale_after: float,
    progress_stale_after: float,
    expected_run_id: str,
) -> dict[str, JsonValue]:
    if heartbeat_stale_after <= 0 or progress_stale_after <= 0:
        raise ValueError("freshness intervals must come from positive declared limits")
    if value is None:
        return {"state": "unknown", "freshness": "controller-not-observed"}
    if (
        value.get("format") != "training-controller-status@1"
        or value.get("writer_scope") != "training-controller"
        or value.get("run_id") != expected_run_id
    ):
        raise ValueError("controller sidecar belongs to a different writer/run")
    state = value["state"]
    if state not in STATES:
        raise ValueError("unsupported controller state")
    age = (
        now - datetime.fromisoformat(value["observed_at"].replace("Z", "+00:00"))
    ).total_seconds()
    progress_age = (
        now - datetime.fromisoformat(value["phase_progress_at"].replace("Z", "+00:00"))
    ).total_seconds()
    freshness = (
        "clock-uncomparable"
        if age < 0 or progress_age < 0
        else "controller-heartbeat-stale"
        if age > heartbeat_stale_after
        else "no-phase-progress"
        if progress_age > progress_stale_after
        else "phase-progress-observed"
    )
    observed_state = (
        state
        if state in TERMINAL_STATES
        or freshness not in {"clock-uncomparable", "controller-heartbeat-stale"}
        else "unknown"
    )
    return {
        "state": observed_state,
        "declared_state": state,
        "freshness": freshness,
        "heartbeat_age_seconds": age if age >= 0 else None,
        "phase_progress_age_seconds": progress_age if progress_age >= 0 else None,
        "process_instance_id": value["process_instance_id"],
        "last_committed_step": value["last_committed_step"],
        "source_commit_id": value.get("source_commit_id"),
        "committed_at": value.get("committed_at"),
        "evidence_status": value.get("evidence_status"),
        "scope": "controller-receipt-not-uploader-liveness; no-timeout-implies-failure",
    }
