from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from skillev.contracts import JsonValue
from skillev.policy import AdapterRole
from skillev.policy.checkpoint import PolicyCheckpointState
from skillev.runtime import EventType, SkillLibraryState
from skillev.runtime.request_journal import read_connection

if TYPE_CHECKING:
    from skillev.application import SKILLEVApplication


@dataclass(frozen=True)
class FreshNamespaces:
    request_journals: tuple[Path, ...]
    evidence_directories: tuple[Path, ...]
    maximum_empty_directory_depth: int = 2

    def __post_init__(self) -> None:
        if (
            type(self.maximum_empty_directory_depth) is not int
            or not 0 <= self.maximum_empty_directory_depth <= 4
        ):
            raise ValueError("empty management directory inspection must have bounded depth 0..4")


def _empty_management_tree(path: Path, depth: int) -> tuple[bool, int]:
    if path.is_symlink():
        return False, 1
    if not path.exists():
        return True, 0
    inspected = 0
    for entry in path.iterdir():
        inspected += 1
        if entry.is_symlink() or not entry.is_dir() or depth == 0:
            return False, inspected
        empty, children = _empty_management_tree(entry, depth - 1)
        inspected += children
        if not empty:
            return False, inspected
    return True, inspected


def _namespace_observations(namespaces: FreshNamespaces) -> list[dict[str, JsonValue]]:
    rows: list[dict[str, JsonValue]] = []
    for path in namespaces.request_journals:
        counts: dict[str, JsonValue] = {}
        error: str | None = None
        try:
            if path.exists():
                with read_connection(path) as db:
                    tables = {
                        r[0]
                        for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    }
                    for table, query in (
                        ("requests", "SELECT COUNT(*) FROM requests"),
                        ("episode_routes", "SELECT COUNT(*) FROM episode_routes"),
                        ("authorized_retries", "SELECT COUNT(*) FROM authorized_retries"),
                    ):
                        counts[table] = db.execute(query).fetchone()[0] if table in tables else None
                if counts.get("requests") is None:
                    error = "request table unavailable"
            else:
                counts = {"requests": 0, "episode_routes": 0, "authorized_retries": 0}
        except (OSError, sqlite3.Error) as exc:
            error = type(exc).__name__
        rows.append(
            {
                "path": str(path),
                "kind": "requests",
                "counts": counts,
                "empty": error is None and all(v in (None, 0) for v in counts.values()),
                "error": error,
            }
        )
    for path in namespaces.evidence_directories:
        error = None
        count: int | None = None
        empty = False
        try:
            empty, count = _empty_management_tree(path, namespaces.maximum_empty_directory_depth)
        except OSError as exc:
            error = type(exc).__name__
        rows.append(
            {
                "path": str(path),
                "kind": "evidence-directory",
                "inspected_entry_count": count,
                "empty": empty,
                "maximum_empty_directory_depth": namespaces.maximum_empty_directory_depth,
                "error": error,
            }
        )
    return rows


def inspect_fresh_state(
    application: SKILLEVApplication,
    *,
    initial_library: SkillLibraryState,
    namespaces: FreshNamespaces,
    preparation_state: PolicyCheckpointState | None = None,
    initial_parameters: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, JsonValue]:
    loop = application.training_loop
    backbone = application.backbone
    cursor = loop.task_provider.runtime_state
    run = application.run_progress.state
    projection = application.projections.runtime_state()
    detector = application.detector.state
    versions: dict[str, JsonValue] = {
        "forward": backbone.adapter_version(AdapterRole.FORWARD_POLICY),
        "backward": backbone.adapter_version(AdapterRole.BACKWARD_POLICY),
        "z": backbone.z_version,
    }
    try:
        loop.ledger.assert_fully_settled()
        ledger_settled = True
    except RuntimeError:
        ledger_settled = False
    checks: dict[str, JsonValue] = {
        "ledger_fully_settled": ledger_settled,
        "no_active_gradient_stream": loop.gradient_progress is None,
        "optimizer_step_zero": loop.optimizer_step == 0,
        "task_cursor_zero": cursor.cursor == 0,
        "run_cursor_zero": (
            run.completed_training_steps,
            run.committed_cycles,
            run.committed_actions,
        )
        == (0, 0, 0),
        "optimizer_state_empty": len(loop.optimizer.state) == 0,
        "posterior_empty": not projection.posterior_provenance.batches,
        "projection_fresh": projection.revision == 0
        and projection.diagnostics_state.expected_library_version
        == initial_library.current_version,
        "detector_fresh": detector.expected_library_version == initial_library.current_version,
        "initial_library_matches": application.library.state == initial_library,
        "preparation_step_zero": None
        if preparation_state is None
        else preparation_state.optimizer_step == 0,
        "preparation_backbone_matches": None
        if preparation_state is None
        else preparation_state.backbone_id == backbone.backbone_id,
        "preparation_versions_match": None
        if preparation_state is None
        else versions
        == {
            "forward": preparation_state.forward_version,
            "backward": preparation_state.backward_version,
            "z": preparation_state.z_version,
        },
    }
    parameters = backbone.named_trainable_parameters()
    mismatched: list[str] = []
    nonfinite = 0
    from skillev.policy.flow_head import FLOW_OFFSET_PARAMETER

    offsets = parameters.pop(FLOW_OFFSET_PARAMETER, None)
    if offsets is not None and bool((offsets.detach() != 0).any()):
        mismatched.append(FLOW_OFFSET_PARAMETER)
    for name, parameter in parameters.items():
        value = parameter.detach().cpu()
        nonfinite += int((~torch.isfinite(value)).sum())
        if initial_parameters is not None:
            initial = initial_parameters.get(name)
            if (
                initial is None
                or initial.device.type != "cpu"
                or initial.dtype != value.dtype
                or initial.shape != value.shape
                or not torch.equal(value, initial)
            ):
                mismatched.append(name)
    forward_ids = {id(p) for p in backbone.parameter_groups().forward}
    forward = {name: p for name, p in parameters.items() if id(p) in forward_ids}
    a_names = {name for name in forward if ".lora_A." in name}
    b_names = {name for name in forward if ".lora_B." in name}
    recognized = (
        bool(a_names and b_names)
        and a_names | b_names == set(forward)
        and {name.replace(".lora_A.", ".lora_B.") for name in a_names} == b_names
    )
    checks["forward_default_lora_b_zero"] = (
        all(int(torch.count_nonzero(forward[name].detach())) == 0 for name in b_names)
        if recognized
        else None
    )
    checks["parameters_finite"] = nonfinite == 0
    checks["preparation_tensors_match"] = (
        None
        if initial_parameters is None
        else not mismatched and set(parameters) == set(initial_parameters)
    )
    counts: dict[str, JsonValue] = {
        kind.value: application.emitter.log.committed_event_count(kind)
        for kind in (
            EventType.TRAINING_STEP_COMMITTED,
            EventType.PHASE_DETECTION_RECORDED,
            EventType.ROLLOUT_STARTED,
            EventType.AGENT_STEP_RECORDED,
            EventType.TERMINAL_REWARD_RECORDED,
        )
    }
    checks["no_prior_execution_events"] = not any(counts.values())
    namespace_rows = _namespace_observations(namespaces)
    checks["request_namespace_empty"] = (
        all(row["empty"] is True for row in namespace_rows if row["kind"] == "requests")
        if namespaces.request_journals
        else None
    )
    checks["evidence_namespace_empty"] = (
        all(row["empty"] is True for row in namespace_rows if row["kind"] == "evidence-directory")
        if namespaces.evidence_directories
        else None
    )
    failures = [name for name, result in checks.items() if result is False]
    unverified = [name for name, result in checks.items() if result is None]
    z_initialization = getattr(backbone, "z_initialization_spec", None)
    return {
        "format": "fresh-application-start@1",
        "declared_z_initialization": None
        if z_initialization is None
        else z_initialization.to_value(),
        "state": "rejected" if failures else "unverified" if unverified else "verified",
        "checks": checks,
        "failures": list(failures),
        "unverified": list(unverified),
        "optimizer_step": loop.optimizer_step,
        "optimizer_state_entries": len(loop.optimizer.state),
        "task_cursor": cursor.to_value(),
        "run_cursor": run.to_value(),
        "versions": versions,
        "library_version": application.library.current_version,
        "active_skill_ids": list(application.library.active_skill_ids),
        "posterior_batch_count": len(projection.posterior_provenance.batches),
        "projection_revision": projection.revision,
        "execution_event_counts": counts,
        "parameter_tensor_count": len(parameters),
        "parameter_nonfinite_count": nonfinite,
        "mismatched_parameter_names": list(mismatched),
        "namespaces": list(namespace_rows),
        "coverage": [
            "actual live state; before quality and rollout",
            "existing build load/bind checks are not repeated",
            "only trainable tensors, no frozen base copy",
            "namespace emptiness is current contents, not historical non-use",
            "preparation provenance and initialization method must be declared by caller; "
            "equality does not prove how preparation was generated",
        ],
    }


class FreshStateError(ValueError):
    def __init__(self, report: dict[str, JsonValue]) -> None:
        self.report = report
        super().__init__("application is not a verified fresh start; see private start report")


def require_fresh_state(report: dict[str, JsonValue], *, allow_unverified: bool = False) -> None:
    if (
        report.get("format") != "fresh-application-start@1"
        or report.get("failures")
        or (report.get("unverified") and not allow_unverified)
    ):
        raise FreshStateError(report)


def save_fresh_state_report(path: Path, report: dict[str, JsonValue]) -> None:
    encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
