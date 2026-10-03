from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import json
import math
import os
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Protocol, cast

from skillev.contracts import JsonValue, canonical_json, normalize_json
from skillev.r2flow_evolution.types import (
    EdgeObs,
    GateDecision,
    LibraryVersion,
    PhaseState,
    TrajectoryObs,
)
from skillev.runtime.skill_library import SkillLibrary

from .r2flow_evolution_config import EvolutionConfig
from .r2flow_library_versions import (
    LibraryVersionStore,
    PhaseCarrier,
    PhaseRecord,
    committed_spec,
    decode_phase_state,
    encode_phase_state,
    initial_library_version,
    library_matches_runtime,
    library_version_from_value,
    library_version_to_value,
    runtime_library_state,
)

TRACE_ROW_FORMAT: Final = "r2flow-improvement-trace-row@1"
PI_EVAL_RULE: Final = "forward-snapshot-at-phase-boundary@1"
SEGMENT_RESET_RULE: Final = "library-segment-reset-no-z@1"
VALIDATION_TASK_PREFIX: Final = "r2flow-validation@1"
VALIDATION_TOKENS_RULE: Final = "validation-tokens=policy-generated@1"
VALIDATION_LATENCY_RULE: Final = "validation-latency=agent-time@1"
VALIDATION_EDGE_RULE: Final = "heldout-edge-residual=nan@1"
STEP_COMMITTED_EVENT: Final = "rollout_step_committed"

TRANSITION_INFRASTRUCTURE_ERRORS: Final = (OSError, TimeoutError, RuntimeError)
RolloutHeldout = Callable[[LibraryVersion, int], Sequence[TrajectoryObs]]
AsyncRolloutHeldout = Callable[[LibraryVersion, int], Awaitable[Sequence[TrajectoryObs]]]


class PolicySnapshotLike(Protocol):
    @property
    def snapshot_id(self) -> str: ...


class HeldoutRolloutFactory(Protocol):
    def __call__(
        self, *, snapshot: Any, phase: int, optimizer_step: int
    ) -> AsyncRolloutHeldout: ...


def _label_hash(event_label: str) -> str:
    from skillev.policy.event_grammar import key_json

    return hashlib.sha256(key_json({"label": event_label}).encode("utf-8")).hexdigest()


def _legal_event_count(legal: Any) -> int:
    total = 0
    for function in legal.functions:
        product = 1
        for param in function.params:
            if param.kind == "enum":
                product *= len(param.values)
        total += product
    return total


def _seconds(started_at: str, completed_at: str) -> float:
    start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    end = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
    return (end - start).total_seconds()


def validation_query_key(task_id: str) -> tuple[str, int]:
    prefix = VALIDATION_TASK_PREFIX + "/"
    if not task_id.startswith(prefix):
        raise ValueError("not a held-out validation task id")
    benchmark, _, rest = task_id[len(prefix) :].partition("/")
    source, _, replica = rest.rpartition("/")
    if not benchmark or not source or not replica.startswith("r") or not replica[1:].isdigit():
        raise ValueError("malformed held-out validation task id")
    return f"{benchmark}:{source}", int(replica[1:])


def last_step_commits(events_path: Path) -> dict[str, tuple[int, str]]:
    commits: dict[str, tuple[int, str]] = {}
    with events_path.open(encoding="utf-8") as stream:
        for line in stream:
            if STEP_COMMITTED_EVENT not in line:
                continue
            event = json.loads(line)
            if event["event_type"] == STEP_COMMITTED_EVENT:
                payload = event["payload"]
                commits[str(payload["trajectory_id"])] = (
                    int(payload["step_index"]),
                    str(event["occurred_at"]),
                )
    return commits


def heldout_trajectory_obs(
    artifact: Any,
    *,
    eta: float,
    epsilon: float,
    executor_calls: Sequence[Any] = (),
    namespace: str = "",
    agent_step_commit: tuple[int, str] | None = None,
) -> TrajectoryObs:
    record, manifest = artifact.record, artifact.manifest
    query_id, _ = validation_query_key(manifest.task_id)
    reward = float(record.reward.value)
    tokens = sum(int(n) for n in manifest.reasoning_token_counts) + sum(
        int(step.action_token_count) for step in record.steps
    )
    if agent_step_commit is None or agent_step_commit[0] != record.steps[-1].index:
        raise ValueError(
            f"{VALIDATION_LATENCY_RULE} needs the commit time of the final step "
            f"{record.steps[-1].index} of {record.trajectory_id}, got {agent_step_commit}"
        )
    latency = _seconds(manifest.started_at, agent_step_commit[1])
    latency += sum(
        (int(call.reference_latency_ms) - int(call.physical_latency_ms)) / 1000.0
        for call in executor_calls
    )
    legal_sets = (
        None if artifact.verifier_inputs is None else artifact.verifier_inputs.legal_event_sets
    )
    edges = []
    for position, step in enumerate(record.steps):
        flow = step.r2flow
        if flow is None:
            raise ValueError("a held-out validation rollout needs sigma-mode steps")
        edges.append(
            EdgeObs(
                step_index=step.index,
                predecessor_key=flow.predecessor_key,
                state_key=flow.state_key,
                in_edges=tuple(flow.in_edges),
                event_label=_label_hash(flow.event_label),
                event_function=flow.event_function,
                skill_id=step.invoked_skill_ids[0] if step.invoked_skill_ids else None,
                legal_event_count=-1
                if legal_sets is None
                else _legal_event_count(legal_sets[position]),
                edge_residual=math.nan,
            )
        )
    domain = (
        artifact.verifier_inputs.domain
        if artifact.verifier_inputs is not None
        else record.task_family.split("/")[0]
    )
    return TrajectoryObs(
        trajectory_id=f"{namespace}{record.trajectory_id}",
        query_id=query_id,
        domain=str(domain),
        family=record.task_family,
        reward=reward,
        reward_eta=float((reward + epsilon) ** eta),
        success=bool(record.reward.success),
        tokens=int(tokens),
        latency_seconds=max(0.0, latency),
        edges=tuple(edges),
    )


def trajectory_obs_to_value(obs: TrajectoryObs) -> dict[str, JsonValue]:
    return {
        "domain": obs.domain,
        "edges": [
            {
                "edge_residual": None if math.isnan(e.edge_residual) else e.edge_residual,
                "event_function": e.event_function,
                "event_label": e.event_label,
                "in_edges": [list(pair) for pair in e.in_edges],
                "legal_event_count": e.legal_event_count,
                "predecessor_key": e.predecessor_key,
                "skill_id": e.skill_id,
                "state_key": e.state_key,
                "step_index": e.step_index,
            }
            for e in obs.edges
        ],
        "family": obs.family,
        "latency_seconds": obs.latency_seconds,
        "query_id": obs.query_id,
        "reward": obs.reward,
        "reward_eta": obs.reward_eta,
        "success": obs.success,
        "tokens": obs.tokens,
        "trajectory_id": obs.trajectory_id,
    }


def trajectory_obs_from_value(value: dict[str, Any]) -> TrajectoryObs:
    return TrajectoryObs(
        trajectory_id=str(value["trajectory_id"]),
        query_id=str(value["query_id"]),
        domain=str(value["domain"]),
        family=str(value["family"]),
        reward=float(value["reward"]),
        reward_eta=float(value["reward_eta"]),
        success=bool(value["success"]),
        tokens=int(value["tokens"]),
        latency_seconds=float(value["latency_seconds"]),
        edges=tuple(
            EdgeObs(
                step_index=int(e["step_index"]),
                predecessor_key=str(e["predecessor_key"]),
                state_key=str(e["state_key"]),
                in_edges=tuple((str(a), str(b)) for a, b in e["in_edges"]),
                event_label=str(e["event_label"]),
                event_function=str(e["event_function"]),
                skill_id=None if e["skill_id"] is None else str(e["skill_id"]),
                legal_event_count=int(e["legal_event_count"]),
                edge_residual=math.nan if e["edge_residual"] is None else float(e["edge_residual"]),
            )
            for e in value["edges"]
        ),
    )


def threadsafe_rollout_heldout(
    collect: AsyncRolloutHeldout, loop: asyncio.AbstractEventLoop
) -> RolloutHeldout:
    def rollout_heldout(
        library: LibraryVersion, seed: int, *, domains: frozenset[str] | None = None
    ) -> list[TrajectoryObs]:
        if not isinstance(library, LibraryVersion) or type(seed) is not int:
            raise TypeError("rollout_heldout(library_version, seed)")
        if domains is not None and (
            not isinstance(domains, frozenset)
            or not domains
            or any(type(domain) is not str for domain in domains)
        ):
            raise TypeError("rollout_heldout(..., domains=) takes a non-empty frozenset of domains")
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            raise RuntimeError("rollout_heldout must be called from the transition worker thread")

        async def arm() -> Sequence[TrajectoryObs]:
            if domains is not None:
                return await cast(Callable[..., Awaitable[Sequence[TrajectoryObs]]], collect)(
                    library, seed, domains=domains
                )
            return await collect(library, seed)

        return list(asyncio.run_coroutine_threadsafe(arm(), loop).result())

    return rollout_heldout


def _default_transition() -> Callable[..., tuple[GateDecision, PhaseState]]:
    module = importlib.import_module("skillev.r2flow_evolution.transition")
    return cast(Callable[..., tuple[GateDecision, PhaseState]], module.run_phase_transition)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


def _edit_summary(decision: GateDecision) -> dict[str, JsonValue]:
    def edit(item: Any) -> dict[str, JsonValue]:
        return {
            "kind": str(item.candidate.kind),
            "targets": list(item.candidate.skill_ids),
            "added": [spec.skill_id for spec in item.added],
            "removed": list(item.removed),
            "author_model": item.author_model,
        }

    return {
        "accepted": [edit(item) for item in decision.accepted],
        "rejected": [{**edit(item), "reason": reason} for item, reason in decision.rejected],
    }


@dataclass(slots=True)
class R2FlowPhaseController:
    run_root: Path
    library: SkillLibrary
    carrier: PhaseCarrier
    reset_segment: Callable[[str, str], None]
    policy_snapshot: Callable[[], PolicySnapshotLike]
    config: EvolutionConfig
    eta: float
    epsilon: float
    family_universe: frozenset[str]
    heldout: HeldoutRolloutFactory
    trace_path: Path
    transition: Callable[..., tuple[GateDecision, PhaseState]] | None = None
    set_state: Callable[[str | None], None] | None = None
    log: Callable[..., None] | None = None

    @property
    def store(self) -> LibraryVersionStore:
        return LibraryVersionStore(self.run_root)

    @property
    def record(self) -> PhaseRecord:
        record = self.carrier.record()
        if record is None:
            raise RuntimeError("the evolve@1 phase record is not initialised")
        return record

    def segment_label(self) -> str:
        return self.record.segment_label

    def phase_start(self) -> int:
        return self.record.since_step

    def start(self, *, optimizer_step: int, initial_specs: Sequence[Any] | None = None) -> None:
        record = self.carrier.record()
        state = self.library.state
        if record is None:
            if optimizer_step != 0:
                raise ValueError("an evolve@1 checkpoint after step 0 lacks its phase record")
            library = initial_library_version(
                state, None if initial_specs is None else tuple(initial_specs)
            )
            record = PhaseRecord(0, library, state.current_version, PhaseState(), 0)
            self.store.save_version(library, digest=state.current_version, committed_at_step=0)
            self.carrier.set(record)
        elif record.library_digest != state.current_version or not library_matches_runtime(
            record.library, state
        ):
            raise ValueError("the carried library version differs from the checkpoint library")
        self.store.save_active(record)

    async def on_boundary(
        self,
        *,
        optimizer_step: int,
        policy_snapshot_id: str,
        reason: str,
        diagnostics: dict[str, JsonValue],
    ) -> None:
        record = self.record
        existing = self.store.load_transition(optimizer_step)
        if existing is not None:
            if existing["phase_before"] == record.phase:
                self._append_trace(existing)
                self._apply(existing)
            elif existing["phase_after"] != record.phase:
                raise RuntimeError("a recorded phase transition belongs to another phase")
            return
        if optimizer_step <= record.since_step:
            raise ValueError("a phase boundary must follow the start of its phase")
        snapshot = self.policy_snapshot()
        if snapshot.snapshot_id != policy_snapshot_id:
            raise ValueError("pi_eval must be the committed forward policy of the boundary")
        loop = asyncio.get_running_loop()
        collect = self.heldout(snapshot=snapshot, phase=record.phase, optimizer_step=optimizer_step)
        transition = self.transition or _default_transition()
        kwargs: dict[str, Any] = {
            "run_root": self.run_root,
            "library": record.library,
            "state": record.phase_state,
            "phase": record.phase,
            "optimizer_step": optimizer_step,
            "since_step": record.since_step,
            "eta": self.eta,
            "epsilon": self.epsilon,
            "config": self.config,
            "rollout_heldout": threadsafe_rollout_heldout(collect, loop),
        }
        try:
            accepted: set[str] = set(inspect.signature(transition).parameters)
        except (TypeError, ValueError):
            accepted = set()
        if "diagnostics" in accepted:
            kwargs["diagnostics"] = {"vq_boundary": diagnostics, "reason": reason}
        if self.set_state is not None:
            self.set_state("validating")
        self._log("r2flow-phase-transition-started", phase=record.phase, step=optimizer_step)
        failure: BaseException | None = None
        try:
            decision, next_state = await asyncio.to_thread(transition, **kwargs)
        except TRANSITION_INFRASTRUCTURE_ERRORS as error:
            failure = error
            decision = GateDecision((), (), record.library, {})
            next_state = replace(record.phase_state, phase=record.phase + 1)
        finally:
            if self.set_state is not None:
                self.set_state(None)
        value = self._commit_value(
            record, decision, next_state, optimizer_step, policy_snapshot_id, reason, diagnostics
        )
        if failure is not None:
            value["transition_failed"] = f"{type(failure).__name__}: {failure}"[:2000]
        if value["changed"]:
            library = library_version_from_value(value["library"])
            self.store.save_version(
                library, digest=str(value["library_after"]), committed_at_step=optimizer_step
            )
        self.store.save_transition(optimizer_step, value)
        committed = self.store.load_transition(optimizer_step)
        assert committed is not None
        self._append_trace(committed)
        self._apply(committed)
        self._log(
            "r2flow-phase-transition-committed",
            phase=record.phase,
            step=optimizer_step,
            library_version=int(value["version_after"]),
            changed=bool(value["changed"]),
        )

    def _commit_value(
        self,
        record: PhaseRecord,
        decision: GateDecision,
        next_state: PhaseState,
        optimizer_step: int,
        policy_snapshot_id: str,
        reason: str,
        diagnostics: dict[str, JsonValue],
    ) -> dict[str, Any]:
        if not isinstance(decision, GateDecision) or not isinstance(next_state, PhaseState):
            raise TypeError("run_phase_transition returns (GateDecision, PhaseState)")
        after = decision.library_after
        if not isinstance(after, LibraryVersion):
            raise TypeError("GateDecision.library_after must be a LibraryVersion")
        library, digest = record.library, record.library_digest
        refused: str | None = None
        try:
            before_specs = {spec.skill_id: committed_spec(spec) for spec in record.library.skills}
            after_specs = {spec.skill_id: committed_spec(spec) for spec in after.skills}
            if before_specs != after_specs:
                if not decision.accepted and not decision.trace_row.get("replayed_from"):
                    raise ValueError("Prop. C.1: the library changed without an accepted edit")
                outside = sorted(
                    spec.skill_id
                    for spec in after_specs.values()
                    if not set(spec.families) <= self.family_universe
                )
                if outside:
                    raise ValueError(f"evolved skills target families outside the run: {outside}")
                candidate = LibraryVersion(
                    record.library.version + 1,
                    tuple(after_specs[skill_id] for skill_id in sorted(after_specs)),
                )
                digest = runtime_library_state(candidate, self.library.state).current_version
                library = candidate
        except ValueError as error:
            library, digest, refused = record.library, record.library_digest, str(error)
        changed = library is not record.library
        snapshot_value = normalize_json(
            {
                "rule": PI_EVAL_RULE,
                "policy_snapshot_id": policy_snapshot_id,
            }
        )
        return {
            "optimizer_step": optimizer_step,
            "reason": reason,
            "phase_before": record.phase,
            "phase_after": record.phase + 1,
            "since_step_before": record.since_step,
            "version_before": record.library.version,
            "version_after": library.version,
            "library_before": record.library_digest,
            "library_after": digest,
            "changed": changed,
            "commit_refused": refused,
            "library": library_version_to_value(library),
            "phase_state": encode_phase_state(next_state),
            "pi_eval": snapshot_value,
            "edits": _json_safe(_edit_summary(decision)),
            "vq_boundary": _json_safe(diagnostics),
            "gate_trace_row": _json_safe(decision.trace_row),
            "segment_reset": SEGMENT_RESET_RULE,
        }

    def _apply(self, value: dict[str, Any]) -> None:
        library = library_version_from_value(value["library"])
        old = self.library.state
        if bool(value["changed"]):
            if old.current_version != value["library_before"]:
                raise ValueError("the committed transition starts from another library")
            state = runtime_library_state(library, old)
            if state.current_version != value["library_after"]:
                raise ValueError("the committed library version does not reproduce its digest")
            self.library.apply(state)
            self.reset_segment(old.current_version, state.current_version)
        elif old.current_version != value["library_after"]:
            raise ValueError("an unchanged transition must keep the active library")
        record = PhaseRecord(
            phase=int(value["phase_after"]),
            library=library,
            library_digest=str(value["library_after"]),
            phase_state=decode_phase_state(value["phase_state"]),
            since_step=int(value["optimizer_step"]),
            last_transition={
                "optimizer_step": value["optimizer_step"],
                "reason": value["reason"],
                "library_before": value["library_before"],
                "library_after": value["library_after"],
                "version_before": value["version_before"],
                "version_after": value["version_after"],
                "changed": value["changed"],
            },
        )
        self.carrier.set(record)
        self.store.save_active(record)

    def _append_trace(self, value: dict[str, Any]) -> None:
        path = self.trace_path
        rows = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        if any(json.loads(row).get("optimizer_step") == value["optimizer_step"] for row in rows):
            return
        gate = value.get("gate_trace_row") or {}
        row = normalize_json(
            {
                "format": TRACE_ROW_FORMAT,
                "action": "evolve@1",
                "optimizer_step": value["optimizer_step"],
                "reason": value["reason"],
                "k": value["phase_before"],
                "v_k": {"version": value["version_before"], "digest": value["library_before"]},
                "v_k1": {"version": value["version_after"], "digest": value["library_after"]},
                "delta_k": value["vq_boundary"],
                "d_k": value["edits"],
                "val_k": gate.get("val_k") if isinstance(gate, dict) else None,
                "gate_trace_row": gate,
                "changed": value["changed"],
                "commit_refused": value.get("commit_refused"),
                "transition_failed": value.get("transition_failed"),
                "pi_eval": value["pi_eval"],
            }
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _log(self, status: str, **fields: JsonValue) -> None:
        if self.log is not None:
            self.log(status, **fields)


def controller_for_application(application: Any, **kwargs: Any) -> R2FlowPhaseController:
    def reset_segment(old: str, new: str) -> None:
        application.detector.reset_for_library(old, new)
        application.projections.reset_library_segment(old, new)

    return R2FlowPhaseController(
        library=application.library,
        carrier=application.phase_carrier,
        reset_segment=reset_segment,
        policy_snapshot=application.generator.snapshot,
        **kwargs,
    )


__all__ = [
    "PI_EVAL_RULE",
    "SEGMENT_RESET_RULE",
    "STEP_COMMITTED_EVENT",
    "TRACE_ROW_FORMAT",
    "VALIDATION_EDGE_RULE",
    "VALIDATION_LATENCY_RULE",
    "VALIDATION_TASK_PREFIX",
    "VALIDATION_TOKENS_RULE",
    "R2FlowPhaseController",
    "controller_for_application",
    "heldout_trajectory_obs",
    "last_step_commits",
    "threadsafe_rollout_heldout",
    "trajectory_obs_from_value",
    "trajectory_obs_to_value",
    "validation_query_key",
]
