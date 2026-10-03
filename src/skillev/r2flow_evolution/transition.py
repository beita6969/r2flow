from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
import os
import tempfile
from collections.abc import Callable, Collection, Mapping, Sequence
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import Any, Final, Protocol

from skillev.contracts import JsonValue

from .verification_budget import apply_verification_budget
from .evidence import (
    build_author_material,
    build_phase_evidence,
    forbidden_strings,
    summarize_evidence,
)
from .amendments import RSI_AMENDMENTS
from .author_memory import load_previous_drafts
from .paired import RolloutHeldout, RunPaired, make_run_paired
from .types import (
    STRUCTURAL_EDITS,
    AuthoredEdit,
    CandidateEdit,
    GateDecision,
    LibraryVersion,
    PairedOutcome,
    PhaseEvidence,
    PhaseState,
    SkillSpec,
    serialized_fields,
)

PHASE_RECORD_FORMAT: Final = "r2flow-phase-record@1"
PHASE_RECORD_MARKER: Final = "phase-record.json"
TRANSITION_RULES: Final = (
    "phase-record=progressive-atomic-marker-last@1",
    "heldout-seed=phase-index@1",
    "author-input=structural-candidates-in-phi-order@1",
    "tost-margins=dict@1",
    "phi-config=signature-fields-from-config@1",
    "next-state-inputs@2",
    "completed-record-replay@1",
    "forbidden-strings-recorded-as-hash@1",
)
RSI_AMENDMENTS_RULE: Final = "rsi-amendments=config-subsets@1"


class EvolutionConfigLike(Protocol):
    @property
    def tau_c(self) -> float: ...
    @property
    def delta(self) -> float: ...
    @property
    def theta_low(self) -> float: ...
    @property
    def theta_mid(self) -> float: ...
    @property
    def theta_high(self) -> float: ...
    @property
    def n_min(self) -> float: ...
    @property
    def theta_h(self) -> float: ...
    @property
    def kappa_u(self) -> float: ...
    @property
    def cooldown_phases(self) -> int: ...
    @property
    def gamma_carry(self) -> float: ...
    @property
    def generate_min_support(self) -> float: ...
    @property
    def generate_min_failure_rate(self) -> float: ...
    @property
    def compress_min_context_overlap(self) -> float: ...
    @property
    def compress_max_reliability_gap(self) -> float: ...
    @property
    def tost_alpha(self) -> float: ...
    @property
    def tost_margins(self) -> Mapping[str, float] | Sequence[Sequence[Any]]: ...
    @property
    def max_edits_per_phase(self) -> int: ...
    @property
    def max_validations_per_phase(self) -> int: ...
    @property
    def author_model(self) -> str: ...
    @property
    def author_base_file(self) -> str | Path: ...
    @property
    def author_key_file(self) -> str | Path: ...
    @property
    def verification_budget_per_family(self) -> int: ...


def to_json(value: object) -> JsonValue:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Enum):
        return to_json(value.value)
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {name: to_json(item) for name, item in serialized_fields(value)}
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: to_json(value[key]) for key in sorted(value)}
        pairs = sorted(
            ((to_json(k), to_json(v)) for k, v in value.items()),
            key=lambda pair: json.dumps(pair[0], sort_keys=True),
        )
        return [{"key": k, "value": v} for k, v in pairs]
    if isinstance(value, set | frozenset):
        return sorted(
            (to_json(item) for item in value), key=lambda v: json.dumps(v, sort_keys=True)
        )
    if isinstance(value, list | tuple):
        return [to_json(item) for item in value]
    to_value = getattr(value, "to_value", None)
    if callable(to_value):
        return to_json(to_value())
    return repr(value)


def _encode(value: object) -> bytes:
    text = json.dumps(to_json(value), sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def atomic_write_json(path: Path, value: object) -> str:
    data = _encode(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return "sha256:" + hashlib.sha256(data).hexdigest()


def phase_record_dir(run_root: Path, phase: int) -> Path:
    return Path(run_root) / "evolution" / f"phase-{phase}"


def _skill_md_text(spec: SkillSpec) -> str:
    families = ", ".join(json.dumps(family) for family in spec.families)
    return (
        "---\n"
        f"name: {spec.name}\n"
        f"description: {json.dumps(spec.description, ensure_ascii=False)}\n"
        f'version: "{spec.version}"\n'
        f"applicability:\n  families: [{families}]\n"
        "---\n\n"
        f"{spec.body}"
    )


def _default_renderer() -> Callable[[SkillSpec], str]:
    try:
        renderer = _lazy("skill_md", "render_skill_md")
    except (ImportError, AttributeError):
        renderer = None
    return renderer if callable(renderer) else _skill_md_text


def _authored_json(edit: AuthoredEdit, render: Callable[[SkillSpec], str]) -> dict[str, JsonValue]:
    added: list[JsonValue] = []
    for spec in edit.added:
        row = to_json(spec)
        assert isinstance(row, dict)
        row["skill_md"] = render(spec)
        added.append(row)
    return {
        "candidate": to_json(edit.candidate),
        "added": added,
        "removed": list(edit.removed),
        "author_model": edit.author_model,
        "author_record": to_json(edit.author_record),
    }


def _margins(raw: Mapping[str, float] | Sequence[Sequence[Any]]) -> dict[str, float]:
    items = raw.items() if isinstance(raw, Mapping) else ((row[0], row[1]) for row in raw)
    return {str(name): float(margin) for name, margin in sorted(items)}


def make_phi_config(config: object, phi_config_cls: Callable[..., Any]) -> Any:
    names = inspect.signature(phi_config_cls).parameters
    return phi_config_cls(
        **{name: getattr(config, name) for name in names if hasattr(config, name)}
    )


def _lazy(module: str, name: str) -> Any:
    return getattr(import_module(f"{__package__}.{module}"), name)


def _skill_from_json(value: Mapping[str, Any]) -> SkillSpec:
    return SkillSpec(
        skill_id=str(value["skill_id"]),
        name=str(value["name"]),
        description=str(value["description"]),
        body=str(value["body"]),
        families=tuple(str(f) for f in value["families"]),
        version=int(value["version"]),
        parent_id=None if value.get("parent_id") is None else str(value["parent_id"]),
    )


def _library_from_json(value: Mapping[str, Any]) -> LibraryVersion:
    return LibraryVersion(
        version=int(value["version"]),
        skills=tuple(_skill_from_json(spec) for spec in value["skills"]),
    )


def _state_from_json(value: Mapping[str, Any]) -> PhaseState:
    carried: dict[tuple[str, tuple[str, ...]], tuple[float, float]] = {}
    raw = value.get("carried_counts") or []
    rows = raw if isinstance(raw, list) else []
    for row in rows:
        skill_id, z = row["key"]
        success, failure = row["value"]
        carried[(str(skill_id), tuple(str(c) for c in z))] = (float(success), float(failure))
    return PhaseState(
        phase=int(value["phase"]),
        carried_counts=carried,
        cooldown={str(k): int(v) for k, v in (value.get("cooldown") or {}).items()},
        retired=tuple(str(u) for u in value.get("retired") or ()),
    )


def replay_completed_phase(record: Path) -> tuple[GateDecision, PhaseState]:
    decision_json = json.loads((record / "decision.json").read_text())
    state_json = json.loads((record / "state.json").read_text())
    trace_row = dict(decision_json["trace_row"])
    trace_row["replayed_from"] = str(record)
    decision = GateDecision(
        accepted=(),
        rejected=(),
        library_after=_library_from_json(decision_json["library_after"]),
        trace_row=trace_row,
    )
    return decision, _state_from_json(state_json["after"])


def accepted_skill_ids(decision: GateDecision) -> tuple[str, ...]:
    ids: set[str] = set()
    for edit in decision.accepted:
        ids.update(edit.candidate.skill_ids)
        ids.update(spec.skill_id for spec in edit.added)
    return tuple(sorted(ids))


def _candidate_rows(candidates: Sequence[CandidateEdit]) -> list[JsonValue]:
    rows: list[JsonValue] = []
    for position, candidate in enumerate(candidates):
        row = to_json(candidate)
        assert isinstance(row, dict)
        row["position"] = position
        row["structural"] = candidate.kind in STRUCTURAL_EDITS
        rows.append(row)
    return rows


def _unchanged(
    library: LibraryVersion,
    *,
    phase: int,
    outcome: str,
    candidates: Sequence[CandidateEdit],
    delta_k: Mapping[str, JsonValue],
) -> GateDecision:
    return GateDecision(
        accepted=(),
        rejected=(),
        library_after=library,
        trace_row={
            "k": phase,
            "v_k": library.version,
            "v_k_plus_1": library.version,
            "delta_k": dict(delta_k),
            "d_k": [
                {
                    "kind": str(c.kind),
                    "skill_ids": list(c.skill_ids),
                    "context": None if c.context is None else list(c.context),
                    "decision": "not-structural" if c.kind not in STRUCTURAL_EDITS else outcome,
                }
                for c in candidates
            ],
            "val_k": [],
            "outcome": outcome,
        },
    )


def _check_gate_invariants(
    decision: GateDecision, library: LibraryVersion, authored: Sequence[AuthoredEdit]
) -> None:
    if not decision.accepted and decision.library_after != library:
        raise RuntimeError("the gate changed the library without an accepted edit (Prop. C.1)")
    for edit in decision.accepted:
        if edit.candidate.kind not in STRUCTURAL_EDITS or not any(edit is a for a in authored):
            raise RuntimeError(
                "the gate accepted an edit that was not an authored structural draft"
            )


def run_phase_transition(
    *,
    run_root: Path,
    library: LibraryVersion,
    state: PhaseState,
    phase: int,
    optimizer_step: int,
    since_step: int,
    eta: float,
    epsilon: float,
    config: EvolutionConfigLike,
    rollout_heldout: RolloutHeldout,
    heldout_seed: int | None = None,
    library_digest: str | None = None,
    dataset: Path | None = None,
    diagnostics: Mapping[str, JsonValue] | None = None,
    author_timeout_s: float = 600.0,
    author_client: object | None = None,
    heldout_query_ids: Collection[str] = (),
    material_chars: int = 1500,
    build_evidence: Callable[..., PhaseEvidence] = build_phase_evidence,
    gold_strings: Callable[..., frozenset[str]] = forbidden_strings,
    author_material: Callable[..., list[dict[str, JsonValue]]] = build_author_material,
    compute_readouts: Callable[..., Any] | None = None,
    verifier_posterior: Callable[..., Any] | None = None,
    phi_config_cls: Callable[..., Any] | None = None,
    propose_edits: Callable[..., Sequence[CandidateEdit]] | None = None,
    next_phase_state: Callable[..., PhaseState] | None = None,
    author_edits: Callable[..., Sequence[AuthoredEdit]] | None = None,
    author_client_factory: Callable[..., object] | None = None,
    gate: Callable[..., GateDecision] | None = None,
    tost: Callable[..., Any] | None = None,
    run_paired_factory: Callable[..., RunPaired] = make_run_paired,
    render_skill_md: Callable[[SkillSpec], str] | None = None,
) -> tuple[GateDecision, PhaseState]:
    run_root = Path(run_root)
    if state.phase != phase:
        raise ValueError(f"state.phase {state.phase} is not the closing phase {phase}")
    record = phase_record_dir(run_root, phase)
    if (record / PHASE_RECORD_MARKER).exists():
        return replay_completed_phase(record)
    compute_readouts = compute_readouts or _lazy("readouts", "compute_readouts")
    verifier_posterior = verifier_posterior or _lazy("posterior", "verifier_posterior")
    phi_config_cls = phi_config_cls or _lazy("phi", "PhiConfig")
    propose_edits = propose_edits or _lazy("phi", "propose_edits")
    next_phase_state = next_phase_state or _lazy("phi", "next_phase_state")
    render = render_skill_md or _default_renderer()
    hashes: dict[str, str] = {}

    def write(name: str, value: object) -> None:
        hashes[name] = atomic_write_json(record / name, value)

    evidence_kwargs: dict[str, Any] = {}
    if library_digest is not None:
        evidence_kwargs["library_digest"] = library_digest
    evidence = build_evidence(
        run_root,
        library,
        phase=phase,
        since_step=since_step,
        until_step=optimizer_step,
        eta=eta,
        epsilon=epsilon,
        **evidence_kwargs,
    )
    evidence, budget_report = apply_verification_budget(
        evidence, config.verification_budget_per_family
    )
    write("verification-budget.json", budget_report)
    summary = summarize_evidence(evidence)
    write("evidence-summary.json", {**summary, "since_step": since_step})
    readouts = compute_readouts(evidence, tau_c=config.tau_c)
    write("readouts.json", readouts)
    posterior = verifier_posterior(evidence, state, kappa_u=config.kappa_u, delta=config.delta)
    write("posterior.json", posterior)
    cfg = make_phi_config(config, phi_config_cls)
    candidates = list(propose_edits(evidence, readouts, posterior, state, cfg))
    structural = [c for c in candidates if c.kind in STRUCTURAL_EDITS]
    write("candidates.json", {"phi_config": cfg, "candidates": _candidate_rows(candidates)})
    delta_k: dict[str, JsonValue] = {
        "evidence": {
            key: summary[key]
            for key in ("trajectories", "queries", "edges", "verifier_records", "skill_events")
        },
        "diagnostics": to_json(dict(diagnostics or {})),
    }
    authored: list[AuthoredEdit] = []
    validations: list[JsonValue] = []
    tests: list[JsonValue] = []
    gold_summary: dict[str, JsonValue] = {"count": 0, "sha256": None}
    if not structural:
        decision = _unchanged(
            library,
            phase=phase,
            outcome="no-structural-candidates",
            candidates=candidates,
            delta_k=delta_k,
        )
    else:
        gold_kwargs: dict[str, Any] = {} if dataset is None else {"dataset": dataset}
        gold = gold_strings(run_root, since_step, optimizer_step, **gold_kwargs)
        gold_summary = {
            "count": len(gold),
            "sha256": "sha256:"
            + hashlib.sha256("\n".join(sorted(gold)).encode("utf-8")).hexdigest(),
        }
        client = author_client
        if client is None:
            factory = author_client_factory or _lazy("author", "GatewayAuthorClient")
            client = factory(
                base_file=config.author_base_file,
                key_file=config.author_key_file,
                model=config.author_model,
                timeout_s=author_timeout_s,
            )
        material = author_material(
            run_root,
            since_step=since_step,
            until_step=optimizer_step,
            forbidden=gold,
            max_chars_per_field=material_chars,
            exclude_query_ids=heldout_query_ids,
        )
        write("author-material.json", {"material": material})
        author = author_edits or _lazy("author", "author_edits")
        author_failures: list[dict[str, JsonValue]] = []
        authored = list(
            author(
                structural,
                evidence,
                library,
                client,
                max_edits=config.max_edits_per_phase,
                forbidden_strings=gold,
                material=material,
                failures=author_failures,
                previous_drafts=load_previous_drafts(run_root, phase),
            )
        )
        write(
            "authored.json",
            {
                "forbidden_strings": gold_summary,
                "drafts": [_authored_json(e, render) for e in authored],
                "failures": author_failures,
            },
        )
        if not authored:
            decision = _unchanged(
                library,
                phase=phase,
                outcome="no-authored-drafts",
                candidates=candidates,
                delta_k=delta_k,
            )
        else:
            rollout_records: list[dict[str, JsonValue]] = []
            base_run_paired = run_paired_factory(
                rollout_heldout,
                seed=phase if heldout_seed is None else heldout_seed,
                observer=rollout_records.append,
            )
            tost_fn = tost or _lazy("tost", "tost_noninferior")

            def run_paired(
                current: LibraryVersion, candidate: LibraryVersion
            ) -> list[PairedOutcome]:
                rollout_records.clear()
                pairs = base_run_paired(current, candidate)
                row: dict[str, JsonValue] = {
                    "call": len(validations),
                    "current_version": current.version,
                    "current_skill_ids": list(current.skill_ids),
                    "candidate_version": candidate.version,
                    "candidate_skill_ids": list(candidate.skill_ids),
                    "pairs": to_json(pairs),
                }
                if rollout_records:
                    row["rollout"] = rollout_records[-1]
                validations.append(row)
                write("validations.json", {"paired": validations, "tost": tests})
                return pairs

            def recorded_tost(pairs: Sequence[PairedOutcome], **kwargs: Any) -> Any:
                result = tost_fn(pairs, **kwargs)
                entry: dict[str, JsonValue] = {
                    "call": len(tests),
                    "n": len(pairs),
                    "result": to_json(result),
                }
                tests.append(entry)
                write("validations.json", {"paired": validations, "tost": tests})
                return result

            gate_fn = gate or _lazy("gate", "gate")
            decision = gate_fn(
                authored,
                library=library,
                run_paired=run_paired,
                tost=recorded_tost,
                margins=_margins(config.tost_margins),
                alpha=config.tost_alpha,
                state=state,
                max_validations=config.max_validations_per_phase,
                phase=phase,
            )
            _check_gate_invariants(decision, library, authored)
    write("validations.json", {"paired": validations, "tost": tests})
    accepted_ids = accepted_skill_ids(decision)
    accepted_targets = {u for edit in decision.accepted for u in edit.candidate.skill_ids}
    state_after = next_phase_state(state, posterior, candidates, accepted_targets, cfg)
    rule_ids: list[JsonValue] = [*TRANSITION_RULES, RSI_AMENDMENTS_RULE, *RSI_AMENDMENTS]
    decision = dataclasses.replace(
        decision,
        trace_row={
            **decision.trace_row,
            "transition": {
                "format": PHASE_RECORD_FORMAT,
                "phase_record": str(record.relative_to(run_root)),
                "optimizer_step": optimizer_step,
                "since_step": since_step,
                "candidates": len(candidates),
                "structural_candidates": len(structural),
                "authored": len(authored),
                "accepted_skill_ids": list(accepted_ids),
                "evidence": delta_k["evidence"],
                "diagnostics": delta_k["diagnostics"],
                "rules": rule_ids,
            },
        },
    )
    write(
        "decision.json",
        {
            "accepted": [_authored_json(e, render) for e in decision.accepted],
            "rejected": [
                {"edit": _authored_json(e, render), "reason": reason}
                for e, reason in decision.rejected
            ],
            "library_before": library,
            "library_after": decision.library_after,
            "trace_row": decision.trace_row,
        },
    )
    write("state.json", {"before": state, "after": state_after, "accepted_skill_ids": accepted_ids})
    atomic_write_json(
        record / PHASE_RECORD_MARKER,
        {
            "format": PHASE_RECORD_FORMAT,
            "phase": phase,
            "optimizer_step": optimizer_step,
            "since_step": since_step,
            "heldout_seed": phase if heldout_seed is None else heldout_seed,
            "library_before": library.version,
            "library_after": decision.library_after.version,
            "forbidden_strings": gold_summary,
            "files": dict(sorted(hashes.items())),
            "rules": rule_ids,
        },
    )
    return decision, state_after


__all__ = [
    "PHASE_RECORD_FORMAT",
    "PHASE_RECORD_MARKER",
    "TRANSITION_RULES",
    "EvolutionConfigLike",
    "accepted_skill_ids",
    "atomic_write_json",
    "make_phi_config",
    "phase_record_dir",
    "run_phase_transition",
    "to_json",
]
