from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Final, cast

from skillev.contracts import JsonValue
from skillev.contracts.answer_writer import written_answer
from skillev.contracts.verifier_record import SOFT_ONLY_DOMAINS, EventClass, VerifierRecord

from .skill_md import forbidden_needle_patterns, needle_sha256, redact_forbidden
from .types import (
    EdgeObs,
    EventVerifierObs,
    LibraryVersion,
    PhaseEvidence,
    TrajectoryObs,
    VerifierObs,
)

SKILL_FUNCTION: Final = "invoke_skill"
COMMIT_EVENT: Final = "training_step_committed"
FLOW_EVENT: Final = "flow_step_recorded"
INDEX_FORMAT: Final = "committed-batch-evidence-index@1"
TRAINING_DATASET_RELPATH: Final = Path("inputs") / "retained-training.jsonl"
EVIDENCE_RULES: Final[Mapping[str, str]] = {
    "window": "committed-window-one-library@1",
    "query_id": "query-id=source-question-id@1",
    "tokens": "tokens=policy-reasoning+policy-action+recorded-executor@1",
    "edge_label": "edge-label=in-edge-label-hash@1",
    "legal_event_count": "legal-event-count=enum-product-text-as-one@1",
    "verifier": "skill-events-with-combined-evidence@1",
    "forbidden_strings": "forbidden-strings@1",
    "author_material": "author-material@1",
}
EVENT_RECORDS_RULE: Final = "event-verifier-records@1"
_ROLLOUT_SUFFIX = re.compile(r"/rollout-\d+$")


def key_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    )


def label_hash(event_label: str) -> str:
    return hashlib.sha256(key_json({"label": event_label}).encode("utf-8")).hexdigest()


def legal_set_sha256(legal: Mapping[str, Any]) -> str:
    return hashlib.sha256(key_json(legal).encode("utf-8")).hexdigest()


def legal_event_count(legal: Mapping[str, Any]) -> int:
    total = 0
    for function in legal.get("functions", ()):
        product = 1
        for param in function.get("params", ()):
            if param.get("kind") == "enum":
                product *= len(param.get("values", ()))
        total += product
    return total


def skill_id_of(event_label: str) -> str:
    try:
        parsed = json.loads(event_label)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invoke_skill event label is not JSON: {event_label[:80]!r}") from exc
    args = parsed.get("args") if isinstance(parsed, dict) else None
    skill = args.get("skill_id") if isinstance(args, dict) else None
    if parsed.get("u") != SKILL_FUNCTION or not isinstance(skill, str) or not skill:
        raise ValueError(f"invoke_skill event label has no skill_id: {event_label[:80]!r}")
    return skill


def _seconds(started_at: str, completed_at: str) -> float:
    start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    end = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
    return (end - start).total_seconds()


def _unwrap(value: Mapping[str, Any]) -> Mapping[str, Any]:
    artifact = value.get("artifact")
    return artifact if isinstance(artifact, Mapping) else value


def query_id_of(record: Mapping[str, Any], task_id: str | None = None) -> str:
    payload = record.get("reward", {}).get("native_payload") or {}
    source = payload.get("training_evidence_source") or payload.get("evaluation_evidence_source")
    if isinstance(source, Mapping) and isinstance(source.get("source_question_id"), str):
        return str(source["source_question_id"])
    if task_id is None:
        raise ValueError(f"trajectory {record.get('trajectory_id')!r} has no query identity")
    return _ROLLOUT_SUFFIX.sub("", task_id)


def _reward_eta(reward: float, *, eta: float, epsilon: float) -> float:
    return float((reward + epsilon) ** eta)


@dataclass(frozen=True, slots=True)
class CommittedStep:
    optimizer_step: int
    batch_id: str
    library_version: str
    records: tuple[Mapping[str, Any], ...]
    flow_records: Mapping[str, Mapping[str, Any]]
    artifacts: tuple[Mapping[str, Any], ...]
    sources: tuple[Mapping[str, Any] | None, ...]


def _read_line_at(path: Path, offset: int) -> tuple[dict[str, Any], int]:
    with path.open("rb") as stream:
        stream.seek(offset)
        line = stream.readline()
        if not line.endswith(b"\n"):
            raise ValueError(f"no complete event line at {path}:{offset}")
        return json.loads(line), stream.tell()


def _iter_lines(path: Path, offset: int = 0) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("rb") as stream:
        stream.seek(offset)
        while True:
            position = stream.tell()
            line = stream.readline()
            if not line or not line.endswith(b"\n"):
                return
            yield position, json.loads(line)


def _commit_offsets(run_root: Path, steps: Sequence[int]) -> dict[int, int]:
    offsets: dict[int, int] = {}
    for step in steps:
        index_path = run_root / "evidence" / f"step-{step:08d}.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text())
            if index.get("format") != INDEX_FORMAT or index.get("optimizer_step") != step:
                raise ValueError(f"{index_path} is not the committed index of step {step}")
            offsets[step] = int(index["source_event"]["byte_offset"])
    missing = set(steps) - set(offsets)
    if missing:
        events = run_root / "events.jsonl"
        for position, event in _iter_lines(events):
            if event.get("event_type") != COMMIT_EVENT:
                continue
            step = event["payload"].get("optimizer_step")
            if step in missing:
                if step in offsets:
                    raise ValueError(f"optimizer step {step} was committed twice")
                offsets[step] = position
        absent = sorted(set(steps) - set(offsets))
        if absent:
            raise ValueError(f"optimizer steps {absent} are not committed in {run_root}")
    return offsets


def load_committed_step(run_root: Path, step: int, *, offset: int | None = None) -> CommittedStep:
    events = run_root / "events.jsonl"
    if offset is None:
        offset = _commit_offsets(run_root, (step,))[step]
    commit, after = _read_line_at(events, offset)
    payload = commit.get("payload", {})
    if commit.get("event_type") != COMMIT_EVENT or payload.get("optimizer_step") != step:
        raise ValueError(f"{events}:{offset} is not the commit of optimizer step {step}")
    index_path = run_root / "evidence" / f"step-{step:08d}.json"
    sources: list[Mapping[str, Any] | None] = [None] * len(payload["records"])
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        if index.get("source_commit_id") != commit.get("event_id"):
            raise ValueError(f"{index_path} indexes another commit event")
        for row in index.get("trajectories", ()):
            sources[int(row["position"]) - 1] = row.get("source")
    flow: dict[str, Any] | None = None
    for _, event in _iter_lines(events, after):
        kind = event.get("event_type")
        if kind == FLOW_EVENT and event["payload"].get("optimizer_step") == step:
            flow = event["payload"]
            break
        if kind == COMMIT_EVENT:
            break
    if flow is None or flow.get("batch_id") != payload.get("batch_id"):
        raise ValueError(f"optimizer step {step} has no flow_step_recorded event after its commit")
    records = tuple(payload["records"])
    ids = [record["trajectory_id"] for record in records]
    flow_records = {record["trajectory_id"]: record for record in flow["records"]}
    if len(set(ids)) != len(ids) or set(flow_records) != set(ids):
        raise ValueError(f"step {step}: flow records do not cover the committed trajectories")
    directory = run_root / "inflight" / f"step-{step:08d}"
    artifacts = []
    for position, record in enumerate(records, 1):
        path = directory / f"trajectory-{position:06d}.json"
        artifact = _unwrap(json.loads(path.read_text()))
        if artifact.get("record") != record:
            raise ValueError(f"{path} differs from the committed record")
        if artifact.get("manifest", {}).get("library_version") != payload["library_version"]:
            raise ValueError(f"{path} was sampled on another library version")
        artifacts.append(artifact)
    return CommittedStep(
        optimizer_step=step,
        batch_id=str(payload["batch_id"]),
        library_version=str(payload["library_version"]),
        records=records,
        flow_records=flow_records,
        artifacts=tuple(artifacts),
        sources=tuple(sources),
    )


def iter_committed_steps(
    run_root: Path, since_step: int, until_step: int
) -> Iterator[CommittedStep]:
    if since_step < 0 or until_step <= since_step:
        raise ValueError("the evidence window (since_step, until_step] is empty")
    steps = list(range(since_step + 1, until_step + 1))
    offsets = _commit_offsets(run_root, steps)
    for step in steps:
        yield load_committed_step(run_root, step, offset=offsets[step])


def _legal_counts(
    artifacts: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for artifact in artifacts:
        inputs = artifact.get("verifier_inputs") or {}
        for legal in inputs.get("legal_event_sets", ()):
            counts.setdefault(legal_set_sha256(legal), legal_event_count(legal))
    return counts


def _edges(
    artifact: Mapping[str, Any],
    flow_record: Mapping[str, Any] | None,
    *,
    library: LibraryVersion | None,
    legal_by_sha: Mapping[str, int],
) -> tuple[EdgeObs, ...]:
    record = artifact["record"]
    steps = record["steps"]
    inputs = artifact.get("verifier_inputs") or {}
    legal_sets = inputs.get("legal_event_sets")
    flow_edges = None if flow_record is None else flow_record["edges"]
    if flow_edges is not None and len(flow_edges) != len(steps):
        raise ValueError(f"{record['trajectory_id']}: flow record and steps differ in horizon")
    edges = []
    for position, step in enumerate(steps):
        flow_step = step.get("r2flow")
        if flow_step is None:
            raise ValueError(f"{record['trajectory_id']}: step {position + 1} has no r2flow record")
        edge = flow_step if flow_edges is None else flow_edges[position]
        index = int(edge.get("step_index", step["index"]))
        if index != position + 1 or step["index"] != index:
            raise ValueError(f"{record['trajectory_id']}: steps are not ordered 1..T")
        for name in ("state_key", "predecessor_key", "event_label", "legal_event_set_sha256"):
            if edge[name] != flow_step[name]:
                raise ValueError(f"{record['trajectory_id']} step {index}: {name} differs")
        in_edges = tuple(
            (str(e["predecessor_key"]), str(e["label_hash"])) for e in edge["in_edges"]
        )
        actual = int(edge["actual_in_edge"])
        label = label_hash(str(edge["event_label"]))
        if not 0 <= actual < len(in_edges) or in_edges[actual] != (edge["predecessor_key"], label):
            raise ValueError(f"{record['trajectory_id']} step {index}: actual in-edge mismatch")
        function = str(edge["event_function"])
        skill = skill_id_of(str(edge["event_label"])) if function == SKILL_FUNCTION else None
        if skill is not None and library is not None and skill not in library.skill_ids:
            raise ValueError(f"skill {skill!r} is not in library version {library.version}")
        sha = str(edge["legal_event_set_sha256"])
        if legal_sets is not None:
            if legal_set_sha256(legal_sets[position]) != sha:
                raise ValueError(f"{record['trajectory_id']} step {index}: legal set hash differs")
            count = legal_event_count(legal_sets[position])
        else:
            count = legal_by_sha.get(sha, -1)
        edges.append(
            EdgeObs(
                step_index=index,
                predecessor_key=str(edge["predecessor_key"]),
                state_key=str(edge["state_key"]),
                in_edges=in_edges,
                event_label=label,
                event_function=function,
                skill_id=skill,
                legal_event_count=count,
                edge_residual=float(edge["edge_residual"]) if flow_edges is not None else math.nan,
            )
        )
    return tuple(edges)


def trajectory_from_artifact(
    artifact: Mapping[str, Any],
    *,
    eta: float,
    epsilon: float,
    flow_record: Mapping[str, Any] | None = None,
    library: LibraryVersion | None = None,
    query_id: str | None = None,
    legal_by_sha: Mapping[str, int] | None = None,
) -> TrajectoryObs:
    artifact = _unwrap(artifact)
    record = artifact["record"]
    manifest = artifact["manifest"]
    reward_block = record["reward"]
    reward = float(reward_block["value"])
    if not 0.0 <= reward <= 1.0:
        raise ValueError(f"{record['trajectory_id']}: task reward {reward} outside [0, 1]")
    shifted = record.get("shifted_reward")
    if shifted is not None and not math.isclose(float(shifted), reward + epsilon, abs_tol=1e-9):
        raise ValueError(f"{record['trajectory_id']}: epsilon differs from the recorded shift")
    reward_eta = _reward_eta(reward, eta=eta, epsilon=epsilon)
    executor_tokens = 0
    if flow_record is not None:
        logged = float(flow_record["log_reward_eta"])
        if not math.isclose(logged, eta * math.log(reward + epsilon), abs_tol=1e-6):
            raise ValueError(f"{record['trajectory_id']}: eta/epsilon differ from the flow record")
        recorded = (flow_record.get("episode") or {}).get("executor_tokens")
        executor_tokens = recorded if type(recorded) is int else 0
    inputs = artifact.get("verifier_inputs") or {}
    payload = reward_block.get("native_payload") or {}
    domain = (
        inputs.get("domain") or payload.get("benchmark_id") or record["task_family"].split("/")[0]
    )
    tokens = (
        sum(int(n) for n in manifest["reasoning_token_counts"])
        + sum(int(step["action_token_count"]) for step in record["steps"])
        + executor_tokens
    )
    return TrajectoryObs(
        trajectory_id=str(record["trajectory_id"]),
        query_id=query_id or query_id_of(record, manifest.get("task_id")),
        domain=str(domain),
        family=str(record["task_family"]),
        reward=reward,
        reward_eta=reward_eta,
        success=reward_block.get("success") is True,
        tokens=tokens,
        latency_seconds=_seconds(manifest["started_at"], manifest["completed_at"]),
        edges=_edges(artifact, flow_record, library=library, legal_by_sha=legal_by_sha or {}),
    )


def _verifier_obs(artifact: Mapping[str, Any], trajectory: TrajectoryObs) -> list[VerifierObs]:
    rows = artifact.get("verifier_records")
    if rows is None:
        return []
    domain = (artifact.get("verifier_inputs") or {}).get("domain")
    observations = []
    for row in rows:
        record = VerifierRecord.from_value(row)
        if record.trajectory_id != trajectory.trajectory_id or record.domain != domain:
            raise ValueError(f"verifier record {record.event_id} belongs to another trajectory")
        edge = trajectory.edges[record.step_index - 1]
        if record.event.event_class is not EventClass.SKILL:
            continue
        if edge.skill_id != record.event.unit:
            raise ValueError(
                f"{trajectory.trajectory_id} step {record.step_index}: verifier unit "
                f"{record.event.unit!r} differs from the event's skill {edge.skill_id!r}"
            )
        if record.combined_passed is None or record.combined_confidence is None:
            continue
        observations.append(
            VerifierObs(
                trajectory_id=trajectory.trajectory_id,
                step_index=record.step_index,
                skill_id=record.event.unit,
                z=(
                    record.z.context_class,
                    record.z.lagged_failure_mode,
                    str(record.z.token_bucket),
                    str(record.z.turn_bucket),
                ),
                y=1.0 if record.combined_passed else 0.0,
                confidence=float(record.combined_confidence),
                gate_eligible=record.gate_eligible,
            )
        )
    return observations


def _event_verifier_obs(
    artifact: Mapping[str, Any], trajectory: TrajectoryObs
) -> list[EventVerifierObs]:
    rows = artifact.get("verifier_records")
    if rows is None:
        return []
    domain = (artifact.get("verifier_inputs") or {}).get("domain")
    observations = []
    for row in rows:
        record = VerifierRecord.from_value(row)
        if record.trajectory_id != trajectory.trajectory_id or record.domain != domain:
            raise ValueError(f"verifier record {record.event_id} belongs to another trajectory")
        if record.event.event_class is EventClass.SKILL:
            continue
        if record.combined_passed is None or record.combined_confidence is None:
            continue
        passed = record.combined_passed
        deciding = tuple(
            sorted(
                item.verifier_id
                for item in record.components
                if (item.evidential if passed else not item.passed)
            )
        )
        observations.append(
            EventVerifierObs(
                trajectory_id=trajectory.trajectory_id,
                step_index=record.step_index,
                event_class=record.event.event_class.value,
                unit=record.event.unit,
                z=(
                    record.z.context_class,
                    record.z.lagged_failure_mode,
                    str(record.z.token_bucket),
                    str(record.z.turn_bucket),
                ),
                y=1.0 if passed else 0.0,
                confidence=float(record.combined_confidence),
                eligible=record.domain not in SOFT_ONLY_DOMAINS
                or any(item.evidential for item in record.components),
                verifiers=deciding,
            )
        )
    return observations


def _fill_legal_lower_bounds(trajectories: list[TrajectoryObs]) -> list[TrajectoryObs]:
    if all(edge.legal_event_count >= 0 for t in trajectories for edge in t.edges):
        return trajectories
    observed: dict[str, set[str]] = defaultdict(set)
    for trajectory in trajectories:
        for edge in trajectory.edges:
            observed[edge.predecessor_key].add(edge.event_label)
    filled = []
    for trajectory in trajectories:
        edges = tuple(
            edge
            if edge.legal_event_count >= 0
            else replace(edge, legal_event_count=len(observed[edge.predecessor_key]))
            for edge in trajectory.edges
        )
        filled.append(replace(trajectory, edges=edges))
    return filled


def build_phase_evidence(
    run_root: Path,
    library: LibraryVersion,
    *,
    phase: int,
    since_step: int,
    until_step: int,
    eta: float,
    epsilon: float,
    library_digest: str | None = None,
) -> PhaseEvidence:
    run_root = Path(run_root)
    steps = list(iter_committed_steps(run_root, since_step, until_step))
    versions = sorted({step.library_version for step in steps})
    if len(versions) != 1 or (library_digest is not None and versions != [library_digest]):
        raise ValueError(f"the evidence window spans library versions {versions}")
    legal_by_sha = _legal_counts(a for step in steps for a in step.artifacts)
    trajectories: list[TrajectoryObs] = []
    artifacts: list[Mapping[str, Any]] = []
    for step in steps:
        for record, artifact, source in zip(
            step.records, step.artifacts, step.sources, strict=True
        ):
            query = None
            if isinstance(source, Mapping) and isinstance(source.get("source_question_id"), str):
                query = str(source["source_question_id"])
            trajectory = trajectory_from_artifact(
                artifact,
                eta=eta,
                epsilon=epsilon,
                flow_record=step.flow_records[record["trajectory_id"]],
                library=library,
                query_id=query,
                legal_by_sha=legal_by_sha,
            )
            if query is not None and query != query_id_of(record, artifact["manifest"]["task_id"]):
                raise ValueError(f"{trajectory.trajectory_id}: index and record sources differ")
            trajectories.append(trajectory)
            artifacts.append(artifact)
    trajectories = _fill_legal_lower_bounds(trajectories)
    verifier: list[VerifierObs] = []
    events: list[EventVerifierObs] = []
    for trajectory, artifact in zip(trajectories, artifacts, strict=True):
        verifier.extend(_verifier_obs(artifact, trajectory))
        events.extend(_event_verifier_obs(artifact, trajectory))
    return PhaseEvidence(
        phase=phase,
        library=library,
        trajectories=tuple(trajectories),
        verifier=tuple(verifier),
        eta=eta,
        epsilon=epsilon,
        optimizer_step=until_step,
        event_verifier=tuple(events),
    )


def summarize_evidence(evidence: PhaseEvidence) -> dict[str, JsonValue]:
    edges = [edge for t in evidence.trajectories for edge in t.edges]
    skill_calls = Counter(edge.skill_id for edge in edges if edge.skill_id is not None)
    labelled = Counter((v.skill_id, v.gate_eligible) for v in evidence.verifier)
    domain_of = {t.trajectory_id: t.domain for t in evidence.trajectories}
    eligible_by_domain = Counter(
        domain_of[v.trajectory_id] for v in evidence.verifier if v.gate_eligible
    )
    labelled_by_domain = Counter(domain_of[v.trajectory_id] for v in evidence.verifier)
    states: dict[str, set[str]] = defaultdict(set)
    rollouts = Counter(t.query_id for t in evidence.trajectories)
    for trajectory in evidence.trajectories:
        for edge in trajectory.edges:
            states[trajectory.query_id].update((edge.predecessor_key, edge.state_key))
    per_query = sorted(len(v) for v in states.values())
    domains = Counter(t.domain for t in evidence.trajectories)
    return {
        "format": "r2flow-phase-evidence-summary@1",
        "rules": dict(EVIDENCE_RULES),
        "phase": evidence.phase,
        "optimizer_step": evidence.optimizer_step,
        "library_version": evidence.library.version,
        "trajectories": len(evidence.trajectories),
        "trajectories_per_domain": dict(sorted(domains.items())),
        "queries": len(rollouts),
        "rollouts_per_query": {
            str(n): count for n, count in sorted(Counter(rollouts.values()).items())
        },
        "edges": len(edges),
        "multi_in_edges": sum(1 for edge in edges if len(edge.in_edges) > 1),
        "skill_events": dict(sorted(skill_calls.items())),
        "skill_events_without_verifier_evidence": dict(
            sorted(
                (
                    skill,
                    count - labelled[(skill, True)] - labelled[(skill, False)],
                )
                for skill, count in skill_calls.items()
            )
        ),
        "verifier_records": len(evidence.verifier),
        "verifier_records_per_domain": dict(sorted(labelled_by_domain.items())),
        "gate_eligible_per_domain": dict(sorted(eligible_by_domain.items())),
        "gate_eligible_per_skill": dict(
            sorted((s, n) for (s, eligible), n in labelled.items() if eligible)
        ),
        "gate_eligible_y1": sum(1 for v in evidence.verifier if v.gate_eligible and v.y == 1.0),
        "distinct_states_per_query": {
            "min": per_query[0] if per_query else 0,
            "median": per_query[len(per_query) // 2] if per_query else 0,
            "max": per_query[-1] if per_query else 0,
            "total": sum(per_query),
        },
        "success_rate": (
            sum(t.success for t in evidence.trajectories) / len(evidence.trajectories)
            if evidence.trajectories
            else None
        ),
        **_event_record_summary(evidence, domain_of),
    }


def _event_record_summary(
    evidence: PhaseEvidence, domain_of: Mapping[str, str]
) -> dict[str, JsonValue]:
    eligible = [v for v in evidence.event_verifier if v.eligible]
    by_verifier = Counter(verifier for v in eligible for verifier in v.verifiers)
    return {
        "event_verifier_rule": EVENT_RECORDS_RULE,
        "event_verifier_records": len(evidence.event_verifier),
        "event_verifier_eligible": len(eligible),
        "event_verifier_eligible_y1": sum(1 for v in eligible if v.y == 1.0),
        "event_verifier_eligible_per_domain": dict(
            sorted(Counter(domain_of[v.trajectory_id] for v in eligible).items())
        ),
        "event_verifier_eligible_per_verifier": dict(sorted(by_verifier.items())),
    }


def locate_training_dataset(run_root: Path) -> Path:
    for directory in (Path(run_root).resolve(), *Path(run_root).resolve().parents):
        candidate = directory / TRAINING_DATASET_RELPATH
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no {TRAINING_DATASET_RELPATH} above {run_root}; pass dataset=")


def _hidden_assert_lines(text: object, public: str) -> list[str]:
    if not isinstance(text, str):
        return []
    lines = (line.strip() for line in text.splitlines())
    return [line for line in lines if line.startswith("assert") and line not in public]


def _target_strings(target: Mapping[str, Any], public: str) -> list[str]:
    found: list[str] = []
    answers = target.get("accepted_answers")
    if isinstance(answers, list):
        found.extend(item for item in answers if isinstance(item, str))
    for name in (
        "canonical_solution",
        "reference_solution",
        "ideal_completion",
        "reference_answer",
    ):
        value = target.get(name)
        if isinstance(value, str):
            found.append(value)
    found.extend(_hidden_assert_lines(target.get("assertion"), public))
    found.extend(_hidden_assert_lines(target.get("test"), public))
    for rubric in target.get("rubrics") or ():
        if isinstance(rubric, Mapping) and isinstance(rubric.get("criterion"), str):
            found.append(rubric["criterion"])
    return found


def forbidden_strings(
    run_root: Path,
    since_step: int,
    until_step: int,
    *,
    dataset: Path | None = None,
) -> frozenset[str]:
    run_root = Path(run_root)
    found: set[str] = set()
    sources: set[str] = set()
    for step in iter_committed_steps(run_root, since_step, until_step):
        for record, artifact in zip(step.records, step.artifacts, strict=True):
            sources.add(query_id_of(record, artifact["manifest"]["task_id"]))
            payload = record.get("reward", {}).get("native_payload") or {}
            diagnostics = payload.get("qa_diagnostics") or {}
            for alias in diagnostics.get("accepted_aliases") or ():
                if isinstance(alias, str):
                    found.add(alias)
    path = dataset if dataset is not None else locate_training_dataset(run_root)
    matched: set[str] = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            source = (row.get("episode") or {}).get("source_id")
            if source not in sources:
                continue
            matched.add(source)
            target = (row.get("output") or {}).get("target") or {}
            public = str((row.get("input") or {}).get("query") or "")
            found.update(_target_strings(target, public))
    missing = sorted(sources - matched)
    if missing:
        raise ValueError(f"training dataset {path} lacks window sources {missing[:5]}")
    return frozenset(s.strip() for s in found if s.strip())


REDACTED: Final = "[redacted]"
TRUNCATED: Final = " [truncated]"


def make_redactor(forbidden: Collection[str]) -> Callable[[str], str]:
    strings = sorted({s.strip() for s in forbidden if s.strip()}, key=lambda s: (-len(s), s))
    if not strings:
        return lambda text: text
    pattern = re.compile("|".join(rf"(?<!\w){re.escape(s)}(?!\w)" for s in strings), re.IGNORECASE)
    return lambda text: pattern.sub(REDACTED, text)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + TRUNCATED


def _event_args(event_label: str) -> dict[str, str] | None:
    try:
        parsed = json.loads(event_label)
    except json.JSONDecodeError:
        return None
    args = parsed.get("args") if isinstance(parsed, dict) else None
    if not isinstance(args, dict):
        return None
    return {str(k): v if isinstance(v, str) else key_json(v) for k, v in args.items()}


def _written_answer(function: str, observation_text: str) -> str | None:
    if function != "submit_answer":
        return None
    try:
        return written_answer(json.loads(observation_text))
    except json.JSONDecodeError:
        return None


def _step_arguments(function: str, label: str, item: Mapping[str, Any]) -> dict[str, str]:
    written = _written_answer(function, str(item["observation_text"]))
    if written is not None:
        return {"answer": written}
    return _event_args(label) or {"raw": str(item.get("action_text", ""))}


def _query_text(artifact: Mapping[str, Any]) -> str:
    contract = (artifact.get("initial_context") or {}).get("contract") or {}
    query = contract.get("query")
    if not isinstance(query, str):
        query = (artifact["record"].get("initial_context") or {}).get("query")
    if not isinstance(query, str):
        raise ValueError(f"{artifact['record']['trajectory_id']}: no public query text")
    return query


def build_author_material(
    run_root: Path,
    *,
    since_step: int,
    until_step: int,
    forbidden: Collection[str],
    max_chars_per_field: int = 1500,
    exclude_query_ids: Collection[str] = (),
) -> list[dict[str, JsonValue]]:
    return _material(
        Path(run_root),
        since_step=since_step,
        until_step=until_step,
        forbidden=forbidden,
        max_chars=max_chars_per_field,
        exclude_query_ids=exclude_query_ids,
    )


SUBMISSION_FUNCTION: Final = "submit_answer"
SOLVER_FAILING_QUERIES: Final = 10
SOLVER_ROLLOUTS_PER_FAILING_QUERY: Final = 2
SOLVER_SUCCESSES: Final = 4
SOLVER_DECISIVE_TURNS: Final = 2
SOLVER_REPEATED_ACTIONS: Final = 5
SOLVER_ACTION_CHARS: Final = 160
SOLVER_PASSAGE_CHARS: Final = 400
SOLVER_PASSAGE_MIN_CHARS: Final = 120
CRITERION_LEDGER_FORMAT: Final = "healthbench-criterion-ledger@1"


def clip_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = max(limit // 2, 1)
    return f"{text[:half]} [... {len(text) - 2 * half} characters omitted ...] {text[-half:]}"


def _string_leaves(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _string_leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from _string_leaves(item)


def _environment_texts(function: str, observation_text: str) -> list[str]:
    if function == SKILL_FUNCTION:
        return []
    try:
        parsed = json.loads(observation_text)
    except json.JSONDecodeError:
        return [observation_text]
    if function == SUBMISSION_FUNCTION and written_answer(parsed) is not None:
        parsed = {key: value for key, value in parsed.items() if key != "answer"}
    return list(_string_leaves(parsed))


def public_needles(needles: Mapping[str, re.Pattern[str]], public: str) -> frozenset[str]:
    nfkc = unicodedata.normalize("NFKC", public)
    folded = nfkc.casefold()
    return frozenset(
        needle
        for needle, pattern in needles.items()
        if pattern.search(nfkc) or pattern.search(folded)
    )


def _public_redactor(
    needles: Mapping[str, re.Pattern[str]], exempt: Collection[str]
) -> Callable[[str], str]:
    active = tuple(needles[needle] for needle in sorted(needles) if needle not in exempt)
    return lambda text: redact_forbidden(text, active, marker=REDACTED)[0]


def _solver_observation(function: str, text: str, max_chars: int) -> tuple[str | None, str | None]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None, text
    if not isinstance(parsed, dict):
        return None, text
    status = parsed.get("status") if isinstance(parsed.get("status"), str) else None
    if function == SKILL_FUNCTION:
        output = parsed.get("output")
        return (output if isinstance(output, str) else None), status
    passages = parsed.get("passages")
    if isinstance(passages, list):
        texts = [
            str(passage["text"])
            for passage in passages
            if isinstance(passage, Mapping) and isinstance(passage.get("text"), str)
        ]
        share = max(max_chars // max(len(texts), 1) - 45, SOLVER_PASSAGE_MIN_CHARS)
        limit = min(SOLVER_PASSAGE_CHARS, share)
        rows = [f"[{number}] {clip_middle(body, limit)}" for number, body in enumerate(texts, 1)]
        return None, "\n".join(rows) if rows else "(no passages returned)"
    body = parsed.get("text")
    if isinstance(body, str):
        title = parsed.get("title")
        return None, f"[{title}] {body}" if isinstance(title, str) and title else body
    return None, status if status is not None else text


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _max_turns(payload: Mapping[str, Any]) -> int | None:
    node: object = payload.get("training_evidence_source") or payload.get(
        "evaluation_evidence_source"
    )
    for key in ("reset_binding", "public_task", "budget_profile"):
        node = node.get(key) if isinstance(node, Mapping) else None
    value = node.get("max_turns") if isinstance(node, Mapping) else None
    return value if type(value) is int and value > 0 else None


def _terminal(step: Mapping[str, Any]) -> bool:
    try:
        parsed = json.loads(str(step["observation_text"]))
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and parsed.get("terminal") is True


def _termination(record: Mapping[str, Any]) -> str:
    steps = record["steps"]
    limit = _max_turns(record["reward"].get("native_payload") or {})
    if not steps:
        return "no step was taken"
    count = len(steps)
    if str(steps[-1]["r2flow"]["event_function"]) == SUBMISSION_FUNCTION:
        bound = f" of at most {limit}" if limit is not None else ""
        return f"the final answer was submitted at step {count}{bound}"
    if record["reward"].get("success") is True and _terminal(steps[-1]):
        return f"the environment ended the episode at step {count}"
    if limit is not None and count >= limit:
        return f"the turn budget ran out: {count} of {limit} steps used"
    if _terminal(steps[-1]):
        return f"the environment ended the episode at step {count}"
    return f"the episode ended after {count} steps without a submission"


def _incomplete_cause(lane: Mapping[str, Any], verdict: Mapping[str, Any]) -> str:
    profile = verdict.get("scorer_profile")
    profile = profile if isinstance(profile, Mapping) else {}
    timeout = _finite(profile.get("per_lane_timeout_seconds"))
    seconds = _finite(lane.get("candidate_seconds"))
    if timeout is None or timeout <= 0 or seconds is None:
        return "the code raised an error or timed out"
    if seconds < 0.5 * timeout:
        return f"the code raised an error after {seconds:.2f} s, no timeout"
    if seconds >= 0.9 * timeout:
        return f"the code timed out ({seconds:.0f} s of a {timeout:.0f} s limit)"
    return "the code raised an error or timed out"


def _code_diagnostics(kind: object, verdict: Mapping[str, Any]) -> str:
    parts = [f"failure kind {kind}" if isinstance(kind, str) and kind else "all tests passed"]
    lanes = verdict.get("lanes")
    for name, label in (("base", "base tests"), ("plus", "extended hidden tests")):
        lane = lanes.get(name) if isinstance(lanes, Mapping) else None
        if not isinstance(lane, Mapping) or type(lane.get("planned_inputs")) is not int:
            continue
        planned = int(lane["planned_inputs"])
        details = lane.get("details")
        passed = sum(item is True for item in details) if isinstance(details, list) else 0
        text = f"{label} {passed}/{planned} passed"
        completed = lane.get("completed_inputs")
        if type(completed) is int and completed < planned:
            cause = _incomplete_cause(lane, verdict)
            text += f" ({planned - completed} did not complete: {cause})"
        parts.append(text)
        for key in ("exception_type", "error_type"):
            if isinstance(lane.get(key), str) and lane[key]:
                parts.append(f"{label} exception {lane[key]}")
    syntax = verdict.get("syntax")
    if isinstance(syntax, Mapping) and syntax.get("status") == "invalid":
        error = syntax.get("error_type")
        line = syntax.get("line")
        where = f" at line {line}" if type(line) is int else ""
        parts.append(
            "syntax check failed" + (f" ({error}{where})" if isinstance(error, str) else "")
        )
    return "code tests: " + "; ".join(parts)


def _qa_diagnostics(qa: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    answer = qa.get("normalized_answer")
    aliases = [a for a in qa.get("normalized_aliases") or () if isinstance(a, str) and a.strip()]
    if not isinstance(answer, str) or not answer.strip():
        return "answer check: no answer (empty after normalisation)"
    if _finite(metrics.get("answer-exact-match")) == 1.0 or qa.get("exact_matching_alias_indices"):
        return "answer check: exact match with an accepted answer"
    f1 = _finite(metrics.get("answer-f1"))
    words = set(answer.split())
    if f1 is not None and f1 > 0.0:
        alias_words = [set(alias.split()) for alias in aliases]
        if any(words < other for other in alias_words):
            relation = "the answer is less specific than an accepted answer (a strict subset of it)"
        elif any(other < words for other in alias_words):
            relation = "the answer is more verbose than an accepted answer (it adds words to it)"
        else:
            relation = "the answer shares some words with an accepted answer"
        return (
            f"answer check: partial overlap with an accepted answer (token F1 {f1:.2f}); "
            f"{relation}: a granularity or alias mismatch, not necessarily a wrong entity"
        )
    squashed = "".join(answer.split())
    if any(
        (other := "".join(alias.split())) and (other in squashed or squashed in other)
        for alias in aliases
    ):
        return (
            "answer check: no word overlap after normalisation, but the answer and an accepted "
            "answer overlap once spaces and punctuation are ignored (a punctuation or "
            "hyphenation mismatch)"
        )
    return (
        "answer check: no word overlap with any accepted answer (a wrong entity, or an answer "
        "of the wrong kind or level of detail)"
    )


def _criterion_ledger(payload: Mapping[str, Any], run_root: Path) -> Mapping[str, Any] | None:
    reference = payload.get("criterion_ledger")
    if not isinstance(reference, Mapping) or reference.get("format") != CRITERION_LEDGER_FORMAT:
        return None
    if not isinstance(reference.get("path"), str) or not reference["path"]:
        return None
    recorded = Path(reference["path"])
    path = recorded if recorded.is_file() else run_root / recorded.parent.name / recorded.name
    if not path.is_file():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    return state if isinstance(state, Mapping) else None


def _tag_counts(rows: Iterable[tuple[str, float]]) -> str:
    counts: dict[str, list[float]] = defaultdict(list)
    for tag, points in rows:
        counts[tag].append(points)
    ordered = sorted(counts.items(), key=lambda item: (-sum(item[1]), item[0]))
    return ", ".join(
        f"{tag} {sum(points):g} pts ({len(points)} criteri{'on' if len(points) == 1 else 'a'})"
        for tag, points in ordered
    )


RUBRIC_THEMES: Final = (
    "communication",
    "complex_responses",
    "context_seeking",
    "emergency_referrals",
    "global_health",
    "health_data_tasks",
    "hedging",
)


def _theme_text(cluster: str) -> str:
    for theme in RUBRIC_THEMES:
        if cluster.startswith(theme + "_"):
            situation, _, behaviour = cluster[len(theme) + 1 :].partition("_")
            if situation and behaviour:
                return (
                    f"{theme.replace('_', ' ')} ({situation.replace('-', ' ')}): "
                    + behaviour.replace("_", " ")
                )
    return cluster.replace("_", " ")


def _rubric_diagnostics(payload: Mapping[str, Any], run_root: Path) -> list[str]:
    raw = _finite(payload.get("native_raw_score"))
    profile = payload.get("grader_profile")
    rule = profile.get("success") if isinstance(profile, Mapping) else None
    lines = [
        f"rubric grade: score {raw:.2f}"
        + (f" (success rule: {rule})" if isinstance(rule, str) and rule else "")
    ]
    triggered = payload.get("triggered-negative-rubric-count")
    negatives = payload.get("negative_criterion_count")
    if type(triggered) is int and type(negatives) is int and negatives > 0:
        lines.append(f"rubric penalties: {triggered} of {negatives} penalty criteria triggered")
    ledger = _criterion_ledger(payload, run_root)
    results = (ledger.get("diagnostics") or {}).get("criterion_results") if ledger else None
    if not isinstance(results, list):
        return lines
    missed: list[tuple[str, float]] = []
    penalties: list[tuple[str, float]] = []
    themes: Counter[str] = Counter()
    for row in results:
        if not isinstance(row, Mapping) or row.get("status") != "graded":
            continue
        points = _finite(row.get("points")) or 0.0
        tags = [tag for tag in row.get("tags") or () if isinstance(tag, str)]
        axes = [tag.split(":", 1)[1] for tag in tags if tag.startswith("axis:")] or ["other"]
        if points > 0 and row.get("criteria_met") is False:
            missed.extend((axis, points) for axis in axes)
            kind = "missed"
        elif points < 0 and row.get("criteria_met") is True:
            penalties.extend((axis, -points) for axis in axes)
            kind = "penalty"
        else:
            continue
        themes.update(
            f"{_theme_text(tag.split(':', 1)[1])} [{kind}]"
            for tag in tags
            if tag.startswith("cluster:")
        )
    if missed:
        lines.append("missed rubric points by axis: " + _tag_counts(missed))
    if penalties:
        lines.append("triggered penalty points by axis: " + _tag_counts(penalties))
    if themes:
        lines.append(
            "rubric themes of those criteria: "
            + "; ".join(
                f"{theme}" + (f" x{count}" if count > 1 else "")
                for theme, count in sorted(themes.items())
            )
        )
    return lines


def _grader_diagnostics(record: Mapping[str, Any], run_root: Path) -> list[str]:
    payload = record["reward"].get("native_payload") or {}
    metrics = payload.get("public_metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    lines: list[str] = []
    verdict = payload.get("native_verdict")
    if isinstance(verdict, Mapping) and "failure_kind" in payload:
        lines.append(_code_diagnostics(payload.get("failure_kind"), verdict))
    qa = payload.get("qa_diagnostics")
    accuracy = _finite(metrics.get("accuracy"))
    if isinstance(qa, Mapping):
        lines.append(_qa_diagnostics(qa, metrics))
    elif accuracy is not None:
        lines.append(
            "answer check: the final answer is "
            + ("correct" if accuracy >= 1.0 else "incorrect")
            + " (exact match)"
        )
    if _finite(payload.get("native_raw_score")) is not None:
        lines.extend(_rubric_diagnostics(payload, run_root))
    return lines


@dataclass(frozen=True, slots=True)
class _Rollout:
    order: tuple[int, int]
    record: Mapping[str, Any]
    artifact: Mapping[str, Any]
    query_id: str
    key: str
    query_text: str
    verifiers: tuple[VerifierRecord, ...]
    success: bool


def _most_recent(rollouts: Iterable[_Rollout]) -> list[_Rollout]:
    return sorted(rollouts, key=lambda rollout: rollout.order, reverse=True)


def _distinct_query_sample(rollouts: Sequence[_Rollout]) -> list[_Rollout]:
    groups: dict[str, list[_Rollout]] = defaultdict(list)
    for rollout in rollouts:
        groups[rollout.key].append(rollout)
    failing = sorted(
        (
            (_most_recent(r for r in members if not r.success), key)
            for key, members in groups.items()
            if any(not r.success for r in members)
        ),
        key=lambda item: (tuple(-n for n in item[0][0].order), item[1]),
    )[:SOLVER_FAILING_QUERIES]
    wins = {
        key: _most_recent(r for r in members if r.success)[0]
        for key, members in groups.items()
        if any(r.success for r in members)
    }
    contrast = [key for _, key in failing if key in wins][:SOLVER_SUCCESSES]
    chosen = {key for _, key in failing}
    others = _most_recent(win for key, win in wins.items() if key not in chosen)
    sample: list[_Rollout] = []
    for fails, key in failing:
        sample.extend(fails[:SOLVER_ROLLOUTS_PER_FAILING_QUERY])
        if key in contrast:
            sample.append(wins[key])
    sample.extend(others[: SOLVER_SUCCESSES - len(contrast)])
    return sample


def _action_key(function: str, arguments: Mapping[str, str]) -> str:
    values = ", ".join(value for _, value in sorted(arguments.items()))
    return clip_middle(f"{function}: {values}", SOLVER_ACTION_CHARS)


def _solver_entry(
    rollout: _Rollout,
    cluster: tuple[str, str],
    *,
    needles: Mapping[str, re.Pattern[str]],
    max_chars: int,
    run_root: Path,
) -> dict[str, JsonValue]:
    record, artifact = rollout.record, rollout.artifact
    items = record["steps"]
    public = [rollout.query_text]
    for item in items:
        public.extend(
            _environment_texts(str(item["r2flow"]["event_function"]), str(item["observation_text"]))
        )
    exempt = public_needles(needles, "\n".join(public))
    redact = _public_redactor(needles, exempt)

    def text(value: str, limit: int = max_chars) -> str:
        return clip_middle(redact(value), limit)

    verdicts: dict[int, JsonValue] = {
        verified.step_index: {
            "y": 1.0 if verified.combined_passed else 0.0,
            "confidence": verified.combined_confidence,
            "gate_eligible": verified.gate_eligible,
        }
        for verified in rollout.verifiers
        if verified.combined_passed is not None
    }
    decisive = {int(item["index"]) for item in items[-SOLVER_DECISIVE_TURNS:]}
    actions: Counter[str] = Counter()
    steps: list[JsonValue] = []
    for item in items:
        index = int(item["index"])
        function = str(item["r2flow"]["event_function"])
        label = str(item["r2flow"]["event_label"])
        arguments = _step_arguments(function, label, item)
        output, observation = _solver_observation(
            function, str(item["observation_text"]), max_chars
        )
        reasoning = item.get("reasoning_text")
        actions[_action_key(function, arguments)] += 1
        steps.append(
            {
                "step_index": index,
                "function": function,
                "arguments": {k: text(v) for k, v in sorted(arguments.items())},
                "skill_id": skill_id_of(label) if function == SKILL_FUNCTION else None,
                "executor_output": None if output is None else text(output),
                "observation": None if observation is None else text(observation),
                "verifier": verdicts.get(index),
                "reasoning": text(reasoning)
                if index in decisive and isinstance(reasoning, str) and reasoning.strip()
                else None,
            }
        )
    repeated: list[JsonValue] = [
        cast(JsonValue, [redact(action), count])
        for action, count in sorted(actions.items(), key=lambda item: (-item[1], item[0]))
        if count >= 2
    ][:SOLVER_REPEATED_ACTIONS]
    family = str(record["task_family"])
    return {
        "trajectory_id": str(record["trajectory_id"]),
        "query_id": rollout.query_id,
        "domain": str(
            (artifact.get("verifier_inputs") or {}).get("domain") or family.split("/")[0]
        ),
        "family": family,
        "cluster": list(cluster),
        "query_text": text(rollout.query_text, 2 * max_chars),
        "reward": float(record["reward"]["value"]),
        "success": rollout.success,
        "optimizer_step": rollout.order[0],
        "position": rollout.order[1],
        "public_forbidden_sha256": [needle_sha256(needle) for needle in sorted(exempt)],
        "diagnostics": [redact(line) for line in _grader_diagnostics(record, run_root)],
        "episode": {
            "steps": len(items),
            "termination": _termination(record),
            "repeated_actions": repeated,
        },
        "steps": steps,
    }


def _material(
    run_root: Path,
    *,
    since_step: int,
    until_step: int,
    forbidden: Collection[str],
    max_chars: int,
    exclude_query_ids: Collection[str],
) -> list[dict[str, JsonValue]]:
    needles = forbidden_needle_patterns(forbidden)
    excluded = set(exclude_query_ids)
    clusters: dict[tuple[str, str], list[_Rollout]] = defaultdict(list)
    for step in iter_committed_steps(run_root, since_step, until_step):
        for position, (record, artifact) in enumerate(
            zip(step.records, step.artifacts, strict=True), 1
        ):
            query = query_id_of(record, artifact["manifest"]["task_id"])
            if query in excluded:
                continue
            verifiers = tuple(
                VerifierRecord.from_value(row) for row in artifact.get("verifier_records") or ()
            )
            context_class = next((v.z.context_class for v in verifiers if v.z.context_class), None)
            family = str(record["task_family"])
            query_text = _query_text(artifact)
            clusters[(family, context_class or family.split("/")[-1])].append(
                _Rollout(
                    order=(step.optimizer_step, position),
                    record=record,
                    artifact=artifact,
                    query_id=query,
                    key=" ".join(query_text.split()),
                    query_text=query_text,
                    verifiers=verifiers,
                    success=record["reward"].get("success") is True,
                )
            )
    return [
        _solver_entry(rollout, cluster, needles=needles, max_chars=max_chars, run_root=run_root)
        for cluster in sorted(clusters)
        for rollout in _distinct_query_sample(clusters[cluster])
    ]


__all__ = [
    "EVIDENCE_RULES",
    "REDACTED",
    "SOLVER_FAILING_QUERIES",
    "SOLVER_ROLLOUTS_PER_FAILING_QUERY",
    "SOLVER_SUCCESSES",
    "SUBMISSION_FUNCTION",
    "CommittedStep",
    "build_author_material",
    "build_phase_evidence",
    "clip_middle",
    "forbidden_strings",
    "iter_committed_steps",
    "label_hash",
    "legal_event_count",
    "legal_set_sha256",
    "load_committed_step",
    "locate_training_dataset",
    "make_redactor",
    "public_needles",
    "query_id_of",
    "skill_id_of",
    "summarize_evidence",
    "trajectory_from_artifact",
]
