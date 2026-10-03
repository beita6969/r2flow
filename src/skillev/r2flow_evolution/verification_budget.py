from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Final

from skillev.contracts import JsonValue

from .readouts import _is_continuous, reconstruct_reference_flow
from .types import PhaseEvidence, TrajectoryObs, VerifierObs

FLOW_RANKED_BUDGET: Final = "flow-ranked-budget@1"
RANKING_RULE: Final = "reference-edge-flow-desc-then-trajectory-step@1"


def invocation_phi0(trajectories: tuple[TrajectoryObs, ...]) -> dict[tuple[str, int], float]:
    groups: dict[tuple[str, str], list[TrajectoryObs]] = {}
    seen: set[str] = set()
    for trajectory in trajectories:
        if trajectory.trajectory_id in seen or not trajectory.edges:
            continue
        seen.add(trajectory.trajectory_id)
        if not _is_continuous(trajectory):
            continue
        root = trajectory.edges[0].predecessor_key
        groups.setdefault((trajectory.query_id, root), []).append(trajectory)
    share: dict[tuple[str, int], float] = {}
    for _, group in sorted(groups.items()):
        try:
            flow = reconstruct_reference_flow(group)
        except ValueError:
            continue
        for trajectory in group:
            for position, edge in enumerate(trajectory.edges, start=1):
                key = (edge.predecessor_key, edge.event_label, edge.state_key)
                if key in flow.edge_flow:
                    share[(trajectory.trajectory_id, position)] = flow.phi0(key)
    return share


def apply_verification_budget(
    evidence: PhaseEvidence, per_family: int
) -> tuple[PhaseEvidence, dict[str, JsonValue]]:
    if type(per_family) is not int or per_family < 1:
        raise ValueError("the verification budget per family must be a positive integer")
    share = invocation_phi0(evidence.trajectories)
    families: dict[str, list[VerifierObs]] = defaultdict(list)
    for observation in evidence.verifier:
        if observation.gate_eligible:
            families[observation.z[0]].append(observation)
    admitted: set[tuple[str, int]] = set()
    report: dict[str, JsonValue] = {}
    for family, rows in sorted(families.items()):
        ranked = sorted(
            rows,
            key=lambda o: (
                -share.get((o.trajectory_id, o.step_index), 0.0),
                o.trajectory_id,
                o.step_index,
            ),
        )
        chosen = ranked[:per_family]
        admitted.update((o.trajectory_id, o.step_index) for o in chosen)
        report[family] = {
            "eligible": len(rows),
            "admitted": len(chosen),
            "min_admitted_phi0": min(
                (share.get((o.trajectory_id, o.step_index), 0.0) for o in chosen), default=0.0
            ),
        }
    kept = tuple(
        o
        for o in evidence.verifier
        if not o.gate_eligible or (o.trajectory_id, o.step_index) in admitted
    )
    summary: dict[str, JsonValue] = {
        "format": "r2flow-verification-budget@1",
        "policy": FLOW_RANKED_BUDGET,
        "ranking": RANKING_RULE,
        "per_family": per_family,
        "families": report,
    }
    return replace(evidence, verifier=kept), summary


__all__ = ["FLOW_RANKED_BUDGET", "apply_verification_budget", "invocation_phi0"]
