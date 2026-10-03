from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from skillev.contracts import JsonValue

from .types import PhaseEvidence, TrajectoryObs

EdgeKey = tuple[str, str, str]

INVOKE_SKILL: Final = "invoke_skill"
RULES: Final[dict[str, str]] = {
    "graph": "union-dag-per-query@1",
    "backward_policy": "uniform-observed-in-edges@1",
    "terminal_anchor": "terminal-stop-mass@1",
    "terminal_conflict": "terminal-reward-mean@1",
    "psi0_pooling": "residual-weighted-rollout-share@1",
    "q_estimator": "per-query-state-event-mean@1",
    "nu": "uniform-observed-events-nu@1",
    "a_util_undefined": "single-event-state-undefined@1",
    "atilde_skill": "mean-over-defined-invocations@1",
}
PSI0_ROLLOUT_RULE: Final = "residual-weighted-rollout-share@1"
_REWARD_TOLERANCE: Final = 1e-12


@dataclass(frozen=True, slots=True)
class Readouts:
    psi0: dict[str, float]
    n_call: dict[str, int]
    n_edge: dict[str, int]
    atilde_edge: dict[tuple[str, int], float]
    atilde_skill: dict[str, float]
    m_skill: float
    diagnostics: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class ReferenceFlow:
    root: str
    z_star: float
    state_flow: dict[str, float]
    edge_flow: dict[EdgeKey, float]
    backward: dict[EdgeKey, float]
    terminal_reward: dict[str, float]
    edge_skill: dict[EdgeKey, str | None]
    rank: dict[str, int]
    rank_inconsistent_states: tuple[str, ...]
    reward_inconsistent_terminals: tuple[str, ...]
    interior_terminal_states: tuple[str, ...]
    unobserved_in_edges: int

    def forward(self, edge: EdgeKey) -> float:
        return self.edge_flow[edge] / self.state_flow[edge[0]]

    def phi0(self, edge: EdgeKey) -> float:
        return self.edge_flow[edge] / self.z_star

    def skill_mass(self) -> dict[str, float]:
        mass: dict[str, float] = {}
        for edge, skill in self.edge_skill.items():
            if skill is not None:
                mass[skill] = mass.get(skill, 0.0) + self.phi0(edge)
        return mass


def _json_list(items: Iterable[str]) -> list[JsonValue]:
    out: list[JsonValue] = []
    out.extend(items)
    return out


def _is_skill_edge(event_function: str, skill_id: str | None) -> bool:
    return event_function == INVOKE_SKILL and skill_id is not None


def _is_continuous(trajectory: TrajectoryObs) -> bool:
    edges = trajectory.edges
    return all(edges[i].predecessor_key == edges[i - 1].state_key for i in range(1, len(edges)))


def reconstruct_reference_flow(
    trajectories: Sequence[TrajectoryObs],
    *,
    backward_weight: Callable[[EdgeKey], float] | None = None,
) -> ReferenceFlow:
    if not trajectories:
        raise ValueError("at least one trajectory is required")
    roots = {t.edges[0].predecessor_key for t in trajectories if t.edges}
    if any(not t.edges for t in trajectories) or len(roots) != 1:
        raise ValueError("trajectories must be non-empty and share one root")
    if not all(_is_continuous(t) for t in trajectories):
        raise ValueError("trajectories must be continuous")
    (root,) = roots

    edge_skill: dict[EdgeKey, str | None] = {}
    in_edges: dict[str, list[EdgeKey]] = {}
    out_edges: dict[str, list[EdgeKey]] = {}
    declared_in: dict[str, int] = {}
    terminal_values: dict[str, list[float]] = {}
    nodes: dict[str, None] = {root: None}
    for trajectory in trajectories:
        for obs in trajectory.edges:
            key = (obs.predecessor_key, obs.event_label, obs.state_key)
            nodes.setdefault(obs.predecessor_key, None)
            nodes.setdefault(obs.state_key, None)
            declared_in[obs.state_key] = max(
                declared_in.get(obs.state_key, 0), len(set(obs.in_edges))
            )
            if key not in edge_skill:
                skill = obs.skill_id if _is_skill_edge(obs.event_function, obs.skill_id) else None
                edge_skill[key] = skill
                in_edges.setdefault(obs.state_key, []).append(key)
                out_edges.setdefault(obs.predecessor_key, []).append(key)
        terminal_values.setdefault(trajectory.terminal_key, []).append(trajectory.reward_eta)

    indegree = {node: len(in_edges.get(node, ())) for node in nodes}
    ready = sorted(node for node, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while ready:
        node = ready.pop()
        order.append(node)
        released = []
        for edge in out_edges.get(node, ()):
            indegree[edge[2]] -= 1
            if indegree[edge[2]] == 0:
                released.append(edge[2])
        ready.extend(sorted(released, reverse=True))
    if len(order) != len(nodes):
        raise ValueError("the union graph has a cycle")
    longest: dict[str, int] = {root: 0}
    shortest: dict[str, int] = {root: 0}
    for node in order:
        if node not in longest:
            raise ValueError(f"state {node!r} is not reachable from the root")
        for edge in out_edges.get(node, ()):
            child = edge[2]
            longest[child] = max(longest.get(child, 0), longest[node] + 1)
            shortest[child] = min(shortest.get(child, shortest[node] + 1), shortest[node] + 1)
    rank_inconsistent = tuple(sorted(n for n in nodes if longest[n] != shortest[n]))

    terminal_reward: dict[str, float] = {}
    reward_conflicts: list[str] = []
    for state, values in terminal_values.items():
        if any(not math.isfinite(v) or v <= 0.0 for v in values):
            raise ValueError(f"reward_eta at terminal {state!r} must be finite and positive")
        terminal_reward[state] = sum(values) / len(values)
        if max(values) - min(values) > _REWARD_TOLERANCE * max(1.0, max(values)):
            reward_conflicts.append(state)

    backward: dict[EdgeKey, float] = {}
    for state, incoming in in_edges.items():
        if backward_weight is None:
            for edge in incoming:
                backward[edge] = 1.0 / len(incoming)
        else:
            weights = [backward_weight(edge) for edge in incoming]
            if any(not math.isfinite(w) or w <= 0.0 for w in weights):
                raise ValueError(f"backward weights into {state!r} must be finite and positive")
            total = sum(weights)
            for edge, weight in zip(incoming, weights, strict=True):
                backward[edge] = weight / total

    state_flow: dict[str, float] = {}
    edge_flow: dict[EdgeKey, float] = {}
    for node in reversed(order):
        flow = terminal_reward.get(node, 0.0)
        for edge in out_edges.get(node, ()):
            mass = state_flow[edge[2]] * backward[edge]
            edge_flow[edge] = mass
            flow += mass
        state_flow[node] = flow

    unobserved = sum(
        max(0, declared_in.get(node, 0) - len(in_edges.get(node, ()))) for node in nodes
    )
    return ReferenceFlow(
        root=root,
        z_star=state_flow[root],
        state_flow=state_flow,
        edge_flow=edge_flow,
        backward=backward,
        terminal_reward=terminal_reward,
        edge_skill=edge_skill,
        rank=longest,
        rank_inconsistent_states=rank_inconsistent,
        reward_inconsistent_terminals=tuple(sorted(reward_conflicts)),
        interior_terminal_states=tuple(sorted(s for s in terminal_reward if out_edges.get(s))),
        unobserved_in_edges=unobserved,
    )


def _signed_utility(
    trajectories: Sequence[TrajectoryObs], *, tau_c: float
) -> tuple[dict[tuple[str, int], float], int, int]:
    rewards: dict[str, dict[str, dict[str, float]]] = {}
    for trajectory in trajectories:
        for obs in trajectory.edges:
            by_event = rewards.setdefault(obs.predecessor_key, {})
            by_event.setdefault(obs.event_label, {})[trajectory.trajectory_id] = (
                trajectory.reward_eta
            )
    q_value: dict[str, dict[str, float]] = {
        state: {label: sum(r.values()) / len(r) for label, r in by_event.items()}
        for state, by_event in rewards.items()
    }
    defined: dict[tuple[str, int], float] = {}
    undefined = 0
    nonfinite = 0
    for trajectory in trajectories:
        for obs in trajectory.edges:
            events = q_value[obs.predecessor_key]
            if len(events) < 2:
                undefined += 1
                continue
            if not math.isfinite(obs.edge_residual):
                nonfinite += 1
                continue
            baseline = sum(events.values()) / len(events)
            a_util = events[obs.event_label] - baseline
            defined[(trajectory.trajectory_id, obs.step_index)] = a_util * math.exp(
                -abs(obs.edge_residual) / tau_c
            )
    return defined, undefined, nonfinite


def compute_readouts(evidence: PhaseEvidence, *, tau_c: float) -> Readouts:
    if not (math.isfinite(tau_c) and tau_c > 0.0):
        raise ValueError("tau_c must be finite and positive")

    library_ids = evidence.library.skill_ids
    n_call: dict[str, int] = dict.fromkeys(library_ids, 0)
    seen_ids: set[str] = set()
    duplicates: list[str] = []
    empty: list[str] = []
    discontinuous: list[str] = []
    components: dict[tuple[str, str], list[TrajectoryObs]] = {}
    call_edges: dict[tuple[str, int], str] = {}
    for trajectory in evidence.trajectories:
        if trajectory.trajectory_id in seen_ids:
            duplicates.append(trajectory.trajectory_id)
            continue
        seen_ids.add(trajectory.trajectory_id)
        for obs in trajectory.edges:
            if _is_skill_edge(obs.event_function, obs.skill_id):
                assert obs.skill_id is not None
                n_call[obs.skill_id] = n_call.get(obs.skill_id, 0) + 1
                call_edges[(trajectory.trajectory_id, obs.step_index)] = obs.skill_id
        if not trajectory.edges:
            empty.append(trajectory.trajectory_id)
            continue
        if not _is_continuous(trajectory):
            discontinuous.append(trajectory.trajectory_id)
            continue
        root = trajectory.edges[0].predecessor_key
        components.setdefault((trajectory.query_id, root), []).append(trajectory)

    roots_per_query: dict[str, int] = {}
    for query_id, _root in components:
        roots_per_query[query_id] = roots_per_query.get(query_id, 0) + 1

    mass: dict[str, float] = dict.fromkeys(library_ids, 0.0)
    n_edge: dict[str, int] = dict.fromkeys(library_ids, 0)
    atilde_edge: dict[tuple[str, int], float] = {}
    cyclic: list[JsonValue] = []
    rank_bad = 0
    reward_bad: list[JsonValue] = []
    interior = 0
    unobserved = 0
    merged_states = 0
    n_states = 0
    n_edges = 0
    undefined_total = 0
    nonfinite_total = 0
    per_query_m: list[float] = []
    for (query_id, root), group in sorted(components.items()):
        defined, undefined, nonfinite = _signed_utility(group, tau_c=tau_c)
        atilde_edge.update(defined)
        undefined_total += undefined
        nonfinite_total += nonfinite
        try:
            flow = reconstruct_reference_flow(group)
        except ValueError as error:
            cyclic.append(_json_list((query_id, root, str(error))))
            continue
        component_mass = flow.skill_mass()
        for skill, value in component_mass.items():
            mass[skill] = mass.get(skill, 0.0) + value
        for edge_skill in flow.edge_skill.values():
            if edge_skill is not None:
                n_edge[edge_skill] = n_edge.get(edge_skill, 0) + 1
        per_query_m.append(sum(component_mass.values()))
        rank_bad += len(flow.rank_inconsistent_states)
        reward_bad.extend(_json_list((query_id, s)) for s in flow.reward_inconsistent_terminals)
        interior += len(flow.interior_terminal_states)
        unobserved += flow.unobserved_in_edges
        n_states += len(flow.state_flow)
        n_edges += len(flow.edge_flow)
        in_degree: dict[str, int] = {}
        for edge in flow.edge_flow:
            in_degree[edge[2]] = in_degree.get(edge[2], 0) + 1
        merged_states += sum(1 for degree in in_degree.values() if degree >= 2)

    m_skill = sum(mass.values())
    psi0, psi0_defined, psi0_diagnostics = _rollout_weighted_psi0(components, library_ids)

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for key, skill in call_edges.items():
        if key in atilde_edge:
            sums[skill] = sums.get(skill, 0.0) + atilde_edge[key]
            counts[skill] = counts.get(skill, 0) + 1
    atilde_skill = {skill: sums[skill] / counts[skill] for skill in sorted(sums)}

    rules: dict[str, JsonValue] = dict(RULES)
    diagnostics: dict[str, JsonValue] = {
        "rules": rules,
        "tau_c": tau_c,
        "psi0_defined": psi0_defined,
        "n_trajectories": len(evidence.trajectories),
        "n_components": len(components),
        "n_queries": len(roots_per_query),
        "m_skill_mean_per_component": (sum(per_query_m) / len(per_query_m) if per_query_m else 0.0),
        "n_states": n_states,
        "n_edges": n_edges,
        "merged_states": merged_states,
        "multi_root_queries": _json_list(sorted(q for q, n in roots_per_query.items() if n > 1)),
        "duplicate_trajectory_ids": _json_list(sorted(duplicates)),
        "empty_trajectories": _json_list(sorted(empty)),
        "discontinuous_trajectories": _json_list(sorted(discontinuous)),
        "skipped_components": cyclic,
        "rank_inconsistent_states": rank_bad,
        "reward_inconsistent_terminals": reward_bad,
        "interior_terminal_states": interior,
        "unobserved_in_edges": unobserved,
        "atilde_defined_edges": len(atilde_edge),
        "atilde_undefined_edges": undefined_total,
        "atilde_nonfinite_residual_edges": nonfinite_total,
        **psi0_diagnostics,
    }
    return Readouts(
        psi0=psi0,
        n_call=dict(sorted(n_call.items())),
        n_edge=dict(sorted(n_edge.items())),
        atilde_edge=atilde_edge,
        atilde_skill=atilde_skill,
        m_skill=m_skill,
        diagnostics=diagnostics,
    )


def _rollout_weighted_psi0(
    components: Mapping[tuple[str, str], Sequence[TrajectoryObs]],
    library_ids: Sequence[str],
) -> tuple[dict[str, float], bool, dict[str, JsonValue]]:
    shares: list[dict[str, float]] = []
    ess_fractions: list[float] = []
    ess_values: list[float] = []
    nonfinite = 0
    no_calls = 0
    for _key, group in sorted(components.items()):
        rows: list[tuple[float, dict[str, int]]] = []
        for trajectory in group:
            residuals = [obs.edge_residual for obs in trajectory.edges]
            if not all(math.isfinite(r) for r in residuals):
                nonfinite += 1
                continue
            counts: dict[str, int] = {}
            for obs in trajectory.edges:
                if _is_skill_edge(obs.event_function, obs.skill_id):
                    assert obs.skill_id is not None
                    counts[obs.skill_id] = counts.get(obs.skill_id, 0) + 1
            rows.append((math.fsum(residuals), counts))
        if not rows:
            continue
        low = min(delta for delta, _counts in rows)
        weights = [math.exp(-(delta - low)) for delta, _counts in rows]
        ess = math.fsum(weights) ** 2 / math.fsum(w * w for w in weights)
        ess_values.append(ess)
        ess_fractions.append(ess / len(weights))
        calling = [(delta, c) for delta, c in rows if sum(c.values()) > 0]
        if not calling:
            no_calls += 1
            continue
        low_calling = min(delta for delta, _c in calling)
        w_calling = [math.exp(-(delta - low_calling)) for delta, _c in calling]
        denominator = math.fsum(
            w * sum(c.values()) for w, (_d, c) in zip(w_calling, calling, strict=True)
        )
        skills = sorted({u for _d, c in calling for u in c})
        shares.append(
            {
                u: math.fsum(w * c.get(u, 0) for w, (_d, c) in zip(w_calling, calling, strict=True))
                / denominator
                for u in skills
            }
        )
    everyone = sorted(set(library_ids) | {u for share in shares for u in share})
    defined = bool(shares)
    psi0 = {
        u: (math.fsum(share.get(u, 0.0) for share in shares) / len(shares) if defined else 0.0)
        for u in everyone
    }
    diagnostics: dict[str, JsonValue] = {
        "psi0_rule": PSI0_ROLLOUT_RULE,
        "psi0_defined": defined,
        "psi0_queries": len(shares),
        "psi0_queries_without_skill_calls": no_calls,
        "psi0_nonfinite_residual_trajectories": nonfinite,
        "psi0_weight_ess_fraction_mean": (
            math.fsum(ess_fractions) / len(ess_fractions) if ess_fractions else None
        ),
        "psi0_weight_ess_min": min(ess_values) if ess_values else None,
    }
    return psi0, defined, diagnostics
