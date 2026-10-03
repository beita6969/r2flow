from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Final, Protocol, cast

from skillev.contracts import JsonValue

from .types import LibraryVersion, PairedOutcome, SkillSpec, TrajectoryObs

PAIRED_RULES: Final = (
    "paired-common-seed@1",
    "paired-by-query-rollout-index@1",
    "paired-query-mean@1",
    "paired-arm-memo=library-value@1",
)
RULE_CHANGED_FAMILIES_ROLLOUT: Final = "validation-rollout=changed-families-only@1"

RolloutHeldout = Callable[[LibraryVersion, int], Sequence[TrajectoryObs]]
RunPaired = Callable[[LibraryVersion, LibraryVersion], list[PairedOutcome]]
RolloutObserver = Callable[[dict[str, JsonValue]], None]


class DomainRolloutHeldout(Protocol):
    def __call__(
        self, library: LibraryVersion, seed: int, *, domains: frozenset[str]
    ) -> Sequence[TrajectoryObs]: ...


def visible_skills_by_family(library: LibraryVersion) -> dict[str, frozenset[SkillSpec]]:
    visible: dict[str, set[SkillSpec]] = {}
    for spec in library.skills:
        if spec.is_empty_slot:
            continue
        for family in spec.families:
            visible.setdefault(family, set()).add(spec)
    return {family: frozenset(specs) for family, specs in visible.items()}


def changed_families(current: LibraryVersion, candidate: LibraryVersion) -> frozenset[str]:
    before, after = visible_skills_by_family(current), visible_skills_by_family(candidate)
    return frozenset(
        family
        for family in before.keys() | after.keys()
        if before.get(family, frozenset()) != after.get(family, frozenset())
    )


def _sorted(values: Iterable[str]) -> list[JsonValue]:
    return [*sorted(set(values))]


def _by_query(arm: Sequence[TrajectoryObs], label: str) -> dict[str, list[TrajectoryObs]]:
    grouped: dict[str, list[TrajectoryObs]] = {}
    for trajectory in arm:
        grouped.setdefault(trajectory.query_id, []).append(trajectory)
    for query, rollouts in grouped.items():
        if len({t.domain for t in rollouts}) != 1:
            raise ValueError(f"arm {label}: query {query!r} spans several domains")
    return grouped


def _mean(values: Sequence[float]) -> float:
    return float(sum(values)) / len(values)


def pair_outcomes(
    arm_a: Sequence[TrajectoryObs], arm_b: Sequence[TrajectoryObs]
) -> list[PairedOutcome]:
    a, b = _by_query(arm_a, "a"), _by_query(arm_b, "b")
    if not a:
        raise ValueError("the held-out arms are empty")
    if set(a) != set(b):
        only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        raise ValueError(f"held-out query sets differ: only a {only_a[:5]}, only b {only_b[:5]}")
    outcomes = []
    for query in sorted(a):
        ra, rb = a[query], b[query]
        if len(ra) != len(rb):
            raise ValueError(f"query {query!r}: {len(ra)} vs {len(rb)} rollouts")
        if ra[0].domain != rb[0].domain:
            raise ValueError(f"query {query!r}: arms disagree on the domain")
        outcomes.append(
            PairedOutcome(
                query_id=query,
                domain=ra[0].domain,
                success_a=_mean([float(t.success) for t in ra]),
                success_b=_mean([float(t.success) for t in rb]),
                reward_eta_a=_mean([t.reward_eta for t in ra]),
                reward_eta_b=_mean([t.reward_eta for t in rb]),
                tokens_a=_mean([float(t.tokens) for t in ra]),
                tokens_b=_mean([float(t.tokens) for t in rb]),
                latency_a=_mean([t.latency_seconds for t in ra]),
                latency_b=_mean([t.latency_seconds for t in rb]),
            )
        )
    return outcomes


def make_run_paired(
    rollout_heldout: RolloutHeldout,
    *,
    seed: int,
    memoize: bool = True,
    observer: RolloutObserver | None = None,
) -> RunPaired:
    cache: dict[LibraryVersion, tuple[TrajectoryObs, ...]] = {}
    records: dict[LibraryVersion, dict[str, JsonValue]] = {}

    def arm(library: LibraryVersion) -> tuple[TrajectoryObs, ...]:
        if memoize and library in cache:
            return cache[library]
        rollouts = tuple(rollout_heldout(library, seed))
        if memoize:
            cache[library] = rollouts
        return rollouts

    def scoped_arm(
        current: LibraryVersion, current_arm: tuple[TrajectoryObs, ...], candidate: LibraryVersion
    ) -> tuple[tuple[TrajectoryObs, ...], dict[str, JsonValue]]:
        if memoize and candidate in records:
            return cache[candidate], records[candidate]
        from .phi import BOOTSTRAP_DOMAIN_FAMILIES

        changed = changed_families(current, candidate)
        domains = frozenset(
            trajectory.domain
            for trajectory in current_arm
            if BOOTSTRAP_DOMAIN_FAMILIES.get(trajectory.domain) in changed
        )
        reused = tuple(t for t in current_arm if t.domain not in domains)
        rolled: tuple[TrajectoryObs, ...] = ()
        if domains:
            rollout = cast(DomainRolloutHeldout, rollout_heldout)
            rolled = tuple(rollout(candidate, seed, domains=domains))
            expected = {t.query_id for t in current_arm if t.domain in domains}
            if {t.domain for t in rolled} - domains or {t.query_id for t in rolled} != expected:
                raise ValueError(
                    f"the candidate arm over {sorted(domains)} does not cover exactly the "
                    "current arm's queries of those domains"
                )
        merged = tuple(sorted(reused + rolled, key=lambda t: t.query_id))
        record: dict[str, JsonValue] = {
            "rule": RULE_CHANGED_FAMILIES_ROLLOUT,
            "changed_families": _sorted(changed),
            "rolled_out_domains": _sorted(domains),
            "rolled_out": _sorted(t.query_id for t in rolled),
            "reused": _sorted(t.query_id for t in reused),
            "rolled_out_trajectories": len(rolled),
            "reused_trajectories": len(reused),
        }
        if memoize:
            cache[candidate], records[candidate] = merged, record
        return merged, record

    def run_paired(current: LibraryVersion, candidate: LibraryVersion) -> list[PairedOutcome]:
        current_arm = arm(current)
        candidate_arm, record = scoped_arm(current, current_arm, candidate)
        if observer is not None:
            observer(dict(record))
        return pair_outcomes(current_arm, candidate_arm)

    return run_paired


__all__ = [
    "PAIRED_RULES",
    "RULE_CHANGED_FAMILIES_ROLLOUT",
    "DomainRolloutHeldout",
    "RolloutHeldout",
    "RolloutObserver",
    "RunPaired",
    "changed_families",
    "make_run_paired",
    "pair_outcomes",
    "visible_skills_by_family",
]
