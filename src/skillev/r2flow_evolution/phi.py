from __future__ import annotations

import dataclasses
import difflib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from skillev.contracts import JsonValue

from .posterior import (
    Context,
    Posterior,
    beta_cdf,
    beta_quantile,
    context_key,
    kish_neff,
)
from .readouts import Readouts
from .types import (
    STRUCTURAL_EDITS,
    CandidateEdit,
    EditKind,
    PhaseEvidence,
    PhaseState,
    SkillSpec,
    TrajectoryObs,
    VerifierObs,
)

SKILL_EVENT_CLASS: Final = "skill"

CLASS_ORDER: Final[dict[EditKind, int]] = {
    EditKind.SPLIT: 0,
    EditKind.REFINE: 1,
    EditKind.PRUNE: 2,
    EditKind.COMPRESS: 3,
    EditKind.GENERATE: 4,
    EditKind.RETAIN: 5,
    EditKind.DEFER: 6,
}
SPLIT_GRID: Final = 512
SPLIT_TAIL: Final = 1e-12
FAILURE_REWARD: Final = 0.5

RULE_DEFER: Final = "defer-insufficient-support@1"
RULE_SPLIT: Final = "split-extreme-pair-lower-riemann-stieltjes@1"
RULE_REFINE: Final = "refine-global-sound-local-weak@1"
RULE_EMPTY_REFINE: Final = "empty-slot-first-procedure@1"
RULE_RETAIN: Final = "retain-lcb-glob-high@1"
RULE_PRUNE: Final = "prune-requires-a-tilde-le-0@1"
RULE_EMPTY_GUARD: Final = "empty-slot-prune-guard@1"
RULE_COOLDOWN: Final = "cooldown-inherits-via-parent-id@1"
RULE_DEFAULT: Final = "default-defer@1"
RULE_GENERATE: Final = "context-class-failure-mass@1"
RULE_VERIFIABLE: Final = "verifiable-cluster@1"
RULE_LAST_COVERAGE: Final = "family-last-coverage@1"
RULE_SPLIT_SUPPORT: Final = "split-support-every-context-nmin@1"
RULE_UNIQUE: Final = "sole-server-of-a-context-class@1"
RULE_CLASS_LAST_COVERAGE: Final = "context-class-last-coverage@1"
RULE_COMPRESS: Final = "pairwise-redundancy@2"
RULE_GENERATE_SUPPORT: Final = "generate-support=cluster-neff@1"
RULE_GENERATE_FAILURE_MASS: Final = "generate-failure-mass=verifier@1"
RULE_GENERATE_ADEQUACY: Final = "generate-adequacy=context-class@1"
RULE_EMPTY_SLOT_BOOTSTRAP: Final = "empty-slot-bootstrap@1"
RULE_EMPTY_SLOT_BOOTSTRAP_ALL: Final = "empty-slot-bootstrap@2"
BOOTSTRAP_CLUSTER: Final = "family-trajectories@1"
BOOTSTRAP_SUPPORT: Final = "trajectory-neff-generate-min-support@1"
BOOTSTRAP_FAILURE_MASS: Final = "one-minus-native-terminal-success@1"
NATIVE_TERMINAL_VERIFIER: Final = "native-terminal-outcome@1"
NATIVE_TERMINAL_LABELS: Final = (
    "binary success of the native scorer: exact match (HotpotQA, TriviaQA), exact integer "
    "(AIME), base + plus unit tests (MBPP+), environment success (ALFWorld)"
)
JUDGE_LABELLED_DOMAINS: Final = frozenset({"healthbench"})
BOOTSTRAP_DOMAIN_FAMILIES: Final[Mapping[str, str]] = {
    "hotpotqa": "multi-hop-qa",
    "triviaqa": "factual-qa",
    "aime-2026": "mathematical-reasoning",
    "healthbench": "health-dialogue",
    "alfworld": "interactive-decision",
    "mbpp-plus": "code-generation",
}
RULE_VERIFIER_EVIDENCE_FLOOR: Final = "verifier-evidence-floor@1"
FAMILY_VERIFIER_EVIDENCE: Final = "family-event-verifier-records@1"
COMPRESS_MIN_OUTPUT_SIMILARITY: Final = 0.8


@dataclass(frozen=True, slots=True)
class PhiConfig:
    tau_c: float = 1.0
    delta: float = 0.05
    theta_low: float = 0.3
    theta_mid: float = 0.5
    theta_high: float = 0.7
    n_min: int = 8
    theta_h: float = 0.2
    kappa_u: float = 8.0
    cooldown_phases: int = 2
    gamma_carry: float = 0.5
    generate_min_support: int = 8
    generate_min_failure_rate: float = 0.5
    compress_min_context_overlap: float = 0.8
    compress_max_reliability_gap: float = 0.1
    verifier_evidence_floor: int = 2

    def __post_init__(self) -> None:
        floor = self.verifier_evidence_floor
        if type(floor) is not int or not 1 <= floor <= self.n_min:
            raise ValueError(f"{RULE_VERIFIER_EVIDENCE_FLOOR} needs an integer floor in [1, n_min]")
        if not 0.0 <= self.theta_low < self.theta_mid < self.theta_high <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= theta_low < theta_mid < theta_high <= 1")
        if not (math.isfinite(self.tau_c) and self.tau_c > 0.0):
            raise ValueError("tau_c must be finite and positive")
        if not 0.0 < self.delta < 0.5:
            raise ValueError("delta must lie in (0, 0.5)")
        if not (math.isfinite(self.kappa_u) and self.kappa_u > 0.0):
            raise ValueError("kappa_u must be finite and positive")
        if self.n_min < 1:
            raise ValueError("n_min must be at least 1")
        if not 0.0 <= self.theta_h < 1.0:
            raise ValueError("theta_h must lie in [0, 1)")
        if self.cooldown_phases < 0:
            raise ValueError("cooldown_phases must be >= 0")
        if not 0.0 <= self.gamma_carry <= 1.0:
            raise ValueError("gamma_carry must lie in [0, 1]")
        if self.generate_min_support < 1:
            raise ValueError("generate_min_support must be at least 1")
        if not 0.0 <= self.generate_min_failure_rate <= 1.0:
            raise ValueError("generate_min_failure_rate must lie in [0, 1]")
        if not 0.0 <= self.compress_min_context_overlap <= 1.0:
            raise ValueError("compress_min_context_overlap must lie in [0, 1]")
        if not self.compress_max_reliability_gap >= 0.0:
            raise ValueError("compress_max_reliability_gap must be >= 0")


def prob_difference_exceeds(
    alpha_hi: float,
    beta_hi: float,
    alpha_lo: float,
    beta_lo: float,
    theta: float,
    *,
    grid: int = SPLIT_GRID,
) -> tuple[float, float]:
    def tail(q: float) -> float:
        x = q + theta
        if x >= 1.0:
            return 0.0
        if x <= 0.0:
            return 1.0
        return 1.0 - beta_cdf(x, alpha_hi, beta_hi)

    lo = beta_quantile(SPLIT_TAIL, alpha_lo, beta_lo)
    hi = min(beta_quantile(1.0 - SPLIT_TAIL, alpha_lo, beta_lo), 1.0 - theta)
    points = [0.0]
    if hi > lo:
        points.extend(lo + (hi - lo) * i / grid for i in range(grid + 1))
    else:
        points.append(lo)
    points.append(1.0)
    points = sorted(set(points))
    cdf = [beta_cdf(t, alpha_lo, beta_lo) for t in points]
    g = [tail(t) for t in points]
    lower = 0.0
    upper = 0.0
    for i in range(len(points) - 1):
        mass = cdf[i + 1] - cdf[i]
        lower += mass * g[i + 1]
        upper += mass * g[i]
    return max(0.0, min(1.0, lower)), max(0.0, min(1.0, upper))


@dataclass(frozen=True, slots=True)
class _SkillView:
    spec: SkillSpec
    neff: float
    lcb_glob: float | None
    pooled_mean: float | None
    cells: tuple[Context, ...]
    psi0: float
    atilde: float | None
    n_call: int
    cooldown: bool
    unique: bool
    empty_guard: bool
    unique_contexts: tuple[str, ...] = ()

    @property
    def skill_id(self) -> str:
        return self.spec.skill_id


def _opt(value: float | None) -> JsonValue:
    return None if value is None or not math.isfinite(value) else value


def family_verifier_evidence(
    evidence: PhaseEvidence, family: str, floor: int
) -> dict[str, JsonValue]:
    if evidence.event_verifier is None:
        raise ValueError(
            f"{RULE_VERIFIER_EVIDENCE_FLOOR} needs phase evidence with event-verifier-records@1"
        )
    members = {
        t.trajectory_id
        for t in evidence.trajectories
        if BOOTSTRAP_DOMAIN_FAMILIES.get(t.domain) == family
    }
    rows: list[tuple[float, float]] = []
    by_class: dict[str, int] = {}
    by_verifier: dict[str, int] = {}
    for record in evidence.verifier:
        if record.trajectory_id in members and record.gate_eligible and record.confidence > 0.0:
            rows.append((record.confidence, record.y))
            by_class[SKILL_EVENT_CLASS] = by_class.get(SKILL_EVENT_CLASS, 0) + 1
    for event in evidence.event_verifier:
        if event.trajectory_id in members and event.eligible and event.confidence > 0.0:
            rows.append((event.confidence, event.y))
            by_class[event.event_class] = by_class.get(event.event_class, 0) + 1
            for verifier in event.verifiers:
                by_verifier[verifier] = by_verifier.get(verifier, 0) + 1
    neff = kish_neff(c for c, _ in rows)
    return {
        "rule": FAMILY_VERIFIER_EVIDENCE,
        "floor_rule": RULE_VERIFIER_EVIDENCE_FLOOR,
        "records": len(rows),
        "neff": neff,
        "floor": floor,
        "success_mass": math.fsum(c * y for c, y in rows),
        "failure_mass": math.fsum(c * (1.0 - y) for c, y in rows),
        "records_per_event_class": dict(sorted(by_class.items())),
        "records_per_verifier": dict(sorted(by_verifier.items())),
        "sufficient": neff >= floor,
    }


def _rank(kind: EditKind, psi0: float, atilde: float | None) -> tuple[float, ...]:
    return (
        float(CLASS_ORDER[kind]),
        -psi0,
        0.0 if atilde is not None else 1.0,
        atilde if atilde is not None else 0.0,
    )


def _skill_evidence(view: _SkillView, posterior: Posterior, cfg: PhiConfig) -> dict[str, JsonValue]:
    sid = view.skill_id
    ucb: dict[str, JsonValue] = {context_key(z): posterior.ucb[(sid, z)] for z in view.cells}
    lcb: dict[str, JsonValue] = {context_key(z): posterior.lcb[(sid, z)] for z in view.cells}
    neff_cell: dict[str, JsonValue] = {
        context_key(z): posterior.neff_cell[(sid, z)] for z in view.cells
    }
    return {
        "skill_id": sid,
        "neff": view.neff,
        "lcb_glob": _opt(view.lcb_glob),
        "pooled_mean": _opt(view.pooled_mean),
        "ucb": ucb,
        "lcb": lcb,
        "neff_cell": neff_cell,
        "psi0": view.psi0,
        "atilde": _opt(view.atilde),
        "n_call": view.n_call,
        "verifier_eligible": view.neff >= cfg.n_min,
        "in_cooldown": view.cooldown,
        "is_empty_slot": view.spec.is_empty_slot,
        "unique_coverage": view.unique,
        "unique_contexts": [*view.unique_contexts],
    }


def _edit(
    kind: EditKind,
    view: _SkillView,
    posterior: Posterior,
    cfg: PhiConfig,
    rule: str,
    *,
    context: Context | None = None,
    checked: Mapping[str, JsonValue] | None = None,
    reason: str | None = None,
) -> CandidateEdit:
    evidence = _skill_evidence(view, posterior, cfg)
    evidence["rule"] = rule
    if checked:
        evidence.update(checked)
    if reason is not None:
        evidence["reason"] = reason
    return CandidateEdit(
        kind=kind,
        skill_ids=(view.skill_id,),
        context=context,
        rank=_rank(kind, view.psi0, view.atilde),
        evidence=evidence,
    )


def _split_test(
    view: _SkillView, posterior: Posterior, cfg: PhiConfig
) -> tuple[bool, dict[str, JsonValue], Context | None]:
    sid = view.skill_id
    cells = view.cells
    per_cell = float(cfg.n_min)
    supported = len(cells) >= 2 and all(posterior.neff_cell[(sid, z)] >= per_cell for z in cells)
    values: dict[str, JsonValue] = {
        "split_support": supported,
        "split_support_rule": RULE_SPLIT_SUPPORT,
    }
    if not supported:
        return False, values, None
    best = max(cells, key=lambda z: (posterior.mean[(sid, z)], z))
    worst = min(cells, key=lambda z: (posterior.mean[(sid, z)], z))
    lower, upper = prob_difference_exceeds(
        posterior.alpha[(sid, best)],
        posterior.beta[(sid, best)],
        posterior.alpha[(sid, worst)],
        posterior.beta[(sid, worst)],
        cfg.theta_h,
    )
    values.update(
        {
            "split_best": context_key(best),
            "split_worst": context_key(worst),
            "split_prob_lower": lower,
            "split_prob_upper": upper,
            "split_lcb_best_minus_ucb_worst": posterior.lcb[(sid, best)]
            - posterior.ucb[(sid, worst)],
        }
    )
    return lower > 1.0 - cfg.delta, values, worst


def _single_skill_decision(view: _SkillView, posterior: Posterior, cfg: PhiConfig) -> CandidateEdit:
    sid = view.skill_id
    if view.neff < cfg.n_min:
        return _edit(EditKind.DEFER, view, posterior, cfg, RULE_DEFER)
    ucbs = {z: posterior.ucb[(sid, z)] for z in view.cells}
    weakest = min(view.cells, key=lambda z: (ucbs[z], z)) if view.cells else None
    max_ucb = max(ucbs.values()) if ucbs else None
    checked: dict[str, JsonValue] = {
        "max_ucb": max_ucb,
        "weak_contexts": [context_key(z) for z in view.cells if ucbs[z] < cfg.theta_low],
    }
    if not view.cooldown:
        fired, split_values, worst = _split_test(view, posterior, cfg)
        checked.update(split_values)
        if fired:
            return _edit(
                EditKind.SPLIT, view, posterior, cfg, RULE_SPLIT, context=worst, checked=checked
            )
        refine_rules: list[str] = []
        if (
            view.lcb_glob is not None
            and view.lcb_glob >= cfg.theta_mid
            and weakest is not None
            and ucbs[weakest] < cfg.theta_low
        ):
            refine_rules.append(RULE_REFINE)
        if view.spec.is_empty_slot and max_ucb is not None and max_ucb < cfg.theta_mid:
            refine_rules.append(RULE_EMPTY_REFINE)
        if refine_rules:
            checked["refine_rules_fired"] = [*refine_rules]
            return _edit(
                EditKind.REFINE,
                view,
                posterior,
                cfg,
                refine_rules[0],
                context=weakest,
                checked=checked,
            )
    if view.lcb_glob is not None and view.lcb_glob >= cfg.theta_high:
        return _edit(EditKind.RETAIN, view, posterior, cfg, RULE_RETAIN, checked=checked)
    if view.cooldown:
        return _edit(
            EditKind.DEFER,
            view,
            posterior,
            cfg,
            RULE_DEFAULT,
            reason=RULE_COOLDOWN,
            checked=checked,
        )
    reason = "no-rule-fired"
    if max_ucb is not None and max_ucb < cfg.theta_low:
        veto: str | None = None
        if view.atilde is None:
            veto = "atilde-undefined"
        elif view.atilde > 0.0:
            veto = "atilde-positive-veto"
        if view.spec.is_empty_slot and view.empty_guard:
            reason = RULE_EMPTY_GUARD
        elif veto is not None:
            reason = veto
        elif view.unique:
            reason = RULE_UNIQUE
        else:
            return _edit(EditKind.PRUNE, view, posterior, cfg, RULE_PRUNE, checked=checked)
    return _edit(EditKind.DEFER, view, posterior, cfg, RULE_DEFAULT, reason=reason, checked=checked)


def procedure_similarity(first: SkillSpec, second: SkillSpec) -> float:
    a = " ".join(first.body.lower().split())
    b = " ".join(second.body.lower().split())
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def _credibly_different(a: str, b: str, z: Context, posterior: Posterior, cfg: PhiConfig) -> bool:
    gap = cfg.compress_max_reliability_gap
    ab, _ = prob_difference_exceeds(
        posterior.alpha[(a, z)],
        posterior.beta[(a, z)],
        posterior.alpha[(b, z)],
        posterior.beta[(b, z)],
        gap,
    )
    ba, _ = prob_difference_exceeds(
        posterior.alpha[(b, z)],
        posterior.beta[(b, z)],
        posterior.alpha[(a, z)],
        posterior.beta[(a, z)],
        gap,
    )
    return ab > 1.0 - cfg.delta or ba > 1.0 - cfg.delta


def _compress_candidates(
    views: Sequence[_SkillView],
    blocked: set[str],
    posterior: Posterior,
    cfg: PhiConfig,
) -> list[CandidateEdit]:
    eligible = [
        v
        for v in views
        if not v.spec.is_empty_slot and v.neff >= cfg.n_min and v.skill_id not in blocked
    ]
    pairs: list[tuple[float, float, float, str, str, _SkillView, _SkillView]] = []
    for i, first in enumerate(eligible):
        for second in eligible[i + 1 :]:
            if not set(first.spec.families) & set(second.spec.families):
                continue
            ctx_a, ctx_b = set(first.cells), set(second.cells)
            shared_cells = ctx_a & ctx_b
            if not shared_cells:
                continue
            jaccard = len(shared_cells) / len(ctx_a | ctx_b)
            if jaccard < cfg.compress_min_context_overlap:
                continue
            similarity = procedure_similarity(first.spec, second.spec)
            if similarity < COMPRESS_MIN_OUTPUT_SIMILARITY:
                continue
            gap = max(
                abs(posterior.mean[(first.skill_id, z)] - posterior.mean[(second.skill_id, z)])
                for z in shared_cells
            )
            if not any(
                _credibly_different(first.skill_id, second.skill_id, z, posterior, cfg)
                for z in sorted(shared_cells)
            ):
                pairs.append(
                    (gap, -jaccard, -similarity, first.skill_id, second.skill_id, first, second)
                )
    pairs.sort(key=lambda item: item[:5])
    used: set[str] = set()
    candidates: list[CandidateEdit] = []
    for gap, neg_jaccard, neg_similarity, id_a, id_b, first, second in pairs:
        if id_a in used or id_b in used:
            continue
        used.update((id_a, id_b))
        defined = [v.atilde for v in (first, second) if v.atilde is not None]
        atilde = sum(defined) / len(defined) if defined else None
        psi0 = first.psi0 + second.psi0
        ucb: dict[str, JsonValue] = {}
        for view in (first, second):
            for z in view.cells:
                ucb[f"{view.skill_id}|{context_key(z)}"] = posterior.ucb[(view.skill_id, z)]
        skills: dict[str, JsonValue] = {
            v.skill_id: _skill_evidence(v, posterior, cfg) for v in (first, second)
        }
        shared: list[JsonValue] = []
        shared.extend(sorted(set(first.spec.families) & set(second.spec.families)))
        lcbs = [v.lcb_glob for v in (first, second) if v.lcb_glob is not None]
        evidence: dict[str, JsonValue] = {
            "rule": RULE_COMPRESS,
            "neff": min(first.neff, second.neff),
            "lcb_glob": _opt(min(lcbs) if len(lcbs) == 2 else None),
            "ucb": ucb,
            "psi0": psi0,
            "atilde": _opt(atilde),
            "verifier_eligible": True,
            "jaccard": -neg_jaccard,
            "max_context_mean_gap": gap,
            "procedure_similarity": -neg_similarity,
            "shared_families": shared,
            "skills": skills,
        }
        candidates.append(
            CandidateEdit(
                kind=EditKind.COMPRESS,
                skill_ids=(id_a, id_b),
                context=None,
                rank=_rank(EditKind.COMPRESS, psi0, atilde),
                evidence=evidence,
            )
        )
    return candidates


def _generate_candidates(
    evidence: PhaseEvidence,
    active: Sequence[SkillSpec],
    posterior: Posterior,
    cfg: PhiConfig,
) -> list[CandidateEdit]:
    records: dict[str, list[VerifierObs]] = {}
    for record in evidence.verifier:
        records.setdefault(record.trajectory_id, []).append(record)
    clusters: dict[tuple[str, str], dict[str, TrajectoryObs]] = {}
    for trajectory in evidence.trajectories:
        classes = sorted({r.z[0] for r in records.get(trajectory.trajectory_id, ()) if r.z})
        for context_class in classes or [trajectory.family]:
            clusters.setdefault((trajectory.family, context_class), {}).setdefault(
                trajectory.trajectory_id, trajectory
            )
    total_failures = sum(
        1
        for trajectory in {t.trajectory_id: t for t in evidence.trajectories}.values()
        if trajectory.reward < FAILURE_REWARD
    )
    candidates: list[CandidateEdit] = []
    for (family, context_class), members in sorted(clusters.items()):
        support = len(members)
        failures = sum(1 for t in members.values() if t.reward < FAILURE_REWARD)
        failure_rate = failures / support
        eligible = [
            r
            for tid in sorted(members)
            for r in records.get(tid, ())
            if r.gate_eligible and r.confidence > 0.0 and r.z and r.z[0] == context_class
        ]
        eligible_confidences = [r.confidence for r in eligible]
        mass = math.fsum(eligible_confidences)
        verifier_failure_rate = (
            math.fsum(r.confidence * (1.0 - r.y) for r in eligible) / mass if mass > 0 else 0.0
        )
        ucb: dict[str, JsonValue] = {}
        adequate: list[JsonValue] = []
        for spec in active:
            for z in posterior.contexts(spec.skill_id):
                if z and z[0] == context_class:
                    value = posterior.ucb[(spec.skill_id, z)]
                    ucb[f"{spec.skill_id}|{context_key(z)}"] = value
                    if value >= cfg.theta_mid and spec.skill_id not in adequate:
                        adequate.append(spec.skill_id)
        neff = kish_neff(eligible_confidences)
        floor = cfg.verifier_evidence_floor
        verifiable = neff >= floor
        supported = neff >= cfg.n_min
        concentrated = verifier_failure_rate >= cfg.generate_min_failure_rate
        if not (supported and concentrated and not adequate and verifiable):
            continue
        share = failures / total_failures if total_failures else 0.0
        candidates.append(
            CandidateEdit(
                kind=EditKind.GENERATE,
                skill_ids=(),
                context=(family, context_class),
                rank=(float(CLASS_ORDER[EditKind.GENERATE]), -share, 0.0, 0.0),
                evidence={
                    "rule": RULE_GENERATE,
                    "verifiable_rule": RULE_VERIFIABLE,
                    "family": family,
                    "context_class": context_class,
                    "support": support,
                    "failures": failures,
                    "failure_rate": failure_rate,
                    "failure_share": share,
                    "gate_eligible_records": len(eligible_confidences),
                    "neff": neff,
                    "support_rule": RULE_GENERATE_SUPPORT,
                    "failure_mass_rule": RULE_GENERATE_FAILURE_MASS,
                    "verifier_failure_rate": verifier_failure_rate,
                    "adequacy_rule": RULE_GENERATE_ADEQUACY,
                    "floor_rule": RULE_VERIFIER_EVIDENCE_FLOOR,
                    "verifier_evidence_floor": floor,
                    "lcb_glob": None,
                    "ucb": ucb,
                    "adequate_skills": adequate,
                    "psi0": 0.0,
                    "atilde": None,
                    "verifier_eligible": verifiable,
                },
            )
        )
    return candidates


def propose_edits(
    evidence: PhaseEvidence,
    readouts: Readouts,
    posterior: Posterior,
    state: PhaseState,
    cfg: PhiConfig,
) -> list[CandidateEdit]:
    if not math.isclose(posterior.delta, cfg.delta, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("posterior.delta differs from PhiConfig.delta")
    if not math.isclose(posterior.kappa_u, cfg.kappa_u, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("posterior.kappa_u differs from PhiConfig.kappa_u")
    retired = set(state.retired)
    active = sorted(
        (spec for spec in evidence.library.skills if spec.skill_id not in retired),
        key=lambda spec: spec.skill_id,
    )
    members: dict[str, list[SkillSpec]] = {}
    for spec in active:
        for family in spec.families:
            members.setdefault(family, []).append(spec)

    def served(spec: SkillSpec) -> tuple[Context, ...]:
        cells = [z for z in posterior.contexts(spec.skill_id) if z and z[0] in spec.families]
        return tuple(sorted({(z[0],) for z in cells}))

    servers: dict[Context, set[str]] = {}
    for spec in active:
        for z in served(spec):
            servers.setdefault(z, set()).add(spec.skill_id)

    def sole_contexts(spec: SkillSpec) -> tuple[Context, ...]:
        return tuple(z for z in served(spec) if servers[z] == {spec.skill_id})

    def unique_coverage(spec: SkillSpec) -> bool:
        family_sole = any(len(members[f]) == 1 for f in spec.families)
        return family_sole or bool(sole_contexts(spec))

    def in_cooldown(spec: SkillSpec) -> bool:
        if state.cooldown.get(spec.skill_id, 0) > 0:
            return True
        return spec.parent_id is not None and state.cooldown.get(spec.parent_id, 0) > 0

    views: list[_SkillView] = []
    for spec in active:
        sid = spec.skill_id
        views.append(
            _SkillView(
                spec=spec,
                neff=posterior.neff.get(sid, 0.0),
                lcb_glob=posterior.lcb_glob.get(sid),
                pooled_mean=posterior.pooled_mean.get(sid),
                cells=posterior.contexts(sid),
                psi0=readouts.psi0.get(sid, 0.0),
                atilde=readouts.atilde_skill.get(sid),
                n_call=readouts.n_call.get(sid, 0),
                cooldown=in_cooldown(spec),
                unique=unique_coverage(spec),
                unique_contexts=tuple(context_key(z) for z in sole_contexts(spec)),
                empty_guard=any(
                    sum(1 for m in members[f] if not m.is_empty_slot) < 2 for f in spec.families
                ),
            )
        )

    bootstrap = _bootstrap_decisions(evidence, views, members, cfg)
    candidates: list[CandidateEdit] = []
    blocked: set[str] = set()
    for view in views:
        decision = _single_skill_decision(view, posterior, cfg)
        checked = bootstrap.get(view.skill_id)
        if checked is not None and decision.kind is EditKind.DEFER:
            decision = _bootstrap_edit(view, posterior, cfg, decision, checked)
        candidates.append(decision)
        if decision.kind in STRUCTURAL_EDITS:
            blocked.add(view.skill_id)
    candidates = _last_coverage_closure(candidates, views, members, servers)
    candidates.extend(_compress_candidates(views, blocked, posterior, cfg))
    candidates.extend(_generate_candidates(evidence, active, posterior, cfg))
    candidates.sort(key=lambda c: (c.rank, c.skill_ids, c.context or ()))
    return candidates


def _bootstrap_decisions(
    evidence: PhaseEvidence,
    views: Sequence[_SkillView],
    members: Mapping[str, Sequence[SkillSpec]],
    cfg: PhiConfig,
) -> dict[str, dict[str, JsonValue]]:
    floor = cfg.verifier_evidence_floor
    clusters: dict[str, dict[str, TrajectoryObs]] = {}
    for trajectory in evidence.trajectories:
        family = BOOTSTRAP_DOMAIN_FAMILIES.get(trajectory.domain)
        if family is not None and trajectory.domain not in JUDGE_LABELLED_DOMAINS:
            clusters.setdefault(family, {}).setdefault(trajectory.trajectory_id, trajectory)
    unique = {t.trajectory_id: t for t in evidence.trajectories}.values()
    native_failures = sum(
        1 for t in unique if not t.success and t.domain not in JUDGE_LABELLED_DOMAINS
    )
    by_id = {view.skill_id: view for view in views}
    out: dict[str, dict[str, JsonValue]] = {}
    for family, cluster in sorted(clusters.items()):
        family_skills = members.get(family, ())
        if not family_skills or any(not spec.is_empty_slot for spec in family_skills):
            continue
        slots = sorted(
            spec.skill_id
            for spec in family_skills
            if spec.families == (family,) and not by_id[spec.skill_id].cooldown
        )
        if not slots:
            continue
        trajectories = [cluster[tid] for tid in sorted(cluster)]
        support = len(trajectories)
        neff = kish_neff(1.0 for _ in trajectories)
        failed = [t.trajectory_id for t in trajectories if not t.success]
        failure_rate = len(failed) / support
        supported = neff >= cfg.generate_min_support
        domains: list[JsonValue] = []
        domains.extend(sorted({t.domain for t in trajectories}))
        succeeded = [t.trajectory_id for t in trajectories if t.success]
        trajectory_ids: list[JsonValue] = []
        trajectory_ids.extend(failed + succeeded)
        labels: dict[str, JsonValue] = {
            "rule": NATIVE_TERMINAL_VERIFIER,
            "labels": NATIVE_TERMINAL_LABELS,
            "records": support,
            "failure_mass": float(len(failed)),
            "success_mass": float(support - len(failed)),
        }
        checked: dict[str, JsonValue] = {
            "rule": RULE_EMPTY_SLOT_BOOTSTRAP,
            "family": family,
            "context_class": family,
            "cluster_rule": BOOTSTRAP_CLUSTER,
            "domains": domains,
            "support_rule": BOOTSTRAP_SUPPORT,
            "cluster_support": support,
            "cluster_neff": neff,
            "failure_mass_rule": BOOTSTRAP_FAILURE_MASS,
            "failures": len(failed),
            "failure_rate": failure_rate,
            "failure_share": len(failed) / native_failures if native_failures else 0.0,
            "supported": supported,
            "concentrated": True,
            "fired": supported,
            "verifier_evidence": labels,
            "trajectory_ids": trajectory_ids,
        }
        _apply_evidence_floor(checked, evidence, family, floor, labels)
        out[slots[0]] = checked
        out[slots[0]]["failure_gate"] = f"waived-by-{RULE_EMPTY_SLOT_BOOTSTRAP_ALL}"
    return out


def _apply_evidence_floor(
    checked: dict[str, JsonValue],
    evidence: PhaseEvidence,
    family: str,
    floor: int,
    native: dict[str, JsonValue],
) -> None:
    family_evidence = family_verifier_evidence(evidence, family, floor)
    verified = family_evidence["sufficient"] is True
    checked.update(
        verifier_evidence=family_evidence,
        native_outcomes=native,
        verified=verified,
        fired=checked["fired"] is True and verified,
    )


def _bootstrap_edit(
    view: _SkillView,
    posterior: Posterior,
    cfg: PhiConfig,
    deferred: CandidateEdit,
    checked: Mapping[str, JsonValue],
) -> CandidateEdit:
    if checked["fired"] is not True:
        values = {key: value for key, value in checked.items() if key != "trajectory_ids"}
        return dataclasses.replace(
            deferred, evidence={**deferred.evidence, "empty_slot_bootstrap": values}
        )
    family = str(checked["family"])
    share = checked["failure_share"]
    assert isinstance(share, float)
    evidence = _skill_evidence(view, posterior, cfg)
    evidence.update(checked)
    evidence.update(
        refine_rules_fired=[RULE_EMPTY_SLOT_BOOTSTRAP],
        verifier_eligible=checked["verified"] is True,
    )
    return CandidateEdit(
        kind=EditKind.REFINE,
        skill_ids=(view.skill_id,),
        context=(family,),
        rank=(float(CLASS_ORDER[EditKind.REFINE]), -share, 0.0, 0.0),
        evidence=evidence,
    )


def _last_coverage_closure(
    decisions: list[CandidateEdit],
    views: Sequence[_SkillView],
    members: Mapping[str, Sequence[SkillSpec]],
    servers: Mapping[Context, set[str]],
) -> list[CandidateEdit]:
    by_id = {view.skill_id: view for view in views}
    out = list(decisions)
    index = {d.skill_ids[0]: i for i, d in enumerate(out) if len(d.skill_ids) == 1}
    groups: list[tuple[str, str, set[str]]] = [
        (RULE_LAST_COVERAGE, family, {spec.skill_id for spec in members[family]})
        for family in sorted(members)
    ]
    groups.extend(
        (RULE_CLASS_LAST_COVERAGE, context_key(z), set(ids)) for z, ids in sorted(servers.items())
    )
    changed = True
    while changed:
        changed = False
        pruned = {d.skill_ids[0] for d in out if d.kind is EditKind.PRUNE and d.skill_ids}
        for rule, name, ids in groups:
            if not ids or not ids <= pruned:
                continue
            order = sorted(ids)
            keeper = max(
                order,
                key=lambda u: (
                    by_id[u].pooled_mean if by_id[u].pooled_mean is not None else -1.0,
                    -order.index(u),
                ),
            )
            view = by_id[keeper]
            old = out[index[keeper]]
            blocked = [*old.evidence.get("prune_blocked_by", ()), rule]
            evidence = {**old.evidence, "prune_blocked_by": blocked, "last_coverage_of": name}
            out[index[keeper]] = dataclasses.replace(
                old,
                kind=EditKind.DEFER,
                context=None,
                rank=_rank(EditKind.DEFER, view.psi0, view.atilde),
                evidence={**evidence, "rule": RULE_DEFAULT, "reason": rule},
            )
            changed = True
            break
    return out


def next_phase_state(
    state: PhaseState,
    posterior: Posterior,
    decisions: Sequence[CandidateEdit],
    accepted_skill_ids: set[str],
    cfg: PhiConfig,
) -> PhaseState:
    accepted = [
        d
        for d in decisions
        if d.kind in STRUCTURAL_EDITS and d.skill_ids and set(d.skill_ids) <= accepted_skill_ids
    ]
    pruned = {u for d in accepted if d.kind is EditKind.PRUNE for u in d.skill_ids}
    changed = {
        u
        for d in accepted
        if d.kind in (EditKind.SPLIT, EditKind.REFINE, EditKind.COMPRESS)
        for u in d.skill_ids
    }
    cooled = {
        u
        for d in accepted
        if d.kind in (EditKind.SPLIT, EditKind.REFINE, EditKind.PRUNE)
        for u in d.skill_ids
    }
    carried: dict[tuple[str, tuple[str, ...]], tuple[float, float]] = {}
    for cell, (s_mass, f_mass) in sorted(posterior.evidence_mass.items()):
        skill_id = cell[0]
        if skill_id in pruned:
            continue
        factor = cfg.gamma_carry if skill_id in changed else 1.0
        carried[cell] = (factor * s_mass, factor * f_mass)
    cooldown: dict[str, int] = {}
    for skill_id, remaining in state.cooldown.items():
        if remaining - 1 > 0:
            cooldown[skill_id] = remaining - 1
    if cfg.cooldown_phases > 0:
        for skill_id in cooled:
            cooldown[skill_id] = cfg.cooldown_phases
    newly_retired = sorted((pruned | changed) - set(state.retired))
    return PhaseState(
        phase=state.phase + 1,
        carried_counts=carried,
        cooldown=dict(sorted(cooldown.items())),
        retired=tuple(state.retired) + tuple(newly_retired),
    )
