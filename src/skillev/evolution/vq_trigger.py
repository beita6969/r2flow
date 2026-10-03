from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Final, Literal, Protocol

from skillev.contracts import JsonValue

VQ_POINT_FORMAT: Final = "r2flow-heldout-vq-point@1"
VQ_DECISION_FORMAT: Final = "r2flow-vq-decision@1"
VQ_POOLING: Final = "random-effects-dl-bias-corrected-logvar@1"
VQ_SLOPE: Final = "wls-delta-method-z-per-optimizer-step-fixed-panel@1"
VQ_ENTROPY_VACUOUS: Final = "vq-entropy-vacuous-when-undefined@1"
VQ_ENTROPY_UNDEFINED_RULES: Final = frozenset({VQ_ENTROPY_VACUOUS})
ENTROPY_AT_MOST_ONE_SKILL: Final = "at-most-one-visible-active-skill"
ENTROPY_H_UNDEFINED: Final = "h-norm-undefined"
ENTROPY_ONE_SKILL_PER_FAMILY: Final = "one-visible-skill-per-called-family"

VqStatus = Literal[
    "boundary",
    "no-plateau",
    "insufficient-points",
    "deferred-missing-point",
]


def digamma(x: float) -> float:
    if not math.isfinite(x) or x <= 0:
        raise ValueError("digamma requires a finite positive argument")
    shift = 0.0
    while x < 10.0:
        shift -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    series = inv2 * (
        1.0 / 12 - inv2 * (1.0 / 120 - inv2 * (1.0 / 252 - inv2 * (1.0 / 240 - inv2 * (1.0 / 132))))
    )
    return shift + math.log(x) - 0.5 * inv - series


def trigamma(x: float) -> float:
    if not math.isfinite(x) or x <= 0:
        raise ValueError("trigamma requires a finite positive argument")
    shift = 0.0
    while x < 10.0:
        shift += 1.0 / (x * x)
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    series = (
        inv
        + 0.5 * inv2
        + inv
        * inv2
        * (1.0 / 6 - inv2 * (1.0 / 30 - inv2 * (1.0 / 42 - inv2 * (1.0 / 30 - inv2 * 5.0 / 66))))
    )
    return shift + series


@dataclass(frozen=True, slots=True)
class QueryResiduals:
    query_key: str
    deltas: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.query_key:
            raise ValueError("query key must be non-empty")
        if not all(math.isfinite(value) for value in self.deltas):
            raise ValueError("held-out residuals must be finite")

    def to_value(self) -> dict[str, JsonValue]:
        return {"deltas": list(self.deltas), "query_key": self.query_key}


@dataclass(frozen=True, slots=True)
class PooledLogVariance:
    log_v: float
    se_log_v: float
    tau2: float
    queries: int


def pool_log_variance(queries: Sequence[QueryResiduals], v_min: float) -> PooledLogVariance:
    if not queries or any(len(q.deltas) < 2 for q in queries):
        raise ValueError("pooling needs at least one query and >= 2 rollouts per query")
    if not v_min > 0:
        raise ValueError("v_min must be positive")
    ys: list[float] = []
    vs: list[float] = []
    for query in queries:
        half_nu = (len(query.deltas) - 1) / 2
        s2 = max(statistics.variance(query.deltas), v_min)
        ys.append(math.log(s2) - (digamma(half_nu) - math.log(half_nu)))
        vs.append(trigamma(half_nu))
    weights = [1.0 / v for v in vs]
    total = math.fsum(weights)
    ybar = math.fsum(w * y for w, y in zip(weights, ys, strict=True)) / total
    q_stat = math.fsum(w * (y - ybar) ** 2 for w, y in zip(weights, ys, strict=True))
    denominator = total - math.fsum(w * w for w in weights) / total
    k = len(ys)
    tau2 = max(0.0, (q_stat - (k - 1)) / denominator) if denominator > 0 else 0.0
    star = [1.0 / (v + tau2) for v in vs]
    star_total = math.fsum(star)
    mu = math.fsum(w * y for w, y in zip(star, ys, strict=True)) / star_total
    return PooledLogVariance(mu, math.sqrt(1.0 / star_total), tau2, k)


def fixed_panel_log_variance(
    queries: Sequence[QueryResiduals], v_min: float
) -> tuple[float, float]:
    if not queries or any(len(q.deltas) < 2 for q in queries):
        raise ValueError("the fixed panel needs at least one query and >= 2 rollouts per query")
    if not v_min > 0:
        raise ValueError("v_min must be positive")
    ys: list[float] = []
    weights: list[float] = []
    for query in queries:
        half_nu = (len(query.deltas) - 1) / 2
        s2 = max(statistics.variance(query.deltas), v_min)
        ys.append(math.log(s2) - (digamma(half_nu) - math.log(half_nu)))
        weights.append(1.0 / trigamma(half_nu))
    total = math.fsum(weights)
    mean = math.fsum(w * y for w, y in zip(weights, ys, strict=True)) / total
    return mean, math.sqrt(1.0 / total)


def normalized_conditional_entropy(
    cell_calls: Mapping[tuple[str, str], Mapping[str, int]], active_skills: int
) -> float | None:
    if active_skills <= 1:
        return None
    cells = {
        key: {skill: n for skill, n in calls.items() if n > 0}
        for key, calls in cell_calls.items()
        if sum(n for n in calls.values() if n > 0) > 0
    }
    if any(n < 0 for calls in cell_calls.values() for n in calls.values()):
        raise ValueError("call counts must be non-negative")
    grand = sum(sum(calls.values()) for calls in cells.values())
    if grand == 0:
        return None
    entropy = math.fsum(
        (sum(calls.values()) / grand)
        * -math.fsum(
            (n / sum(calls.values())) * math.log(n / sum(calls.values())) for n in calls.values()
        )
        for _, calls in sorted(cells.items())
    )
    return entropy / math.log(active_skills)


@dataclass(frozen=True, slots=True)
class VqPoint:
    optimizer_step: int
    library_version: str
    status: Literal["complete", "gap"]
    queries: tuple[QueryResiduals, ...]
    log_v: float | None
    se_log_v: float | None
    tau2: float | None
    v: float | None
    h_norm: float | None
    calls_total: int
    active_skills: int
    format: str = VQ_POINT_FORMAT

    def __post_init__(self) -> None:
        if self.format != VQ_POINT_FORMAT:
            raise ValueError("unsupported V_q point format")
        if type(self.optimizer_step) is not int or self.optimizer_step < 0:
            raise ValueError("V_q point step must be a non-negative integer")
        complete = (self.log_v, self.se_log_v, self.tau2, self.v)
        if self.status == "complete" and any(value is None for value in complete):
            raise ValueError("a complete V_q point carries the pooled statistics")
        if self.status == "gap" and any(value is not None for value in complete):
            raise ValueError("a V_q gap carries no pooled statistics")
        if self.status not in ("complete", "gap"):
            raise ValueError("unsupported V_q point status")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "active_skills": self.active_skills,
            "calls_total": self.calls_total,
            "format": self.format,
            "h_norm": self.h_norm,
            "library_version": self.library_version,
            "log_v": self.log_v,
            "optimizer_step": self.optimizer_step,
            "queries": [q.to_value() for q in self.queries],
            "se_log_v": self.se_log_v,
            "status": self.status,
            "tau2": self.tau2,
            "v": self.v,
        }


def build_vq_point(
    *,
    optimizer_step: int,
    library_version: str,
    queries: Sequence[QueryResiduals],
    cell_calls: Mapping[tuple[str, str], Mapping[str, int]],
    active_skills: int,
    v_min: float,
) -> VqPoint:
    calls_total = sum(n for calls in cell_calls.values() for n in calls.values() if n > 0)
    h_norm = normalized_conditional_entropy(cell_calls, active_skills)
    ordered = tuple(sorted(queries, key=lambda q: q.query_key))
    if not ordered or any(len(q.deltas) < 2 for q in ordered):
        return VqPoint(
            optimizer_step,
            library_version,
            "gap",
            ordered,
            None,
            None,
            None,
            None,
            h_norm,
            calls_total,
            active_skills,
        )
    pooled = pool_log_variance(ordered, v_min)
    return VqPoint(
        optimizer_step,
        library_version,
        "complete",
        ordered,
        pooled.log_v,
        pooled.se_log_v,
        pooled.tau2,
        math.exp(pooled.log_v),
        h_norm,
        calls_total,
        active_skills,
    )


class VqThresholds(Protocol):
    @property
    def cadence_steps(self) -> int: ...
    @property
    def window_points(self) -> int: ...
    @property
    def alpha(self) -> float: ...
    @property
    def epsilon_b_fraction(self) -> float: ...
    @property
    def gamma_var(self) -> float: ...
    @property
    def v_min(self) -> float: ...
    @property
    def h0(self) -> float: ...


@dataclass(frozen=True, slots=True)
class VqDecision:
    status: VqStatus
    library_version: str | None
    window_steps: tuple[int, ...]
    slope: float | None = None
    half_width: float | None = None
    epsilon_b: float | None = None
    rho: float | None = None
    delta_h: float | None = None
    slope_ok: bool | None = None
    rho_ok: bool | None = None
    entropy_ok: bool | None = None
    entropy_vacuous: str | None = None
    format: str = VQ_DECISION_FORMAT

    def to_value(self) -> dict[str, JsonValue]:
        value: dict[str, JsonValue] = {
            "delta_h": self.delta_h,
            "entropy_ok": self.entropy_ok,
            "epsilon_b": self.epsilon_b,
            "format": self.format,
            "half_width": self.half_width,
            "library_version": self.library_version,
            "rho": self.rho,
            "rho_ok": self.rho_ok,
            "slope": self.slope,
            "slope_ok": self.slope_ok,
            "status": self.status,
            "window_steps": list(self.window_steps),
        }
        if self.entropy_vacuous is not None:
            value["entropy_vacuous"] = self.entropy_vacuous
        return value


def _vacuous_entropy(
    ends: tuple[VqPoint, VqPoint], entropy_degenerate_steps: frozenset[int]
) -> str | None:
    if any(p.active_skills <= 1 for p in ends):
        return ENTROPY_AT_MOST_ONE_SKILL
    if any(p.h_norm is None for p in ends):
        return ENTROPY_H_UNDEFINED
    if any(p.optimizer_step in entropy_degenerate_steps for p in ends):
        return ENTROPY_ONE_SKILL_PER_FAMILY
    return None


def evaluate_vq_trigger(
    series: Sequence[VqPoint],
    config: VqThresholds,
    *,
    entropy_degenerate_steps: frozenset[int] = frozenset(),
) -> VqDecision:
    entropy_rule = getattr(config, "entropy_undefined", None)
    if entropy_rule not in VQ_ENTROPY_UNDEFINED_RULES:
        raise ValueError(f"unsupported V_q entropy-undefined rule {entropy_rule!r}")
    if not series:
        return VqDecision("insufficient-points", None, ())
    library = max(series, key=lambda p: p.optimizer_step).library_version
    points = sorted(
        (p for p in series if p.library_version == library and p.status == "complete"),
        key=lambda p: p.optimizer_step,
    )
    width = config.window_points + 1
    if len(points) < width:
        return VqDecision("insufficient-points", library, tuple(p.optimizer_step for p in points))
    window = points[-width:]
    steps = tuple(p.optimizer_step for p in window)
    ordinals = [step // config.cadence_steps for step in steps]
    if any(step % config.cadence_steps for step in steps) or any(
        b - a != 1 for a, b in pairwise(ordinals)
    ):
        return VqDecision("deferred-missing-point", library, steps)
    vacuous = _vacuous_entropy((window[0], window[-1]), entropy_degenerate_steps)
    first_h, last_h = window[0].h_norm, window[-1].h_norm
    panels = [p.queries for p in window]
    keys = {tuple(sorted(q.query_key for q in queries)) for queries in panels}
    if len(keys) != 1:
        raise ValueError("the fixed-panel slope needs the same held-out queries at every point")
    stats = [fixed_panel_log_variance(queries, config.v_min) for queries in panels]
    values = [math.exp(mean) for mean, _ in stats]
    sds = [v * se for v, (_, se) in zip(values, stats, strict=True)]
    if any(sd <= 0 for sd in sds):
        raise ValueError("V_q standard errors must be positive")
    weights = [1.0 / (sd * sd) for sd in sds]
    total = math.fsum(weights)
    xs = [float(step) for step in steps]
    x_bar = math.fsum(w * x for w, x in zip(weights, xs, strict=True)) / total
    v_bar = math.fsum(w * v for w, v in zip(weights, values, strict=True)) / total
    sxx = math.fsum(w * (x - x_bar) ** 2 for w, x in zip(weights, xs, strict=True))
    slope = (
        math.fsum(
            w * (x - x_bar) * (v - v_bar) for w, x, v in zip(weights, xs, values, strict=True)
        )
        / sxx
    )
    half_width = statistics.NormalDist().inv_cdf(1 - config.alpha / 2) / math.sqrt(sxx)
    epsilon_b = config.epsilon_b_fraction * statistics.median(values)
    slope_ok = -epsilon_b <= slope - half_width and slope + half_width <= epsilon_b
    rho = (values[0] - values[-1]) / max(values[0], config.v_min)
    rho_ok = 0.0 <= rho < config.gamma_var
    delta_h = None if first_h is None or last_h is None else last_h - first_h
    if vacuous is not None:
        entropy_ok = True
    else:
        assert delta_h is not None
        entropy_ok = delta_h < -config.h0
    return VqDecision(
        "boundary" if slope_ok and rho_ok and entropy_ok else "no-plateau",
        library,
        steps,
        slope=slope,
        half_width=half_width,
        epsilon_b=epsilon_b,
        rho=rho,
        delta_h=delta_h,
        slope_ok=slope_ok,
        rho_ok=rho_ok,
        entropy_ok=entropy_ok,
        entropy_vacuous=vacuous,
    )


__all__ = [
    "ENTROPY_AT_MOST_ONE_SKILL",
    "ENTROPY_H_UNDEFINED",
    "ENTROPY_ONE_SKILL_PER_FAMILY",
    "VQ_DECISION_FORMAT",
    "VQ_ENTROPY_UNDEFINED_RULES",
    "VQ_ENTROPY_VACUOUS",
    "VQ_POINT_FORMAT",
    "VQ_POOLING",
    "VQ_SLOPE",
    "PooledLogVariance",
    "QueryResiduals",
    "VqDecision",
    "VqPoint",
    "VqStatus",
    "VqThresholds",
    "build_vq_point",
    "digamma",
    "evaluate_vq_trigger",
    "fixed_panel_log_variance",
    "normalized_conditional_entropy",
    "pool_log_variance",
    "trigamma",
]
