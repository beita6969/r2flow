from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from skillev.contracts import JsonValue

from .types import PhaseEvidence, PhaseState

Context = tuple[str, ...]
Cell = tuple[str, Context]

MU_RULE: Final = "eb-skill-mean-beta11@1"
CELL_RULE: Final = "observed-cells-only@1"
CARRY_RULE: Final = "carry-to-children-via-parent-id@1"
GLOBAL_LCB_RULE: Final = "skill-pooled-beta11-quantile@1"
NEFF_RULE: Final = "kish-phase-records@1"
QUANTILE_TOLERANCE: Final = 1e-10

_CF_EPS: Final = 3e-16
_CF_FPMIN: Final = 1e-300
_CF_MAXIT: Final = 100_000


def _betacf(a: float, b: float, x: float) -> float:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _CF_FPMIN:
        d = _CF_FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _CF_MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _CF_FPMIN:
            d = _CF_FPMIN
        c = 1.0 + aa / c
        if abs(c) < _CF_FPMIN:
            c = _CF_FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _CF_FPMIN:
            d = _CF_FPMIN
        c = 1.0 + aa / c
        if abs(c) < _CF_FPMIN:
            c = _CF_FPMIN
        d = 1.0 / d
        step = d * c
        h *= step
        if abs(step - 1.0) < _CF_EPS:
            return h
    raise ArithmeticError(f"betacf did not converge for a={a}, b={b}, x={x}")


def _check_shape(a: float, b: float) -> None:
    if not (math.isfinite(a) and math.isfinite(b) and a > 0.0 and b > 0.0):
        raise ValueError(f"Beta shape parameters must be finite and positive, got ({a}, {b})")


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    _check_shape(a, b)
    if math.isnan(x):
        raise ValueError("x must not be NaN")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        value = front * _betacf(a, b, x) / a
    else:
        value = 1.0 - front * _betacf(b, a, 1.0 - x) / b
    return min(1.0, max(0.0, value))


def beta_cdf(x: float, a: float, b: float) -> float:
    return regularized_incomplete_beta(x, a, b)


def beta_quantile(q: float, a: float, b: float, *, tol: float = QUANTILE_TOLERANCE) -> float:
    _check_shape(a, b)
    if math.isnan(q):
        raise ValueError("q must not be NaN")
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if regularized_incomplete_beta(mid, a, b) >= q:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def beta_mean(a: float, b: float) -> float:
    _check_shape(a, b)
    return a / (a + b)


def beta_variance(a: float, b: float) -> float:
    _check_shape(a, b)
    total = a + b
    return a * b / (total * total * (total + 1.0))


def kish_neff(weights: Iterable[float]) -> float:
    total = 0.0
    squares = 0.0
    for weight in weights:
        total += weight
        squares += weight * weight
    return 0.0 if squares <= 0.0 else total * total / squares


def context_key(z: Context) -> str:
    return "|".join(z)


@dataclass(frozen=True, slots=True)
class Posterior:
    alpha: dict[Cell, float]
    beta: dict[Cell, float]
    lcb: dict[Cell, float]
    ucb: dict[Cell, float]
    mean: dict[Cell, float]
    variance: dict[Cell, float]
    neff_cell: dict[Cell, float]
    evidence_mass: dict[Cell, tuple[float, float]]
    lcb_glob: dict[str, float]
    pooled_mean: dict[str, float]
    neff: dict[str, float]
    mu: dict[str, float]
    n_records: dict[str, int]
    delta: float
    kappa_u: float
    rules: dict[str, str]
    diagnostics: dict[str, JsonValue]

    def contexts(self, skill_id: str) -> tuple[Context, ...]:
        return tuple(sorted(z for (u, z) in self.alpha if u == skill_id))

    def skills(self) -> tuple[str, ...]:
        return tuple(sorted({u for (u, _z) in self.alpha}))


def _unit_interval(value: float, name: str) -> float:
    if not (math.isfinite(value) and 0.0 <= value <= 1.0):
        raise ValueError(f"{name} must lie in [0, 1], got {value!r}")
    return value


def verifier_posterior(
    evidence: PhaseEvidence,
    state: PhaseState,
    *,
    kappa_u: float,
    delta: float,
    mu_rule: str = MU_RULE,
) -> Posterior:
    if not (math.isfinite(kappa_u) and kappa_u > 0.0):
        raise ValueError("kappa_u must be finite and positive")
    if not (math.isfinite(delta) and 0.0 < delta < 0.5):
        raise ValueError("delta must lie in (0, 0.5)")
    if mu_rule != MU_RULE:
        raise ValueError(f"unsupported mu rule {mu_rule!r}; only {MU_RULE!r} is implemented")

    success: dict[Cell, float] = {}
    failure: dict[Cell, float] = {}
    cell_weights: dict[Cell, list[float]] = {}
    skill_weights: dict[str, list[float]] = {}
    skill_success: dict[str, float] = {}
    skill_mass: dict[str, float] = {}
    excluded = 0
    for record in evidence.verifier:
        if not record.gate_eligible:
            excluded += 1
            continue
        y = _unit_interval(record.y, "verifier outcome y")
        c = _unit_interval(record.confidence, "verifier confidence")
        cell = (record.skill_id, tuple(record.z))
        success[cell] = success.get(cell, 0.0) + c * y
        failure[cell] = failure.get(cell, 0.0) + c * (1.0 - y)
        cell_weights.setdefault(cell, []).append(c)
        skill_weights.setdefault(record.skill_id, []).append(c)
        skill_success[record.skill_id] = skill_success.get(record.skill_id, 0.0) + c * y
        skill_mass[record.skill_id] = skill_mass.get(record.skill_id, 0.0) + c

    library_ids = set(evidence.library.skill_ids)
    children: dict[str, list[str]] = {}
    for spec in evidence.library.skills:
        if spec.parent_id is not None:
            children.setdefault(spec.parent_id, []).append(spec.skill_id)
    carried_success: dict[Cell, float] = {}
    carried_failure: dict[Cell, float] = {}
    orphans: list[JsonValue] = []
    for (skill_id, z), (s_mass, f_mass) in sorted(state.carried_counts.items()):
        if not (math.isfinite(s_mass) and math.isfinite(f_mass) and s_mass >= 0 and f_mass >= 0):
            raise ValueError(f"carried counts of {(skill_id, z)!r} must be finite and >= 0")
        if skill_id in library_ids:
            targets = [skill_id]
        else:
            targets = sorted(children.get(skill_id, ()))
        if not targets:
            orphans.append([skill_id, context_key(tuple(z))])
            continue
        for target in targets:
            cell = (target, tuple(z))
            carried_success[cell] = carried_success.get(cell, 0.0) + s_mass
            carried_failure[cell] = carried_failure.get(cell, 0.0) + f_mass

    mu: dict[str, float] = {}
    for skill_id in sorted(library_ids | set(skill_mass) | {u for (u, _z) in carried_success}):
        mu[skill_id] = (1.0 + skill_success.get(skill_id, 0.0)) / (
            2.0 + skill_mass.get(skill_id, 0.0)
        )

    alpha: dict[Cell, float] = {}
    beta: dict[Cell, float] = {}
    lcb: dict[Cell, float] = {}
    ucb: dict[Cell, float] = {}
    mean: dict[Cell, float] = {}
    variance: dict[Cell, float] = {}
    neff_cell: dict[Cell, float] = {}
    evidence_mass: dict[Cell, tuple[float, float]] = {}
    pooled_success: dict[str, float] = {}
    pooled_failure: dict[str, float] = {}
    for cell in sorted(set(success) | set(carried_success)):
        s_total = success.get(cell, 0.0) + carried_success.get(cell, 0.0)
        f_total = failure.get(cell, 0.0) + carried_failure.get(cell, 0.0)
        if s_total + f_total <= 0.0:
            continue
        skill_id = cell[0]
        a = kappa_u * mu[skill_id] + s_total
        b = kappa_u * (1.0 - mu[skill_id]) + f_total
        alpha[cell] = a
        beta[cell] = b
        lcb[cell] = beta_quantile(delta, a, b)
        ucb[cell] = beta_quantile(1.0 - delta, a, b)
        mean[cell] = beta_mean(a, b)
        variance[cell] = beta_variance(a, b)
        neff_cell[cell] = kish_neff(cell_weights.get(cell, ()))
        evidence_mass[cell] = (s_total, f_total)
        pooled_success[skill_id] = pooled_success.get(skill_id, 0.0) + s_total
        pooled_failure[skill_id] = pooled_failure.get(skill_id, 0.0) + f_total

    lcb_glob: dict[str, float] = {}
    pooled_mean: dict[str, float] = {}
    for skill_id in sorted(pooled_success):
        a = 1.0 + pooled_success[skill_id]
        b = 1.0 + pooled_failure[skill_id]
        lcb_glob[skill_id] = beta_quantile(delta, a, b)
        pooled_mean[skill_id] = beta_mean(a, b)

    neff = {skill_id: kish_neff(skill_weights.get(skill_id, ())) for skill_id in sorted(mu)}
    n_records = {skill_id: len(skill_weights.get(skill_id, ())) for skill_id in sorted(mu)}
    diagnostics: dict[str, JsonValue] = {
        "gate_eligible_records": sum(n_records.values()),
        "excluded_non_eligible_records": excluded,
        "cells": len(alpha),
        "orphan_carried_cells": orphans,
        "quantile_tolerance": QUANTILE_TOLERANCE,
    }
    return Posterior(
        alpha=alpha,
        beta=beta,
        lcb=lcb,
        ucb=ucb,
        mean=mean,
        variance=variance,
        neff_cell=neff_cell,
        evidence_mass=evidence_mass,
        lcb_glob=lcb_glob,
        pooled_mean=pooled_mean,
        neff=neff,
        mu=mu,
        n_records=n_records,
        delta=delta,
        kappa_u=kappa_u,
        rules={
            "mu": mu_rule,
            "cells": CELL_RULE,
            "carry": CARRY_RULE,
            "lcb_glob": GLOBAL_LCB_RULE,
            "neff": NEFF_RULE,
        },
        diagnostics=diagnostics,
    )
