from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from skillev.contracts import JsonValue

from .posterior import regularized_incomplete_beta
from .types import PairedOutcome

RULE: Final = "paired-one-sided-t-noninferiority@1"
ABSOLUTE_METRICS: Final = ("success_abs", "tempered_reward_abs")
LOG_RATIO_METRICS: Final = ("tokens_log_ratio", "latency_log_ratio")
RULE_LOG_COST: Final = "tost-cost=paired-log-ratio@1"
RULE_IMPROVEMENT_DIAGNOSTIC: Final = "gate-improvement-diagnostic=family-posterior@1"
RULES: Final = (RULE_IMPROVEMENT_DIAGNOSTIC, RULE_LOG_COST)
LOG_RATIO_FLOORS: Final[Mapping[str, float]] = {
    "tokens_log_ratio": 1.0,
    "latency_log_ratio": 1e-3,
}
IMPROVEMENT_METRIC: Final = "tempered_reward_gain"
POSTERIOR_MODEL: Final = "normal-mean-sample-variance-floor@1"
POSTERIOR_THRESHOLD: Final = 0.8
POSTERIOR_VARIANCE_FLOOR: Final = 1e-4
INCONCLUSIVE: Final = "inconclusive"
T_TOLERANCE: Final = 1e-10


@dataclass(frozen=True, slots=True)
class TostResult:
    passed: bool
    per_metric: dict[str, dict[str, JsonValue]]
    alpha: float
    improvement: dict[str, JsonValue]
    rule: str = RULE
    rules: tuple[str, ...] = RULES


def student_t_cdf(t: float, df: float) -> float:
    if not (math.isfinite(df) and df > 0.0):
        raise ValueError("df must be finite and positive")
    if math.isnan(t):
        raise ValueError("t must not be NaN")
    if math.isinf(t):
        return 1.0 if t > 0 else 0.0
    tail = 0.5 * regularized_incomplete_beta(df / (df + t * t), 0.5 * df, 0.5)
    return 1.0 - tail if t >= 0.0 else tail


def student_t_quantile(p: float, df: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie in (0, 1)")
    if p < 0.5:
        return -student_t_quantile(1.0 - p, df)
    if p == 0.5:
        return 0.0
    lo, hi = 0.0, 1.0
    while student_t_cdf(hi, df) < p:
        lo, hi = hi, 2.0 * hi
    while hi - lo > T_TOLERANCE * max(1.0, hi):
        mid = 0.5 * (lo + hi)
        if student_t_cdf(mid, df) >= p:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _log_ratio(a: float, b: float, floor: float) -> float:
    return math.log(max(float(b), floor)) - math.log(max(float(a), floor))


def _differences(pairs: Sequence[PairedOutcome], metric: str) -> list[float]:
    if metric == "success_abs":
        values = [p.success_b - p.success_a for p in pairs]
    elif metric == "tempered_reward_abs":
        values = [p.reward_eta_b - p.reward_eta_a for p in pairs]
    elif metric == "tokens_log_ratio":
        floor = LOG_RATIO_FLOORS[metric]
        values = [_log_ratio(p.tokens_a, p.tokens_b, floor) for p in pairs]
    elif metric == "latency_log_ratio":
        floor = LOG_RATIO_FLOORS[metric]
        values = [_log_ratio(p.latency_a, p.latency_b, floor) for p in pairs]
    else:
        raise ValueError(metric)
    if any(not math.isfinite(v) for v in values):
        raise ValueError(f"non-finite paired difference for {metric}")
    return values


def normal_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def posterior_improvement(
    pairs: Sequence[PairedOutcome], *, noninferiority_passed: bool
) -> dict[str, JsonValue]:
    gains = _differences(pairs, "tempered_reward_abs")
    n = len(gains)
    row: dict[str, JsonValue] = {
        "rule": RULE_IMPROVEMENT_DIAGNOSTIC,
        "metric": IMPROVEMENT_METRIC,
        "posterior": POSTERIOR_MODEL,
        "n": n,
        "mean_diff": None,
        "sample_variance": None,
        "variance_floor": POSTERIOR_VARIANCE_FLOOR,
        "s": None,
        "se": None,
        "p_gain_positive": None,
        "threshold": POSTERIOR_THRESHOLD,
        "noninferiority_passed": noninferiority_passed,
        "passed": False,
    }
    if n < 2:
        row["reason"] = f"n = {n} < 2 paired differences: no posterior"
        return row
    mean = math.fsum(gains) / n
    variance = math.fsum((g - mean) ** 2 for g in gains) / (n - 1)
    floored = max(variance, POSTERIOR_VARIANCE_FLOOR)
    se = math.sqrt(floored / n)
    probability = normal_cdf(mean / se)
    row.update(
        mean_diff=mean,
        sample_variance=variance,
        s=math.sqrt(floored),
        se=se,
        p_gain_positive=probability,
        passed=probability >= POSTERIOR_THRESHOLD,
    )
    return row


def tost_noninferior(
    pairs: Sequence[PairedOutcome],
    *,
    margins: Mapping[str, float],
    alpha: float,
) -> TostResult:
    metrics = ABSOLUTE_METRICS + LOG_RATIO_METRICS
    if set(margins) != set(metrics):
        raise ValueError(f"margins must name exactly {sorted(metrics)}, got {sorted(margins)}")
    for metric, margin in margins.items():
        if not (math.isfinite(margin) and margin >= 0.0):
            raise ValueError(f"margin of {metric} must be finite and >= 0")
    if not (math.isfinite(alpha) and 0.0 < alpha < 0.5):
        raise ValueError("alpha must lie in (0, 0.5)")

    n_pairs = len(pairs)
    per_metric: dict[str, dict[str, JsonValue]] = {}
    for metric in metrics:
        margin = float(margins[metric])
        relative = metric in LOG_RATIO_METRICS
        values = _differences(pairs, metric)
        n = len(values)
        t_critical = student_t_quantile(1.0 - alpha, n - 1) if n_pairs >= 2 else None
        mean = sum(values) / n if n else None
        row: dict[str, JsonValue] = {
            "n": n,
            "mean_diff": mean,
            "se": None,
            "lower_bound": None,
            "upper_bound": None,
            "margin": margin,
            "t_critical": t_critical,
            "direction": "higher-is-worse" if relative else "higher-is-better",
            "passed": False,
        }
        if metric in LOG_RATIO_FLOORS:
            row["floor"] = LOG_RATIO_FLOORS[metric]
        if n_pairs >= 2 and mean is not None and t_critical is not None:
            variance = sum((v - mean) ** 2 for v in values) / (n - 1)
            se = math.sqrt(variance / n)
            lower = mean - t_critical * se
            upper = mean + t_critical * se
            row.update(
                {
                    "se": se,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "passed": upper <= margin if relative else lower >= -margin,
                }
            )
            failed = lower > margin if relative else upper < -margin
            row["status"] = "pass" if row["passed"] else ("fail" if failed else INCONCLUSIVE)
        else:
            row["status"] = INCONCLUSIVE
        per_metric[metric] = row
    passed = n_pairs >= 2 and all(row["passed"] is True for row in per_metric.values())
    improvement = posterior_improvement(pairs, noninferiority_passed=passed)
    improvement["decides"] = False
    return TostResult(passed=passed, per_metric=per_metric, alpha=alpha, improvement=improvement)
