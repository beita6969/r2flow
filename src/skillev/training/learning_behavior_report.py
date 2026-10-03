from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

FIELDS = (
    "raw_delta_squared",
    "normalized_ttb_loss",
    "delta",
    "log_z",
    "sum_forward",
    "sum_backward",
    "forward_per_edge",
    "backward_per_edge",
    "reward_term",
    "signed_backward_term",
    "signed_reward_term",
    "raw_reward",
)

_R2FLOW_GROUPS = ("forward", "backward", "z", "psi")


def _number(value: Any) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("committed diagnostic scalar must be finite or absent")
    return float(value)


def _source(record: dict[str, Any]) -> tuple[str, str] | None:
    native = record.get("reward", {}).get("native_payload", {})
    source = native.get("training_evidence_source")
    if isinstance(source, dict) and all(
        isinstance(source.get(k), str) and source[k] for k in ("benchmark_id", "source_question_id")
    ):
        return source["benchmark_id"], source["source_question_id"]
    return None


def _summary(rows: list[dict[str, Any]], batch_size: int) -> dict[str, Any]:
    result: dict[str, Any] = {"trajectory_count": len(rows)}
    for field in FIELDS:
        values = [r[field] for r in rows]
        known = [v for v in values if v is not None]
        result[field + "_mean"] = (
            math.fsum(known) / len(rows) if rows and len(known) == len(rows) else None
        )
        result[field + "_observed_count"] = len(known)
    deltas = [r["delta"] for r in rows]
    complete = all(v is not None for v in deltas)
    result["positive_delta_count"] = sum(v > 0 for v in deltas) if complete else None
    result["negative_delta_count"] = sum(v < 0 for v in deltas) if complete else None
    result["zero_delta_count"] = sum(v == 0 for v in deltas) if complete else None
    mean = result["normalized_ttb_loss_mean"]
    result["global_batch_loss_contribution"] = (
        mean * len(rows) / batch_size if mean is not None else None
    )
    return result


def _contrasts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        source = _source(record)
        if source is not None:
            groups.setdefault(source, []).append(record)
    results = []
    for (domain, source_id), members in sorted(groups.items()):
        pairs = []
        for left, right in itertools.combinations(members, 2):
            successes = [r.get("reward", {}).get("success") for r in (left, right)]
            steps = [r.get("steps") for r in (left, right)]
            actions = [
                [s.get("action_text") for s in seq] if isinstance(seq, list) else None
                for seq in steps
            ]
            known = all(seq is not None and all(isinstance(a, str) for a in seq) for seq in actions)
            difference = None
            if known:
                a, b = actions
                assert a is not None
                assert b is not None
                difference = sum(x != y for x, y in itertools.zip_longest(a, b))
            pairs.append(
                {
                    "trajectory_ids": [left["trajectory_id"], right["trajectory_id"]],
                    "terminal_success_disagrees": successes[0] != successes[1]
                    if all(type(v) is bool for v in successes)
                    else None,
                    "raw_action_sequence_equal": difference == 0
                    if difference is not None
                    else None,
                    "differing_action_positions": difference,
                }
            )
        results.append(
            {
                "benchmark_id": domain,
                "canonical_source_id": source_id,
                "trajectory_count": len(members),
                "pair_count": len(pairs),
                "pairs": pairs,
                "scope": (
                    "within-committed-batch; population aliases grouped; "
                    "observational-not-independent-or-causal"
                ),
            }
        )
    return results


def committed_learning_report(event: dict[str, Any], *, condition_id: str) -> dict[str, Any]:
    if event.get("event_type") != "training_step_committed":
        raise ValueError("learning report requires the original committed event")
    payload = event["payload"]
    stats_format = payload.get("stats", {}).get("format")
    if stats_format != "r2flow-batch-stats@1":
        raise ValueError("unknown training stats format")
    records = payload.get("records", [])
    residuals = payload.get("stats", {}).get("residuals", [])
    residuals = [
        {
            **r,
            "delta": r.get("delta_0T"),
            "log_shifted_reward": None,
            "temperature_beta": None,
            "sum_forward": None,
            "sum_backward": None,
        }
        for r in residuals
    ]
    ids = [r["trajectory_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate committed trajectory")
    by_id = {r["trajectory_id"]: r for r in residuals}
    if len(by_id) != len(residuals) or (records and residuals and ids != list(by_id)):
        raise ValueError("residual population/order differs from committed records")
    rows = []
    for record in records:
        residual = by_id.get(record["trajectory_id"], {})
        horizon = residual.get("horizon")
        if horizon is not None and (type(horizon) is not int or horizon < 1):
            raise ValueError("original residual horizon must be positive")
        if (
            horizon is not None
            and isinstance(record.get("steps"), list)
            and horizon != len(record["steps"])
        ):
            raise ValueError("original residual/action horizon differs")
        delta = _number(residual.get("delta"))
        forward, backward = (
            _number(residual.get("sum_forward")),
            _number(residual.get("sum_backward")),
        )
        log_reward, beta = (
            _number(residual.get("log_shifted_reward")),
            _number(residual.get("temperature_beta")),
        )
        term = beta * log_reward if beta is not None and log_reward is not None else None
        source = _source(record)
        native = record.get("reward", {}).get("native_payload", {})
        rows.append(
            {
                "trajectory_id": record["trajectory_id"],
                "benchmark_id": native.get("benchmark_id"),
                "canonical_source": source,
                "horizon": horizon,
                "delta": delta,
                "raw_delta_squared": delta**2 if delta is not None else None,
                "normalized_ttb_loss": (delta / horizon) ** 2
                if delta is not None and horizon is not None
                else None,
                "log_z": _number(residual.get("log_z")),
                "sum_forward": forward,
                "sum_backward": backward,
                "forward_per_edge": forward / horizon
                if forward is not None and horizon is not None
                else None,
                "backward_per_edge": backward / horizon
                if backward is not None and horizon is not None
                else None,
                "reward_term": term,
                "signed_reward_term": -term if term is not None else None,
                "signed_backward_term": -backward if backward is not None else None,
                "raw_reward": _number(record.get("reward", {}).get("value")),
            }
        )
        rows[-1]["normalized_ttb_loss"] = _number(residual.get("loss"))
    grouped = {}
    for field in ("benchmark_id", "canonical_source", "horizon"):
        groups: dict[Any, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(row[field], []).append(row)
        grouped[field] = [
            {"group": key, **_summary(members, len(rows))} for key, members in groups.items()
        ]
    report = payload.get("report", {})
    transition = report.get("optimizer_transition")
    components = transition.get("components") if isinstance(transition, dict) else None
    return {
        "format": "committed-learning-behavior@1",
        "run_id": event["run_id"],
        "condition_id": condition_id,
        "source_commit_id": event["event_id"],
        "optimizer_step": payload["optimizer_step"],
        "sampled_policy_step": payload["optimizer_step"] - 1,
        "policy_snapshot_id": payload.get("policy_snapshot_before"),
        "library_version": payload.get("library_version"),
        "summary": _summary(rows, len(rows)) if rows else None,
        "groups": grouped,
        "trajectories": rows,
        "source_contrasts": _contrasts(records),
        "component_gradient_norms": {
            group: _number(report.get("grad_norm_" + group)) for group in _R2FLOW_GROUPS
        },
        "optimizer_transition": transition,
        "component_parameter_changes": {
            name: data.get("parameters") for name, data in components.items()
        }
        if isinstance(components, dict)
        else None,
        "optimizer_state_anomalies": {name: data.get("adam") for name, data in components.items()}
        if isinstance(components, dict)
        else None,
        "uncovered": [
            *(
                ["parameter changes/Adam state require original before-after evidence"]
                if components is None
                else []
            ),
            "fixed-panel before-after comparison is a separate probe",
            "same-source pairs are not independent causal trials",
        ],
        "units": {
            "raw_delta_squared": "delta squared, without horizon division",
            "normalized_ttb_loss": "(delta / T)^2; group contribution keeps original global B",
            "forward_backward": "sum over edges of original per-action-token mean logprob (nats)",
            "signed_terms": "delta = logZ + F - beta*log_shifted_reward - B",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Private, read-only committed TTB/source contrasts; never an alternate scorer."
    )
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--first-step", type=int, default=1)
    parser.add_argument("--last-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.first_step < 1 or args.last_step < args.first_step:
        parser.error("positive ordered committed step range required")
    reports = []
    run = None
    with args.events.open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            event = json.loads(line)
            if event.get("event_type") != "training_step_committed":
                continue
            if run is not None and run != event["run_id"]:
                raise ValueError("one learning report cannot mix source runs")
            run = event["run_id"]
            if args.first_step <= event["payload"]["optimizer_step"] <= args.last_step:
                reports.append(committed_learning_report(event, condition_id=args.condition_id))
    with args.output.open("x") as stream:
        json.dump(
            {"format": "learning-behavior-export@1", "reports": reports}, stream, allow_nan=False
        )
        stream.write("\n")


if __name__ == "__main__":
    main()
