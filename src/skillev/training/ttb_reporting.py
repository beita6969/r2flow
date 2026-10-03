import math
from collections import defaultdict

from skillev.contracts import JsonValue, R2FlowEdgeRecord, TrajectoryRecord

from .step_math import PreparedTTBStep


def _domain(record: TrajectoryRecord) -> str:
    source = record.reward.native_payload.get("training_evidence_source")
    source = source if isinstance(source, dict) else {}
    return str(
        source.get("benchmark_id", record.initial_context.meta.get("benchmark_id", "unknown"))
    )


def ttb_update_diagnostics(prepared: PreparedTTBStep) -> dict[str, JsonValue]:
    total = len(prepared.batch.artifacts)
    residuals = {r.trajectory_id: r for r in prepared.residuals}
    edges: dict[str, list[R2FlowEdgeRecord]] = defaultdict(list)
    for edge in prepared.edges:
        edges[edge.trajectory_id].append(edge)
    rows: list[JsonValue] = []
    by_domain: dict[str, list[float]] = defaultdict(list)
    for artifact in prepared.batch.artifacts:
        record = artifact.record
        r = residuals[record.trajectory_id]
        terms = r.terms
        domain = _domain(record)
        by_domain[domain].append(r.loss)
        rows.append(
            {
                "trajectory_id": record.trajectory_id,
                "domain": domain,
                "horizon": r.horizon,
                "delta_0T": r.delta_0T,
                "loss": r.loss,
                "log_flows": [r.log_z, *r.log_flows, r.log_reward_eta],
                "edges": [
                    {
                        "step_index": e.step_index,
                        "log_pf_reasoning": e.log_pf_reasoning,
                        "log_pf_event": e.log_pf_event,
                        "log_q_reasoning": e.log_q_reasoning,
                        "log_pb_in_edge": e.log_pb_in_edge,
                        "edge_log_ratio": e.edge_log_ratio,
                        "edge_residual": e.edge_residual,
                        "edge_coefficient": e.edge_coefficient,
                    }
                    for e in sorted(edges[record.trajectory_id], key=lambda e: e.step_index)
                ],
                "flow_coefficients": list(terms.flow_coefficients),
                "binary_success": record.reward.success,
            }
        )
    return {
        "format": "read-only-r2flow-decomposition@1",
        "interpretation": "loss contribution is not gradient contribution or causal task benefit",
        "sampling_policy_snapshot_id": prepared.snapshot_before.snapshot_id,
        "sampled_policy_step": prepared.batch.optimizer_step - 1,
        "optimizer_step": prepared.batch.optimizer_step,
        "trajectory_count": total,
        "trajectories": rows,
        "domains": {
            domain: {
                "trajectory_count": len(values),
                "mean_subtb_loss": math.fsum(values) / len(values),
                "batch_loss_contribution": math.fsum(values) / total,
            }
            for domain, values in sorted(by_domain.items())
        },
    }
