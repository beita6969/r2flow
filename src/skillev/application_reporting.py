from __future__ import annotations

from typing import TYPE_CHECKING

from skillev.contracts import JsonValue
from skillev.policy import AdapterRole
from skillev.training.coverage_reporting import batch_coverage
from skillev.training.skill_lifecycle import skill_lifecycle_report

if TYPE_CHECKING:
    from skillev.application import SKILLEVApplication


def resolved_method_state(application: SKILLEVApplication) -> dict[str, JsonValue]:
    state = application.projections.runtime_state()
    provenance = state.posterior_provenance
    latest_batch = provenance.batches[-1] if provenance.batches else None
    backbone = application.backbone
    return {
        "application_config": application.public_identity.application_config.to_value(),
        "optimizer_step": application.training_loop.optimizer_step,
        "policy_snapshot_id": application.training_loop.policy_snapshot_id,
        "forward_adapter_version": backbone.adapter_version(AdapterRole.FORWARD_POLICY),
        "backward_adapter_version": backbone.adapter_version(AdapterRole.BACKWARD_POLICY),
        "z_version": backbone.z_version,
        "library_version": application.library.current_version,
        "active_skill_ids": list(application.library.active_skill_ids),
        "projection_revision": state.revision,
        "terminal_evaluation_conditions": application.snapshot_identity.to_value().get(
            "terminal_evaluation_conditions"
        ),
        "task_feature_mapping_version": application.snapshot_identity.task_feature_mapping_version,
        "posterior_batch_count": len(provenance.batches),
        "actual_library_mutation_count": application.run_progress.state.committed_cycles,
        "committed_proposal_count": application.run_progress.state.committed_actions,
        "last_evidence_batch": None if latest_batch is None else latest_batch.posterior.batch_id,
        "last_evidence_policy": None if latest_batch is None else latest_batch.policy_snapshot_id,
        "last_evidence_library": None if latest_batch is None else latest_batch.library_version,
        "skill_coverage": [batch_coverage(batch) for batch in provenance.batches],
        "skill_lifecycle": skill_lifecycle_report(application.library, provenance),
        "residual_statistic": "mean-raw-delta-squared",
        "optimization_statistic": "mean-horizon-normalized-delta-squared",
    }
