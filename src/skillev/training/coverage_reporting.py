from collections import defaultdict
from collections.abc import Iterable
from typing import cast

from skillev.contracts import JsonValue

from .evidence_context import TrajectoryEvidenceContext
from .posterior_state import PosteriorEvidenceBatch


def _skill_ids(values: Iterable[str]) -> list[JsonValue]:
    return cast(list[JsonValue], sorted(set(values)))


def batch_coverage(batch: PosteriorEvidenceBatch) -> dict[str, JsonValue]:
    groups: dict[str, list[TrajectoryEvidenceContext]] = defaultdict(list)
    for context in batch.trajectory_contexts:
        groups[context.benchmark_id or "unknown"].append(context)
    benchmarks: dict[str, JsonValue] = {}
    for name, contexts in sorted(groups.items()):
        edge_count = sum(context.executed_edge_count for context in contexts)
        invoking = sum(context.invoking_edge_count for context in contexts)
        benchmarks[name] = {
            "canonical_source_question_count": len(
                {c.canonical_source_key for c in contexts if c.canonical_source_key is not None}
            ),
            "canonical_source_unknown_trajectory_count": sum(
                c.canonical_source_key is None for c in contexts
            ),
            "rejected_declaration_count": sum(
                not link.admitted for context in contexts for link in context.invocation_links or ()
            ),
            "admitted_without_following_execution_count": sum(
                link.admitted and not link.following_execution_steps
                for context in contexts
                for link in context.invocation_links or ()
            ),
            "admitted_terminal_failure_count": sum(
                link.admitted and not link.terminal_success
                for context in contexts
                for link in context.invocation_links or ()
            ),
            "exposed_but_uncalled_trajectory_count": sum(
                bool(context.visible_skill_ids) and context.invoking_edge_count == 0
                for context in contexts
            ),
            "invocation_link_unknown_trajectory_count": sum(
                context.invocation_links is None for context in contexts
            ),
            "trajectory_count": len(contexts),
            "distinct_source_question_count": len(
                {context.source_key for context in contexts if context.source_key is not None}
            ),
            "source_unknown_trajectory_count": sum(
                context.source_key is None for context in contexts
            ),
            "matched_skill_ids": _skill_ids(
                {skill for context in contexts for skill in context.matched_skill_ids or ()}
            ),
            "match_unknown_trajectory_count": sum(
                context.matched_skill_ids is None for context in contexts
            ),
            "retrieved_skill_ids": _skill_ids(
                {skill for context in contexts for skill in context.retrieved_skill_ids}
            ),
            "catalog_or_inline_visible_skill_ids": _skill_ids(
                skill for context in contexts for skill in context.visible_skill_ids
            ),
            "catalog_or_inline_visible_trajectory_count": sum(
                bool(context.visible_skill_ids) for context in contexts
            ),
            "invoking_trajectory_count": sum(
                context.invoking_edge_count > 0 for context in contexts
            ),
            "invoking_edge_count": invoking,
            "execution_edge_count": edge_count,
            "invocation_edge_fraction": invoking / edge_count if edge_count else 0.0,
        }
    return {
        "format": "skill-coverage-report@2",
        "batch_id": batch.posterior.batch_id,
        "optimizer_step": batch.optimizer_step,
        "policy_snapshot_id": batch.policy_snapshot_id,
        "library_version": batch.library_version,
        "trajectory_count": len(batch.trajectory_ids),
        "metadata_missing_trajectory_count": len(batch.trajectory_ids)
        - len(batch.trajectory_contexts),
        "benchmarks": benchmarks,
        "credit_unit": "actual-invocation-only-not-retrieval-or-exposure",
    }
