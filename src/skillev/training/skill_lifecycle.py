from skillev.contracts import JsonValue
from skillev.runtime import SkillLibrary

from .posterior_state import PosteriorEventProvenance


def skill_lifecycle_report(
    library: SkillLibrary, provenance: PosteriorEventProvenance
) -> dict[str, JsonValue]:
    catalogs: list[JsonValue] = []
    by_skill: dict[str, list[JsonValue]] = {}
    documents = library.all_documents()
    for batch in provenance.batches:
        for context in batch.trajectory_contexts:
            catalogs.append(
                {
                    "batch_id": batch.posterior.batch_id,
                    "optimizer_step": batch.optimizer_step,
                    "trajectory_id": context.trajectory_id,
                    "policy_snapshot_id": batch.policy_snapshot_id,
                    "library_version": batch.library_version,
                    "task_family": context.task_family,
                    "skill_exposure": context.skill_exposure,
                    "terminal_verifier_version": context.terminal_verifier_version,
                    "source": list(context.canonical_source_key)
                    if context.canonical_source_key is not None
                    else None,
                    "visible_ids": list(context.visible_skill_ids),
                    "matched_ids": list(context.matched_skill_ids)
                    if context.matched_skill_ids is not None
                    else None,
                    "reads": [link.to_value() for link in context.invocation_links]
                    if context.invocation_links is not None
                    else None,
                }
            )
        for document in documents:
            skill_id = document.manifest.skill_id
            contexts = [c for c in batch.trajectory_contexts if skill_id in c.active_skill_ids]
            if not contexts:
                continue
            links = [
                link
                for c in contexts
                for link in c.invocation_links or ()
                if link.declared_skill_id == skill_id
            ]
            links_known = all(c.invocation_links is not None for c in contexts)
            by_skill.setdefault(skill_id, []).append(
                {
                    "optimizer_step": batch.optimizer_step,
                    "library_version": batch.library_version,
                    "matched_trajectories": sum(
                        skill_id in (c.matched_skill_ids or ()) for c in contexts
                    )
                    if all(c.matched_skill_ids is not None for c in contexts)
                    else None,
                    "visible_trajectories": sum(skill_id in c.visible_skill_ids for c in contexts),
                    "admitted_call_count": sum(link.admitted for link in links)
                    if links_known
                    else None,
                    "calls_with_following_actions": sum(
                        link.admitted and bool(link.following_execution_steps) for link in links
                    )
                    if links_known
                    else None,
                }
            )
    return {
        "format": "skill-discovery-evidence-chain@2",
        "interpretation": (
            "observed-discovery-and-temporal-following-not-causal-efficacy; no-minimum-call-count"
        ),
        "credit_rule": (
            "admitted-explicit-read/declaration; terminal-success-label; "
            "full-batch-mean-one-invocation-flow; not-independent-execution-or-causal-use"
        ),
        "catalogs": catalogs,
        "skills": [
            {
                "skill_id": document.manifest.skill_id,
                "version": document.manifest.version,
                "title": document.title,
                "summary": document.summary,
                "applicability": document.applicability.to_value(),
                "active_now": document.manifest.skill_id in library.active_skill_ids,
                "batches": by_skill.get(document.manifest.skill_id, []),
                "availability_status": "available-in-training-batch"
                if document.manifest.skill_id in by_skill
                else "not-yet-available-in-a-training-batch",
            }
            for document in documents
        ],
    }
