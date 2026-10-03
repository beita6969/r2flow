from __future__ import annotations

from skillev.contracts import JsonValue

from .qa_metrics import normalize_hotpotqa_answer, normalize_triviaqa_answer


def qa_answer_diagnostics(
    benchmark: str,
    *,
    original_submission: str | None,
    projected_answer: str | None,
    accepted_aliases: tuple[str, ...],
) -> dict[str, JsonValue]:
    if benchmark not in {"hotpotqa", "triviaqa"}:
        raise ValueError("QA normalization diagnostics require the actual QA benchmark")
    normalize = normalize_hotpotqa_answer if benchmark == "hotpotqa" else normalize_triviaqa_answer
    predicted = None if projected_answer is None else normalize(projected_answer)
    normalized_aliases = [normalize(alias) for alias in accepted_aliases]
    return {
        "format": "private-qa-normalization@1",
        "benchmark": benchmark,
        "original_submission": original_submission,
        "projected_answer": projected_answer,
        "normalized_answer": predicted,
        "accepted_aliases": list(accepted_aliases),
        "normalized_aliases": list(normalized_aliases),
        "exact_matching_alias_indices": None
        if predicted is None
        else [index for index, alias in enumerate(normalized_aliases) if alias == predicted],
        "cause": None,
    }
