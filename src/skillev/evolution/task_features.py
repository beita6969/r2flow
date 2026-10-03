from dataclasses import dataclass

TASK_FEATURE_MAPPING_VERSION = "benchmark-public-task@1"
_FAMILIES = {
    "hotpotqa": "multi-hop-qa",
    "triviaqa": "factual-qa",
    "aime-2026": "integer-answer",
    "healthbench": "health-dialogue",
    "webshop": "shopping",
    "alfworld": "unspecified",
    "mbpp-plus": "code-generation",
    "musique": "multi-hop-qa",
    "nq-open": "factual-qa",
    "math-hard": "mathematical-reasoning",
    "gpqa-diamond-bioorganic": "scientific-multiple-choice",
}


TRANSFERABLE_FAMILY_MAPPING_VERSION = "transferable-task-family@1"
_TRANSFERABLE = {
    "hotpotqa": "multi-hop-qa",
    "musique": "multi-hop-qa",
    "triviaqa": "factual-qa",
    "nq-open": "factual-qa",
    "aime-2026": "mathematical-reasoning",
    "math-hard": "mathematical-reasoning",
    "healthbench": "health-dialogue",
    "alfworld": "interactive-decision",
    "webshop": "interactive-decision",
    "mbpp-plus": "code-generation",
}


def transferable_family(benchmark_id: str | None) -> str | None:
    return None if benchmark_id is None else _TRANSFERABLE.get(benchmark_id)


@dataclass(frozen=True, slots=True)
class PublicTaskFeatures:
    task_family: str
    context_id: str
    mapping_version: str = TASK_FEATURE_MAPPING_VERSION

    def to_value(self) -> dict[str, str]:
        return {
            "task_family": self.task_family,
            "context_id": self.context_id,
            "mapping_version": self.mapping_version,
        }


def public_task_features(
    benchmark: str, *, task_family: str | None = None, context_id: str | None = None
) -> PublicTaskFeatures:
    if benchmark not in _FAMILIES:
        raise ValueError("no public feature mapping for this benchmark")
    family = task_family or _FAMILIES[benchmark]
    if family == "public-task":
        family = _FAMILIES[benchmark]
    if not family.startswith(f"{benchmark}/"):
        family = f"{benchmark}/{family}"
    return PublicTaskFeatures(family, context_id or f"{benchmark}:task")


def configured_public_task_features(
    benchmark: str, settings: dict[str, object]
) -> PublicTaskFeatures:
    if settings.keys() - {"task_family", "context_id", "mapping_version"}:
        raise ValueError("only public task family and context may configure retrieval")
    family, context = settings.get("task_family"), settings.get("context_id")
    if any(
        value is not None and (not isinstance(value, str) or not value.strip())
        for value in (family, context)
    ):
        raise ValueError("public retrieval features must be nonempty text")
    if family is not None and not isinstance(family, str):
        raise TypeError("task family must be text")
    if context is not None and not isinstance(context, str):
        raise TypeError("context must be text")
    if (
        settings.get("mapping_version", TASK_FEATURE_MAPPING_VERSION)
        != TASK_FEATURE_MAPPING_VERSION
    ):
        raise ValueError("unsupported public task feature mapping")
    return public_task_features(benchmark, task_family=family, context_id=context)
