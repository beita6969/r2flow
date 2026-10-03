from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

from skillev.rollout import (
    RolloutSessionBundle,
    RolloutTask,
    UnskilledRolloutSessionBundle,
)
from skillev.runtime import (
    FullRetrievedSkillContext,
    RetrievalInclusionReason,
    SkillDocument,
    SkillLibrary,
    SkillMetadata,
    model_visible_skill_content,
)

from .task_features import transferable_family


@dataclass(frozen=True, slots=True)
class TaskRetrievalFeatures:
    task_id: str
    task_family: str
    context: str
    available_tools: tuple[str, ...]
    benchmark_id: str | None = None
    root_query: str = ""


def task_retrieval_features(task: RolloutTask) -> TaskRetrievalFeatures:
    benchmark_id: str | None = None
    if isinstance(task.public_context, dict):
        candidate = task.public_context.get("benchmark_id")
        if type(candidate) is str and candidate:
            benchmark_id = candidate
    return TaskRetrievalFeatures(
        task_id=task.task_id,
        task_family=task.task_family,
        context=task.context_id,
        available_tools=task.available_tools,
        benchmark_id=benchmark_id,
        root_query=task.query,
    )


def _features_benchmark(features: TaskRetrievalFeatures) -> str | None:
    if features.benchmark_id is not None:
        return features.benchmark_id
    prefix, separator, _ = features.task_family.partition("/")
    return prefix if separator else None


def document_mismatch_reasons(
    document: SkillDocument, features: TaskRetrievalFeatures
) -> tuple[str, ...]:
    if not isinstance(features, TaskRetrievalFeatures):
        raise TypeError("retrieval requires pre-action public features, not post-hoc calibration")
    family = transferable_family(_features_benchmark(features))
    if family is None or family not in document.applicability.task_families:
        return ("transferable-family-mismatch",)
    return ()


class TaskConditionedSkillRetriever:
    def __init__(
        self,
        *,
        library: SkillLibrary,
    ) -> None:
        self._library = library

    @property
    def library(self) -> SkillLibrary:
        return self._library

    def retrieve(self, task: RolloutTask) -> tuple[FullRetrievedSkillContext, ...]:
        return self.retrieve_features(task_retrieval_features(task))

    def retrieve_features(
        self, features: TaskRetrievalFeatures
    ) -> tuple[FullRetrievedSkillContext, ...]:
        matches = sorted(
            (
                document
                for document in self._library.active_documents()
                if not document_mismatch_reasons(document, features)
            ),
            key=lambda document: document.manifest.skill_id,
        )
        return tuple(
            FullRetrievedSkillContext(
                metadata=SkillMetadata.from_document(document),
                content=model_visible_skill_content(document),
                inclusion_reason=RetrievalInclusionReason.APPLICABILITY_MATCH,
            )
            for document in matches
        )


class BaseRolloutSessionFactory(Protocol):
    def create(self, task: RolloutTask) -> UnskilledRolloutSessionBundle: ...


class RetrievingRolloutSessionFactory:
    def __init__(
        self,
        *,
        base_factory: BaseRolloutSessionFactory,
        retriever: TaskConditionedSkillRetriever,
    ) -> None:
        self._base_factory = base_factory
        self._retriever = retriever

    async def prepare_tasks(self, tasks: tuple[RolloutTask, ...]) -> tuple[RolloutTask, ...]:
        prepare = getattr(self._base_factory, "prepare_tasks", None)
        return tasks if prepare is None else cast(tuple[RolloutTask, ...], await prepare(tasks))

    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        base = self._base_factory.create(task)
        return RolloutSessionBundle(
            environment=base.environment,
            evaluator=base.evaluator,
            retrieved_skills=self._retriever.retrieve(task),
            cleanup=base.cleanup,
            verifier=base.verifier,
        )


__all__ = [
    "BaseRolloutSessionFactory",
    "RetrievingRolloutSessionFactory",
    "TaskConditionedSkillRetriever",
    "TaskRetrievalFeatures",
    "document_mismatch_reasons",
    "task_retrieval_features",
]
