from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from skillev.contracts import JsonValue, normalize_json, stable_hash
from skillev.rollout import RolloutTask
from skillev.runtime import (
    ActionKind,
    BudgetVector,
    EnvironmentMethodFailedError,
    EnvironmentObservation,
    OrderedTaskCursorState,
    StructuredAction,
)
from skillev.runtime.skill_invocation import skill_invocation_observation

from .retrieval import RetrievalIndexManifest
from .task_family import require_benchmark_task_family


def _text(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value


@dataclass(frozen=True, slots=True)
class BenchmarkPublicItem:
    benchmark_id: str
    dataset_revision: str
    split: str
    task_id: str
    task_family: str
    query: str
    public_context: JsonValue

    def __post_init__(self) -> None:
        for field_name in (
            "benchmark_id",
            "dataset_revision",
            "split",
            "task_id",
            "task_family",
            "query",
        ):
            _text(getattr(self, field_name), field_name=field_name)
        require_benchmark_task_family(
            benchmark_id=self.benchmark_id,
            task_family=self.task_family,
        )
        object.__setattr__(self, "public_context", normalize_json(self.public_context))

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "benchmark_id": self.benchmark_id,
            "dataset_revision": self.dataset_revision,
            "public_context": self.public_context,
            "query": self.query,
            "split": self.split,
            "task_family": self.task_family,
            "task_id": self.task_id,
        }

    @classmethod
    def from_value(cls, value: object) -> BenchmarkPublicItem:
        normalized = normalize_json(value)
        fields = {
            "benchmark_id",
            "dataset_revision",
            "public_context",
            "query",
            "split",
            "task_family",
            "task_id",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("BenchmarkPublicItem has incompatible fields")
        for field_name in fields - {"public_context"}:
            if type(normalized[field_name]) is not str:
                raise TypeError(f"{field_name} must be text")
        return cls(
            benchmark_id=cast(str, normalized["benchmark_id"]),
            dataset_revision=cast(str, normalized["dataset_revision"]),
            split=cast(str, normalized["split"]),
            task_id=cast(str, normalized["task_id"]),
            task_family=cast(str, normalized["task_family"]),
            query=cast(str, normalized["query"]),
            public_context=normalized["public_context"],
        )

    @property
    def environment_id(self) -> str:
        return f"benchmark:{self.benchmark_id}@{self.dataset_revision}"

    def to_rollout_task(self) -> RolloutTask:
        return RolloutTask(
            task_id=self.task_id,
            environment_id=self.environment_id,
            task_family=self.task_family,
            context_id=f"{self.benchmark_id}:{self.split}",
            query=self.query,
            available_tools=(),
            public_context={
                "benchmark_id": self.benchmark_id,
                "dataset_revision": self.dataset_revision,
                "payload": self.public_context,
                "split": self.split,
            },
        )

    def to_retrieval_rollout_task(self, manifest: RetrievalIndexManifest) -> RolloutTask:
        if not isinstance(manifest, RetrievalIndexManifest):
            raise TypeError("manifest must be a RetrievalIndexManifest")
        return RolloutTask(
            task_id=self.task_id,
            environment_id=(
                f"benchmark:{self.benchmark_id}@{self.dataset_revision}:"
                f"retrieval:{manifest.index_id}"
            ),
            task_family=self.task_family,
            context_id=f"{self.benchmark_id}:{self.split}:retrieval",
            query=self.query,
            available_tools=("read", "search"),
            public_context={
                "benchmark_id": self.benchmark_id,
                "dataset_revision": self.dataset_revision,
                "payload": self.public_context,
                "retrieval_index": manifest.to_value(),
                "split": self.split,
                "tools": {
                    "read": {"arguments": ["passage_id"]},
                    "search": {"arguments": ["limit", "query"]},
                },
            },
        )


@dataclass(slots=True)
class OrderedBenchmarkTaskProvider:
    items: tuple[BenchmarkPublicItem, ...]
    cursor: int = 0

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("benchmark task provider requires at least one item")
        identities = tuple(item.task_id for item in self.items)
        if len(set(identities)) != len(identities):
            raise ValueError("benchmark task identities must be unique")

    def next_task(self) -> RolloutTask:
        if self.cursor >= len(self.items):
            raise RuntimeError("benchmark public curriculum is exhausted")
        task = self.items[self.cursor].to_rollout_task()
        self.cursor += 1
        return task

    @property
    def runtime_state(self) -> OrderedTaskCursorState:
        return OrderedTaskCursorState(
            curriculum_id=stable_hash(
                {"kind": "completion", "task_ids": [item.task_id for item in self.items]}
            ),
            cursor=self.cursor,
        )


@dataclass(slots=True)
class OrderedRetrievalTaskProvider:
    items: tuple[BenchmarkPublicItem, ...]
    manifest: RetrievalIndexManifest
    cursor: int = 0

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("retrieval task provider requires at least one item")
        identities = tuple(item.task_id for item in self.items)
        if len(set(identities)) != len(identities):
            raise ValueError("retrieval benchmark task identities must be unique")
        if not isinstance(self.manifest, RetrievalIndexManifest):
            raise TypeError("manifest must be a RetrievalIndexManifest")

    def next_task(self) -> RolloutTask:
        if self.cursor >= len(self.items):
            raise RuntimeError("retrieval benchmark curriculum is exhausted")
        task = self.items[self.cursor].to_retrieval_rollout_task(self.manifest)
        self.cursor += 1
        return task

    @property
    def runtime_state(self) -> OrderedTaskCursorState:
        return OrderedTaskCursorState(
            curriculum_id=stable_hash(
                {
                    "index_id": self.manifest.index_id,
                    "kind": "retrieval",
                    "task_ids": [item.task_id for item in self.items],
                }
            ),
            cursor=self.cursor,
        )


@dataclass(slots=True)
class CompletionBenchmarkEnvironment:
    item: BenchmarkPublicItem
    _skill_invocations: list[str] = field(default_factory=list, init=False, repr=False)
    _last_environment_step_index: int = field(default=0, init=False, repr=False)

    @property
    def environment_id(self) -> str:
        return self.item.environment_id

    @property
    def task_family(self) -> str:
        return self.item.task_family

    async def execute(
        self,
        action: StructuredAction,
        *,
        step_index: int,
    ) -> EnvironmentObservation:
        if step_index <= self._last_environment_step_index:
            raise RuntimeError("benchmark environment requires an increasing step")
        self._last_environment_step_index = step_index
        if action.kind is not ActionKind.SKILL or action.skill_id is None:
            raise EnvironmentMethodFailedError(
                budget_usage=BudgetVector(tool_calls=1),
                public_error_code="unsupported_benchmark_action",
            )
        self._skill_invocations.append(action.skill_id)
        return skill_invocation_observation(action.skill_id)

    def validate_completion(self, submission: JsonValue) -> bool:
        normalized = normalize_json(submission)
        return (
            isinstance(normalized, dict)
            and set(normalized) == {"answer"}
            and type(normalized["answer"]) is str
            and bool(normalized["answer"].strip())
        )


__all__ = [
    "BenchmarkPublicItem",
    "CompletionBenchmarkEnvironment",
    "OrderedBenchmarkTaskProvider",
    "OrderedRetrievalTaskProvider",
]
