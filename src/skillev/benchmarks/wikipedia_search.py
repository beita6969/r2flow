from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, TypeVar

from skillev.contracts import JsonValue, normalize_json
from skillev.contracts.wikipedia_search import (
    PASSAGE_TEXT_CAP,
    PASSAGES_PER_QUERY,
    SEARCH_RESULTS_STATUS,
    WIKIPEDIA_SEARCH_RESOURCE,
    WIKIPEDIA_SEARCH_TOOL_NAME,
    render_wikipedia_passage,
)
from skillev.rollout.errors import EpisodeInfrastructureError
from skillev.runtime import ActionKind, BudgetVector, StructuredAction
from skillev.runtime.execution import EnvironmentObservation, RolloutEnvironmentSession

SEARCH_TIMEOUT_ERROR: Final = "corpus_search_timeout"
SEARCH_TIMEOUT_RULE: Final = "corpus-search-timeout=infrastructure@1"
SEARCH_ATTEMPTS: Final = 3

_T = TypeVar("_T")


class CorpusSearchInfrastructureError(EpisodeInfrastructureError):
    pass


async def search_with_deadline_retries(
    search: Callable[[], Awaitable[_T]], on_timeout: Callable[[], None], *, corpus: str
) -> _T:
    for attempt in range(1, SEARCH_ATTEMPTS + 1):
        try:
            return await search()
        except TimeoutError as error:
            on_timeout()
            if attempt == SEARCH_ATTEMPTS:
                raise CorpusSearchInfrastructureError(
                    f"{SEARCH_TIMEOUT_ERROR}: {SEARCH_ATTEMPTS} attempts exceeded the "
                    f"{corpus} search deadline"
                ) from error
    raise AssertionError("unreachable")


@dataclass(frozen=True, slots=True)
class WikipediaPassage:
    id: str
    title: str
    text: str

    def to_value(self) -> dict[str, JsonValue]:
        rendered = render_wikipedia_passage(self.title, self.text)
        return {
            "id": self.id,
            "text": rendered[:PASSAGE_TEXT_CAP],
            "truncated": len(rendered) > PASSAGE_TEXT_CAP,
        }


class WikipediaSearchBackend(Protocol):
    async def search(self, query: str) -> tuple[WikipediaPassage, ...]: ...


def wikipedia_search_observation(
    query: str, passages: Sequence[WikipediaPassage]
) -> EnvironmentObservation:
    if len(passages) > PASSAGES_PER_QUERY:
        raise ValueError("corpus_search returns at most five passages")
    return EnvironmentObservation(
        normalize_json(
            {
                "status": SEARCH_RESULTS_STATUS,
                "resource": WIKIPEDIA_SEARCH_RESOURCE,
                "query": query,
                "passages": [passage.to_value() for passage in passages],
            }
        ),
        "success",
        budget_usage=BudgetVector(tool_calls=1),
    )


class WikipediaSearchEnvironment:
    def __init__(
        self, delegate: RolloutEnvironmentSession, backend: WikipediaSearchBackend
    ) -> None:
        self.delegate = delegate
        self.backend = backend
        self._last_step = 0
        self.timeouts = 0

    @property
    def environment_id(self) -> str:
        return self.delegate.environment_id

    @property
    def task_family(self) -> str:
        return self.delegate.task_family

    def validate_completion(self, submission: JsonValue) -> bool:
        return self.delegate.validate_completion(submission)

    async def execute(self, action: StructuredAction, *, step_index: int) -> EnvironmentObservation:
        if (
            action.kind is ActionKind.TOOL
            and action.resource_id == WIKIPEDIA_SEARCH_RESOURCE
            and action.name == WIKIPEDIA_SEARCH_TOOL_NAME
        ):
            if step_index <= self._last_step:
                raise RuntimeError("corpus_search environment requires an increasing step")
            self._last_step = step_index
            arguments = action.arguments
            query = arguments.get("query") if isinstance(arguments, dict) else None
            if not isinstance(query, str) or not query.strip():
                return EnvironmentObservation({"error": "query_required"}, "schema_invalid")
            text: str = query

            def missed() -> None:
                self.timeouts += 1

            passages = await search_with_deadline_retries(
                lambda: self.backend.search(text), missed, corpus="Wikipedia"
            )
            return wikipedia_search_observation(text, passages)
        return await self.delegate.execute(action, step_index=step_index)


__all__ = [
    "SEARCH_ATTEMPTS",
    "SEARCH_TIMEOUT_ERROR",
    "SEARCH_TIMEOUT_RULE",
    "CorpusSearchInfrastructureError",
    "WikipediaPassage",
    "WikipediaSearchBackend",
    "WikipediaSearchEnvironment",
    "wikipedia_search_observation",
]
