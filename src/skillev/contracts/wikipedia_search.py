from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from skillev.contracts import JsonValue

WIKIPEDIA_SEARCH_TOOL_NAME: Final = "corpus_search"
WIKIPEDIA_SEARCH_RESOURCE: Final = "wikipedia-passages"
WIKIPEDIA_SEARCH_PROFILE: Final = "dpr-wikipedia-fts5-bm25@1"
WIKIPEDIA_QUERY_POLICY: Final = "snowball-english-regex-tokens@1"
WIKIPEDIA_RANKING: Final = "bm25-title5-text1@1"
WIKIPEDIA_CORPUS_ID: Final = "dpr-wikipedia-psgs-w100-20181220"
WIKIPEDIA_PASSAGE_FORMAT: Final = "wikipedia-title-text@1"
PASSAGES_PER_QUERY: Final = 5
PASSAGE_TEXT_CAP: Final = 1000
SEARCH_RESULTS_STATUS: Final = "corpus-search-results"
NO_PASSAGE_TEXT: Final = "No Wikipedia passage matched the query."
WIKIPEDIA_SEARCH_FUNCTION_NOTE: Final = (
    "Returns the five Wikipedia passages that best match the query."
)


def render_wikipedia_passage(title: str, text: str) -> str:
    for field, value in (("title", title), ("text", text)):
        if type(value) is not str or not value.strip():
            raise ValueError(f"a Wikipedia passage {field} is non-empty text")
    return f"Title: {' '.join(title.split())} Text: {' '.join(text.split())}"


def render_search_results(passages: Sequence[Mapping[str, JsonValue]]) -> str:
    if not passages:
        return NO_PASSAGE_TEXT
    blocks = []
    for index, passage in enumerate(passages, start=1):
        text = passage.get("text")
        if type(text) is not str:
            raise ValueError("a corpus_search passage carries its text")
        blocks.append(f"[{index}] {text}")
    return "\n\n".join(blocks)


__all__ = [
    "NO_PASSAGE_TEXT",
    "PASSAGES_PER_QUERY",
    "PASSAGE_TEXT_CAP",
    "SEARCH_RESULTS_STATUS",
    "WIKIPEDIA_CORPUS_ID",
    "WIKIPEDIA_PASSAGE_FORMAT",
    "WIKIPEDIA_QUERY_POLICY",
    "WIKIPEDIA_RANKING",
    "WIKIPEDIA_SEARCH_FUNCTION_NOTE",
    "WIKIPEDIA_SEARCH_PROFILE",
    "WIKIPEDIA_SEARCH_RESOURCE",
    "WIKIPEDIA_SEARCH_TOOL_NAME",
    "render_search_results",
    "render_wikipedia_passage",
]
