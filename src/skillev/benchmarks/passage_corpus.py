from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from skillev.contracts import JsonValue
from skillev.contracts.canonical import stable_hash
from skillev.runtime import ActionKind, BudgetVector, StructuredAction
from skillev.runtime.execution import EnvironmentObservation, RolloutEnvironmentSession

PASSAGE_TOOL_RESOURCE = "hotpotqa-documents"
PASSAGE_TOOL_NAME = "open_passage"
MAX_DOCUMENTS = 16
_HEAD = "Based on the following passages, answer the question.\n\n"
_QUESTION = "\n\nQuestion: "
_EVIDENCE = "\n\nEvidence:\n"
_PARAGRAPH = re.compile(r"\[([^\]\n]+)\] (.*)", re.S)


@dataclass(frozen=True, slots=True)
class PassageDocument:
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class PassageCorpus:
    documents: tuple[PassageDocument, ...]

    def __post_init__(self) -> None:
        titles = self.titles()
        if not 1 <= len(titles) <= MAX_DOCUMENTS or len(set(titles)) != len(titles):
            raise ValueError("a passage corpus has 1..16 uniquely titled documents")

    def titles(self) -> tuple[str, ...]:
        return tuple(document.title for document in self.documents)

    def open(self, title: str) -> PassageDocument:
        for document in self.documents:
            if document.title == title:
                return document
        raise KeyError(title)

    def corpus_hash(self) -> str:
        return stable_hash(
            [
                [document.title, hashlib.sha256(document.text.encode("utf-8")).hexdigest()]
                for document in self.documents
            ]
        )


def hotpot_open_passage_projection(baked_query: str) -> tuple[str, PassageCorpus]:
    if (
        type(baked_query) is not str
        or not baked_query.startswith(_HEAD)
        or baked_query.count(_QUESTION) != 1
        or baked_query.count(_EVIDENCE) != 1
    ):
        raise ValueError("HotpotQA query does not have the verified baked layout")
    embedded_part, _, rest = baked_query.partition(_QUESTION)
    question, _, evidence = rest.partition(_EVIDENCE)
    if not question.strip() or "\n" in question:
        raise ValueError("HotpotQA question must be one non-empty line")
    paragraphs = evidence.split("\n\n")
    embedded = embedded_part[len(_HEAD) :].split("\n\n")
    if embedded != ["[" + paragraph + "]" for paragraph in paragraphs]:
        raise ValueError("embedded passages differ from the evidence list")
    documents: list[PassageDocument] = []
    for paragraph in paragraphs:
        match = _PARAGRAPH.fullmatch(paragraph)
        if match is None or not match[2].strip():
            raise ValueError("evidence paragraph is not '[title] text'")
        documents.append(PassageDocument(match[1], match[2]))
    return question, PassageCorpus(tuple(documents))


def distractor_context_query(question: str, corpus: PassageCorpus) -> str:
    return (
        question
        + f"\n\nDocuments ({len(corpus.documents)}):\n"
        + "\n\n".join(f"[{document.title}] {document.text}" for document in corpus.documents)
    )


def passage_observation(document: PassageDocument) -> EnvironmentObservation:
    return EnvironmentObservation(
        {"status": "passage-opened", "title": document.title, "text": document.text},
        "success",
        budget_usage=BudgetVector(tool_calls=1),
    )


class PassageCorpusEnvironment:
    def __init__(self, delegate: RolloutEnvironmentSession, corpus: PassageCorpus) -> None:
        self.delegate = delegate
        self.corpus = corpus
        self._last_step = 0

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
            and action.resource_id == PASSAGE_TOOL_RESOURCE
            and action.name == PASSAGE_TOOL_NAME
        ):
            if step_index <= self._last_step:
                raise RuntimeError("passage environment requires an increasing step")
            self._last_step = step_index
            arguments = action.arguments
            title = arguments.get("title") if isinstance(arguments, dict) else None
            if not isinstance(title, str) or title not in self.corpus.titles():
                return EnvironmentObservation({"error": "passage_not_available"}, "schema_invalid")
            return passage_observation(self.corpus.open(title))
        return await self.delegate.execute(action, step_index=step_index)


__all__ = [
    "PASSAGE_TOOL_NAME",
    "PASSAGE_TOOL_RESOURCE",
    "PassageCorpus",
    "PassageCorpusEnvironment",
    "PassageDocument",
    "distractor_context_query",
    "hotpot_open_passage_projection",
    "passage_observation",
]
