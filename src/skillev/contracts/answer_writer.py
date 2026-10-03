from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from .canonical import JsonValue
from .integer_answer import INTEGER_ANSWER_PATTERN

EXECUTOR_ANSWER: Final = "executor-answer@1"
COMPLETION_WRITERS: Final = frozenset({EXECUTOR_ANSWER})
ANSWER_WRITER_PROMPT_VERSION: Final = "answer-writer-prompt@1"
ANSWER_WRITER_WINDOW: Final = "answer-writer-window@1"
ANSWER_WRITER_RESERVATION: Final = "answer-writer"
INTEGER_ANSWER_REGEX: Final = INTEGER_ANSWER_PATTERN.removeprefix("^").removesuffix("$")
ACCEPTED_STATUS: Final = "accepted_for_evaluation"
WRITER_REASONING_SCORINGS: Final = frozenset({"sampled-reasoning-conditioned@1"})


def declared_completion_writer(spec_or_contract: Any) -> str | None:
    raw = getattr(spec_or_contract, "action_contract_json", spec_or_contract)
    contract = json.loads(raw) if isinstance(raw, str) else raw
    surface = contract.get("surface") if isinstance(contract, Mapping) else None
    writer = surface.get("completion_writer") if isinstance(surface, Mapping) else None
    if writer is not None and writer not in COMPLETION_WRITERS:
        raise ValueError("unsupported completion writer")
    return writer if isinstance(writer, str) else None


def accepted_observation(answer: str) -> dict[str, JsonValue]:
    return {"status": ACCEPTED_STATUS, "answer": answer}


def written_answer(value: object) -> str | None:
    if (
        isinstance(value, Mapping)
        and set(value) == {"status", "answer"}
        and value["status"] == ACCEPTED_STATUS
        and isinstance(value["answer"], str)
    ):
        return value["answer"]
    return None


@dataclass(frozen=True, slots=True)
class AnswerWriterPrompt:
    system: str
    task: str
    order: str
    calls: tuple[str, ...]
    notes: str

    def __post_init__(self) -> None:
        for name in ("system", "task", "order"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"answer writer {name} must be non-empty text")
        if type(self.notes) is not str:
            raise ValueError("answer writer notes must be text")
        if not isinstance(self.calls, tuple) or any(
            type(block) is not str or not block for block in self.calls
        ):
            raise ValueError("answer writer call blocks must be a tuple of text")

    def user_text(self, *, omitted: int = 0) -> str:
        if type(omitted) is not int or not 0 <= omitted <= len(self.calls):
            raise ValueError("omitted call blocks must be within the listed calls")
        kept = self.calls[omitted:]
        listing = "".join(kept) if kept or omitted else "None.\n"
        if omitted:
            noun = "call" if omitted == 1 else "calls"
            listing = f"({omitted} earlier {noun} omitted.)\n" + listing
        return (
            "### Task\n"
            + self.task.rstrip("\n")
            + "\n### Calls and results\n"
            + self.order
            + "\n"
            + listing
            + "### Supervisor notes\n"
            + (self.notes.strip() or "(empty)")
            + "\n"
        )

    def messages(self, *, omitted: int = 0) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user_text(omitted=omitted)},
        ]

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "version": ANSWER_WRITER_PROMPT_VERSION,
            "system": self.system,
            "task": self.task,
            "order": self.order,
            "calls": list(self.calls),
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class AnswerWriterWindowOutcome:
    listed_calls: int
    omitted_calls: int
    truncated: bool
    prompt_tokens: int
    sent_tokens: int
    sent_sha256: str

    def __post_init__(self) -> None:
        counts = (self.listed_calls, self.omitted_calls, self.prompt_tokens, self.sent_tokens)
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("answer writer window counts are non-negative integers")
        if type(self.truncated) is not bool:
            raise ValueError("the answer writer window truncation flag is a bool")
        if self.omitted_calls > self.listed_calls or (
            self.truncated and self.omitted_calls != self.listed_calls
        ):
            raise ValueError("the window omits listed call blocks only, all before a cut")
        if not 0 < self.sent_tokens <= self.prompt_tokens or (
            (self.sent_tokens == self.prompt_tokens)
            != (self.omitted_calls == 0 and not self.truncated)
        ):
            raise ValueError("the window sends the full prompt exactly when it cuts nothing")
        if (
            type(self.sent_sha256) is not str
            or len(self.sent_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.sent_sha256)
        ):
            raise ValueError("the sent prompt hash is a lowercase sha256 hex digest")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "version": ANSWER_WRITER_WINDOW,
            "listed_calls": self.listed_calls,
            "omitted_calls": self.omitted_calls,
            "truncated": self.truncated,
            "prompt_tokens": self.prompt_tokens,
            "sent_tokens": self.sent_tokens,
            "sent_sha256": self.sent_sha256,
        }


__all__ = [
    "ACCEPTED_STATUS",
    "ANSWER_WRITER_PROMPT_VERSION",
    "ANSWER_WRITER_RESERVATION",
    "ANSWER_WRITER_WINDOW",
    "COMPLETION_WRITERS",
    "EXECUTOR_ANSWER",
    "INTEGER_ANSWER_REGEX",
    "WRITER_REASONING_SCORINGS",
    "AnswerWriterPrompt",
    "AnswerWriterWindowOutcome",
    "accepted_observation",
    "declared_completion_writer",
    "written_answer",
]
