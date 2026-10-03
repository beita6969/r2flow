from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
import unicodedata
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

from skillev.contracts import JsonValue
from skillev.contracts.answer_writer import (
    ANSWER_WRITER_PROMPT_VERSION,
    ANSWER_WRITER_RESERVATION,
    ANSWER_WRITER_WINDOW,
    EXECUTOR_ANSWER,
    AnswerWriterPrompt,
    AnswerWriterWindowOutcome,
)
from skillev.contracts.canonical import canonical_json, stable_hash

from .budget_ledger import BudgetLedger
from .contracts import BudgetReservation, BudgetSettlement, BudgetVector
from .emitter import RuntimeEventEmitter
from .event_log import EventType
from .execution import EnvironmentObservation
from .executor_ledger import ExecutorCallRecord
from .request_journal import shared_database
from .skill_md import parse_skill_md
from .skills import SkillMetadata

FROZEN_EXECUTOR_VERSION = "frozen-base-executor@1"
EXECUTOR_PROMPT_VERSION = "skill-md-executor-prompt@1"
EXECUTOR_OUTPUT_VERSION = "executor-output-canonical@1"
EXECUTOR_MEMO_FILE = "executor-memo.sqlite"
EXECUTOR_PREAMBLE = (
    "You execute one skill. Follow the skill instructions below on the user input. "
    "Reply with the result only."
)
TRANSPORT_ATTEMPTS = 3


class FrozenExecutorError(RuntimeError):
    pass


class ExecutorInputTooLongError(FrozenExecutorError):
    pass


@dataclass(frozen=True, slots=True)
class FrozenExecutorSpec:
    backbone_id: str
    tokenizer_id: str
    serving_profile_id: str
    max_input_tokens: int
    max_output_tokens: int
    seed: int
    temperature: float = 0.0
    top_k: int = 1
    top_p: float = 1.0
    enable_thinking: bool = False
    prompt_version: str = EXECUTOR_PROMPT_VERSION
    output_version: str = EXECUTOR_OUTPUT_VERSION
    format: str = FROZEN_EXECUTOR_VERSION

    def __post_init__(self) -> None:
        for name in ("backbone_id", "tokenizer_id", "serving_profile_id"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"executor {name} must be non-empty text")
        for name in ("max_input_tokens", "max_output_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"executor {name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("executor seed must be a non-negative integer")
        if (self.temperature, self.top_k, self.top_p, self.enable_thinking) != (
            0.0,
            1,
            1.0,
            False,
        ):
            raise ValueError("the frozen executor decodes greedily without thinking")
        if (self.prompt_version, self.output_version, self.format) != (
            EXECUTOR_PROMPT_VERSION,
            EXECUTOR_OUTPUT_VERSION,
            FROZEN_EXECUTOR_VERSION,
        ):
            raise ValueError("unsupported frozen executor version")

    def to_value(self) -> dict[str, JsonValue]:
        return dict(asdict(self))

    @classmethod
    def from_value(cls, value: object) -> FrozenExecutorSpec:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("frozen executor spec has an incompatible field set")
        fields = dict(value)
        for name in ("temperature", "top_p"):
            if type(fields[name]) in (int, float):
                fields[name] = float(fields[name])
        spec = cls(**fields)
        if spec.to_value() != value:
            raise ValueError("frozen executor spec is not canonical")
        return spec

    def identity(self) -> str:
        return stable_hash(self.to_value())


def executor_messages(skill_md_text: str, input_text: str) -> list[dict[str, str]]:
    skill = parse_skill_md(skill_md_text)
    if skill.is_empty_slot:
        return [{"role": "user", "content": input_text}]
    return [
        {"role": "system", "content": EXECUTOR_PREAMBLE + "\n\n" + skill.body},
        {"role": "user", "content": input_text},
    ]


def executor_memo_key(spec: FrozenExecutorSpec, meta: SkillMetadata, input_text: str) -> str:
    return stable_hash(
        {
            "executor": spec.identity(),
            "skill_id": meta.skill_id,
            "version": meta.version,
            "content_hash": meta.content_hash,
            "input": input_text,
        }
    )


def canonical_executor_output(text: str) -> str:
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    return "\n".join(line.rstrip() for line in text.split("\n")).rstrip()


@dataclass(frozen=True, slots=True)
class ExecutorOutput:
    token_ids: tuple[int, ...]
    text: str
    finish: str
    prompt_tokens: int
    latency_ms: int
    endpoint: str


@dataclass(frozen=True, slots=True)
class WrittenAnswer:
    output: ExecutorOutput
    window: AnswerWriterWindowOutcome


def executor_observation(
    meta: SkillMetadata, library_version: str, output: ExecutorOutput
) -> EnvironmentObservation:
    value: dict[str, JsonValue] = {
        "status": "skill-executed",
        "skill_id": meta.skill_id,
        "version": meta.version,
        "library_version": library_version,
        "output": output.text,
        "output_truncated": output.finish == "length",
    }
    status = "success" if output.text.strip() else "tool_error"
    if status == "tool_error":
        value["error"] = "empty_executor_output"
    return EnvironmentObservation(
        value, status, (meta.skill_id,), budget_usage=BudgetVector(tool_calls=1)
    )


class ExecutorMemoStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.touch(mode=0o600, exist_ok=True)
        path.chmod(0o600)
        self._shared = shared_database(path, isolation_level=None)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS executor_memo ("
                "memo_key TEXT PRIMARY KEY, executor_identity TEXT NOT NULL, "
                "token_ids TEXT NOT NULL, text TEXT NOT NULL, finish TEXT NOT NULL, "
                "prompt_tokens INTEGER NOT NULL, latency_ms INTEGER NOT NULL, "
                "endpoint TEXT NOT NULL)"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with self._shared.transaction() as db:
            yield db

    def get(self, key: str) -> ExecutorOutput | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT token_ids, text, finish, prompt_tokens, latency_ms, endpoint "
                "FROM executor_memo WHERE memo_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return ExecutorOutput(tuple(json.loads(row[0])), row[1], row[2], row[3], row[4], row[5])

    def put_first(self, key: str, spec_identity: str, output: ExecutorOutput) -> ExecutorOutput:
        with self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO executor_memo VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    spec_identity,
                    json.dumps(list(output.token_ids)),
                    output.text,
                    output.finish,
                    output.prompt_tokens,
                    output.latency_ms,
                    output.endpoint,
                ),
            )
        winner = self.get(key)
        if winner is None:
            raise FrozenExecutorError("executor memo lost a committed row")
        return winner


class ExecutorTokenizer(Protocol):
    def encode_executor_prompt(self, messages: list[dict[str, str]]) -> list[int]: ...


class ExecutorTransport(Protocol):
    def __call__(
        self, input_ids: tuple[int, ...], spec: FrozenExecutorSpec, /, *, regex: str | None = None
    ) -> Awaitable[ExecutorOutput]: ...


ExecutorRecordSink = Callable[[ExecutorCallRecord], None]
_Outcome = tuple[ExecutorOutput, bool, int, bool]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def answer_writer_memo_key(
    spec: FrozenExecutorSpec, prompt: AnswerWriterPrompt, max_output_tokens: int, regex: str | None
) -> str:
    return stable_hash(
        {
            "executor": spec.identity(),
            "writer": EXECUTOR_ANSWER,
            "prompt": prompt.to_value(),
            "window": ANSWER_WRITER_WINDOW,
            "max_output_tokens": max_output_tokens,
            "regex": regex,
        }
    )


class FrozenSkillExecutor:
    def __init__(
        self,
        *,
        spec: FrozenExecutorSpec,
        tokenizer: ExecutorTokenizer,
        memo: ExecutorMemoStore,
        transport: ExecutorTransport,
        ledger: BudgetLedger,
        record_sink: ExecutorRecordSink,
        emitter: RuntimeEventEmitter | None = None,
    ) -> None:
        self.spec = spec
        self.tokenizer = tokenizer
        self.memo = memo
        self.transport = transport
        self.ledger = ledger
        self.record_sink = record_sink
        self.emitter = emitter
        self._inflight: dict[str, asyncio.Future[_Outcome]] = {}

    async def run(
        self,
        *,
        trajectory_id: str,
        step_index: int,
        meta: SkillMetadata,
        skill_md_text: str,
        input_text: str,
        library_version: str,
    ) -> EnvironmentObservation:
        key = executor_memo_key(self.spec, meta, input_text)

        def encode() -> tuple[int, ...]:
            return tuple(
                self.tokenizer.encode_executor_prompt(executor_messages(skill_md_text, input_text))
            )

        outcome = await self._single_flight(
            key,
            lambda: self._compute(
                key,
                trajectory_id,
                step_index,
                encode,
                call_spec=self.spec,
                regex=None,
                suffix="executor",
            ),
        )
        output, cache_hit, physical_ms, divergent = outcome
        self.record_sink(
            ExecutorCallRecord(
                trajectory_id=trajectory_id,
                step_index=step_index,
                skill_id=meta.skill_id,
                skill_version=meta.version,
                skill_content_hash=meta.content_hash,
                executor_identity=self.spec.identity(),
                memo_key=key,
                input_sha256=_sha256(input_text),
                input_chars=len(input_text),
                prompt_tokens=output.prompt_tokens,
                output_tokens=len(output.token_ids),
                finish_reason=output.finish,
                cache_hit=cache_hit,
                physical_latency_ms=physical_ms,
                reference_latency_ms=output.latency_ms,
                endpoint=output.endpoint,
                output_sha256=_sha256(output.text),
                divergent_recompute=divergent,
            )
        )
        return executor_observation(meta, library_version, output)

    async def write_answer(
        self,
        *,
        trajectory_id: str,
        step_index: int,
        prompt: AnswerWriterPrompt,
        max_output_tokens: int,
        regex: str | None = None,
    ) -> WrittenAnswer:
        if type(max_output_tokens) is not int or not (
            1 <= max_output_tokens <= self.spec.max_output_tokens
        ):
            raise ValueError("the answer writer output cap must lie in [1, max_output_tokens]")
        if regex is not None and (type(regex) is not str or not regex):
            raise ValueError("the answer writer regex must be non-empty text")
        key = answer_writer_memo_key(self.spec, prompt, max_output_tokens, regex)
        input_ids, window = self._writer_window(prompt)
        outcome = await self._single_flight(
            key,
            lambda: self._compute(
                key,
                trajectory_id,
                step_index,
                lambda: input_ids,
                call_spec=replace(self.spec, max_output_tokens=max_output_tokens),
                regex=regex,
                suffix=ANSWER_WRITER_RESERVATION,
            ),
        )
        output, cache_hit, physical_ms, divergent = outcome
        messages = prompt.messages()
        self.record_sink(
            ExecutorCallRecord(
                trajectory_id=trajectory_id,
                step_index=step_index,
                skill_id=EXECUTOR_ANSWER,
                skill_version=ANSWER_WRITER_PROMPT_VERSION,
                skill_content_hash=stable_hash(prompt.system),
                executor_identity=self.spec.identity(),
                memo_key=key,
                input_sha256=_sha256(canonical_json(messages)),
                input_chars=len(messages[1]["content"]),
                prompt_tokens=output.prompt_tokens,
                output_tokens=len(output.token_ids),
                finish_reason=output.finish,
                cache_hit=cache_hit,
                physical_latency_ms=physical_ms,
                reference_latency_ms=output.latency_ms,
                endpoint=output.endpoint,
                output_sha256=_sha256(output.text),
                divergent_recompute=divergent,
                answer_writer_window=window,
            )
        )
        return WrittenAnswer(output, window)

    def _writer_window(
        self, prompt: AnswerWriterPrompt
    ) -> tuple[tuple[int, ...], AnswerWriterWindowOutcome]:
        cap = self.spec.max_input_tokens

        def encode(omitted: int) -> tuple[int, ...]:
            return tuple(self.tokenizer.encode_executor_prompt(prompt.messages(omitted=omitted)))

        full = encode(0)
        if not full:
            raise FrozenExecutorError("the answer writer prompt encodes to no tokens")
        ids, omitted, truncated = full, 0, False
        if len(full) > cap:
            fitting: tuple[int, ...] | None = None
            low, high = 1, len(prompt.calls)
            while low <= high:
                middle = (low + high) // 2
                candidate = encode(middle)
                if len(candidate) <= cap:
                    fitting, omitted, high = candidate, middle, middle - 1
                else:
                    low = middle + 1
            if fitting is not None:
                ids = fitting
            else:
                alone = encode(len(prompt.calls))
                head = cap // 2
                ids = alone[:head] + alone[len(alone) - (cap - head) :]
                omitted, truncated = len(prompt.calls), True
        return ids, AnswerWriterWindowOutcome(
            listed_calls=len(prompt.calls),
            omitted_calls=omitted,
            truncated=truncated,
            prompt_tokens=len(full),
            sent_tokens=len(ids),
            sent_sha256=_sha256(canonical_json(list(ids))),
        )

    async def _single_flight(
        self, key: str, compute: Callable[[], Awaitable[_Outcome]]
    ) -> _Outcome:
        pending = self._inflight.get(key)
        if pending is not None:
            output = (await asyncio.shield(pending))[0]
            return (output, True, 0, False)
        future: asyncio.Future[_Outcome] = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            outcome = await compute()
            future.set_result(outcome)
        except asyncio.CancelledError:
            future.cancel()
            raise
        except Exception as error:
            future.set_exception(error)
            future.exception()
            raise
        finally:
            del self._inflight[key]
        return outcome

    async def _compute(
        self,
        key: str,
        trajectory_id: str,
        step_index: int,
        encode: Callable[[], tuple[int, ...]],
        *,
        call_spec: FrozenExecutorSpec,
        regex: str | None,
        suffix: str,
    ) -> _Outcome:
        stored = self.memo.get(key)
        if stored is not None:
            return stored, True, 0, False
        input_ids = encode()
        if not input_ids or len(input_ids) > self.spec.max_input_tokens:
            raise ExecutorInputTooLongError("executor prompt exceeds max_input_tokens")
        reservation = BudgetReservation(
            reservation_id=f"{trajectory_id}:{step_index}:{suffix}",
            run_id=self.ledger.run_id,
            attempt_id=self.ledger.attempt_id,
            invocation_id=f"{trajectory_id}:{step_index}:{suffix}",
            maximum=BudgetVector(
                input_tokens=len(input_ids),
                output_tokens=call_spec.max_output_tokens,
                model_calls=1,
            ),
        )
        self.ledger.reserve(reservation)
        self._emit(EventType.BUDGET_RESERVED, reservation.reservation_id, reservation.maximum)
        started = time.perf_counter()
        local: ExecutorOutput | None = None
        try:
            for attempt in range(TRANSPORT_ATTEMPTS):
                try:
                    local = await (
                        self.transport(input_ids, call_spec)
                        if regex is None
                        else self.transport(input_ids, call_spec, regex=regex)
                    )
                    break
                except FrozenExecutorError:
                    raise
                except Exception as error:
                    if attempt + 1 == TRANSPORT_ATTEMPTS:
                        raise FrozenExecutorError("frozen executor transport failed") from error
            assert local is not None
            if local.prompt_tokens != len(input_ids):
                raise FrozenExecutorError("executor prompt usage differs from the input IDs")
            if len(local.token_ids) > call_spec.max_output_tokens:
                raise FrozenExecutorError("executor output exceeds max_output_tokens")
            actual = BudgetVector(
                input_tokens=local.prompt_tokens,
                output_tokens=len(local.token_ids),
                model_calls=1,
            )
        except BaseException:
            self._settle(reservation.reservation_id, reservation.maximum)
            raise
        self._settle(reservation.reservation_id, actual)
        physical_ms = max(1, round((time.perf_counter() - started) * 1000))
        winner = self.memo.put_first(key, self.spec.identity(), local)
        divergent = (winner.token_ids, winner.text) != (local.token_ids, local.text)
        return winner, False, physical_ms, divergent

    def _settle(self, reservation_id: str, actual: BudgetVector) -> None:
        self.ledger.settle(BudgetSettlement(reservation_id=reservation_id, actual=actual))
        self._emit(EventType.BUDGET_SETTLED, reservation_id, actual)

    def _emit(self, event: EventType, reservation_id: str, vector: BudgetVector) -> None:
        if self.emitter is None:
            return
        key = "maximum" if event is EventType.BUDGET_RESERVED else "actual"
        self.emitter.emit(event, {key: vector.to_value(), "reservation_id": reservation_id})


__all__ = [
    "EXECUTOR_MEMO_FILE",
    "EXECUTOR_OUTPUT_VERSION",
    "EXECUTOR_PREAMBLE",
    "EXECUTOR_PROMPT_VERSION",
    "FROZEN_EXECUTOR_VERSION",
    "ExecutorInputTooLongError",
    "ExecutorMemoStore",
    "ExecutorOutput",
    "FrozenExecutorError",
    "FrozenExecutorSpec",
    "FrozenSkillExecutor",
    "WrittenAnswer",
    "answer_writer_memo_key",
    "canonical_executor_output",
    "executor_memo_key",
    "executor_messages",
    "executor_observation",
]
