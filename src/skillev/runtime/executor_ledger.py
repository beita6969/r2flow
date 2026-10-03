from __future__ import annotations

import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from skillev.contracts import JsonValue, canonical_json
from skillev.contracts.answer_writer import AnswerWriterWindowOutcome

EXECUTOR_CALL_RECORD_FORMAT = "executor-call-record@1"
EXECUTOR_CALLS_FILE = "executor-calls.jsonl"


@dataclass(frozen=True, slots=True)
class ExecutorCallRecord:
    trajectory_id: str
    step_index: int
    skill_id: str
    skill_version: str
    skill_content_hash: str
    executor_identity: str
    memo_key: str
    input_sha256: str
    input_chars: int
    prompt_tokens: int
    output_tokens: int
    finish_reason: str
    cache_hit: bool
    physical_latency_ms: int
    reference_latency_ms: int
    endpoint: str
    output_sha256: str
    divergent_recompute: bool
    format: str = EXECUTOR_CALL_RECORD_FORMAT
    answer_writer_window: AnswerWriterWindowOutcome | None = None

    def to_value(self) -> dict[str, JsonValue]:
        value: dict[str, JsonValue] = dict(asdict(self))
        if self.answer_writer_window is None:
            del value["answer_writer_window"]
        else:
            value["answer_writer_window"] = self.answer_writer_window.to_value()
        return value


class JsonlExecutorCallSink:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, record: ExecutorCallRecord) -> None:
        line = canonical_json(record.to_value()) + "\n"
        with self._lock:
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, line.encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)


def logical_executor_tokens(records: tuple[ExecutorCallRecord, ...]) -> int:
    return sum(record.prompt_tokens + record.output_tokens for record in records)


__all__ = [
    "EXECUTOR_CALLS_FILE",
    "EXECUTOR_CALL_RECORD_FORMAT",
    "ExecutorCallRecord",
    "JsonlExecutorCallSink",
    "logical_executor_tokens",
]
