from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from skillev.contracts import JsonValue, normalize_json


class RolloutTraceStage(StrEnum):
    EPISODE_STARTED = "episode-started"
    PLANNER_REQUEST = "planner-request"
    PLANNER_RESULT = "planner-result"
    OPERATOR_SELECTED = "operator-selected"
    OPERATOR_REQUEST = "operator-request"
    OPERATOR_RESULT = "operator-result"
    REASONING_REQUEST = "reasoning-request"
    REASONING_RESULT = "reasoning-result"
    ACTION_REQUEST = "action-request"
    ACTION_RESULT = "action-result"
    ACTION_PARSED = "action-parsed"
    ACTION_ADMITTED = "action-admitted"
    TOOL_REQUEST = "tool-request"
    TOOL_RESULT = "tool-result"
    ANSWER_WRITER_RESULT = "answer-writer-result"
    STATE_BEFORE = "state-before"
    STATE_DELTA = "state-delta"
    STATE_AFTER = "state-after"
    NEXT_PLANNER_CONTEXT_BUILT = "next-planner-context-built"
    ENVIRONMENT_RESULT = "environment-result"
    TERMINAL_REQUEST = "terminal-request"
    TERMINAL_RESULT = "terminal-result"
    REWARD_PROJECTION_INPUT = "reward-projection-input"
    REWARD_PROJECTION_RESULT = "reward-projection-result"
    TRAINING_SAMPLE_BUILT = "training-sample-built"
    TEACHER_FORCED_LOGPROB_COMPUTED = "teacher-forced-logprob-computed"
    TTB_EDGE_BUILT = "ttb-edge-built"
    GRADIENT_PREPARED = "gradient-prepared"
    OPTIMIZER_STEP_APPLIED = "optimizer-step-applied"
    PROJECTION_COMMITTED = "projection-committed"
    CHECKPOINT_PUBLISHED = "checkpoint-published"
    EPISODE_FINISHED = "episode-finished"
    EPISODE_FAILED = "episode-failed"


@dataclass(frozen=True, slots=True)
class RolloutTraceEvent:
    trajectory_id: str
    task_id: str
    step_index: int | None
    stage: RolloutTraceStage
    public_payload: dict[str, JsonValue]
    run_id: str | None = None
    batch_id: str | None = None
    optimizer_step: int | None = None
    event_sequence: int | None = None
    condition_id: str | None = None
    policy_snapshot_id: str | None = None
    library_version: str | None = None
    operator_id: str | None = None
    tool_call_id: str | None = None
    monotonic_time: float = dataclass_field(default_factory=time.perf_counter, compare=False)

    def __post_init__(self) -> None:
        if not self.trajectory_id.strip() or not self.task_id.strip():
            raise ValueError("trace event identity must be non-empty")
        if self.step_index is not None and self.step_index < 1:
            raise ValueError("trace event step must be positive or null")
        normalized = normalize_json(self.public_payload)
        if not isinstance(normalized, dict):
            raise ValueError("trace payload must be a JSON object")
        object.__setattr__(self, "public_payload", normalized)
        for field in (
            "run_id",
            "batch_id",
            "condition_id",
            "policy_snapshot_id",
            "library_version",
            "operator_id",
            "tool_call_id",
        ):
            value = getattr(self, field)
            if value is not None and (type(value) is not str or not value.strip()):
                raise ValueError(f"trace {field} must be non-empty text or null")
        for field in ("optimizer_step", "event_sequence"):
            value = getattr(self, field)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"trace {field} must be non-negative or null")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "public_payload": self.public_payload,
            "stage": self.stage.value,
            "step_index": self.step_index,
            "task_id": self.task_id,
            "trajectory_id": self.trajectory_id,
            "run_id": self.run_id,
            "batch_id": self.batch_id,
            "optimizer_step": self.optimizer_step,
            "event_sequence": self.event_sequence,
            "condition_id": self.condition_id,
            "policy_snapshot_id": self.policy_snapshot_id,
            "library_version": self.library_version,
            "operator_id": self.operator_id,
            "tool_call_id": self.tool_call_id,
            "monotonic_time": self.monotonic_time,
        }


class RolloutTraceSink(Protocol):
    async def record(self, event: RolloutTraceEvent) -> None: ...


@dataclass(frozen=True, slots=True)
class NullRolloutTraceSink:
    async def record(self, event: RolloutTraceEvent) -> None:
        del event


@dataclass(frozen=True, slots=True)
class RolloutTracePolicy:
    enabled: bool
    maximum_trajectories: int
    allowed_task_ids: frozenset[str] | None
    private_output_path: Path

    def __post_init__(self) -> None:
        if self.maximum_trajectories < 0:
            raise ValueError("trace trajectory limit cannot be negative")
        if not self.private_output_path.is_absolute():
            raise ValueError("private trace output must be absolute")


@dataclass(slots=True)
class JsonlRolloutTraceSink:
    path: Path
    queue: asyncio.Queue[RolloutTraceEvent | None]
    writer: asyncio.Task[None] | None = None

    @classmethod
    def create(cls, path: Path) -> JsonlRolloutTraceSink:
        if not path.is_absolute():
            raise ValueError("private trace path must be absolute")
        return cls(path, asyncio.Queue())

    async def __aenter__(self) -> JsonlRolloutTraceSink:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = asyncio.create_task(self._write_loop())
        return self

    async def record(self, event: RolloutTraceEvent) -> None:
        if self.writer is None:
            raise RuntimeError("trace sink must be entered before recording")
        await self.queue.put(event)

    async def __aexit__(self, *_args: object) -> None:
        await self.queue.put(None)
        if self.writer is None:
            raise RuntimeError("trace writer was not started")
        await self.writer

    async def _write_loop(self) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            while True:
                event = await self.queue.get()
                if event is None:
                    stream.flush()
                    return
                stream.write(
                    json.dumps(event.to_value(), ensure_ascii=False, separators=(",", ":")) + "\n"
                )


FORBIDDEN_KEYS = frozenset(
    {
        "accepted_answers",
        "rubrics",
        "reference_response",
        "tests",
        "gold_patch",
        "golden_workbook",
        "oracle",
    }
)


def reject_private_keys(value: JsonValue, *, location: str = "trace") -> None:
    if isinstance(value, dict):
        forbidden = set(value).intersection(FORBIDDEN_KEYS)
        if forbidden:
            raise ValueError(f"{location} contains private keys: {sorted(forbidden)!r}")
        for key, child in value.items():
            reject_private_keys(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_private_keys(child, location=f"{location}[{index}]")


__all__ = [
    "FORBIDDEN_KEYS",
    "JsonlRolloutTraceSink",
    "NullRolloutTraceSink",
    "RolloutTraceEvent",
    "RolloutTracePolicy",
    "RolloutTraceSink",
    "RolloutTraceStage",
    "reject_private_keys",
]
