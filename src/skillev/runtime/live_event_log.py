from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

from skillev.contracts import canonical_json

from .attempt_failures import EventAppendFailedError
from .event_log import EventEnvelope, EventType


class LiveAttemptEventLog:
    def __init__(self, path: Path, *, run_id: str, attempt_id: str) -> None:
        if type(run_id) is not str or not run_id.strip():
            raise ValueError("run_id must be non-empty text")
        if type(attempt_id) is not str or not attempt_id.strip():
            raise ValueError("attempt_id must be non-empty text")
        self.path = path.resolve()
        self.run_id = run_id
        self.attempt_id = attempt_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("xb"):
            pass
        self._lock = Lock()
        self._event_ids: set[str] = set()
        self._events_by_id: dict[str, EventEnvelope] = {}
        self._producer_sequences: dict[str, int] = {}

    @classmethod
    def resume(cls, path: Path, *, run_id: str, attempt_id: str) -> LiveAttemptEventLog:
        from .event_log_reader import read_event_history

        resolved = path.resolve()
        events = read_event_history(resolved)
        if any(event.run_id != run_id or event.attempt_id != attempt_id for event in events):
            raise ValueError("event history belongs to another live attempt")
        self = cls.__new__(cls)
        self.path = resolved
        self.run_id = run_id
        self.attempt_id = attempt_id
        self._lock = Lock()
        self._event_ids = {event.event_id for event in events}
        self._events_by_id = {event.event_id: event for event in events}
        self._producer_sequences = {}
        for event in events:
            self._producer_sequences[event.producer_id] = event.producer_seq
        return self

    def next_producer_sequence(self, producer_id: str) -> int:
        if type(producer_id) is not str or not producer_id.strip():
            raise ValueError("producer_id must be non-empty text")
        with self._lock:
            return self._producer_sequences.get(producer_id, 0) + 1

    def committed_event_count(self, event_type: EventType) -> int:
        with self._lock:
            return sum(event.event_type is event_type for event in self._events_by_id.values())

    def append(self, event: EventEnvelope) -> None:
        if event.run_id != self.run_id or event.attempt_id != self.attempt_id:
            raise ValueError("event belongs to another live attempt")
        with self._lock:
            self._append_locked(event)

    def append_idempotent(self, event: EventEnvelope) -> bool:
        with self._lock:
            existing = self._events_by_id.get(event.event_id)
            if existing is not None:
                if existing != event:
                    raise ValueError("event ID is bound to different content")
                return False
            self._append_locked(event)
            return True

    def _append_locked(self, event: EventEnvelope) -> None:
        if event.run_id != self.run_id or event.attempt_id != self.attempt_id:
            raise ValueError("event belongs to another live attempt")
        if event.event_id in self._event_ids:
            raise ValueError("event ID is already present")
        expected = self._producer_sequences.get(event.producer_id, 0) + 1
        if event.producer_seq != expected:
            raise ValueError(f"producer sequence {event.producer_seq} differs from {expected}")
        encoded = canonical_json(event.to_value()).encode("utf-8") + b"\n"
        try:
            with self.path.open("ab") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise EventAppendFailedError("live attempt event append failed") from error
        self._event_ids.add(event.event_id)
        self._events_by_id[event.event_id] = event
        self._producer_sequences[event.producer_id] = event.producer_seq


__all__ = ["LiveAttemptEventLog"]
