from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock

from skillev.contracts.identity import utc_timestamp

from .event_log import EventEnvelope, EventType
from .live_event_log import LiveAttemptEventLog


def _utc_now() -> str:
    return utc_timestamp(datetime.now(UTC))


@dataclass(slots=True)
class RuntimeEventEmitter:
    log: LiveAttemptEventLog
    producer_id: str
    clock: Callable[[], str] = _utc_now
    _next_sequence: int = field(default=1, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.producer_id) is not str or not self.producer_id.strip():
            raise ValueError("producer_id must be non-empty text")
        self._next_sequence = self.log.next_producer_sequence(self.producer_id)

    def emit(self, event_type: EventType, payload: object) -> EventEnvelope:
        with self._lock:
            event = self._prepare_locked(event_type, payload, offset=0)
            self._publish_locked(event)
        return event

    def prepare(self, event_type: EventType, payload: object) -> EventEnvelope:
        with self._lock:
            return self._prepare_locked(event_type, payload, offset=0)

    def prepare_many(
        self,
        events: tuple[tuple[EventType, object], ...],
    ) -> tuple[EventEnvelope, ...]:
        if not events:
            raise ValueError("event preparation requires at least one event")
        with self._lock:
            return tuple(
                self._prepare_locked(event_type, payload, offset=offset)
                for offset, (event_type, payload) in enumerate(events)
            )

    def publish_prepared(self, event: EventEnvelope) -> EventEnvelope:
        with self._lock:
            self._publish_locked(event)
        return event

    def reconcile_prepared(self, events: tuple[EventEnvelope, ...]) -> None:
        if not events:
            raise ValueError("event reconciliation requires prepared events")
        with self._lock:
            for event in events:
                if event.producer_id != self.producer_id:
                    raise ValueError("prepared event belongs to another producer")
                if event.producer_seq < self._next_sequence:
                    self.log.append_idempotent(event)
                    continue
                self._publish_locked(event)

    def _prepare_locked(
        self,
        event_type: EventType,
        payload: object,
        *,
        offset: int,
    ) -> EventEnvelope:
        return EventEnvelope.create(
            event_type=event_type,
            run_id=self.log.run_id,
            attempt_id=self.log.attempt_id,
            producer_id=self.producer_id,
            producer_seq=self._next_sequence + offset,
            occurred_at=self.clock(),
            payload=payload,
        )

    def _publish_locked(self, event: EventEnvelope) -> None:
        if event.producer_id != self.producer_id:
            raise ValueError("prepared event belongs to another producer")
        if event.producer_seq != self._next_sequence:
            raise ValueError("prepared event is not the next producer event")
        self.log.append_idempotent(event)
        self._next_sequence += 1

    def child(self, producer_id: str) -> RuntimeEventEmitter:
        return RuntimeEventEmitter(
            log=self.log,
            producer_id=producer_id,
            clock=self.clock,
        )
