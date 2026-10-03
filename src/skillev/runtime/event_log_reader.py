from __future__ import annotations

import json
from pathlib import Path

from .event_log import EventEnvelope


def read_event_history(path: Path) -> tuple[EventEnvelope, ...]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    events: list[EventEnvelope] = []
    event_ids: set[str] = set()
    producer_sequences: dict[str, int] = {}
    for line_number, line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line:
            raise ValueError(f"event log contains an empty line at {line_number}")
        event = EventEnvelope.from_value(json.loads(line))
        if event.event_id in event_ids:
            raise ValueError(f"duplicate event ID at line {line_number}")
        expected = producer_sequences.get(event.producer_id, 0) + 1
        if event.producer_seq != expected:
            raise ValueError(f"non-contiguous producer sequence at line {line_number}")
        events.append(event)
        event_ids.add(event.event_id)
        producer_sequences[event.producer_id] = event.producer_seq
    return tuple(events)


__all__ = ["read_event_history"]
