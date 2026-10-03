from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from skillev.contracts.canonical import JsonValue, normalize_json, stable_hash
from skillev.contracts.identity import event_id


class EventType(StrEnum):
    LIBRARY_INITIALIZED = "library_initialized"
    TRAINING_STEP_COMMITTED = "training_step_committed"
    METHOD_STATE_RECORDED = "method_state_recorded"
    PHASE_DETECTION_RECORDED = "phase_detection_recorded"
    FLOW_STEP_RECORDED = "flow_step_recorded"

    TERMINAL_REWARD_RECORDED = "terminal_reward_recorded"
    ROLLOUT_STARTED = "rollout_started"
    ROLLOUT_POLICY_PINNED = "rollout_policy_pinned"
    ROLLOUT_STEP_COMMITTED = "rollout_step_committed"
    ROLLOUT_REJECTED = "rollout_rejected"
    ROLLOUT_COMPLETED = "rollout_completed"
    BUDGET_RESERVED = "budget_reserved"
    BUDGET_SETTLED = "budget_settled"
    AGENT_TURN_STARTED = "agent_turn_started"
    AGENT_ACTION_PARSED = "agent_action_parsed"
    AGENT_STEP_RECORDED = "agent_step_recorded"
    AGENT_COMPLETED = "agent_completed"


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    event_id: str
    event_type: EventType
    run_id: str
    attempt_id: str
    producer_id: str
    producer_seq: int
    occurred_at: str
    payload: JsonValue
    payload_hash: str

    @classmethod
    def create(
        cls,
        *,
        event_type: EventType,
        run_id: str,
        attempt_id: str,
        producer_id: str,
        producer_seq: int,
        occurred_at: str,
        payload: object,
    ) -> EventEnvelope:
        normalized = normalize_json(payload)
        payload_hash = stable_hash(normalized)
        return cls(
            event_id=event_id(
                producer_id=producer_id,
                producer_seq=producer_seq,
                payload_hash=payload_hash,
            ),
            event_type=event_type,
            run_id=run_id,
            attempt_id=attempt_id,
            producer_id=producer_id,
            producer_seq=producer_seq,
            occurred_at=occurred_at,
            payload=normalized,
            payload_hash=payload_hash,
        )

    def __post_init__(self) -> None:
        if not all(
            (
                self.event_id,
                self.run_id,
                self.attempt_id,
                self.producer_id,
                self.occurred_at,
            )
        ):
            raise ValueError("Event identity fields cannot be empty")
        if type(self.producer_seq) is not int or self.producer_seq < 1:
            raise ValueError("Event producer sequence must be positive")
        if normalize_json(self.payload) != self.payload:
            raise ValueError("Event payload must be normalized JSON")
        if stable_hash(self.payload) != self.payload_hash:
            raise ValueError("Event payload does not match its recorded identity")
        expected_id = event_id(
            producer_id=self.producer_id,
            producer_seq=self.producer_seq,
            payload_hash=self.payload_hash,
        )
        if self.event_id != expected_id:
            raise ValueError("Event ID does not match its producer sequence and payload")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "attempt_id": self.attempt_id,
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "occurred_at": self.occurred_at,
            "payload": self.payload,
            "payload_hash": self.payload_hash,
            "producer_id": self.producer_id,
            "producer_seq": self.producer_seq,
            "run_id": self.run_id,
        }

    @classmethod
    def from_value(cls, value: object) -> EventEnvelope:
        if not isinstance(value, dict):
            raise TypeError("Event envelope must be a JSON object")
        normalized = normalize_json(value)
        if not isinstance(normalized, dict) or normalized != value:
            raise TypeError("Event envelope must already use exact JSON wire types")
        expected_fields = {
            "attempt_id",
            "event_id",
            "event_type",
            "occurred_at",
            "payload",
            "payload_hash",
            "producer_id",
            "producer_seq",
            "run_id",
        }
        if set(normalized) != expected_fields:
            raise ValueError("Event envelope has an invalid field set")
        text_fields = (
            "attempt_id",
            "event_id",
            "event_type",
            "occurred_at",
            "payload_hash",
            "producer_id",
            "run_id",
        )
        if any(not isinstance(normalized[field], str) for field in text_fields):
            raise TypeError("Event envelope identity fields must be text")
        if type(normalized["producer_seq"]) is not int:
            raise TypeError("Event envelope producer_seq must be an integer")
        return cls(
            event_id=cast(str, normalized["event_id"]),
            event_type=EventType(cast(str, normalized["event_type"])),
            run_id=cast(str, normalized["run_id"]),
            attempt_id=cast(str, normalized["attempt_id"]),
            producer_id=cast(str, normalized["producer_id"]),
            producer_seq=normalized["producer_seq"],
            occurred_at=cast(str, normalized["occurred_at"]),
            payload=normalize_json(normalized["payload"]),
            payload_hash=cast(str, normalized["payload_hash"]),
        )
