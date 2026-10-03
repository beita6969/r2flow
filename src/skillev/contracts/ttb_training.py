from __future__ import annotations

from dataclasses import dataclass

from .canonical import JsonValue
from .ttb_common import (
    require_non_empty_text,
)


def _require_non_empty(value: str, *, field_name: str) -> None:
    require_non_empty_text(value, field=field_name)


@dataclass(frozen=True, slots=True)
class EdgeScoreContext:
    batch_id: str
    policy_snapshot_id: str
    library_version: str
    action_token_count: int
    action_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in ("batch_id", "policy_snapshot_id", "library_version"):
            _require_non_empty(getattr(self, name), field_name=f"edge context {name}")
        if type(self.action_token_count) is not int or self.action_token_count < 1:
            raise ValueError("edge action token count must be positive")
        if not isinstance(self.action_token_ids, tuple) or any(
            type(token) is not int or token < 0 for token in self.action_token_ids
        ):
            raise ValueError("edge action span must be an exact token tuple")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "batch_id": self.batch_id,
            "policy_snapshot_id": self.policy_snapshot_id,
            "library_version": self.library_version,
            "action_token_count": self.action_token_count,
            "action_token_ids": list(self.action_token_ids),
        }

    @classmethod
    def from_value(cls, value: object) -> EdgeScoreContext:
        if not isinstance(value, dict):
            raise TypeError("edge scoring context must be an object")
        return cls(**{**value, "action_token_ids": tuple(value.get("action_token_ids", ()))})
