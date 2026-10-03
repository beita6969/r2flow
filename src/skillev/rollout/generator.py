from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from skillev.contracts import JsonValue, TokenizerProtocol, normalize_json
from skillev.policy.interface import (
    AdapterRole,
    PolicyBackbone,
    PolicyGenerationRequest,
    RolloutPromptTokenizerProtocol,
)
from skillev.runtime import BudgetVector

from .types import GenerationPhase, PolicySnapshot

_BACKEND_ID = "hf-local"


def _validate_token_ids(value: object, *, field_name: str, non_empty: bool) -> None:
    if not isinstance(value, tuple):
        raise ValueError(f"{field_name} must be a tuple")
    if non_empty and not value:
        raise ValueError(f"{field_name} must not be empty")
    if any(type(token_id) is not int or token_id < 0 for token_id in value):
        raise ValueError(f"{field_name} must contain non-negative integer token ids")


def _require_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value


def _wire_text(value: object, *, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be text")
    return value


def _wire_token(value: object, *, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must contain integers")
    return value


class RolloutTokenizerProtocol(
    TokenizerProtocol,
    RolloutPromptTokenizerProtocol,
    Protocol,
):
    def decode(self, token_ids: tuple[int, ...]) -> str: ...

    def encode_rollout_prompt(self, text: str) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class RolloutGenerationRequest:
    phase: GenerationPhase
    input_ids: tuple[int, ...]
    max_new_tokens: int
    seed: int
    decoding_snapshot_id: str
    expected_policy_snapshot_id: str
    episode_id: str | None = None
    turn_index: int | None = None
    library_version: str | None = None
    sampling_constraint: str | None = None
    action_budget_tokens: int | None = None
    extra_stop_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self._validate_constraint()
        if isinstance(self.extra_stop_token_ids, list):
            object.__setattr__(self, "extra_stop_token_ids", tuple(self.extra_stop_token_ids))
        _validate_token_ids(
            self.extra_stop_token_ids, field_name="extra_stop_token_ids", non_empty=False
        )
        if self.extra_stop_token_ids and (
            self.phase is not GenerationPhase.REASONING
            or len(set(self.extra_stop_token_ids)) != len(self.extra_stop_token_ids)
        ):
            raise ValueError("extra stop tokens are distinct ids of a reasoning request")
        if (self.episode_id is None) != (self.turn_index is None):
            raise ValueError("episode and turn coordinates must be supplied together")
        if self.episode_id is not None:
            _require_text(self.episode_id, field_name="episode_id")
            if type(self.turn_index) is not int or self.turn_index < 1:
                raise ValueError("turn coordinates are one-based")
        if not isinstance(self.phase, GenerationPhase):
            raise ValueError("phase must be a GenerationPhase")
        _validate_token_ids(self.input_ids, field_name="input_ids", non_empty=True)
        if type(self.max_new_tokens) is not int or self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        _require_text(
            self.decoding_snapshot_id,
            field_name="decoding_snapshot_id",
        )
        _require_text(
            self.expected_policy_snapshot_id,
            field_name="expected_policy_snapshot_id",
        )

    def _validate_constraint(self) -> None:
        constrained = self.phase is GenerationPhase.ACTION
        if constrained != (self.sampling_constraint is not None):
            raise ValueError("a sampling constraint is required iff an event-boundary action")
        if (self.action_budget_tokens is None) != (self.sampling_constraint is None):
            raise ValueError("the action budget accompanies the sampling constraint")
        if self.sampling_constraint is None:
            return
        from skillev.policy.event_grammar import parse_event_grammar_key

        key = parse_event_grammar_key(self.sampling_constraint)
        if type(self.action_budget_tokens) is not int or key.budget != self.action_budget_tokens:
            raise ValueError("action budget differs from the grammar key budget")
        if self.max_new_tokens != self.action_budget_tokens + 1:
            raise ValueError("event-grammar requests need max_new_tokens == budget + 1")


@dataclass(frozen=True, slots=True)
class RolloutGenerationResult:
    content_token_ids: tuple[int, ...]
    stop_token_ids: tuple[int, ...]
    finish_reason: str
    policy_snapshot_id: str
    backend_id: str
    usage: BudgetVector = field(default_factory=BudgetVector)
    server_token_logq: tuple[float, ...] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        _validate_token_ids(
            self.content_token_ids,
            field_name="content_token_ids",
            non_empty=False,
        )
        if self.server_token_logq is not None and len(self.server_token_logq) != len(
            self.content_token_ids
        ) + len(self.stop_token_ids):
            raise ValueError("server_token_logq must cover every generated token")
        _validate_token_ids(
            self.stop_token_ids,
            field_name="stop_token_ids",
            non_empty=False,
        )
        _require_text(self.finish_reason, field_name="finish_reason")
        _require_text(self.policy_snapshot_id, field_name="policy_snapshot_id")
        _require_text(self.backend_id, field_name="backend_id")
        if not isinstance(self.usage, BudgetVector):
            raise ValueError("usage must be a BudgetVector")
        expected_output_tokens = len(self.content_token_ids) + len(self.stop_token_ids)
        if self.usage.output_tokens != expected_output_tokens:
            raise ValueError("usage.output_tokens does not match generated token IDs")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "backend_id": self.backend_id,
            "content_token_ids": list(self.content_token_ids),
            "finish_reason": self.finish_reason,
            "policy_snapshot_id": self.policy_snapshot_id,
            "stop_token_ids": list(self.stop_token_ids),
            "usage": normalize_json(self.usage.to_value()),
        }

    @classmethod
    def from_value(cls, value: object) -> RolloutGenerationResult:
        if not isinstance(value, dict):
            raise ValueError("rollout generation result must be a JSON object")
        normalized = normalize_json(value)
        if not isinstance(normalized, dict) or normalized != value:
            raise ValueError("rollout generation result must be a JSON object")
        expected = {
            "backend_id",
            "content_token_ids",
            "finish_reason",
            "policy_snapshot_id",
            "stop_token_ids",
            "usage",
        }
        if set(normalized) != expected:
            raise ValueError("rollout generation result has an incompatible field set")
        content = normalized["content_token_ids"]
        stop = normalized["stop_token_ids"]
        if type(content) is not list or type(stop) is not list:
            raise ValueError("generation token fields must be arrays")
        content_ids = tuple(_wire_token(item, field_name="content_token_ids") for item in content)
        stop_ids = tuple(_wire_token(item, field_name="stop_token_ids") for item in stop)
        finish_reason = _wire_text(normalized["finish_reason"], field_name="finish_reason")
        policy_snapshot_id = _wire_text(
            normalized["policy_snapshot_id"],
            field_name="policy_snapshot_id",
        )
        backend_id = _wire_text(normalized["backend_id"], field_name="backend_id")
        return cls(
            content_token_ids=content_ids,
            stop_token_ids=stop_ids,
            finish_reason=finish_reason,
            policy_snapshot_id=policy_snapshot_id,
            backend_id=backend_id,
            usage=BudgetVector.from_value(normalized["usage"]),
        )


class RolloutGenerator(Protocol):
    @property
    def tokenizer(self) -> RolloutTokenizerProtocol: ...

    def snapshot(self) -> PolicySnapshot: ...

    def begin_episode(self, episode_id: str, expected_policy_snapshot_id: str) -> None: ...

    def end_episode(self, episode_id: str) -> None: ...

    async def generate(self, request: RolloutGenerationRequest) -> RolloutGenerationResult: ...


class PolicySnapshotMismatchError(RuntimeError):
    pass


class LocalPolicyGenerator:
    def __init__(self, backbone: PolicyBackbone, *, episode_cache_enabled: bool = True) -> None:
        if type(episode_cache_enabled) is not bool:
            raise TypeError("episode_cache_enabled must be boolean")
        self._backbone = backbone
        self._episode_cache_enabled = episode_cache_enabled

    @property
    def tokenizer(self) -> RolloutTokenizerProtocol:
        return self._backbone.tokenizer

    def snapshot(self) -> PolicySnapshot:
        return PolicySnapshot.create(
            backbone_id=self._backbone.backbone_id,
            forward_adapter_version=self._backbone.adapter_version(AdapterRole.FORWARD_POLICY),
            tokenizer_id=self.tokenizer.tokenizer_id,
            backend_id=_BACKEND_ID,
            initial_trainable_state_hash=self._backbone.initial_trainable_state_hash,
        )

    def begin_episode(self, episode_id: str, expected_policy_snapshot_id: str) -> None:
        if self.snapshot().snapshot_id != expected_policy_snapshot_id:
            raise PolicySnapshotMismatchError("generator policy snapshot changed before episode")
        if self._episode_cache_enabled:
            self._backbone.begin_policy_episode(episode_id)

    def end_episode(self, episode_id: str) -> None:
        if self._episode_cache_enabled:
            self._backbone.end_policy_episode(episode_id)

    async def generate(self, request: RolloutGenerationRequest) -> RolloutGenerationResult:
        if request.sampling_constraint is not None:
            raise ValueError("local policy generation cannot apply a sampling constraint")
        if request.extra_stop_token_ids:
            raise ValueError("local policy generation cannot apply extra stop tokens")
        before = self.snapshot()
        if before.snapshot_id != request.expected_policy_snapshot_id:
            raise PolicySnapshotMismatchError("generator policy snapshot changed before generation")
        result = self._backbone.generate_policy(
            PolicyGenerationRequest(
                input_ids=request.input_ids,
                max_new_tokens=request.max_new_tokens,
                seed=request.seed,
                decoding_snapshot_id=request.decoding_snapshot_id,
            )
        )
        after = self.snapshot()
        if after != before:
            raise PolicySnapshotMismatchError("generator policy snapshot changed during generation")

        content_ids = result.content_token_ids
        stop_ids = result.stop_token_ids
        finish_reason = result.finish_reason

        return RolloutGenerationResult(
            content_token_ids=content_ids,
            stop_token_ids=stop_ids,
            finish_reason=finish_reason,
            policy_snapshot_id=after.snapshot_id,
            backend_id=after.backend_id,
            usage=BudgetVector(
                input_tokens=len(request.input_ids),
                output_tokens=(len(content_ids) + len(stop_ids)),
                model_calls=1,
            ),
        )
