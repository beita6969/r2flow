from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from itertools import takewhile
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

from skillev.contracts import JsonValue, TokenizerProtocol, canonical_json

from .phase_context import PhaseContextSpec as PhaseContextSpec
from .phase_context import phase_initial_text
from .skill_availability import context_for_turn
from .skill_availability import skill_available_from_turn as skill_available_from_turn
from .skill_availability import wrap_skill_availability as wrap_skill_availability
from .tokenizer_identity import PublicTokenizerIdentity
from .trainable_state import TrainableStateIdentity
from .versions import TrainableVersions

if TYPE_CHECKING:
    import torch

_FINISH_REASONS: Final = frozenset({"length", "stop"})
ROLLOUT_PROMPT_ENCODER_VERSION: Final = "qwen-chat-nonthinking@5"
THINKING_ROLLOUT_PROMPT_ENCODER_VERSION: Final = "qwen-native-reasoning-structured-action@1"
ROLLOUT_SOURCE_MESSAGES_BEGIN: Final = "<skillev-source-messages>\n"
ROLLOUT_SOURCE_MESSAGES_END: Final = "\n</skillev-source-messages>\n"
ROLLOUT_CONTROLLER_SYSTEM_MESSAGE: Final = (
    "Solve the task using the supplied information and available actions. The final "
    "prompt suffix indicates whether to reason about the task or send an action. "
    "Choose your own approach; the action interfaces are described in Available Actions."
)


def rollout_chat_messages(
    text: str,
    *,
    system_message: str = ROLLOUT_CONTROLLER_SYSTEM_MESSAGE,
) -> list[dict[str, str]]:
    if not isinstance(text, str) or not text:
        raise ValueError("rollout prompt must be non-empty text")
    if not isinstance(system_message, str) or not system_message.strip():
        raise ValueError("rollout system message must be non-empty text")
    begin = text.find(ROLLOUT_SOURCE_MESSAGES_BEGIN)
    if begin < 0:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text},
        ]
    payload_start = begin + len(ROLLOUT_SOURCE_MESSAGES_BEGIN)
    end = text.find(ROLLOUT_SOURCE_MESSAGES_END, payload_start)
    if end < 0 or text.find(ROLLOUT_SOURCE_MESSAGES_BEGIN, payload_start) >= 0:
        raise ValueError("rollout source-message envelope is malformed")
    raw_messages = text[payload_start:end]
    try:
        value = json.loads(raw_messages)
    except json.JSONDecodeError as error:
        raise ValueError("rollout source-message envelope is invalid JSON") from error
    if not isinstance(value, list) or canonical_json(value) != raw_messages:
        raise ValueError("rollout source-message envelope is not canonical")
    source_messages: list[dict[str, str]] = []
    for item in value:
        if (
            not isinstance(item, dict)
            or set(item) != {"content", "role"}
            or item.get("role") not in {"system", "user", "assistant"}
            or type(item.get("content")) is not str
            or not item["content"].strip()
        ):
            raise ValueError("rollout source-message entry is incompatible")
        source_messages.append({"role": item["role"], "content": item["content"]})

    leading_system: list[str] = []
    first_non_system = 0
    while (
        first_non_system < len(source_messages)
        and source_messages[first_non_system]["role"] == "system"
    ):
        leading_system.append(source_messages[first_non_system]["content"])
        first_non_system += 1
    remaining = source_messages[first_non_system:]
    if any(message["role"] == "system" for message in remaining):
        raise ValueError("rollout source system messages must form one leading block")
    system_content = system_message
    if leading_system:
        system_content += "\n\nSource system instructions:\n" + "\n\n".join(leading_system)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_content},
        *remaining,
    ]
    controller = text[:begin] + text[end + len(ROLLOUT_SOURCE_MESSAGES_END) :]
    messages.append({"role": "user", "content": controller})
    return messages


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite number")
    try:
        normalized = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be a finite number")
    return normalized


def _validate_token_ids(
    value: object,
    *,
    field: str,
    non_empty: bool,
) -> None:
    if not isinstance(value, tuple):
        raise ValueError(f"{field} must be a tuple of token ids")
    if non_empty and not value:
        raise ValueError(f"{field} must not be empty")
    if any(type(token_id) is not int or token_id < 0 for token_id in value):
        raise ValueError(f"{field} must contain non-negative integer token ids")


class AdapterRole(StrEnum):
    FORWARD_POLICY = "forward-policy"
    BACKWARD_POLICY = "backward-policy"


@runtime_checkable
class RolloutPromptTokenizerProtocol(Protocol):
    def encode_rollout_prompt(self, text: str) -> list[int]: ...


def encode_rollout_prompt(tokenizer: TokenizerProtocol, text: str) -> list[int]:
    text = context_for_turn(text, 1)
    if not isinstance(text, str) or not text:
        raise ValueError("rollout prompt must be non-empty text")
    if isinstance(tokenizer, RolloutPromptTokenizerProtocol):
        encoded = tokenizer.encode_rollout_prompt(text)
    else:
        encoded = tokenizer.encode(text)
    if not isinstance(encoded, list) or not encoded:
        raise ValueError("rollout prompt must encode to at least one token")
    if any(type(token_id) is not int or token_id < 0 for token_id in encoded):
        raise ValueError("rollout prompt encoding must contain non-negative token ids")
    return encoded


@runtime_checkable
class ThinkingRolloutTokenizerProtocol(Protocol):
    def encode_rollout_prompt_with_thinking(self, text: str) -> list[int]: ...


def encode_reasoning_prompt(
    tokenizer: TokenizerProtocol, text: str, *, native_thinking: bool
) -> list[int]:
    text = context_for_turn(text, 1)
    if not native_thinking:
        return encode_rollout_prompt(tokenizer, text)
    if not isinstance(tokenizer, ThinkingRolloutTokenizerProtocol):
        raise TypeError("native reasoning requires a thinking-capable tokenizer")
    encoded = tokenizer.encode_rollout_prompt_with_thinking(text)
    if not encoded or any(type(t) is not int or t < 0 for t in encoded):
        raise ValueError("native reasoning prompt must contain valid token IDs")
    return encoded


INPUT_WINDOW_VERSION = "h0-head-tail-recent-tokens@2"
INPUT_WINDOW_META_KEY = "model_input_window"


def _common_prefix_tokens(ids: list[int], initial_ids: list[int]) -> int:
    return sum(
        1 for _ in takewhile(lambda pair: pair[0] == pair[1], zip(ids, initial_ids, strict=False))
    )


@dataclass(frozen=True, slots=True)
class ModelInputWindow:
    max_tokens: int
    format: str = INPUT_WINDOW_VERSION

    def __post_init__(self) -> None:
        if type(self.max_tokens) is not int or self.max_tokens < 4:
            raise ValueError("model input window must allow at least four tokens")
        if self.format != INPUT_WINDOW_VERSION:
            raise ValueError("unsupported model input window")

    def to_value(self) -> dict[str, JsonValue]:
        return {"format": self.format, "max_tokens": self.max_tokens}

    @classmethod
    def from_value(cls, value: object) -> ModelInputWindow:
        if not isinstance(value, Mapping):
            raise ValueError("model input window must be an object")
        maximum, version = value.get("max_tokens"), value.get("format")
        if type(maximum) is not int or not isinstance(version, str):
            raise ValueError("invalid model input window fields")
        return cls(maximum, version)

    @classmethod
    def from_meta(cls, meta: Mapping[str, JsonValue]) -> ModelInputWindow | None:
        value = meta.get(INPUT_WINDOW_META_KEY)
        return None if value is None else cls.from_value(value)

    def retained_ranges(
        self, ids: list[int], initial_ids: list[int]
    ) -> tuple[tuple[int, int], ...]:
        if len(ids) <= self.max_tokens:
            return ((0, len(ids)),)
        common = _common_prefix_tokens(ids, initial_ids)
        recent_minimum = self.max_tokens // 2
        initial_end = min(common, len(ids) - recent_minimum)
        initial_count = min(initial_end, self.max_tokens - recent_minimum)
        head = (initial_count + 1) // 2
        tail = initial_count - head
        recent = self.max_tokens - initial_count
        ranges: list[tuple[int, int]] = []
        for start, end in (
            (0, head),
            (initial_end - tail, initial_end),
            (len(ids) - recent, len(ids)),
        ):
            if start == end:
                continue
            if ranges and ranges[-1][1] == start:
                ranges[-1] = (ranges[-1][0], end)
            else:
                ranges.append((start, end))
        return tuple(ranges)

    def apply(self, ids: list[int], initial_ids: list[int]) -> tuple[int, ...]:
        return tuple(
            token
            for start, end in self.retained_ranges(ids, initial_ids)
            for token in ids[start:end]
        )


@dataclass(frozen=True, slots=True)
class EncodedPolicyPrompt:
    ids: tuple[int, ...]
    original_tokens: int
    retained_segment_lengths: tuple[int, ...] | None = None
    protected_initial_tokens: int | None = None

    @property
    def removed_tokens(self) -> int:
        return self.original_tokens - len(self.ids)


def encode_policy_prompt(
    tokenizer: TokenizerProtocol,
    text: str,
    *,
    initial_text: str,
    window: ModelInputWindow | None,
    native_thinking: bool = False,
    step_index: int = 1,
) -> EncodedPolicyPrompt:
    ids = encode_reasoning_prompt(tokenizer, text, native_thinking=native_thinking)
    if window is None or len(ids) <= window.max_tokens:
        return EncodedPolicyPrompt(tuple(ids), len(ids))
    initial = phase_initial_text(text, context_for_turn(initial_text, step_index))
    initial_ids = encode_reasoning_prompt(tokenizer, initial, native_thinking=native_thinking)
    ranges = window.retained_ranges(ids, initial_ids)
    return EncodedPolicyPrompt(
        tuple(token for start, end in ranges for token in ids[start:end]),
        len(ids),
        tuple(end - start for start, end in ranges),
        _common_prefix_tokens(ids, initial_ids),
    )


class PolicyScoringMemoryError(RuntimeError):
    def __init__(
        self,
        *,
        prefix_token_count: int,
        action_token_count: int,
        role: AdapterRole,
    ) -> None:
        super().__init__("teacher-forced policy scoring exhausted device memory")
        self.prefix_token_count = prefix_token_count
        self.action_token_count = action_token_count
        self.role = role


def _validate_generation_common(
    *,
    input_ids: tuple[int, ...],
    max_new_tokens: int,
    seed: int,
) -> None:
    _validate_token_ids(input_ids, field="input_ids", non_empty=True)
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")


@dataclass(frozen=True, slots=True)
class PolicyGenerationRequest:
    input_ids: tuple[int, ...]
    max_new_tokens: int
    seed: int
    decoding_snapshot_id: str

    def __post_init__(self) -> None:
        _validate_generation_common(
            input_ids=self.input_ids,
            max_new_tokens=self.max_new_tokens,
            seed=self.seed,
        )
        if not isinstance(self.decoding_snapshot_id, str) or not self.decoding_snapshot_id.strip():
            raise ValueError("decoding_snapshot_id must be non-empty text")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    content_token_ids: tuple[int, ...]
    stop_token_ids: tuple[int, ...]
    finish_reason: str

    def __post_init__(self) -> None:
        _validate_token_ids(
            self.content_token_ids,
            field="content_token_ids",
            non_empty=False,
        )
        _validate_token_ids(
            self.stop_token_ids,
            field="stop_token_ids",
            non_empty=False,
        )
        if not isinstance(self.finish_reason, str) or self.finish_reason not in _FINISH_REASONS:
            raise ValueError("finish_reason is unsupported")
        if self.finish_reason == "length" and self.stop_token_ids:
            raise ValueError("non-stop generation cannot carry stop_token_ids")
        if self.finish_reason == "stop" and not self.stop_token_ids:
            raise ValueError("stop-finished generation must carry stop_token_ids")


class PolicyTokenizerProtocol(
    TokenizerProtocol,
    RolloutPromptTokenizerProtocol,
    Protocol,
):
    @property
    def public_identity(self) -> PublicTokenizerIdentity: ...

    def decode(self, token_ids: tuple[int, ...]) -> str: ...

    def encode_rollout_prompt(self, text: str) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class PolicyParameterGroups:
    forward: tuple[torch.nn.Parameter, ...]
    backward: tuple[torch.nn.Parameter, ...]
    z_head: tuple[torch.nn.Parameter, ...]
    psi_head: tuple[torch.nn.Parameter, ...] = ()


class PolicyBackbone(Protocol):
    @property
    def backbone_id(self) -> str: ...

    @property
    def tokenizer(self) -> PolicyTokenizerProtocol: ...

    def adapter_version(self, role: AdapterRole) -> str: ...

    @property
    def z_version(self) -> str: ...

    @property
    def trainable_state_identity(self) -> TrainableStateIdentity: ...

    @property
    def initial_trainable_state_hash(self) -> str: ...

    def bind_initial_trainable_state(self, expected: TrainableStateIdentity) -> None: ...

    def mark_policy_update(self, optimizer_step: int) -> None: ...

    def generate_policy(self, request: PolicyGenerationRequest) -> GenerationResult: ...

    def synchronize_trainable_versions(self, versions: TrainableVersions) -> None: ...

    def begin_policy_episode(self, episode_id: str) -> None: ...

    def end_policy_episode(self, episode_id: str) -> None: ...

    def score(
        self,
        prefix_ids: tuple[int, ...],
        action_ids: tuple[int, ...],
        role: AdapterRole,
    ) -> torch.Tensor: ...

    def z_value(self, query_ids: tuple[int, ...]) -> torch.Tensor: ...

    def reset_z(self, seed: int) -> str: ...

    def parameter_groups(self) -> PolicyParameterGroups: ...

    def named_trainable_parameters(self) -> dict[str, torch.nn.Parameter]: ...

    def save_checkpoint(self, directory: str) -> None: ...

    def load_checkpoint(self, directory: str) -> None: ...
