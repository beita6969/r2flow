from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from skillev.contracts import JsonValue
from skillev.contracts.action_wire import NATIVE_EVENT_CALL_WIRE

from .action_contract import ActionContract
from .action_surface import ActionSurface
from .errors import RolloutBoundaryError
from .generator import RolloutTokenizerProtocol
from .native_wire import NativeToolWire


@dataclass(frozen=True, slots=True)
class DecodedSegment:
    text: str
    token_ids: tuple[int, ...]


def decode_reasoning_segment(
    tokenizer: RolloutTokenizerProtocol,
    token_ids: tuple[int, ...],
) -> DecodedSegment:
    if not isinstance(token_ids, tuple) or any(
        type(token_id) is not int or token_id < 0 for token_id in token_ids
    ):
        raise RolloutBoundaryError("reasoning token IDs are invalid")
    text = tokenizer.decode(token_ids)
    if not isinstance(text, str):
        raise RolloutBoundaryError("tokenizer.decode must return text")
    return DecodedSegment(text=text, token_ids=token_ids)


def decode_action_segment(
    tokenizer: RolloutTokenizerProtocol,
    token_ids: tuple[int, ...],
) -> DecodedSegment:
    if not isinstance(token_ids, tuple) or not token_ids:
        raise RolloutBoundaryError("action token IDs must be non-empty")
    if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
        raise RolloutBoundaryError("action token IDs are invalid")
    text = tokenizer.decode(token_ids)
    if not isinstance(text, str) or not text:
        raise RolloutBoundaryError("decoded action text must be non-empty")
    return DecodedSegment(text=text, token_ids=token_ids)


def codec_for_initial_meta(meta: Mapping[str, JsonValue]) -> NativeToolWire:
    if meta.get("action_wire") != NATIVE_EVENT_CALL_WIRE:
        raise ValueError("unsupported persisted action wire")
    contract = meta.get("action_contract")
    if not isinstance(contract, dict) or not isinstance(contract.get("surface"), dict):
        raise ValueError("native wire metadata lacks its public contract")
    retrieved, active = contract.get("retrieved_skill_ids"), contract.get("active_skill_ids")
    if (
        not isinstance(retrieved, list)
        or not isinstance(active, list)
        or any(not isinstance(sid, str) for sid in [*retrieved, *active])
    ):
        raise ValueError("native contract lacks its exact skill identity")
    return NativeToolWire(
        ActionContract.freeze(
            ActionSurface.from_value(contract["surface"]),
            retrieved_skill_ids=tuple(cast(list[str], retrieved)),
            active_skill_ids=tuple(cast(list[str], active)),
        )
    )
