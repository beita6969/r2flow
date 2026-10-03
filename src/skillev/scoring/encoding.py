from __future__ import annotations

from skillev.contracts import TokenizerProtocol, TrajectoryRecord
from skillev.policy.interface import encode_rollout_prompt

from .rendering import assembled_context_hash


def encoded_initial_context(
    tokenizer: TokenizerProtocol,
    record: TrajectoryRecord,
    initial_text: str,
) -> tuple[int, ...]:
    if not isinstance(initial_text, str):
        raise ValueError(f"trajectory {record.trajectory_id!r}: initial_text must be text")
    actual_tokenizer_id = tokenizer.tokenizer_id
    if record.tokenizer_id != actual_tokenizer_id:
        raise ValueError(
            f"trajectory {record.trajectory_id!r}: tokenizer_id does not match backbone"
        )
    if assembled_context_hash(initial_text) != record.initial_context.assembled_hash:
        raise ValueError(
            f"trajectory {record.trajectory_id!r}: initial assembled context hash mismatch"
        )
    encoded = encode_rollout_prompt(tokenizer, initial_text)
    if len(encoded) != record.initial_context.assembled_token_count:
        raise ValueError(
            f"trajectory {record.trajectory_id!r}: initial assembled token count mismatch"
        )
    return tuple(encoded)


def encoded_query(
    tokenizer: TokenizerProtocol,
    record: TrajectoryRecord,
) -> tuple[int, ...]:
    encoded = tokenizer.encode(record.initial_context.query)
    if not encoded:
        raise ValueError(f"trajectory {record.trajectory_id!r}: query encoded to invalid token ids")
    return tuple(encoded)


__all__ = ["encoded_initial_context", "encoded_query"]
