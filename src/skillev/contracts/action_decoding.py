from __future__ import annotations

from typing import Final

ACTION_GREEDY_UNSEEDED: Final = "action-greedy-unseeded@1"
ACTION_DECODING_RULES: Final = frozenset({ACTION_GREEDY_UNSEEDED})


def require_action_decoding_coupling(declared: str | None, generator: object) -> None:
    applied = getattr(generator, "action_decoding", None)
    if applied != declared:
        raise ValueError(
            f"the rollout condition declares action decoding {declared!r} but its generator "
            f"applies {applied!r} ({ACTION_GREEDY_UNSEEDED} must be declared on both)"
        )


__all__ = [
    "ACTION_DECODING_RULES",
    "ACTION_GREEDY_UNSEEDED",
    "require_action_decoding_coupling",
]
