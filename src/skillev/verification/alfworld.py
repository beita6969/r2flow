from __future__ import annotations

from collections.abc import Collection

from skillev.contracts import JsonValue

NOTHING_HAPPENS = "Nothing happens"


def feedback_text(observation: JsonValue) -> str | None:
    if isinstance(observation, dict):
        text = observation.get("text", observation.get("feedback"))
        return text if isinstance(text, str) else None
    return observation if isinstance(observation, str) else None


def act_outcome(
    command: str,
    admissible: Collection[str],
    observation: JsonValue,
    observation_status: str,
) -> tuple[bool, str]:
    if command not in admissible:
        return False, "not-admissible"
    if observation_status != "success":
        return False, "execution-error"
    text = feedback_text(observation)
    if text is None:
        return False, "no-feedback"
    if text.lstrip().startswith(NOTHING_HAPPENS):
        return False, "nothing-happens"
    return True, "admissible-executed"


__all__ = ["NOTHING_HAPPENS", "act_outcome", "feedback_text"]
