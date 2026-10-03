from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .canonical import JsonValue
from .identity import validate_identifier


class SkillInvocationAdmissionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ParsedActionInvocation:
    kind: str
    skill_id: str | None


def _require_skill_id_tuple(value: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{field} must be a tuple")
    if any(type(skill_id) is not str for skill_id in value):
        raise ValueError(f"{field} must contain text skill IDs")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} must not repeat a skill ID")
    for skill_id in value:
        validate_identifier(skill_id)
    return value


def parse_action_invocation(
    action_text: str, *, initial_meta: Mapping[str, JsonValue]
) -> ParsedActionInvocation | None:
    if type(action_text) is not str:
        raise TypeError("action_text must be text")
    from skillev.rollout.codec import codec_for_initial_meta

    action = codec_for_initial_meta(initial_meta).parse(action_text).action
    return (
        None
        if action is None
        else ParsedActionInvocation(kind=action.kind.value, skill_id=action.skill_id)
    )


def canonical_invoked_skill_ids(
    *,
    action_kind: str,
    action_skill_id: str | None,
    retrieved_skill_ids: tuple[str, ...],
    active_skill_ids: tuple[str, ...],
) -> tuple[str, ...]:
    _require_skill_id_tuple(retrieved_skill_ids, field="retrieved_skill_ids")
    _require_skill_id_tuple(active_skill_ids, field="active_skill_ids")
    if action_kind != "skill":
        if action_skill_id is not None:
            raise SkillInvocationAdmissionError("only skill actions may carry skill_id")
        return ()
    if type(action_skill_id) is not str:
        raise SkillInvocationAdmissionError("skill action is missing skill_id")
    try:
        validate_identifier(action_skill_id)
    except ValueError as error:
        raise SkillInvocationAdmissionError("skill action skill_id is invalid") from error
    if action_skill_id not in active_skill_ids:
        raise SkillInvocationAdmissionError("skill action targets an inactive skill")
    if action_skill_id not in retrieved_skill_ids:
        raise SkillInvocationAdmissionError("skill action targets a skill absent from H0")
    return (action_skill_id,)


def validate_trajectory_skill_invocations(
    *,
    retrieved_skill_ids: tuple[str, ...],
    active_skill_ids: tuple[str, ...],
    steps: tuple[object, ...],
    initial_meta: Mapping[str, JsonValue],
) -> None:
    _require_skill_id_tuple(retrieved_skill_ids, field="retrieved_skill_ids")
    _require_skill_id_tuple(active_skill_ids, field="active_skill_ids")
    if not set(retrieved_skill_ids) <= set(active_skill_ids):
        raise ValueError("retrieved_skill_ids reference inactive skills")
    for position, step in enumerate(steps, start=1):
        action_text = getattr(step, "action_text", None)
        observed = getattr(step, "invoked_skill_ids", None)
        observation_status = getattr(step, "observation_status", None)
        if type(action_text) is not str or not isinstance(observed, tuple):
            raise TypeError("trajectory step lacks invocation admission fields")
        if observation_status in {"parse_error", "schema_invalid"} and observed == ():
            continue
        parsed = parse_action_invocation(action_text, initial_meta=initial_meta)
        if parsed is None:
            expected: tuple[str, ...] = ()
        else:
            try:
                expected = canonical_invoked_skill_ids(
                    action_kind=parsed.kind,
                    action_skill_id=parsed.skill_id,
                    retrieved_skill_ids=retrieved_skill_ids,
                    active_skill_ids=active_skill_ids,
                )
            except SkillInvocationAdmissionError as error:
                if observation_status != "schema_invalid" or observed != ():
                    raise ValueError(
                        f"trajectory step {position} contains unavailable skill credit"
                    ) from error
                continue
        if observed != expected:
            raise ValueError(f"trajectory step {position} invoked_skill_ids differ from its action")


__all__ = [
    "ParsedActionInvocation",
    "SkillInvocationAdmissionError",
    "canonical_invoked_skill_ids",
    "parse_action_invocation",
    "validate_trajectory_skill_invocations",
]
