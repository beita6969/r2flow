from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Final, Literal

from skillev.contracts import JsonValue, canonical_json
from skillev.contracts.native_call_format import tool_name
from skillev.contracts.reasoning_call_line import plain_call_view
from skillev.contracts.state_map import chain_functions

from .phase_context import (
    ACTION_PHASE_SUFFIX,
    PhaseContextSpec,
    deliverable_instruction,
    event_call_action_instruction,
    reasoning_tool_reference,
    token_budget_notice,
)

STATE_VIEW_VERSION = "state-view@1"
STATE_VIEW_HEADER = "SKILLEV_STATE_VIEW_V1\n"
StatePhase = Literal["reasoning", "forward", "hindsight"]
STATE_VIEW_BOUNDARY = (
    "Completed calls and their results are listed in a canonical order, not necessarily "
    "the order they ran; a call that used another call's result is listed after it. "
    "Only listed results record what happened."
)
_CHRONOLOGICAL_BOUNDARY = (
    "Completed calls and their results are listed in the order they ran. "
    "Only listed results record what happened."
)


IN_EDGE_HINDSIGHT_CHOICE_INSTRUCTION: Final = (
    "Hindsight: choose one listed completed call as the step into this state.\nChosen call:\n"
)


def _result_text(output: Any) -> str:
    if output.status != "success":
        error = dict(output.fields).get("error")
        return f"error: {error}" if isinstance(error, str) else "error"
    if output.handle is not None:
        return f"{output.handle}\n{output.content}"
    if output.env is not None:
        return str(output.env["text"])
    if output.terminal:
        return "accepted"
    return str(output.content)


def event_block(call_text: str, output: Any) -> str:
    return f"Call:\n{call_text}\nResult (status: {output.status}):\n{_result_text(output)}\n"


def is_chain(state: Any) -> bool:
    return tuple(state.edges) == tuple((index, index + 1) for index in range(state.rank - 1))


def declares_chain_order(spec: PhaseContextSpec) -> bool:
    return spec.reasoning_call_line is not None and chain_functions(
        tool_name(tool) for tool in json.loads(spec.tools_json)
    )


def forward_draft_view(spec: PhaseContextSpec, draft: str) -> str:
    return plain_call_view(draft) if spec.reasoning_call_line is not None else draft


def state_payload(
    initial_body: str,
    state: Any,
    *,
    phase: StatePhase,
    instruction: str,
    current: str | None = None,
    candidates: Sequence[str] | None = None,
    chain_order: bool = False,
) -> dict[str, JsonValue]:
    if chain_order and not is_chain(state):
        raise AssertionError("a state of a declared chain domain is not a chain")
    env = state.env
    payload: dict[str, JsonValue] = {
        "initial_text": initial_body,
        "phase": phase,
        "instruction": instruction,
        "order": "chronological" if chain_order else "canonical",
        "calls": [event_block(item.event.call_text(), item.output) for item in state.events],
        "environment": None
        if env is None
        else {"text": env["text"], "admissible_commands": list(env["admissible_commands"])},
        "controller": (
            {"completed_calls": state.rank, "max_turns": state.horizon}
            if phase == "hindsight"
            else {
                "turn": state.rank + 1,
                "max_turns": state.horizon,
                "turns_remaining_including_current": state.horizon - state.rank,
            }
        ),
    }
    if phase == "forward":
        if current is None:
            raise ValueError("the forward view carries the current draft")
        payload["current"] = current
    elif current is not None:
        raise ValueError("only the forward view carries a current draft")
    if candidates is not None:
        if phase != "hindsight":
            raise ValueError("candidate last calls belong to the hindsight view")
        payload["candidates"] = list(candidates)
    return payload


def render_state_view(spec: PhaseContextSpec, payload: dict[str, JsonValue]) -> str:
    return spec.wrap(STATE_VIEW_HEADER + canonical_json(payload))


def state_phase_chat_messages(
    spec: PhaseContextSpec, body: str
) -> tuple[list[dict[str, str]], list[dict[str, JsonValue]] | None]:
    from .interface import rollout_chat_messages

    value = (
        json.loads(body[len(STATE_VIEW_HEADER) :]) if body.startswith(STATE_VIEW_HEADER) else None
    )
    phase: StatePhase = (
        value["phase"]
        if value is not None
        else ("forward" if body.endswith(ACTION_PHASE_SUFFIX) else "reasoning")
    )
    canonical = (
        value["order"] == "canonical" if value is not None else not declares_chain_order(spec)
    )
    boundary = STATE_VIEW_BOUNDARY if canonical else _CHRONOLOGICAL_BOUNDARY
    if phase == "reasoning":
        system = (
            "Reason about the public task and current state. This is the reasoning phase; "
            "an executable action is requested separately. "
            + boundary
            + reasoning_tool_reference(spec)
        )
    else:
        system = event_call_action_instruction(spec) + " " + boundary
    system += deliverable_instruction(spec, reasoning=phase == "reasoning")
    system += token_budget_notice(spec, reasoning=phase == "reasoning")
    messages = rollout_chat_messages(
        value["initial_text"] if value is not None else body, system_message=system
    )
    tools = json.loads(spec.tools_json) if phase != "reasoning" else None
    if value is None:
        return messages, tools
    if value["calls"]:
        messages.append(
            {
                "role": "user",
                "content": f"Completed calls ({value['order']} order):\n" + "".join(value["calls"]),
            }
        )
    if value["environment"] is not None:
        messages.append(
            {
                "role": "user",
                "content": "Current environment state:\n" + canonical_json(value["environment"]),
            }
        )
    if phase == "forward":
        messages.append(
            {
                "role": "user",
                "content": "Your reasoning notes for this call (not executed):\n"
                + value["current"],
            }
        )
    tail = "Controller state (not a model prediction):\n" + canonical_json(value["controller"])
    if value.get("candidates") is not None:
        label = (
            "Candidate calls"
            if value["instruction"] == IN_EDGE_HINDSIGHT_CHOICE_INSTRUCTION
            else "Candidate last calls"
        )
        tail += f"\n{label}:\n" + "".join(f"{text}\n" for text in value["candidates"])
    messages.append({"role": "user", "content": tail + "\n" + value["instruction"]})
    return messages, tools


__all__ = [
    "IN_EDGE_HINDSIGHT_CHOICE_INSTRUCTION",
    "STATE_VIEW_BOUNDARY",
    "STATE_VIEW_HEADER",
    "STATE_VIEW_VERSION",
    "declares_chain_order",
    "event_block",
    "forward_draft_view",
    "is_chain",
    "render_state_view",
    "state_payload",
    "state_phase_chat_messages",
]
