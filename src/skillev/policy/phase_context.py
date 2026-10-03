from __future__ import annotations

import json
from dataclasses import dataclass

from skillev.contracts import JsonValue, canonical_json
from skillev.contracts.action_wire import NATIVE_EVENT_CALL_WIRE
from skillev.contracts.answer_writer import declared_completion_writer
from skillev.contracts.final_turn import (
    COMPLETION_FUNCTION,
    EXPLICIT_COMPLETION_TERMINAL_MODE,
    FINAL_TURN_COMPLETION_RULES,
)
from skillev.contracts.native_call_format import (
    call_example_values,
    primary_tool,
    render_function_summary,
    render_xml_call_example,
    tool_name,
)
from skillev.contracts.reasoning_call_line import (
    ENVIRONMENT_ACT_FUNCTION,
    EXACT_COMMAND_PLACEHOLDER,
    NEXT_CALL,
    NEXT_COMMAND,
    REASONING_CALL_LINE_INSTRUCTION,
    REASONING_CALL_LINE_RULES,
    plain_call,
)
from skillev.contracts.skill_call_budget import SKILL_FUNCTION
from skillev.contracts.skill_visibility import NONEMPTY_ONLY, SKILL_VISIBILITY_RULES
from skillev.contracts.state_map import SIGMA_TRACE_QUOTIENT
from skillev.contracts.wikipedia_search import (
    WIKIPEDIA_SEARCH_FUNCTION_NOTE,
    WIKIPEDIA_SEARCH_RESOURCE,
    WIKIPEDIA_SEARCH_TOOL_NAME,
)

from .skill_availability import context_for_turn

PHASE_CONTEXT_VERSION = "phase-context@10"
REASONING_CALL_LINE_PHASE_CONTEXT_VERSION = "phase-context@11"
PHASE_CONTEXT_HEADER = "SKILLEV_PHASE_CONTEXT_V1 "
WRITER_DELIVERABLES = frozenset({"conversation-reply@3", "integer-answer@3"})
ACTION_PHASE_SUFFIX = "Action:\n"


@dataclass(frozen=True, slots=True)
class PhaseContextSpec:
    tools_json: str
    action_contract_json: str
    initial_public_state_json: str
    max_turns: int
    reasoning_token_cap: int
    action_token_cap: int
    state_map: str
    skill_visibility: str
    deliverable: str | None = None
    final_turn_completion: str | None = None
    skill_call_budget: int | None = None
    reasoning_call_line: str | None = None

    def __post_init__(self) -> None:
        if self.deliverable not in {
            None,
            "conversation-reply@2",
            "integer-answer@2",
            "environment-event@2",
            *WRITER_DELIVERABLES,
        }:
            raise ValueError("unsupported public phase deliverable")
        if self.deliverable in WRITER_DELIVERABLES and declared_completion_writer(self) is None:
            raise ValueError("the answer-writer phase notes require an executor-answer@1 H0")
        if not isinstance(json.loads(self.tools_json), list) or not json.loads(self.tools_json):
            raise ValueError("native action phase requires declared tools")
        if self.state_map != SIGMA_TRACE_QUOTIENT or type(self.max_turns) is not int:
            raise ValueError("state_map requires max_turns")
        if self.max_turns < 1:
            raise ValueError("phase turn limit must be a positive integer")
        if self.final_turn_completion is not None:
            self._require_final_turn_rule()
        if self.skill_call_budget is not None and (
            type(self.skill_call_budget) is not int
            or self.skill_call_budget < 1
            or SKILL_FUNCTION not in {tool_name(tool) for tool in json.loads(self.tools_json)}
        ):
            raise ValueError(
                "skill-call-budget@1 requires a positive integer budget in a sigma H0 "
                f"(state map, max_turns) that declares {SKILL_FUNCTION}"
            )
        if self.skill_visibility not in SKILL_VISIBILITY_RULES:
            raise ValueError("skill visibility nonempty-only@1 requires a sigma H0 (state map)")
        if self.reasoning_call_line is not None and (
            self.reasoning_call_line not in REASONING_CALL_LINE_RULES
        ):
            raise ValueError("reasoning-call-line@1 requires a declared reasoning rule")
        json.loads(self.initial_public_state_json)
        if any(
            type(cap) is not int or cap < 1
            for cap in (self.reasoning_token_cap, self.action_token_cap)
        ):
            raise ValueError("token budget notice requires both actual positive decoding caps")

    def _require_final_turn_rule(self) -> None:
        contract = json.loads(self.action_contract_json)
        surface = contract.get("surface") if isinstance(contract, dict) else None
        if (
            self.final_turn_completion not in FINAL_TURN_COMPLETION_RULES
            or COMPLETION_FUNCTION not in {tool_name(tool) for tool in json.loads(self.tools_json)}
            or (
                isinstance(surface, dict)
                and surface.get("terminal_mode") != EXPLICIT_COMPLETION_TERMINAL_MODE
            )
        ):
            raise ValueError(
                "final-turn-submit@1 requires a sigma H0 (state map, max_turns) of an "
                "explicit-completion surface that declares the completion function "
                f"{COMPLETION_FUNCTION}"
            )

    def to_value(self) -> dict[str, JsonValue]:
        value: dict[str, JsonValue] = {
            "version": PHASE_CONTEXT_VERSION,
            "action_wire": NATIVE_EVENT_CALL_WIRE,
            "tools": json.loads(self.tools_json),
            "action_contract": json.loads(self.action_contract_json),
            "state_map": self.state_map,
            "max_turns": self.max_turns,
            "initial_public_state": json.loads(self.initial_public_state_json),
            "reasoning_tool_catalog": True,
            "reasoning_token_cap": self.reasoning_token_cap,
            "action_token_cap": self.action_token_cap,
            "deliverable": self.deliverable,
            "submission_feedback": False,
            "skill_advice": True,
            "prompt_profile": None,
            "final_turn_completion": self.final_turn_completion,
            "skill_call_budget": self.skill_call_budget,
            "skill_visibility": self.skill_visibility,
        }
        if self.reasoning_call_line is not None:
            value.update(
                version=REASONING_CALL_LINE_PHASE_CONTEXT_VERSION,
                reasoning_call_line=self.reasoning_call_line,
            )
        return value

    def wrap(self, initial_text: str) -> str:
        return PHASE_CONTEXT_HEADER + canonical_json(self.to_value()) + "\n" + initial_text

    @classmethod
    def split(cls, text: str) -> tuple[PhaseContextSpec | None, str]:
        text = context_for_turn(text, 1)
        if not text.startswith(PHASE_CONTEXT_HEADER):
            return None, text
        line, separator, body = text.partition("\n")
        if not separator:
            raise ValueError("phase context lacks a body")
        value = json.loads(line[len(PHASE_CONTEXT_HEADER) :])
        if (
            not isinstance(value, dict)
            or value.get("version")
            not in {PHASE_CONTEXT_VERSION, REASONING_CALL_LINE_PHASE_CONTEXT_VERSION}
            or value["action_wire"] != NATIVE_EVENT_CALL_WIRE
        ):
            raise ValueError("unsupported phase context metadata")
        call_line = (
            value["reasoning_call_line"]
            if value["version"] == REASONING_CALL_LINE_PHASE_CONTEXT_VERSION
            else None
        )
        if value["version"] == REASONING_CALL_LINE_PHASE_CONTEXT_VERSION and call_line is None:
            raise ValueError("phase-context@11 declares its reasoning rule")
        return cls(
            tools_json=canonical_json(value["tools"]),
            action_contract_json=canonical_json(value["action_contract"]),
            initial_public_state_json=canonical_json(value["initial_public_state"]),
            max_turns=value["max_turns"],
            reasoning_token_cap=value["reasoning_token_cap"],
            action_token_cap=value["action_token_cap"],
            state_map=value["state_map"],
            skill_visibility=value["skill_visibility"],
            deliverable=value["deliverable"],
            final_turn_completion=value["final_turn_completion"],
            skill_call_budget=value["skill_call_budget"],
            reasoning_call_line=call_line,
        ), body


def deliverable_instruction(spec: PhaseContextSpec, *, reasoning: bool) -> str:
    if spec.deliverable is None:
        return ""
    if spec.deliverable == "conversation-reply@2":
        instruction = (
            "This reasoning draft is not delivered to the user. The action phase delivers the "
            "reply by calling submit_answer."
            if reasoning
            else "The answer parameter of submit_answer is the reply the user sees, not a "
            "summary of the reasoning or a pointer to an earlier draft."
        )
    elif spec.deliverable == "conversation-reply@3":
        instruction = (
            "This reasoning draft is not delivered to the user. Calling submit_answer() ends the "
            "episode; the frozen answer writer then writes the reply the user sees from the "
            "conversation, the listed call results and this draft."
            if reasoning
            else "submit_answer() has no parameters: the frozen answer writer writes the reply "
            "the user sees from the conversation, the listed call results and the reasoning "
            "draft of this turn."
        )
    elif spec.deliverable == "integer-answer@3":
        instruction = (
            "This reasoning draft is not submitted. Calling submit_answer() ends the episode; "
            "the frozen answer writer then writes the final integer from 0 through 999 from the "
            "problem, the listed call results and this draft."
            if reasoning
            else "submit_answer() has no parameters: the frozen answer writer writes the final "
            "integer from 0 through 999 from the problem, the listed call results and the "
            "reasoning draft of this turn."
        )
    elif spec.deliverable == "integer-answer@2":
        instruction = (
            "This reasoning draft is not submitted. The action phase submits the final integer "
            "by calling submit_answer."
            if reasoning
            else "Deliver your final integer from 0 through 999 in submit_answer's answer "
            "parameter. The tool call, not surrounding mathematical prose or a promise to "
            "compute later, is the submission."
        )
    else:
        hidden = spec.skill_visibility == NONEMPTY_ONLY and SKILL_FUNCTION not in {
            tool_name(tool) for tool in json.loads(spec.tools_json)
        }
        instruction = (
            (
                "Decide the next single action. "
                "A possible future plan is not an executed sequence: do not supply imagined "
                "environment replies. For an environment action, choose the exact current command "
                "rather than a paraphrase."
                + (
                    " A useful skill can instead be run through invoke_skill; no skill call is "
                    "required."
                    if not hidden
                    else ""
                )
            )
            if reasoning
            else "Execute the one next action you choose using its actual declared function. "
            "For an environment action, send the exact current admissible command; do not "
            "execute several imagined future steps or claim they already happened. Only "
            "actual environment feedback advances the known state."
        )
    return "\nPhase deliverable:\n" + instruction


def token_budget_notice(spec: PhaseContextSpec, *, reasoning: bool) -> str:
    cap = spec.reasoning_token_cap if reasoning else spec.action_token_cap
    phase = "Reasoning" if reasoning else "Action"
    return (
        f"\n{phase} output limit: at most {cap} tokens for this request, enforced by the runtime. "
        + (
            "Aim to finish your reasoning within this allowance."
            if reasoning
            else "Begin with <tool_call> and keep the whole call within this allowance; "
            "a truncated call is not executed."
        )
    )


def reasoning_call_lines(tools: list[dict[str, JsonValue]]) -> tuple[str, ...]:
    return tuple(
        f"{NEXT_COMMAND} {EXACT_COMMAND_PLACEHOLDER}"
        if tool_name(tool) == ENVIRONMENT_ACT_FUNCTION
        else f"{NEXT_CALL} {plain_call(tool_name(tool), call_example_values(tool))}"
        for tool in tools
    )


def reasoning_tool_reference(spec: PhaseContextSpec) -> str:
    tools = json.loads(spec.tools_json)
    summaries = "\n".join("- " + render_function_summary(t, with_description=True) for t in tools)
    if spec.reasoning_call_line is not None:
        lines = reasoning_call_lines(tools)
        return (
            "\nFunctions for the next action phase (reference only; nothing is executed "
            "while reasoning):\n"
            + summaries
            + "\n"
            + REASONING_CALL_LINE_INSTRUCTION
            + ("in this form:\n" if len(lines) == 1 else "in one of these forms:\n")
            + "\n".join(lines)
        )
    return (
        "\nFunctions for the next action phase (reference only; nothing is executed "
        "while reasoning):\n"
        + summaries
        + "\nThe action phase replies with exactly one call in this XML format and "
        "nothing else:\n" + render_xml_call_example(primary_tool(tools))
    )


_TOOL_NOTE = "Pass the chosen tool input through this function's declared parameters."
_FUNCTION_NOTES = {
    "submit_answer": "Deliver your complete response as this function's parameter value.",
    "invoke_skill": "Runs the skill on your input with the frozen executor; the output is "
    "returned as the observation.",
    "open_passage": "Returns the full text of one listed document.",
    "corpus_search": "Returns the five stored question-answer pairs that best match the query.",
    "act": "Pass one current admissible command.",
}
EVENT_CALL_ACTION_RULE = (
    "Your reply is exactly one tool call and nothing else: begin with <tool_call> and end "
    "with </tool_call>."
)


def _declares_wikipedia_search(spec: PhaseContextSpec) -> bool:
    contract = json.loads(spec.action_contract_json)
    surface = contract.get("surface") if isinstance(contract, dict) else None
    tools = surface.get("tools") if isinstance(surface, dict) else None
    return isinstance(tools, list) and any(
        isinstance(tool, dict)
        and (tool.get("resource_id"), tool.get("name"))
        == (WIKIPEDIA_SEARCH_RESOURCE, WIKIPEDIA_SEARCH_TOOL_NAME)
        for tool in tools
    )


WRITER_SUBMIT_NOTE = (
    "Ends the episode; the frozen answer writer writes the final response from the task, the "
    "listed call results and the reasoning draft of this turn."
)


def function_notes(spec: PhaseContextSpec) -> dict[str, str]:
    notes = _FUNCTION_NOTES
    if _declares_wikipedia_search(spec):
        notes = {**notes, "corpus_search": WIKIPEDIA_SEARCH_FUNCTION_NOTE}
    if declared_completion_writer(spec) is not None:
        notes = {**notes, "submit_answer": WRITER_SUBMIT_NOTE}
    return notes


def event_call_action_instruction(spec: PhaseContextSpec) -> str:
    tools = json.loads(spec.tools_json)
    notes = function_notes(spec)
    lines = [
        "Action phase: " + EVENT_CALL_ACTION_RULE,
        f"Required format (example for {tool_name(primary_tool(tools))}):",
        render_xml_call_example(primary_tool(tools)),
        "Available functions for this action:",
    ]
    for tool in tools:
        lines.append(f"- {render_function_summary(tool)} {notes.get(tool_name(tool), _TOOL_NOTE)}")
    return "\n".join(lines)


def native_action_reminder() -> str:
    return (
        "Action phase: reply now with exactly one <tool_call> block in the XML format given "
        "in the system instructions, with nothing before or after it."
    )


def phase_chat_messages(
    text: str,
) -> tuple[list[dict[str, str]], list[dict[str, JsonValue]] | None]:
    from .interface import rollout_chat_messages

    spec, body = PhaseContextSpec.split(text)
    if spec is None:
        return rollout_chat_messages(text), None
    from .state_view import state_phase_chat_messages

    return state_phase_chat_messages(spec, body)


def phase_initial_text(prompt_text: str, initial_text: str) -> str:
    spec, body = PhaseContextSpec.split(prompt_text)
    if spec is None:
        return initial_text
    from .state_view import STATE_VIEW_HEADER

    reasoning = (
        body.startswith(STATE_VIEW_HEADER)
        and json.loads(body[len(STATE_VIEW_HEADER) :])["phase"] == "reasoning"
    )
    return initial_text if reasoning else initial_text + ACTION_PHASE_SUFFIX
