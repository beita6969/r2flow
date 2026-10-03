from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from skillev.contracts import JsonValue
from skillev.contracts.final_turn import COMPLETION_FUNCTION, FINAL_TURN_SUBMIT
from skillev.contracts.integer_answer import INTEGER_ANSWER_PATTERN, INTEGER_ANSWER_VALUES
from skillev.contracts.skill_call_budget import SKILL_FUNCTION
from skillev.policy.event_grammar import (
    EVENT_GRAMMAR_VERSION,
    EventGrammarSpec,
    FunctionSpec,
    ParamSpec,
    key_json,
)
from skillev.policy.skill_availability import context_for_turn, skill_available_from_turn

if TYPE_CHECKING:
    from skillev.contracts.ttb_trajectory import TrajectoryStep

LEGAL_EVENT_SET_VERSION = "legal-event-set@1"
DYNAMIC_ENUMS: Mapping[tuple[str, str], str] = {("act", "command"): "admissible_commands"}
ACT_FUNCTION = "act"


@dataclass(frozen=True, slots=True)
class LegalEventSet:
    functions: tuple[FunctionSpec, ...]

    def __post_init__(self) -> None:
        names = [function.name for function in self.functions]
        if names != sorted(set(names)):
            raise ValueError("legal functions must be unique and sorted by name")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(function.name for function in self.functions)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "version": LEGAL_EVENT_SET_VERSION,
            "functions": [
                {
                    "name": function.name,
                    "params": [
                        {"name": param.name, "kind": param.kind, "values": list(param.values)}
                        for param in function.params
                    ],
                }
                for function in self.functions
            ],
        }

    @classmethod
    def from_value(cls, value: object) -> LegalEventSet:
        if not isinstance(value, dict) or set(value) != {"version", "functions"}:
            raise ValueError("legal event set has an incompatible field set")
        if value["version"] != LEGAL_EVENT_SET_VERSION:
            raise ValueError("unsupported legal event set version")
        functions = value["functions"]
        if not isinstance(functions, list):
            raise ValueError("legal event set functions must be a list")
        parsed: list[FunctionSpec] = []
        for function in functions:
            if not isinstance(function, dict) or set(function) != {"name", "params"}:
                raise ValueError("legal function has an incompatible field set")
            params = function["params"]
            if not isinstance(params, list):
                raise ValueError("legal function params must be a list")
            specs: list[ParamSpec] = []
            for param in params:
                if not isinstance(param, dict) or set(param) != {"name", "kind", "values"}:
                    raise ValueError("legal parameter has an incompatible field set")
                values = param["values"]
                if not isinstance(values, list) or any(type(item) is not str for item in values):
                    raise ValueError("legal parameter values must be strings")
                kind = param["kind"]
                if kind not in ("enum", "text"):
                    raise ValueError("unsupported legal parameter kind")
                specs.append(ParamSpec(str(param["name"]), kind, tuple(values)))
            parsed.append(FunctionSpec(str(function["name"]), tuple(specs)))
        result = cls(tuple(parsed))
        if result.to_value() != value:
            raise ValueError("legal event set is not in canonical form")
        return result

    @property
    def sha256(self) -> str:
        return hashlib.sha256(key_json(self.to_value()).encode("utf-8")).hexdigest()

    def allows(self, function: str, args: Mapping[str, str]) -> bool:
        declared = next((item for item in self.functions if item.name == function), None)
        if declared is None or set(args) != {param.name for param in declared.params}:
            return False
        for param in declared.params:
            value = args[param.name]
            if type(value) is not str:
                return False
            if param.kind == "enum" and value not in param.values:
                return False
        return True

    def grammar_spec(self, *, budget: int, stop_token_id: int) -> EventGrammarSpec:
        if not self.functions:
            raise ValueError("a terminal state has no legal events")
        return EventGrammarSpec(EVENT_GRAMMAR_VERSION, self.functions, budget, stop_token_id)


def _phase_spec(initial_text: str, turn: int):
    from skillev.policy.phase_context import PhaseContextSpec

    spec, _ = PhaseContextSpec.split(context_for_turn(initial_text, turn))
    if spec is None:
        raise ValueError("the legal event set requires a phase-context H0")
    return spec


def horizon(initial_text: str) -> int:
    spec = _phase_spec(initial_text, 1)
    if type(spec.max_turns) is not int:
        raise ValueError("the legal event set requires the H0 max_turns")
    return int(spec.max_turns)


def final_turn_rule(initial_text: str) -> str | None:
    rule: str | None = _phase_spec(initial_text, 1).final_turn_completion
    return rule


def _commands(env: Mapping[str, JsonValue] | None, field: str) -> tuple[str, ...]:
    if env is None:
        return ()
    values = env.get(field)
    if not isinstance(values, list):
        return ()
    return tuple(sorted({value for value in values if isinstance(value, str) and value}))


def legal_event_set(
    initial_text: str,
    *,
    rank: int,
    omega_env: Mapping[str, JsonValue] | None,
    terminal: bool = False,
    skill_calls: int = 0,
) -> LegalEventSet:
    if type(rank) is not int or rank < 0:
        raise ValueError("rank must be a non-negative integer")
    if type(skill_calls) is not int or not 0 <= skill_calls <= rank:
        raise ValueError("skill_calls must be an integer in [0, rank]")
    if skill_available_from_turn(initial_text) != 1:
        raise ValueError("R2 Flow legal events require skills legal from turn 1")
    total = horizon(initial_text)
    if terminal or rank >= total:
        return LegalEventSet(())
    spec = _phase_spec(initial_text, rank + 1)
    functions: list[FunctionSpec] = []
    for tool in json.loads(spec.tools_json):
        function = tool["function"]
        name = function["name"]
        parameters = function["parameters"]
        properties = parameters["properties"]
        if set(parameters.get("required", ())) != set(properties):
            raise ValueError(f"{name}: every event parameter must be required")
        enums: list[ParamSpec] = []
        texts: list[ParamSpec] = []
        legal = True
        for param_name in sorted(properties):
            schema = properties[param_name]
            dynamic = DYNAMIC_ENUMS.get((name, param_name))
            if dynamic is not None:
                values = _commands(omega_env, dynamic)
                legal = legal and bool(values)
                if values:
                    enums.append(ParamSpec(param_name, "enum", values))
            elif isinstance(schema.get("enum"), list):
                enums.append(ParamSpec(param_name, "enum", tuple(schema["enum"])))
            elif schema.get("type") == "string" and "pattern" in schema:
                if schema["pattern"] != INTEGER_ANSWER_PATTERN:
                    raise ValueError(f"{name}.{param_name}: unsupported pattern")
                enums.append(ParamSpec(param_name, "enum", INTEGER_ANSWER_VALUES))
            elif schema.get("type") == "string":
                texts.append(ParamSpec(param_name, "text"))
            else:
                raise ValueError(f"{name}.{param_name}: unsupported event parameter type")
        if legal:
            functions.append(FunctionSpec(name, (*enums, *texts)))
    if spec.final_turn_completion == FINAL_TURN_SUBMIT and rank == total - 1:
        functions = [item for item in functions if item.name == COMPLETION_FUNCTION]
        if not functions:
            raise ValueError("final-turn-submit@1: the completion function is not declared")
    if spec.skill_call_budget is not None and skill_calls >= spec.skill_call_budget:
        functions = [item for item in functions if item.name != SKILL_FUNCTION]
        if not functions:
            raise ValueError("skill-call-budget@1: no legal event besides invoke_skill")
    return LegalEventSet(tuple(sorted(functions, key=lambda item: item.name)))


def initial_environment(initial_text: str) -> dict[str, JsonValue] | None:
    spec = _phase_spec(initial_text, 1)
    value = json.loads(spec.initial_public_state_json)
    return environment_from_observation(value, initial=True)


def environment_from_observation(
    value: object, *, initial: bool = False
) -> dict[str, JsonValue] | None:
    if not isinstance(value, dict):
        return None
    commands = value.get("admissible_commands")
    text = value.get("text", value.get("initial_observation") if initial else None)
    if (
        not isinstance(commands, list)
        or any(type(item) is not str for item in commands)
        or not isinstance(text, str)
    ):
        return None
    terminal = value.get("terminal", False)
    return {
        "text": text,
        "admissible_commands": list(commands),
        "terminal": terminal if type(terminal) is bool else False,
    }


def current_admissible_commands(
    initial_text: str, steps: Sequence[TrajectoryStep]
) -> tuple[str, ...]:
    env = initial_environment(initial_text)
    for step in steps:
        try:
            value = json.loads(step.observation_text)
        except json.JSONDecodeError:
            continue
        observed = environment_from_observation(value)
        if observed is not None and step.observation_status == "success":
            env = observed
    return _commands(env, "admissible_commands")


__all__ = [
    "ACT_FUNCTION",
    "DYNAMIC_ENUMS",
    "LEGAL_EVENT_SET_VERSION",
    "LegalEventSet",
    "current_admissible_commands",
    "environment_from_observation",
    "final_turn_rule",
    "horizon",
    "initial_environment",
    "legal_event_set",
]
