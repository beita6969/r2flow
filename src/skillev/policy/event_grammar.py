from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

EVENT_GRAMMAR_VERSION = "event-grammar@1"
EVENT_IDENTITY = "event-token-path@1"
BUDGET_FORCED_CLOSING = "budget-forced-closing@1"

EX: tuple[str, ...] = (
    "<tool_call>",
    "</tool_call>",
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<think>",
    "</think>",
    "</parameter>",
)
OPEN = "<tool_call>\n<function="
OPEN_PREFIX = OPEN[:-1]
AFTER_NAME = ">\n"
ENUM_CLOSE = "\n</parameter>\n"
CALL_CLOSE = "</function>\n</tool_call>"
T = "\n</parameter>"
F = "\n</function>\n</tool_call>"
assert T + F == ENUM_CLOSE + CALL_CLOSE

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")


def key_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    )


class EventParseError(ValueError):
    pass


def _require_name(value: object, *, what: str) -> str:
    if type(value) is not str or _NAME.fullmatch(value) is None:
        raise ValueError(f"{what} must match [A-Za-z_][A-Za-z0-9_]{{0,63}}")
    return value


def _require_enum_value(value: object) -> str:
    if type(value) is not str or not value:
        raise ValueError("enum values must be non-empty text")
    if "\n" in value or any(item in value for item in EX):
        raise ValueError("enum values must not contain a newline or an excluded string")
    return value


@dataclass(frozen=True, slots=True)
class ParamSpec:
    name: str
    kind: Literal["enum", "text"]
    values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_name(self.name, what="parameter name")
        if self.kind == "text":
            if self.values != ():
                raise ValueError("text parameters carry no values")
            return
        if self.kind != "enum":
            raise ValueError("parameter kind must be 'enum' or 'text'")
        if type(self.values) is not tuple or not self.values:
            raise ValueError("enum parameters need a non-empty tuple of values")
        for value in self.values:
            _require_enum_value(value)
        if len(set(self.values)) != len(self.values):
            raise ValueError("enum values must be unique")
        object.__setattr__(self, "values", tuple(sorted(self.values)))


@dataclass(frozen=True, slots=True)
class FunctionSpec:
    name: str
    params: tuple[ParamSpec, ...]

    def __post_init__(self) -> None:
        _require_name(self.name, what="function name")
        if type(self.params) is not tuple or any(
            not isinstance(param, ParamSpec) for param in self.params
        ):
            raise ValueError("function parameters must be a tuple of ParamSpec")
        names = [param.name for param in self.params]
        if len(set(names)) != len(names):
            raise ValueError("duplicate parameter name")
        text = [index for index, param in enumerate(self.params) if param.kind == "text"]
        if len(text) > 1:
            raise ValueError("at most one text parameter")
        if text and text[0] != len(self.params) - 1:
            raise ValueError("the text parameter must be last")


@dataclass(frozen=True, slots=True)
class EventGrammarSpec:
    version: str
    functions: tuple[FunctionSpec, ...]
    budget: int
    stop_token_id: int

    def __post_init__(self) -> None:
        if self.version != EVENT_GRAMMAR_VERSION:
            raise ValueError("unsupported event grammar version")
        if type(self.functions) is not tuple or not self.functions:
            raise ValueError("an event grammar needs at least one function")
        if any(not isinstance(function, FunctionSpec) for function in self.functions):
            raise ValueError("functions must be FunctionSpec")
        names = [function.name for function in self.functions]
        if len(set(names)) != len(names):
            raise ValueError("duplicate function name")
        if type(self.budget) is not int or self.budget < 1:
            raise ValueError("budget must be a positive integer")
        if type(self.stop_token_id) is not int or self.stop_token_id < 0:
            raise ValueError("stop token id must be a non-negative integer")
        object.__setattr__(
            self, "functions", tuple(sorted(self.functions, key=lambda item: item.name))
        )

    def function(self, name: str) -> FunctionSpec:
        for function in self.functions:
            if function.name == name:
                return function
        raise ValueError(f"function {name!r} is not legal")

    def to_value(self) -> dict[str, object]:
        return {
            "version": self.version,
            "budget": self.budget,
            "stop_token_id": self.stop_token_id,
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


def render_event_call(spec: EventGrammarSpec, function: str, args: Mapping[str, str]) -> str:
    declared = spec.function(function)
    if set(args) != {param.name for param in declared.params}:
        raise ValueError("arguments must name exactly the declared parameters")
    parts = [OPEN, declared.name, AFTER_NAME]
    for param in declared.params:
        value = args[param.name]
        if type(value) is not str:
            raise ValueError("argument values must be text")
        if param.kind == "enum":
            if value not in param.values:
                raise ValueError(f"{value!r} is not a legal {param.name}")
        elif any(item in value for item in EX):
            raise ValueError("text values must not contain an excluded string")
        parts.extend((f"<parameter={param.name}>\n", value, ENUM_CLOSE))
    parts.append(CALL_CLOSE)
    return "".join(parts)


def parse_event_call(spec: EventGrammarSpec, text: str) -> tuple[str, dict[str, str]]:
    if type(text) is not str or not text.startswith(OPEN):
        raise EventParseError("call must start with the canonical opening")
    position = len(OPEN)
    end = text.find(AFTER_NAME, position)
    if end < 0:
        raise EventParseError("function name is not terminated")
    name = text[position:end]
    try:
        declared = spec.function(name)
    except ValueError as error:
        raise EventParseError(str(error)) from None
    position = end + len(AFTER_NAME)
    args: dict[str, str] = {}
    for param in declared.params:
        header = f"<parameter={param.name}>\n"
        if not text.startswith(header, position):
            raise EventParseError(f"expected parameter {param.name!r}")
        position += len(header)
        if param.kind == "enum":
            matched = [
                value for value in param.values if text.startswith(value + ENUM_CLOSE, position)
            ]
            if len(matched) != 1:
                raise EventParseError(f"illegal value for {param.name!r}")
            args[param.name] = matched[0]
            position += len(matched[0]) + len(ENUM_CLOSE)
            continue
        close = text.find("</parameter>", position)
        if close < 1 or close - 1 < position or text[close - 1] != "\n":
            raise EventParseError("text value is not closed by the framing terminator")
        value = text[position : close - 1]
        if any(item in value for item in EX):
            raise EventParseError("text value contains an excluded string")
        if not text.startswith(ENUM_CLOSE, close - 1):
            raise EventParseError("text value terminator is not canonical")
        args[param.name] = value
        position = close - 1 + len(ENUM_CLOSE)
    if text[position:] != CALL_CLOSE:
        raise EventParseError("call must end with the canonical close and nothing after it")
    return declared.name, args


def _const(value: str) -> dict[str, object]:
    return {"type": "const_string", "value": value}


def _param_elements(param: ParamSpec) -> list[dict[str, object]]:
    header = _const(f"<parameter={param.name}>\n")
    if param.kind == "enum":
        values = [_const(value) for value in param.values]
        choice = values[0] if len(values) == 1 else {"type": "or", "elements": values}
        return [header, choice, _const(ENUM_CLOSE)]
    return [
        header,
        {
            "type": "tag",
            "begin": "",
            "content": {"type": "any_text", "excludes": list(EX)},
            "end": ENUM_CLOSE,
        },
    ]


def to_structural_tag(spec: EventGrammarSpec) -> dict[str, object]:
    calls: list[dict[str, object]] = []
    for function in spec.functions:
        elements: list[dict[str, object]] = [_const(function.name + AFTER_NAME)]
        for param in function.params:
            elements.extend(_param_elements(param))
        elements.append(_const(CALL_CLOSE))
        calls.append({"type": "sequence", "elements": elements})
    body = calls[0] if len(calls) == 1 else {"type": "or", "elements": calls}
    return {"type": "sequence", "elements": [_const(OPEN), body]}


def _encode(encode: Callable[[str], Sequence[int]], text: str) -> list[int]:
    ids = [int(token) for token in encode(text)] if text else []
    if text and not ids:
        raise ValueError(f"encoding of {text!r} is empty")
    if any(token < 0 for token in ids):
        raise ValueError("token ids must be non-negative")
    return ids


def _suffix_consistent(encode: Callable[[str], Sequence[int]], text: str) -> list[list[int]]:
    closing = [_encode(encode, text[index:]) for index in range(len(text))]
    for index, ids in enumerate(closing):
        rest = ids[1:]
        if not any(
            (closing[index + step] if index + step < len(text) else []) == rest
            for step in range(1, len(text) - index + 1)
        ):
            raise ValueError(f"closing encodings of {text!r} are not suffix-consistent")
    return closing


def _open_and_name_tokens(
    spec: EventGrammarSpec, encode: Callable[[str], Sequence[int]]
) -> tuple[dict[str, list[int]], list[int]]:
    joints = {function.name: _encode(encode, OPEN + function.name) for function in spec.functions}
    rows = list(joints.values())
    common = 0
    while all(len(row) > common for row in rows) and len({row[common] for row in rows}) == 1:
        common += 1
    while common and any(len(row) <= common for row in rows):
        common -= 1
    return joints, rows[0][:common]


def noncanonical_name_segments(
    spec: EventGrammarSpec, encode: Callable[[str], Sequence[int]]
) -> tuple[str, ...]:
    plan = build_token_plan(spec, encode)
    open_ids = list(cast(Sequence[int], plan["open"]))
    bad: list[str] = []
    for function, row in zip(
        spec.functions, cast(Sequence[Mapping[str, object]], plan["functions"]), strict=True
    ):
        program = cast(Sequence[Sequence[object]], row["program"])
        first_text = AFTER_NAME + (
            f"<parameter={function.params[0].name}>\n" if function.params else CALL_CLOSE
        )
        segmented = (
            open_ids
            + list(cast(Sequence[int], row["name_ids"]))
            + list(cast(Sequence[int], program[0][1]))
        )
        if segmented != _encode(encode, OPEN + function.name + first_text):
            bad.append(function.name)
    return tuple(bad)


def build_token_plan(
    spec: EventGrammarSpec, encode: Callable[[str], Sequence[int]]
) -> dict[str, object]:
    joints, open_ids = _open_and_name_tokens(spec, encode)
    functions: list[dict[str, object]] = []
    names: set[tuple[int, ...]] = set()
    for function in spec.functions:
        program: list[list[object]] = []
        run = AFTER_NAME
        ends_in_text = False
        for param in function.params:
            run += f"<parameter={param.name}>\n"
            program.append(["forced", _encode(encode, run)])
            if param.kind == "enum":
                choices = [_encode(encode, value) for value in param.values]
                if len({tuple(ids) for ids in choices}) != len(choices):
                    raise ValueError("enum value encodings collide")
                program.append(["choice", choices])
                run = ENUM_CLOSE
            else:
                program.append(["text"])
                ends_in_text = True
        if not ends_in_text:
            program.append(["forced", _encode(encode, run + CALL_CLOSE)])
        name_ids = joints[function.name][len(open_ids) :]
        if tuple(name_ids) in names:
            raise ValueError("function name encodings collide")
        names.add(tuple(name_ids))
        functions.append({"name": function.name, "name_ids": name_ids, "program": program})
    return {
        "open": open_ids,
        "functions": functions,
        "terminator_closing": _suffix_consistent(encode, T),
        "tail_closing": _suffix_consistent(encode, F),
        "stop": [spec.stop_token_id],
    }


def _program_minimum(program: Sequence[Sequence[object]], plan: Mapping[str, object]) -> int:
    total = 0
    for step in program:
        kind = step[0]
        if kind == "forced":
            total += len(step[1])
        elif kind == "choice":
            total += min(len(ids) for ids in step[1])
        elif kind == "text":
            terminator = plan["terminator_closing"]
            tail = plan["tail_closing"]
            total += len(terminator[0]) + len(tail[0])
        else:
            raise ValueError(f"unknown program step {kind!r}")
    return total


def minimum_closing_tokens(plan: Mapping[str, object]) -> int:
    functions = plan["functions"]
    best = min(
        len(item["name_ids"]) + _program_minimum(item["program"], plan) for item in functions
    )
    return len(plan["open"]) + best + 1


def require_budget(spec: EventGrammarSpec, plan: Mapping[str, object]) -> None:
    if spec.budget < minimum_closing_tokens(plan):
        raise ValueError("action budget below shortest legal event")


def event_grammar_key(
    spec: EventGrammarSpec, plan: Mapping[str, object], tokenizer_info_sha256: str
) -> str:
    if (
        type(tokenizer_info_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", tokenizer_info_sha256) is None
    ):
        raise ValueError("tokenizer_info_sha256 must be a sha256 hex digest")
    if plan.get("stop") != [spec.stop_token_id]:
        raise ValueError("plan stop token differs from the spec")
    require_budget(spec, plan)
    return key_json(
        {
            "type": EVENT_GRAMMAR_VERSION,
            "closing": BUDGET_FORCED_CLOSING,
            "format": to_structural_tag(spec),
            "plan": dict(plan),
            "budget": spec.budget,
            "tokenizer_info_sha256": tokenizer_info_sha256,
        }
    )


@dataclass(frozen=True, slots=True)
class EventGrammarKey:
    text: str
    format: dict[str, object]
    plan: dict[str, object]
    budget: int
    tokenizer_info_sha256: str

    @property
    def sha256(self) -> str:
        return event_grammar_sha256(self.text)

    @property
    def stop_token_id(self) -> int:
        stop = self.plan["stop"]
        return int(stop[0])

    def structural_tag_json(self) -> str:
        return json.dumps({"type": "structural_tag", "format": self.format})


def parse_event_grammar_key(key: str) -> EventGrammarKey:
    if type(key) is not str:
        raise ValueError("event grammar key must be text")
    try:
        value = json.loads(key)
    except json.JSONDecodeError as error:
        raise ValueError("event grammar key is not JSON") from error
    if not isinstance(value, dict) or value.get("type") != EVENT_GRAMMAR_VERSION:
        raise ValueError("not an event-grammar@1 key")
    if set(value) != {"type", "closing", "format", "plan", "budget", "tokenizer_info_sha256"}:
        raise ValueError("event grammar key has an incompatible field set")
    if value["closing"] != BUDGET_FORCED_CLOSING:
        raise ValueError("unsupported closing rule")
    if key_json(value) != key:
        raise ValueError("event grammar key is not canonical JSON")
    budget = value["budget"]
    plan = value["plan"]
    fmt = value["format"]
    digest = value["tokenizer_info_sha256"]
    if type(budget) is not int or budget < 1:
        raise ValueError("event grammar budget is invalid")
    if not isinstance(plan, dict) or not isinstance(fmt, dict) or type(digest) is not str:
        raise ValueError("event grammar key fields are invalid")
    if budget < minimum_closing_tokens(plan):
        raise ValueError("action budget below shortest legal event")
    return EventGrammarKey(key, fmt, plan, budget, digest)


def event_grammar_sha256(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


__all__ = [
    "AFTER_NAME",
    "BUDGET_FORCED_CLOSING",
    "CALL_CLOSE",
    "ENUM_CLOSE",
    "EVENT_GRAMMAR_VERSION",
    "EVENT_IDENTITY",
    "EX",
    "OPEN",
    "EventGrammarKey",
    "EventGrammarSpec",
    "EventParseError",
    "F",
    "FunctionSpec",
    "ParamSpec",
    "T",
    "build_token_plan",
    "event_grammar_key",
    "event_grammar_sha256",
    "key_json",
    "minimum_closing_tokens",
    "parse_event_call",
    "parse_event_grammar_key",
    "render_event_call",
    "require_budget",
    "to_structural_tag",
]
