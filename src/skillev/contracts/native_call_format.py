from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Final

from .canonical import JsonValue, canonical_json
from .integer_answer import INTEGER_ANSWER_PATTERN

NATIVE_CALL_FORMAT_VERSION: Final = "qwen-xml-tool-call@1"
NATIVE_XML_TASK_SENTENCE: Final = (
    "Only a tool call in the Qwen XML format is executed: <tool_call><function=NAME>"
    "<parameter=FIELD>VALUE</parameter></function></tool_call>. The action-phase reply is "
    "that one call and nothing else; JSON objects, code fences and described calls are not "
    "executed."
)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}")


def public_identifier(value: object) -> str | None:
    return value if isinstance(value, str) and _IDENTIFIER.fullmatch(value) else None


def _layout(name: str, values: Sequence[tuple[str, str]]) -> str:
    parts = ["<tool_call>", f"<function={name}>"]
    for key, value in values:
        parts += [f"<parameter={key}>", value, "</parameter>"]
    parts += ["</function>", "</tool_call>"]
    return "\n".join(parts)


def render_native_call(
    name: str,
    properties: Mapping[str, Mapping[str, JsonValue]],
    arguments: Mapping[str, JsonValue],
) -> str:
    if public_identifier(name) is None:
        raise ValueError("declared function names are identifiers")
    if not set(arguments) <= set(properties):
        raise ValueError("arguments differ from the declared function")
    values: list[tuple[str, str]] = []
    for key, schema in properties.items():
        if key not in arguments:
            continue
        value, kind = arguments[key], schema.get("type")
        literal = isinstance(value, str) and (
            kind == "string" or (isinstance(kind, list) and "string" in kind and value != "null")
        )
        values.append((key, value if isinstance(value, str) and literal else canonical_json(value)))
    return _layout(name, values)


def _function(tool: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    function = tool.get("function")
    if not isinstance(function, dict) or public_identifier(function.get("name")) is None:
        raise ValueError("a declared tool needs a named function")
    parameters = function.get("parameters")
    if not isinstance(parameters, dict) or not isinstance(parameters.get("properties"), dict):
        raise ValueError("a declared function needs an object parameter schema")
    return function


def _properties(function: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    return properties


def _required(function: Mapping[str, JsonValue]) -> tuple[str, ...]:
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    properties = _properties(function)
    required = parameters.get("required", list(properties))
    if not isinstance(required, list):
        raise ValueError("required fields must be a list")
    return tuple(key for key in properties if key in required)


def tool_name(tool: Mapping[str, JsonValue]) -> str:
    name = _function(tool)["name"]
    assert isinstance(name, str)
    return name


def primary_tool(tools: Sequence[Mapping[str, JsonValue]]) -> Mapping[str, JsonValue]:
    if not tools:
        raise ValueError("the action phase declares no functions")
    return tools[0]


def _placeholder(key: str, schema: JsonValue) -> str:
    if isinstance(schema, dict) and "const" in schema:
        return canonical_json(schema["const"])
    return key.upper() + "_VALUE"


def call_example_values(tool: Mapping[str, JsonValue]) -> tuple[tuple[str, str], ...]:
    function = _function(tool)
    properties = _properties(function)
    return tuple((key, _placeholder(key, properties[key])) for key in _required(function))


def render_xml_call_example(tool: Mapping[str, JsonValue]) -> str:
    return _layout(tool_name(tool), call_example_values(tool))


def _describe(schema: JsonValue) -> str:
    if not isinstance(schema, dict):
        return "JSON value"
    if "const" in schema:
        return "exactly " + canonical_json(schema["const"])
    if schema.get("pattern") == INTEGER_ANSWER_PATTERN:
        return "integer 0..999 written as plain digits"
    enum = schema.get("enum")
    if isinstance(enum, list):
        return "one of: " + ", ".join(
            value if isinstance(value, str) else canonical_json(value) for value in enum
        )
    kind = schema.get("type")
    text = (
        kind
        if isinstance(kind, str)
        else " or ".join(str(item) for item in kind)
        if isinstance(kind, list)
        else "JSON value"
    )
    low, high = schema.get("minimum"), schema.get("maximum")
    if low is not None and high is not None:
        text += f" {canonical_json(low)}..{canonical_json(high)}"
    elif low is not None:
        text += f" >= {canonical_json(low)}"
    elif high is not None:
        text += f" <= {canonical_json(high)}"
    return text


def render_function_summary(
    tool: Mapping[str, JsonValue], *, with_description: bool = False
) -> str:
    function = _function(tool)
    properties = _properties(function)
    required = _required(function)
    fields = "; ".join(
        f"{key} ({_describe(schema)}; {'required' if key in required else 'optional'})"
        for key, schema in properties.items()
    )
    text = f"{tool_name(tool)}: " + (f"parameters {fields}." if fields else "no parameters.")
    description = function.get("description")
    if with_description and isinstance(description, str) and description:
        text += " " + description
    return text
