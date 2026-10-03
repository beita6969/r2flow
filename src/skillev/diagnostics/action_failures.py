from __future__ import annotations

import json
import re

from skillev.contracts.action_wire import NATIVE_EVENT_CALL_WIRE


ACTION_FAILURE_FORMAT = "observed-action-outcomes@2"


def classify_action_outcome(
    text: str | None,
    *,
    parse_status: str | None,
    finish_reason: str | None,
    action_kind: str | None,
    completed: bool | None,
    observation_status: str | None,
    interface_matches: bool | None = None,
    action_wire: str | None = NATIVE_EVENT_CALL_WIRE,
    public_error_code: str | None = None,
    available_native_names: tuple[str, ...] | None = None,
    admitted: bool | None = None,
    executed: bool | None = None,
    terminal_success: bool | None = None,
    output_token_count: int | None = None,
) -> tuple[str, ...]:
    labels: list[str] = []
    if finish_reason == "length":
        labels.append("output-length-limit")
    if text is None:
        labels.append("raw-action-unavailable")
    elif not text.strip() or output_token_count == 0:
        labels.append("empty-stop" if finish_reason == "stop" else "empty-action-output")
    elif action_wire is None:
        labels.append("action-wire-unavailable")
    else:
        labels.extend(_native_labels(text, parse_status, available_native_names))
    if public_error_code in {
        "unsupported_resource",
        "unsupported_tool",
        "skill_not_available_in_h0",
    }:
        labels.append("action-not-in-current-surface")
    if public_error_code in {"invalid_arguments", "invalid_completion"}:
        labels.append("arguments-or-completion-invalid")
    if interface_matches is False:
        labels.append("declared-interface-mismatch")
    if parse_status == "valid" and action_kind == "complete":
        if completed is None:
            labels.append("completion-terminal-state-unavailable")
        elif completed is False:
            labels.append("complete-not-terminal")
    if executed is True and observation_status not in (None, "success", "completed"):
        labels.append(
            "admitted-execution-unsuccessful" if admitted is True else "execution-unsuccessful"
        )
    elif observation_status is not None and observation_status not in {"success", "completed"}:
        labels.append("execution-rejected-or-unsuccessful")
    if admitted is True and terminal_success is False:
        labels.append("admitted-action-in-terminal-failed-trajectory")
    if not labels:
        labels.append(
            "accepted-action"
            if parse_status == "valid" and observation_status in {"success", "completed"}
            else "outcome-evidence-unavailable"
        )
    return tuple(dict.fromkeys(labels))


def _json_errors(text: str) -> list[str]:
    try:
        json.loads(text)
    except json.JSONDecodeError as error:
        return [
            "json-string-escaping"
            if "escape" in error.msg or "control character" in error.msg
            else "json-syntax"
        ]
    return []


def _native_labels(text: str, status: str | None, names: tuple[str, ...] | None) -> list[str]:
    if status == "valid":
        return []
    raw = text.strip()
    labels = (
        ["native-carrier-invalid" if status == "parse-error" else "native-schema-mismatch"]
        if status in {"parse-error", "schema-invalid"}
        else []
    )
    if raw.startswith("<tool_call>"):
        if not raw.endswith("</tool_call>"):
            return [*labels, "unclosed-native-carrier"]
        raw = raw[len("<tool_call>") : -len("</tool_call>")].strip()
    name = None
    if raw.startswith("<function="):
        match = re.match(r"<function=([A-Za-z_][A-Za-z0-9_]*)>", raw)
        if match:
            name = match[1]
        if not raw.endswith("</function>"):
            labels.append("unclosed-native-function")
    elif raw.startswith(("{", "[")):
        labels.extend(_json_errors(raw))
        try:
            value = json.loads(raw)
            if isinstance(value, dict):
                name = value.get("name")
        except ValueError:
            pass
    elif status == "parse-error":
        labels.append("non-carrier-action-text")
    if isinstance(name, str) and names is not None and name not in names:
        labels.append("action-not-in-current-surface")
    return labels
