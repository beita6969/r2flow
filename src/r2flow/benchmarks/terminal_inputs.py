from __future__ import annotations

from skillev.contracts import JsonValue, SuccessRule, TerminalReward, normalize_json
from skillev.rollout import (
    NoTerminalSubmission,
    SubmittedTerminalValue,
    TerminalEvaluationRequest,
)


def submitted_value(request: TerminalEvaluationRequest) -> JsonValue:
    terminal_input = request.evaluation_input
    if not isinstance(terminal_input, SubmittedTerminalValue):
        raise TypeError("submitted_value requires SubmittedTerminalValue")
    return terminal_input.value


def no_submission_reward(
    request: TerminalEvaluationRequest,
    *,
    native_metric_name: str,
    native_payload: dict[str, JsonValue],
    environment_id: str,
    verifier_version: str,
) -> TerminalReward:
    terminal_input = request.evaluation_input
    if not isinstance(terminal_input, NoTerminalSubmission):
        raise TypeError("no_submission_reward requires NoTerminalSubmission")
    payload = normalize_json(
        {
            **native_payload,
            "no_submission_reason": terminal_input.reason.value,
        }
    )
    if not isinstance(payload, dict):
        raise TypeError("no-submission native payload must be an object")
    return TerminalReward(
        value=0.0,
        success=False,
        success_rule=SuccessRule.R_EQUALS_ONE,
        success_threshold=None,
        native_metric_name=native_metric_name,
        native_payload=payload,
        environment_id=environment_id,
        verifier_version=verifier_version,
    )


__all__ = ["no_submission_reward", "submitted_value"]
