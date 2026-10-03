from dataclasses import asdict, dataclass
from typing import cast

from skillev.contracts import JsonValue
from skillev.runtime.execution import ActionParseResult, ActionParseStatus

OUTCOME_FORMAT = "action-submission-outcome@2"


@dataclass(frozen=True, slots=True)
class ActionSubmissionOutcome:
    finish_reason: str
    raw_output_token_count: int
    carrier_status: str
    action_token_cap: int
    controller_turns_remaining: int
    public_error_code: str | None
    response_received: bool = True
    admitted: bool | None = None
    executed: bool | None = None
    execution_status: str | None = None
    reason: str | None = None
    reason_detail: tuple[str, ...] = ()

    @classmethod
    def observe(
        cls,
        parsed: ActionParseResult,
        *,
        finish_reason: str,
        output_tokens: int,
        action_token_cap: int,
        turns_remaining: int,
    ) -> "ActionSubmissionOutcome":
        if parsed.status is not ActionParseStatus.VALID:
            raise ValueError("the declared codec is the sole authority for admission")
        return cls(
            finish_reason,
            output_tokens,
            "unique-complete",
            action_token_cap,
            max(0, turns_remaining),
            parsed.public_error_code,
        )

    def to_value(self) -> dict[str, JsonValue]:
        value = cast(dict[str, JsonValue], asdict(self))
        value["reason_detail"] = list(self.reason_detail)
        return {"format": OUTCOME_FORMAT, **value}
