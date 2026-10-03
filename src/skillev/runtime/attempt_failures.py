from __future__ import annotations

from .attempt_protocol import AttemptFailureCode, AttemptFailureStage


class AttemptDomainError(RuntimeError):
    def __init__(
        self,
        *,
        code: AttemptFailureCode,
        stage: AttemptFailureStage,
        private_detail: str,
    ) -> None:
        if not isinstance(code, AttemptFailureCode):
            raise TypeError("attempt domain failure code must be AttemptFailureCode")
        if not isinstance(stage, AttemptFailureStage):
            raise TypeError("attempt domain failure stage must be AttemptFailureStage")
        if type(private_detail) is not str or not private_detail:
            raise ValueError("attempt domain failure requires private detail")
        super().__init__(private_detail)
        self.code = code
        self.stage = stage


class EventAppendFailedError(AttemptDomainError):
    def __init__(self, private_detail: str) -> None:
        super().__init__(
            code=AttemptFailureCode.EVENT_APPEND_FAILED,
            stage=AttemptFailureStage.EXECUTION,
            private_detail=private_detail,
        )


__all__ = [
    "AttemptDomainError",
    "EventAppendFailedError",
]
