from __future__ import annotations

from typing import Final

INTEGER_ANSWER_RULE: Final = "integer-answer@1"
INTEGER_ANSWER_PATTERN: Final = "^(0|[1-9][0-9]{0,2})$"
INTEGER_ANSWER_VALUES: Final[tuple[str, ...]] = tuple(str(value) for value in range(1000))

__all__ = ["INTEGER_ANSWER_PATTERN", "INTEGER_ANSWER_RULE", "INTEGER_ANSWER_VALUES"]
