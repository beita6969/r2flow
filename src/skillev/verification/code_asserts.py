from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal, Protocol

PUBLIC_ASSERT_TIMEOUT_SECONDS = 10.0
_FENCE = re.compile(r"```[ \t]*(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


class PublicAssertInfrastructureError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PublicAssertRequest:
    code: str
    asserts: tuple[str, ...]
    timeout_s: float = PUBLIC_ASSERT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if type(self.code) is not str or not self.code.strip():
            raise ValueError("public-assert request needs candidate code")
        if (
            type(self.asserts) is not tuple
            or not self.asserts
            or any(
                type(item) is not str or not item.lstrip().startswith("assert ")
                for item in self.asserts
            )
        ):
            raise ValueError("public-assert request needs assert statements")
        if (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, int | float)
            or not math.isfinite(self.timeout_s)
            or self.timeout_s <= 0
        ):
            raise ValueError("timeout_s must be positive")


PublicAssertStatus = Literal["pass", "fail", "timeout", "error"]


@dataclass(frozen=True, slots=True)
class PublicAssertResult:
    status: PublicAssertStatus
    failed_index: int | None = None
    elapsed_ms: int = 0

    def __post_init__(self) -> None:
        if self.status not in ("pass", "fail", "timeout", "error"):
            raise ValueError("unsupported public-assert status")


class PublicAssertBackend(Protocol):
    async def run(self, request: PublicAssertRequest) -> PublicAssertResult: ...


def public_asserts(query: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in query.split("\n") if line.strip().startswith("assert "))


def extract_code(output: str) -> str | None:
    blocks = [block for block in _FENCE.findall(output) if block.strip()]
    return blocks[-1] if blocks else None


__all__ = [
    "PUBLIC_ASSERT_TIMEOUT_SECONDS",
    "PublicAssertBackend",
    "PublicAssertInfrastructureError",
    "PublicAssertRequest",
    "PublicAssertResult",
    "extract_code",
    "public_asserts",
]
