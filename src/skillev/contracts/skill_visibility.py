from __future__ import annotations

from typing import Final

NONEMPTY_ONLY: Final = "nonempty-only@1"
SKILL_VISIBILITY_RULES: Final = frozenset({NONEMPTY_ONLY})

__all__ = ["NONEMPTY_ONLY", "SKILL_VISIBILITY_RULES"]
