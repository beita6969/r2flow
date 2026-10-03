from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from skillev.runtime import FullRetrievedSkillContext, RolloutEnvironmentSession

from .environment import TerminalEvaluator

if TYPE_CHECKING:
    from skillev.verification import VerifierSuite


@dataclass(frozen=True, slots=True)
class UnskilledRolloutSessionBundle:
    environment: RolloutEnvironmentSession
    evaluator: TerminalEvaluator
    cleanup: Callable[[], Awaitable[None]] | None = None
    verifier: VerifierSuite | None = None


@dataclass(frozen=True, slots=True)
class RolloutSessionBundle:
    environment: RolloutEnvironmentSession
    evaluator: TerminalEvaluator
    retrieved_skills: tuple[FullRetrievedSkillContext, ...]
    cleanup: Callable[[], Awaitable[None]] | None = None
    verifier: VerifierSuite | None = None


__all__ = ["RolloutSessionBundle", "UnskilledRolloutSessionBundle"]
