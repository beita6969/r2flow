from __future__ import annotations

from enum import StrEnum
from typing import Final


class TrainingBenchmark(StrEnum):
    HOTPOT_QA = "hotpotqa"
    TRIVIA_QA = "triviaqa"
    AIME_2026 = "aime-2026"
    HEALTHBENCH = "healthbench"
    ALF_WORLD = "alfworld"
    MBPP_PLUS = "mbpp-plus"


QUESTIONS_PER_DOMAIN: Final = 250
TRAJECTORIES_PER_QUESTION: Final = 4
TRAINING_STEPS: Final = QUESTIONS_PER_DOMAIN


__all__ = [
    "QUESTIONS_PER_DOMAIN",
    "TRAINING_STEPS",
    "TRAJECTORIES_PER_QUESTION",
    "TrainingBenchmark",
]
