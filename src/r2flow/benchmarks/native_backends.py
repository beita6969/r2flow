from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from skillev.contracts import JsonValue, normalize_json
from skillev.experiments.run_protocol import Benchmark
from skillev.rollout import (
    NoTerminalSubmission,
    TerminalEvaluationRequest,
    TerminalEvaluatorError,
)

from .static import PrivateStaticBenchmarkCase, StaticScoringRule
from .terminal_evaluator import NativeResult
from .terminal_inputs import submitted_value


def _submitted_answer(request: TerminalEvaluationRequest) -> str | None:
    if isinstance(request.evaluation_input, NoTerminalSubmission):
        return None
    submitted = normalize_json(submitted_value(request))
    if not isinstance(submitted, dict) or set(submitted) != {"answer"}:
        raise TerminalEvaluatorError("completion has incompatible fields")
    answer = submitted["answer"]
    if type(answer) is not str:
        raise TerminalEvaluatorError("completion answer must be text")
    return answer


@dataclass(frozen=True, slots=True)
class StaticBackend:
    case: PrivateStaticBenchmarkCase
    benchmark: Benchmark
    verifier_version: str
    environment_id: str | None = None

    def __post_init__(self) -> None:
        if self.case.public.benchmark_id != self.benchmark.value:
            raise ValueError("static case and Protocol 10 benchmark differ")
        expected = {
            Benchmark.HOTPOT_QA: StaticScoringRule.TOKEN_F1,
            Benchmark.TRIVIA_QA: StaticScoringRule.TOKEN_F1,
            Benchmark.AIME_2026: StaticScoringRule.INTEGER,
        }.get(self.benchmark)
        if expected is None or self.case.target.scoring_rule is not expected:
            raise ValueError("static backend does not implement this benchmark rule")
        if not self.verifier_version.strip():
            raise ValueError("static verifier version must be non-empty")
        if self.environment_id is not None and not self.environment_id.strip():
            raise ValueError("static environment override must be non-empty")

    async def evaluate_native(
        self,
        request: TerminalEvaluationRequest,
    ) -> NativeResult:
        if request.task_id != self.case.public.task_id:
            raise TerminalEvaluatorError("terminal request reached another static case")
        answer = _submitted_answer(request)
        if answer is None:
            score = exact = 0.0
        else:
            score = self.case.target.score(answer)
            exact = self.case.target.exact_match(answer)
        fields: dict[str, JsonValue]
        if self.benchmark in {Benchmark.HOTPOT_QA, Benchmark.TRIVIA_QA}:
            fields = {"answer-exact-match": exact, "answer-f1": score}
        else:
            fields = {"accuracy": score}
        return NativeResult(
            task_id=request.task_id,
            benchmark=self.benchmark,
            native_fields=fields,
            environment_id=self.environment_id or self.case.public.environment_id,
            verifier_version=self.verifier_version,
        )


@dataclass(frozen=True, slots=True)
class HealthBenchGrade:
    official_rubric_score: float
    triggered_negative_rubric_count: int
    grader_cost: dict[str, float] | None = None
    criterion_ledger: dict[str, JsonValue] | None = None
    refused_criterion_count: int | None = None

    def __post_init__(self) -> None:
        if self.refused_criterion_count is not None and (
            type(self.refused_criterion_count) is not int or self.refused_criterion_count < 0
        ):
            raise ValueError("HealthBench refused-criterion count must be non-negative")
        if (
            isinstance(self.official_rubric_score, bool)
            or not isinstance(self.official_rubric_score, int | float)
            or not math.isfinite(float(self.official_rubric_score))
        ):
            raise ValueError("HealthBench rubric score must be finite")
        if (
            type(self.triggered_negative_rubric_count) is not int
            or self.triggered_negative_rubric_count < 0
        ):
            raise ValueError("HealthBench negative-rubric count must be non-negative")


class HealthBenchOfficialGrader(Protocol):
    @property
    def verifier_version(self) -> str: ...

    async def grade(self, task_id: str, candidate_answer: str) -> HealthBenchGrade: ...


@dataclass(frozen=True, slots=True)
class HealthBenchNativeBackend:
    task_id: str
    environment_id: str
    grader: HealthBenchOfficialGrader

    async def evaluate_native(
        self,
        request: TerminalEvaluationRequest,
    ) -> NativeResult:
        if request.task_id != self.task_id:
            raise TerminalEvaluatorError("terminal request reached another HealthBench case")
        answer = _submitted_answer(request)
        if answer is None:
            grade = HealthBenchGrade(0.0, 0)
        else:
            try:
                grade = await self.grader.grade(self.task_id, answer)
            except Exception as error:
                raise TerminalEvaluatorError("HealthBench grader failed") from error
        return NativeResult(
            task_id=self.task_id,
            benchmark=Benchmark.HEALTHBENCH,
            native_fields={
                "official-rubric-score": float(grade.official_rubric_score),
                "triggered-negative-rubric-count": grade.triggered_negative_rubric_count,
            },
            environment_id=self.environment_id,
            verifier_version=self.grader.verifier_version,
        )


__all__ = [
    "HealthBenchGrade",
    "HealthBenchNativeBackend",
    "HealthBenchOfficialGrader",
    "StaticBackend",
]
