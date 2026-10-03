from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from skillev.benchmarks import (
    BenchmarkPublicItem,
    CompletionBenchmarkEnvironment,
    QABenchmark,
    QARetrievalEnvironment,
    RetrievalIndex,
)
from skillev.contracts import JsonValue, SuccessRule, TerminalReward, normalize_json
from skillev.rollout import (
    NoTerminalSubmission,
    TerminalEvaluationRequest,
    TerminalEvaluatorError,
)
from skillev.training import RolloutSessionBundle

from .terminal_inputs import no_submission_reward, submitted_value

_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_AIME_CANONICAL = re.compile(r"[0-9]{1,3}\Z")


class StaticScoringRule(StrEnum):
    EXACT_TEXT = "exact-text"
    TOKEN_F1 = "token-f1"
    OPTION = "option"
    INTEGER = "integer"


_FORMAL_STATIC_METRIC_NAMES: dict[str, tuple[StaticScoringRule, str]] = {
    "aime-2026": (StaticScoringRule.INTEGER, "accuracy"),
    "gpqa-diamond": (StaticScoringRule.OPTION, "accuracy"),
    "hotpotqa": (StaticScoringRule.TOKEN_F1, "token-f1"),
    "medqa": (StaticScoringRule.OPTION, "accuracy"),
    "musique": (StaticScoringRule.TOKEN_F1, "token-f1"),
    "nq-open": (StaticScoringRule.EXACT_TEXT, "exact-match"),
    "triviaqa": (StaticScoringRule.TOKEN_F1, "token-f1"),
}


def _normalize_text(value: str) -> str:
    lowered = value.lower()
    no_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    no_articles = _ARTICLES.sub(" ", no_punctuation)
    return _WHITESPACE.sub(" ", no_articles).strip()


def normalize_hotpotqa_answer(value: str) -> str:
    return _normalize_text(value)


def normalize_triviaqa_answer(value: str) -> str:
    return _normalize_text(value.replace("_", " "))


def parse_aime_answer(value: str) -> str | None:
    stripped = value.strip()
    if _AIME_CANONICAL.fullmatch(stripped) is None:
        return None
    return str(int(stripped))


def _token_f1(
    prediction: str,
    target: str,
    *,
    normalizer: Callable[[str], str] = _normalize_text,
) -> float:
    predicted = normalizer(prediction).split()
    expected = normalizer(target).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2.0 * precision * recall / (precision + recall)


def _native_metric_name(case: PrivateStaticBenchmarkCase) -> str:
    declared = _FORMAL_STATIC_METRIC_NAMES.get(case.public.benchmark_id)
    if declared is None:
        return case.target.scoring_rule.value
    expected_rule, metric_name = declared
    if case.target.scoring_rule is not expected_rule:
        raise ValueError("formal static benchmark uses another scoring rule")
    return metric_name


@dataclass(frozen=True, slots=True)
class PrivateStaticTarget:
    task_id: str
    scoring_rule: StaticScoringRule
    accepted_answers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.task_id or not self.accepted_answers:
            raise ValueError("private static target requires identity and answers")
        if any(type(answer) is not str or not answer.strip() for answer in self.accepted_answers):
            raise ValueError("private static answers must be non-empty text")
        if len(set(self.accepted_answers)) != len(self.accepted_answers):
            raise ValueError("private static answers must be unique")

    def score(self, prediction: str) -> float:
        if self.scoring_rule is StaticScoringRule.TOKEN_F1:
            normalizer = (
                normalize_triviaqa_answer
                if self.task_id.startswith("triviaqa/")
                else normalize_hotpotqa_answer
            )
            return max(
                _token_f1(prediction, answer, normalizer=normalizer)
                for answer in self.accepted_answers
            )
        if self.scoring_rule is StaticScoringRule.INTEGER:
            predicted = parse_aime_answer(prediction)
            try:
                expected = {
                    str(int(answer.strip()))
                    for answer in self.accepted_answers
                    if 0 <= int(answer.strip()) <= 999
                }
            except ValueError:
                return 0.0
            if predicted is None or len(expected) != len(self.accepted_answers):
                return 0.0
            return float(predicted in expected)
        normalized = _normalize_text(prediction)
        expected = {_normalize_text(answer) for answer in self.accepted_answers}
        return float(normalized in expected)

    def exact_match(self, prediction: str) -> float:
        normalizer = (
            normalize_triviaqa_answer
            if self.task_id.startswith("triviaqa/")
            else normalize_hotpotqa_answer
        )
        normalized = normalizer(prediction)
        expected = {normalizer(answer) for answer in self.accepted_answers}
        return float(normalized in expected)


@dataclass(frozen=True, slots=True)
class PrivateStaticBenchmarkCase:
    public: BenchmarkPublicItem
    target: PrivateStaticTarget

    def __post_init__(self) -> None:
        if self.public.task_id != self.target.task_id:
            raise ValueError("public benchmark item and private target identities differ")


@dataclass(slots=True)
class PrivateStaticBenchmarkEvaluator:
    case: PrivateStaticBenchmarkCase
    environment_id: str | None = None

    def __post_init__(self) -> None:
        if self.environment_id is not None and not self.environment_id.strip():
            raise ValueError("evaluator environment identity must be non-empty")

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        if request.task_id != self.case.public.task_id:
            raise TerminalEvaluatorError("terminal request reached a different benchmark case")
        native_metric_name = _native_metric_name(self.case)
        environment_id = self.environment_id or self.case.public.environment_id
        if isinstance(request.evaluation_input, NoTerminalSubmission):
            no_submission_metrics: dict[str, JsonValue] = {}
            if self.case.target.scoring_rule is StaticScoringRule.TOKEN_F1:
                no_submission_metrics["exact-match"] = 0.0
            return no_submission_reward(
                request,
                native_metric_name=native_metric_name,
                native_payload={
                    "benchmark_id": self.case.public.benchmark_id,
                    "public_metrics": no_submission_metrics,
                    "split": self.case.public.split,
                },
                environment_id=environment_id,
                verifier_version="static-benchmark-evaluator@1",
            )
        submission = normalize_json(submitted_value(request))
        if not isinstance(submission, dict) or set(submission) != {"answer"}:
            raise TerminalEvaluatorError("admitted completion has incompatible fields")
        prediction = submission["answer"]
        if type(prediction) is not str:
            raise TerminalEvaluatorError("admitted completion answer must be text")
        score = self.case.target.score(prediction)
        public_metrics: dict[str, JsonValue] = {}
        if self.case.target.scoring_rule is StaticScoringRule.TOKEN_F1:
            public_metrics = {"exact-match": self.case.target.exact_match(prediction)}
        native_payload: dict[str, JsonValue] = {
            "benchmark_id": self.case.public.benchmark_id,
            "public_metrics": public_metrics,
            "split": self.case.public.split,
        }
        return TerminalReward(
            value=score,
            success=score == 1.0,
            success_rule=SuccessRule.R_EQUALS_ONE,
            success_threshold=None,
            native_metric_name=native_metric_name,
            native_payload=native_payload,
            environment_id=environment_id,
            verifier_version="static-benchmark-evaluator@1",
        )


@dataclass(slots=True)
class PrivateStaticBenchmarkSessionFactory:
    cases: tuple[PrivateStaticBenchmarkCase, ...]

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("private benchmark session factory requires cases")
        task_ids = tuple(case.public.task_id for case in self.cases)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("private benchmark cases must have unique task identities")

    def create(self, task: object) -> RolloutSessionBundle:
        task_id = getattr(task, "task_id", None)
        matches = tuple(case for case in self.cases if case.public.task_id == task_id)
        if len(matches) != 1:
            raise ValueError("public task has no unique private benchmark case")
        case = matches[0]
        if task != case.public.to_rollout_task():
            raise ValueError("public task projection differs from its private case")
        return RolloutSessionBundle(
            environment=CompletionBenchmarkEnvironment(case.public),
            evaluator=PrivateStaticBenchmarkEvaluator(case),
            retrieved_skills=(),
        )


@dataclass(slots=True)
class PrivateRetrievalBenchmarkSessionFactory:
    cases: tuple[PrivateStaticBenchmarkCase, ...]
    index: RetrievalIndex
    benchmark: QABenchmark

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("private retrieval session factory requires cases")
        if not isinstance(self.index, RetrievalIndex):
            raise TypeError("index must be a RetrievalIndex")
        if not isinstance(self.benchmark, QABenchmark):
            raise TypeError("benchmark must be a QABenchmark")
        task_ids = tuple(case.public.task_id for case in self.cases)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("private retrieval cases must have unique task identities")
        if any(case.public.benchmark_id != self.benchmark.value for case in self.cases):
            raise ValueError("private retrieval cases do not match the benchmark identity")

    def create(self, task: object) -> RolloutSessionBundle:
        task_id = getattr(task, "task_id", None)
        matches = tuple(case for case in self.cases if case.public.task_id == task_id)
        if len(matches) != 1:
            raise ValueError("public task has no unique private retrieval case")
        case = matches[0]
        expected_task = case.public.to_retrieval_rollout_task(self.index.manifest)
        if task != expected_task:
            raise ValueError("public retrieval task projection differs from its private case")
        return RolloutSessionBundle(
            environment=QARetrievalEnvironment(
                index=self.index,
                benchmark=self.benchmark,
                dataset_revision=case.public.dataset_revision,
                task_family=case.public.task_family,
            ),
            evaluator=PrivateStaticBenchmarkEvaluator(
                case,
                environment_id=expected_task.environment_id,
            ),
            retrieved_skills=(),
        )


__all__ = [
    "PrivateRetrievalBenchmarkSessionFactory",
    "PrivateStaticBenchmarkCase",
    "PrivateStaticBenchmarkEvaluator",
    "PrivateStaticBenchmarkSessionFactory",
    "PrivateStaticTarget",
    "StaticScoringRule",
    "normalize_hotpotqa_answer",
    "normalize_triviaqa_answer",
    "parse_aime_answer",
]
