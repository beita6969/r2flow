from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Callable, Sequence


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value, flags=re.UNICODE)

    def remove_punc(value: str) -> str:
        return "".join(ch for ch in value if ch not in string.punctuation)

    return " ".join(remove_articles(remove_punc(text.lower())).split())


def get_tokens(text: str) -> list[str]:
    return normalize_answer(text).split() if text else []


def compute_exact(gold: str, prediction: str) -> float:
    return float(normalize_answer(gold) == normalize_answer(prediction))


def compute_f1(gold: str, prediction: str) -> float:
    gold_tokens, prediction_tokens = get_tokens(gold), get_tokens(prediction)
    common = Counter(gold_tokens) & Counter(prediction_tokens)
    same = sum(common.values())
    if not gold_tokens or not prediction_tokens:
        return float(gold_tokens == prediction_tokens)
    if same == 0:
        return 0.0
    precision = same / len(prediction_tokens)
    recall = same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(
    metric: Callable[[str, str], float], prediction: str, ground_truths: Sequence[str]
) -> float:
    return max(metric(prediction, answer) for answer in ground_truths)


class AnswerMetric:
    def __init__(self) -> None:
        self.reset()

    def __call__(self, predicted_answer: str, ground_truth_answers: Sequence[str]) -> None:
        self._total_em += metric_max_over_ground_truths(
            compute_exact, predicted_answer, ground_truth_answers
        )
        self._total_f1 += metric_max_over_ground_truths(
            compute_f1, predicted_answer, ground_truth_answers
        )
        self._count += 1

    def get_metric(self, reset: bool = False) -> tuple[float, float]:
        em = self._total_em / self._count if self._count else 0.0
        f1 = self._total_f1 / self._count if self._count else 0.0
        if reset:
            self.reset()
        return em, f1

    def reset(self) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0
