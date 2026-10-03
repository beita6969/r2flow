from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import regex

from .musique_answer import AnswerMetric
from .musique_answer import normalize_answer as _musique_normalize

_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_HOTPOT_SPECIAL = frozenset({"yes", "no", "noanswer"})


def _squad_like(value: str) -> str:
    lowered = value.lower()
    no_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    return " ".join(_ARTICLES.sub(" ", no_punctuation).split())


def normalize_hotpotqa_answer(value: str) -> str:
    return _squad_like(value)


def normalize_triviaqa_answer(value: str) -> str:
    punctuation = string.punctuation + "\u2018\u2019\u00b4" + chr(96)
    text = "".join(" " if character in punctuation else character for character in value.lower())
    return " ".join(_ARTICLES.sub(" ", text).split())


def normalize_musique_answer(value: str) -> str:
    return _musique_normalize(value)


def normalize_nq_open_answer(value: str) -> str:
    text = "".join(ch for ch in value.lower() if ch not in string.punctuation)
    return " ".join(regex.sub(r"\b(a|an|the)\b", " ", text).split())


@dataclass(frozen=True, slots=True)
class QAMetrics:
    em: float
    f1: float


def score_musique_answers(prediction: str, accepted_answers: tuple[str, ...]) -> QAMetrics:
    if not accepted_answers:
        raise ValueError("QA scorer requires accepted answers")
    metric = AnswerMetric()
    metric(prediction, accepted_answers)
    return QAMetrics(*metric.get_metric())


def score_qa_answers(
    prediction: str,
    accepted_answers: tuple[str, ...],
    *,
    normalizer: Callable[[str], str],
) -> QAMetrics:
    if not accepted_answers:
        raise ValueError("QA scorer requires accepted answers")
    predicted = normalizer(prediction)
    predicted_tokens = predicted.split()
    results: list[QAMetrics] = []
    for answer in accepted_answers:
        expected = normalizer(answer)
        expected_tokens = expected.split()
        em = float(predicted == expected)
        if not predicted_tokens or not expected_tokens:
            f1 = 0.0
        else:
            overlap = sum((Counter(predicted_tokens) & Counter(expected_tokens)).values())
            if overlap == 0:
                f1 = 0.0
            else:
                precision = overlap / len(predicted_tokens)
                recall = overlap / len(expected_tokens)
                f1 = 2.0 * precision * recall / (precision + recall)
        results.append(QAMetrics(em, f1))
    return QAMetrics(
        em=max(result.em for result in results),
        f1=max(result.f1 for result in results),
    )


def best_alias_metrics(
    prediction: str,
    aliases: tuple[str, ...],
    *,
    normalize: Callable[[str], str],
) -> QAMetrics:
    return score_qa_answers(prediction, aliases, normalizer=normalize)


def score_hotpotqa_answers(
    prediction: str,
    accepted_answers: tuple[str, ...],
) -> QAMetrics:
    if not accepted_answers:
        raise ValueError("HotpotQA scorer requires accepted answers")
    predicted = normalize_hotpotqa_answer(prediction)
    rows: list[QAMetrics] = []
    for answer in accepted_answers:
        expected = normalize_hotpotqa_answer(answer)
        if predicted != expected and (predicted in _HOTPOT_SPECIAL or expected in _HOTPOT_SPECIAL):
            rows.append(QAMetrics(0.0, 0.0))
        else:
            rows.append(
                score_qa_answers(
                    prediction,
                    (answer,),
                    normalizer=normalize_hotpotqa_answer,
                )
            )
    return QAMetrics(max(row.em for row in rows), max(row.f1 for row in rows))


__all__ = [
    "QAMetrics",
    "best_alias_metrics",
    "normalize_hotpotqa_answer",
    "normalize_musique_answer",
    "normalize_nq_open_answer",
    "normalize_triviaqa_answer",
    "score_hotpotqa_answers",
    "score_musique_answers",
    "score_qa_answers",
]
