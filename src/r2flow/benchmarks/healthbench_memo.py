from __future__ import annotations

import json
import os
import tempfile
import threading
import unicodedata
from pathlib import Path

from skillev.contracts import JsonValue, canonical_json, normalize_json
from skillev.contracts.canonical import stable_hash

from .native_backends import HealthBenchGrade

MEMO_FORMAT = "healthbench-verdict-memo@1"
ANSWER_CANON = "healthbench-answer-canon@1"
MEMO_KEY_FORMAT = "healthbench-verdict-memo-key@1"
MEMO_DIRECTORY = "healthbench-verdict-memo"


def canonical_answer_text(answer: str) -> str:
    text = answer.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text).strip()


def memo_key(private_case: object, answer: str, settings: object) -> str:
    return stable_hash(
        {
            "format": MEMO_KEY_FORMAT,
            "question": stable_hash(normalize_json(private_case)),
            "answer": canonical_answer_text(answer),
            "profile": stable_hash(normalize_json(settings)),
        }
    )


def _grade_value(grade: HealthBenchGrade) -> dict[str, JsonValue]:
    return {
        "official_rubric_score": grade.official_rubric_score,
        "triggered_negative_rubric_count": grade.triggered_negative_rubric_count,
        "grader_cost": normalize_json(grade.grader_cost),
        "criterion_ledger": grade.criterion_ledger,
        **(
            {"refused_criterion_count": grade.refused_criterion_count}
            if grade.refused_criterion_count is not None
            else {}
        ),
    }


def _grade_from_value(value: dict[str, JsonValue]) -> HealthBenchGrade:
    return HealthBenchGrade(
        value["official_rubric_score"],
        value["triggered_negative_rubric_count"],
        value["grader_cost"],
        value["criterion_ledger"],
        value.get("refused_criterion_count"),
    )


class HealthBenchVerdictMemo:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def key_lock(self, key: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(key, threading.Lock())

    def _path(self, key: str) -> Path:
        return self.root / (key.removeprefix("sha256:") + ".json")

    def lookup(self, key: str) -> HealthBenchGrade | None:
        path = self._path(key)
        if not path.exists():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("format") != MEMO_FORMAT or value.get("key") != key:
            raise ValueError("HealthBench verdict memo entry is incompatible")
        return _grade_from_value(value["grade"])

    def store(self, key: str, grade: HealthBenchGrade) -> HealthBenchGrade:
        payload = canonical_json({"format": MEMO_FORMAT, "key": key, "grade": _grade_value(grade)})
        fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, self._path(key))
            except FileExistsError:
                saved = self.lookup(key)
                assert saved is not None
                return saved
        finally:
            os.unlink(temporary)
        return _grade_from_value(json.loads(payload)["grade"])


def with_memo_reference(grade: HealthBenchGrade, key: str, *, hit: bool) -> HealthBenchGrade:
    reference: dict[str, JsonValue] = dict(grade.criterion_ledger or {})
    reference["verdict_memo"] = {"format": MEMO_FORMAT, "key": key, "hit": hit}
    return HealthBenchGrade(
        grade.official_rubric_score,
        grade.triggered_negative_rubric_count,
        grade.grader_cost,
        reference,
        grade.refused_criterion_count,
    )


__all__ = [
    "ANSWER_CANON",
    "MEMO_DIRECTORY",
    "MEMO_FORMAT",
    "MEMO_KEY_FORMAT",
    "HealthBenchVerdictMemo",
    "canonical_answer_text",
    "memo_key",
    "with_memo_reference",
]
