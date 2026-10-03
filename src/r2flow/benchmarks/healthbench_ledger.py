from __future__ import annotations

import json
import os
import threading
import weakref
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

from skillev.contracts import canonical_json, normalize_json
from skillev.runtime.request_journal import UnknownRequestOutcomeError
from skillev.training.inflight import durable_json

_INCOMPLETE_FINISH = frozenset({"length", "content_filter"})


class StaleLedgerError(RuntimeError):
    pass


_OWNERS: weakref.WeakValueDictionary[str, HealthBenchCriterionLedger] = (
    weakref.WeakValueDictionary()
)
_PATH_LOCKS: dict[str, threading.RLock] = {}
_REGISTRY_LOCK = threading.Lock()


def _path_lock(key: str) -> threading.RLock:
    with _REGISTRY_LOCK:
        return _PATH_LOCKS.setdefault(key, threading.RLock())


def _replay_key(request: Mapping[str, Any]) -> str:
    from r2flow.benchmarks.healthbench_grader_usage import RECORDED_REQUEST_KEYS

    return canonical_json(
        normalize_json(
            {k: request[k] for k in RECORDED_REQUEST_KEYS if k in request and k != "timeout"}
        )
    )


def _chat_completion(raw: Any, request_id: Any) -> Any:
    from openai.types.chat import ChatCompletion

    response = ChatCompletion.model_validate(raw)
    response._request_id = request_id
    return response


class CriterionReplay:
    def __init__(self) -> None:
        self._pool: dict[str, deque[tuple[Any, dict[str, Any], dict[str, Any]]]] = {}
        self._raw_seen: set[str] = set()
        self._lock = threading.Lock()
        self.available = 0
        self.taken = 0
        self.late_spool_responses: list[str] = []

    def add(
        self, request: Mapping[str, Any], response: dict[str, Any], source: dict[str, Any]
    ) -> bool:
        raw = response.get("raw_response")
        if raw is None or response.get("finish_reason") in _INCOMPLETE_FINISH:
            return False
        identity = canonical_json(normalize_json(raw))
        if identity in self._raw_seen:
            return False
        try:
            _chat_completion(raw, response.get("request_id"))
        except ValueError:
            return False
        self._raw_seen.add(identity)
        self._pool.setdefault(_replay_key(request), deque()).append((raw, response, source))
        self.available += 1
        return True

    def take(self, kwargs: Mapping[str, Any]) -> tuple[Any, dict[str, Any], dict[str, Any]] | None:
        with self._lock:
            queue = self._pool.get(_replay_key(kwargs))
            if not queue:
                return None
            raw, response, source = queue.popleft()
            self.taken += 1
        return _chat_completion(raw, response.get("request_id")), response, source

    def to_value(self) -> dict[str, Any]:
        return {
            "available_responses": self.available,
            "late_spool_responses": list(self.late_spool_responses),
        }

    @classmethod
    def from_judgments(
        cls, judgments: Sequence[tuple[str, Mapping[str, Any]]], spool_root: Path | None
    ) -> CriterionReplay:
        from r2flow.benchmarks.healthbench_grader_usage import response_evidence
        from skillev.evaluation.healthbench_judge_recovery import SPOOL_DEADLINE_ERROR

        replay = cls()
        for name, state in judgments:
            for row in state.get("requests", []):
                request, response = row.get("request"), row.get("response")
                if not isinstance(request, dict):
                    continue
                if isinstance(response, dict) and "error_type" not in row:
                    source = row.get("replayed_from") or {
                        "ledger": name,
                        "attempt": row.get("attempt"),
                    }
                    replay.add(request, response, dict(source))
                    continue
                spool_id = row.get("spool_request_id")
                if (
                    spool_root is None
                    or row.get("spool_error_type") != SPOOL_DEADLINE_ERROR
                    or not isinstance(spool_id, str)
                ):
                    continue
                path = spool_root / "responses" / f"{spool_id}.json"
                if not path.is_file():
                    continue
                late = json.loads(path.read_text(encoding="utf-8"))
                if (
                    late.get("request_id") != spool_id
                    or late.get("status") != "completed"
                    or not isinstance(late.get("request"), dict)
                    or _replay_key(late["request"]) != _replay_key(request)
                ):
                    continue
                try:
                    completion = _chat_completion(late["raw_response"], late.get("api_request_id"))
                except (KeyError, ValueError):
                    continue
                added = replay.add(
                    request,
                    response_evidence(completion),
                    {
                        "ledger": name,
                        "attempt": row.get("attempt"),
                        "late_spool_response": spool_id,
                    },
                )
                if added:
                    replay.late_spool_responses.append(spool_id)
        return replay


class HealthBenchCriterionLedger:
    def __init__(
        self,
        root: Path,
        task_id: str,
        binding: dict[str, Any],
        *,
        supersede_transport_incomplete: bool = False,
        spool_root: Path | None = None,
    ) -> None:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = root / (quote(task_id, safe="") + ".json")
        self._key = str(self.path.resolve())
        self._lock = threading.RLock()
        expected = normalize_json({"task_id": task_id, **binding})
        with _path_lock(self._key):
            self._open(root, expected, supersede_transport_incomplete, spool_root)
            with _REGISTRY_LOCK:
                _OWNERS[self._key] = self

    def _open(
        self,
        root: Path,
        expected: dict[str, Any],
        supersede_transport_incomplete: bool,
        spool_root: Path | None,
    ) -> None:
        from skillev.evaluation.healthbench_judge_recovery import ledger_supersedable

        superseded: list[str] = []
        self.replay: CriterionReplay | None = None
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state["binding"] != expected:
                raise ValueError(
                    "saved HealthBench judgment belongs to another submission/condition"
                )
            if self.state["status"] != "completed":
                if not (supersede_transport_incomplete and ledger_supersedable(self.state)):
                    raise UnknownRequestOutcomeError(
                        "prior HealthBench judgment is incomplete; do not resample"
                    )
                earlier = list(self.state.get("supersedes", []))
                target = self._archive_target()
                self.replay = CriterionReplay.from_judgments(
                    [
                        *(
                            (name, json.loads((root / name).read_text(encoding="utf-8")))
                            for name in earlier
                        ),
                        (target.name, self.state),
                    ],
                    spool_root,
                )
                superseded = [*earlier, self._archive(target)]
        if not self.path.exists():
            self.state = {
                "format": "healthbench-criterion-ledger@1",
                "binding": expected,
                "status": "started",
                "requests": [],
                "result": None,
                **({"supersedes": superseded} if superseded else {}),
                **({"replay": self.replay.to_value()} if self.replay is not None else {}),
            }
            with self.path.open("x", encoding="utf-8") as stream:
                os.chmod(self.path, 0o600)
                json.dump(self.state, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def _archive_target(self) -> Path:
        k = 1
        while True:
            target = self.path.with_name(f"{self.path.stem}.superseded-{k}.json")
            if not target.exists():
                return target
            k += 1

    def _archive(self, target: Path) -> str:
        os.replace(self.path, target)
        fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return target.name

    def update(self, **fields: Any) -> None:
        with self._lock, _path_lock(self._key):
            if self.state["status"] == "completed":
                raise ValueError("a completed HealthBench judgment is immutable")
            if _OWNERS.get(self._key) is not self:
                raise StaleLedgerError(
                    "this HealthBench judgment was superseded; its writer must not touch the ledger"
                )
            self.state.update(fields)
            durable_json(self.path, self.state)

    def record_requests(self, rows: list[dict[str, Any]]) -> None:
        self.update(requests=rows)

    @property
    def completed(self) -> bool:
        return bool(self.state["status"] == "completed")

    def reference(self) -> dict[str, Any]:
        return {
            "format": self.state["format"],
            "path": str(self.path),
            "status": self.state["status"],
            "request_count": len(self.state["requests"]),
        }


def official_criterion_messages(
    grade_sample: Any,
    prompt: list[dict[str, str]],
    candidate: str,
    rubrics: list[Any],
) -> list[list[dict[str, str]]] | None:
    template = getattr(grade_sample, "__globals__", {}).get("GRADER_TEMPLATE")
    if not isinstance(template, str):
        return None
    conversation = "\n\n".join(
        f"{message['role']}: {message['content']}"
        for message in [*prompt, {"role": "assistant", "content": candidate}]
    )
    return [
        [
            {
                "role": "user",
                "content": template.replace("<<conversation>>", conversation).replace(
                    "<<rubric_item>>", str(rubric)
                ),
            }
        ]
        for rubric in rubrics
    ]
