from __future__ import annotations

import atexit
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4


def write_spool_json(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            os.chmod(temp, 0o600)
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temp.unlink(missing_ok=True)


_OUTSTANDING: dict[str, Path] = {}
_OUTSTANDING_LOCK = threading.Lock()
_EXIT_HOOK: list[bool] = []


def _withdraw_outstanding() -> None:
    with _OUTSTANDING_LOCK:
        paths = list(_OUTSTANDING.values())
        _OUTSTANDING.clear()
    for path in paths:
        path.unlink(missing_ok=True)


def _track_outstanding(request_id: str, path: Path) -> None:
    with _OUTSTANDING_LOCK:
        _OUTSTANDING[request_id] = path
        if not _EXIT_HOOK:
            atexit.register(_withdraw_outstanding)
            _EXIT_HOOK.append(True)


def _untrack_outstanding(request_id: str) -> None:
    with _OUTSTANDING_LOCK:
        _OUTSTANDING.pop(request_id, None)


class JudgeSpoolError(RuntimeError):
    def __init__(self, request_id: str, error_type: str) -> None:
        self.request_id = request_id
        self.error_type = error_type
        super().__init__(f"External Judge operation {request_id} failed: {error_type}")


class JudgeSpoolClient:
    def __init__(
        self,
        root: Path,
        *,
        poll_seconds: float = 1.0,
        recovery: Any = None,
        clock: Any = None,
        sleeper: Any = None,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("Spool poll interval must be positive")
        self.root = root
        self.poll_seconds = poll_seconds
        self.recovery = recovery
        self._clock = clock
        self._sleep = sleeper
        self.chat = SimpleNamespace(completions=self)

    @property
    def deadline_seconds(self) -> float | None:
        return None if self.recovery is None else float(self.recovery.spool_deadline_seconds)

    def create(self, **kwargs: Any) -> Any:
        from openai.types.chat import ChatCompletion

        from .healthbench_judge_recovery import SPOOL_DEADLINE_ERROR

        request_id = str(uuid4())
        request = {"request_id": request_id, "request": kwargs, "created_unix": time.time()}
        request_path = self.root / "requests" / f"{request_id}.json"
        write_spool_json(request_path, request)
        response_path = self.root / "responses" / f"{request_id}.json"
        clock = self._clock or time.monotonic
        deadline = self.deadline_seconds
        started = clock()
        withdrawn: float | None = None
        grace = 0.0
        if self.recovery is not None:
            _track_outstanding(request_id, request_path)
        try:
            while not response_path.exists():
                now = clock()
                if withdrawn is None and deadline is not None and now - started >= deadline:
                    request_path.unlink(missing_ok=True)
                    withdrawn = now
                    grace = float(self.recovery.withdrawal_grace_seconds(kwargs.get("timeout")))
                if withdrawn is not None and now - withdrawn >= grace:
                    if not response_path.exists():
                        raise JudgeSpoolError(request_id, SPOOL_DEADLINE_ERROR)
                    break
                (self._sleep or time.sleep)(self.poll_seconds)
        finally:
            _untrack_outstanding(request_id)
        response = json.loads(response_path.read_text(encoding="utf-8"))
        if response.get("request_id") != request_id or response.get("request") != kwargs:
            raise ValueError("Judge spool response belongs to another operation")
        (self.root / "requests" / f"{request_id}.json").unlink(missing_ok=True)
        if response.get("status") != "completed":
            raise JudgeSpoolError(request_id, response.get("error_type", "UnknownAPIOutcome"))
        result = ChatCompletion.model_validate(response["raw_response"])
        result._request_id = response.get("api_request_id")
        return result

    def close(self) -> None:
        pass
