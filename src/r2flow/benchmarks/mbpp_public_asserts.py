from __future__ import annotations

import asyncio
import json
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Protocol

from skillev.verification.code_asserts import (
    PublicAssertInfrastructureError,
    PublicAssertRequest,
    PublicAssertResult,
)

MBPP_PUBLIC_ASSERT_PROFILE = "mbpp-public-asserts-isolated@1"
_MAX_REQUEST_BYTES = 2 * 1024 * 1024
_MAX_RESULT_BYTES = 4096
_CHILD_STARTED = b"skillev-public-asserts-started\n"


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    wall_timeout_seconds: float = 3.0
    cpu_seconds: int = 2
    address_space_bytes: int = 512 * 1024 * 1024
    process_count: int = 1
    file_size_bytes: int = 1024 * 1024
    open_file_count: int = 32


_LIMITS = ResourceLimits()

_CHILD_RUNNER = rf"""
import contextlib
import json
import os
import resource
import sys

resource.setrlimit(resource.RLIMIT_CPU, ({_LIMITS.cpu_seconds}, {_LIMITS.cpu_seconds}))
_AS = {_LIMITS.address_space_bytes}
resource.setrlimit(resource.RLIMIT_AS, (_AS, _AS))
resource.setrlimit(resource.RLIMIT_NPROC, ({_LIMITS.process_count}, {_LIMITS.process_count}))
resource.setrlimit(resource.RLIMIT_FSIZE, ({_LIMITS.file_size_bytes}, {_LIMITS.file_size_bytes}))
resource.setrlimit(resource.RLIMIT_NOFILE, ({_LIMITS.open_file_count}, {_LIMITS.open_file_count}))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


_verdict = os.fdopen(os.dup(1), "wb")


def emit(value):
    _verdict.write(json.dumps(value, sort_keys=True).encode("utf-8"))
    _verdict.flush()
    os._exit(0)


try:
    payload = json.loads(sys.stdin.buffer.read())
    if set(payload) != {{"asserts", "code"}} or not isinstance(payload["code"], str):
        raise ValueError("invalid child request")
    if not isinstance(payload["asserts"], list) or not all(
        isinstance(item, str) for item in payload["asserts"]
    ):
        raise ValueError("invalid child request")
except BaseException:
    emit({{"infrastructure_error": True}})

os.write(2, b"skillev-public-asserts-started\n")
_null = os.open(os.devnull, os.O_WRONLY)
os.dup2(_null, 1)
os.dup2(_null, 2)
namespace = {{"__name__": "__mbpp_candidate__"}}
with open(os.devnull, "w", encoding="utf-8") as null:
    with contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
        try:
            exec(compile(payload["code"], "<candidate>", "exec"), namespace, namespace)
        except BaseException:
            emit({{"status": "error", "failed_index": None}})
        for index, line in enumerate(payload["asserts"]):
            try:
                exec(compile(line, "<public-assert>", "exec"), namespace, namespace)
            except AssertionError:
                emit({{"status": "fail", "failed_index": index}})
            except BaseException:
                emit({{"status": "error", "failed_index": index}})
emit({{"status": "pass", "failed_index": None}})
"""


class ProcessLimiter(Protocol):
    def lease(self, *, token_cost: int | None = None, role: str = ...) -> Any: ...


def _run_child(request: PublicAssertRequest, command_prefix: tuple[str, ...]) -> PublicAssertResult:
    payload = json.dumps(
        {"asserts": list(request.asserts), "code": request.code},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    if len(payload) > _MAX_REQUEST_BYTES:
        raise PublicAssertInfrastructureError("public-assert request exceeds its transport limit")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="skillev-public-asserts-") as directory:
        try:
            completed = subprocess.run(
                [*command_prefix, sys.executable, "-I", "-S", "-c", _CHILD_RUNNER],
                input=payload,
                cwd=directory,
                env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONHASHSEED": "0"},
                capture_output=True,
                start_new_session=True,
                timeout=request.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            if not (error.stderr or b"").startswith(_CHILD_STARTED):
                raise PublicAssertInfrastructureError(
                    "isolated interpreter did not start before its deadline"
                ) from error
            return PublicAssertResult("timeout", None, _elapsed(started))
        except (OSError, subprocess.SubprocessError) as error:
            raise PublicAssertInfrastructureError("isolated process could not start") from error
    elapsed = _elapsed(started)
    if not completed.stderr.startswith(_CHILD_STARTED):
        raise PublicAssertInfrastructureError("isolated interpreter did not start the runner")
    if completed.returncode < 0:
        timed_out = -completed.returncode in (signal.SIGXCPU, signal.SIGKILL)
        return PublicAssertResult("timeout" if timed_out else "error", None, elapsed)
    if completed.returncode != 0 or len(completed.stdout) > _MAX_RESULT_BYTES:
        return PublicAssertResult("error", None, elapsed)
    try:
        value = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return PublicAssertResult("error", None, elapsed)
    if value == {"infrastructure_error": True}:
        raise PublicAssertInfrastructureError("public-assert child rejected its request")
    if (
        not isinstance(value, dict)
        or set(value) != {"failed_index", "status"}
        or value["status"] not in ("pass", "fail", "error")
        or not (value["failed_index"] is None or type(value["failed_index"]) is int)
    ):
        return PublicAssertResult("error", None, elapsed)
    return PublicAssertResult(value["status"], value["failed_index"], elapsed)


def _elapsed(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


@dataclass(frozen=True, slots=True)
class IsolatedPublicAssertBackend:
    command_prefix: tuple[str, ...] = ()
    limiter: ProcessLimiter | None = None
    profile_id: str = MBPP_PUBLIC_ASSERT_PROFILE

    async def run(self, request: PublicAssertRequest) -> PublicAssertResult:
        if not isinstance(request, PublicAssertRequest):
            raise TypeError("public-assert backend requires PublicAssertRequest")
        lease: AbstractAsyncContextManager[Any] = (
            nullcontext()
            if self.limiter is None
            else self.limiter.lease(role="mbpp-public-asserts")
        )
        async with lease:
            return await asyncio.to_thread(_run_child, request, self.command_prefix)


__all__ = ["MBPP_PUBLIC_ASSERT_PROFILE", "IsolatedPublicAssertBackend"]
