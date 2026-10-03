from __future__ import annotations

import asyncio
import inspect
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from skillev.contracts import JsonValue, normalize_json

if TYPE_CHECKING:
    from skillev.training import AsyncResourceLimiter


class WorkerError(RuntimeError):
    pass


class PrivateWorkerTransport(Protocol):
    async def request(
        self,
        *,
        command: tuple[str, ...],
        working_directory: Path,
        encoded: bytes,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> bytes: ...


async def _stop_process(
    process: asyncio.subprocess.Process,
    wait_task: asyncio.Task[int] | None,
) -> None:
    if process.returncode is not None or (wait_task is not None and wait_task.done()):
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        waiter = wait_task if wait_task is not None else asyncio.create_task(process.wait())
        await asyncio.wait_for(asyncio.shield(waiter), timeout=1.0)
    except TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            return
        waiter = wait_task if wait_task is not None else asyncio.create_task(process.wait())
        await asyncio.shield(waiter)


@dataclass(frozen=True, slots=True)
class AsyncSubprocessWorkerTransport:
    async def request(
        self,
        *,
        command: tuple[str, ...],
        working_directory: Path,
        encoded: bytes,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> bytes:
        process: asyncio.subprocess.Process | None = None
        wait_task: asyncio.Task[int] | None = None
        try:
            async with asyncio.timeout(timeout_seconds):
                environment = os.environ.copy()
                environment.pop("PYTHONPATH", None)
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=working_directory,
                    env=environment,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                if process.stdin is None or process.stdout is None:
                    raise WorkerError("official grader pipes are unavailable")
                wait_task = asyncio.create_task(process.wait())
                try:
                    process.stdin.write(encoded)
                    await process.stdin.drain()
                    process.stdin.close()
                    await process.stdin.wait_closed()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                try:
                    response = await process.stdout.readexactly(max_response_bytes + 1)
                except asyncio.IncompleteReadError as error:
                    response = error.partial
                if len(response) > max_response_bytes:
                    raise WorkerError("official grader returned an invalid response size")
                return_code = await asyncio.shield(wait_task)
                if return_code != 0:
                    raise WorkerError("official grader process returned failure")
                return response
        except asyncio.CancelledError:
            if process is not None:
                await asyncio.shield(_stop_process(process, wait_task))
            raise
        except WorkerError:
            if process is not None:
                await _stop_process(process, wait_task)
            raise
        except (OSError, TimeoutError) as error:
            if process is not None:
                await _stop_process(process, wait_task)
            raise WorkerError("official grader process did not complete") from error


InMemoryWorkerResponder = Callable[[bytes], bytes | Awaitable[bytes]]


@dataclass(slots=True)
class InMemoryWorkerTransport:
    responder: InMemoryWorkerResponder
    requests: list[bytes] = field(default_factory=list)

    async def request(
        self,
        *,
        command: tuple[str, ...],
        working_directory: Path,
        encoded: bytes,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> bytes:
        del command, working_directory
        self.requests.append(encoded)
        try:
            async with asyncio.timeout(timeout_seconds):
                result = self.responder(encoded)
                response = await result if inspect.isawaitable(result) else result
        except TimeoutError as error:
            raise WorkerError("official grader process did not complete") from error
        if not isinstance(response, bytes):
            raise TypeError("in-memory worker response must be bytes")
        if len(response) > max_response_bytes:
            raise WorkerError("official grader returned an invalid response size")
        return response


@dataclass(frozen=True, slots=True)
class PrivateJSONWorker:
    command: tuple[str, ...]
    working_directory: Path
    timeout_seconds: float
    max_response_bytes: int = 1024 * 1024
    process_limiter: AsyncResourceLimiter | None = None
    transport: PrivateWorkerTransport | None = None

    def __post_init__(self) -> None:
        if not self.command or any(not part for part in self.command):
            raise ValueError("official worker command must be non-empty")
        if not self.working_directory.is_absolute():
            raise ValueError("official worker directory must be absolute")
        if self.timeout_seconds <= 0 or self.max_response_bytes <= 0:
            raise ValueError("official worker limits must be positive")

    async def request(self, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        normalized = normalize_json(value)
        if not isinstance(normalized, dict):
            raise TypeError("official worker request must be an object")
        encoded = json.dumps(normalized, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"
        if self.process_limiter is None:
            return await self._request_async(encoded)
        async with self.process_limiter.lease():
            return await self._request_async(encoded)

    async def _request_async(self, encoded: bytes) -> dict[str, JsonValue]:
        transport = self.transport or AsyncSubprocessWorkerTransport()
        response = await transport.request(
            command=self.command,
            working_directory=self.working_directory,
            encoded=encoded,
            timeout_seconds=self.timeout_seconds,
            max_response_bytes=self.max_response_bytes,
        )
        if not response:
            raise WorkerError("official grader returned an invalid response size")
        try:
            value = normalize_json(json.loads(response))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise WorkerError("official grader returned malformed JSON") from error
        if not isinstance(value, dict):
            raise WorkerError("official grader response must be an object")
        return value


__all__ = [
    "AsyncSubprocessWorkerTransport",
    "InMemoryWorkerTransport",
    "PrivateJSONWorker",
    "PrivateWorkerTransport",
    "WorkerError",
]
