from __future__ import annotations

import os
import select
import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class DeadlinePipe:
    request_fd: int
    response_fd: int
    maximum_bytes: int
    _buffer: bytearray = field(default_factory=bytearray, init=False)

    def __post_init__(self) -> None:
        os.set_blocking(self.request_fd, False)
        os.set_blocking(self.response_fd, False)

    @staticmethod
    def _wait(fd: int, deadline: float, *, writing: bool = False) -> None:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("official environment request deadline expired")
            try:
                readable, writable, _ = select.select(
                    () if writing else (fd,), (fd,) if writing else (), (), remaining
                )
            except InterruptedError:
                continue
            if readable or writable:
                return
            raise TimeoutError("official environment request deadline expired")

    def write(self, request: bytes, *, deadline: float) -> None:
        if len(request) > self.maximum_bytes:
            raise ValueError("official environment request is too large")
        pending = memoryview(request + b"\n")
        while pending:
            self._wait(self.request_fd, deadline, writing=True)
            try:
                count = os.write(self.request_fd, pending)
            except (BlockingIOError, InterruptedError):
                continue
            if not count:
                raise BrokenPipeError("official environment request channel closed")
            pending = pending[count:]

    def read(self, *, deadline: float) -> bytes:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("official environment request deadline expired")
            end = self._buffer.find(b"\n")
            if end >= 0:
                if end > self.maximum_bytes:
                    raise OSError("official environment response is too large")
                line = bytes(self._buffer[:end])
                del self._buffer[: end + 1]
                return line
            if len(self._buffer) > self.maximum_bytes:
                raise OSError("official environment response is too large")
            self._wait(self.response_fd, deadline)
            try:
                chunk = os.read(self.response_fd, 65536)
            except (BlockingIOError, InterruptedError):
                continue
            if not chunk:
                raise BrokenPipeError("official environment exited before a complete response")
            self._buffer.extend(chunk)
