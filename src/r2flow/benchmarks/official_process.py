from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import time
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager, nullcontext, suppress
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import cast

from skillev.contracts import JsonValue, normalize_json
from skillev.diagnostics.rollout_progress import current_progress
from skillev.rollout.errors import EpisodeInfrastructureError
from skillev.rollout.evaluation_deadline import REQUEST_WINDOW

from .alfworld_official import (
    OfficialALFWorldResetResult,
    OfficialALFWorldStepResult,
    OfficialALFWorldTask,
    OfficialALFWorldTextEnv,
)
from .deadline_pipe import DeadlinePipe

_PROTOCOL_VERSION = "skillev-official-environment-worker@1"
_MAX_MESSAGE_BYTES = 2 * 1024 * 1024
_REVISION_LENGTH = 40
_WORKER_SCRIPT = Path(__file__).with_name("official_environment_worker.py")


class OfficialEnvironmentInfrastructureError(EpisodeInfrastructureError):
    pass


def _text(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{field_name} must be non-empty text without NUL")
    return value


def _seed(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("official environment seed must be a non-negative integer")
    return value


def _positive_timeout(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("official environment timeout must be numeric")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("official environment timeout must be positive and finite")
    return timeout


def _path(value: Path | str, *, field_name: str, directory: bool) -> Path:
    path = Path(value).expanduser().resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise ValueError(f"{field_name} must identify an existing {kind}")
    return path


def _executable_path(value: Path | str, *, field_name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = path.absolute()
    if not path.is_file():
        raise ValueError(f"{field_name} must identify an existing file")
    return path


def _revision(value: object) -> str:
    revision = _text(value, field_name="source_revision").lower()
    if len(revision) != _REVISION_LENGTH or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError("source_revision must be a full 40-character Git commit")
    return revision


def _operation_deadline(seconds: float, *, closing: bool = False) -> float:
    deadline = time.monotonic() + seconds
    window = REQUEST_WINDOW.get()
    if window is not None:
        deadline = min(deadline, window.cleanup_deadline if closing else window.work_deadline)
    return deadline


def _time_left(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("episode has no native worker time remaining")
    return remaining


def _verify_checkout(*, source_root: Path, source_revision: str, timeout: float) -> None:
    deadline = _operation_deadline(timeout)
    try:
        completed = subprocess.run(
            ("git", "-C", str(source_root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
            timeout=_time_left(deadline),
        )
        subprocess.run(
            ("git", "-C", str(source_root), "diff", "--quiet", "HEAD", "--"),
            check=True,
            capture_output=True,
            timeout=_time_left(deadline),
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        raise OfficialEnvironmentInfrastructureError(
            "official source revision could not be resolved"
        ) from error
    if completed.stdout.strip().lower() != source_revision:
        raise ValueError("official source checkout does not match its pinned revision")


def _object(
    value: object, *, fields: set[str], label: str, optional_fields: frozenset[str] = frozenset()
) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if (
        not isinstance(normalized, dict)
        or not fields.issubset(normalized)
        or set(normalized) - fields - optional_fields
    ):
        raise OfficialEnvironmentInfrastructureError(f"{label} has an incompatible shape")
    return normalized


def _string_array(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise OfficialEnvironmentInfrastructureError(f"{label} must be an array")
    items = tuple(_text(item, field_name=label) for item in value)
    if len(set(items)) != len(items):
        raise OfficialEnvironmentInfrastructureError(f"{label} must be unique")
    return items


@dataclass(frozen=True, slots=True)
class PinnedOfficialProcess:
    interpreter_path: Path
    source_root: Path
    source_revision: str
    request_timeout_seconds: float = 60.0
    worker_script_path: Path = _WORKER_SCRIPT
    worker_stderr_path: Path | None = None

    def __post_init__(self) -> None:
        interpreter = _executable_path(
            self.interpreter_path,
            field_name="interpreter_path",
        )
        source_root = _path(self.source_root, field_name="source_root", directory=True)
        worker = _path(
            self.worker_script_path,
            field_name="worker_script_path",
            directory=False,
        )
        revision = _revision(self.source_revision)
        timeout = _positive_timeout(self.request_timeout_seconds)
        _verify_checkout(source_root=source_root, source_revision=revision, timeout=timeout)
        object.__setattr__(self, "interpreter_path", interpreter)
        object.__setattr__(self, "source_root", source_root)
        object.__setattr__(self, "source_revision", revision)
        object.__setattr__(self, "request_timeout_seconds", timeout)
        object.__setattr__(self, "worker_script_path", worker)


class WorkerLifecycleState(StrEnum):
    OPEN = "open"
    CLOSED_SUCCESSFULLY = "closed-successfully"
    DISCARDED_AFTER_INITIALIZATION_FAILURE = "discarded-after-initialization-failure"
    DISCARDED_AFTER_REQUEST_FAILURE = "discarded-after-request-failure"


@dataclass(slots=True)
class OfficialWorkerClient:
    runtime: PinnedOfficialProcess
    _process: subprocess.Popen[bytes] = field(init=False, repr=False)
    _response_fd: int = field(init=False, repr=False)
    _channel: DeadlinePipe = field(init=False, repr=False)
    _request_id: int = field(default=0, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)
    _state: WorkerLifecycleState = field(
        default=WorkerLifecycleState.OPEN,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        read_fd, write_fd = os.pipe()
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        with ExitStack() as files:
            error_log = self.runtime.worker_stderr_path
            if error_log is not None:
                error_log.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._process = subprocess.Popen(
                (
                    str(self.runtime.interpreter_path),
                    str(self.runtime.worker_script_path),
                    "--response-fd",
                    str(write_fd),
                ),
                cwd=self.runtime.source_root,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=files.enter_context(error_log.open("ab"))
                if error_log
                else subprocess.DEVNULL,
                pass_fds=(write_fd,),
                env=environment,
            )
        os.close(write_fd)
        self._response_fd = read_fd
        assert self._process.stdin is not None
        self._channel = DeadlinePipe(self._process.stdin.fileno(), read_fd, _MAX_MESSAGE_BYTES)

    @contextmanager
    def _serialized(self, deadline: float) -> Iterator[None]:
        if not self._lock.acquire(timeout=_time_left(deadline)):
            raise TimeoutError("native worker is still processing its earlier operation")
        try:
            yield
        finally:
            self._lock.release()

    def request(self, operation: str, payload: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        deadline = _operation_deadline(
            self.runtime.request_timeout_seconds, closing=operation == "close"
        )
        with self._serialized(deadline):
            if self._state is not WorkerLifecycleState.OPEN:
                raise RuntimeError("official worker is not open")
            try:
                return self._request(operation, payload, deadline=deadline)
            except (
                OSError,
                ValueError,
                TypeError,
                OfficialEnvironmentInfrastructureError,
            ) as error:
                self._discard(WorkerLifecycleState.DISCARDED_AFTER_REQUEST_FAILURE)
                raise OfficialEnvironmentInfrastructureError(
                    f"official environment request failed: {type(error).__name__}: {error}"
                ) from error

    def _request(
        self, operation: str, payload: Mapping[str, JsonValue], *, deadline: float
    ) -> dict[str, JsonValue]:
        self._request_id += 1
        request = normalize_json(
            {
                "operation": _text(operation, field_name="worker operation"),
                "payload": dict(payload),
                "protocol_version": _PROTOCOL_VERSION,
                "request_id": self._request_id,
            }
        )
        encoded = json.dumps(
            request,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if len(encoded) > _MAX_MESSAGE_BYTES:
            raise ValueError("official environment worker request is too large")
        self._channel.write(encoded, deadline=deadline)
        response_bytes = self._channel.read(deadline=deadline)
        response_value: object = json.loads(response_bytes)
        response = _object(
            response_value,
            fields={"protocol_version", "request_id", "result"},
            label="official environment worker response",
        )
        if response["protocol_version"] != _PROTOCOL_VERSION:
            raise OfficialEnvironmentInfrastructureError("official worker protocol version differs")
        if response["request_id"] != self._request_id:
            raise OfficialEnvironmentInfrastructureError(
                "official worker response request ID differs"
            )
        result = response["result"]
        if not isinstance(result, dict):
            raise OfficialEnvironmentInfrastructureError(
                "official environment worker result must be an object"
            )
        return result

    def close_successfully(self) -> None:
        deadline = _operation_deadline(self.runtime.request_timeout_seconds, closing=True)
        with self._serialized(deadline):
            try:
                self._close_successfully(deadline)
            except (TimeoutError, subprocess.TimeoutExpired):
                self._discard(WorkerLifecycleState.DISCARDED_AFTER_REQUEST_FAILURE)
                raise

    def _close_successfully(self, deadline: float) -> None:
        if self._state is WorkerLifecycleState.DISCARDED_AFTER_REQUEST_FAILURE:
            return
        if self._state is not WorkerLifecycleState.OPEN:
            raise RuntimeError("official worker was closed more than once")
        if self._process.poll() is not None:
            raise OfficialEnvironmentInfrastructureError(
                "official worker exited before the explicit close"
            )
        result = self.request("close", {})
        closed = _object(result, fields={"closed"}, label="worker close result")["closed"]
        if closed is not True:
            raise OfficialEnvironmentInfrastructureError("official worker did not confirm close")
        self._process.wait(timeout=_time_left(deadline))
        if self._process.returncode != 0:
            raise OfficialEnvironmentInfrastructureError(
                "official worker close returned non-zero status"
            )
        stdin = self._process.stdin
        if stdin is None:
            raise RuntimeError("official worker request channel disappeared")
        stdin.close()
        os.close(self._response_fd)
        self._state = WorkerLifecycleState.CLOSED_SUCCESSFULLY

    def _discard(self, state: WorkerLifecycleState) -> None:
        if self._state is WorkerLifecycleState.OPEN:
            if self._process.poll() is None:
                with suppress(OSError):
                    self._process.terminate()
                try:
                    self._process.wait(timeout=min(2.0, self.runtime.request_timeout_seconds))
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=2.0)
            if self._process.stdin is not None:
                with suppress(OSError):
                    self._process.stdin.close()
            with suppress(OSError):
                os.close(self._response_fd)
        self._state = state


def _initialization_payload(
    *,
    benchmark: str,
    runtime: PinnedOfficialProcess,
    deployment: Mapping[str, JsonValue],
    task: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    return {
        "benchmark": benchmark,
        "deployment": dict(deployment),
        "source_revision": runtime.source_revision,
        "source_root": str(runtime.source_root),
        "task": dict(task),
    }


def _initialize(client: OfficialWorkerClient, payload: Mapping[str, JsonValue]) -> None:
    result = client.request("initialize", payload)
    ready = _object(result, fields={"ready"}, label="worker initialize result")["ready"]
    if ready is not True:
        raise OfficialEnvironmentInfrastructureError("official environment worker is not ready")


def _initialize_or_discard(
    client: OfficialWorkerClient,
    payload: Mapping[str, JsonValue],
) -> None:
    try:
        _initialize(client, payload)
        _time_left(_operation_deadline(client.runtime.request_timeout_seconds))
    except Exception:
        client._discard(WorkerLifecycleState.DISCARDED_AFTER_INITIALIZATION_FAILURE)
        raise


@dataclass(frozen=True, slots=True)
class ALFWorldGameDeployment:
    data_directory: Path
    train_eval: str

    def __post_init__(self) -> None:
        directory = _path(
            self.data_directory,
            field_name="ALFWorld data_directory",
            directory=True,
        )
        if self.train_eval not in {
            "train",
            "eval_in_distribution",
            "eval_out_of_distribution",
        }:
            raise ValueError("ALFWorld train_eval is unsupported")
        _single_alfworld_game(directory)
        object.__setattr__(self, "data_directory", directory)


def _single_alfworld_game(directory: Path) -> Path:
    game_files = tuple(directory.rglob("game.tw-pddl"))
    trajectories = tuple(directory.rglob("traj_data.json"))
    if (
        len(game_files) != 1
        or len(trajectories) != 1
        or game_files[0].parent != trajectories[0].parent
    ):
        raise ValueError("ALFWorld case data directory must contain exactly one complete game")
    return game_files[0]


@dataclass(frozen=True, slots=True)
class OfficialALFWorldProcessFactory:
    runtime: PinnedOfficialProcess
    config_path: Path
    games: Mapping[str, ALFWorldGameDeployment]
    seed: int
    simulator_max_steps: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, PinnedOfficialProcess):
            raise TypeError("ALFWorld process factory requires a pinned runtime")
        config = _path(self.config_path, field_name="ALFWorld config_path", directory=False)
        _seed(self.seed)
        games = dict(self.games)
        if not games:
            raise ValueError("ALFWorld process factory requires game deployments")
        for game_id, deployment in games.items():
            _text(game_id, field_name="ALFWorld game_id")
            if not isinstance(deployment, ALFWorldGameDeployment):
                raise TypeError("ALFWorld game deployment is incompatible")
        if self.simulator_max_steps is not None and (
            type(self.simulator_max_steps) is not int or self.simulator_max_steps <= 0
        ):
            raise ValueError("ALFWorld simulator_max_steps must be positive or null")
        object.__setattr__(self, "config_path", config)
        object.__setattr__(self, "games", games)

    def create(self, task: OfficialALFWorldTask) -> OfficialALFWorldTextEnv:
        if not isinstance(task, OfficialALFWorldTask):
            raise TypeError("ALFWorld process factory requires OfficialALFWorldTask")
        if task.seed != self.seed:
            raise ValueError("ALFWorld task seed differs from its deployment identity")
        deployment = self.games.get(task.game_id)
        if deployment is None:
            raise ValueError("ALFWorld game has no pinned data deployment")
        client = OfficialWorkerClient(self.runtime)
        _initialize_or_discard(
            client,
            _initialization_payload(
                benchmark="alfworld",
                runtime=self.runtime,
                deployment={
                    "config_path": str(self.config_path),
                    "data_directory": str(deployment.data_directory),
                    "seed": self.seed,
                    "train_eval": deployment.train_eval,
                },
                task={
                    "game_id": task.game_id,
                    "max_steps": task.max_steps,
                    "simulator_max_steps": self.simulator_max_steps or task.max_steps,
                },
            ),
        )
        return _ALFWorldProcessEnv(task=task, client=client)


@dataclass(slots=True)
class _ALFWorldProcessEnv:
    task: OfficialALFWorldTask
    client: OfficialWorkerClient = field(repr=False)

    @property
    def game_id(self) -> str:
        return self.task.game_id

    @property
    def seed(self) -> int:
        return self.task.seed

    @property
    def max_steps(self) -> int:
        return self.task.max_steps

    def reset(self, seed: int) -> OfficialALFWorldResetResult:
        if seed != self.task.seed:
            raise ValueError("ALFWorld reset seed differs from its pinned identity")
        result = _object(
            self.client.request("reset", {}),
            fields={"admissible_commands", "instruction_text", "observation_text"},
            label="ALFWorld reset result",
        )
        return OfficialALFWorldResetResult(
            observation_text=cast(str, result["observation_text"]),
            instruction_text=cast(str, result["instruction_text"]),
            admissible_commands=_string_array(
                result["admissible_commands"], label="ALFWorld admissible commands"
            ),
        )

    def step(self, action: str) -> OfficialALFWorldStepResult:
        row = current_progress()
        with row.environment_command(action) if row is not None else nullcontext():
            result = _object(
                self.client.request("step", {"action": _text(action, field_name="action")}),
                fields={"admissible_commands", "observation_text", "success", "terminal"},
                label="ALFWorld step result",
            )
        if row is not None:
            row.observation(result["observation_text"])
        step = OfficialALFWorldStepResult(
            observation_text=cast(str, result["observation_text"]),
            admissible_commands=_string_array(
                result["admissible_commands"], label="ALFWorld admissible commands"
            ),
            terminal=cast(bool, result["terminal"]),
            success=cast(bool | None, result["success"]),
        )
        if step.terminal:
            self.client.close_successfully()
        return step

    def close_after_preparation_failure(self) -> None:
        if self.client._state is WorkerLifecycleState.OPEN:
            self.client.close_successfully()

    async def close(self) -> None:
        if self.client._state is WorkerLifecycleState.OPEN:
            await asyncio.to_thread(self.client.close_successfully)


__all__ = [
    "ALFWorldGameDeployment",
    "OfficialALFWorldProcessFactory",
    "OfficialEnvironmentInfrastructureError",
    "OfficialWorkerClient",
    "PinnedOfficialProcess",
    "WorkerLifecycleState",
]
