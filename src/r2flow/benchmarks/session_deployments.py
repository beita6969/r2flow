from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from .triviaqa_wikipedia_search import WikipediaCorpusDeployment

SESSION_DEPLOYMENTS_FORMAT: Final = "r2flow-session-deployments@1"


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise TypeError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _fields(value: object, *, expected: set[str], label: str) -> dict[str, object]:
    result = _mapping(value, label=label)
    if set(result) != expected:
        raise ValueError(f"{label} has incompatible fields")
    return result


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be non-empty text")
    return value


def _path(value: object, *, label: str, directory: bool) -> Path:
    result = Path(_text(value, label=label))
    if not result.is_absolute():
        raise ValueError(f"{label} must be absolute")
    exists = result.is_dir() if directory else result.is_file()
    if not exists:
        raise ValueError(f"{label} does not identify an existing deployment asset")
    return result


def _positive(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or float(value) <= 0:
        raise ValueError(f"{label} must be positive")
    return float(value)


def _path_array(value: object, *, label: str, directory: bool = True) -> tuple[Path, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty array")
    return tuple(_path(item, label=f"{label} item", directory=directory) for item in value)


@dataclass(frozen=True, slots=True)
class ALFWorldDeployment:
    interpreter: Path
    source_root: Path
    source_revision: str
    config_path: Path
    dataset_root: Path
    timeout_seconds: float

    @classmethod
    def from_value(cls, value: object) -> ALFWorldDeployment:
        data = _fields(
            value,
            expected={
                "config_path",
                "dataset_root",
                "interpreter",
                "source_revision",
                "source_root",
                "timeout_seconds",
            },
            label="ALFWorld deployment",
        )
        return cls(
            interpreter=_path(data["interpreter"], label="ALFWorld interpreter", directory=False),
            source_root=_path(data["source_root"], label="ALFWorld source", directory=True),
            source_revision=_text(data["source_revision"], label="ALFWorld revision"),
            config_path=_path(data["config_path"], label="ALFWorld config", directory=False),
            dataset_root=_path(data["dataset_root"], label="ALFWorld dataset", directory=True),
            timeout_seconds=_positive(data["timeout_seconds"], label="ALFWorld timeout"),
        )


@dataclass(frozen=True, slots=True)
class HealthBenchDeployment:
    source_root: Path

    @classmethod
    def from_value(cls, value: object) -> HealthBenchDeployment:
        data = _fields(value, expected={"source_root"}, label="HealthBench deployment")
        return cls(
            source_root=_path(data["source_root"], label="HealthBench source", directory=True)
        )


@dataclass(frozen=True, slots=True)
class TrainingDeployments:
    alfworld: ALFWorldDeployment
    healthbench: HealthBenchDeployment
    triviaqa_wikipedia: WikipediaCorpusDeployment | None = None

    @classmethod
    def read(cls, path: Path) -> TrainingDeployments:
        if not path.is_absolute() or not path.is_file():
            raise ValueError("training deployment input must be an absolute file")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("format") != SESSION_DEPLOYMENTS_FORMAT:
            raise ValueError("training deployment format is unsupported")
        return cls(
            alfworld=ALFWorldDeployment.from_value(data.get("alfworld")),
            healthbench=HealthBenchDeployment.from_value(data.get("healthbench")),
            triviaqa_wikipedia=WikipediaCorpusDeployment.from_value(data["triviaqa_wikipedia"])
            if data.get("triviaqa_wikipedia") is not None
            else None,
        )
