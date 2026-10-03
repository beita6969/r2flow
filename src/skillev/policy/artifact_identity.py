from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

from skillev.contracts import JsonValue, normalize_json, stable_hash
from skillev.contracts.identity import validate_sha256

ARTIFACT_FILE_IDENTITY_FORMAT: Final = "skillev-artifact-file-identity@1"
BASE_MODEL_ARTIFACT_IDENTITY_FORMAT: Final = "skillev-base-model-artifact@1"

_WEIGHT_INDEX_NAMES: Final = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _relative_path(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be non-empty relative text")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"{field} must be a normalized relative path")
    return value


@dataclass(frozen=True, slots=True)
class ArtifactFileIdentity:
    relative_path: str
    size_bytes: int
    sha256: str
    format: str = ARTIFACT_FILE_IDENTITY_FORMAT

    def __post_init__(self) -> None:
        _relative_path(self.relative_path, field="relative_path")
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise ValueError("artifact file size_bytes must be a positive integer")
        validate_sha256(self.sha256)
        if self.format != ARTIFACT_FILE_IDENTITY_FORMAT:
            raise ValueError("unsupported artifact file identity format")

    @classmethod
    def from_root(cls, *, root: Path, path: Path) -> ArtifactFileIdentity:
        if not root.is_dir():
            raise NotADirectoryError(root)
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            relative_path = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError("artifact file must belong to its declared root") from error
        return cls(
            relative_path=relative_path,
            size_bytes=path.stat().st_size,
            sha256=_sha256_file(path),
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_value(cls, value: object) -> ArtifactFileIdentity:
        normalized = normalize_json(value)
        fields = {"format", "relative_path", "sha256", "size_bytes"}
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("artifact file identity has incompatible fields")
        if any(type(normalized[field]) is not str for field in fields - {"size_bytes"}):
            raise TypeError("artifact file identity text fields must be strings")
        if type(normalized["size_bytes"]) is not int:
            raise TypeError("artifact file identity size_bytes must be an integer")
        return cls(
            relative_path=normalized["relative_path"],
            size_bytes=normalized["size_bytes"],
            sha256=normalized["sha256"],
            format=normalized["format"],
        )


@dataclass(frozen=True, slots=True)
class BaseModelArtifactIdentity:
    backend_class: str
    upstream_revision: str
    dtype_conversion_policy: str
    model_config: ArtifactFileIdentity
    generation_config: ArtifactFileIdentity | None
    weight_index: ArtifactFileIdentity | None
    weight_shards: tuple[ArtifactFileIdentity, ...]
    content_hash: str
    format: str = BASE_MODEL_ARTIFACT_IDENTITY_FORMAT

    def __post_init__(self) -> None:
        for field in ("backend_class", "upstream_revision", "dtype_conversion_policy"):
            if type(getattr(self, field)) is not str or not getattr(self, field):
                raise ValueError(f"{field} must be non-empty text")
        if not isinstance(self.model_config, ArtifactFileIdentity):
            raise TypeError("base model identity requires a model config file")
        if self.model_config.relative_path != "config.json":
            raise ValueError("base model identity model_config must be config.json")
        if self.generation_config is not None and not isinstance(
            self.generation_config, ArtifactFileIdentity
        ):
            raise TypeError("base model generation_config must be an artifact file or None")
        if (
            self.generation_config is not None
            and self.generation_config.relative_path != "generation_config.json"
        ):
            raise ValueError("base model identity generation_config must be generation_config.json")
        if self.weight_index is not None and not isinstance(
            self.weight_index, ArtifactFileIdentity
        ):
            raise TypeError("base model weight_index must be an artifact file or None")
        if not isinstance(self.weight_shards, tuple) or not self.weight_shards:
            raise ValueError("base model identity requires weight shards")
        if any(not isinstance(item, ArtifactFileIdentity) for item in self.weight_shards):
            raise TypeError("base model weight shards must be artifact files")
        shard_paths = tuple(item.relative_path for item in self.weight_shards)
        if shard_paths != tuple(sorted(shard_paths)) or len(set(shard_paths)) != len(shard_paths):
            raise ValueError("base model weight shards must be unique and sorted")
        validate_sha256(self.content_hash)
        if self.format != BASE_MODEL_ARTIFACT_IDENTITY_FORMAT:
            raise ValueError("unsupported base model artifact identity format")
        if self.content_hash != stable_hash(self._content_value()):
            raise ValueError("base model artifact content_hash differs from manifest")

    def _content_value(self) -> dict[str, JsonValue]:
        return {
            "backend_class": self.backend_class,
            "dtype_conversion_policy": self.dtype_conversion_policy,
            "generation_config": (
                self.generation_config.to_value() if self.generation_config else None
            ),
            "model_config": self.model_config.to_value(),
            "upstream_revision": self.upstream_revision,
            "weight_index": self.weight_index.to_value() if self.weight_index else None,
            "weight_shards": [item.to_value() for item in self.weight_shards],
        }

    @classmethod
    def create(
        cls,
        *,
        backend_class: str,
        upstream_revision: str,
        dtype_conversion_policy: str,
        model_config: ArtifactFileIdentity,
        generation_config: ArtifactFileIdentity | None,
        weight_index: ArtifactFileIdentity | None,
        weight_shards: tuple[ArtifactFileIdentity, ...],
    ) -> BaseModelArtifactIdentity:
        payload: dict[str, JsonValue] = {
            "backend_class": backend_class,
            "dtype_conversion_policy": dtype_conversion_policy,
            "generation_config": generation_config.to_value() if generation_config else None,
            "model_config": model_config.to_value(),
            "upstream_revision": upstream_revision,
            "weight_index": weight_index.to_value() if weight_index else None,
            "weight_shards": [item.to_value() for item in weight_shards],
        }
        return cls(
            backend_class=backend_class,
            upstream_revision=upstream_revision,
            dtype_conversion_policy=dtype_conversion_policy,
            model_config=model_config,
            generation_config=generation_config,
            weight_index=weight_index,
            weight_shards=weight_shards,
            content_hash=stable_hash(payload),
        )

    @classmethod
    def from_directory(
        cls,
        *,
        directory: Path,
        backend_class: str,
        upstream_revision: str,
        dtype_conversion_policy: str,
    ) -> BaseModelArtifactIdentity:
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        for field, value in (
            ("backend_class", backend_class),
            ("upstream_revision", upstream_revision),
            ("dtype_conversion_policy", dtype_conversion_policy),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{field} must be non-empty text")
        model_config = ArtifactFileIdentity.from_root(
            root=directory, path=directory / "config.json"
        )
        generation_config_path = directory / "generation_config.json"
        generation_config = (
            ArtifactFileIdentity.from_root(
                root=directory,
                path=generation_config_path,
            )
            if generation_config_path.is_file()
            else None
        )
        index_paths = [
            directory / name for name in _WEIGHT_INDEX_NAMES if (directory / name).is_file()
        ]
        if len(index_paths) > 1:
            raise ValueError("base model directory contains multiple incompatible weight indexes")
        weight_index = (
            ArtifactFileIdentity.from_root(root=directory, path=index_paths[0])
            if index_paths
            else None
        )
        shard_paths = _weight_paths(
            directory=directory, index_path=index_paths[0] if index_paths else None
        )
        weight_shards = tuple(
            ArtifactFileIdentity.from_root(root=directory, path=path)
            for path in sorted(shard_paths, key=lambda item: item.relative_to(directory).as_posix())
        )
        return cls.create(
            backend_class=backend_class,
            upstream_revision=upstream_revision,
            dtype_conversion_policy=dtype_conversion_policy,
            model_config=model_config,
            generation_config=generation_config,
            weight_index=weight_index,
            weight_shards=weight_shards,
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            **self._content_value(),
            "content_hash": self.content_hash,
            "format": self.format,
        }

    @classmethod
    def from_value(cls, value: object) -> BaseModelArtifactIdentity:
        normalized = normalize_json(value)
        fields = {
            "backend_class",
            "content_hash",
            "dtype_conversion_policy",
            "format",
            "generation_config",
            "model_config",
            "upstream_revision",
            "weight_index",
            "weight_shards",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("base model artifact identity has incompatible fields")
        for field in (
            "backend_class",
            "content_hash",
            "dtype_conversion_policy",
            "format",
            "upstream_revision",
        ):
            if type(normalized[field]) is not str:
                raise TypeError("base model artifact identity text fields must be strings")
        raw_shards = normalized["weight_shards"]
        if not isinstance(raw_shards, list):
            raise TypeError("base model weight_shards must be an array")
        raw_index = normalized["weight_index"]
        if raw_index is not None and not isinstance(raw_index, dict):
            raise TypeError("base model weight_index must be an object or null")
        raw_generation_config = normalized["generation_config"]
        if raw_generation_config is not None and not isinstance(raw_generation_config, dict):
            raise TypeError("base model generation_config must be an object or null")
        return cls(
            backend_class=normalized["backend_class"],
            upstream_revision=normalized["upstream_revision"],
            dtype_conversion_policy=normalized["dtype_conversion_policy"],
            model_config=ArtifactFileIdentity.from_value(normalized["model_config"]),
            generation_config=(
                ArtifactFileIdentity.from_value(raw_generation_config)
                if raw_generation_config
                else None
            ),
            weight_index=(ArtifactFileIdentity.from_value(raw_index) if raw_index else None),
            weight_shards=tuple(ArtifactFileIdentity.from_value(item) for item in raw_shards),
            content_hash=normalized["content_hash"],
            format=normalized["format"],
        )


def _weight_paths(*, directory: Path, index_path: Path | None) -> tuple[Path, ...]:
    if index_path is not None:
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("base model weight index is not valid JSON") from error
        if not isinstance(payload, dict) or set(payload) - {"metadata", "weight_map"}:
            raise ValueError("base model weight index has unsupported fields")
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("base model weight index requires a non-empty weight_map")
        names = tuple(sorted(set(weight_map.values())))
        if any(type(name) is not str or not name for name in names):
            raise ValueError("base model weight index shard names must be non-empty text")
        paths: list[Path] = []
        for name in names:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
                raise ValueError("base model weight index contains an invalid shard path")
            path = directory / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(path)
        return tuple(paths)
    candidates = tuple(
        sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file()
                and (path.name == "model.safetensors" or path.name == "pytorch_model.bin")
            ),
            key=lambda item: item.name,
        )
    )
    if len(candidates) != 1:
        raise ValueError("unindexed base model must contain exactly one recognized weight file")
    return candidates


__all__ = [
    "ARTIFACT_FILE_IDENTITY_FORMAT",
    "BASE_MODEL_ARTIFACT_IDENTITY_FORMAT",
    "ArtifactFileIdentity",
    "BaseModelArtifactIdentity",
]
