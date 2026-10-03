from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from skillev.contracts import JsonValue, normalize_json, stable_hash
from skillev.contracts.identity import validate_sha256

PUBLIC_TOKENIZER_IDENTITY_FORMAT: Final = "skillev-public-tokenizer-identity@2"
TOKENIZER_ARTIFACT_IDENTITY_FORMAT: Final = "skillev-tokenizer-artifact@1"


class PublicTokenizerKind(StrEnum):
    QWEN = "qwen"


@dataclass(frozen=True, slots=True)
class TokenizerArtifactIdentity:
    kind: PublicTokenizerKind
    tokenizer_id: str
    revision: str
    backend_serialization_hash: str
    tokenizer_config_hash: str
    chat_template_hash: str
    special_tokens_hash: str
    added_tokens_hash: str
    transformers_version: str
    tokenizers_version: str
    content_hash: str
    format: str = TOKENIZER_ARTIFACT_IDENTITY_FORMAT

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PublicTokenizerKind):
            raise TypeError("tokenizer artifact kind must be PublicTokenizerKind")
        for field in (
            "tokenizer_id",
            "revision",
            "transformers_version",
            "tokenizers_version",
        ):
            if type(getattr(self, field)) is not str or not getattr(self, field):
                raise ValueError(f"{field} must be non-empty")
        for field in (
            "backend_serialization_hash",
            "tokenizer_config_hash",
            "chat_template_hash",
            "special_tokens_hash",
            "added_tokens_hash",
            "content_hash",
        ):
            validate_sha256(getattr(self, field))
        if self.format != TOKENIZER_ARTIFACT_IDENTITY_FORMAT:
            raise ValueError("unsupported tokenizer artifact identity format")
        if self.content_hash != stable_hash(self._content_value()):
            raise ValueError("tokenizer artifact content_hash differs from components")

    @classmethod
    def create(
        cls,
        *,
        kind: PublicTokenizerKind,
        tokenizer_id: str,
        revision: str,
        backend_serialization_hash: str,
        tokenizer_config_hash: str,
        chat_template_hash: str,
        special_tokens_hash: str,
        added_tokens_hash: str,
        transformers_version: str,
        tokenizers_version: str,
    ) -> TokenizerArtifactIdentity:
        content = {
            "added_tokens_hash": added_tokens_hash,
            "backend_serialization_hash": backend_serialization_hash,
            "chat_template_hash": chat_template_hash,
            "kind": kind.value,
            "revision": revision,
            "special_tokens_hash": special_tokens_hash,
            "tokenizer_config_hash": tokenizer_config_hash,
            "tokenizer_id": tokenizer_id,
            "tokenizers_version": tokenizers_version,
            "transformers_version": transformers_version,
        }
        return cls(
            kind=kind,
            tokenizer_id=tokenizer_id,
            revision=revision,
            backend_serialization_hash=backend_serialization_hash,
            tokenizer_config_hash=tokenizer_config_hash,
            chat_template_hash=chat_template_hash,
            special_tokens_hash=special_tokens_hash,
            added_tokens_hash=added_tokens_hash,
            transformers_version=transformers_version,
            tokenizers_version=tokenizers_version,
            content_hash=stable_hash(content),
        )

    def _content_value(self) -> dict[str, JsonValue]:
        return {
            "added_tokens_hash": self.added_tokens_hash,
            "backend_serialization_hash": self.backend_serialization_hash,
            "chat_template_hash": self.chat_template_hash,
            "kind": self.kind.value,
            "revision": self.revision,
            "special_tokens_hash": self.special_tokens_hash,
            "tokenizer_config_hash": self.tokenizer_config_hash,
            "tokenizer_id": self.tokenizer_id,
            "tokenizers_version": self.tokenizers_version,
            "transformers_version": self.transformers_version,
        }

    def to_value(self) -> dict[str, JsonValue]:
        return {
            **self._content_value(),
            "content_hash": self.content_hash,
            "format": self.format,
        }

    @classmethod
    def from_value(cls, value: object) -> TokenizerArtifactIdentity:
        normalized = normalize_json(value)
        fields = {
            "added_tokens_hash",
            "backend_serialization_hash",
            "chat_template_hash",
            "content_hash",
            "format",
            "kind",
            "revision",
            "special_tokens_hash",
            "tokenizer_config_hash",
            "tokenizer_id",
            "tokenizers_version",
            "transformers_version",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("tokenizer artifact identity has incompatible fields")
        if any(type(normalized[field]) is not str for field in fields):
            raise TypeError("tokenizer artifact identity fields must be text")
        return cls(
            kind=PublicTokenizerKind(normalized["kind"]),
            tokenizer_id=normalized["tokenizer_id"],
            revision=normalized["revision"],
            backend_serialization_hash=normalized["backend_serialization_hash"],
            tokenizer_config_hash=normalized["tokenizer_config_hash"],
            chat_template_hash=normalized["chat_template_hash"],
            special_tokens_hash=normalized["special_tokens_hash"],
            added_tokens_hash=normalized["added_tokens_hash"],
            transformers_version=normalized["transformers_version"],
            tokenizers_version=normalized["tokenizers_version"],
            content_hash=normalized["content_hash"],
            format=normalized["format"],
        )


@dataclass(frozen=True, slots=True)
class PublicTokenizerIdentity:
    kind: PublicTokenizerKind
    tokenizer_id: str
    revision: str
    content_hash: str
    format: str = PUBLIC_TOKENIZER_IDENTITY_FORMAT

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PublicTokenizerKind):
            raise TypeError("tokenizer kind must be PublicTokenizerKind")
        for field in ("tokenizer_id", "revision"):
            if type(getattr(self, field)) is not str or not getattr(self, field):
                raise ValueError(f"{field} must be non-empty")
        validate_sha256(self.content_hash)
        if self.format != PUBLIC_TOKENIZER_IDENTITY_FORMAT:
            raise ValueError("unsupported public tokenizer identity format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "content_hash": self.content_hash,
            "format": self.format,
            "kind": self.kind.value,
            "revision": self.revision,
            "tokenizer_id": self.tokenizer_id,
        }

    @classmethod
    def from_value(cls, value: object) -> PublicTokenizerIdentity:
        normalized = normalize_json(value)
        fields = {"content_hash", "format", "kind", "revision", "tokenizer_id"}
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("PublicTokenizerIdentity has incompatible fields")
        if any(type(normalized[field]) is not str for field in fields):
            raise TypeError("PublicTokenizerIdentity fields must be text")
        return cls(
            kind=PublicTokenizerKind(normalized["kind"]),
            tokenizer_id=normalized["tokenizer_id"],
            revision=normalized["revision"],
            content_hash=normalized["content_hash"],
            format=normalized["format"],
        )


__all__ = [
    "PUBLIC_TOKENIZER_IDENTITY_FORMAT",
    "TOKENIZER_ARTIFACT_IDENTITY_FORMAT",
    "PublicTokenizerIdentity",
    "PublicTokenizerKind",
    "TokenizerArtifactIdentity",
]
