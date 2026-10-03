from __future__ import annotations

import os
import re
import shutil
import sqlite3
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from urllib.parse import quote

from skillev.contracts import (
    JsonValue,
    canonical_json,
    canonical_json_bytes,
    normalize_json,
    parse_canonical_json,
    stable_hash,
    validate_sha256,
)
from skillev.runtime import (
    ActionKind,
    BudgetVector,
    EnvironmentMethodFailedError,
    EnvironmentObservation,
    StructuredAction,
)

INDEX_FORMAT = "skillev-public-retrieval-index@2"
RETRIEVAL_BACKEND = "sqlite-fts5-lexical"
FTS_TOKENIZER = "unicode61 remove_diacritics 2"
_SNIPPET_CODEPOINTS = 320
_QUERY_TERM = re.compile(r"\w+", flags=re.UNICODE)
_EPHEMERAL_BUILD_CACHE_KIB = 2 * 1024 * 1024
_COPY_BUFFER_BYTES = 16 * 1024 * 1024


def _object(value: object, *, fields: frozenset[str], label: str) -> dict[str, JsonValue]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be a JSON object")
    normalized = normalize_json(value)
    if normalized != value or not isinstance(normalized, dict):
        raise TypeError(f"{label} must be normalized JSON")
    if set(normalized) != fields:
        raise ValueError(f"{label} has an incompatible field set")
    return normalized


def _text(value: object, *, field: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field} must be text")
    normalized = normalize_json(value)
    if type(normalized) is not str or normalized != value:
        raise ValueError(f"{field} must be canonical text")
    if not value.strip() or "\x00" in value:
        raise ValueError(f"{field} must be non-empty text without NUL")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class DocumentPassage:
    passage_id: str
    document_id: str
    title: str
    text: str
    source_rowid: int | None = dataclass_field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        for field in ("passage_id", "document_id", "title", "text"):
            _text(getattr(self, field), field=field)
        if self.source_rowid is not None:
            _positive_int(self.source_rowid, field="source_rowid")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "document_id": self.document_id,
            "passage_id": self.passage_id,
            "text": self.text,
            "title": self.title,
        }

    @classmethod
    def from_value(cls, value: object) -> DocumentPassage:
        data = _object(
            value,
            fields=frozenset({"document_id", "passage_id", "text", "title"}),
            label="document passage",
        )
        return cls(
            passage_id=_text(data["passage_id"], field="passage_id"),
            document_id=_text(data["document_id"], field="document_id"),
            title=_text(data["title"], field="title"),
            text=_text(data["text"], field="text"),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


class RetrievalBuildPhase(StrEnum):
    INGEST = "ingest"
    UNIQUE_INDEX = "unique-index"
    FTS_INDEX = "fts-index"
    CORPUS_HASH = "corpus-hash"
    VERIFY = "verify"
    PUBLISH = "publish"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class RetrievalBuildProgress:
    phase: RetrievalBuildPhase
    rows_completed: int
    expected_rows: int | None
    elapsed_seconds: float
    rows_per_second: float
    eta_seconds: float | None


@dataclass(frozen=True, slots=True)
class SearchHit:
    passage_id: str
    document_id: str
    title: str
    snippet: str
    rank: int

    def __post_init__(self) -> None:
        for field in ("passage_id", "document_id", "title", "snippet"):
            _text(getattr(self, field), field=field)
        _positive_int(self.rank, field="rank")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "document_id": self.document_id,
            "passage_id": self.passage_id,
            "rank": self.rank,
            "snippet": self.snippet,
            "title": self.title,
        }

    @classmethod
    def from_value(cls, value: object) -> SearchHit:
        data = _object(
            value,
            fields=frozenset({"document_id", "passage_id", "rank", "snippet", "title"}),
            label="search hit",
        )
        return cls(
            passage_id=_text(data["passage_id"], field="passage_id"),
            document_id=_text(data["document_id"], field="document_id"),
            title=_text(data["title"], field="title"),
            snippet=_text(data["snippet"], field="snippet"),
            rank=_positive_int(data["rank"], field="rank"),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class RetrievalIndexManifest:
    index_id: str
    corpus_name: str
    corpus_version: str
    passage_count: int
    corpus_hash: str
    retrieval_backend: str = RETRIEVAL_BACKEND
    fts_tokenizer: str = FTS_TOKENIZER
    format: str = INDEX_FORMAT

    def __post_init__(self) -> None:
        if self.format != INDEX_FORMAT:
            raise ValueError("retrieval index format is unsupported")
        _text(self.corpus_name, field="corpus_name")
        _text(self.corpus_version, field="corpus_version")
        _positive_int(self.passage_count, field="passage_count")
        validate_sha256(self.corpus_hash)
        if self.retrieval_backend != RETRIEVAL_BACKEND:
            raise ValueError("retrieval backend identity is unsupported")
        if self.fts_tokenizer != FTS_TOKENIZER:
            raise ValueError("retrieval tokenizer identity is unsupported")
        validate_sha256(self.index_id)
        if self.index_id != stable_hash(self._identity_value()):
            raise ValueError("retrieval index ID does not match its content identity")

    @classmethod
    def create(
        cls,
        passages: tuple[DocumentPassage, ...],
        *,
        corpus_name: str,
        corpus_version: str,
    ) -> RetrievalIndexManifest:
        ordered = _ordered_passages(passages)
        passage_count, corpus_hash = _hash_ordered_passages(iter(ordered))
        return cls.create_from_corpus_hash(
            corpus_name=corpus_name,
            corpus_version=corpus_version,
            passage_count=passage_count,
            corpus_hash=corpus_hash,
        )

    @classmethod
    def create_from_corpus_hash(
        cls,
        *,
        corpus_name: str,
        corpus_version: str,
        passage_count: int,
        corpus_hash: str,
    ) -> RetrievalIndexManifest:
        count = _positive_int(passage_count, field="passage_count")
        validate_sha256(corpus_hash)
        identity: dict[str, JsonValue] = {
            "corpus_hash": corpus_hash,
            "corpus_name": _text(corpus_name, field="corpus_name"),
            "corpus_version": _text(corpus_version, field="corpus_version"),
            "format": INDEX_FORMAT,
            "fts_tokenizer": FTS_TOKENIZER,
            "passage_count": count,
            "retrieval_backend": RETRIEVAL_BACKEND,
        }
        return cls(
            index_id=stable_hash(identity),
            corpus_name=corpus_name,
            corpus_version=corpus_version,
            passage_count=count,
            corpus_hash=corpus_hash,
        )

    def _identity_value(self) -> dict[str, JsonValue]:
        return {
            "corpus_hash": self.corpus_hash,
            "corpus_name": self.corpus_name,
            "corpus_version": self.corpus_version,
            "format": self.format,
            "fts_tokenizer": self.fts_tokenizer,
            "passage_count": self.passage_count,
            "retrieval_backend": self.retrieval_backend,
        }

    def to_value(self) -> dict[str, JsonValue]:
        return {"index_id": self.index_id, **self._identity_value()}

    @classmethod
    def from_value(cls, value: object) -> RetrievalIndexManifest:
        data = _object(
            value,
            fields=frozenset(
                {
                    "corpus_hash",
                    "corpus_name",
                    "corpus_version",
                    "format",
                    "fts_tokenizer",
                    "index_id",
                    "passage_count",
                    "retrieval_backend",
                }
            ),
            label="retrieval index manifest",
        )
        return cls(
            index_id=_text(data["index_id"], field="index_id"),
            corpus_name=_text(data["corpus_name"], field="corpus_name"),
            corpus_version=_text(data["corpus_version"], field="corpus_version"),
            passage_count=_positive_int(data["passage_count"], field="passage_count"),
            corpus_hash=_text(data["corpus_hash"], field="corpus_hash"),
            retrieval_backend=_text(data["retrieval_backend"], field="retrieval_backend"),
            fts_tokenizer=_text(data["fts_tokenizer"], field="fts_tokenizer"),
            format=_text(data["format"], field="format"),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


def _ordered_passages(passages: tuple[DocumentPassage, ...]) -> tuple[DocumentPassage, ...]:
    if type(passages) is not tuple or not passages:
        raise ValueError("retrieval corpus must be a non-empty passage tuple")
    if any(not isinstance(passage, DocumentPassage) for passage in passages):
        raise TypeError("retrieval corpus contains a non-passage item")
    ordered = tuple(sorted(passages, key=lambda passage: passage.passage_id))
    identities = tuple(passage.passage_id for passage in ordered)
    if len(set(identities)) != len(identities):
        raise ValueError("retrieval passage IDs must be unique")
    return ordered


def _hash_ordered_passages(
    passages: Iterator[DocumentPassage],
) -> tuple[int, str]:
    digest = sha256()
    digest.update(b"[")
    count = 0
    previous_id: str | None = None
    for passage in passages:
        if not isinstance(passage, DocumentPassage):
            raise TypeError("retrieval corpus contains a non-passage item")
        if previous_id is not None and passage.passage_id <= previous_id:
            raise ValueError("retrieval passages must be strictly ordered by passage ID")
        if count:
            digest.update(b",")
        digest.update(canonical_json_bytes(passage.to_value()))
        previous_id = passage.passage_id
        count += 1
    digest.update(b"]")
    if count < 1:
        raise ValueError("retrieval corpus must contain at least one passage")
    return count, f"sha256:{digest.hexdigest()}"


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _configure_ephemeral_index_build(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(f"PRAGMA cache_size=-{_EPHEMERAL_BUILD_CACHE_KIB}")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA locking_mode=EXCLUSIVE")


def _same_filesystem(first: Path, second_directory: Path) -> bool:
    return first.stat().st_dev == second_directory.stat().st_dev


def _copy_fsync(source: Path, destination: Path) -> None:
    with source.open("rb") as source_stream, destination.open("wb") as destination_stream:
        shutil.copyfileobj(source_stream, destination_stream, length=_COPY_BUFFER_BYTES)
        destination_stream.flush()
        os.fsync(destination_stream.fileno())


def _publish_completed_index(source: Path, target: Path) -> None:
    if _same_filesystem(source, target.parent):
        os.replace(source, target)
        _fsync_directory(target.parent)
        return

    descriptor, publication_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".publish",
        dir=target.parent,
    )
    os.close(descriptor)
    publication = Path(publication_name)
    try:
        _copy_fsync(source, publication)
        os.replace(publication, target)
        _fsync_directory(target.parent)
    finally:
        if publication.exists():
            publication.unlink()


def _report_build_progress(
    callback: Callable[[RetrievalBuildProgress], None] | None,
    *,
    phase: RetrievalBuildPhase,
    rows_completed: int,
    expected_rows: int | None,
    started_at: float,
) -> None:
    if callback is None:
        return
    elapsed = max(0.0, time.monotonic() - started_at)
    rate = rows_completed / elapsed if elapsed > 0.0 else 0.0
    eta = None
    if (
        phase
        in (
            RetrievalBuildPhase.INGEST,
            RetrievalBuildPhase.FTS_INDEX,
        )
        and expected_rows is not None
        and rows_completed < expected_rows
        and rate > 0.0
    ):
        eta = (expected_rows - rows_completed) / rate
    callback(
        RetrievalBuildProgress(
            phase=phase,
            rows_completed=rows_completed,
            expected_rows=expected_rows,
            elapsed_seconds=elapsed,
            rows_per_second=rate,
            eta_seconds=eta,
        )
    )


def build_retrieval_index(
    path: str | os.PathLike[str],
    passages: tuple[DocumentPassage, ...],
    *,
    corpus_name: str,
    corpus_version: str,
) -> RetrievalIndexManifest:
    ordered = _ordered_passages(passages)
    return build_retrieval_index_stream(
        path,
        ordered,
        corpus_name=corpus_name,
        corpus_version=corpus_version,
    )


def build_retrieval_index_stream(
    path: str | os.PathLike[str],
    passages: Iterable[DocumentPassage],
    *,
    corpus_name: str,
    corpus_version: str,
    insert_batch_size: int = 10_000,
    staging_directory: str | os.PathLike[str] | None = None,
    expected_passage_count: int | None = None,
    progress_callback: Callable[[RetrievalBuildProgress], None] | None = None,
    progress_interval_rows: int = 100_000,
) -> RetrievalIndexManifest:
    started_at = time.monotonic()
    target = Path(path)
    _text(corpus_name, field="corpus_name")
    _text(corpus_version, field="corpus_version")
    batch_size = _positive_int(insert_batch_size, field="insert_batch_size")
    expected_rows = (
        None
        if expected_passage_count is None
        else _positive_int(expected_passage_count, field="expected_passage_count")
    )
    progress_interval = _positive_int(progress_interval_rows, field="progress_interval_rows")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")
    if not isinstance(passages, Iterable):
        raise TypeError("retrieval corpus must be iterable")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging_root = target.parent if staging_directory is None else Path(staging_directory)
    staging_root.mkdir(parents=True, exist_ok=True)
    if not staging_root.is_dir():
        raise ValueError("retrieval staging directory must be a directory")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=staging_root,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        connection = sqlite3.connect(temporary)
        try:
            _configure_ephemeral_index_build(connection)
            connection.execute("PRAGMA user_version=1")
            connection.execute(
                "CREATE TABLE passages ("
                "rowid INTEGER PRIMARY KEY, "
                "passage_id TEXT NOT NULL, "
                "document_id TEXT NOT NULL, "
                "title TEXT NOT NULL, "
                "text TEXT NOT NULL)"
            )
            connection.execute("CREATE TABLE retrieval_manifest (manifest_json TEXT NOT NULL)")
            connection.commit()
            batch: list[tuple[int | None, str, str, str, str]] = []
            passage_count = 0
            next_progress_row = progress_interval
            last_progress_row = 0
            previous_source_rowid: int | None = None
            previous_passage_id: str | None = None
            passage_order_is_canonical = True
            streaming_corpus_digest = sha256()
            streaming_corpus_digest.update(b"[")
            source_rows_seen = 0
            for passage in passages:
                if not isinstance(passage, DocumentPassage):
                    raise TypeError("retrieval corpus contains a non-passage item")
                if passage.source_rowid is not None:
                    if (
                        previous_source_rowid is not None
                        and passage.source_rowid <= previous_source_rowid
                    ):
                        raise ValueError("retrieval source rowids must be strictly increasing")
                    previous_source_rowid = passage.source_rowid
                if previous_passage_id is not None and passage.passage_id <= previous_passage_id:
                    passage_order_is_canonical = False
                if passage_order_is_canonical:
                    if source_rows_seen:
                        streaming_corpus_digest.update(b",")
                    streaming_corpus_digest.update(canonical_json_bytes(passage.to_value()))
                previous_passage_id = passage.passage_id
                source_rows_seen += 1
                batch.append(
                    (
                        passage.source_rowid,
                        passage.passage_id,
                        passage.document_id,
                        passage.title,
                        passage.text,
                    )
                )
                if len(batch) == batch_size:
                    _insert_passage_batch(connection, batch)
                    connection.commit()
                    passage_count += len(batch)
                    batch.clear()
                    if passage_count >= next_progress_row:
                        _report_build_progress(
                            progress_callback,
                            phase=RetrievalBuildPhase.INGEST,
                            rows_completed=passage_count,
                            expected_rows=expected_rows,
                            started_at=started_at,
                        )
                        last_progress_row = passage_count
                        while next_progress_row <= passage_count:
                            next_progress_row += progress_interval
            if batch:
                _insert_passage_batch(connection, batch)
                connection.commit()
                passage_count += len(batch)
                _report_build_progress(
                    progress_callback,
                    phase=RetrievalBuildPhase.INGEST,
                    rows_completed=passage_count,
                    expected_rows=expected_rows,
                    started_at=started_at,
                )
                last_progress_row = passage_count
            if passage_count < 1:
                raise ValueError("retrieval corpus must contain at least one passage")
            if last_progress_row != passage_count:
                _report_build_progress(
                    progress_callback,
                    phase=RetrievalBuildPhase.INGEST,
                    rows_completed=passage_count,
                    expected_rows=expected_rows,
                    started_at=started_at,
                )
            if expected_rows is not None and passage_count != expected_rows:
                raise ValueError("retrieval corpus passage count differs from its pinned identity")
            _report_build_progress(
                progress_callback,
                phase=RetrievalBuildPhase.UNIQUE_INDEX,
                rows_completed=passage_count,
                expected_rows=expected_rows,
                started_at=started_at,
            )
            try:
                connection.execute(
                    "CREATE UNIQUE INDEX passages_passage_id_uq ON passages(passage_id)"
                )
            except sqlite3.IntegrityError as error:
                raise ValueError("retrieval passage IDs must be unique") from error
            connection.commit()
            connection.execute(
                "CREATE VIRTUAL TABLE passage_fts USING fts5("
                "title, text, content='passages', content_rowid='rowid', "
                f"tokenize='{FTS_TOKENIZER}')"
            )
            fts_started_at = time.monotonic()
            _populate_fts_in_batches(
                connection,
                passage_count=passage_count,
                batch_size=max(batch_size, progress_interval),
                progress_interval=progress_interval,
                progress_callback=progress_callback,
                started_at=fts_started_at,
            )
            _report_build_progress(
                progress_callback,
                phase=RetrievalBuildPhase.CORPUS_HASH,
                rows_completed=passage_count,
                expected_rows=expected_rows,
                started_at=started_at,
            )
            if passage_order_is_canonical:
                streaming_corpus_digest.update(b"]")
                hashed_count = source_rows_seen
                corpus_hash = f"sha256:{streaming_corpus_digest.hexdigest()}"
            else:
                ordered_rows = connection.execute(
                    "SELECT passage_id, document_id, title, text FROM passages ORDER BY passage_id"
                )
                hashed_count, corpus_hash = _hash_ordered_passages(
                    _passages_from_rows(ordered_rows)
                )
            if hashed_count != passage_count:
                raise ValueError("retrieval index row count changed during construction")
            manifest = RetrievalIndexManifest.create_from_corpus_hash(
                corpus_name=corpus_name,
                corpus_version=corpus_version,
                passage_count=passage_count,
                corpus_hash=corpus_hash,
            )
            connection.execute(
                "INSERT INTO retrieval_manifest(manifest_json) VALUES (?)",
                (canonical_json(manifest.to_value()),),
            )
            connection.commit()
        finally:
            connection.close()
        _report_build_progress(
            progress_callback,
            phase=RetrievalBuildPhase.VERIFY,
            rows_completed=manifest.passage_count,
            expected_rows=expected_rows,
            started_at=started_at,
        )
        verify_built_index(temporary, expected_manifest=manifest)
        _fsync_file(temporary)
        _report_build_progress(
            progress_callback,
            phase=RetrievalBuildPhase.PUBLISH,
            rows_completed=manifest.passage_count,
            expected_rows=expected_rows,
            started_at=started_at,
        )
        _publish_completed_index(temporary, target)
        _report_build_progress(
            progress_callback,
            phase=RetrievalBuildPhase.COMPLETE,
            rows_completed=manifest.passage_count,
            expected_rows=expected_rows,
            started_at=started_at,
        )
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifest


def _insert_passage_batch(
    connection: sqlite3.Connection,
    batch: list[tuple[int | None, str, str, str, str]],
) -> None:
    connection.executemany(
        "INSERT INTO passages(rowid, passage_id, document_id, title, text) VALUES (?, ?, ?, ?, ?)",
        batch,
    )


def _populate_fts_in_batches(
    connection: sqlite3.Connection,
    *,
    passage_count: int,
    batch_size: int,
    progress_interval: int,
    progress_callback: Callable[[RetrievalBuildProgress], None] | None,
    started_at: float,
) -> None:
    last_rowid = 0
    indexed_rows = 0
    next_progress_row = progress_interval
    last_progress_row = 0
    while indexed_rows < passage_count:
        boundary = connection.execute(
            "SELECT max(rowid), count(*) FROM ("
            "SELECT rowid FROM passages WHERE rowid > ? ORDER BY rowid LIMIT ?)",
            (last_rowid, batch_size),
        ).fetchone()
        if (
            boundary is None
            or type(boundary[0]) is not int
            or type(boundary[1]) is not int
            or boundary[1] < 1
        ):
            raise ValueError("retrieval FTS source rows ended before the passage table")
        upper_rowid, batch_count = boundary
        connection.execute(
            "INSERT INTO passage_fts(rowid, title, text) "
            "SELECT rowid, title, text FROM passages "
            "WHERE rowid > ? AND rowid <= ? ORDER BY rowid",
            (last_rowid, upper_rowid),
        )
        connection.commit()
        last_rowid = upper_rowid
        indexed_rows += batch_count
        if indexed_rows >= next_progress_row:
            _report_build_progress(
                progress_callback,
                phase=RetrievalBuildPhase.FTS_INDEX,
                rows_completed=indexed_rows,
                expected_rows=passage_count,
                started_at=started_at,
            )
            last_progress_row = indexed_rows
            while next_progress_row <= indexed_rows:
                next_progress_row += progress_interval
    if indexed_rows != passage_count:
        raise ValueError("retrieval FTS indexed row count differs from the passage table")
    if last_progress_row != indexed_rows:
        _report_build_progress(
            progress_callback,
            phase=RetrievalBuildPhase.FTS_INDEX,
            rows_completed=indexed_rows,
            expected_rows=passage_count,
            started_at=started_at,
        )


def _passages_from_rows(rows: Iterable[tuple[object, ...]]) -> Iterator[DocumentPassage]:
    for row in rows:
        if len(row) != 4:
            raise ValueError("retrieval passage row has an incompatible shape")
        yield DocumentPassage(
            passage_id=_text(row[0], field="passage_id"),
            document_id=_text(row[1], field="document_id"),
            title=_text(row[2], field="title"),
            text=_text(row[3], field="text"),
        )


def _read_index_manifest(connection: sqlite3.Connection) -> RetrievalIndexManifest:
    manifest_rows = connection.execute("SELECT manifest_json FROM retrieval_manifest").fetchall()
    if len(manifest_rows) != 1 or type(manifest_rows[0][0]) is not str:
        raise ValueError("retrieval index must contain exactly one manifest")
    return RetrievalIndexManifest.from_value(parse_canonical_json(manifest_rows[0][0]))


def _validate_index_schema(connection: sqlite3.Connection) -> None:
    if connection.execute("PRAGMA user_version").fetchone() != (1,):
        raise ValueError("retrieval SQLite schema version is unsupported")
    required = {
        ("index", "passages_passage_id_uq"),
        ("table", "passage_fts"),
        ("table", "passages"),
        ("table", "retrieval_manifest"),
    }
    schema = set(
        connection.execute(
            "SELECT type, name FROM sqlite_master WHERE name IN (?, ?, ?, ?)",
            tuple(name for _kind, name in sorted(required)),
        ).fetchall()
    )
    if schema != required:
        raise ValueError("retrieval SQLite schema is incomplete")


def verify_built_index(
    path: str | os.PathLike[str],
    *,
    expected_manifest: RetrievalIndexManifest,
) -> RetrievalIndexManifest:
    if not isinstance(expected_manifest, RetrievalIndexManifest):
        raise TypeError("expected_manifest must be RetrievalIndexManifest")
    connection = sqlite3.connect(Path(path).resolve(strict=True))
    try:
        _validate_index_schema(connection)
        manifest = _read_index_manifest(connection)
        if manifest != expected_manifest:
            raise ValueError("built retrieval manifest differs from its expected identity")
        passage_count = connection.execute("SELECT count(*) FROM passages").fetchone()
        fts_count = connection.execute("SELECT count(*) FROM passage_fts").fetchone()
        if passage_count != (manifest.passage_count,) or fts_count != (manifest.passage_count,):
            raise ValueError("built retrieval row counts differ from the manifest")
        connection.execute(
            "INSERT INTO passage_fts(passage_fts, rank) VALUES ('integrity-check', 1)"
        )
        integrity = connection.execute("PRAGMA quick_check").fetchall()
        if integrity != [("ok",)]:
            raise ValueError("built retrieval SQLite artifact failed quick check")
        return manifest
    finally:
        connection.close()


def audit_retrieval_index_content(
    path: str | os.PathLike[str],
) -> RetrievalIndexManifest:
    connection = sqlite3.connect(Path(path).resolve(strict=True))
    try:
        _validate_index_schema(connection)
        manifest = _read_index_manifest(connection)
        rows = connection.execute(
            "SELECT passage_id, document_id, title, text FROM passages ORDER BY passage_id"
        )
        passage_count, corpus_hash = _hash_ordered_passages(_passages_from_rows(rows))
        expected = RetrievalIndexManifest.create_from_corpus_hash(
            corpus_name=manifest.corpus_name,
            corpus_version=manifest.corpus_version,
            passage_count=passage_count,
            corpus_hash=corpus_hash,
        )
        if manifest != expected:
            raise ValueError("retrieval index content does not match its manifest")
        return manifest
    finally:
        connection.close()


class RetrievalIndex:
    def __init__(
        self,
        *,
        path: Path,
        connection: sqlite3.Connection,
        manifest: RetrievalIndexManifest,
    ) -> None:
        self._path = path
        self._connection = connection
        self._manifest = manifest
        self._closed = False

    @classmethod
    def open(cls, path: str | os.PathLike[str]) -> RetrievalIndex:
        resolved = Path(path).resolve(strict=True)
        uri = f"file:{quote(str(resolved), safe='/')}?mode=ro&immutable=1"
        with ExitStack() as cleanup:
            connection = sqlite3.connect(uri, uri=True)
            cleanup.callback(connection.close)
            connection.execute("PRAGMA query_only=ON")
            _validate_index_schema(connection)
            manifest = _read_index_manifest(connection)
            cleanup.pop_all()
            return cls(path=resolved, connection=connection, manifest=manifest)

    @property
    def manifest(self) -> RetrievalIndexManifest:
        return self._manifest

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def __enter__(self) -> RetrievalIndex:
        if self._closed:
            raise RuntimeError("retrieval index is closed")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("retrieval index is closed")

    def read(self, passage_id: str) -> DocumentPassage:
        self._require_open()
        identity = _text(passage_id, field="passage_id")
        row = self._connection.execute(
            "SELECT passage_id, document_id, title, text FROM passages WHERE passage_id = ?",
            (identity,),
        ).fetchone()
        if row is None:
            raise KeyError(identity)
        return DocumentPassage(
            passage_id=_text(row[0], field="passage_id"),
            document_id=_text(row[1], field="document_id"),
            title=_text(row[2], field="title"),
            text=_text(row[3], field="text"),
        )

    def search(self, query: str, *, limit: int) -> tuple[SearchHit, ...]:
        self._require_open()
        match_query = _compile_match_query(query)
        count = _positive_int(limit, field="limit")
        rows = self._connection.execute(
            "SELECT p.passage_id, p.document_id, p.title, p.text, "
            "bm25(passage_fts, 5.0, 1.0) AS score "
            "FROM passage_fts "
            "JOIN passages AS p ON p.rowid = passage_fts.rowid "
            "WHERE passage_fts MATCH ? "
            "ORDER BY score ASC, p.passage_id ASC LIMIT ?",
            (match_query, count),
        ).fetchall()
        return tuple(
            SearchHit(
                passage_id=_text(row[0], field="passage_id"),
                document_id=_text(row[1], field="document_id"),
                title=_text(row[2], field="title"),
                snippet=_snippet(_text(row[3], field="text")),
                rank=rank,
            )
            for rank, row in enumerate(rows, start=1)
        )


def _compile_match_query(query: str) -> str:
    public_query = _text(query, field="query")
    terms = tuple(dict.fromkeys(_QUERY_TERM.findall(public_query)))
    if not terms:
        raise ValueError("query must contain at least one searchable term")
    return " OR ".join(f'"{term}"' for term in terms)


def _snippet(text: str) -> str:
    if len(text) <= _SNIPPET_CODEPOINTS:
        return text
    return f"{text[:_SNIPPET_CODEPOINTS]}…"


class QABenchmark(StrEnum):
    HOTPOT_QA = "hotpotqa"
    TRIVIA_QA = "triviaqa"
    NATURAL_QUESTIONS = "nq-open"


@dataclass(frozen=True, slots=True)
class _ExecutionRecord:
    step_index: int
    action: StructuredAction
    observation: EnvironmentObservation


class QARetrievalEnvironment:
    def __init__(
        self,
        *,
        index: RetrievalIndex,
        benchmark: QABenchmark,
        dataset_revision: str,
        task_family: str,
        resource_id: str = "qa-retrieval",
    ) -> None:
        if not isinstance(index, RetrievalIndex):
            raise TypeError("index must be a RetrievalIndex")
        if not isinstance(benchmark, QABenchmark):
            raise TypeError("benchmark must be a QABenchmark")
        self._index = index
        self._benchmark = benchmark
        self._dataset_revision = _text(dataset_revision, field="dataset_revision")
        self._task_family = _text(task_family, field="task_family")
        self._resource_id = _text(resource_id, field="resource_id")
        self._records: list[_ExecutionRecord] = []

    @property
    def environment_id(self) -> str:
        return (
            f"benchmark:{self._benchmark.value}@{self._dataset_revision}:"
            f"retrieval:{self._index.manifest.index_id}"
        )

    @property
    def task_family(self) -> str:
        return self._task_family

    async def execute(
        self,
        action: StructuredAction,
        *,
        step_index: int,
    ) -> EnvironmentObservation:
        if not isinstance(action, StructuredAction):
            raise TypeError("action must be a StructuredAction")
        step = _positive_int(step_index, field="step_index")
        if step != len(self._records) + 1:
            raise ValueError("environment step index is not contiguous")

        observation = self._outcome_for_action(action)
        record = _ExecutionRecord(
            step_index=step,
            action=action,
            observation=observation,
        )
        self._records.append(record)
        return self._publish_outcome(observation)

    @staticmethod
    def _publish_outcome(observation: EnvironmentObservation) -> EnvironmentObservation:
        if observation.observation_status != "tool_error":
            return observation
        public_value = observation.public_value
        if type(public_value) is not dict or set(public_value) != {"error"}:
            raise ValueError("stored tool failure has an incompatible public projection")
        error_code = public_value["error"]
        if type(error_code) is not str or not error_code:
            raise ValueError("stored tool failure has an invalid public error code")
        raise EnvironmentMethodFailedError(
            budget_usage=observation.budget_usage,
            public_error_code=error_code,
        )

    def _outcome_for_action(self, action: StructuredAction) -> EnvironmentObservation:
        usage = BudgetVector(tool_calls=1)
        if action.kind is ActionKind.SKILL:
            if action.skill_id is None:
                raise TypeError("skill action is missing its skill identity")
            return EnvironmentObservation(
                public_value={"operation": "skill", "skill_id": action.skill_id},
                observation_status="success",
                invoked_skill_ids=(action.skill_id,),
                budget_usage=usage,
            )
        if action.kind is not ActionKind.TOOL or action.resource_id != self._resource_id:
            return _tool_failure("unsupported_retrieval_action")
        if action.name == "search":
            if type(action.arguments) is not dict or set(action.arguments) != {"limit", "query"}:
                return _tool_failure("invalid_search_arguments")
            query = action.arguments["query"]
            limit = action.arguments["limit"]
            if type(query) is not str or not query.strip() or type(limit) is not int or limit < 1:
                return _tool_failure("invalid_search_arguments")
            try:
                hits = self._index.search(query, limit=limit)
            except ValueError:
                return _tool_failure("invalid_search_arguments")
            return EnvironmentObservation(
                public_value={
                    "hits": [hit.to_value() for hit in hits],
                    "operation": "search",
                    "query": query,
                },
                observation_status="success",
                budget_usage=usage,
            )
        if action.name == "read":
            if type(action.arguments) is not dict or set(action.arguments) != {"passage_id"}:
                return _tool_failure("invalid_read_arguments")
            passage_id = action.arguments["passage_id"]
            if type(passage_id) is not str or not passage_id.strip():
                return _tool_failure("invalid_read_arguments")
            try:
                passage = self._index.read(passage_id)
            except KeyError:
                return _tool_failure("passage_not_found")
            return EnvironmentObservation(
                public_value={"operation": "read", "passage": passage.to_value()},
                observation_status="success",
                budget_usage=usage,
            )
        return _tool_failure("unsupported_retrieval_action")

    def validate_completion(self, submission: JsonValue) -> bool:
        normalized = normalize_json(submission)
        return (
            type(normalized) is dict
            and set(normalized) == {"answer"}
            and type(normalized["answer"]) is str
            and bool(normalized["answer"].strip())
        )


def _tool_failure(public_error_code: str) -> EnvironmentObservation:
    return EnvironmentObservation(
        public_value={"error": public_error_code},
        observation_status="tool_error",
        budget_usage=BudgetVector(tool_calls=1),
    )


__all__ = [
    "RETRIEVAL_BACKEND",
    "DocumentPassage",
    "QABenchmark",
    "QARetrievalEnvironment",
    "RetrievalBuildPhase",
    "RetrievalBuildProgress",
    "RetrievalIndex",
    "RetrievalIndexManifest",
    "SearchHit",
    "audit_retrieval_index_content",
    "build_retrieval_index",
    "build_retrieval_index_stream",
    "verify_built_index",
]
