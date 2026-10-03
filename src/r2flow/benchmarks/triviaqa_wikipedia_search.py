from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import time
from collections import OrderedDict
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from skillev.benchmarks.r2flow_tool_set import wikipedia_corpus_identity
from skillev.benchmarks.wikipedia_search import WikipediaPassage
from skillev.contracts import JsonValue
from skillev.contracts.wikipedia_search import (
    PASSAGES_PER_QUERY,
    WIKIPEDIA_CORPUS_ID,
    WIKIPEDIA_QUERY_POLICY,
    WIKIPEDIA_SEARCH_PROFILE,
)

DEFAULT_SEARCH_TIMEOUT_SECONDS: Final = 60.0
_DIGEST_MEMO: dict[tuple[str, int, int, int], str] = {}


def sha256_file_memoised(path: Path) -> str:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    key = (str(resolved), stat.st_ino, stat.st_mtime_ns, stat.st_size)
    if key not in _DIGEST_MEMO:
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        _DIGEST_MEMO[key] = digest.hexdigest()
    return _DIGEST_MEMO[key]


DEFAULT_MAX_CONCURRENCY: Final = 4
MEMO_ENTRIES: Final = 4096
FTS_RANK: Final = "bm25(5.0,1.0)"
FTS_SCHEMA: Final = (
    "CREATE VIRTUAL TABLE passages USING fts5(title, text, tokenize='porter unicode61')"
)


class WikipediaCorpusError(ValueError):
    pass


def _read_only(path: Path) -> closing[sqlite3.Connection]:
    return closing(
        sqlite3.connect(path.resolve(strict=True).as_uri() + "?mode=ro&immutable=1", uri=True)
    )


@dataclass(frozen=True, slots=True)
class WikipediaCorpusDeployment:
    path: Path
    sha256: str
    size_bytes: int
    passages: int
    corpus_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise WikipediaCorpusError("the Wikipedia corpus path must be absolute")
        if self.corpus_id != WIKIPEDIA_CORPUS_ID:
            raise WikipediaCorpusError(
                f"TriviaQA corpus_search uses the full DPR Wikipedia corpus {WIKIPEDIA_CORPUS_ID}"
            )
        try:
            self.identity()
        except ValueError as error:
            raise WikipediaCorpusError(str(error)) from error

    @classmethod
    def from_value(cls, value: object) -> WikipediaCorpusDeployment:
        if not isinstance(value, dict) or set(value) != {
            "path",
            "sha256",
            "size_bytes",
            "passages",
            "corpus_id",
        }:
            raise WikipediaCorpusError(
                "triviaqa_wikipedia pins path, sha256, size_bytes, passages, corpus_id"
            )
        return cls(
            Path(value["path"]),
            value["sha256"],
            value["size_bytes"],
            value["passages"],
            value["corpus_id"],
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "passages": self.passages,
            "corpus_id": self.corpus_id,
        }

    def identity(self) -> dict[str, JsonValue]:
        return wikipedia_corpus_identity(
            sha256=self.sha256,
            size_bytes=self.size_bytes,
            passages=self.passages,
            corpus_id=self.corpus_id,
        )

    def verify(self, *, full_hash: bool = False) -> dict[str, str]:
        if not self.path.is_file():
            raise WikipediaCorpusError(f"the pinned Wikipedia corpus is missing: {self.path}")
        size = self.path.stat().st_size
        if size != self.size_bytes:
            raise WikipediaCorpusError(
                f"Wikipedia corpus size {size} differs from the pin {self.size_bytes}"
            )
        with _read_only(self.path) as db:
            metadata = dict(db.execute("SELECT key,value FROM metadata"))
            schema = db.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='passages'"
            ).fetchone()
            rank = db.execute("SELECT v FROM passages_config WHERE k='rank'").fetchone()
        expected = {
            "profile": WIKIPEDIA_SEARCH_PROFILE,
            "corpus_id": self.corpus_id,
            "passages": str(self.passages),
            "completed": "true",
        }
        wrong = sorted(key for key, value in expected.items() if metadata.get(key) != value)
        if wrong:
            raise WikipediaCorpusError(f"Wikipedia corpus metadata differs from the pin: {wrong}")
        if schema is None or schema[0] != FTS_SCHEMA:
            raise WikipediaCorpusError("the Wikipedia corpus is not the declared FTS5 table")
        if rank is None or rank[0] != FTS_RANK:
            raise WikipediaCorpusError(f"the Wikipedia corpus rank is not {FTS_RANK}")
        if full_hash:
            digest = sha256_file_memoised(self.path)
            if digest != self.sha256:
                raise WikipediaCorpusError(f"Wikipedia corpus sha256 {digest} differs from the pin")
        return metadata


def ranked_passages(
    path: Path,
    query: str,
    *,
    limit: int = PASSAGES_PER_QUERY,
    timeout: float = DEFAULT_SEARCH_TIMEOUT_SECONDS,
) -> tuple[WikipediaPassage, ...]:
    from r2flow.evaluation.wikipedia_corpus import search_corpus

    started = time.monotonic()
    try:
        rows = search_corpus(
            path, query, limit=limit, timeout=timeout, query_policy=WIKIPEDIA_QUERY_POLICY
        )
    except sqlite3.OperationalError as error:
        if "interrupt" in str(error).lower() or time.monotonic() - started >= timeout:
            raise TimeoutError("Wikipedia search exceeded its deadline") from error
        raise
    return tuple(
        WikipediaPassage(str(row["passage_id"]), str(row["title"]), str(row["text"]))
        for row in rows
    )


@dataclass(slots=True)
class WikipediaSearchBackend:
    deployment: WikipediaCorpusDeployment
    timeout: float = DEFAULT_SEARCH_TIMEOUT_SECONDS
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    memo_entries: int = MEMO_ENTRIES
    _memo: OrderedDict[str, tuple[WikipediaPassage, ...]] = field(default_factory=OrderedDict)
    _limits: dict[int, asyncio.Semaphore] = field(default_factory=dict)
    searches: int = 0
    timeouts: int = 0

    def __post_init__(self) -> None:
        if (
            type(self.max_concurrency) is not int
            or self.max_concurrency < 1
            or self.timeout <= 0
            or type(self.memo_entries) is not int
            or self.memo_entries < 0
        ):
            raise WikipediaCorpusError(
                "the Wikipedia backend needs a positive deadline and concurrency"
            )

    def identity(self) -> dict[str, JsonValue]:
        return self.deployment.identity()

    def stats(self) -> dict[str, JsonValue]:
        return {"searches": self.searches, "timeouts": self.timeouts, "memoised": len(self._memo)}

    async def search(self, query: str) -> tuple[WikipediaPassage, ...]:
        if query in self._memo:
            self._memo.move_to_end(query)
            return self._memo[query]
        loop = id(asyncio.get_running_loop())
        limit = self._limits.setdefault(loop, asyncio.Semaphore(self.max_concurrency))
        async with limit:
            self.searches += 1
            try:
                result = await asyncio.to_thread(
                    ranked_passages, self.deployment.path, query, timeout=self.timeout
                )
            except TimeoutError:
                self.timeouts += 1
                raise
        if self.memo_entries:
            self._memo[query] = result
            while len(self._memo) > self.memo_entries:
                self._memo.popitem(last=False)
        return result


__all__ = [
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_SEARCH_TIMEOUT_SECONDS",
    "FTS_RANK",
    "FTS_SCHEMA",
    "MEMO_ENTRIES",
    "WikipediaCorpusDeployment",
    "WikipediaCorpusError",
    "WikipediaSearchBackend",
    "ranked_passages",
]
