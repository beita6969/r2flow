from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from skillev.evaluation.corpus_search import (
    CORPUS_ID,
    GLASGOW_QUERY_POLICY,
    PHRASE_QUERY_POLICY,
    QUERY_POLICY,
    REQUIRED_PHRASE_QUERY_POLICY,
    SEARCH_PROFILE,
    SNAPSHOT_DATE,
    SNAPSHOT_REFERENCE,
    SOFT_CONTEXT_QUERY_POLICY,
)

from .english_stopwords import SNOWBALL_ENGLISH
from .glasgow_stopwords import GLASGOW_ENGLISH

CORPUS_URL = "DPR psgs_w100"
PROGRESS_HANDLER_OPCODES = 1_000_000


def build_index(source: Path, destination: Path, *, corpus_id: str = CORPUS_ID) -> int:
    if destination.exists():
        raise FileExistsError(destination)
    partial = destination.with_suffix(destination.suffix + ".building")
    with partial.open("xb"):
        pass
    count, started = 0, time.monotonic()
    connection = sqlite3.connect(partial)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA cache_size=-131072")
        connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute(
            "CREATE VIRTUAL TABLE passages USING fts5(title, text, tokenize='porter unicode61')"
        )
        with gzip.open(source, "rt", encoding="utf-8", newline="") as stream:
            rows = csv.DictReader(stream, delimiter="\t")
            if rows.fieldnames != ["id", "text", "title"]:
                raise ValueError("expected the public DPR id/text/title passage TSV")
            batch = []
            for row in rows:
                if None in row or any(row[k] is None for k in ("id", "text", "title")):
                    raise ValueError("incomplete public passage row")
                batch.append((int(row["id"]), row["title"], row["text"]))
                if len(batch) == 10000:
                    connection.executemany(
                        "INSERT INTO passages(rowid,title,text) VALUES(?,?,?)", batch
                    )
                    count += len(batch)
                    batch.clear()
                    connection.commit()
                    if count % 100000 == 0:
                        print(
                            json.dumps(
                                {
                                    "passages": count,
                                    "elapsed_s": round(time.monotonic() - started, 1),
                                }
                            ),
                            flush=True,
                        )
            connection.executemany("INSERT INTO passages(rowid,title,text) VALUES(?,?,?)", batch)
            count += len(batch)
        if not count:
            raise ValueError("empty corpus")
        connection.execute("INSERT INTO passages(passages,rank) VALUES('rank','bm25(5.0,1.0)')")
        connection.executemany(
            "INSERT INTO metadata VALUES(?,?)",
            [
                ("profile", SEARCH_PROFILE),
                ("corpus_id", corpus_id),
                ("passages", str(count)),
                ("source_url", CORPUS_URL),
                ("completed", "true"),
            ],
        )
        if corpus_id == CORPUS_ID:
            connection.executemany(
                "INSERT INTO metadata VALUES(?,?)",
                [("snapshot_date", SNAPSHOT_DATE), ("snapshot_reference", SNAPSHOT_REFERENCE)],
            )
        connection.commit()
    finally:
        connection.close()
    partial.chmod(0o400)
    partial.rename(destination)
    return count


def corpus_metadata(path: Path) -> dict[str, str]:
    with closing(
        sqlite3.connect(path.resolve(strict=True).as_uri() + "?mode=ro&immutable=1", uri=True)
    ) as db:
        return dict(db.execute("SELECT key,value FROM metadata"))


def compile_query(query: str, query_policy: str = QUERY_POLICY) -> str:
    required: list[str] = []
    if query_policy == QUERY_POLICY:
        terms = re.findall(r"\w+", query.casefold())
        terms = [word for word in terms if word not in SNOWBALL_ENGLISH]
    elif query_policy in {
        PHRASE_QUERY_POLICY,
        GLASGOW_QUERY_POLICY,
        REQUIRED_PHRASE_QUERY_POLICY,
        SOFT_CONTEXT_QUERY_POLICY,
    }:
        terms = []
        stopwords = (
            SNOWBALL_ENGLISH | GLASGOW_ENGLISH
            if query_policy
            in {GLASGOW_QUERY_POLICY, REQUIRED_PHRASE_QUERY_POLICY, SOFT_CONTEXT_QUERY_POLICY}
            else SNOWBALL_ENGLISH
        )
        for phrase, word in re.findall(r'"([^"]*)"|(\w+)', query.casefold()):
            if phrase:
                term = " ".join(re.findall(r"\w+", phrase))
                if term:
                    if query_policy in {REQUIRED_PHRASE_QUERY_POLICY, SOFT_CONTEXT_QUERY_POLICY}:
                        required.append(term)
                    else:
                        terms.append(term)
            elif word and word not in stopwords:
                terms.append(word)
    else:
        raise ValueError("unsupported corpus query policy")
    required = list(dict.fromkeys(required))[:32]
    optional = [term for term in dict.fromkeys(terms) if term not in required][: 32 - len(required)]
    alternative = " OR ".join(f'"{term}"' for term in optional)
    if not required:
        return alternative
    clauses = [f'"{term}"' for term in required]
    if alternative:
        if query_policy == SOFT_CONTEXT_QUERY_POLICY and any(
            " " in term or term not in SNOWBALL_ENGLISH | GLASGOW_ENGLISH for term in required
        ):
            clauses.append(f"({' AND '.join(clauses)} OR {alternative})")
        else:
            clauses.append(f"({alternative})")
    return " AND ".join(clauses)


def search_corpus(
    path: Path,
    query: str,
    *,
    limit: int,
    timeout: float = 30.0,
    query_policy: str = QUERY_POLICY,
) -> list[dict[str, object]]:
    match = compile_query(query, query_policy)
    if not match:
        return []
    deadline = time.monotonic() + timeout
    with closing(
        sqlite3.connect(path.resolve(strict=True).as_uri() + "?mode=ro&immutable=1", uri=True)
    ) as db:
        db.set_progress_handler(lambda: int(time.monotonic() > deadline), PROGRESS_HANDLER_OPCODES)
        rows = db.execute(
            "SELECT rowid,title,text FROM passages WHERE passages MATCH ? ORDER BY rank LIMIT ?",
            (match, limit),
        ).fetchall()
    return [
        {"passage_id": str(pid), "title": title, "text": text, "rank": rank}
        for rank, (pid, title, text) in enumerate(rows, 1)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Question-independent DPR passage import and read-only lexical retrieval."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(
        json.dumps({"completed_passages": build_index(args.source, args.destination)}), flush=True
    )


if __name__ == "__main__":
    main()
