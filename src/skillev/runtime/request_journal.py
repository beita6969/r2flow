from __future__ import annotations

import json
import sqlite3
import threading
import weakref
import zlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from skillev.contracts import JsonValue, canonical_json

SQLITE_BUSY_TIMEOUT_SECONDS = 900


class SharedDatabase:
    def __init__(self, path: Path, *, isolation_level: Literal["DEFERRED"] | None) -> None:
        self.lock = threading.RLock()
        uri = "file:" + quote(str(path.resolve()), safe="/") + "?vfs=unix-excl"
        self.connection = sqlite3.connect(
            uri,
            uri=True,
            timeout=SQLITE_BUSY_TIMEOUT_SECONDS,
            isolation_level=isolation_level,
            check_same_thread=False,
        )
        self.connection.execute("PRAGMA locking_mode=EXCLUSIVE")
        weakref.finalize(self, self.connection.close)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.lock:
            with self.connection as db:
                yield db


_DATABASES: weakref.WeakValueDictionary[str, SharedDatabase] = weakref.WeakValueDictionary()
_DATABASES_GUARD = threading.Lock()


def shared_database(
    path: Path, *, isolation_level: Literal["DEFERRED"] | None = "DEFERRED"
) -> SharedDatabase:
    key = str(path.resolve())
    with _DATABASES_GUARD:
        database = _DATABASES.get(key)
        if database is None:
            database = SharedDatabase(path, isolation_level=isolation_level)
            _DATABASES[key] = database
        return database


@contextmanager
def read_connection(path: Path) -> Iterator[sqlite3.Connection]:
    with _DATABASES_GUARD:
        shared = _DATABASES.get(str(path.resolve()))
    if shared is None:
        db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            yield db
        finally:
            db.close()
        return
    with shared.lock:
        try:
            yield shared.connection
        finally:
            shared.connection.rollback()


class UnknownRequestOutcomeError(RuntimeError):
    pass


class DurableRequestJournal:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.touch(mode=0o600, exist_ok=True)
        path.chmod(0o600)
        for suffix in ("-wal", "-shm"):
            sidefile = path.with_name(path.name + suffix)
            if sidefile.exists():
                sidefile.chmod(0o600)
        self._shared = shared_database(path)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("""CREATE TABLE IF NOT EXISTS requests (
                identity TEXT PRIMARY KEY, endpoint TEXT NOT NULL,
                payload BLOB NOT NULL, state TEXT NOT NULL,
                status INTEGER, response BLOB
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS episode_routes (
                episode TEXT PRIMARY KEY, policy TEXT NOT NULL, endpoint TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS authorized_retries (
                identity TEXT PRIMARY KEY, authorization_id TEXT NOT NULL,
                reason TEXT NOT NULL, prior_endpoint TEXT NOT NULL,
                prior_payload BLOB NOT NULL, prior_state TEXT NOT NULL,
                prior_status INTEGER, prior_response BLOB, retry_state TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS authorized_route_migrations (
                identity TEXT NOT NULL, episode TEXT NOT NULL, policy TEXT NOT NULL,
                prior_endpoint TEXT NOT NULL, target_endpoint TEXT NOT NULL,
                authorization_id TEXT NOT NULL, reason TEXT NOT NULL,
                PRIMARY KEY(identity, authorization_id)
            )""")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with self._shared.transaction() as db:
            yield db

    def episode_route(self, episode: str, policy: str) -> str | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT policy,endpoint FROM episode_routes WHERE episode=?", (episode,)
            ).fetchone()
        if row is None:
            return None
        if row[0] != policy:
            raise ValueError("saved episode route belongs to another policy")
        return str(row[1])

    def save_episode_route(self, episode: str, policy: str, endpoint: str) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT policy,endpoint FROM episode_routes WHERE episode=?", (episode,)
            ).fetchone()
            if row is not None and row != (policy, endpoint):
                raise ValueError("cannot replace an episode's original serving route")
            db.execute(
                "INSERT OR IGNORE INTO episode_routes VALUES(?,?,?)", (episode, policy, endpoint)
            )

    def require_resolved_prefix(self, prefix: tuple[str, str]) -> None:
        if len(prefix) != 2 or any(not isinstance(v, str) or not v for v in prefix):
            raise ValueError("request scope needs non-empty coordinates")
        with self._connect() as db:
            found = db.execute(
                "SELECT 1 FROM requests WHERE state!='COMPLETED' "
                "AND json_extract(identity,'$[0]')=? "
                "AND json_extract(identity,'$[1]')=? LIMIT 1",
                prefix,
            ).fetchone()
        if found is not None:
            raise UnknownRequestOutcomeError("logical operation has an unresolved prior dispatch")

    def migrate_episode_route(
        self, *, episode: str, policy: str, endpoint: str, authorization_id: str, reason: str
    ) -> None:
        if not all(v.strip() for v in (episode, policy, endpoint, authorization_id, reason)):
            raise ValueError("route migration needs identity and explicit authorization")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            route = db.execute(
                "SELECT policy,endpoint FROM episode_routes WHERE episode=?", (episode,)
            ).fetchone()
            if route is None or route[0] != policy:
                raise ValueError("migration requires the original episode policy")
            if route[1] == endpoint:
                return
            rows = db.execute(
                "SELECT identity,endpoint FROM requests WHERE json_extract(identity,'$[0]')=?",
                (episode,),
            ).fetchall()
            for key, old_endpoint in rows:
                coordinates = json.loads(key)
                if len(coordinates) != 6 or coordinates[3] != policy:
                    raise ValueError("only matching rollout requests can migrate")
                db.execute(
                    "INSERT INTO authorized_route_migrations VALUES(?,?,?,?,?,?,?)",
                    (
                        key,
                        episode,
                        policy,
                        old_endpoint,
                        endpoint + "/generate",
                        authorization_id,
                        reason,
                    ),
                )
            db.execute("UPDATE episode_routes SET endpoint=? WHERE episode=?", (endpoint, episode))

    def authorize_aborted_retry(
        self, *, identity: tuple[str, ...], authorization_id: str, reason: str
    ) -> None:
        if not identity or any(not isinstance(v, str) or not v for v in identity):
            raise ValueError("retry requires the exact original request identity")
        if not authorization_id.strip() or not reason.strip():
            raise ValueError("retry requires explicit authorization and abort evidence")
        key = canonical_json(list(identity))
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                "SELECT authorization_id,reason FROM authorized_retries WHERE identity=?", (key,)
            ).fetchone()
            if prior is not None:
                if prior != (authorization_id, reason):
                    raise ValueError("cannot replace or renew an existing retry authorization")
                return
            row = db.execute(
                "SELECT endpoint,payload,state,status,response FROM requests WHERE identity=?",
                (key,),
            ).fetchone()
            if row is None or row[2] != "DISPATCHED":
                raise ValueError("only an unresolved dispatched operation can be authorized")
            db.execute(
                "INSERT INTO authorized_retries VALUES(?,?,?,?,?,?,?,?,?)",
                (key, authorization_id, reason, *row, "READY"),
            )

    def authorize_preexecution_input_length_retry(
        self, *, identity: tuple[str, ...], authorization_id: str, reason: str
    ) -> None:
        if not identity or any(not isinstance(v, str) or not v for v in identity):
            raise ValueError("retry requires the exact original request identity")
        if not authorization_id.strip() or not reason.strip():
            raise ValueError("retry requires explicit authorization and rejection evidence")
        key = canonical_json(list(identity))
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                "SELECT authorization_id,reason FROM authorized_retries WHERE identity=?", (key,)
            ).fetchone()
            if prior is not None:
                if prior != (authorization_id, reason):
                    raise ValueError("cannot replace or renew an existing retry authorization")
                return
            row = db.execute(
                "SELECT endpoint,payload,state,status,response FROM requests WHERE identity=?",
                (key,),
            ).fetchone()
            if row is None or row[2] != "COMPLETED" or row[3] != 400 or row[4] is None:
                raise ValueError("retry requires a completed HTTP 400 response")
            response = json.loads(zlib.decompress(row[4]))
            message = (
                response.get("error", {}).get("message")
                if isinstance(response, dict) and isinstance(response.get("error"), dict)
                else None
            )
            if not isinstance(message, str) or not (
                message.startswith("Input length (")
                and " exceeds the maximum allowed length (" in message
            ):
                raise ValueError("HTTP 400 was not a pre-generation input-length rejection")
            db.execute(
                "INSERT INTO authorized_retries VALUES(?,?,?,?,?,?,?,?,?)",
                (key, authorization_id, reason, *row, "READY"),
            )
            db.execute(
                "UPDATE requests SET state='DISPATCHED',status=NULL,response=NULL WHERE identity=?",
                (key,),
            )

    def request(
        self,
        *,
        identity: tuple[str, ...],
        endpoint: str,
        payload: dict[str, JsonValue],
        send: Callable[[], tuple[int, JsonValue]],
    ) -> tuple[int, JsonValue]:
        if not identity or any(not isinstance(v, str) or not v for v in identity):
            raise ValueError("persistent requests need non-empty execution coordinates")
        key = canonical_json(list(identity))
        encoded = canonical_json(payload).encode()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT endpoint,payload,state,status,response FROM requests WHERE identity=?",
                (key,),
            ).fetchone()
            if row is not None:
                old_endpoint, old_payload, state, status, response = row
                migrated = (
                    old_endpoint != endpoint
                    and db.execute(
                        "SELECT 1 FROM authorized_route_migrations m JOIN episode_routes e "
                        "ON e.episode=m.episode AND e.policy=m.policy "
                        "WHERE m.identity=? AND m.prior_endpoint=? AND m.target_endpoint=? "
                        "AND e.endpoint || '/generate'=m.target_endpoint LIMIT 1",
                        (key, old_endpoint, endpoint),
                    ).fetchone()
                    is not None
                )
                if (old_endpoint != endpoint and not migrated) or zlib.decompress(
                    old_payload
                ) != encoded:
                    raise ValueError("saved request differs from its exact input/route")
                if state == "COMPLETED":
                    restored = json.loads(zlib.decompress(response))
                    if isinstance(restored, dict):
                        restored["skillev_restored_response"] = True
                    return status, restored
                permit = db.execute(
                    "UPDATE authorized_retries SET retry_state='DISPATCHED' "
                    "WHERE identity=? AND retry_state='READY'",
                    (key,),
                )
                if permit.rowcount != 1:
                    raise UnknownRequestOutcomeError("previous dispatch has no durable response")
                if migrated:
                    db.execute("UPDATE requests SET endpoint=? WHERE identity=?", (endpoint, key))
            else:
                db.execute(
                    "INSERT INTO requests(identity,endpoint,payload,state) VALUES(?,?,?,?)",
                    (key, endpoint, zlib.compress(encoded), "DISPATCHED"),
                )
        status, response = send()
        saved = zlib.compress(canonical_json(response).encode())
        with self._connect() as db:
            db.execute(
                "UPDATE requests SET state='COMPLETED',status=?,response=? WHERE identity=?",
                (status, saved, key),
            )
            db.execute(
                "UPDATE authorized_retries SET retry_state='COMPLETED' "
                "WHERE identity=? AND retry_state='DISPATCHED'",
                (key,),
            )
        return status, response
