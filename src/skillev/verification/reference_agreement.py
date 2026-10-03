from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import threading
import time
import weakref
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol

from skillev.contracts import JsonValue

from .qa_grounding import squad_normalize

if TYPE_CHECKING:
    from skillev.r2flow_evolution.author import AuthorClient

REFERENCE_PROMPT_VERSION: Final = "reference-prompt@2"
REFERENCE_CACHE_FORMAT: Final = "reference-cache@1"
REFERENCE_CACHE_FILE: Final = "verifier-reference-cache.sqlite3"
REFERENCE_NO_FABRICATION: Final = "reference-no-fabrication@1"
REFERENCE_EFFORT_RULE: Final = "reference-reasoning-effort@1"
TRIVIAQA_CONTAINMENT: Final = "qa-normalized-containment@2"
AIME_FINAL_NUMBER: Final = "final-number@1"
REFERENCE_DOMAINS: Final = ("aime-2026", "triviaqa")
REFERENCE_PROMPTS: Final[Mapping[str, str]] = {
    "triviaqa": (
        "Answer the question. Reply with only the shortest answer (a name, number, date or "
        "short phrase) without explanation or qualifiers, or with N/A if the text is not a "
        "question with a short factual answer.\n\nQuestion:\n{question}"
    ),
    "aime-2026": (
        "Solve the following. If it asks for a single numerical result, end your reply with a "
        "final line of the form 'ANSWER: <number>'. If it does not ask for a single numerical "
        "result, reply 'ANSWER: N/A'.\n\n{question}"
    ),
}
REFERENCE_REASONING_EFFORT: Final[Mapping[str, str]] = {"triviaqa": "low", "aime-2026": "medium"}
REFERENCE_MAX_OUTPUT_TOKENS: Final[Mapping[str, int]] = {"triviaqa": 4096, "aime-2026": 32768}
COMPARISON_RULES: Final[Mapping[str, str]] = {
    "triviaqa": TRIVIAQA_CONTAINMENT,
    "aime-2026": AIME_FINAL_NUMBER,
}
NUMBER_TOLERANCE: Final = 1e-6
REPLY_DETAIL_CHARS: Final = 2000

ReferenceStatus = Literal["answer", "not-applicable", "unavailable"]
_STATUSES: Final = frozenset({"answer", "not-applicable", "unavailable"})
_CLIENT_ERRORS: Final = (RuntimeError, OSError, ValueError)
_CACHED_STATUSES: Final = frozenset({"answer", "not-applicable"})


@dataclass(frozen=True, slots=True)
class ReferenceAnswer:
    status: ReferenceStatus
    value: str | None
    detail: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise ValueError(f"unsupported reference status {self.status!r}")
        if (self.status == "answer") != (type(self.value) is str and bool(self.value.strip())):
            raise ValueError("exactly an 'answer' reference carries a non-empty value")
        if not isinstance(self.detail, dict):
            raise ValueError("reference detail must be a mapping")


class ReferenceAnswerBackend(Protocol):
    async def reference(self, domain: str, question: str) -> ReferenceAnswer: ...


def reference_prompt(domain: str, question: str) -> str:
    template = REFERENCE_PROMPTS.get(domain)
    if template is None:
        raise ValueError(f"{domain!r} has no reference prompt")
    return template.replace("{question}", question)


_ANSWER_LINE = re.compile(r"^[\s*_`>#-]*answer[\s*_`]*[:\N{FULLWIDTH COLON}](.*)$", re.IGNORECASE)
_NOT_APPLICABLE = re.compile(r"^n\s*/\s*a$", re.IGNORECASE)


def _is_not_applicable(text: str) -> bool:
    return bool(_NOT_APPLICABLE.match(text.strip().strip(".*_`'\"").strip()))


def parse_reference_reply(domain: str, text: str) -> tuple[ReferenceStatus, str | None]:
    if domain == "triviaqa":
        answer = text.strip()
        label = _ANSWER_LINE.match(answer)
        if label is not None and "\n" not in answer:
            answer = label.group(1).strip()
        if not answer or _is_not_applicable(answer):
            return "not-applicable", None
        return "answer", answer
    if domain == "aime-2026":
        claim = None
        for line in text.split("\n"):
            match = _ANSWER_LINE.match(line.strip())
            if match is not None:
                claim = match.group(1)
        if claim is None or _is_not_applicable(claim):
            return "not-applicable", None
        number = parse_number(_strip_decoration(claim))
        if number is None:
            return "not-applicable", None
        return "answer", _canonical_number(number)
    raise ValueError(f"{domain!r} has no reference parser")


_INT = r"(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)"
_UNSIGNED = rf"(?:{_INT}(?:\.\d+)?|\.\d+)"
_SIGN = r"[-+\N{MINUS SIGN}]"
_NUMBER_TOKEN = re.compile(
    rf"(?:(?<![\w)\]}}])(?P<sign>{_SIGN}))?(?P<num>{_UNSIGNED})"
    rf"(?:\s*/\s*(?P<den>{_UNSIGNED}))?"
)
_FRAC = re.compile(r"\\[dt]?frac\s*\{\s*([^{}]*)\}\s*\{\s*([^{}]*)\}")
_LATEX_SPACING = re.compile(r"\\[,!;: ]|\\q?quad|~")
_DECORATION: Final = " \t\r\n$*_`"


def _strip_decoration(text: str) -> str:
    value = text.strip()
    boxed = last_boxed(value)
    if boxed is not None:
        value = boxed
    value = _latex_to_plain(value)
    return value.strip(_DECORATION).rstrip(".").strip(_DECORATION)


def _latex_to_plain(text: str) -> str:
    value = text.replace("{,}", ",")
    value = _LATEX_SPACING.sub("", value)
    previous = None
    while previous != value:
        previous = value
        value = _FRAC.sub(lambda m: f"{m.group(1).strip()}/{m.group(2).strip()}", value)
    return value.replace("\\left", "").replace("\\right", "")


def _token_value(match: re.Match[str]) -> Fraction | None:
    try:
        number = Fraction(match.group("num").replace(",", ""))
        if match.group("den") is not None:
            denominator = Fraction(match.group("den").replace(",", ""))
            if denominator == 0:
                return None
            number /= denominator
    except (ValueError, ZeroDivisionError):
        return None
    return -number if match.group("sign") in ("-", "\N{MINUS SIGN}") else number


def parse_number(text: str) -> Fraction | None:
    match = _NUMBER_TOKEN.fullmatch(text.strip())
    return None if match is None else _token_value(match)


def last_number(text: str) -> Fraction | None:
    last = None
    for match in _NUMBER_TOKEN.finditer(text):
        value = _token_value(match)
        if value is not None:
            last = value
    return last


def last_boxed(text: str) -> str | None:
    content = None
    start = text.find("\\boxed")
    while start != -1:
        cursor = start + len("\\boxed")
        while cursor < len(text) and text[cursor] == " ":
            cursor += 1
        if cursor < len(text) and text[cursor] == "{":
            depth, end = 0, None
            for index in range(cursor, len(text)):
                if text[index] == "{":
                    depth += 1
                elif text[index] == "}":
                    depth -= 1
                    if depth == 0:
                        end = index
                        break
            if end is not None:
                content = text[cursor + 1 : end]
        start = text.find("\\boxed", cursor)
    return content


def final_number(output: str) -> Fraction | None:
    boxed = last_boxed(output)
    if boxed is not None:
        plain = _latex_to_plain(boxed).strip(_DECORATION)
        whole = parse_number(plain)
        return whole if whole is not None else last_number(plain)
    return last_number(_latex_to_plain(output))


def _canonical_number(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def compare_final_number(output: str, reference: str) -> tuple[bool | None, str]:
    expected = parse_number(reference)
    if expected is None:
        return None, "reference-not-a-number"
    actual = final_number(output)
    if actual is None:
        return False, "no-final-number"
    tolerance = NUMBER_TOLERANCE * max(1.0, abs(float(expected)))
    if abs(float(actual) - float(expected)) <= tolerance:
        return True, "final-number-agrees"
    return False, "final-number-differs"


def compare_containment(output: str, reference: str) -> tuple[bool | None, str]:
    expected = squad_normalize(reference)
    if not expected:
        return None, "reference-empty"
    haystack = f" {squad_normalize(output)} "
    if f" {expected} " in haystack:
        return True, "reference-contained"
    head = squad_normalize(reference.split(",", 1)[0]) if "," in reference else ""
    if head and f" {head} " in haystack:
        return True, "reference-head-contained"
    return False, "reference-not-contained"


def compare_with_reference(domain: str, output: str, reference: str) -> tuple[bool | None, str]:
    if domain == "triviaqa":
        return compare_containment(output, reference)
    if domain == "aime-2026":
        return compare_final_number(output, reference)
    raise ValueError(f"{domain!r} has no reference comparison rule")


def reference_identity(prompt_version: str = REFERENCE_PROMPT_VERSION) -> dict[str, JsonValue]:
    return {
        "cache": REFERENCE_CACHE_FORMAT,
        "comparison_rules": dict(sorted(COMPARISON_RULES.items())),
        "domains": list[JsonValue](REFERENCE_DOMAINS),
        "effort_rule": REFERENCE_EFFORT_RULE,
        "max_output_tokens": dict(sorted(REFERENCE_MAX_OUTPUT_TOKENS.items())),
        "no_fabrication": REFERENCE_NO_FABRICATION,
        "prompt_sha256": {
            domain: hashlib.sha256(REFERENCE_PROMPTS[domain].encode("utf-8")).hexdigest()
            for domain in sorted(REFERENCE_PROMPTS)
        },
        "prompt_version": prompt_version,
        "reasoning_effort": dict(sorted(REFERENCE_REASONING_EFFORT.items())),
    }


def reference_cache_key(model: str, prompt_version: str, domain: str, question: str) -> str:
    payload = json.dumps(
        [model, prompt_version, domain, question], ensure_ascii=False, separators=(",", ":")
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS reference_answers (
    key TEXT PRIMARY KEY,
    format TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    domain TEXT NOT NULL,
    question TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('answer', 'not-applicable')),
    value TEXT,
    detail TEXT NOT NULL,
    created_at REAL NOT NULL
)
"""


class ReferenceCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._connection: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            connection = sqlite3.connect(self.path, timeout=60.0, check_same_thread=False)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(_SCHEMA)
            connection.commit()
            self._connection = connection
        return self._connection

    def get(self, key: str) -> ReferenceAnswer | None:
        with self._lock:
            row = (
                self._connect()
                .execute(
                    "SELECT status, value, detail FROM reference_answers WHERE key = ?", (key,)
                )
                .fetchone()
            )
        if row is None:
            return None
        detail = json.loads(row[2])
        return ReferenceAnswer(row[0], row[1], detail if isinstance(detail, dict) else {})

    def put(
        self,
        key: str,
        *,
        model: str,
        prompt_version: str,
        domain: str,
        question: str,
        answer: ReferenceAnswer,
    ) -> ReferenceAnswer:
        if answer.status not in _CACHED_STATUSES:
            raise ValueError("only answer / not-applicable references are cached")
        with self._lock:
            connection = self._connect()
            connection.execute(
                "INSERT OR IGNORE INTO reference_answers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    REFERENCE_CACHE_FORMAT,
                    model,
                    prompt_version,
                    domain,
                    question,
                    answer.status,
                    answer.value,
                    json.dumps(answer.detail, ensure_ascii=False, sort_keys=True),
                    time.time(),
                ),
            )
            connection.commit()
        stored = self.get(key)
        assert stored is not None
        return stored

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None


class GatewayReferenceBackend:
    def __init__(
        self,
        client: AuthorClient | Mapping[str, AuthorClient],
        cache_path: str | Path,
        *,
        max_concurrency: int = 4,
        prompt_version: str = REFERENCE_PROMPT_VERSION,
    ) -> None:
        clients = (
            dict(client)
            if isinstance(client, Mapping)
            else dict.fromkeys(REFERENCE_DOMAINS, client)
        )
        if not clients or any(domain not in REFERENCE_DOMAINS for domain in clients):
            raise ValueError(f"reference clients must serve a subset of {REFERENCE_DOMAINS}")
        if type(max_concurrency) is not int or max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")
        if prompt_version != REFERENCE_PROMPT_VERSION:
            raise ValueError(f"only {REFERENCE_PROMPT_VERSION} is declared")
        self._clients = clients
        self.prompt_version = prompt_version
        self.max_concurrency = max_concurrency
        self.cache = ReferenceCache(cache_path)
        self._semaphores: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, asyncio.Semaphore
        ] = weakref.WeakKeyDictionary()
        self._inflight: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, dict[str, asyncio.Future[ReferenceAnswer]]
        ] = weakref.WeakKeyDictionary()

    def __repr__(self) -> str:
        models = {domain: client.model for domain, client in sorted(self._clients.items())}
        return f"GatewayReferenceBackend(models={models!r}, cache={str(self.cache.path)!r})"

    @property
    def domains(self) -> frozenset[str]:
        return frozenset(self._clients)

    def model(self, domain: str) -> str:
        return self._clients[domain].model

    async def reference(self, domain: str, question: str) -> ReferenceAnswer:
        if domain not in self._clients:
            raise ValueError(f"the reference backend does not serve {domain!r}")
        if type(question) is not str or not question.strip():
            return ReferenceAnswer("not-applicable", None, {"reason": "empty-question"})
        client = self._clients[domain]
        key = reference_cache_key(client.model, self.prompt_version, domain, question)
        cached = self.cache.get(key)
        if cached is not None:
            return _with_cache_flag(cached, "hit")
        loop = asyncio.get_running_loop()
        inflight = self._inflight.setdefault(loop, {})
        pending = inflight.get(key)
        if pending is None:
            pending = loop.create_task(self._resolve(domain, question, key))
            inflight[key] = pending
            pending.add_done_callback(lambda _: inflight.pop(key, None))
        return await asyncio.shield(pending)

    async def _resolve(self, domain: str, question: str, key: str) -> ReferenceAnswer:
        client = self._clients[domain]
        semaphore = self._semaphores.get(asyncio.get_running_loop())
        if semaphore is None:
            semaphore = asyncio.Semaphore(self.max_concurrency)
            self._semaphores[asyncio.get_running_loop()] = semaphore
        async with semaphore:
            cached = self.cache.get(key)
            if cached is not None:
                return _with_cache_flag(cached, "hit")
            started = time.monotonic()
            try:
                response = await asyncio.to_thread(
                    client.complete,
                    reference_prompt(domain, question),
                    max_output_tokens=REFERENCE_MAX_OUTPUT_TOKENS[domain],
                )
            except _CLIENT_ERRORS as error:
                return ReferenceAnswer(
                    "unavailable",
                    None,
                    {
                        "error": type(error).__name__,
                        "http_status": _status_of(error),
                        "model": client.model,
                        "prompt_version": self.prompt_version,
                    },
                )
        elapsed_ms = int((time.monotonic() - started) * 1000)
        detail: dict[str, JsonValue] = {
            "elapsed_ms": elapsed_ms,
            "model": response.model,
            "prompt_version": self.prompt_version,
            "reasoning_effort": REFERENCE_REASONING_EFFORT[domain],
            "reasoning_sent": response.reasoning_sent,
            "response_id": response.response_id,
            "response_status": response.status,
            "usage": response.usage,
        }
        if response.status not in (None, "completed"):
            return ReferenceAnswer("unavailable", None, {**detail, "error": "incomplete-response"})
        status, value = parse_reference_reply(domain, response.text)
        detail["reply"] = response.text[:REPLY_DETAIL_CHARS]
        stored = self.cache.put(
            key,
            model=client.model,
            prompt_version=self.prompt_version,
            domain=domain,
            question=question,
            answer=ReferenceAnswer(status, value, detail),
        )
        return _with_cache_flag(stored, "miss")


def _status_of(error: BaseException) -> int | None:
    status = getattr(error, "status", None)
    return status if type(status) is int else None


def _with_cache_flag(answer: ReferenceAnswer, flag: str) -> ReferenceAnswer:
    return ReferenceAnswer(answer.status, answer.value, {**answer.detail, "cache": flag})


_BACKENDS: dict[tuple[Any, ...], GatewayReferenceBackend] = {}
_BACKENDS_LOCK = threading.Lock()


def gateway_reference_backend(
    *,
    model: str,
    base_file: str | Path,
    key_file: str | Path,
    cache_path: str | Path,
    domains: Collection[str] = REFERENCE_DOMAINS,
    timeout_s: float = 600.0,
    max_concurrency: int = 4,
    client_factory: Callable[..., AuthorClient] | None = None,
) -> GatewayReferenceBackend:
    if client_factory is None:
        from skillev.r2flow_evolution.author import GatewayAuthorClient

        client_factory = GatewayAuthorClient
    clients = {
        domain: client_factory(
            base_file,
            key_file,
            model,
            timeout_s,
            reasoning_effort=REFERENCE_REASONING_EFFORT[domain],
        )
        for domain in sorted(set(domains))
    }
    return GatewayReferenceBackend(clients, cache_path, max_concurrency=max_concurrency)


def reference_backend_from_config(
    config: Any, run_root: str | Path
) -> GatewayReferenceBackend | None:
    model = getattr(config, "reference_verifier_model", None)
    if model is None:
        return None
    domains = tuple(sorted(config.reference_verifier_domains))
    timeout_s = float(config.reference_verifier_timeout_s)
    concurrency = int(config.reference_verifier_max_concurrency)
    base_file = str(config.author_base_file)
    key_file = str(config.author_key_file)
    cache_path = Path(run_root).resolve() / REFERENCE_CACHE_FILE
    identity = (str(cache_path), model, domains, timeout_s, concurrency, base_file, key_file)
    with _BACKENDS_LOCK:
        backend = _BACKENDS.get(identity)
        if backend is None:
            backend = gateway_reference_backend(
                model=model,
                base_file=base_file,
                key_file=key_file,
                cache_path=cache_path,
                domains=domains,
                timeout_s=timeout_s,
                max_concurrency=concurrency,
            )
            _BACKENDS[identity] = backend
    return backend


__all__ = [
    "AIME_FINAL_NUMBER",
    "COMPARISON_RULES",
    "REFERENCE_CACHE_FILE",
    "REFERENCE_CACHE_FORMAT",
    "REFERENCE_DOMAINS",
    "REFERENCE_MAX_OUTPUT_TOKENS",
    "REFERENCE_NO_FABRICATION",
    "REFERENCE_PROMPTS",
    "REFERENCE_PROMPT_VERSION",
    "REFERENCE_REASONING_EFFORT",
    "TRIVIAQA_CONTAINMENT",
    "GatewayReferenceBackend",
    "ReferenceAnswer",
    "ReferenceAnswerBackend",
    "ReferenceCache",
    "compare_containment",
    "compare_final_number",
    "compare_with_reference",
    "final_number",
    "gateway_reference_backend",
    "last_boxed",
    "last_number",
    "parse_number",
    "parse_reference_reply",
    "reference_backend_from_config",
    "reference_cache_key",
    "reference_identity",
    "reference_prompt",
]
