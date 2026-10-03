from __future__ import annotations

import hashlib
import importlib
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from .event_grammar import (
    BUDGET_FORCED_CLOSING,
    EX,
    EventGrammarKey,
    EventGrammarSpec,
    F,
    T,
    build_token_plan,
    event_grammar_key,
    parse_event_grammar_key,
)

CLOSING_VERSION = BUDGET_FORCED_CLOSING
ACTION_MASK_PLAN_VERSION = "event-action-mask-plan@1"
TOKEN_BITMASK_FORMAT = "xgrammar-bitmask@1"

_T = T.encode("utf-8")
_F = F.encode("utf-8")
_DONE = -1
_TEXT = -2
_INF = 1 << 40
_TAIL = 100
_INVALID = -1

Int32Row = npt.NDArray[np.int32]


class EventConstraintError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Only:
    token: int


@dataclass(frozen=True, slots=True)
class Subset:
    tokens: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Minus:
    tokens: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Keep:
    pass


Restriction = Only | Subset | Minus | Keep


def restriction_allows(restriction: Restriction, token: int) -> bool:
    if isinstance(restriction, Only):
        return token == restriction.token
    if isinstance(restriction, Subset):
        return token in restriction.tokens
    if isinstance(restriction, Minus):
        return token not in restriction.tokens
    return True


def _bits(tokens: Sequence[int], words: int) -> npt.NDArray[np.uint32]:
    mask = np.zeros(words, dtype=np.uint32)
    ids = np.asarray(tokens, dtype=np.int64)
    if ids.size:
        if int(ids.min()) < 0 or int(ids.max()) >= words * 32:
            raise EventConstraintError("restriction token outside the bitmask")
        np.bitwise_or.at(mask, ids >> 5, np.left_shift(np.uint32(1), (ids & 31).astype(np.uint32)))
    return mask


def apply_restriction(row: Int32Row, restriction: Restriction) -> None:
    view = row.view(np.uint32)
    if isinstance(restriction, Keep):
        return
    if isinstance(restriction, Only):
        word, bit = divmod(restriction.token, 32)
        if word >= view.shape[0] or not (int(view[word]) >> bit) & 1:
            raise EventConstraintError("forced token is not grammar-legal")
        view[:] = 0
        view[word] = np.uint32(1 << bit)
        return
    mask = _bits(restriction.tokens, view.shape[0])
    if isinstance(restriction, Subset):
        kept = view & mask
    else:
        kept = view & ~mask
    if not kept.any():
        raise EventConstraintError("restricted mask is empty")
    view[:] = kept


def row_allows(row: Int32Row, token: int) -> bool:
    word, bit = divmod(token, 32)
    view = row.view(np.uint32)
    return word < view.shape[0] and bool((int(view[word]) >> bit) & 1)


def row_popcount(row: Int32Row) -> int:
    return int(np.unpackbits(row.view(np.uint8)).sum())


def _failure(pattern: bytes) -> list[int]:
    fail = [0] * len(pattern)
    k = 0
    for index in range(1, len(pattern)):
        while k and pattern[index] != pattern[k]:
            k = fail[k - 1]
        if pattern[index] == pattern[k]:
            k += 1
        fail[index] = k
    return fail


_T_FAIL = _failure(_T)


def text_outcome(j: int, data: bytes) -> int:
    for index, byte in enumerate(data):
        while j and byte != _T[j]:
            j = _T_FAIL[j - 1]
        if byte == _T[j]:
            j += 1
        if j == len(_T):
            leftover = data[index + 1 :]
            if len(leftover) <= len(_F) and _F.startswith(leftover):
                return _TAIL + len(leftover)
            return _INVALID
    return j


def _kmp(j: int, byte: int) -> int:
    while j and byte != _T[j]:
        j = _T_FAIL[j - 1]
    return j + 1 if byte == _T[j] else j


def _aho_corasick(patterns: Sequence[bytes]) -> tuple[list[list[int]], list[bool]]:
    goto: list[dict[int, int]] = [{}]
    out = [False]
    for pattern in patterns:
        node = 0
        for byte in pattern:
            if byte not in goto[node]:
                goto.append({})
                out.append(False)
                goto[node][byte] = len(goto) - 1
            node = goto[node][byte]
        out[node] = True
    fail = [0] * len(goto)
    delta = [[0] * 256 for _ in goto]
    for byte in range(256):
        delta[0][byte] = goto[0].get(byte, 0)
    queue = list(goto[0].values())
    while queue:
        node = queue.pop(0)
        out[node] = out[node] or out[fail[node]]
        for byte in range(256):
            child = goto[node].get(byte)
            if child is None:
                delta[node][byte] = delta[fail[node]][byte]
            else:
                fail[child] = delta[fail[node]][byte]
                delta[node][byte] = child
                queue.append(child)
    return delta, out


_EX_DELTA, _EX_OUT = _aho_corasick([item.encode("utf-8") for item in EX])
_TEXT_STRIDE = 16
assert len(_T) < _TEXT_STRIDE
assert all(item.endswith(">") for item in EX)


def text_step(j: int, e: int, data: bytes) -> tuple[str, int]:
    for index, byte in enumerate(data):
        while j and byte != _T[j]:
            j = _T_FAIL[j - 1]
        if byte == _T[j]:
            j += 1
        e = _EX_DELTA[e][byte]
        if j == len(_T):
            leftover = data[index + 1 :]
            if len(leftover) <= len(_F) and _F.startswith(leftover):
                return ("tail", len(leftover))
            return ("invalid", 0)
        if _EX_OUT[e]:
            return ("invalid", 0)
    return ("text", j + _TEXT_STRIDE * e)


@dataclass(frozen=True, slots=True)
class _Exceptions:
    tokens: npt.NDArray[np.int64]
    codes: npt.NDArray[np.int64]


class VocabIndex:
    def __init__(self, decoded_vocab: Sequence[bytes]) -> None:
        self.decoded_vocab = decoded_vocab
        self.size = len(decoded_vocab)
        by_bytes: dict[bytes, list[int]] = {}
        by_first: dict[int, list[int]] = {}
        newline: list[int] = []
        for token, data in enumerate(decoded_vocab):
            by_bytes.setdefault(data, []).append(token)
            if b"\n" in data:
                newline.append(token)
            elif data:
                by_first.setdefault(data[0], []).append(token)
        self.exceptions: list[_Exceptions] = []
        for j in range(len(_T)):
            suffix = _T[j:]
            candidates = set(newline)
            for size in range(len(suffix) + 1):
                candidates.update(by_bytes.get(suffix[:size], ()))
            if b"\n" not in suffix:
                candidates.update(
                    token
                    for token in by_first.get(suffix[0], ())
                    if decoded_vocab[token].startswith(suffix)
                )
            by_token: dict[int, int] = {}
            for token in candidates:
                code = text_outcome(j, decoded_vocab[token])
                if code != 0:
                    by_token[token] = code
            ordered = sorted(by_token)
            self.exceptions.append(
                _Exceptions(
                    np.asarray(ordered, dtype=np.int64),
                    np.asarray([by_token[token] for token in ordered], dtype=np.int64),
                )
            )
        self.closers_by_first: dict[int, list[int]] = {}
        for token, data in enumerate(decoded_vocab):
            if b">" in data:
                self.closers_by_first.setdefault(data[0], []).append(token)
        self.invalid_from_start = sorted(
            token
            for tokens in self.closers_by_first.values()
            for token in tokens
            if text_step(0, 0, decoded_vocab[token])[0] == "invalid"
        )
        self._invalid: dict[int, npt.NDArray[np.int64]] = {}

    def invalid_text_tokens(self, position: int) -> npt.NDArray[np.int64]:
        cached = self._invalid.get(position)
        if cached is None:
            j, e = position % _TEXT_STRIDE, position // _TEXT_STRIDE
            divergent = {
                byte
                for byte in range(256)
                if _EX_DELTA[e][byte] != _EX_DELTA[0][byte] or _kmp(j, byte) != _kmp(0, byte)
            }
            invalid = {
                token
                for token in self.invalid_from_start
                if self.decoded_vocab[token][0] not in divergent
            }
            invalid.update(
                token
                for byte in divergent
                for token in self.closers_by_first.get(byte, ())
                if text_step(j, e, self.decoded_vocab[token])[0] == "invalid"
            )
            cached = np.asarray(sorted(invalid), dtype=np.int64)
            self._invalid[position] = cached
        return cached


_VOCAB_INDEX: dict[int, tuple[Sequence[bytes], VocabIndex]] = {}


def vocab_index(decoded_vocab: Sequence[bytes]) -> VocabIndex:
    cached = _VOCAB_INDEX.get(id(decoded_vocab))
    if cached is not None and cached[0] is decoded_vocab:
        return cached[1]
    index = VocabIndex(decoded_vocab)
    if len(_VOCAB_INDEX) >= 4:
        _VOCAB_INDEX.pop(next(iter(_VOCAB_INDEX)))
    _VOCAB_INDEX[id(decoded_vocab)] = (decoded_vocab, index)
    return index


def _ids(value: object, *, what: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"{what} must be a list of token ids")
    return tuple(value)


class _Graph:
    def __init__(self, plan: Mapping[str, object]) -> None:
        self.edges: list[dict[int, int]] = []
        stop = _ids(plan["stop"], what="stop")
        if len(stop) != 1:
            raise ValueError("exactly one stop token")
        self.stop_token = stop[0]
        self.stop_node = self._node()
        self.edges[self.stop_node][self.stop_token] = _DONE
        name_root = self._node()
        functions = plan["functions"]
        if not isinstance(functions, list) or not functions:
            raise ValueError("plan needs functions")
        for function in functions:
            if not isinstance(function, dict):
                raise ValueError("plan function must be an object")
            program = function["program"]
            if not isinstance(program, list) or not program:
                raise ValueError("plan program must be a non-empty list")
            entry = self._steps(program, 0)
            end = self._path(name_root, _ids(function["name_ids"], what="name_ids"))
            self._merge(end, entry)
        self.root = self._chain(_ids(plan["open"], what="open"), name_root)

    def _node(self) -> int:
        self.edges.append({})
        return len(self.edges) - 1

    def _link(self, node: int, token: int, target: int) -> None:
        existing = self.edges[node].get(token)
        if existing is not None and existing != target:
            raise ValueError("ambiguous token plan")
        self.edges[node][token] = target

    def _merge(self, node: int, entry: int) -> None:
        if entry < 0:
            raise ValueError("a choice or name cannot be followed directly by free text")
        for token, target in self.edges[entry].items():
            self._link(node, token, target)

    def _chain(self, ids: tuple[int, ...], target: int) -> int:
        if not ids:
            raise ValueError("forced runs must be non-empty")
        current = target
        for token in reversed(ids):
            node = self._node()
            self._link(node, token, current)
            current = node
        return current

    def _path(self, root: int, ids: tuple[int, ...]) -> int:
        if not ids:
            raise ValueError("choice encodings must be non-empty")
        node = root
        for token in ids:
            target = self.edges[node].get(token)
            if target is not None and target < 0:
                raise ValueError("ambiguous token plan")
            if target is None:
                target = self._node()
                self._link(node, token, target)
            node = target
        return node

    def _steps(self, program: list[Any], index: int) -> int:
        if index == len(program):
            return self.stop_node
        step = program[index]
        if not isinstance(step, list) or not step:
            raise ValueError("plan steps must be non-empty lists")
        kind = step[0]
        if kind == "text":
            if index != len(program) - 1 or len(step) != 1:
                raise ValueError("text must be the final program step")
            return _TEXT
        rest = self._steps(program, index + 1)
        if kind == "forced" and len(step) == 2:
            return self._chain(_ids(step[1], what="forced"), rest)
        if kind == "choice" and len(step) == 2 and isinstance(step[1], list) and step[1]:
            root = self._node()
            for choice in step[1]:
                self._merge(self._path(root, _ids(choice, what="choice")), rest)
            return root
        raise ValueError(f"invalid program step {kind!r}")


class ClosingController:
    def __init__(
        self, plan: Mapping[str, object], decoded_vocab: Sequence[bytes], budget: int
    ) -> None:
        if type(budget) is not int or budget < 1:
            raise ValueError("budget must be a positive integer")
        self.budget = budget
        self.index = vocab_index(decoded_vocab)
        self.graph = _Graph(plan)
        size = self.index.size
        for node in self.graph.edges:
            if any(token >= size for token in node):
                raise ValueError("plan token id outside the vocabulary")
        self.terminator = [
            _ids(item, what="terminator_closing") for item in plan["terminator_closing"]
        ]
        self.tail = [_ids(item, what="tail_closing") for item in plan["tail_closing"]]
        if len(self.terminator) != len(_T) or len(self.tail) != len(_F):
            raise ValueError("closing tables do not match the terminator and tail")
        self._check_closing(self.terminator, _T)
        self._check_closing(self.tail, _F)
        self.l_tail = [len(ids) + 1 for ids in self.tail] + [1]
        self.l_text = [len(ids) + self.l_tail[0] for ids in self.terminator]
        lookup = np.full(_TAIL + len(_F) + 2, _INF, dtype=np.int64)
        lookup[: len(_T)] = self.l_text
        lookup[_TAIL : _TAIL + len(_F) + 1] = self.l_tail
        self._lookup = lookup
        self._exc_l = [lookup[exc.codes] for exc in self.index.exceptions]
        self.l_text_max = max(self.l_text)
        self._l_node = self._node_lengths()
        self.stage = "graph"
        self.position = self.graph.root
        self.accepted = 0
        if budget < self.closing_length():
            raise ValueError("action budget below shortest legal event")

    def _check_closing(self, table: list[tuple[int, ...]], text: bytes) -> None:
        vocab = self.index.decoded_vocab
        for offset, ids in enumerate(table):
            if not ids or ids[0] >= self.index.size:
                raise ValueError("closing encodings must be non-empty vocabulary ids")
            first = vocab[ids[0]]
            if not first or not text[offset:].startswith(first):
                raise ValueError("closing token does not spell the closing text")
            following = offset + len(first)
            expected = table[following] if following < len(text) else ()
            if ids[1:] != expected:
                raise ValueError("closing encodings are not suffix-consistent")

    def _node_lengths(self) -> list[int]:
        edges = self.graph.edges
        lengths = [-1] * len(edges)

        def target_length(target: int) -> int:
            if target == _DONE:
                return 0
            if target == _TEXT:
                return self.l_text[0]
            return lengths[target]

        stack = [(self.graph.root, False)]
        while stack:
            node, expanded = stack.pop()
            if lengths[node] >= 0:
                continue
            pending = [t for t in edges[node].values() if t >= 0 and lengths[t] < 0]
            if not expanded and pending:
                stack.append((node, True))
                stack.extend((target, False) for target in pending)
                continue
            lengths[node] = 1 + min(target_length(t) for t in edges[node].values())
        return lengths

    def _target_length(self, target: int) -> int:
        if target == _DONE:
            return 0
        if target == _TEXT:
            return self.l_text[0]
        return self._l_node[target]

    @property
    def remaining(self) -> int:
        return self.budget - self.accepted

    @property
    def done(self) -> bool:
        return self.stage == "done"

    def closing_length(self) -> int:
        if self.stage == "graph":
            return self._l_node[self.position]
        if self.stage == "text":
            return self.l_text[self.position % _TEXT_STRIDE]
        if self.stage == "tail":
            return self.l_tail[self.position]
        return 0

    def c1(self) -> int:
        if self.stage == "graph":
            edges = self.graph.edges[self.position]
            return min(edges, key=lambda token: (self._target_length(edges[token]), token))
        if self.stage == "text":
            return self.terminator[self.position % _TEXT_STRIDE][0]
        if self.stage == "tail":
            return self.tail[self.position][0]
        raise EventConstraintError("the event is complete")

    @property
    def text_state(self) -> tuple[int, int]:
        if self.stage != "text":
            raise EventConstraintError("not in free text")
        return self.position % _TEXT_STRIDE, self.position // _TEXT_STRIDE

    def is_free(self) -> bool:
        if self.stage == "graph":
            return len(self.graph.edges[self.position]) > 1
        return self.stage == "text"

    def budget_forced(self) -> bool:
        return self.is_free() and self.remaining == self.closing_length()

    def snapshot(self) -> tuple[str, int, int]:
        return (self.stage, self.position, self.accepted)

    def restore(self, state: tuple[str, int, int]) -> None:
        self.stage, self.position, self.accepted = state

    def restriction(self) -> Restriction:
        if self.stage == "done":
            raise EventConstraintError("the event is complete")
        remaining = self.remaining
        if remaining == self.closing_length():
            return Only(self.c1())
        if self.stage == "tail":
            return Only(self.tail[self.position][0])
        if self.stage == "graph":
            edges = self.graph.edges[self.position]
            if len(edges) == 1:
                return Only(next(iter(edges)))
            allowed = tuple(
                sorted(
                    token
                    for token, target in edges.items()
                    if self._target_length(target) <= remaining - 1
                )
            )
            return Only(allowed[0]) if len(allowed) == 1 else Subset(allowed)
        invalid = self.index.invalid_text_tokens(self.position)
        j = self.position % _TEXT_STRIDE
        if remaining - 1 >= self.l_text_max:
            return Minus(tuple(int(t) for t in invalid)) if invalid.size else Keep()
        exc = self.index.exceptions[j]
        lengths = self._exc_l[j]
        if self.l_text[0] <= remaining - 1:
            bad = np.union1d(exc.tokens[lengths > remaining - 1], invalid)
            return Minus(tuple(int(t) for t in bad)) if bad.size else Keep()
        ok = np.setdiff1d(exc.tokens[lengths <= remaining - 1], invalid)
        return Subset(tuple(int(t) for t in ok))

    def _next(self, token: int) -> tuple[str, int]:
        if self.stage == "graph":
            target = self.graph.edges[self.position].get(token)
            if target is None:
                raise EventConstraintError("token is not on the canonical token plan")
            if target == _DONE:
                return ("done", 0)
            if target == _TEXT:
                return ("text", 0)
            return ("graph", target)
        if self.stage == "text":
            j, e = self.position % _TEXT_STRIDE, self.position // _TEXT_STRIDE
            stage, position = text_step(j, e, self.index.decoded_vocab[token])
            if stage == "invalid":
                raise EventConstraintError("free-text token breaks the value language")
            if stage == "tail":
                return self._tail(position)
            return (stage, position)
        if self.stage == "tail":
            if token != self.tail[self.position][0]:
                raise EventConstraintError("call tail is not canonical")
            return self._tail(self.position + len(self.index.decoded_vocab[token]))
        raise EventConstraintError("the event is complete")

    def _tail(self, offset: int) -> tuple[str, int]:
        if offset == len(_F):
            return ("graph", self.graph.stop_node)
        return ("tail", offset)

    def length_after(self, token: int) -> int | None:
        try:
            stage, position = self._next(token)
        except EventConstraintError:
            return None
        saved = self.snapshot()
        self.stage, self.position = stage, position
        try:
            return self.closing_length()
        finally:
            self.restore(saved)

    def advance(self, token: int) -> None:
        if type(token) is not int or not 0 <= token < self.index.size:
            raise EventConstraintError("token id outside the vocabulary")
        if not restriction_allows(self.restriction(), token):
            raise EventConstraintError("token outside the canonical/budget restriction")
        self.stage, self.position = self._next(token)
        self.accepted += 1
        if self.remaining < self.closing_length():
            raise EventConstraintError("budget invariant violated")


def _xgrammar() -> Any:
    return importlib.import_module("xgrammar")


_DECODED_VOCAB: dict[int, tuple[Any, list[bytes], str]] = {}


def _tokenizer_entry(info: Any) -> tuple[list[bytes], str]:
    cached = _DECODED_VOCAB.get(id(info))
    if cached is not None and cached[0] is info:
        return cached[1], cached[2]
    vocab = list(info.decoded_vocab)
    digest = hashlib.sha256()
    digest.update(b"xgrammar-tokenizer-info@1\0")
    digest.update(int(info.vocab_size).to_bytes(8, "little"))
    stops = sorted(int(token) for token in info.stop_token_ids)
    digest.update(len(stops).to_bytes(8, "little"))
    for token in stops:
        digest.update(token.to_bytes(8, "little"))
    digest.update(len(vocab).to_bytes(8, "little"))
    for data in vocab:
        digest.update(len(data).to_bytes(4, "little"))
        digest.update(data)
    if len(_DECODED_VOCAB) >= 4:
        _DECODED_VOCAB.pop(next(iter(_DECODED_VOCAB)))
    _DECODED_VOCAB[id(info)] = (info, vocab, digest.hexdigest())
    return vocab, digest.hexdigest()


def tokenizer_info_sha256(info: Any) -> str:
    return _tokenizer_entry(info)[1]


def decoded_vocab(info: Any) -> list[bytes]:
    return _tokenizer_entry(info)[0]


def formal_tokenizer_info(
    tokenizer: Any, *, vocab_size: int = 248320, stop_token_ids: Sequence[int] = (248044,)
) -> Any:
    xgr = _xgrammar()

    return xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=vocab_size, stop_token_ids=list(stop_token_ids)
    )


class BudgetedEventMatcher:
    def __init__(
        self,
        compiled_grammar: Any,
        plan: Mapping[str, object],
        budget: int,
        tokenizer_info: Any,
    ) -> None:
        xgr = _xgrammar()

        self.vocab_size = int(tokenizer_info.vocab_size)
        self.matcher = xgr.GrammarMatcher(compiled_grammar, max_rollback_tokens=200)
        self.ctl = ClosingController(plan, decoded_vocab(tokenizer_info), budget)
        self._bitmask = xgr.allocate_token_bitmask(1, self.vocab_size)

    def fill(self) -> Int32Row:
        self.matcher.fill_next_token_bitmask(self._bitmask, 0)
        row: Int32Row = self._bitmask[0].numpy().copy()
        apply_restriction(row, self.ctl.restriction())
        return row

    def accept(self, token: int) -> None:
        if not self.matcher.accept_token(token):
            raise EventConstraintError("xgrammar rejected the token")
        self.ctl.advance(token)

    def is_terminated(self) -> bool:
        terminated = bool(self.matcher.is_terminated())
        if terminated != self.ctl.done:
            raise EventConstraintError("xgrammar termination disagrees with the controller")
        return terminated


_COMPILERS: dict[str, Any] = {}
_COMPILED: OrderedDict[tuple[str, str], Any] = OrderedDict()
_COMPILED_LIMIT = 256


def compile_event_grammar(key: EventGrammarKey, info: Any) -> Any:
    xgr = _xgrammar()

    info_sha = tokenizer_info_sha256(info)
    if info_sha != key.tokenizer_info_sha256:
        raise EventConstraintError("event grammar tokenizer mismatch")
    cache_key = (info_sha, key.sha256)
    compiled = _COMPILED.get(cache_key)
    if compiled is not None:
        _COMPILED.move_to_end(cache_key)
        return compiled
    compiler = _COMPILERS.get(info_sha)
    if compiler is None:
        compiler = xgr.GrammarCompiler(info, max_threads=1, cache_enabled=False)
        _COMPILERS[info_sha] = compiler
    compiled = compiler.compile_structural_tag(key.structural_tag_json())
    _COMPILED[cache_key] = compiled
    while len(_COMPILED) > _COMPILED_LIMIT:
        _COMPILED.popitem(last=False)
    return compiled


@dataclass(frozen=True, slots=True)
class ActionMaskPlan:
    key_sha256: str
    vocab_size: int
    token_ids: tuple[int, ...]
    forced: tuple[bool, ...]
    free_positions: tuple[int, ...]
    free_rows: Int32Row
    budget_forced_positions: tuple[int, ...]
    terminated: bool
    digest: str
    format: str = ACTION_MASK_PLAN_VERSION

    @property
    def words(self) -> int:
        return (self.vocab_size + 31) // 32

    def row(self, position: int) -> Int32Row:
        if self.forced[position]:
            row = np.zeros(self.words, dtype=np.int32)
            row.view(np.uint32)[self.token_ids[position] // 32] = np.uint32(
                1 << (self.token_ids[position] % 32)
            )
            return row
        row = self.free_rows[self.free_positions.index(position)]
        return np.array(row, dtype=np.int32)

    def packed_rows(self) -> Int32Row:
        if not self.token_ids:
            return np.zeros((0, self.words), dtype=np.int32)
        return np.stack([self.row(position) for position in range(len(self.token_ids))])

    def allowed(self, position: int) -> npt.NDArray[np.bool_]:
        bits = np.unpackbits(self.row(position).view(np.uint8), bitorder="little")
        return bits[: self.vocab_size].astype(bool)


def replay_action_masks(
    key: str,
    action_ids: Sequence[int],
    stop_ids: Sequence[int],
    tokenizer_info: Any,
    *,
    require_terminated: bool = True,
) -> ActionMaskPlan:
    parsed = parse_event_grammar_key(key)
    compiled = compile_event_grammar(parsed, tokenizer_info)
    matcher = BudgetedEventMatcher(compiled, parsed.plan, parsed.budget, tokenizer_info)
    tokens = tuple(int(token) for token in (*action_ids, *stop_ids))
    forced: list[bool] = []
    free_positions: list[int] = []
    rows: list[Int32Row] = []
    budget_positions: list[int] = []
    digest = hashlib.sha256()
    digest.update(ACTION_MASK_PLAN_VERSION.encode() + b"\0" + parsed.sha256.encode())
    for position, token in enumerate(tokens):
        if matcher.ctl.done:
            raise EventConstraintError("tokens follow the completed event")
        if matcher.ctl.budget_forced():
            budget_positions.append(position)
        row = matcher.fill()
        if not row_allows(row, token):
            raise EventConstraintError(f"token {token} at position {position} is masked")
        is_forced = row_popcount(row) == 1
        forced.append(is_forced)
        digest.update(position.to_bytes(4, "little") + token.to_bytes(4, "little"))
        if not is_forced:
            free_positions.append(position)
            rows.append(row)
            digest.update(row.tobytes())
        matcher.accept(token)
    terminated = matcher.is_terminated()
    if require_terminated and (
        not terminated
        or tuple(stop_ids) != (parsed.stop_token_id,)
        or len(action_ids) + 1 > parsed.budget
    ):
        raise EventConstraintError("event did not terminate on the stop token within budget")
    words = (matcher.vocab_size + 31) // 32
    return ActionMaskPlan(
        key_sha256=parsed.sha256,
        vocab_size=matcher.vocab_size,
        token_ids=tokens,
        forced=tuple(forced),
        free_positions=tuple(free_positions),
        free_rows=np.stack(rows) if rows else np.zeros((0, words), dtype=np.int32),
        budget_forced_positions=tuple(budget_positions),
        terminated=terminated,
        digest=digest.hexdigest(),
    )


def event_action_masks(
    spec: EventGrammarSpec,
    encode: Any,
    tokenizer_info: Any,
    action_ids: Sequence[int],
    stop_ids: Sequence[int],
    *,
    require_terminated: bool = True,
) -> tuple[str, ActionMaskPlan]:
    key = event_grammar_key(
        spec, build_token_plan(spec, encode), tokenizer_info_sha256(tokenizer_info)
    )
    return key, replay_action_masks(
        key, action_ids, stop_ids, tokenizer_info, require_terminated=require_terminated
    )


__all__ = [
    "ACTION_MASK_PLAN_VERSION",
    "CLOSING_VERSION",
    "TOKEN_BITMASK_FORMAT",
    "ActionMaskPlan",
    "BudgetedEventMatcher",
    "ClosingController",
    "EventConstraintError",
    "Keep",
    "Minus",
    "Only",
    "Restriction",
    "Subset",
    "VocabIndex",
    "apply_restriction",
    "compile_event_grammar",
    "decoded_vocab",
    "event_action_masks",
    "formal_tokenizer_info",
    "replay_action_masks",
    "restriction_allows",
    "row_allows",
    "row_popcount",
    "text_outcome",
    "text_step",
    "tokenizer_info_sha256",
    "vocab_index",
]
