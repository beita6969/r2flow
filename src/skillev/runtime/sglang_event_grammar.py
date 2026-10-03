from __future__ import annotations

import importlib
import importlib.metadata
import logging
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

from skillev.policy.event_grammar import EVENT_GRAMMAR_VERSION, parse_event_grammar_key

EVENT_GRAMMAR_PATCH = "skillev-event-grammar-patch@1"
PINNED_VERSIONS = {"sglang": "0.5.15.post1", "xgrammar": "0.2.1"}
EVENT_GRAMMAR_CACHE_LIMIT = 1024
COMPILER_CACHE_LIMIT_BYTES = 512 * 1024 * 1024
SERVER_INFO_PATCH_FIELD = "skillev_event_grammar_patch"
SERVER_INFO_XGRAMMAR_FIELD = "xgrammar_version"
_EVENT_KEY_SUFFIX = f'"type":"{EVENT_GRAMMAR_VERSION}"}}'
_MARK = "_skillev_event_grammar"

logger = logging.getLogger(__name__)


def is_event_grammar_key(key_string: object) -> bool:
    return isinstance(key_string, str) and key_string.endswith(_EVENT_KEY_SUFFIX)


def require_pinned_versions() -> dict[str, str]:
    versions = {name: importlib.metadata.version(name) for name in PINNED_VERSIONS}
    if versions != PINNED_VERSIONS:
        raise RuntimeError(f"event grammar patch is pinned to {PINNED_VERSIONS}, found {versions}")
    return versions


def _base_grammar_object() -> type:
    module = importlib.import_module("sglang.srt.constrained.base_grammar_backend")
    return module.BaseGrammarObject


def _make_grammar_class() -> type:
    from skillev.policy.event_closing import (
        ClosingController,
        EventConstraintError,
        apply_restriction,
    )

    base = _base_grammar_object()

    class _SkillevEventGrammar(base):
        def __init__(
            self,
            inner: Any,
            plan: Mapping[str, object],
            budget: int,
            vocab: Sequence[bytes],
            key_string: str,
        ) -> None:
            super().__init__()
            self.inner = inner
            self.plan = plan
            self.budget = budget
            self.vocab = vocab
            self.key_string = key_string
            self.ctl = ClosingController(plan, vocab, budget)
            self.grammar_stats = inner.grammar_stats
            self._history: list[tuple[str, int, int]] = []
            self._fault: str | None = None

        def accept_token(self, token: int) -> None:
            self.current_token = token
            if self._fault is not None:
                raise ValueError(self._fault)
            if self.ctl.done:
                return
            self.inner.accept_token(token)
            state = self.ctl.snapshot()
            self.ctl.advance(token)
            self._history.append(state)

        def rollback(self, k: int) -> None:
            if k <= 0:
                return
            self.inner.rollback(k)
            self.ctl.restore(self._history[-k])
            del self._history[-k:]

        def is_terminated(self) -> bool:
            return bool(self.inner.is_terminated())

        def fill_vocab_mask(self, vocab_mask: Any, idx: int) -> None:
            self.inner.fill_vocab_mask(vocab_mask, idx)
            if self._fault is not None or self.ctl.done:
                return
            try:
                apply_restriction(vocab_mask[idx].numpy(), self.ctl.restriction())
            except EventConstraintError as error:
                self._fault = f"event grammar restriction failed: {error}"
                logger.error("%s key=%s", self._fault, self.key_string[:200])

        def allocate_vocab_mask(self, vocab_size: int, batch_size: int, device: Any) -> Any:
            return self.inner.allocate_vocab_mask(vocab_size, batch_size, device)

        @property
        def move_vocab_mask(self) -> Any:
            return self.inner.move_vocab_mask

        @property
        def apply_vocab_mask(self) -> Any:
            return self.inner.apply_vocab_mask

        def copy(self) -> Any:
            return type(self)(
                self.inner.copy(), self.plan, self.budget, self.vocab, self.key_string
            )

        def try_jump_forward(self, tokenizer: Any) -> None:
            return None

        def jump_forward_str_state(self, helper: Any) -> Any:
            raise NotImplementedError("event grammars never jump forward")

        def jump_and_retokenize(self, old: Any, new: Any, state: Any) -> None:
            raise NotImplementedError("event grammars never jump forward")

        def __repr__(self) -> str:
            return f"SkillevEventGrammar(state={self.ctl.snapshot()}, fault={self._fault!r})"

    _SkillevEventGrammar.__name__ = "SkillevEventGrammar"
    _SkillevEventGrammar.__qualname__ = "SkillevEventGrammar"
    return _SkillevEventGrammar


_GRAMMAR_CLASS: type | None = None


def event_grammar_class() -> type:
    global _GRAMMAR_CLASS
    if _GRAMMAR_CLASS is None:
        _GRAMMAR_CLASS = _make_grammar_class()
    return _GRAMMAR_CLASS


def _backend_vocab(backend: Any) -> tuple[list[bytes], str]:
    from skillev.policy.event_closing import decoded_vocab, tokenizer_info_sha256, vocab_index

    cached: tuple[list[bytes], str] | None = getattr(backend, "_skillev_vocab", None)
    if cached is None:
        info = backend._skillev_tokenizer_info
        vocab = decoded_vocab(info)
        cached = (vocab, tokenizer_info_sha256(info))
        vocab_index(vocab)
        backend._skillev_vocab = cached
    return cached


def install_event_grammar(*, server_info: bool = False) -> None:
    versions = require_pinned_versions()
    xgr = importlib.import_module("xgrammar")
    xgb = importlib.import_module("sglang.srt.constrained.xgrammar_backend")
    bgb = importlib.import_module("sglang.srt.constrained.base_grammar_backend")
    backend_class = xgb.XGrammarGrammarBackend
    if server_info:
        _install_server_info(versions)
    if getattr(backend_class.dispatch_structural_tag, _MARK, False):
        return
    invalid = bgb.InvalidGrammarObject
    original_init = backend_class.__init__
    original_dispatch = backend_class.dispatch_structural_tag
    original_set_cache = bgb.BaseGrammarBackend.set_cache

    @wraps(original_init)
    def init(self: Any, tokenizer: Any, vocab_size: int, *args: Any, **kwargs: Any) -> None:
        original_init(self, tokenizer, vocab_size, *args, **kwargs)
        model_eos = kwargs.get("model_eos_token_ids", args[0] if args else None)
        if hasattr(tokenizer, "init_xgrammar"):
            info, _ = tokenizer.init_xgrammar()
        else:
            info = xgr.TokenizerInfo.from_huggingface(
                tokenizer, vocab_size=vocab_size, stop_token_ids=model_eos
            )
        self._skillev_tokenizer_info = info
        self.grammar_compiler = xgr.GrammarCompiler(
            tokenizer_info=info, cache_limit_bytes=COMPILER_CACHE_LIMIT_BYTES
        )
        _backend_vocab(self)

    @wraps(original_dispatch)
    def dispatch(self: Any, key_string: str) -> Any:
        if not is_event_grammar_key(key_string):
            return original_dispatch(self, key_string)
        try:
            key = parse_event_grammar_key(key_string)
        except ValueError as error:
            return invalid(f"invalid event grammar key: {error}")
        vocab, info_sha = _backend_vocab(self)
        if info_sha != key.tokenizer_info_sha256:
            return invalid("event grammar tokenizer mismatch")
        inner = original_dispatch(self, key.structural_tag_json())
        if isinstance(inner, invalid):
            return inner
        try:
            return event_grammar_class()(inner, key.plan, key.budget, vocab, key_string)
        except ValueError as error:
            return invalid(f"invalid event grammar plan: {error}")

    @wraps(original_set_cache)
    def set_cache(self: Any, key: tuple[str, str], value: Any) -> None:
        original_set_cache(self, key, value)
        if key[0] != "structural_tag" or not is_event_grammar_key(key[1]):
            return
        order = self.__dict__.setdefault("_skillev_event_keys", OrderedDict())
        order[key] = None
        order.move_to_end(key)
        while len(order) > EVENT_GRAMMAR_CACHE_LIMIT:
            oldest, _ = order.popitem(last=False)
            self.cache.pop(oldest, None)

    setattr(dispatch, _MARK, True)
    backend_class.__init__ = init
    backend_class.dispatch_structural_tag = dispatch
    bgb.BaseGrammarBackend.set_cache = set_cache


def _install_server_info(versions: Mapping[str, str]) -> None:
    http = importlib.import_module("sglang.srt.entrypoints.http_server")
    original = http.set_global_state
    if getattr(original, _MARK, False):
        return

    @wraps(original)
    def set_global_state(global_state: Any) -> None:
        info = getattr(global_state, "scheduler_info", None)
        if isinstance(info, dict):
            info[SERVER_INFO_PATCH_FIELD] = EVENT_GRAMMAR_PATCH
            info[SERVER_INFO_XGRAMMAR_FIELD] = versions["xgrammar"]
        original(global_state)

    setattr(set_global_state, _MARK, True)
    setattr(http, "set_global_state", set_global_state)


__all__ = [
    "COMPILER_CACHE_LIMIT_BYTES",
    "EVENT_GRAMMAR_CACHE_LIMIT",
    "EVENT_GRAMMAR_PATCH",
    "PINNED_VERSIONS",
    "SERVER_INFO_PATCH_FIELD",
    "SERVER_INFO_XGRAMMAR_FIELD",
    "event_grammar_class",
    "install_event_grammar",
    "is_event_grammar_key",
    "require_pinned_versions",
]
