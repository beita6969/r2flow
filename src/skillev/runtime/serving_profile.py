from __future__ import annotations

from collections.abc import Mapping

from skillev.contracts import JsonValue, normalize_json
from skillev.runtime.sglang_event_grammar import (
    EVENT_GRAMMAR_PATCH,
    PINNED_VERSIONS,
    SERVER_INFO_PATCH_FIELD,
    SERVER_INFO_XGRAMMAR_FIELD,
)

_FIELDS = (
    "model_path",
    "tokenizer_path",
    "revision",
    "served_model_name",
    "dtype",
    "quantization",
    "kv_cache_dtype",
    "context_length",
    "enable_lora",
    "max_lora_rank",
    "lora_target_modules",
    "enable_deterministic_inference",
    "sampling_backend",
    "attention_backend",
    "linear_attn_backend",
    "disable_radix_cache",
    "mamba_radix_cache_strategy",
    "page_size",
    "chunked_prefill_size",
    "max_prefill_tokens",
    "disable_overlap_schedule",
    "mamba_ssm_dtype",
    "enable_int8_mamba_checkpoint",
    "enable_tf32_matmul",
    "disable_decode_cuda_graph",
    "disable_prefill_cuda_graph",
)
_REQUIRED = (
    "model_path",
    "tokenizer_path",
    "served_model_name",
    "dtype",
    "context_length",
    "enable_lora",
    "enable_deterministic_inference",
    "sampling_backend",
)


_EVENT_GRAMMAR_FIELDS = ("grammar_backend", "reasoning_parser", "enable_strict_thinking")


def serving_profile(value: object, *, event_grammar: bool = False) -> dict[str, JsonValue]:
    data = normalize_json(value)
    if not isinstance(data, dict):
        raise ValueError("SGLang server info must be an object")
    args = data.get("server_args", data)
    if not isinstance(args, dict) or any(k not in args for k in _REQUIRED):
        raise ValueError("SGLang server info omits required execution settings")
    version = data.get("version", args.get("version"))
    if not isinstance(version, str) or not version:
        raise ValueError("SGLang server info omits its actual version")
    for name in ("model_path", "tokenizer_path", "served_model_name", "dtype"):
        if not isinstance(args[name], str) or not args[name]:
            raise ValueError(f"SGLang server info has invalid {name}")
    if type(args["context_length"]) is not int or args["context_length"] < 1:
        raise ValueError("SGLang server context length is invalid")
    profile: dict[str, JsonValue] = {
        "version": version,
        **{name: args.get(name) for name in _FIELDS},
    }
    if event_grammar:
        profile.update({name: args.get(name) for name in _EVENT_GRAMMAR_FIELDS})
        profile["xgrammar_version"] = data.get(SERVER_INFO_XGRAMMAR_FIELD)
        profile["event_grammar_patch"] = data.get(SERVER_INFO_PATCH_FIELD)
    return profile


def require_same_profile(
    expected: Mapping[str, JsonValue], actual: Mapping[str, JsonValue]
) -> None:
    changed = sorted(k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k))
    if changed:
        raise ValueError("serving execution differs in: " + ", ".join(changed))


def require_training_service(
    profile: Mapping[str, JsonValue],
    *,
    model_path: str,
    tokenizer_path: str,
    base_model: str,
    minimum_context: int,
    actor: bool,
    event_grammar: bool = False,
) -> None:
    require_same_profile(
        {
            "model_path": model_path,
            "tokenizer_path": tokenizer_path,
            "served_model_name": base_model,
        },
        {k: profile[k] for k in ("model_path", "tokenizer_path", "served_model_name")},
    )
    context = profile["context_length"]
    if type(context) is not int or context < minimum_context:
        raise ValueError("serving context cannot admit the declared input/output budget")
    if actor and (
        profile["enable_lora"] is not True
        or profile["enable_deterministic_inference"] is not True
        or profile["sampling_backend"] != "pytorch"
    ):
        raise ValueError("actor service does not support the declared LoRA/raw sampling path")
    if (
        actor
        and event_grammar
        and (
            profile.get("grammar_backend") != "xgrammar"
            or profile.get("enable_strict_thinking") is not False
            or profile.get("event_grammar_patch") != EVENT_GRAMMAR_PATCH
            or profile.get("version") != PINNED_VERSIONS["sglang"]
            or profile.get("xgrammar_version") != PINNED_VERSIONS["xgrammar"]
        )
    ):
        raise ValueError("actor service does not run the pinned event-grammar patch")
