from __future__ import annotations

import importlib
import inspect
from functools import wraps
from typing import Any


def install_fp32_mamba_checkpoints() -> None:
    import torch

    state = importlib.import_module("sglang.srt.layers.attention.fla.chunk_delta_h")
    chunk: Any = importlib.import_module("sglang.srt.layers.attention.fla.chunk")
    original = chunk.chunk_gated_delta_rule_fwd_h
    if getattr(original, "_skillev_fp32_checkpoints", False):
        return
    signature = inspect.signature(original)
    output = chunk.chunk_fwd_o
    output_signature = inspect.signature(output)

    @wraps(original)
    def checkpoints(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        k, u, initial = values["k"], values["u"], values["initial_state"]
        if initial is None or initial.dtype != torch.float32:
            raise ValueError("FP32 Mamba checkpoints require a real FP32 SSM pool")
        b, t, hg, key_dim = k.shape
        heads, value_dim = u.shape[-2:]
        block = state.CHUNK_SIZE
        sequences, indices = values["cu_seqlens"], values["chunk_indices"]
        if sequences is None:
            count, chunks, offsets = b, (t + block - 1) // block, None
        else:
            if indices is None:
                indices = state.prepare_chunk_indices(sequences, block)
            count, chunks = len(sequences) - 1, len(indices)
            offsets = state.prepare_chunk_offsets(sequences, block)
        if key_dim > 256:
            raise ValueError("upstream Mamba checkpoint kernel requires head dimension <= 256")
        history = torch.empty(
            (b, chunks, heads, value_dim, key_dim), device=k.device, dtype=initial.dtype
        )
        new_values = torch.empty_like(u) if values["save_new_value"] else None

        def grid(meta: dict[str, int]) -> tuple[int, int]:
            return (value_dim + meta["BV"] - 1) // meta["BV"], count * heads

        state.chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
            k=k,
            v=u,
            w=values["w"],
            v_new=new_values,
            g=values["g"],
            gk=values["gk"],
            h=history,
            initial_state=initial,
            initial_state_indices=values["initial_state_indices"],
            cu_seqlens=sequences,
            chunk_offsets=offsets,
            T=t,
            H=heads,
            Hg=hg,
            K=key_dim,
            V=value_dim,
            BT=block,
            USE_G=values["g"] is not None,
            USE_GK=values["gk"] is not None,
            USE_INITIAL_STATE=True,
            INPLACE_UPDATE=True,
            SAVE_NEW_VALUE=new_values is not None,
            IS_VARLEN=sequences is not None,
            NT_BUCKET=0 if chunks <= 32 else (1 if chunks <= 128 else 2),
        )
        return history, new_values

    @wraps(output)
    def output_with_original_precision(*args: Any, **kwargs: Any) -> Any:
        bound = output_signature.bind(*args, **kwargs)
        bound.arguments["h"] = bound.arguments["h"].to(bound.arguments["q"].dtype)
        return output(*bound.args, **bound.kwargs)

    checkpoints._skillev_fp32_checkpoints = True
    chunk.chunk_gated_delta_rule_fwd_h = checkpoints
    chunk.chunk_fwd_o = output_with_original_precision
