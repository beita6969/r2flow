from __future__ import annotations

import math
from collections.abc import Iterable
from typing import cast

from skillev.contracts import JsonValue

from .step_math import TTBArtifactMath


def scoring_telemetry(artifacts: Iterable[TTBArtifactMath]) -> dict[str, JsonValue]:
    rows = [row for artifact in artifacts for row in artifact.scoring_metrics]
    result: dict[str, JsonValue] = {}
    for role in sorted({str(row["role"]) for row in rows}):
        selected = [row for row in rows if row["role"] == role]
        lengths = sorted(length for row in selected for length in cast(list[int], row["lengths"]))
        result[role] = {
            "groups": len(selected),
            "edges": len(lengths),
            "prefix_tokens": sum(cast(int, row["prefix_tokens"]) for row in selected),
            "action_tokens": sum(cast(int, row["action_tokens"]) for row in selected),
            "padded_tokens": sum(cast(int, row["padded_tokens"]) for row in selected),
            "length_p50": lengths[(len(lengths) - 1) // 2],
            "length_p95": lengths[math.ceil(len(lengths) * 0.95) - 1],
            "length_max": max(lengths),
            "checkpoint_groups": sum(bool(row["checkpointed"]) for row in selected),
            "offload_groups": sum(bool(row["offloaded"]) for row in selected),
            "offload_saved_bytes": sum(cast(int, row["offload_saved_bytes"]) for row in selected),
            **{
                field: sum(cast(int, row.get(field, 0)) for row in selected)
                for field in ("offload_restored_bytes", "offload_resident_parameter_bytes")
            },
            **{
                field: math.fsum(cast(float, row.get(field, 0.0)) for row in selected)
                for field in ("offload_pack_host_seconds", "offload_unpack_host_seconds")
            },
            "forward_host_seconds": math.fsum(
                cast(float, row["forward_host_seconds"]) for row in selected
            ),
            "cuda_forward_backward_ms": (
                math.fsum(cast(float, row["cuda_forward_backward_ms"]) for row in selected)
                if all("cuda_forward_backward_ms" in row for row in selected)
                else None
            ),
        }
    return result
