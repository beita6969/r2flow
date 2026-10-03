from __future__ import annotations

import json
from collections.abc import Mapping
from multiprocessing.connection import Connection
from typing import Any, cast

import torch

from skillev.contracts import JsonValue
from skillev.contracts.r2flow_training import R2FlowEdgeRecord, R2FlowTrajectoryResidual

from .gradient_buckets import PackedGradients, pack_gradients, tensor_buckets
from .step_math import TTBArtifactMath, TTBGradientShard

_MAX_MESSAGE_BYTES = 128 * 1024 * 1024


def send_message(connection: Connection, value: dict[str, object]) -> None:
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    if len(payload) > _MAX_MESSAGE_BYTES:
        raise ValueError("gradient control message exceeds its private transport bound")
    connection.send_bytes(payload)


def receive_message(connection: Connection, *, timeout: float = 10800.0) -> dict[str, object]:
    if not connection.poll(timeout):
        raise TimeoutError("gradient control peer did not respond")
    value = json.loads(connection.recv_bytes(_MAX_MESSAGE_BYTES))
    if not isinstance(value, dict) or not isinstance(value.get("kind"), str):
        raise ValueError("invalid gradient control message")
    return value


def send_contribution(connection: Connection, contribution: TTBGradientShard) -> int:
    (item,) = contribution.artifacts
    packed = pack_gradients(contribution.gradients)
    send_message(
        connection,
        {
            "kind": "contribution",
            "batch_id": contribution.batch_id,
            "optimizer_step": contribution.optimizer_step,
            "global_batch_size": contribution.global_batch_size,
            "position": item.position,
            "trajectory_id": item.trajectory_id,
            "residual": item.residual.to_value(),
            "edges": [e.to_value() for e in item.edges],
            "loss": item.loss,
            "scoring_metrics": list(item.scoring_metrics),
            "buckets": [list(names) for names in packed.names],
        },
    )
    for flat in packed.buffers:
        connection.send_bytes(memoryview(cast(Any, flat.view(torch.uint8).numpy())))
    return packed.nbytes


def receive_contribution(
    connection: Connection, header: dict[str, object], *, parameters: Mapping[str, torch.Tensor]
) -> tuple[TTBArtifactMath, PackedGradients]:
    names = tuple(tensor_buckets(parameters))
    if header.get("buckets") != [list(bucket) for bucket in names]:
        raise ValueError("worker contribution parameter layout differs")
    buffers = []
    for bucket in names:
        flat = torch.empty(
            sum(parameters[n].numel() for n in bucket),
            dtype=parameters[bucket[0]].dtype,
            device="cpu",
        )
        if not connection.poll(10800.0):
            raise TimeoutError("worker contribution transfer did not finish")
        count = connection.recv_bytes_into(memoryview(cast(Any, flat.view(torch.uint8).numpy())))
        if count != flat.numel() * flat.element_size():
            raise ValueError("worker contribution bucket has the wrong size")
        buffers.append(flat)
    residual_value = header["residual"]
    edge_values = cast(list[object], header["edges"])
    residual = R2FlowTrajectoryResidual.from_value(residual_value)
    edges = tuple(R2FlowEdgeRecord.from_value(v) for v in edge_values)
    item = TTBArtifactMath(
        cast(int, header["position"]),
        cast(str, header["trajectory_id"]),
        residual,
        edges,
        cast(float, header["loss"]),
        tuple(cast(list[dict[str, JsonValue]], header["scoring_metrics"])),
    )
    return item, PackedGradients(names, tuple(buffers))
