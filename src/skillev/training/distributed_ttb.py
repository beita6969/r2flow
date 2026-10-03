from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from pathlib import Path

import torch
import torch.distributed as dist

from skillev.policy import PolicyBackbone, PolicyParameterGroups
from skillev.rollout import PolicySnapshot
from skillev.rollout.artifact import RolloutArtifact

from .config import TTBMethodConfig
from .gradient_buckets import flatten_bucket, tensor_buckets, unpack_bucket
from .gradient_worker import GradientWorkItem, partition_items_by_token_cost
from .planning import CollectedTrainingBatch
from .step_math import (
    PreparedTTBStep,
)
from .vq_scoring import (
    FlowTrajectoryScore,
    FlowTrajectoryScorer,
    score_flow_no_grad,
    token_cost_partition,
)


class DistributedTTBError(RuntimeError):
    def __init__(self, message: str, *, diagnostics: tuple[object, ...] = ()) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


class DistributedTTBWorkerOOMError(DistributedTTBError):
    pass


class DistributedTTBOOMKind(StrEnum):
    PARTITIONABLE_WORKLOAD = "partitionable-workload"
    SINGLE_ARTIFACT = "single-artifact"


class DistributedTTBPartitionableOOMError(DistributedTTBWorkerOOMError):
    pass


class DistributedTTBSingleArtifactOOMError(DistributedTTBWorkerOOMError):
    pass


@dataclass(frozen=True, slots=True)
class DistributedTTBTopology:
    rank: int
    world_size: int
    local_rank: int
    backend: str

    def __post_init__(self) -> None:
        if self.world_size < 2:
            raise ValueError("distributed TTB requires a coordinator and a worker")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("distributed TTB rank lies outside the world")
        if self.local_rank < 0 or not self.backend.strip():
            raise ValueError("distributed TTB local rank or backend is invalid")


def initialize_distributed_ttb(*, timeout_minutes: int = 30) -> DistributedTTBTopology:
    if type(timeout_minutes) is not int or timeout_minutes <= 0:
        raise ValueError("distributed timeout must be a positive integer")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not torch.cuda.is_available():
        raise RuntimeError("formal distributed TTB requires CUDA")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=timeout_minutes),
        device_id=torch.device("cuda", local_rank),
    )
    return DistributedTTBTopology(rank, world_size, local_rank, "nccl")


def training_artifact_token_cost(batch: CollectedTrainingBatch, position: int) -> int:
    artifact = batch.artifacts[position]
    context_tokens = artifact.initial_context.contract.assembled_token_count
    action_tokens = sum(step.action_token_count for step in artifact.record.steps)
    observation_tokens = sum(
        max(1, (len(step.observation_text) + 3) // 4) for step in artifact.record.steps
    )
    return max(1, context_tokens + action_tokens + observation_tokens)


def partition_training_batch(
    batch: CollectedTrainingBatch,
    *,
    worker_count: int,
) -> tuple[tuple[int, ...], ...]:
    work = tuple(
        GradientWorkItem(
            item_id=str(position),
            token_cost=training_artifact_token_cost(batch, position),
            payload_ref=f"private-batch-position://{position}",
        )
        for position in range(len(batch.artifacts))
    )
    return tuple(
        tuple(int(item.item_id) for item in partition)
        for partition in partition_items_by_token_cost(work, worker_count)
    )


@dataclass(slots=True)
class DistributedTTBGradientCoordinator:
    topology: DistributedTTBTopology
    coordinator_participates: bool = False
    pipeline_mode: str = "sealed-batch"
    stream_directory: Path | None = None

    def __post_init__(self) -> None:
        if self.topology.rank != 0:
            raise ValueError("distributed TTB coordinator must be rank zero")
        if type(self.coordinator_participates) is not bool:
            raise TypeError("coordinator participation must be boolean")

    def prepare(
        self,
        *,
        backbone: PolicyBackbone,
        optimizer: torch.optim.Optimizer,
        parameters: PolicyParameterGroups,
        batch: CollectedTrainingBatch,
        snapshot_before: PolicySnapshot,
        clock: Callable[[], str],
        method: TTBMethodConfig,
    ) -> PreparedTTBStep:
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        from .streaming_step import GradientStepStream

        performance = getattr(backbone, "performance_config", None)
        stream = GradientStepStream(
            coordinator=self,
            backbone=backbone,
            parameters=parameters,
            optimizer=optimizer,
            method=method,
            clock=clock,
            max_buffer_bytes=512 * 1024 * 1024
            if performance is None
            else performance.gradient_buffer_bytes,
            gradient_worker_weights=()
            if performance is None
            else performance.gradient_worker_weights,
            gradient_worker_max_sequence_tokens=()
            if performance is None
            else performance.gradient_worker_max_sequence_tokens,
        )

        async def run() -> PreparedTTBStep:
            await stream.begin_collected(batch, snapshot_before)
            try:
                for position, artifact in enumerate(batch.artifacts):
                    await stream.accept(position, artifact)
                return await stream.seal(batch)
            finally:
                if stream.prepared is None:
                    await stream.discard_uncommitted()

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="ttb-sealed") as pool:
            return pool.submit(lambda: asyncio.run(run())).result()

    def score_flow(
        self,
        *,
        backbone: PolicyBackbone,
        artifacts: tuple[RolloutArtifact, ...],
        method: TTBMethodConfig,
        scorer: FlowTrajectoryScorer | None = None,
    ) -> tuple[FlowTrajectoryScore, ...]:
        if not artifacts:
            return ()
        costs = [_artifact_score_cost(artifact) for artifact in artifacts]
        shares = token_cost_partition(costs, self.topology.world_size)
        command: dict[str, object] = {
            "kind": "score",
            "shares": [list(share) for share in shares],
            "artifacts": [artifact.to_value() for artifact in artifacts],
            "method": method.to_value(),
        }
        _broadcast_object(command, source=0)
        local = _score_share(command, rank=0, backbone=backbone, scorer=scorer)
        gathered = _all_gather_object(local, self.topology.world_size)
        return _merge_scores(gathered, len(artifacts))

    def close(self) -> None:
        _broadcast_object({"kind": "stop"}, source=0)


def serve_distributed_ttb_worker(
    *,
    topology: DistributedTTBTopology,
    backbone: PolicyBackbone,
) -> None:
    if topology.rank == 0:
        raise ValueError("rank zero cannot enter the gradient worker loop")
    while True:
        command = _broadcast_object(None, source=0)
        if not isinstance(command, dict) or not isinstance(command.get("kind"), str):
            raise DistributedTTBError("distributed TTB command is invalid")
        if command["kind"] == "stop":
            return
        if command["kind"] == "stream":
            from .streaming_step import serve_stream_worker

            serve_stream_worker(command, topology=topology, backbone=backbone)
            continue
        if command["kind"] == "score":
            local = _score_share(command, rank=topology.rank, backbone=backbone, scorer=None)
            _all_gather_object(local, topology.world_size)
            continue
        raise DistributedTTBError("distributed TTB command kind is unsupported")


def classify_distributed_ttb_oom(
    failures: tuple[object, ...],
) -> DistributedTTBOOMKind:
    oom_rows = tuple(
        row for row in failures if isinstance(row, dict) and row.get("status") == "oom"
    )
    if not oom_rows:
        raise ValueError("distributed OOM classification requires an OOM diagnostic")
    counts: list[int] = []
    for row in oom_rows:
        count = row.get("artifact_count")
        positions = row.get("artifact_positions")
        token_counts = row.get("artifact_token_counts")
        if (
            type(count) is not int
            or count < 1
            or not isinstance(positions, list)
            or not isinstance(token_counts, list)
            or len(positions) != count
            or len(token_counts) != count
            or any(type(position) is not int or position < 0 for position in positions)
            or any(type(tokens) is not int or tokens < 1 for tokens in token_counts)
        ):
            raise DistributedTTBError("distributed OOM diagnostic is incomplete")
        counts.append(count)
    if any(count == 1 for count in counts):
        return DistributedTTBOOMKind.SINGLE_ARTIFACT
    return DistributedTTBOOMKind.PARTITIONABLE_WORKLOAD


def _artifact_score_cost(artifact: RolloutArtifact) -> int:
    context = artifact.initial_context.contract.assembled_token_count
    return max(1, context + sum(step.action_token_count for step in artifact.record.steps))


def _score_share(
    command: dict[str, object],
    *,
    rank: int,
    backbone: PolicyBackbone,
    scorer: FlowTrajectoryScorer | None,
) -> dict[str, object]:
    shares = command["shares"]
    values = command["artifacts"]
    assert isinstance(shares, list)
    assert isinstance(values, list)
    positions = [int(p) for p in shares[rank]]
    try:
        method = TTBMethodConfig.from_value(command["method"])
        artifacts = tuple(
            RolloutArtifact.from_value(values[p], tokenizer=backbone.tokenizer) for p in positions
        )
        scores = score_flow_no_grad(backbone, artifacts, method, scorer=scorer)
    except Exception as error:
        return {"status": "failed", "rank": rank, "error_type": type(error).__name__}
    return {
        "status": "ok",
        "rank": rank,
        "scores": [[p, s.to_value()] for p, s in zip(positions, scores, strict=True)],
    }


def _merge_scores(gathered: tuple[object, ...], count: int) -> tuple[FlowTrajectoryScore, ...]:
    failures = tuple(
        row for row in gathered if not isinstance(row, dict) or row.get("status") != "ok"
    )
    if failures:
        raise DistributedTTBError("held-out flow scoring failed", diagnostics=failures)
    merged: dict[int, FlowTrajectoryScore] = {}
    for row in gathered:
        assert isinstance(row, dict)
        for position, value in row["scores"]:
            if position in merged:
                raise DistributedTTBError("held-out flow scoring duplicated a position")
            merged[int(position)] = FlowTrajectoryScore.from_value(value)
    if sorted(merged) != list(range(count)):
        raise DistributedTTBError("held-out flow scoring lost a position")
    return tuple(merged[p] for p in range(count))


def _broadcast_object(value: object, *, source: int) -> object:
    values = [value]
    dist.broadcast_object_list(
        values,
        src=source,
        device=_collective_device(),
    )
    return values[0]


def _all_gather_object(value: object, world_size: int) -> tuple[object, ...]:
    values: list[object] = [None] * world_size
    dist.all_gather_object(
        values,
        value,
    )
    return tuple(values)


def _synchronize_parameters(
    parameters: Mapping[str, torch.nn.Parameter],
    *,
    source: int,
) -> None:
    for names in tensor_buckets(parameters):
        flat = flatten_bucket(names, parameters)
        dist.broadcast(flat, src=source)
        with torch.no_grad():
            for name, value in unpack_bucket(names, flat, parameters).items():
                parameters[name].copy_(value)


def _collective_device() -> torch.device:
    backend = dist.get_backend()
    if backend == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


__all__ = [
    "DistributedTTBError",
    "DistributedTTBGradientCoordinator",
    "DistributedTTBOOMKind",
    "DistributedTTBPartitionableOOMError",
    "DistributedTTBSingleArtifactOOMError",
    "DistributedTTBTopology",
    "DistributedTTBWorkerOOMError",
    "classify_distributed_ttb_oom",
    "initialize_distributed_ttb",
    "partition_training_batch",
    "serve_distributed_ttb_worker",
    "training_artifact_token_cost",
]
