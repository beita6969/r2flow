from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing.connection import Listener
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast

import torch

from skillev.contracts import JsonValue
from skillev.policy import PolicyBackbone, PolicyParameterGroups
from skillev.policy.fla_execution import fla_execution_metrics
from skillev.policy.versions import TrainableVersions
from skillev.rollout import PolicySnapshot, RolloutArtifact
from skillev.scoring.r2flow_plan import R2FlowEdgePlan

from .gradient_buckets import PackedGradients
from .gradient_stream_transport import receive_contribution, receive_message, send_message
from .gradient_worker_limits import accepts_plan, validate_sequence_limits
from .ordered_gradients import OrderedGradientAccumulator
from .planning import CollectedTrainingBatch, TrainingBatchPlan
from .scoring_telemetry import scoring_telemetry
from .step_math import (
    PreparedTTBStep,
    TTBArtifactMath,
    TTBGradientShard,
    compute_ttb_artifact_contribution,
    merge_ttb_gradient_shards,
    named_ttb_parameters,
    r2flow_edge_plan,
)
from .stream_worker import serve_stream_worker as serve_stream_worker

if TYPE_CHECKING:
    from collections.abc import Callable

    from .config import TTBMethodConfig
    from .distributed_ttb import DistributedTTBGradientCoordinator, DistributedTTBTopology


class GradientStepStream:
    def __init__(
        self,
        *,
        coordinator: DistributedTTBGradientCoordinator | None = None,
        backbone: PolicyBackbone,
        parameters: PolicyParameterGroups,
        optimizer: torch.optim.Optimizer,
        clock: Callable[[], str],
        method: TTBMethodConfig,
        max_buffer_bytes: int = 512 * 1024 * 1024,
        gradient_worker_weights: tuple[int, ...] = (),
        gradient_worker_max_sequence_tokens: tuple[int, ...] = (),
    ) -> None:
        self.method = method
        if type(max_buffer_bytes) is not int or max_buffer_bytes < 1:
            raise ValueError("gradient buffer byte limit must be positive")
        self.max_buffer_bytes = max_buffer_bytes
        from .gradient_work_log import GradientWorkLog

        self._work_log = GradientWorkLog()
        self._wait_reasons: dict[int, str] = {}
        self._wait_totals: dict[str, float] = {}
        self._waiting_since: dict[int, float] = {}
        self._stage, self._stage_started = "not-started", time.perf_counter()
        self._owners: dict[int, int] = {}
        self._versions = TrainableVersions.from_backbone(backbone)
        self.coordinator = coordinator
        self.backbone, self.parameters, self.optimizer = backbone, parameters, optimizer
        self.clock = clock
        self.plan: TrainingBatchPlan | CollectedTrainingBatch | None = None
        self.prepared: PreparedTTBStep | None = None
        self._condition = threading.Condition()
        self._artifacts: dict[int, RolloutArtifact] = {}
        self._edges: dict[int, R2FlowEdgePlan] = {}
        self._unassigned: set[int] = set()
        self._reserved: set[int] = set()
        self._inbox: dict[int, tuple[TTBArtifactMath, PackedGradients]] = {}
        self._cursor = 0
        self._count = 0
        self._world = 1 if coordinator is None else coordinator.topology.world_size
        if gradient_worker_weights and (
            len(gradient_worker_weights) != self._world
            or any(type(weight) is not int or weight < 1 for weight in gradient_worker_weights)
        ):
            raise ValueError("gradient worker capacities must match all training ranks")
        self.gradient_worker_weights = gradient_worker_weights
        self._participates = coordinator is None or coordinator.coordinator_participates
        validate_sequence_limits(gradient_worker_max_sequence_tokens)
        if gradient_worker_max_sequence_tokens and (
            len(gradient_worker_max_sequence_tokens) != self._world
            or 0 not in gradient_worker_max_sequence_tokens[0 if self._participates else 1 :]
        ):
            raise ValueError("active ranks must cover all sequences without dropping work")
        self.gradient_worker_max_sequence_tokens = gradient_worker_max_sequence_tokens
        self._assignments: list[list[int]] = [[] for _ in range(self._world)]
        self._progress: dict[int, dict[str, object]] = {}
        self._aborted = self._sealed = self._begun = False
        self._error: BaseException | None = None
        self._pool: ThreadPoolExecutor | None = None
        self._prep_pool: ThreadPoolExecutor | None = None
        self._futures: list[Future[object]] = []
        self._prep_futures: list[Future[R2FlowEdgePlan]] = []
        self._directory: TemporaryDirectory[str] | None = None
        self._listeners: list[Listener] = []
        self._shard: TTBGradientShard | None = None
        self._accumulator: OrderedGradientAccumulator | None = None
        self.gradient_started: float | None = None
        self.gradient_finished: float | None = None
        self.last_artifact_ready: float | None = None
        self.completed_before_last_rollout = 0
        self._completed = 0
        self._compute_seconds = self._edge_plan_seconds = 0.0
        self._queue_seconds = self._artifact_wait_seconds = self._buffer_wait_seconds = 0.0
        self._remote_wait_seconds = 0.0
        self._wait_totals_by_rank: dict[int, dict[str, float]] = {}
        self._edge_tokens = self._buffer_peak = 0
        self.rank_metrics: tuple[dict[str, JsonValue], ...] = ()
        self.capacity: dict[str, JsonValue] = {}

    async def begin(self, plan: TrainingBatchPlan, snapshot: PolicySnapshot) -> None:
        await self._begin(plan, snapshot)

    async def begin_collected(
        self, batch: CollectedTrainingBatch, snapshot: PolicySnapshot
    ) -> None:
        await self._begin(batch, snapshot)

    async def _begin(
        self, plan: TrainingBatchPlan | CollectedTrainingBatch, snapshot: PolicySnapshot
    ) -> None:
        if self.plan is not None or plan.policy_snapshot_id != snapshot.snapshot_id:
            raise ValueError("stream cannot change its fixed step identity")
        self.plan, self.snapshot, self.started_at = plan, snapshot, self.clock()
        if isinstance(plan, TrainingBatchPlan):
            self._trajectory_ids = [r.trajectory_id for r in plan.rollouts]
            self._task_ids = [r.task.task_id for r in plan.rollouts]
            self._decoding_ids = [r.decoding.snapshot_id for r in plan.rollouts]
            domains = [r.task.task_family for r in plan.rollouts]
        else:
            self._trajectory_ids = [a.record.trajectory_id for a in plan.artifacts]
            self._task_ids = [a.manifest.task_id for a in plan.artifacts]
            self._decoding_ids = [a.manifest.decoding_snapshot_id for a in plan.artifacts]
            domains = ["fixed-artifact-replay"] * len(plan.artifacts)
        self._count = len(self._trajectory_ids)
        self._unassigned = set(range(self._count))
        self._progress = {
            p: {"position": p, "task_domain": domain, "stage": "waiting-artifact"}
            for p, domain in enumerate(domains)
        }
        named = named_ttb_parameters(self.parameters)
        self._contribution_bytes = sum(p.numel() * p.element_size() for p in named.values())
        workers = self._world if self._participates else self._world - 1
        maximum = self.max_buffer_bytes // self._contribution_bytes
        profile = getattr(self.backbone, "performance_config", None)
        self.capacity = {
            "performance": None if profile is None else profile.to_value(),
            "gradient_world_size": self._world,
            "gradient_worker_count": workers,
            "coordinator_participates": self._participates,
            "contribution_bytes": self._contribution_bytes,
            "coordinator_buffer_bytes": self.max_buffer_bytes,
            "maximum_out_of_order_contributions": maximum,
            "worst_case_out_of_order_bytes": max(0, self._count - 1) * self._contribution_bytes,
            "worker_host_staging_bytes": self._contribution_bytes,
            "edge_plan_workers": min(workers, 4),
            "reduction": "global-canonical-trajectory-order",
            "gradient_worker_weights": list(self.gradient_worker_weights or (1,) * self._world),
            "gradient_worker_max_sequence_tokens": list(self.gradient_worker_max_sequence_tokens),
            "buffer_status": "full-batch-headroom"
            if maximum >= self._count - 1
            else "bounded-backpressure",
        }
        print(
            json.dumps(
                {"status": "gradient-step-ready", "batch_id": plan.batch_id, **self.capacity}
            ),
            flush=True,
        )
        if maximum < self._count - 1:
            logging.getLogger(__name__).warning(
                "Gradient buffer has %d of %d worst-case slots; backpressure remains enabled",
                maximum,
                self._count - 1,
            )
        self._prep_pool = ThreadPoolExecutor(
            max_workers=min(workers, 4), thread_name_prefix="ttb-edge-plan"
        )
        self._stage, self._stage_started = "collecting", time.perf_counter()
        self._pool = ThreadPoolExecutor(max_workers=self._world, thread_name_prefix="ttb-owner")
        self.optimizer.zero_grad(set_to_none=True)
        if self.coordinator is not None:
            root = self.coordinator.stream_directory
            if root is not None:
                root.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._directory = TemporaryDirectory(prefix="ttb-step-", dir=root)
            addresses = [str(Path(self._directory.name) / f"r{r}") for r in range(1, self._world)]
            self._listeners = [Listener(address, family="AF_UNIX") for address in addresses]
            await asyncio.to_thread(self._initialize, addresses)
        self._begun = True
        self._futures = [self._pool.submit(self._local)]
        self._futures.extend(
            self._pool.submit(self._remote, rank) for rank in range(1, self._world)
        )

    def _initialize(self, addresses: list[str]) -> None:
        from .distributed_ttb import _broadcast_object, _synchronize_parameters

        assert self.coordinator is not None
        assert self.plan is not None
        _set_device(self.coordinator.topology)
        _broadcast_object(
            {
                "kind": "stream",
                "addresses": addresses,
                "batch_id": self.plan.batch_id,
                "optimizer_step": self.plan.optimizer_step,
                "policy_snapshot_id": self.plan.policy_snapshot_id,
                "library_version": self.plan.library_version,
                "global_batch_size": self._count,
                "method": self.method.to_value(),
                "trainable_versions": TrainableVersions.from_backbone(self.backbone),
                "trajectory_ids": self._trajectory_ids,
                "task_ids": self._task_ids,
            },
            source=0,
        )
        _synchronize_parameters(named_ttb_parameters(self.parameters), source=0)

    async def accept(self, position: int, artifact: RolloutArtifact) -> None:
        plan = self.plan
        if plan is None or not 0 <= position < self._count:
            raise ValueError("stream position is outside its plan")
        if (
            artifact.record.trajectory_id != self._trajectory_ids[position]
            or artifact.manifest.task_id != self._task_ids[position]
            or artifact.manifest.policy_snapshot.snapshot_id != plan.policy_snapshot_id
            or artifact.manifest.library_version != plan.library_version
            or artifact.manifest.decoding_snapshot_id != self._decoding_ids[position]
        ):
            raise ValueError("stream artifact differs from its planned identity")
        with self._condition:
            if self._aborted:
                cause = self._error
                if cause is not None:
                    raise RuntimeError("gradient stream aborted") from cause
                raise RuntimeError("gradient stream is aborted")
            if self._sealed or position in self._artifacts:
                raise RuntimeError("stream is closed or position was already supplied")
            self._artifacts[position] = artifact
            self.last_artifact_ready = time.perf_counter()
            self.completed_before_last_rollout = self._completed
            self._progress[position].update(
                stage="prepare-edges",
                artifact_ready=self.last_artifact_ready,
                horizon=artifact.record.horizon,
            )
            assert self._prep_pool is not None
            future = self._prep_pool.submit(self._prepare_edges, position, artifact)
            self._prep_futures.append(future)
            self._condition.notify_all()
        for owner in self._futures:
            if owner.done() and owner.exception() is not None:
                raise RuntimeError("gradient owner failed") from owner.exception()

    def _prepare_edges(self, position: int, artifact: RolloutArtifact) -> R2FlowEdgePlan:
        started = time.perf_counter()
        complete = False
        try:
            edges = r2flow_edge_plan(self.backbone, artifact, self.method)
            with self._condition:
                self._edge_plan_seconds += time.perf_counter() - started
                self._edges[position] = edges
                self._progress[position].update(
                    stage="ready", edge_plan_ready=time.perf_counter(), edge_tokens=edges.token_cost
                )
                self._condition.notify_all()
            complete = True
            return edges
        finally:
            if not complete:
                self._signal_abort(sys.exception())

    def _claim(self, rank: int) -> tuple[int, RolloutArtifact, R2FlowEdgePlan] | None:
        all_ready = self._unassigned & self._edges.keys()
        limit = (
            self.gradient_worker_max_sequence_tokens[rank]
            if self.gradient_worker_max_sequence_tokens
            else 0
        )
        ready = {p for p in all_ready if accepts_plan(limit, self._edges.get(p))}
        reserved = len(self._reserved - {self._cursor}) * self._contribution_bytes
        allowed = [
            p
            for p in ready
            if p == self._cursor
            or p in self._reserved
            or reserved + self._contribution_bytes <= self.max_buffer_bytes
        ]
        if not allowed:
            if ready:
                reason = "buffer-capacity"
            elif all_ready:
                reason = "worker-sequence-limit"
            elif any(p not in self._edges for p in self._artifacts if p in self._unassigned):
                reason = "cpu-preparation"
            elif self._unassigned:
                reason = "next-edge-or-artifact"
            else:
                reason = "remote-contribution"
            self._wait_reasons[rank] = reason
            return None
        position = (
            self._cursor
            if self._cursor in allowed
            else min(allowed, key=lambda p: (self._edges[p].token_cost, p))
            if limit
            else max(allowed, key=lambda p: (self._edges[p].token_cost, -p))
        )
        self._reserved.add(position)
        self._owners[position] = rank
        self._assignments[rank].append(position)
        self._progress[position].update(worker_rank=rank, gradient_start=time.perf_counter())
        self._progress[position]["stage"] = "computing"
        if self.gradient_started is None:
            self.gradient_started = time.perf_counter()
        self._unassigned.remove(position)
        return position, self._artifacts[position], self._edges[position]

    def _wait(self, *, local_owner: bool, rank: int = 0) -> None:
        reason = self._wait_reasons.get(rank, "remote-contribution")
        started = self._waiting_since[rank] = time.perf_counter()
        self._condition.wait()
        self._waiting_since.pop(rank, None)
        duration = time.perf_counter() - started
        totals = self._wait_totals_by_rank.setdefault(rank, {})
        totals[reason] = totals.get(reason, 0.0) + duration
        if local_owner:
            self._queue_seconds += duration
            self._wait_totals[reason] = self._wait_totals.get(reason, 0.0) + duration
            if reason == "buffer-capacity":
                self._buffer_wait_seconds += duration
            elif reason == "next-edge-or-artifact":
                self._artifact_wait_seconds += duration
            else:
                self._remote_wait_seconds += duration

    def _update_progress(self, position: int, value: dict[str, object]) -> None:
        with self._condition:
            self._work_log.observe(position, value, time.perf_counter())
            self._progress[position].update(value)
            self._progress[position]["updated"] = time.perf_counter()

    def progress_snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "batch_id": None if self.plan is None else self.plan.batch_id,
                "capacity": dict(self.capacity),
                "edge_work": self._work_log.snapshot(),
                "timing_basis": (
                    "host-scheduling-and-backward-return; CUDA time unknown unless profiled"
                ),
                "stage": self._stage,
                "stage_age_seconds": time.perf_counter() - self._stage_started,
                "wait_seconds_by_reason": dict(self._wait_totals),
                "dispatch_wait_seconds_by_rank": {
                    str(rank): dict(totals) for rank, totals in self._wait_totals_by_rank.items()
                },
                "active_waits": [
                    {
                        "rank": r,
                        "reason": self._wait_reasons.get(r),
                        "age_seconds": time.perf_counter() - t,
                    }
                    for r, t in self._waiting_since.items()
                ],
                "artifacts_ready": len(self._artifacts),
                "edge_plans_ready": len(self._edges),
                "gradient_contributions_completed": self._completed,
                "canonical_contributions_merged": self._cursor,
                "aborted": self._aborted,
                "trajectories": [dict(self._progress[p]) for p in range(self._count)],
            }

    def _record_complete(self, position: int) -> None:
        with self._condition:
            self._edges.pop(position, None)
            self._completed += 1
            self._condition.notify_all()

    def _drain(self, accumulator: OrderedGradientAccumulator) -> None:
        while True:
            with self._condition:
                if not self._inbox:
                    break
                position = next(iter(self._inbox))
                item, packed = self._inbox.pop(position)
            accumulator.add_packed(item, packed)
        with self._condition:
            self._cursor = accumulator.cursor
            self._reserved.difference_update(accumulator.positions[: self._cursor])
            self._buffer_peak = max(self._buffer_peak, accumulator.peak_buffer_bytes)
            self._condition.notify_all()

    def _local(self) -> object:
        named = named_ttb_parameters(self.parameters)
        device = next(iter(named.values())).device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        assert self.plan is not None
        accumulator = self._accumulator = OrderedGradientAccumulator(
            tuple(range(self._count)), named, self.max_buffer_bytes
        )
        try:
            while True:
                self._drain(accumulator)
                with self._condition:
                    if self._aborted:
                        return None
                    if self._completed == self._count:
                        break
                    work = self._claim(0) if self._participates else None
                    if work is None:
                        if not self._inbox:
                            self._wait(local_owner=True)
                        continue
                position, artifact, edges = work
                started = time.perf_counter()
                self._edge_tokens += edges.token_cost

                def progress(value: dict[str, object], position: int = position) -> None:
                    self._update_progress(position, value)
                    self._drain(accumulator)

                contribution = compute_ttb_artifact_contribution(
                    backbone=self.backbone,
                    parameters=self.parameters,
                    artifact=artifact,
                    position=position,
                    batch_id=self.plan.batch_id,
                    optimizer_step=self.plan.optimizer_step,
                    policy_snapshot_id=self.plan.policy_snapshot_id,
                    library_version=self.plan.library_version,
                    global_batch_size=self._count,
                    prepared_edges=edges,
                    progress=progress,
                    method=self.method,
                )
                self._compute_seconds += time.perf_counter() - started
                self._update_progress(
                    position, {"stage": "buffer-or-merge", "gradient_finish": time.perf_counter()}
                )
                accumulator.add(contribution)
                del contribution
                self._record_complete(position)
            self._drain(accumulator)
            if accumulator.cursor != self._count:
                raise RuntimeError(
                    "complete worker results did not produce complete canonical coverage"
                )
            self._shard = TTBGradientShard(
                self.plan.batch_id,
                self.plan.optimizer_step,
                self._count,
                tuple(accumulator.items),
                accumulator.gradients,
            )
        except BaseException as error:
            self._signal_abort(error)
            raise
        finally:
            if self._shard is None and not self._aborted:
                accumulator.clear()
                self._signal_abort(sys.exception())
        return None

    def _remote(self, rank: int) -> object:
        assert self.plan is not None
        connection = self._listeners[rank - 1].accept()
        terminal_sent = False
        try:
            while True:
                with self._condition:
                    while not self._aborted and self._unassigned:
                        work = self._claim(rank)
                        if work is not None:
                            break
                        self._wait(local_owner=False, rank=rank)
                    else:
                        work = None
                if work is None:
                    break
                position, artifact, edges = work
                send_message(
                    connection,
                    {
                        "kind": "artifact",
                        "position": position,
                        "artifact": artifact.to_value(),
                        "r2flow_plan": edges.wire_steps(),
                    },
                )
                while True:
                    reply = receive_message(connection)
                    if reply["kind"] == "progress":
                        self._update_progress(position, cast(dict[str, object], reply["progress"]))
                        continue
                    if reply["kind"] != "contribution":
                        self._signal_abort(
                            RuntimeError(str(reply.get("error_class", "worker failed")))
                        )
                        break
                    assert self.plan is not None
                    if (
                        reply["position"] != position
                        or reply["trajectory_id"] != self._trajectory_ids[position]
                        or reply["batch_id"] != self.plan.batch_id
                        or reply["optimizer_step"] != self.plan.optimizer_step
                        or reply["global_batch_size"] != self._count
                    ):
                        raise ValueError(
                            "worker contribution differs from its assigned complete trajectory"
                        )
                    item, packed = receive_contribution(
                        connection, reply, parameters=named_ttb_parameters(self.parameters)
                    )
                    with self._condition:
                        self._inbox[position] = item, packed
                        pending = sum(p.nbytes for _, p in self._inbox.values())
                        if self._accumulator is not None:
                            pending += self._accumulator.buffer_bytes
                        self._buffer_peak = max(self._buffer_peak, pending)
                        self._progress[position].update(
                            stage="received", gradient_finish=time.perf_counter()
                        )
                        self._record_complete(position)
                    break
            with self._condition:
                self._condition.wait_for(lambda: self._aborted or self._sealed)
                valid = not self._aborted
            send_message(connection, {"kind": "seal" if valid else "abort"})
            terminal_sent = True
        finally:
            if not terminal_sent:
                self._signal_abort()
                try:
                    send_message(connection, {"kind": "abort"})
                except (OSError, EOFError):
                    pass
            connection.close()
        return None

    def _signal_abort(self, error: BaseException | None = None) -> None:
        with self._condition:
            self._aborted = True
            self._error = self._error or error
            self._condition.notify_all()

    async def seal(self, batch: CollectedTrainingBatch) -> PreparedTTBStep:
        plan = self.plan
        if plan is None or self.prepared is not None:
            raise RuntimeError("stream is not active")
        if (
            batch.batch_id != plan.batch_id
            or batch.policy_snapshot_id != plan.policy_snapshot_id
            or batch.library_version != plan.library_version
            or batch.optimizer_step != plan.optimizer_step
            or len(batch.artifacts) != self._count
            or any(self._artifacts.get(i) is not item for i, item in enumerate(batch.artifacts))
        ):
            raise ValueError("sealed batch differs from the streamed population")
        with self._condition:
            self._sealed = True
            self._condition.notify_all()
        try:
            self.prepared = await self._settle_owners(batch)
        finally:
            if self.prepared is None:
                self.optimizer.zero_grad(set_to_none=True)
                self._shard = None
        assert self.prepared is not None
        return self.prepared

    async def discard_uncommitted(self) -> None:
        if self.prepared is not None:
            return
        self._signal_abort()
        if self._begun:
            await self._settle_owners(None)
        else:
            self._cleanup()

    async def _settle_owners(self, batch: CollectedTrainingBatch | None) -> PreparedTTBStep | None:
        task = asyncio.create_task(asyncio.to_thread(self._finish, batch))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            self._signal_abort()
            drained = asyncio.gather(task, return_exceptions=True)
            while not drained.done():
                try:
                    await asyncio.shield(drained)
                except asyncio.CancelledError:
                    continue
            raise

    def _finish(self, batch: CollectedTrainingBatch | None) -> PreparedTTBStep | None:
        from .distributed_ttb import (
            DistributedTTBError,
            DistributedTTBSingleArtifactOOMError,
            _all_gather_object,
        )

        if self.coordinator is not None:
            _set_device(self.coordinator.topology)
        try:
            self._stage, self._stage_started = "seal-draining-owners", time.perf_counter()
            for future in self._futures:
                self._error = future.exception() or self._error
            for preparation in self._prep_futures:
                self._error = preparation.exception() or self._error
            valid = (
                batch is not None
                and not self._aborted
                and self._error is None
                and self._shard is not None
            )
            accumulator = self._accumulator
            status = {
                "rank": 0,
                "status": "ok" if valid else "abort",
                "error_class": None if self._error is None else type(self._error).__name__,
                "positions": self._assignments[0],
                "compute_seconds": self._compute_seconds,
                "compute_timing_kind": "host-wall-including-enqueue-and-remote-drain",
                "edge_plan_cpu_seconds": self._edge_plan_seconds,
                "edge_tokens": self._edge_tokens,
                "queue_wait_seconds": self._queue_seconds,
                "wait_seconds_by_reason": self._wait_totals,
                "artifact_wait_seconds": self._artifact_wait_seconds,
                "buffer_backpressure_seconds": self._buffer_wait_seconds,
                "remote_compute_wait_seconds": self._remote_wait_seconds,
                "gradient_buffer_peak_bytes": self._buffer_peak,
                "gradient_pack_host_seconds": 0.0
                if accumulator is None
                else accumulator.host_pack_seconds,
                "gradient_restore_host_seconds": 0.0
                if accumulator is None
                else accumulator.host_restore_seconds,
                "execution_capacity": self.capacity,
                "scoring": scoring_telemetry(
                    ()
                    if self._shard is None
                    else (a for a in self._shard.artifacts if a.position in self._assignments[0])
                ),
                "trajectories": self.progress_snapshot()["trajectories"],
                "z_cache": getattr(self.backbone, "z_feature_cache_metrics", {}),
                "fla_process_cumulative": fla_execution_metrics(),
            }
            self._stage, self._stage_started = "seal-collective", time.perf_counter()
            statuses = (
                (status,) if self.coordinator is None else _all_gather_object(status, self._world)
            )
            for row in statuses:
                if isinstance(row, dict):
                    row["dispatch_wait_seconds_by_reason"] = dict(
                        self._wait_totals_by_rank.get(row["rank"], {})
                    )
            self.rank_metrics = tuple(cast(dict[str, JsonValue], row) for row in statuses)
            self._begun = False
            if any(not isinstance(row, dict) or row.get("status") != "ok" for row in statuses):
                self.optimizer.zero_grad(set_to_none=True)
                self._shard = None
                if batch is None:
                    return None
                if any(
                    isinstance(row, dict)
                    and row.get("error_class")
                    in {"OutOfMemoryError", "TrajectoryScoringMemoryError"}
                    for row in statuses
                ):
                    raise DistributedTTBSingleArtifactOOMError(
                        "one complete trajectory exhausted memory", diagnostics=statuses
                    )
                raise DistributedTTBError(
                    "complete gradient batch aborted", diagnostics=statuses
                ) from self._error
            assert batch is not None
            assert self._shard is not None
            self._stage, self._stage_started = "seal-final-merge", time.perf_counter()
            result = merge_ttb_gradient_shards(
                parameters=self.parameters,
                batch=batch,
                snapshot_before=self.snapshot,
                shards=(self._shard,),
                clock=self.clock,
                started_at=self.started_at,
            )
            self.gradient_finished = time.perf_counter()
            self._stage, self._stage_started = "gradients-prepared", time.perf_counter()
            return result
        finally:
            self._begun = False
            self._cleanup()

    def _cleanup(self) -> None:
        if self._prep_pool is not None:
            self._prep_pool.shutdown(wait=True)
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        for listener in self._listeners:
            listener.close()
        if self._directory is not None:
            self._directory.cleanup()
        self._inbox.clear()


class LocalGradientStepStream(GradientStepStream):
    pass


def _set_device(topology: DistributedTTBTopology) -> None:
    if topology.backend == "nccl":
        torch.cuda.set_device(topology.local_rank)
