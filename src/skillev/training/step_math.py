from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import torch

from skillev.contracts import JsonValue, TrainingStepReportValue
from skillev.contracts.r2flow_training import (
    R2FlowBatchStats,
    R2FlowEdgeRecord,
    R2FlowTrajectoryResidual,
)
from skillev.contracts.subtb import SUBTB_RESIDUAL_ID
from skillev.policy import AdapterRole, PolicyBackbone, PolicyParameterGroups
from skillev.rollout import PolicySnapshot, RolloutArtifact
from skillev.scoring.r2flow_plan import R2FlowEdgePlan

from .config import OptimizerConfig, TTBMethodConfig
from .optimizer_observation import OptimizerTransitionObservation
from .planning import CollectedTrainingBatch

if TYPE_CHECKING:
    from skillev.scoring.r2flow_objective import R2FlowScore

    from .edge_gradient_bank import EdgeGradientBank

TrajectoryRecordLike = Any


@dataclass(frozen=True, slots=True)
class ComponentGradientNorms:
    forward: float
    backward: float
    z_head: float
    psi_head: float | None = None


@dataclass(frozen=True, slots=True)
class PreparedTTBStep:
    batch: CollectedTrainingBatch
    snapshot_before: PolicySnapshot
    residuals: tuple[R2FlowTrajectoryResidual, ...]
    edges: tuple[R2FlowEdgeRecord, ...]
    stats: R2FlowBatchStats
    detached_losses: tuple[float, ...]
    gradient_norms: ComponentGradientNorms
    started_at: str


class TTBStepPreparer(Protocol):
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
    ) -> PreparedTTBStep: ...


@dataclass(frozen=True, slots=True)
class TTBArtifactMath:
    position: int
    trajectory_id: str
    residual: R2FlowTrajectoryResidual
    edges: tuple[R2FlowEdgeRecord, ...]
    loss: float
    scoring_metrics: tuple[dict[str, JsonValue], ...] = field(default=(), compare=False)


@dataclass(frozen=True, slots=True)
class TTBGradientShard:
    batch_id: str
    optimizer_step: int
    global_batch_size: int
    artifacts: tuple[TTBArtifactMath, ...]
    gradients: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        if not self.artifacts or not self.gradients:
            raise ValueError("TTB gradient shard cannot be empty")
        positions = tuple(item.position for item in self.artifacts)
        if len(positions) != len(set(positions)):
            raise ValueError("TTB gradient shard repeats a trajectory position")
        require_finite_gradients(self.gradients)


def create_ttb_optimizer(
    backbone: PolicyBackbone,
    config: OptimizerConfig,
) -> tuple[torch.optim.Optimizer, PolicyParameterGroups]:
    groups = backbone.parameter_groups()
    if not groups.psi_head:
        raise ValueError("the R2 Flow optimizer requires a backbone with an F_psi head")
    offset = getattr(backbone, "flow_offset_parameter", None)
    psi_group: list[dict[str, object]] = [
        {
            "name": "psi-head",
            "params": tuple(p for p in groups.psi_head if p is not offset),
            "lr": config.psi_learning_rate,
        }
    ]
    if offset is not None:
        from skillev.policy.flow_head import FLOW_OFFSET_LEARNING_RATE

        if not any(p is offset for p in groups.psi_head):
            raise ValueError("flow offsets must be part of the psi parameter group")
        psi_group.append(
            {"name": "flow-offset", "params": (offset,), "lr": FLOW_OFFSET_LEARNING_RATE}
        )
    optimizer = torch.optim.AdamW(
        [
            {
                "name": AdapterRole.FORWARD_POLICY.value,
                "params": groups.forward,
                "lr": config.adapter_learning_rate,
            },
            {
                "name": AdapterRole.BACKWARD_POLICY.value,
                "params": groups.backward,
                "lr": config.backward_learning_rate or config.adapter_learning_rate,
            },
            {
                "name": "z-head",
                "params": groups.z_head,
                "lr": config.z_learning_rate,
            },
            *psi_group,
        ],
        weight_decay=0.0,
    )
    if config.stability is not None:
        for group in optimizer.param_groups:
            group["stability_condition"] = config.stability.to_value()
    return optimizer, groups


def prepare_ttb_step(
    *,
    backbone: PolicyBackbone,
    optimizer: torch.optim.Optimizer,
    parameters: PolicyParameterGroups,
    batch: CollectedTrainingBatch,
    snapshot_before: PolicySnapshot,
    clock: Callable[[], str],
    method: TTBMethodConfig,
) -> PreparedTTBStep:
    started_at = clock()
    optimizer.zero_grad(set_to_none=True)
    shard = compute_ttb_gradient_shard(
        backbone=backbone,
        parameters=parameters,
        batch=batch,
        positions=tuple(range(len(batch.artifacts))),
        global_batch_size=len(batch.artifacts),
        method=method,
    )
    return merge_ttb_gradient_shards(
        parameters=parameters,
        batch=batch,
        snapshot_before=snapshot_before,
        shards=(shard,),
        clock=clock,
        started_at=started_at,
    )


def compute_ttb_gradient_shard(
    *,
    backbone: PolicyBackbone,
    parameters: PolicyParameterGroups,
    batch: CollectedTrainingBatch,
    positions: tuple[int, ...],
    global_batch_size: int,
    method: TTBMethodConfig,
) -> TTBGradientShard:
    if not positions or len(positions) != len(set(positions)):
        raise ValueError("TTB gradient positions must be non-empty and unique")
    if global_batch_size != len(batch.artifacts):
        raise ValueError("TTB gradient global batch size differs from the sealed batch")
    if any(position < 0 or position >= global_batch_size for position in positions):
        raise ValueError("TTB gradient position lies outside the sealed batch")
    accumulated: dict[str, torch.Tensor] = {}
    artifact_math: list[TTBArtifactMath] = []
    for position in positions:
        shard = compute_ttb_artifact_contribution(
            backbone=backbone,
            parameters=parameters,
            artifact=batch.artifacts[position],
            position=position,
            batch_id=batch.batch_id,
            optimizer_step=batch.optimizer_step,
            policy_snapshot_id=batch.policy_snapshot_id,
            library_version=batch.library_version,
            global_batch_size=global_batch_size,
            method=method,
        )
        accumulate_ttb_gradients(accumulated, shard.gradients)
        artifact_math.extend(shard.artifacts)
    return TTBGradientShard(
        batch.batch_id, batch.optimizer_step, global_batch_size, tuple(artifact_math), accumulated
    )


def accumulate_ttb_gradients(
    accumulated: dict[str, torch.Tensor],
    contributions: Mapping[str, torch.Tensor],
) -> None:
    for name, contribution in contributions.items():
        if name not in accumulated:
            accumulated[name] = contribution.clone()
        else:
            accumulated[name].add_(contribution)


def require_finite_gradients(gradients: Mapping[str, torch.Tensor]) -> None:
    flags: dict[torch.device, list[torch.Tensor]] = {}
    for tensor in gradients.values():
        flags.setdefault(tensor.device, []).append(torch.isfinite(tensor).all())
    if not flags or any(not bool(torch.stack(values).all().item()) for values in flags.values()):
        raise ValueError("gradient contribution contains non-finite values")


def compute_ttb_artifact_contribution(
    *,
    backbone: PolicyBackbone,
    parameters: PolicyParameterGroups,
    artifact: RolloutArtifact,
    position: int,
    batch_id: str,
    optimizer_step: int,
    policy_snapshot_id: str,
    library_version: str,
    global_batch_size: int,
    method: TTBMethodConfig,
    prepared_edges: R2FlowEdgePlan | None = None,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> TTBGradientShard:
    if type(global_batch_size) is not int or not 0 <= position < global_batch_size:
        raise ValueError("gradient position lies outside the planned batch")
    if (
        artifact.manifest.policy_snapshot.snapshot_id != policy_snapshot_id
        or artifact.manifest.library_version != library_version
        or artifact.manifest.sampling_coordinate.optimizer_step_or_anchor_ordinal != optimizer_step
    ):
        raise ValueError("gradient artifact belongs to another policy, library or step")
    return _r2flow_artifact_contribution(
        backbone=backbone,
        parameters=parameters,
        artifact=artifact,
        position=position,
        batch_id=batch_id,
        policy_snapshot_id=policy_snapshot_id,
        library_version=library_version,
        optimizer_step=optimizer_step,
        global_batch_size=global_batch_size,
        method=method,
        plan=prepared_edges,
        progress=progress,
    )


def merge_ttb_gradient_shards(
    *,
    parameters: PolicyParameterGroups,
    batch: CollectedTrainingBatch,
    snapshot_before: PolicySnapshot,
    shards: tuple[TTBGradientShard, ...],
    clock: Callable[[], str],
    started_at: str,
) -> PreparedTTBStep:
    if not shards:
        raise ValueError("TTB gradient merge requires shards")
    if snapshot_before.snapshot_id != batch.policy_snapshot_id:
        raise ValueError("gradient merge policy differs from the collected batch")
    expected_positions = tuple(range(len(batch.artifacts)))
    indexed = {item.position: item for shard in shards for item in shard.artifacts}
    artifact_count = sum(len(shard.artifacts) for shard in shards)
    if tuple(sorted(indexed)) != expected_positions or artifact_count != len(indexed):
        raise ValueError("TTB gradient shards do not cover each trajectory exactly once")
    for shard in shards:
        if (
            shard.batch_id != batch.batch_id
            or shard.optimizer_step != batch.optimizer_step
            or shard.global_batch_size != len(batch.artifacts)
        ):
            raise ValueError("TTB gradient shard belongs to another sealed batch")
        for item in shard.artifacts:
            if item.trajectory_id != batch.artifacts[item.position].record.trajectory_id:
                raise ValueError("TTB gradient shard trajectory identity differs")
    named_parameters = named_ttb_parameters(parameters)
    if any(set(shard.gradients) != set(named_parameters) for shard in shards):
        raise ValueError("TTB gradient shards use different parameter sets")
    for name, parameter in named_parameters.items():
        combined = torch.zeros_like(parameter)
        for shard in shards:
            combined.add_(shard.gradients[name].to(device=parameter.device, dtype=parameter.dtype))
        parameter.grad = combined
    require_finite_gradients({n: p.grad for n, p in named_parameters.items() if p.grad is not None})
    ordered = tuple(indexed[position] for position in expected_positions)
    flow = tuple(item.residual for item in ordered)
    flow_edges = tuple(edge for item in ordered for edge in item.edges)
    losses = tuple(item.loss for item in ordered)
    stats = R2FlowBatchStats(
        batch_id=batch.batch_id,
        optimizer_step=batch.optimizer_step,
        library_version=batch.library_version,
        objective_id=SUBTB_RESIDUAL_ID,
        residuals=flow,
        batch_loss=math.fsum(item.loss for item in flow) / len(flow),
        mean_reward=math.fsum(item.raw_reward for item in flow) / len(flow),
        created_at=clock(),
    )
    return PreparedTTBStep(
        batch=batch,
        snapshot_before=snapshot_before,
        residuals=flow,
        edges=flow_edges,
        stats=stats,
        detached_losses=losses,
        gradient_norms=component_gradient_norms(parameters),
        started_at=started_at,
    )


def r2flow_edge_plan(
    backbone: PolicyBackbone, artifact: RolloutArtifact, method: TTBMethodConfig
) -> R2FlowEdgePlan:
    from skillev.scoring.r2flow_masks import artifact_action_masks
    from skillev.scoring.r2flow_plan import prepare_r2flow_artifact_plan

    return prepare_r2flow_artifact_plan(
        backbone.tokenizer,
        artifact,
        masks=artifact_action_masks(backbone, artifact.action_grammars),
    )


def _require_r2flow_scoring(
    backbone: PolicyBackbone, artifact: RolloutArtifact, method: TTBMethodConfig
) -> None:
    config = getattr(backbone, "teacher_forcing_config", None)
    if config is not None and config.microbatch_size > 1:
        raise ValueError("method@4 requires teacher-forcing microbatch_size 1 (per-edge bank)")
    if artifact.manifest.state_map != method.objective.state_map:
        raise ValueError("artifact state map differs from the method@4 state map")


def score_flow_trajectory(
    backbone: PolicyBackbone,
    artifact: RolloutArtifact,
    method: TTBMethodConfig,
    *,
    requires_grad: bool,
    plan: R2FlowEdgePlan | None = None,
    bank: EdgeGradientBank | None = None,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> tuple[R2FlowScore, torch.Tensor | None]:
    from skillev.scoring.r2flow_objective import backward_r2flow_streaming

    from .edge_gradient_bank import EdgeGradientBank as Bank

    _require_r2flow_scoring(backbone, artifact, method)
    if requires_grad != isinstance(bank, Bank):
        raise ValueError("gradient scoring needs exactly one edge gradient bank")
    plan = plan or r2flow_edge_plan(backbone, artifact, method)
    if plan.record is not artifact.record and plan.record != artifact.record:
        raise ValueError("R2 Flow plan belongs to another trajectory")
    return backward_r2flow_streaming(
        backbone,
        plan,
        method,
        bank=bank if requires_grad else None,
        progress=progress,
    )


def materialize_r2flow_records(
    score: R2FlowScore,
    record: TrajectoryRecordLike,
    *,
    batch_id: str,
    policy_snapshot_id: str,
    library_version: str,
    forward_adapter_version: str,
    backward_adapter_version: str,
    flow_head_version: str,
) -> tuple[R2FlowEdgeRecord, ...]:
    from skillev.contracts.ttb_training import EdgeScoreContext
    from skillev.scoring.r2flow_objective import R2FLOW_SCORING_STACK_ID

    residual = score.residual
    flows = (residual.log_z, *residual.log_flows, residual.log_reward_eta)
    records = []
    for index, edge in enumerate(score.edges):
        step = record.steps[index]
        assert step.r2flow is not None
        event_ids = (*step.action_token_ids, *step.r2flow.action_stop_token_ids)
        records.append(
            R2FlowEdgeRecord(
                trajectory_id=residual.trajectory_id,
                step_index=edge.step_index,
                context=EdgeScoreContext(
                    batch_id, policy_snapshot_id, library_version, len(event_ids), event_ids
                ),
                log_pf_reasoning=edge.pf_reasoning,
                log_pf_event=edge.pf_event,
                log_q_reasoning=edge.q_reasoning,
                log_pb_in_edge=edge.log_pb,
                in_edge_count=edge.in_edge_count,
                edge_log_ratio=edge.edge_log_ratio,
                log_flow_source=flows[index],
                log_flow_target=flows[index + 1],
                edge_residual=score.terms.edge_residuals[index],
                edge_coefficient=score.terms.edge_coefficients[index],
                raw_logp_event=edge.raw_logp_event,
                log_mask_mass_event=edge.log_mask_mass_event,
                event_token_count=edge.event_token_count,
                forced_event_token_count=edge.forced_event_token_count,
                reasoning_token_count=edge.reasoning_token_count,
                reasoning_stopped=edge.reasoning_stopped,
                forward_adapter_version=forward_adapter_version,
                backward_adapter_version=backward_adapter_version,
                flow_head_version=flow_head_version,
                scoring_stack_id=R2FLOW_SCORING_STACK_ID,
                backward_scores=edge.f_scores,
            )
        )
    return tuple(records)


def _r2flow_artifact_contribution(
    *,
    backbone: PolicyBackbone,
    parameters: PolicyParameterGroups,
    artifact: RolloutArtifact,
    position: int,
    batch_id: str,
    optimizer_step: int,
    policy_snapshot_id: str,
    library_version: str,
    global_batch_size: int,
    method: TTBMethodConfig,
    plan: R2FlowEdgePlan | None,
    progress: Callable[[dict[str, object]], None] | None,
) -> TTBGradientShard:
    from .edge_gradient_bank import EdgeGradientBank
    from .performance_config import DEFAULT_EDGE_GRADIENT_DEVICE_BYTES

    named = named_ttb_parameters(parameters)
    adapter_names = {name for name in named if name.startswith(("forward.", "backward."))}
    drain_scoring = getattr(backbone, "drain_scoring_metrics", list)
    drain_scoring()
    forward_version = backbone.adapter_version(AdapterRole.FORWARD_POLICY)
    backward_version = backbone.adapter_version(AdapterRole.BACKWARD_POLICY)
    if forward_version != artifact.manifest.policy_snapshot.forward_adapter_version:
        raise ValueError("gradient scorer uses another forward policy")
    performance = getattr(backbone, "performance_config", None)
    budget = getattr(performance, "edge_gradient_device_bytes", DEFAULT_EDGE_GRADIENT_DEVICE_BYTES)
    bank = EdgeGradientBank(
        {name: named[name] for name in adapter_names}, device_budget_bytes=budget
    )
    for parameter in named.values():
        parameter.grad = None
    try:
        score, surrogate = score_flow_trajectory(
            backbone, artifact, method, requires_grad=True, plan=plan, bank=bank, progress=progress
        )
        if (
            backbone.adapter_version(AdapterRole.FORWARD_POLICY) != forward_version
            or backbone.adapter_version(AdapterRole.BACKWARD_POLICY) != backward_version
        ):
            raise ValueError("policy changed during forward/backward scoring")
        scale = 1.0 / global_batch_size
        gradients = bank.combine(score.terms.edge_coefficients, scale)
        bank.clear()
        for parameter in named.values():
            if parameter.grad is not None:
                raise RuntimeError("adapter gradients escaped the edge gradient bank")
        assert surrogate is not None
        (surrogate * scale).backward()
        for name, parameter in named.items():
            if name in adapter_names:
                continue
            gradients[name] = (
                torch.zeros_like(parameter)
                if parameter.grad is None
                else parameter.grad.detach().clone()
            )
        if set(gradients) != set(named):
            raise RuntimeError("artifact did not produce all trainable gradients")
        records = materialize_r2flow_records(
            score,
            artifact.record,
            batch_id=batch_id,
            policy_snapshot_id=policy_snapshot_id,
            library_version=library_version,
            forward_adapter_version=forward_version,
            backward_adapter_version=backward_version,
            flow_head_version=backbone.z_version,
        )
        item = TTBArtifactMath(
            position,
            artifact.record.trajectory_id,
            score.residual,
            records,
            score.loss,
            tuple(drain_scoring()),
        )
        return TTBGradientShard(batch_id, optimizer_step, global_batch_size, (item,), gradients)
    finally:
        bank.clear()
        for parameter in named.values():
            parameter.grad = None


def named_ttb_parameters(
    parameters: PolicyParameterGroups,
) -> dict[str, torch.nn.Parameter]:
    result: dict[str, torch.nn.Parameter] = {}
    for group_name, group in (
        ("forward", parameters.forward),
        ("backward", parameters.backward),
        ("z_head", parameters.z_head),
        ("psi_head", parameters.psi_head),
    ):
        for index, parameter in enumerate(group):
            result[f"{group_name}.{index}"] = parameter
    if len({id(parameter) for parameter in result.values()}) != len(result):
        raise ValueError("TTB trainable parameter groups overlap")
    return result


def apply_optimizer_step(
    *,
    optimizer: torch.optim.Optimizer,
    backbone: PolicyBackbone,
    prepared: PreparedTTBStep,
    clock: Callable[[], str],
    observe_transition: bool = False,
) -> TrainingStepReportValue:
    observation = (
        OptimizerTransitionObservation.capture(optimizer, backbone.parameter_groups())
        if observe_transition
        else None
    )
    optimizer.step()
    transition = observation.finish() if observation is not None else None
    backbone.mark_policy_update(prepared.batch.optimizer_step)
    norms = prepared.gradient_norms
    assert norms.psi_head is not None
    return TrainingStepReportValue(
        optimizer_transition=transition,
        optimizer_step=prepared.batch.optimizer_step,
        batch_id=prepared.batch.batch_id,
        torch_batch_loss=(math.fsum(prepared.detached_losses) / len(prepared.detached_losses)),
        audited_batch_loss=prepared.stats.batch_loss,
        mean_reward=prepared.stats.mean_reward,
        grad_norm_forward=norms.forward,
        grad_norm_backward=norms.backward,
        grad_norm_z=norms.z_head,
        forward_adapter_version=backbone.adapter_version(AdapterRole.FORWARD_POLICY),
        backward_adapter_version=backbone.adapter_version(AdapterRole.BACKWARD_POLICY),
        z_version=backbone.z_version,
        started_at=prepared.started_at,
        completed_at=clock(),
        grad_norm_psi=norms.psi_head,
        optimization_diagnostics=None,
    )


def component_gradient_norms(parameters: PolicyParameterGroups) -> ComponentGradientNorms:
    return ComponentGradientNorms(
        forward=_gradient_l2_norm(parameters.forward),
        backward=_gradient_l2_norm(parameters.backward),
        z_head=_gradient_l2_norm(parameters.z_head),
        psi_head=_gradient_l2_norm(parameters.psi_head) if parameters.psi_head else None,
    )


def _gradient_l2_norm(parameters: tuple[torch.nn.Parameter, ...]) -> float:
    values: dict[torch.device, list[torch.Tensor]] = {}
    for parameter in parameters:
        if parameter.grad is not None:
            grad = parameter.grad.detach()
            values.setdefault(grad.device, []).append(grad.to(dtype=torch.float32).pow(2).sum())
    if not values:
        raise RuntimeError("trainable component produced no gradients")
    squared = [float(v) for group in values.values() for v in torch.stack(group).cpu().tolist()]
    return math.sqrt(math.fsum(squared))


__all__ = [
    "ComponentGradientNorms",
    "PreparedTTBStep",
    "TTBArtifactMath",
    "TTBGradientShard",
    "TTBStepPreparer",
    "apply_optimizer_step",
    "component_gradient_norms",
    "compute_ttb_gradient_shard",
    "create_ttb_optimizer",
    "merge_ttb_gradient_shards",
    "named_ttb_parameters",
    "prepare_ttb_step",
    "r2flow_edge_plan",
    "score_flow_trajectory",
]
