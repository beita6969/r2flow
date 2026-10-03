from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import torch

from skillev.contracts import TrajectoryRecord, stable_hash
from skillev.contracts.r2flow_training import R2FlowTrajectoryResidual
from skillev.contracts.subtb import SubTBTerms
from skillev.policy.interface import AdapterRole
from skillev.policy.teacher_forcing import SequenceScores, backward_signed_span_sums

from .backward_policy import log_pb_from_scores, log_pb_tensor, log_pb_without_scores
from .r2flow_plan import R2FlowEdgePlan, R2FlowStepPasses

if TYPE_CHECKING:
    from skillev.training.config import TTBMethodConfig
    from skillev.training.edge_gradient_bank import EdgeGradientBank

R2FLOW_SCORING_STACK_ID: Final = "r2flow-training-stack@1"

FORWARD = AdapterRole.FORWARD_POLICY
BACKWARD = AdapterRole.BACKWARD_POLICY


@dataclass(frozen=True, slots=True)
class R2FlowEdgeScalars:
    step_index: int
    pf_reasoning: float
    pf_event: float
    q_reasoning: float
    log_pb: float
    in_edge_count: int
    f_scores: tuple[float, ...]
    raw_logp_event: float
    log_mask_mass_event: float
    event_token_count: int
    forced_event_token_count: int
    reasoning_token_count: int
    reasoning_stopped: bool

    @property
    def edge_log_ratio(self) -> float:
        return math.fsum((self.pf_reasoning, self.pf_event, -self.log_pb, -self.q_reasoning))


@dataclass(frozen=True, slots=True)
class R2FlowScore:
    residual: R2FlowTrajectoryResidual
    terms: SubTBTerms
    edges: tuple[R2FlowEdgeScalars, ...]

    @property
    def loss(self) -> float:
        return self.residual.loss

    @property
    def trajectory_id(self) -> str:
        return self.residual.trajectory_id


def _session(backbone: Any, role: AdapterRole) -> Any:
    session = getattr(backbone, "scoring_session", None)
    return nullcontext() if session is None else session(role)


def _span_sum(scores: SequenceScores, index: int) -> float:
    return float(scores.spans[index].detach().double().sum())


def reference_in_edge_scores(backbone: Any, passes: R2FlowStepPasses) -> tuple[float, ...]:
    if passes.hindsight is None:
        raise ValueError("the learned in-edge softmax needs the hindsight pass")
    with torch.no_grad():
        others = [
            _span_sum(backbone.score_reference_sequence(sequence), 0)
            for sequence in passes.candidates
        ]
        actual = _span_sum(backbone.score_reference_sequence(passes.hindsight), 0)
    index = passes.actual_in_edge
    return (*others[:index], actual, *others[index:])


def _event_diagnostics(scores: SequenceScores, masked: float) -> tuple[float, float]:
    raw = scores.unmasked[0] if scores.unmasked else None
    if raw is None:
        return masked, 0.0
    mass = min(0.0, float(raw.detach().double().sum()) - masked)
    return masked + mass, mass


def score_edge_into_bank(
    backbone: Any,
    passes: R2FlowStepPasses,
    bank: EdgeGradientBank | None,
) -> tuple[R2FlowEdgeScalars, torch.Tensor | None]:
    t = passes.step_index
    grad = bank is not None
    context = nullcontext() if grad else torch.no_grad()
    feature: torch.Tensor | None = None
    with context:
        with _session(backbone, FORWARD):
            if passes.forward_reasoning is not None:
                with torch.no_grad():
                    feature = backbone.score_sequence(passes.forward_reasoning, FORWARD).feature
            scores = backbone.score_sequence(passes.forward_event, FORWARD)
            if grad:
                (pf_e,) = backward_signed_span_sums(scores, (1.0,))
            else:
                pf_e = _span_sum(scores, 0)
            raw_e, mass_e = _event_diagnostics(scores, pf_e)
            del scores
        if bank is not None:
            bank.capture(t)
        with _session(backbone, BACKWARD):
            reference_f = (
                reference_in_edge_scores(backbone, passes)
                if passes.learned_backward_policy
                else None
            )
            candidate_f: list[float] = []
            for c, sequence in enumerate(passes.candidates):
                scores = backbone.score_sequence(sequence, BACKWARD)
                if grad:
                    (value,) = backward_signed_span_sums(scores, (1.0,))
                    assert bank is not None
                    bank.capture_candidate(t, c)
                else:
                    value = _span_sum(scores, 0)
                candidate_f.append(value)
                del scores
            hindsight = passes.hindsight
            scores = None if hindsight is None else backbone.score_sequence(hindsight, BACKWARD)
            actual = passes.actual_in_edge
            f_scores: tuple[float, ...] = ()
            fold: dict[int, float] = {}
            if passes.learned_backward_policy:
                assert scores is not None
                f_actual = _span_sum(scores, 0)
                f_scores = (*candidate_f[:actual], f_actual, *candidate_f[actual:])
                if reference_f is not None:
                    f_scores = tuple(f - r for f, r in zip(f_scores, reference_f, strict=True))
                log_pb = log_pb_from_scores(f_scores, actual)
                top = max(f_scores)
                normaliser = top + math.log(math.fsum(math.exp(f - top) for f in f_scores))
                weights = [math.exp(f - normaliser) for f in f_scores]
                sign = -(1.0 - weights[actual])
                others = [c for c in range(passes.in_edge_count) if c != actual]
                fold = {index: weights[c] for index, c in enumerate(others)}
            else:
                log_pb = log_pb_without_scores(actual, passes.in_edge_count)
                sign = 0.0
            if scores is not None and grad:
                backward_signed_span_sums(scores, (sign,))
            del scores
        if bank is not None:
            bank.capture(t)
            if fold:
                bank.fold_candidates(t, fold)
    scalars = R2FlowEdgeScalars(
        step_index=t,
        pf_reasoning=0.0,
        pf_event=pf_e,
        q_reasoning=0.0,
        log_pb=log_pb,
        in_edge_count=passes.in_edge_count,
        f_scores=f_scores,
        raw_logp_event=raw_e,
        log_mask_mass_event=mass_e,
        event_token_count=len(passes.event_token_ids),
        forced_event_token_count=passes.forced_event_tokens,
        reasoning_token_count=passes.reasoning_token_count,
        reasoning_stopped=passes.reasoning_stopped,
    )
    return scalars, feature


def _query_ids(backbone: Any, record: TrajectoryRecord) -> tuple[int, ...]:
    from .encoding import encoded_query

    return encoded_query(backbone.tokenizer, record)


def _require_method(record: TrajectoryRecord, method: TTBMethodConfig) -> float:
    if not math.isclose(record.epsilon_min, method.epsilon_min, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("trajectory epsilon differs from the method epsilon")
    return method.objective.subtb_lambda


def trajectory_domain(record: TrajectoryRecord) -> str:
    return record.task_family.partition("/")[0]


def flow_offset(backbone: Any, record: TrajectoryRecord) -> Any:
    return backbone.flow_offset(trajectory_domain(record))


def r2flow_residual(
    record: TrajectoryRecord,
    method: TTBMethodConfig,
    *,
    log_z: float,
    log_psi: Sequence[float],
    edges: Sequence[R2FlowEdgeScalars],
) -> R2FlowTrajectoryResidual:
    lam = _require_method(record, method)
    return R2FlowTrajectoryResidual.from_terms(
        trajectory_id=record.trajectory_id,
        query_hash=stable_hash({"query": record.initial_context.query}),
        log_z=log_z,
        log_flows=tuple(float(value) for value in log_psi),
        edge_log_ratios=tuple(edge.edge_log_ratio for edge in edges),
        raw_reward=record.reward.value,
        eta=method.temperature_beta,
        epsilon=method.epsilon_min,
        subtb_lambda=lam,
    )


def finalize_r2flow(
    backbone: Any,
    record: TrajectoryRecord,
    method: TTBMethodConfig,
    edges: Sequence[R2FlowEdgeScalars],
    state_features: Sequence[torch.Tensor],
    *,
    requires_grad: bool = True,
) -> tuple[R2FlowScore, torch.Tensor | None]:
    horizon = record.horizon
    if len(edges) != horizon or len(state_features) != horizon - 1:
        raise ValueError("R2 Flow finalisation needs T edges and T - 1 state features")
    context = nullcontext() if requires_grad else torch.no_grad()
    with context:
        z = backbone.z_value(_query_ids(backbone, record))
        psi = (
            backbone.psi_values(torch.stack([f.detach() for f in state_features]))
            if horizon > 1
            else None
        )
        offset = flow_offset(backbone, record)
        z = z + offset.to(dtype=z.dtype, device=z.device)
        if psi is not None:
            psi = psi + offset.to(dtype=psi.dtype, device=psi.device)
    log_z = float(z.detach().double())
    log_psi = () if psi is None else tuple(float(v) for v in psi.detach().double().tolist())
    residual = r2flow_residual(record, method, log_z=log_z, log_psi=log_psi, edges=edges)
    terms = residual.terms_with_gradient_clip(method.residual_gradient_clip)
    surrogate = None
    if requires_grad:
        d = terms.flow_coefficients
        surrogate = z * d[0]
        if psi is not None:
            surrogate = (
                surrogate + (psi * torch.tensor(d[1:], dtype=psi.dtype, device=psi.device)).sum()
            )
    return R2FlowScore(residual, terms, tuple(edges)), surrogate


def backward_r2flow_streaming(
    backbone: Any,
    plan: R2FlowEdgePlan,
    method: TTBMethodConfig,
    *,
    bank: EdgeGradientBank | None,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> tuple[R2FlowScore, torch.Tensor | None]:
    edges: list[R2FlowEdgeScalars] = []
    features: list[torch.Tensor] = []
    for passes in plan.steps:
        if progress is not None:
            progress(
                {
                    "stage": "r2flow-edge-score-and-backward",
                    "step_index": passes.step_index,
                    "passes": len(passes.sequences),
                    "tokens": passes.token_cost,
                }
            )
        scalars, feature = score_edge_into_bank(backbone, passes, bank)
        edges.append(scalars)
        if passes.step_index > 1:
            if feature is None:
                raise RuntimeError("the reasoning pass returned no F_psi feature")
            features.append(feature)
        finish = getattr(backbone, "finish_edge_backward_profile", None)
        if finish is not None and bank is not None:
            finish()
    if progress is not None:
        progress({"stage": "r2flow-finalize"})
    return finalize_r2flow(
        backbone, plan.record, method, edges, features, requires_grad=bank is not None
    )


@contextmanager
def _keep(role: AdapterRole, backbone: Any) -> Iterator[None]:
    with _session(backbone, role):
        yield


def score_r2flow_trajectory(
    backbone: Any, plan: R2FlowEdgePlan, method: TTBMethodConfig
) -> tuple[torch.Tensor, R2FlowScore]:
    pf: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    with _keep(FORWARD, backbone):
        for passes in plan.steps:
            event = backbone.score_sequence(passes.forward_event, FORWARD)
            term = event.spans[0].double().sum()
            if passes.forward_reasoning is not None and passes.step_index > 1:
                features.append(backbone.score_sequence(passes.forward_reasoning, FORWARD).feature)
            pf.append(term)
    pb: list[torch.Tensor] = []
    with _keep(BACKWARD, backbone):
        for passes in plan.steps:
            hindsight = (
                None
                if passes.hindsight is None
                else backbone.score_sequence(passes.hindsight, BACKWARD)
            )
            if passes.learned_backward_policy:
                assert hindsight is not None
                f = [
                    backbone.score_sequence(c, BACKWARD).spans[0].double().sum()
                    for c in passes.candidates
                ]
                f.insert(passes.actual_in_edge, hindsight.spans[0].double().sum())
                stacked = torch.stack(f) - torch.tensor(
                    reference_in_edge_scores(backbone, passes),
                    dtype=torch.float64,
                )
                log_pb = log_pb_tensor(stacked, passes.actual_in_edge)
            else:
                log_pb = torch.tensor(
                    log_pb_without_scores(passes.actual_in_edge, passes.in_edge_count),
                    dtype=torch.float64,
                )
            pb.append(log_pb)
    record = plan.record
    z = backbone.z_value(_query_ids(backbone, record)).double()
    shift = flow_offset(backbone, record).double()
    flows = [z + shift]
    if record.horizon > 1:
        flows.extend((backbone.psi_values(torch.stack(features)).double() + shift).unbind(0))
    anchor = method.temperature_beta * math.log(record.reward.value + method.epsilon_min)
    flows.append(torch.tensor(anchor, dtype=torch.float64))
    a = [forward - backward for forward, backward in zip(pf, pb, strict=True)]
    lam = _require_method(record, method)
    pairs = [(i, j) for i in range(record.horizon) for j in range(i + 1, record.horizon + 1)]
    raw = [lam ** (j - i) for i, j in pairs]
    norm = math.fsum(raw)
    loss = torch.zeros((), dtype=torch.float64)
    for (i, j), weight in zip(pairs, raw, strict=True):
        delta = flows[i] + sum(a[i:j], torch.zeros((), dtype=torch.float64)) - flows[j]
        loss = loss + (weight / norm) * delta * delta
    edges = []
    for passes, forward, backward in zip(plan.steps, pf, pb, strict=True):
        del forward, backward
        scalars, _ = score_edge_into_bank(backbone, passes, None)
        edges.append(scalars)
    residual = r2flow_residual(
        record,
        method,
        log_z=float(flows[0].detach()),
        log_psi=[float(v.detach()) for v in flows[1:-1]],
        edges=edges,
    )
    return loss, R2FlowScore(residual, residual.terms, tuple(edges))


__all__ = [
    "R2FLOW_SCORING_STACK_ID",
    "R2FlowEdgeScalars",
    "R2FlowScore",
    "backward_r2flow_streaming",
    "finalize_r2flow",
    "r2flow_residual",
    "score_edge_into_bank",
    "score_r2flow_trajectory",
]
