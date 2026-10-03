from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, cast

from skillev.contracts import TokenizerProtocol, TrajectoryRecord
from skillev.policy.event_grammar import EventGrammarSpec, FunctionSpec
from skillev.policy.interface import ModelInputWindow, encode_policy_prompt
from skillev.policy.teacher_forcing import ScoredSpan, ScoringSequence
from skillev.policy.token_mask import TokenMask

from skillev.contracts.state_map import PB_IN_EDGE_SOFTMAX

from .backward_policy import needs_scores
from .rendering import (
    in_edge_scoring_inputs,
    render_forward_prefix,
    render_reasoning_prefix,
)

REASONING_CONDITIONED: Final = "sampled-reasoning-conditioned@1"
R2FLOW_PASS_PLAN_VERSION: Final = "r2flow-pass-plan@1"
EVENT_SCORING_FREE_TEXT_CONDITIONED: Final = "grammar-masked-sum-free-text-conditioned@1"
_PARAMETER_CLOSE: Final = "\n</parameter>"


def free_text_opener(function: FunctionSpec) -> str | None:
    if not function.params or function.params[-1].kind != "text":
        return None
    return f"<parameter={function.params[-1].name}>\n"


def free_text_value_tokens(
    ids: Sequence[int],
    decode: Callable[[tuple[int, ...]], str],
    opener: str,
) -> tuple[int, int] | None:
    ids = tuple(ids)
    text = decode(ids)
    opened = text.find(opener)
    if opened < 0:
        return None
    start = opened + len(opener)
    close = text.find(_PARAMETER_CLOSE, start)
    if close < 0:
        return None

    def offset(k: int) -> int:
        return len(decode(ids[:k]))

    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if offset(mid) >= start:
            hi = mid
        else:
            lo = mid + 1
    first = lo
    lo, hi = first, len(ids)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if offset(mid) <= close:
            lo = mid
        else:
            hi = mid - 1
    stop = lo
    return (first, stop) if first < stop else None


def force_sampled_rows(mask: TokenMask, ids: Sequence[int], start: int, stop: int) -> TokenMask:
    import torch

    bits = mask.bitmask.clone()
    bits[start:stop] = 0
    for row in range(start, stop):
        word, bit = divmod(int(ids[row]), 32)
        value = 1 << bit
        bits[row, word] = torch.tensor(value - (1 << 32) if value >= 1 << 31 else value)
    return TokenMask(mask.constraint_hash, mask.vocab_size, bits.contiguous())


class ReplayedMask(Protocol):
    @property
    def mask(self) -> TokenMask: ...

    @property
    def digest(self) -> str: ...

    @property
    def forced_count(self) -> int: ...


class ActionMaskSource(Protocol):
    def replay(self, *, constraint_hash: str, action_token_ids: Sequence[int]) -> ReplayedMask: ...

    def grammar_sha_for(self, spec: EventGrammarSpec) -> str: ...


def _ids(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(t) is not int or t < 0 for t in value):
        raise ValueError(f"r2flow plan wire: {field} must be a list of token ids")
    return tuple(value)


def _sequence_wire(sequence: ScoringSequence) -> dict[str, object]:
    return {
        "ids": list(sequence.token_ids),
        "spans": [
            [span.start, span.stop, None if span.mask is None else span.mask.constraint_hash]
            for span in sequence.spans
        ],
        "feature": sequence.feature_position,
    }


def _sequence_from_wire(
    value: object, masks: Callable[[str, tuple[int, ...]], TokenMask] | None
) -> ScoringSequence:
    if not isinstance(value, dict) or set(value) != {"ids", "spans", "feature"}:
        raise ValueError("r2flow plan wire: invalid scoring sequence")
    ids = _ids(value["ids"], "ids")
    spans = []
    raw_spans = value["spans"]
    if not isinstance(raw_spans, list):
        raise ValueError("r2flow plan wire: spans must be a list")
    for raw in raw_spans:
        if not isinstance(raw, list) or len(raw) != 3:
            raise ValueError("r2flow plan wire: invalid span")
        start, stop, constraint = raw
        if type(start) is not int or type(stop) is not int:
            raise ValueError("r2flow plan wire: invalid span bounds")
        mask = None
        if constraint is not None:
            if masks is None or not isinstance(constraint, str):
                raise ValueError("r2flow plan wire: masked span needs a mask provider")
            mask = masks(constraint, ids[start:stop])
        spans.append(ScoredSpan(start, stop, mask))
    feature = value["feature"]
    if feature is not None and type(feature) is not int:
        raise ValueError("r2flow plan wire: invalid feature position")
    return ScoringSequence(ids, tuple(spans), feature)


def _optional_wire(sequence: ScoringSequence | None) -> dict[str, object] | None:
    return None if sequence is None else _sequence_wire(sequence)


def _optional_from_wire(value: object) -> ScoringSequence | None:
    return None if value is None else _sequence_from_wire(value, None)


@dataclass(frozen=True, slots=True)
class R2FlowStepPasses:
    step_index: int
    forward_reasoning: ScoringSequence | None
    forward_event: ScoringSequence
    hindsight: ScoringSequence | None
    candidates: tuple[ScoringSequence, ...]
    in_edge_count: int
    actual_in_edge: int
    backward_policy: str
    event_mask_digest: str
    forced_event_tokens: int
    reasoning_stopped: bool
    reasoning_scoring: str = REASONING_CONDITIONED
    reasoning_tokens: int | None = None
    conditioned_event_tokens: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index < 1:
            raise ValueError("step passes use one-based step indices")
        if self.reasoning_scoring != REASONING_CONDITIONED:
            raise ValueError("unsupported reasoning scoring")
        if self.backward_policy != PB_IN_EDGE_SOFTMAX:
            raise ValueError("unsupported backward policy mode")
        event = self.forward_event.spans
        if len(event) != 1 or event[0].mask is None:
            raise ValueError("the event pass scores one grammar-masked span")
        if self.conditioned_event_tokens is not None:
            a, b = self.conditioned_event_tokens
            if not 0 <= a < b <= event[0].stop - event[0].start:
                raise ValueError("conditioned event tokens lie outside the event span")
        if not 0 <= self.actual_in_edge < self.in_edge_count:
            raise ValueError("actual in-edge out of range")
        expected = self.in_edge_count - 1 if self.learned_backward_policy else 0
        if len(self.candidates) != expected:
            raise ValueError("candidate passes exist exactly for the learned |In| > 1 case")
        if any(len(c.spans) != 1 or c.spans[0].mask is not None for c in self.candidates):
            raise ValueError("candidate passes score one unmasked event span")
        self._check_conditioned()

    def _check_conditioned(self) -> None:
        reasoning = self.forward_reasoning
        if (self.step_index == 1) != (reasoning is None):
            raise ValueError("the conditioned feature pass exists exactly for t >= 2")
        if reasoning is not None and (
            reasoning.spans or reasoning.feature_position != len(reasoning.token_ids) - 1
        ):
            raise ValueError("the conditioned feature pass is the reasoning prompt only")
        if (self.hindsight is not None) != self.learned_backward_policy:
            raise ValueError("the conditioned hindsight pass exists only for learned P_B")
        if self.hindsight is not None and (
            len(self.hindsight.spans) != 1 or self.hindsight.spans[0].mask is not None
        ):
            raise ValueError("the conditioned hindsight pass scores the event span only")
        if type(self.reasoning_tokens) is not int or self.reasoning_tokens < 0:
            raise ValueError("conditioned passes record the sampled reasoning token count")

    @property
    def learned_backward_policy(self) -> bool:
        return needs_scores(self.in_edge_count)

    @property
    def sequences(self) -> tuple[ScoringSequence, ...]:
        return tuple(
            sequence
            for sequence in (
                self.forward_reasoning,
                self.forward_event,
                self.hindsight,
                *self.candidates,
            )
            if sequence is not None
        )

    @property
    def token_cost(self) -> int:
        return sum(len(sequence.token_ids) for sequence in self.sequences)

    @property
    def max_sequence_tokens(self) -> int:
        return max(len(sequence.token_ids) for sequence in self.sequences)

    @property
    def event_token_ids(self) -> tuple[int, ...]:
        span = self.forward_event.spans[0]
        return self.forward_event.token_ids[span.start : span.stop]

    @property
    def reasoning_token_count(self) -> int:
        assert self.reasoning_tokens is not None
        return self.reasoning_tokens

    def to_wire(self) -> dict[str, object]:
        return {
            "step": self.step_index,
            "forward_reasoning": _optional_wire(self.forward_reasoning),
            "forward_event": _sequence_wire(self.forward_event),
            "hindsight": _optional_wire(self.hindsight),
            "candidates": [_sequence_wire(c) for c in self.candidates],
            "in_edge_count": self.in_edge_count,
            "actual_in_edge": self.actual_in_edge,
            "backward_policy": self.backward_policy,
            "event_mask_digest": self.event_mask_digest,
            "forced_event_tokens": self.forced_event_tokens,
            "reasoning_stopped": self.reasoning_stopped,
            "reasoning_scoring": self.reasoning_scoring,
            "reasoning_tokens": self.reasoning_tokens,
            "conditioned_event_tokens": None
            if self.conditioned_event_tokens is None
            else list(self.conditioned_event_tokens),
        }

    @classmethod
    def from_wire(cls, value: object, masks: ActionMaskSource) -> R2FlowStepPasses:
        if not isinstance(value, dict):
            raise ValueError("r2flow plan wire: step must be an object")
        digest = value.get("event_mask_digest")
        replayed: list[ReplayedMask] = []

        def replay(constraint: str, ids: tuple[int, ...]) -> TokenMask:
            result = masks.replay(constraint_hash=constraint, action_token_ids=ids)
            if result.digest != digest:
                raise ValueError("worker mask replay digest differs from the plan")
            replayed.append(result)
            return result.mask

        candidates = value.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("r2flow plan wire: candidates must be a list")
        raw_conditioned = value.get("conditioned_event_tokens")
        conditioned: tuple[int, int] | None = None
        if raw_conditioned is not None:
            if not (
                isinstance(raw_conditioned, list)
                and len(raw_conditioned) == 2
                and all(type(x) is int for x in raw_conditioned)
            ):
                raise ValueError("r2flow plan wire: invalid conditioned event tokens")
            conditioned = (raw_conditioned[0], raw_conditioned[1])
        forward_event = _sequence_from_wire(value.get("forward_event"), replay)
        if conditioned is not None:
            span = forward_event.spans[0]
            assert span.mask is not None
            forced = force_sampled_rows(
                span.mask, forward_event.token_ids[span.start : span.stop], *conditioned
            )
            forward_event = ScoringSequence(
                forward_event.token_ids,
                (ScoredSpan(span.start, span.stop, forced),),
                forward_event.feature_position,
            )
        passes = cls(
            step_index=cast(int, value.get("step")),
            forward_reasoning=_optional_from_wire(value.get("forward_reasoning")),
            forward_event=forward_event,
            hindsight=_optional_from_wire(value.get("hindsight")),
            candidates=tuple(_sequence_from_wire(c, None) for c in candidates),
            in_edge_count=cast(int, value.get("in_edge_count")),
            actual_in_edge=cast(int, value.get("actual_in_edge")),
            backward_policy=cast(str, value.get("backward_policy")),
            event_mask_digest=cast(str, digest),
            forced_event_tokens=cast(int, value.get("forced_event_tokens")),
            reasoning_stopped=cast(bool, value.get("reasoning_stopped")),
            reasoning_scoring=cast(str, value.get("reasoning_scoring")),
            reasoning_tokens=cast(int | None, value.get("reasoning_tokens")),
            conditioned_event_tokens=conditioned,
        )
        if len(replayed) != 1 or replayed[0].forced_count != passes.forced_event_tokens:
            raise ValueError("worker mask replay differs from the plan")
        return passes


@dataclass(frozen=True, slots=True)
class R2FlowEdgePlan:
    record: TrajectoryRecord
    initial_text: str
    tokenizer_id: str
    steps: tuple[R2FlowStepPasses, ...]
    format: str = R2FLOW_PASS_PLAN_VERSION

    def __post_init__(self) -> None:
        if len(self.steps) != self.record.horizon or any(
            passes.step_index != index for index, passes in enumerate(self.steps, 1)
        ):
            raise ValueError("an R2 Flow plan covers every edge of the trajectory in order")

    @property
    def token_cost(self) -> int:
        return max(1, sum(passes.token_cost for passes in self.steps))

    @property
    def max_sequence_tokens(self) -> int:
        return max(passes.max_sequence_tokens for passes in self.steps)

    def wire_steps(self) -> list[dict[str, object]]:
        return [passes.to_wire() for passes in self.steps]

    @classmethod
    def from_wire_steps(
        cls,
        value: object,
        *,
        record: TrajectoryRecord,
        initial_text: str,
        tokenizer_id: str,
        masks: ActionMaskSource,
    ) -> R2FlowEdgePlan:
        if not isinstance(value, list) or len(value) != record.horizon:
            raise ValueError("R2 Flow worker plan does not cover the complete trajectory")
        steps = tuple(R2FlowStepPasses.from_wire(item, masks) for item in value)
        for passes, step in zip(steps, record.steps, strict=True):
            assert step.r2flow is not None
            span = passes.forward_event.spans[0]
            if (
                passes.event_token_ids
                != (*step.action_token_ids, *step.r2flow.action_stop_token_ids)
                or span.mask is None
                or span.mask.constraint_hash != step.r2flow.action_grammar_sha256
            ):
                raise ValueError("R2 Flow worker plan event span differs from the record")
        return cls(record, initial_text, tokenizer_id, steps)


def _check_initial_context(
    tokenizer: TokenizerProtocol, record: TrajectoryRecord, text: str
) -> None:
    from .encoding import encoded_initial_context

    encoded_initial_context(tokenizer, record, text)


def prepare_r2flow_step(
    tokenizer: TokenizerProtocol,
    record: TrajectoryRecord,
    initial_text: str,
    t: int,
    *,
    masks: ActionMaskSource,
    native_thinking: bool,
) -> R2FlowStepPasses:
    step = record.steps[t - 1]
    flow = step.r2flow
    if flow is None:
        raise ValueError("R2 Flow scoring requires step-r2flow@1 records (sigma mode)")
    window = ModelInputWindow.from_meta(record.initial_context.meta)

    def encode(text: str, *, thinking: bool = False) -> tuple[int, ...]:
        return encode_policy_prompt(
            tokenizer,
            text,
            initial_text=initial_text,
            window=window,
            step_index=t,
            native_thinking=thinking,
        ).ids

    reasoning_prompt = encode(
        render_reasoning_prefix(initial_text, record.steps[: t - 1], t).text,
        thinking=native_thinking,
    )
    r_ids = (
        *flow.reasoning_token_ids,
        *(flow.reasoning_stop_token_ids if flow.reasoning_token_ids_include_stop else ()),
    )
    if not r_ids:
        raise ValueError("a reasoning edge must contain at least one sampled token")
    n = len(reasoning_prompt)
    forward_reasoning = (
        None if t == 1 else ScoringSequence(reasoning_prompt, (), feature_position=n - 1)
    )

    forward = render_forward_prefix(initial_text, record.steps, t)
    if forward.prefix_hash != step.forward_prefix_hash:
        raise ValueError(f"step {t}: forward prefix hash differs from the recorded prefix")
    from .quotient import legal_events_at, sigma

    legal = legal_events_at(initial_text, sigma(initial_text, record.steps[: t - 1]))
    if legal.sha256 != flow.legal_event_set_sha256:
        raise ValueError(f"step {t}: legal event set differs from the recorded one")
    if len(flow.action_stop_token_ids) != 1:
        raise ValueError(f"step {t}: an event ends on exactly one grammar stop token")
    spec = legal.grammar_spec(
        budget=flow.action_budget_tokens, stop_token_id=flow.action_stop_token_ids[0]
    )
    if masks.grammar_sha_for(spec) != flow.action_grammar_sha256:
        raise ValueError(f"step {t}: recomputed action grammar sha differs from the record")
    e_ids = (*step.action_token_ids, *flow.action_stop_token_ids)
    replayed = masks.replay(constraint_hash=flow.action_grammar_sha256, action_token_ids=e_ids)
    if replayed.digest != flow.action_masks_digest:
        raise ValueError(f"step {t}: replayed action mask digest differs from the record")
    if replayed.mask.constraint_hash != flow.action_grammar_sha256 or replayed.mask.rows != len(
        e_ids
    ):
        raise ValueError(f"step {t}: replayed mask does not cover the scored event")
    action_prompt = encode(forward.text)
    m = len(action_prompt)
    event_mask = replayed.mask
    conditioned_event: tuple[int, int] | None = None
    opener = free_text_opener(spec.function(flow.event_function))
    if opener is not None:
        conditioned_event = free_text_value_tokens(e_ids, tokenizer.decode, opener)
        if conditioned_event is not None:
            event_mask = force_sampled_rows(event_mask, e_ids, *conditioned_event)
    forward_event = ScoringSequence(
        (*action_prompt, *e_ids), (ScoredSpan(m, m + len(e_ids), event_mask),)
    )

    inputs = in_edge_scoring_inputs(initial_text, record.steps, t)
    if inputs.prefix.prefix_hash != step.hindsight_prefix_hash:
        raise ValueError(f"step {t}: hindsight prefix hash differs from the recorded prefix")
    if (
        len(inputs.candidate_texts) != len(flow.in_edges)
        or inputs.actual_index != flow.actual_in_edge
        or inputs.predecessor_keys != tuple(edge[0] for edge in flow.in_edges)
    ):
        raise ValueError(f"step {t}: in-edges differ from the recorded state record")
    psi_prompt = encode(inputs.prefix.text)
    h = len(psi_prompt)
    event_text_ids = tuple(tokenizer.encode(inputs.actual_event_text))
    bridge_ids = tuple(tokenizer.encode(inputs.bridge))
    if not event_text_ids or not bridge_ids:
        raise ValueError("canonical event and bridge texts must encode to tokens")
    count = len(inputs.candidate_texts)
    hindsight: ScoringSequence | None
    if needs_scores(count):
        hindsight = ScoringSequence(
            (*psi_prompt, *event_text_ids), (ScoredSpan(h, h + len(event_text_ids)),)
        )
    else:
        hindsight = None
    candidates: tuple[ScoringSequence, ...] = ()
    if needs_scores(count):
        encoded = []
        for index, text in enumerate(inputs.candidate_texts):
            if index == inputs.actual_index:
                continue
            ids = tuple(tokenizer.encode(text))
            if not ids:
                raise ValueError("candidate event text must encode to tokens")
            encoded.append(ScoringSequence((*psi_prompt, *ids), (ScoredSpan(h, h + len(ids)),)))
        candidates = tuple(encoded)
    return R2FlowStepPasses(
        step_index=t,
        forward_reasoning=forward_reasoning,
        forward_event=forward_event,
        hindsight=hindsight,
        candidates=candidates,
        in_edge_count=count,
        actual_in_edge=inputs.actual_index,
        backward_policy=PB_IN_EDGE_SOFTMAX,
        event_mask_digest=replayed.digest,
        forced_event_tokens=replayed.forced_count,
        reasoning_stopped=flow.reasoning_token_ids_include_stop,
        reasoning_tokens=len(r_ids),
        conditioned_event_tokens=conditioned_event,
    )


def prepare_r2flow_plan(
    tokenizer: TokenizerProtocol,
    record: TrajectoryRecord,
    initial_text: str,
    *,
    masks: ActionMaskSource,
    native_thinking: bool,
) -> R2FlowEdgePlan:
    if tokenizer.tokenizer_id != record.tokenizer_id:
        raise ValueError("R2 Flow plan tokenizer differs from the recorded tokenizer")
    _check_initial_context(tokenizer, record, initial_text)
    steps = tuple(
        prepare_r2flow_step(
            tokenizer,
            record,
            initial_text,
            t,
            masks=masks,
            native_thinking=native_thinking,
        )
        for t in range(1, record.horizon + 1)
    )
    return R2FlowEdgePlan(record, initial_text, tokenizer.tokenizer_id, steps)


def artifact_native_thinking(prompt_encoder_version: str) -> bool:
    from skillev.policy.interface import THINKING_ROLLOUT_PROMPT_ENCODER_VERSION

    return prompt_encoder_version == THINKING_ROLLOUT_PROMPT_ENCODER_VERSION


def prepare_r2flow_artifact_plan(
    tokenizer: TokenizerProtocol,
    artifact: object,
    *,
    masks: ActionMaskSource,
) -> R2FlowEdgePlan:
    from skillev.rollout import RolloutArtifact

    if not isinstance(artifact, RolloutArtifact):
        raise TypeError("expected a RolloutArtifact")
    return prepare_r2flow_plan(
        tokenizer,
        artifact.record,
        artifact.initial_context.text,
        masks=masks,
        native_thinking=artifact_native_thinking(artifact.manifest.prompt_encoder_version),
    )


__all__ = [
    "EVENT_SCORING_FREE_TEXT_CONDITIONED",
    "R2FLOW_PASS_PLAN_VERSION",
    "REASONING_CONDITIONED",
    "ActionMaskSource",
    "R2FlowEdgePlan",
    "R2FlowStepPasses",
    "artifact_native_thinking",
    "force_sampled_rows",
    "free_text_opener",
    "free_text_value_tokens",
    "prepare_r2flow_artifact_plan",
    "prepare_r2flow_plan",
    "prepare_r2flow_step",
]
