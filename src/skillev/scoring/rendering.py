from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, cast

from skillev.contracts.canonical import stable_hash
from skillev.contracts.ttb_trajectory import TrajectoryStep
from skillev.policy.phase_context import native_action_reminder
from skillev.policy.skill_availability import context_for_turn
from skillev.policy.state_view import IN_EDGE_HINDSIGHT_CHOICE_INSTRUCTION

if TYPE_CHECKING:
    from skillev.policy.phase_context import PhaseContextSpec

    from .quotient import SharedState, StepLike

TEMPLATE_VERSION: Final = "ttb-render@5"
STATE_TEMPLATE_VERSION: Final = "ttb-render-state@1"


REASONING_POSTERIOR_BRIDGE: Final = "\nReasoning that led to this call:\n"

_REASONING_GENERATION_REMINDER: Final = (
    "Reasoning pass: reason about the task and current public state.\n"
)

PrefixKind: TypeAlias = Literal["forward", "in-edge-hindsight"]

_PREFIX_KINDS: Final = frozenset({"forward", "in-edge-hindsight"})


def _require_prefix_kind(kind: str) -> PrefixKind:
    if kind not in _PREFIX_KINDS:
        raise ValueError("prefix kind must be 'forward' or 'in-edge-hindsight'")
    return cast(PrefixKind, kind)


def _step_at(
    steps: tuple[TrajectoryStep, ...],
    step_index: int,
) -> TrajectoryStep:
    if type(step_index) is not int or not 1 <= step_index <= len(steps):
        raise ValueError("step_index must use one-based indexing within steps")
    for expected_index, step in enumerate(steps[:step_index], start=1):
        if step.index != expected_index:
            raise ValueError("steps must have contiguous one-based indices")
    return steps[step_index - 1]


def _require_previous_steps(
    previous_steps: tuple[TrajectoryStep, ...],
    step_index: int,
) -> None:
    if type(step_index) is not int or step_index < 1:
        raise ValueError("step_index must use one-based indexing")
    if len(previous_steps) != step_index - 1:
        raise ValueError("previous_steps must contain exactly the history before step_index")
    for expected_index, step in enumerate(previous_steps, start=1):
        if step.index != expected_index:
            raise ValueError("previous_steps must have contiguous one-based indices")


@dataclass(frozen=True, slots=True)
class RenderedPrefix:
    kind: PrefixKind
    step_index: int
    text: str
    prefix_hash: str

    def __post_init__(self) -> None:
        _require_prefix_kind(self.kind)
        if type(self.step_index) is not int or self.step_index < 1:
            raise ValueError("step_index must use one-based indexing")


@dataclass(frozen=True, slots=True)
class RenderedReasoningPrompt:
    step_index: int
    text: str
    prompt_hash: str

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index < 1:
            raise ValueError("step_index must use one-based indexing")


def assembled_context_hash(initial_text: str) -> str:
    return stable_hash(
        {
            "initial_text": initial_text,
            "template_version": TEMPLATE_VERSION,
        }
    )


def prefix_content_hash(*, kind: str, text: str) -> str:
    checked_kind = _require_prefix_kind(kind)
    return stable_hash(
        {
            "kind": checked_kind,
            "template_version": STATE_TEMPLATE_VERSION,
            "text": text,
        }
    )


def _turn_spec(raw_initial_text: str, step_index: int) -> tuple[PhaseContextSpec, str]:
    from skillev.policy.phase_context import PhaseContextSpec

    spec, body = PhaseContextSpec.split(context_for_turn(raw_initial_text, step_index))
    if spec is None:
        raise ValueError("sigma mode requires a phase-context H0")
    return spec, body


def _state_view_text(
    raw_initial_text: str,
    state: SharedState,
    step_index: int,
    *,
    phase: Literal["reasoning", "forward", "hindsight"],
    instruction: str,
    current: str | None = None,
    candidates: Sequence[str] | None = None,
) -> str:
    from skillev.policy.state_view import (
        declares_chain_order,
        forward_draft_view,
        render_state_view,
        state_payload,
    )

    spec, body = _turn_spec(raw_initial_text, step_index)
    return render_state_view(
        spec,
        state_payload(
            body,
            state,
            phase=phase,
            instruction=instruction,
            current=None if current is None else forward_draft_view(spec, current),
            candidates=candidates,
            chain_order=declares_chain_order(spec),
        ),
    )


def _sigma_state(
    raw_initial_text: str, previous_steps: Sequence[StepLike], step_index: int
) -> SharedState:
    from .quotient import sigma

    state = sigma(raw_initial_text, previous_steps)
    if state.rank != step_index - 1:
        raise AssertionError("sigma-mode rank must equal step_index - 1")
    return state


def render_reasoning_prefix(
    initial_text: str,
    previous_steps: tuple[TrajectoryStep, ...],
    step_index: int,
) -> RenderedReasoningPrompt:
    _require_previous_steps(previous_steps, step_index)
    state = _sigma_state(initial_text, previous_steps, step_index)
    text = _state_view_text(
        initial_text,
        state,
        step_index,
        phase="reasoning",
        instruction=_REASONING_GENERATION_REMINDER,
    )
    return RenderedReasoningPrompt(
        step_index=step_index,
        text=text,
        prompt_hash=stable_hash(
            {
                "step_index": step_index,
                "template_version": STATE_TEMPLATE_VERSION,
                "text": text,
            }
        ),
    )


def render_forward_prefix_from_parts(
    initial_text: str,
    previous_steps: tuple[TrajectoryStep, ...],
    step_index: int,
    reasoning_text: str,
) -> RenderedPrefix:
    _require_previous_steps(previous_steps, step_index)
    state = _sigma_state(initial_text, previous_steps, step_index)
    text = _state_view_text(
        initial_text,
        state,
        step_index,
        phase="forward",
        instruction=native_action_reminder() + "\n",
        current=reasoning_text,
    )
    return RenderedPrefix(
        kind="forward",
        step_index=step_index,
        text=text,
        prefix_hash=prefix_content_hash(kind="forward", text=text),
    )


def render_forward_prefix(
    initial_text: str,
    steps: tuple[TrajectoryStep, ...],
    step_index: int,
) -> RenderedPrefix:
    current_step = _step_at(steps, step_index)
    return render_forward_prefix_from_parts(
        initial_text,
        steps[: step_index - 1],
        step_index,
        current_step.reasoning_text,
    )


def render_in_edge_hindsight_prefix(
    initial_text: str, steps_through_t: Sequence[StepLike]
) -> RenderedPrefix:
    from .quotient import in_edges, sigma

    t = len(steps_through_t)
    if t < 1:
        raise ValueError("in-edge hindsight needs at least one committed event")
    state = sigma(initial_text, steps_through_t)
    edges = in_edges(initial_text, state)
    text = _state_view_text(
        initial_text,
        state,
        t,
        phase="hindsight",
        instruction=IN_EDGE_HINDSIGHT_CHOICE_INSTRUCTION,
        candidates=[edge.call_text() for edge in edges] if len(edges) > 1 else None,
    )
    return RenderedPrefix(
        kind="in-edge-hindsight",
        step_index=t,
        text=text,
        prefix_hash=prefix_content_hash(kind="in-edge-hindsight", text=text),
    )


@dataclass(frozen=True, slots=True)
class InEdgeScoringInputs:
    prefix: RenderedPrefix
    candidate_texts: tuple[str, ...]
    actual_index: int
    predecessor_keys: tuple[str, ...]
    state_key: str
    bridge: str
    reasoning_text: str

    @property
    def actual_event_text(self) -> str:
        return self.candidate_texts[self.actual_index]


def in_edge_scoring_inputs(
    initial_text: str, steps: Sequence[TrajectoryStep], t: int
) -> InEdgeScoringInputs:
    from .quotient import (
        actual_in_edge_index,
        in_edges,
        legal_events_at,
        parse_step_event,
        sigma,
    )

    if type(t) is not int or not 1 <= t <= len(steps):
        raise ValueError("t must index a recorded step (one-based)")
    previous = sigma(initial_text, steps[: t - 1])
    current = sigma(initial_text, steps[:t])
    edges = in_edges(initial_text, current)
    event = parse_step_event(legal_events_at(initial_text, previous), steps[t - 1].action_text)
    actual = actual_in_edge_index(initial_text, previous, current, event, edges)
    return InEdgeScoringInputs(
        prefix=render_in_edge_hindsight_prefix(initial_text, steps[:t]),
        candidate_texts=tuple(edge.call_text() for edge in edges),
        actual_index=actual,
        predecessor_keys=tuple(edge.predecessor_key for edge in edges),
        state_key=current.key(),
        bridge=REASONING_POSTERIOR_BRIDGE,
        reasoning_text=steps[t - 1].reasoning_text,
    )
