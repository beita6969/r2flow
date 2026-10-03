from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

from skillev.contracts.answer_writer import INTEGER_ANSWER_REGEX, AnswerWriterPrompt
from skillev.contracts.reasoning_call_line import plain_call, plain_call_view
from skillev.task_semantic_guidance import (
    HEALTHBENCH_AUDITED_TASK_SENTENCE,
    HOTPOT_AUDITED_ANSWER_FORMAT,
    MBPP_PLUS_TASK_SEMANTICS,
)

if TYPE_CHECKING:
    from skillev.contracts.ttb_trajectory import TrajectoryStep

    from .types import RolloutTask

ANSWER_WRITER_PREAMBLE: Final = (
    "You write the final answer to a task. The input gives the task, the calls a supervisor made "
    "with their results, and the supervisor's notes from the step where it chose to answer. "
    "Write the answer that the notes and results reach; if they reach none, answer the task from "
    "the input. Reply with the final answer only."
)
ANSWER_WRITER_FORMATS: Final[Mapping[str, str]] = {
    "hotpotqa": HOTPOT_AUDITED_ANSWER_FORMAT,
    "triviaqa": "Return the short answer itself, rather than a sentence restating the question.",
    "aime-2026": "The answer is one integer from 0 through 999, written as plain digits.",
    "mbpp-plus": MBPP_PLUS_TASK_SEMANTICS,
    "healthbench": HEALTHBENCH_AUDITED_TASK_SENTENCE
    + " The reply is shown to the user exactly as written.",
}


def answer_writer_domain(task: RolloutTask) -> str:
    context = task.public_context
    domain = context.get("benchmark_id") if isinstance(context, Mapping) else None
    if not isinstance(domain, str) or domain not in ANSWER_WRITER_FORMATS:
        raise ValueError("the answer writer declares no answer format for this domain")
    return domain


def answer_writer_regex(domain: str) -> str | None:
    if domain not in ANSWER_WRITER_FORMATS:
        raise ValueError("the answer writer declares no answer format for this domain")
    return INTEGER_ANSWER_REGEX if domain == "aime-2026" else None


def answer_writer_prompt(
    *,
    initial_text: str,
    previous_steps: Sequence[TrajectoryStep],
    step_index: int,
    reasoning_text: str,
    task: RolloutTask,
) -> AnswerWriterPrompt:
    from skillev.policy.phase_context import PhaseContextSpec
    from skillev.policy.skill_availability import context_for_turn
    from skillev.policy.state_view import (
        _CHRONOLOGICAL_BOUNDARY,
        STATE_VIEW_BOUNDARY,
        declares_chain_order,
        event_block,
    )
    from skillev.scoring.quotient import sigma

    if len(previous_steps) != step_index - 1:
        raise ValueError("the answering turn follows exactly the previous steps")
    spec, _ = PhaseContextSpec.split(context_for_turn(initial_text, step_index))
    if spec is None:
        raise ValueError("the answer writer requires a phase-context H0")
    state = sigma(initial_text, tuple(previous_steps))
    canonical = not declares_chain_order(spec)
    domain = answer_writer_domain(task)
    return AnswerWriterPrompt(
        system=ANSWER_WRITER_PREAMBLE + "\n\n" + ANSWER_WRITER_FORMATS[domain],
        task=task.query,
        order=STATE_VIEW_BOUNDARY if canonical else _CHRONOLOGICAL_BOUNDARY,
        calls=tuple(
            plain_call_view(event_block(plain_call(item.event.u, item.event.args), item.output))
            for item in state.events
        ),
        notes=plain_call_view(reasoning_text),
    )


__all__ = [
    "ANSWER_WRITER_FORMATS",
    "ANSWER_WRITER_PREAMBLE",
    "answer_writer_domain",
    "answer_writer_prompt",
    "answer_writer_regex",
]
