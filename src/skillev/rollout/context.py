from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final, Protocol

from skillev.contracts import (
    InitialContext,
    JsonValue,
    canonical_json,
    normalize_json,
)
from skillev.contracts.action_wire import NATIVE_EVENT_CALL_WIRE
from skillev.contracts.final_turn import FINAL_TURN_COMPLETION_RULES
from skillev.contracts.native_call_format import NATIVE_XML_TASK_SENTENCE
from skillev.contracts.reasoning_call_line import (
    REASONING_CALL_LINE,
    REASONING_CALL_LINE_TASK_SENTENCE,
)
from skillev.contracts.skill_call_budget import SKILL_CALL_BUDGET
from skillev.contracts.skill_exposure import SKILL_EXPOSURE
from skillev.contracts.skill_visibility import SKILL_VISIBILITY_RULES
from skillev.contracts.state_map import SIGMA_TRACE_QUOTIENT
from skillev.policy.interface import (
    INPUT_WINDOW_META_KEY,
    ROLLOUT_SOURCE_MESSAGES_BEGIN,
    ROLLOUT_SOURCE_MESSAGES_END,
    ModelInputWindow,
    PhaseContextSpec,
    encode_rollout_prompt,
)
from skillev.policy.state_view import STATE_VIEW_VERSION
from skillev.runtime.full_skill_context import FullRetrievedSkillContext
from skillev.scoring import assembled_context_hash
from skillev.task_semantic_guidance import (
    TASK_SEMANTIC_GUIDANCE,
    TRAINING_PUBLIC_INPUT,
    TRAINING_SUBMISSION_INSTRUCTION,
    WRITER_SUBMISSION_INSTRUCTION,
    phase_deliverable,
    public_task_semantics,
    writer_deliverable,
)

from .action_contract import ActionContract
from .action_surface import (
    TerminalMode,
)
from .generator import RolloutTokenizerProtocol
from .types import DecodingSnapshot, RolloutTask

INITIAL_CONTEXT_FORMAT_VERSION: Final = "ttb-initial-context@6"


def _task_input_profile(task: RolloutTask) -> str:
    context = task.public_context
    profile = (
        context.get("input_profile", TRAINING_PUBLIC_INPUT)
        if isinstance(context, dict)
        else TRAINING_PUBLIC_INPUT
    )
    if not isinstance(profile, str):
        raise ValueError("public input profile must be declared text")
    return profile


def _task_benchmark_id(task: RolloutTask) -> str | None:
    context = task.public_context
    if not isinstance(context, dict):
        return None
    benchmark_id = context.get("benchmark_id")
    return benchmark_id if type(benchmark_id) is str and benchmark_id else None


def _require_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be non-empty text")
    return value


@dataclass(frozen=True, slots=True)
class AssembledInitialContext:
    text: str
    contract: InitialContext

    def __post_init__(self) -> None:
        _require_text(self.text, field="initial context text")
        if not isinstance(self.contract, InitialContext):
            raise ValueError("contract must be an InitialContext")
        if assembled_context_hash(self.text) != self.contract.assembled_hash:
            raise ValueError("initial context text does not match its assembled hash")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "contract": self.contract.to_value(),
            "text": self.text,
        }

    @classmethod
    def from_value(cls, value: object) -> AssembledInitialContext:
        if not isinstance(value, dict):
            raise ValueError("assembled initial context must be a JSON object")
        normalized = normalize_json(value)
        if not isinstance(normalized, dict) or normalized != value:
            raise ValueError("assembled initial context must be a JSON object")
        if set(normalized) != {"contract", "text"}:
            raise ValueError("assembled initial context has an incompatible field set")
        text = normalized["text"]
        if type(text) is not str:
            raise ValueError("assembled initial context text must be text")
        return cls(
            text=text,
            contract=InitialContext.from_value(normalized["contract"]),
        )


class InitialContextAssembler(Protocol):
    @property
    def assembler_version(self) -> str: ...

    def assemble(
        self,
        *,
        task: RolloutTask,
        retrieved_skills: tuple[FullRetrievedSkillContext, ...],
        active_skill_ids: tuple[str, ...],
        library_version: str,
        tokenizer: RolloutTokenizerProtocol,
        decoding: DecodingSnapshot | None = None,
    ) -> AssembledInitialContext: ...


class CanonicalInitialContextAssembler:
    def __init__(
        self,
        *,
        maximum_h0_tokens: int,
        state_map: str,
        final_turn_completion: str,
        skill_visibility: str,
        input_window: ModelInputWindow | None = None,
        skill_call_budget_by_domain: tuple[tuple[str, int], ...] = (),
        reasoning_call_line_domains: tuple[str, ...] = (),
    ) -> None:
        call_line = tuple(reasoning_call_line_domains)
        if call_line and (
            any(type(domain) is not str or not domain for domain in call_line)
            or list(call_line) != sorted(set(call_line))
        ):
            raise ValueError(f"{REASONING_CALL_LINE} requires unique sorted domains")
        self._reasoning_call_line_domains = call_line
        if skill_visibility not in SKILL_VISIBILITY_RULES:
            raise ValueError("skill visibility nonempty-only@1 is the only skill visibility rule")
        self._skill_visibility = skill_visibility
        budgets = tuple(skill_call_budget_by_domain)
        if budgets and (
            any(
                not isinstance(item, tuple)
                or len(item) != 2
                or type(item[0]) is not str
                or type(item[1]) is not int
                or item[1] < 1
                for item in budgets
            )
            or [item[0] for item in budgets] != sorted({item[0] for item in budgets})
        ):
            raise ValueError("skill-call-budget@1 requires unique sorted positive domain budgets")
        self._skill_call_budget_by_domain = budgets
        if final_turn_completion not in FINAL_TURN_COMPLETION_RULES:
            raise ValueError("final-turn-submit@1 is the only final-turn completion rule")
        self._final_turn_completion = final_turn_completion
        if state_map != SIGMA_TRACE_QUOTIENT:
            raise ValueError("state_map must be a declared state map")
        self._state_map = state_map
        if type(maximum_h0_tokens) is not int or maximum_h0_tokens < 1:
            raise ValueError("maximum_h0_tokens must be positive")
        self._maximum_h0_tokens = maximum_h0_tokens
        self._input_window = input_window

    @property
    def assembler_version(self) -> str:
        return (
            "phase-context@3/"
            + NATIVE_EVENT_CALL_WIRE
            + "/"
            + SKILL_EXPOSURE
            + "/public-action-semantics@1"
            + f"/{TASK_SEMANTIC_GUIDANCE}/hotpot=0"
            + f"/{self._state_map}/{STATE_VIEW_VERSION}"
            + f"/{self._final_turn_completion}"
            + (
                f"/{SKILL_CALL_BUDGET}="
                + ",".join(f"{d}:{k}" for d, k in self._skill_call_budget_by_domain)
                if self._skill_call_budget_by_domain
                else ""
            )
            + f"/skill-visibility={self._skill_visibility}"
            + (
                f"/{REASONING_CALL_LINE}=" + ",".join(self._reasoning_call_line_domains)
                if self._reasoning_call_line_domains
                else ""
            )
        )

    @property
    def state_map(self) -> str:
        return self._state_map

    @property
    def reasoning_call_line_domains(self) -> tuple[str, ...]:
        return self._reasoning_call_line_domains

    @property
    def skill_visibility(self) -> str:
        return self._skill_visibility

    @property
    def final_turn_completion(self) -> str:
        return self._final_turn_completion

    @property
    def skill_call_budget_by_domain(self) -> tuple[tuple[str, int], ...]:
        return self._skill_call_budget_by_domain

    def assemble(
        self,
        *,
        task: RolloutTask,
        retrieved_skills: tuple[FullRetrievedSkillContext, ...],
        active_skill_ids: tuple[str, ...],
        library_version: str,
        tokenizer: RolloutTokenizerProtocol,
        decoding: DecodingSnapshot | None = None,
    ) -> AssembledInitialContext:
        if not isinstance(task, RolloutTask):
            raise ValueError("task must be a RolloutTask")
        if decoding is None:
            raise ValueError("token budget notice requires actual request decoding caps")
        if not isinstance(retrieved_skills, tuple):
            raise ValueError("retrieved_skills must be a tuple")
        if (
            not isinstance(active_skill_ids, tuple)
            or tuple(sorted(set(active_skill_ids))) != active_skill_ids
            or any(type(skill_id) is not str or not skill_id for skill_id in active_skill_ids)
        ):
            raise ValueError("active_skill_ids must be sorted unique non-empty text")
        if not isinstance(library_version, str) or not library_version.strip():
            raise ValueError("library_version must be non-empty text")
        if self._skill_visibility is not None:
            retrieved_skills = tuple(
                skill for skill in retrieved_skills if not is_empty_slot_skill(skill)
            )
        skill_ids: list[str] = []
        retrieval_inclusions: list[JsonValue] = []
        rendered_skills: list[str] = []
        for position, skill in enumerate(retrieved_skills, start=1):
            metadata = skill.metadata
            if metadata.skill_id in skill_ids:
                raise ValueError("retrieved skill IDs must be unique")
            if metadata.skill_id not in active_skill_ids:
                raise ValueError("retrieved skill is not active in the pinned library")
            skill_ids.append(metadata.skill_id)
            retrieval_inclusions.append(
                {
                    "inclusion_reason": skill.inclusion_reason.value,
                    "skill_id": metadata.skill_id,
                }
            )
            rendered_skills.append(render_skill_md_catalog_entry(skill, position=position))

        skill_section = SKILL_CATALOG_HEADER + "".join(rendered_skills) if rendered_skills else ""
        surface = task.action_surface
        benchmark = _task_benchmark_id(task)
        if surface is None or benchmark is None:
            raise ValueError(
                "shared task guidance requires an explicit public action surface/domain"
            )
        instructions = (public_task_semantics(benchmark, input_profile=_task_input_profile(task)),)
        if surface.completion is not None:
            instructions += (
                WRITER_SUBMISSION_INSTRUCTION
                if surface.completion_writer is not None
                else TRAINING_SUBMISSION_INSTRUCTION,
            )
        task = replace(
            task,
            action_surface=replace(surface, instructions=instructions),
        )
        call_line = _task_benchmark_id(task) in self._reasoning_call_line_domains
        task_sentence = REASONING_CALL_LINE_TASK_SENTENCE if call_line else NATIVE_XML_TASK_SENTENCE
        action_contract = ActionContract.freeze(
            task.action_surface,
            retrieved_skill_ids=tuple(skill_ids),
            active_skill_ids=active_skill_ids,
        )
        action_guidance = (
            "Reasoning drafts do not execute actions. The action phase calls one "
            "of the supplied functions using its declared parameters. "
            + task_sentence
            + (
                ""
                if not skill_ids
                else " Skill IDs are values for invoke_skill's skill_id parameter; "
                "input is the text the executor receives."
            )
        )
        action_guidance = "\n".join((*action_contract.render_native_semantics(), action_guidance))
        text = (
            f"### Query\n{task.query}\n"
            f"{_render_source_messages(task)}"
            f"{skill_section}"
            f"### Public Task Context\n{canonical_json(task.public_context)}\n"
            f"### Available Actions\n{action_guidance}\n"
        )
        public_state = (
            task.public_context.get("payload", {})
            if benchmark == "alfworld" and isinstance(task.public_context, dict)
            else None
        )
        if isinstance(public_state, dict) and isinstance(task.public_context, dict):
            native_limit = task.public_context.get("max_steps")
            if type(native_limit) is int:
                public_state = {**public_state, "max_steps": native_limit}
        if task.budget_profile is None:
            raise ValueError("sigma-mode H0 requires the task turn horizon")
        deliverable = phase_deliverable(benchmark)
        if task.action_surface is not None and task.action_surface.completion_writer:
            deliverable = writer_deliverable(deliverable)
        spec = PhaseContextSpec(
            state_map=self._state_map,
            deliverable=deliverable,
            reasoning_token_cap=decoding.max_reasoning_tokens,
            action_token_cap=decoding.max_action_tokens,
            tools_json=canonical_json(list(action_contract.to_native_tools())),
            action_contract_json=canonical_json(action_contract.to_scoring_metadata()),
            initial_public_state_json=canonical_json(public_state),
            max_turns=task.budget_profile.max_turns,
            final_turn_completion=self._final_turn_completion
            if action_contract.surface is not None
            and action_contract.surface.terminal_mode is TerminalMode.EXPLICIT_COMPLETION
            else None,
            skill_call_budget=dict(self._skill_call_budget_by_domain).get(benchmark)
            if skill_ids
            else None,
            skill_visibility=self._skill_visibility,
            reasoning_call_line=REASONING_CALL_LINE if call_line else None,
        )
        text = spec.wrap(text)
        token_count = len(encode_rollout_prompt(tokenizer, text))
        if token_count <= 0:
            raise ValueError("assembled initial context must encode to at least one token")
        if token_count > self._maximum_h0_tokens and self._input_window is None:
            raise ValueError("canonical initial context exceeds maximum_h0_tokens")
        meta: dict[str, JsonValue] = {
            "environment_id": task.environment_id,
            "context_id": task.context_id,
            "available_tools": list(task.available_tools),
            "format_version": INITIAL_CONTEXT_FORMAT_VERSION,
            "library_version": library_version,
            "retrieval_inclusions": retrieval_inclusions,
            "root_query_source": "rollout-task-query",
            "root_query_stable_across_episode": True,
            "root_query_token_count": len(tokenizer.encode(task.query)),
            "task_family": task.task_family,
            "task_id": task.task_id,
        }
        if isinstance(task.public_context, dict):
            reset_identity = task.public_context.get("environment_reset_identity")
            if isinstance(reset_identity, str) and reset_identity:
                meta["environment_reset_identity"] = reset_identity
        meta["task_semantic_guidance"] = TASK_SEMANTIC_GUIDANCE
        meta["task_semantic_input_profile"] = _task_input_profile(task)
        meta["hotpot_deliberation"] = False
        meta["public_action_semantics"] = "public-action-semantics@1"
        meta.update(
            {
                "action_wire": NATIVE_EVENT_CALL_WIRE,
                "action_contract": action_contract.to_scoring_metadata(),
                "skill_exposure": SKILL_EXPOSURE,
            }
        )
        contract = InitialContext(
            query=task.query,
            retrieved_skill_ids=tuple(skill_ids),
            active_skill_ids=active_skill_ids,
            meta={
                **meta,
                **(
                    {INPUT_WINDOW_META_KEY: self._input_window.to_value()}
                    if self._input_window
                    else {}
                ),
            },
            assembler_version=self.assembler_version,
            assembled_hash=assembled_context_hash(text),
            assembled_token_count=token_count,
        )
        return AssembledInitialContext(text=text, contract=contract)


SKILL_CATALOG_HEADER = (
    "### Available Skills\nEach skill is a procedure learned from earlier tasks of this kind; "
    "its full text is shown below. Follow a procedure in your own reasoning when it fits the "
    "task. A skill's reply format applies only when it runs on the frozen executor via "
    "invoke_skill(skill_id, input), whose output returns as an observation; your own answer "
    "keeps the format this task requires.\n"
)


_EMPTY_BODY_PROBE = "empty-slot-probe\n"


def skill_md_name_description(content: str) -> tuple[str, str]:
    from skillev.runtime.skill_md import parse_skill_md

    try:
        document = parse_skill_md(content)
    except ValueError:
        if not content.endswith("\n---\n\n"):
            raise
        document = parse_skill_md(content + _EMPTY_BODY_PROBE)
    return document.name, document.description


def is_empty_slot_skill(skill: FullRetrievedSkillContext) -> bool:
    content = skill.content
    if not content.endswith("\n---\n\n"):
        return False
    try:
        skill_md_name_description(content)
    except ValueError:
        return False
    return True


def render_skill_md_catalog_entry(skill: FullRetrievedSkillContext, *, position: int) -> str:
    from skillev.runtime.skill_md import parse_skill_md

    document = parse_skill_md(skill.content)
    if document.name != skill.metadata.skill_id:
        raise ValueError("SKILL.md name differs from the retrieved skill ID")
    entry = f"[{position}] {document.name}: {document.description}\n"
    if document.body.strip():
        entry += document.body.strip() + "\n"
    return entry


def _render_source_messages(task: RolloutTask) -> str:
    value = [message.to_value() for message in task.model_visible_messages]
    return ROLLOUT_SOURCE_MESSAGES_BEGIN + canonical_json(value) + ROLLOUT_SOURCE_MESSAGES_END
