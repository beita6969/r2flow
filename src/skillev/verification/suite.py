from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

from skillev.contracts import JsonValue
from skillev.contracts.answer_writer import written_answer
from skillev.contracts.canonical import stable_hash
from skillev.contracts.ttb_trajectory import TrajectoryStep
from skillev.contracts.verifier_record import (
    ALFWORLD_ACT,
    ALFWORLD_SKILL_ADVICE,
    CALL_SCHEMA,
    EXECUTION_STATUS,
    GATE_ELIGIBILITY_RULE,
    MBPP_PUBLIC_ASSERTS,
    QA_GROUNDING,
    QA_GROUNDING_SUBMIT,
    REFERENCE_AGREEMENT,
    REFERENCE_AGREEMENT_SUBMIT,
    REGISTERED_VERIFIERS,
    SKILL_FUNCTION,
    SOFT_ONLY_DOMAINS,
    SUBMIT_FUNCTION,
    VERIFIER_COMBINATION_RULE,
    VERIFIER_RECORD_FORMAT,
    VERIFIER_Z_FORMAT,
    EventClass,
    EventLabel,
    VerifierComponent,
    VerifierRecord,
    component,
    pre_invocation_context,
)
from skillev.evolution.task_features import transferable_family
from skillev.policy.event_grammar import (
    EVENT_GRAMMAR_VERSION,
    EventGrammarSpec,
    EventParseError,
    FunctionSpec,
    parse_event_call,
)

from .alfworld import act_outcome
from .code_asserts import (
    PublicAssertBackend,
    PublicAssertRequest,
    extract_code,
    public_asserts,
)
from .qa_grounding import distractor_documents, grounding_outcome, submit_grounding_outcome
from .reference_agreement import (
    REFERENCE_DOMAINS,
    REFERENCE_PROMPT_VERSION,
    ReferenceAnswer,
    ReferenceAnswerBackend,
    compare_with_reference,
    reference_identity,
)

if TYPE_CHECKING:
    from skillev.rollout.artifact import RolloutArtifact

SUITE_ID: Final = "r2flow-verifier-suite@2"
SUBMIT_QUESTION_RULE: Final = "submit-question=public-task-query@1"
BUDGET_POLICY: Final = "verify-all@1"
SUITE_DOMAINS: Final = frozenset(
    {"aime-2026", "alfworld", "healthbench", "hotpotqa", "mbpp-plus", "triviaqa"}
)
CLOSED_BOOK_DOMAINS: Final = frozenset({"triviaqa"})
OPEN_PASSAGE: Final = "open_passage"
ALFWORLD_ACT_FUNCTION: Final = "act"


class LegalSetView(Protocol):
    @property
    def functions(self) -> tuple[FunctionSpec, ...]: ...


@dataclass(frozen=True, slots=True)
class VerifiableStep:
    index: int
    action_text: str
    observation: JsonValue
    observation_status: str
    legal_functions: tuple[FunctionSpec, ...]
    forward_prefix_token_count: int


@dataclass(frozen=True, slots=True)
class VerificationInput:
    trajectory_id: str
    task_id: str
    domain: str
    query: str
    steps: tuple[VerifiableStep, ...]

    def __post_init__(self) -> None:
        if [step.index for step in self.steps] != list(range(1, len(self.steps) + 1)):
            raise ValueError("verifiable steps must be indexed 1..T")


def verification_input(
    *,
    trajectory_id: str,
    task_id: str,
    domain: str,
    query: str,
    steps: Sequence[TrajectoryStep],
    legal_sets: Sequence[LegalSetView],
    forward_prefix_token_counts: Sequence[int],
) -> VerificationInput:
    if not len(steps) == len(legal_sets) == len(forward_prefix_token_counts):
        raise ValueError("one legal set and one prefix token count per step")
    return VerificationInput(
        trajectory_id=trajectory_id,
        task_id=task_id,
        domain=domain,
        query=query,
        steps=tuple(
            VerifiableStep(
                index=step.index,
                action_text=step.action_text,
                observation=json.loads(step.observation_text),
                observation_status=step.observation_status,
                legal_functions=tuple(legal.functions),
                forward_prefix_token_count=count,
            )
            for step, legal, count in zip(
                steps, legal_sets, forward_prefix_token_counts, strict=True
            )
        ),
    )


def verification_input_from_artifact(
    artifact: RolloutArtifact,
    *,
    domain: str,
    query: str,
    legal_sets: Sequence[LegalSetView],
    forward_prefix_token_counts: Sequence[int],
) -> VerificationInput:
    record = artifact.record
    return verification_input(
        trajectory_id=record.trajectory_id,
        task_id=artifact.manifest.task_id,
        domain=domain,
        query=query,
        steps=record.steps,
        legal_sets=legal_sets,
        forward_prefix_token_counts=forward_prefix_token_counts,
    )


def recorded_verification_input(artifact: RolloutArtifact) -> VerificationInput:
    inputs = artifact.verifier_inputs
    if inputs is None:
        raise ValueError("the artifact carries no verifier inputs")
    return verification_input_from_artifact(
        artifact,
        domain=inputs.domain,
        query=artifact.initial_context.contract.query,
        legal_sets=inputs.legal_event_sets,
        forward_prefix_token_counts=inputs.forward_prefix_token_counts,
    )


def suite_identity() -> dict[str, JsonValue]:
    return {
        "budget": BUDGET_POLICY,
        "combination": VERIFIER_COMBINATION_RULE,
        "domains": list[JsonValue](sorted(SUITE_DOMAINS)),
        "alfworld_advice_actionable": ADVICE_ACTIONABLE_RULE,
        "alfworld_advice_match": ADVICE_MATCH_RULE,
        "gate_eligibility": GATE_ELIGIBILITY_RULE,
        "record": VERIFIER_RECORD_FORMAT,
        "reference_agreement": reference_identity(REFERENCE_PROMPT_VERSION),
        "soft_only_domains": list[JsonValue](sorted(SOFT_ONLY_DOMAINS)),
        "suite": SUITE_ID,
        "verifiers": {
            key: {"confidence": value[0], "evidential": value[1]}
            for key, value in sorted(REGISTERED_VERIFIERS.items())
        },
        "z": VERIFIER_Z_FORMAT,
        "submit_evidence": {
            "aime-2026": REFERENCE_AGREEMENT_SUBMIT,
            "hotpotqa": QA_GROUNDING_SUBMIT,
            "question": SUBMIT_QUESTION_RULE,
            "triviaqa": REFERENCE_AGREEMENT_SUBMIT,
        },
    }


def suite_identity_hash() -> str:
    return stable_hash(suite_identity())


def _parse(step: VerifiableStep) -> tuple[str, dict[str, str]] | None:
    if not step.legal_functions:
        return None
    spec = EventGrammarSpec(EVENT_GRAMMAR_VERSION, step.legal_functions, 1, 0)
    try:
        return parse_event_call(spec, step.action_text)
    except EventParseError:
        return None


ADVICE_MATCH_RULE: Final = "advice-match=word-bounded-phrase@1"
ADVICE_ACTIONABLE_RULE: Final = "advice-actionable-commands@1"
NON_ACTIONABLE_COMMANDS: Final = frozenset({"look", "inventory", "help"})


def _normalized_command(text: str) -> str:
    return " ".join(text.lower().split())


def _advice_names(output: str, normalized_command: str) -> bool:
    pattern = r"(?<![a-z0-9])" + re.escape(normalized_command) + r"(?![a-z0-9])"
    return re.search(pattern, _normalized_command(output)) is not None


def _admissible(step: VerifiableStep, function: str) -> tuple[str, ...]:
    return next(
        (
            tuple(param.values)
            for fn in step.legal_functions
            if fn.name == function
            for param in fn.params
            if param.name == "command"
        ),
        (),
    )


def _projected_python(answer: str) -> str | None:
    from skillev.evaluation.owner_final import project_owner_final
    from skillev.evaluation.terminal_projection import TerminalMode

    return project_owner_final(TerminalMode.PYTHON_SOURCE, answer)


def _submitted_answer(step: VerifiableStep, arguments: Mapping[str, str]) -> str:
    written = written_answer(step.observation)
    return arguments.get("answer", "") if written is None else written


def _executor_output(step: VerifiableStep) -> str | None:
    value = step.observation
    output = value.get("output") if isinstance(value, dict) else None
    return output if isinstance(output, str) else None


def _execution_component(step: VerifiableStep, event_class: EventClass) -> VerifierComponent:
    value = step.observation
    checked = json.dumps(value, sort_keys=True, ensure_ascii=False)
    if step.observation_status != "success" or (isinstance(value, dict) and "error" in value):
        return component(EXECUTION_STATUS, False, "execution-error", checked)
    if event_class is EventClass.SKILL:
        output = _executor_output(step)
        if output is None or not output.strip():
            return component(EXECUTION_STATUS, False, "empty-output", checked)
        if isinstance(value, dict) and value.get("output_truncated") is True:
            return component(EXECUTION_STATUS, False, "truncated-output", checked)
    elif value in (None, "", {}, []):
        return component(EXECUTION_STATUS, False, "empty-output", checked)
    return component(EXECUTION_STATUS, True, "ok", checked)


def _opened_passages(steps: Sequence[VerifiableStep], before: int) -> list[str]:
    passages: list[str] = []
    for step in steps[: before - 1]:
        parsed = _parse(step)
        value = step.observation
        text = value.get("text") if isinstance(value, dict) else None
        if (
            parsed is not None
            and parsed[0] == OPEN_PASSAGE
            and step.observation_status == "success"
            and isinstance(text, str)
        ):
            passages.append(text)
    return passages


class VerifierSuite:
    def __init__(
        self,
        *,
        domain: str,
        code_backend: PublicAssertBackend | None = None,
        reference_backend: ReferenceAnswerBackend | None = None,
    ) -> None:
        if domain not in SUITE_DOMAINS:
            raise ValueError(f"{domain!r} is outside {SUITE_ID}")
        self.suite = SUITE_ID
        if (code_backend is not None) != (domain == "mbpp-plus"):
            raise ValueError("exactly the MBPP+ suite takes a public-assert backend")
        if reference_backend is not None and domain not in REFERENCE_DOMAINS:
            raise ValueError("only the TriviaQA and AIME suites take a reference backend")
        self.domain = domain
        self._code_backend = code_backend
        self._reference_backend = reference_backend
        family = transferable_family(domain)
        if family is None:
            raise ValueError(f"{domain!r} has no transferable task family")
        self.context_class = family

    async def verify(self, value: VerificationInput) -> tuple[VerifierRecord, ...]:
        if value.domain != self.domain:
            raise ValueError("verification input belongs to another domain")
        references = await self._prefetch_references(value)
        records = []
        for step in value.steps:
            records.append(await self._verify_step(value, step, references))
        return tuple(records)

    async def _prefetch_references(self, value: VerificationInput) -> dict[str, ReferenceAnswer]:
        backend = self._reference_backend
        if backend is None:
            return {}
        questions: dict[str, None] = {}
        submit_question = self._submit_question(value)
        for step in value.steps:
            parsed = _parse(step)
            output = _executor_output(step)
            if parsed is not None and parsed[0] == SKILL_FUNCTION and output and output.strip():
                questions.setdefault(parsed[1].get("input", ""), None)
            if (
                submit_question is not None
                and parsed is not None
                and parsed[0] == SUBMIT_FUNCTION
                and _submitted_answer(step, parsed[1]).strip()
            ):
                questions.setdefault(submit_question, None)
        answers = await asyncio.gather(
            *(backend.reference(self.domain, question) for question in questions)
        )
        return dict(zip(questions, answers, strict=True))

    async def _verify_step(
        self,
        value: VerificationInput,
        step: VerifiableStep,
        references: Mapping[str, ReferenceAnswer],
    ) -> VerifierRecord:
        components: list[VerifierComponent] = []
        skipped: list[tuple[str, str]] = []
        parsed = _parse(step)
        if parsed is None:
            event = EventLabel.unparsed(step.action_text)
            components.append(component(CALL_SCHEMA, False, "not-a-legal-event", step.action_text))
        else:
            function, arguments = parsed
            event = EventLabel.from_call(function, arguments)
            components.append(component(CALL_SCHEMA, True, "ok", step.action_text))
            if event.event_class in (EventClass.SKILL, EventClass.TOOL):
                components.append(_execution_component(step, event.event_class))
            if (
                event.event_class is EventClass.TOOL
                and function == ALFWORLD_ACT_FUNCTION
                and self.domain == "alfworld"
            ):
                passed, reason = act_outcome(
                    arguments.get("command", ""),
                    _admissible(step, function),
                    step.observation,
                    step.observation_status,
                )
                components.append(
                    component(ALFWORLD_ACT, passed, reason, arguments.get("command", ""))
                )
            if event.event_class is EventClass.SKILL and self.domain == "alfworld":
                self._alfworld_advice(value, step, components, skipped)
            if event.event_class is EventClass.SKILL and self.domain in ("hotpotqa", "triviaqa"):
                if self.domain in CLOSED_BOOK_DOMAINS:
                    skipped.append((QA_GROUNDING, "closed-book"))
                else:
                    output = _executor_output(step) or ""
                    grounded, reason = grounding_outcome(
                        output, _opened_passages(value.steps, step.index)
                    )
                    if grounded is None:
                        skipped.append((QA_GROUNDING, reason))
                    else:
                        components.append(component(QA_GROUNDING, grounded, reason, output))
            if event.event_class is EventClass.SKILL and self.domain in REFERENCE_DOMAINS:
                self._reference(step, arguments, references, components, skipped)
            if event.event_class is EventClass.SUBMIT:
                if self.domain in REFERENCE_DOMAINS:
                    self._reference_submit(value, step, arguments, references, components, skipped)
                elif self.domain == "hotpotqa":
                    self._grounding_submit(value, step, arguments, components, skipped)
            if self.domain == "mbpp-plus" and event.event_class in (
                EventClass.SKILL,
                EventClass.SUBMIT,
            ):
                await self._mbpp(value, step, event, arguments, components, skipped)
        z = pre_invocation_context(
            context_class=self.context_class,
            step_index=step.index,
            previous_status=(
                None if step.index == 1 else value.steps[step.index - 2].observation_status
            ),
            forward_prefix_token_count=step.forward_prefix_token_count,
        )
        return VerifierRecord(
            trajectory_id=value.trajectory_id,
            step_index=step.index,
            task_id=value.task_id,
            domain=self.domain,
            event=event,
            z=z,
            components=tuple(sorted(components, key=lambda item: item.verifier_id)),
            not_applicable=tuple(skipped),
            post_invocation_status=step.observation_status,
        )

    @staticmethod
    def _alfworld_advice(
        value: VerificationInput,
        step: VerifiableStep,
        components: list[VerifierComponent],
        skipped: list[tuple[str, str]],
    ) -> None:
        output = _executor_output(step)
        if output is None or not output.strip():
            skipped.append((ALFWORLD_SKILL_ADVICE, "no-output"))
            return
        admissible = _admissible(step, ALFWORLD_ACT_FUNCTION)
        if not admissible:
            skipped.append((ALFWORLD_SKILL_ADVICE, "no-admissible-set"))
            return
        advised = {
            normalized
            for command in admissible
            if (normalized := _normalized_command(command)) not in NON_ACTIONABLE_COMMANDS
            and _advice_names(output, normalized)
        }
        if not advised:
            components.append(
                component(ALFWORLD_SKILL_ADVICE, False, "no-admissible-command", output)
            )
            return
        if step.index >= len(value.steps):
            skipped.append((ALFWORLD_SKILL_ADVICE, "no-next-event"))
            return
        nxt = value.steps[step.index]
        parsed = _parse(nxt)
        if parsed is None or parsed[0] != ALFWORLD_ACT_FUNCTION:
            skipped.append((ALFWORLD_SKILL_ADVICE, "next-not-act"))
            return
        command = parsed[1].get("command", "")
        if _normalized_command(command) not in advised:
            skipped.append((ALFWORLD_SKILL_ADVICE, "advice-not-followed"))
            return
        passed, reason = act_outcome(
            command,
            _admissible(nxt, ALFWORLD_ACT_FUNCTION),
            nxt.observation,
            nxt.observation_status,
        )
        components.append(component(ALFWORLD_SKILL_ADVICE, passed, reason, command))

    def _reference(
        self,
        step: VerifiableStep,
        arguments: dict[str, str],
        references: Mapping[str, ReferenceAnswer],
        components: list[VerifierComponent],
        skipped: list[tuple[str, str]],
    ) -> None:
        if self._reference_backend is None:
            skipped.append((REFERENCE_AGREEMENT, "disabled"))
            return
        output = _executor_output(step)
        if output is None or not output.strip():
            skipped.append((REFERENCE_AGREEMENT, "no-output"))
            return
        reference = references[arguments.get("input", "")]
        if reference.status == "unavailable":
            skipped.append((REFERENCE_AGREEMENT, "reference-unavailable"))
            return
        if reference.status == "not-applicable" or reference.value is None:
            skipped.append((REFERENCE_AGREEMENT, "reference-not-applicable"))
            return
        passed, reason = compare_with_reference(self.domain, output, reference.value)
        if passed is None:
            skipped.append((REFERENCE_AGREEMENT, reason))
            return
        components.append(component(REFERENCE_AGREEMENT, passed, reason, output))

    def _submit_question(self, value: VerificationInput) -> str | None:
        question = value.query.strip()
        if not question or (self.domain == "triviaqa" and "\n" in question):
            return None
        return question

    def _reference_submit(
        self,
        value: VerificationInput,
        step: VerifiableStep,
        arguments: dict[str, str],
        references: Mapping[str, ReferenceAnswer],
        components: list[VerifierComponent],
        skipped: list[tuple[str, str]],
    ) -> None:
        if self._reference_backend is None:
            skipped.append((REFERENCE_AGREEMENT_SUBMIT, "disabled"))
            return
        answer = _submitted_answer(step, arguments)
        if not answer.strip():
            skipped.append((REFERENCE_AGREEMENT_SUBMIT, "no-answer"))
            return
        question = self._submit_question(value)
        if question is None:
            reason = "query-not-one-line" if value.query.strip() else "no-question"
            skipped.append((REFERENCE_AGREEMENT_SUBMIT, reason))
            return
        reference = references[question]
        if reference.status == "unavailable":
            skipped.append((REFERENCE_AGREEMENT_SUBMIT, "reference-unavailable"))
            return
        if reference.status == "not-applicable" or reference.value is None:
            skipped.append((REFERENCE_AGREEMENT_SUBMIT, "reference-not-applicable"))
            return
        passed, reason = compare_with_reference(self.domain, answer, reference.value)
        if passed is None:
            skipped.append((REFERENCE_AGREEMENT_SUBMIT, reason))
            return
        components.append(component(REFERENCE_AGREEMENT_SUBMIT, passed, reason, answer))

    @staticmethod
    def _grounding_submit(
        value: VerificationInput,
        step: VerifiableStep,
        arguments: dict[str, str],
        components: list[VerifierComponent],
        skipped: list[tuple[str, str]],
    ) -> None:
        answer = _submitted_answer(step, arguments)
        documents = distractor_documents(value.query)
        contexts = ([documents] if documents is not None else []) + _opened_passages(
            value.steps, step.index
        )
        grounded, reason = submit_grounding_outcome(answer, contexts)
        if grounded is None:
            skipped.append((QA_GROUNDING_SUBMIT, reason))
        else:
            components.append(component(QA_GROUNDING_SUBMIT, grounded, reason, answer))

    async def _mbpp(
        self,
        value: VerificationInput,
        step: VerifiableStep,
        event: EventLabel,
        arguments: dict[str, str],
        components: list[VerifierComponent],
        skipped: list[tuple[str, str]],
    ) -> None:
        asserts = public_asserts(value.query)
        written = written_answer(step.observation)
        code = (
            (arguments.get("answer") if written is None else _projected_python(written))
            if event.event_class is EventClass.SUBMIT
            else extract_code(_executor_output(step) or "")
        )
        if not asserts:
            skipped.append((MBPP_PUBLIC_ASSERTS, "no-public-assert"))
            return
        if code is None or not code.strip():
            skipped.append((MBPP_PUBLIC_ASSERTS, "no-code"))
            return
        assert self._code_backend is not None
        result = await self._code_backend.run(PublicAssertRequest(code, asserts))
        reason = {
            "pass": "asserts-passed",
            "fail": "assert-failed",
            "timeout": "timeout",
            "error": "runtime-error",
        }[result.status]
        components.append(component(MBPP_PUBLIC_ASSERTS, result.status == "pass", reason, code))


__all__ = [
    "ALFWORLD_ACT_FUNCTION",
    "BUDGET_POLICY",
    "SUBMIT_QUESTION_RULE",
    "SUITE_DOMAINS",
    "SUITE_ID",
    "LegalSetView",
    "VerifiableStep",
    "VerificationInput",
    "VerifierSuite",
    "recorded_verification_input",
    "suite_identity",
    "suite_identity_hash",
    "verification_input",
    "verification_input_from_artifact",
]
