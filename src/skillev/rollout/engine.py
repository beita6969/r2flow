from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NoReturn, Protocol

from skillev.contracts import JsonValue, StepR2FlowRecord, TrajectoryStep, stable_hash
from skillev.contracts.answer_writer import (
    EXECUTOR_ANSWER,
    AnswerWriterPrompt,
    declared_completion_writer,
)
from skillev.contracts.reasoning_call_line import REASONING_STOP_TEXT
from skillev.diagnostics.rollout_progress import current_progress, progress_stage
from skillev.diagnostics.rollout_trace import (
    NullRolloutTraceSink,
    RolloutTraceEvent,
    RolloutTraceSink,
    RolloutTraceStage,
)
from skillev.policy.interface import (
    THINKING_ROLLOUT_PROMPT_ENCODER_VERSION,
    ModelInputWindow,
    PhaseContextSpec,
    encode_policy_prompt,
    skill_available_from_turn,
)
from skillev.runtime import (
    ActionKind,
    ActionParseResult,
    ActionParseStatus,
    BoundedAgent,
    BoundedAgentState,
    BoundedAgentTurnRequest,
    BudgetLedger,
    BudgetReservation,
    BudgetSettlement,
    BudgetVector,
    EnvironmentSkillInvocationMismatchError,
    EventType,
    RuntimeEventEmitter,
)
from skillev.scoring import render_forward_prefix_from_parts, render_reasoning_prefix

from .action_contract import ActionContract
from .artifact import (
    ActionDraft,
    CompletedStepDraft,
    R2FlowStepInputs,
    ReasoningDraft,
    RolloutArtifact,
    RolloutManifest,
    finalize_trajectory_record,
    materialize_trajectory_step,
)
from .codec import decode_action_segment, decode_reasoning_segment
from .context import AssembledInitialContext, InitialContextAssembler
from .environment import (
    NoSubmissionReason,
    NoTerminalSubmission,
    SubmittedTerminalValue,
    TerminalEvaluationInput,
    TerminalEvaluationRequest,
    TerminalEvaluator,
)
from .errors import (
    RolloutBoundaryError,
    RolloutInfrastructureError,
    RolloutInfrastructureFailure,
    RolloutInfrastructureKind,
)
from .event_grammar_runtime import EventGrammarRuntime, ReplayedActionMasks
from .generator import (
    PolicySnapshotMismatchError,
    RolloutGenerationRequest,
    RolloutGenerationResult,
    RolloutGenerator,
)
from .native_wire import NativeToolWire
from .types import (
    INITIAL_CONTEXT_PROFILE,
    GenerationPhase,
    PolicySnapshot,
    RolloutRequest,
    RolloutTermination,
    derive_generation_seed,
)

if TYPE_CHECKING:
    from skillev.policy.event_grammar import EventGrammarSpec
    from skillev.runtime.frozen_executor import FrozenExecutorSpec, WrittenAnswer
    from skillev.training.rollout_workflow import RolloutWorkflowResources
    from skillev.verification import VerifierSuite

    from .legal_events import LegalEventSet


class AnswerWriter(Protocol):
    @property
    def spec(self) -> FrozenExecutorSpec: ...

    async def write_answer(
        self,
        *,
        trajectory_id: str,
        step_index: int,
        prompt: AnswerWriterPrompt,
        max_output_tokens: int,
        regex: str | None = None,
    ) -> WrittenAnswer: ...


class RolloutEngine:
    def __init__(
        self,
        *,
        generator: RolloutGenerator,
        context_assembler: InitialContextAssembler,
        bounded_agent: BoundedAgent,
        terminal_evaluator: TerminalEvaluator,
        ledger: BudgetLedger,
        reasoning_call_maximum: BudgetVector,
        action_call_maximum: BudgetVector,
        emitter: RuntimeEventEmitter,
        clock: Callable[[], str],
        workflow_resources: RolloutWorkflowResources | None = None,
        trace_sink: RolloutTraceSink | None = None,
        event_grammar: EventGrammarRuntime | None = None,
        verifier: VerifierSuite | None = None,
        answer_writer: AnswerWriter | None = None,
    ) -> None:
        if reasoning_call_maximum.model_calls != 1:
            raise ValueError("reasoning reservation must cover one model call")
        if reasoning_call_maximum.agent_turns != 0:
            raise ValueError("reasoning reservation cannot consume an agent turn")
        if action_call_maximum.model_calls != 1:
            raise ValueError("action reservation must cover one model call")
        if action_call_maximum.agent_turns != 1:
            raise ValueError("action reservation must cover one agent turn")
        self._generator = generator
        self._context_assembler = context_assembler
        self._bounded_agent = bounded_agent
        self._terminal_evaluator = terminal_evaluator
        self._ledger = ledger
        self._reasoning_call_maximum = reasoning_call_maximum
        self._action_call_maximum = action_call_maximum
        self._emitter = emitter
        self._clock = clock
        self._workflow_resources = workflow_resources
        self._trace_sink = trace_sink or NullRolloutTraceSink()
        self._event_grammar = event_grammar
        self._verifier = verifier
        self._answer_writer = answer_writer

    async def run(self, request: RolloutRequest) -> RolloutArtifact:
        started_at = self._clock()
        pinned = self._generator.snapshot()
        if pinned.tokenizer_id != self._generator.tokenizer.tokenizer_id:
            self._reject(
                request,
                kind=RolloutInfrastructureKind.POLICY_SNAPSHOT_MISMATCH,
                stage="pin-policy",
                step_index=None,
                message="generator tokenizer does not match its policy snapshot",
            )
        if (
            request.task.environment_id != self._bounded_agent.environment_id
            or request.task.task_family != self._bounded_agent.task_family
        ):
            self._reject(
                request,
                kind=RolloutInfrastructureKind.INITIAL_CONTEXT_MISMATCH,
                stage="bind-environment",
                step_index=None,
                message="rollout task does not match the execution environment",
            )
        assembled = self._context_assembler.assemble(
            decoding=request.decoding,
            task=request.task,
            retrieved_skills=request.retrieved_skills,
            active_skill_ids=request.active_skill_ids,
            library_version=request.library_version,
            tokenizer=self._generator.tokenizer,
        )
        await self._trace_sink.record(
            RolloutTraceEvent(
                request.trajectory_id,
                request.task.task_id,
                None,
                RolloutTraceStage.EPISODE_STARTED,
                {
                    "active_skill_ids": list(assembled.contract.active_skill_ids),
                    "condition_id": request.condition_id,
                    "initial_context": assembled.text,
                    "initial_context_profile": INITIAL_CONTEXT_PROFILE,
                    "library_version": request.library_version,
                    "policy_snapshot_id": pinned.snapshot_id,
                    "retrieved_skill_ids": list(assembled.contract.retrieved_skill_ids),
                },
            )
        )
        self._generator.begin_episode(request.trajectory_id, pinned.snapshot_id)
        try:
            return await self._run_started_episode(
                request=request,
                assembled=assembled,
                pinned=pinned,
                started_at=started_at,
            )
        finally:
            self._generator.end_episode(request.trajectory_id)

    async def _run_started_episode(
        self,
        *,
        request: RolloutRequest,
        assembled: AssembledInitialContext,
        pinned: PolicySnapshot,
        started_at: str,
    ) -> RolloutArtifact:
        phase_spec, _ = PhaseContextSpec.split(assembled.text)
        if phase_spec is None:
            raise ValueError("sigma mode requires the native event wire")
        action_codec = NativeToolWire(
            ActionContract.freeze(
                request.task.action_surface,
                retrieved_skill_ids=assembled.contract.retrieved_skill_ids,
                active_skill_ids=assembled.contract.active_skill_ids,
            )
        )
        self._require_sigma_setup(request, phase_spec)
        if self._verifier is not None:
            self._require_verifier_domain(request)
        writes_answer = self._require_answer_writer_setup(request, phase_spec)
        reasoning_stop_ids = self._reasoning_stop_ids(request, phase_spec)
        reasoning_stop_trace: list[JsonValue] = list(reasoning_stop_ids)
        action_grammars: dict[str, str] = {}
        legal_sets: list[LegalEventSet] = []
        prefix_token_counts: list[int] = []
        state = BoundedAgentState(invocation_id=request.trajectory_id)
        completed_steps: tuple[TrajectoryStep, ...] = ()
        reasoning_token_counts: tuple[int, ...] = ()
        reasoning_finish_reasons: tuple[str, ...] = ()
        action_finish_reasons: tuple[str, ...] = ()
        input_window = ModelInputWindow.from_meta(assembled.contract.meta)
        self._emitter.emit(
            EventType.ROLLOUT_STARTED,
            {
                "assembled_hash": assembled.contract.assembled_hash,
                "decoding_snapshot_id": request.decoding.snapshot_id,
                "library_version": request.library_version,
                "trajectory_id": request.trajectory_id,
            },
        )
        self._emitter.emit(
            EventType.ROLLOUT_POLICY_PINNED,
            {
                "policy_snapshot": pinned.to_value(),
                "trajectory_id": request.trajectory_id,
            },
        )

        skill_start_turn = skill_available_from_turn(assembled.text)
        while not state.completed and len(completed_steps) < self._bounded_agent.max_turns:
            step_index = len(completed_steps) + 1
            progress_stage("reasoning-prefix", turn_index=step_index)
            self._require_current_snapshot(request, pinned, step_index=step_index)
            reasoning_prompt = render_reasoning_prefix(
                assembled.text,
                completed_steps,
                step_index,
            )
            reasoning_input = encode_policy_prompt(
                self._generator.tokenizer,
                reasoning_prompt.text,
                initial_text=assembled.text,
                window=input_window,
                step_index=step_index,
                native_thinking=(
                    request.decoding.prompt_encoder_version
                    == THINKING_ROLLOUT_PROMPT_ENCODER_VERSION
                ),
            )
            reasoning_request = RolloutGenerationRequest(
                episode_id=request.trajectory_id,
                library_version=request.library_version,
                turn_index=step_index,
                phase=GenerationPhase.REASONING,
                input_ids=reasoning_input.ids,
                max_new_tokens=request.decoding.max_reasoning_tokens,
                seed=derive_generation_seed(
                    base_seed=request.decoding.base_seed,
                    coordinate=request.sampling_coordinate,
                    step_index=step_index,
                    phase=GenerationPhase.REASONING,
                ),
                decoding_snapshot_id=request.decoding.snapshot_id,
                expected_policy_snapshot_id=pinned.snapshot_id,
                extra_stop_token_ids=reasoning_stop_ids,
            )
            await self._trace_sink.record(
                RolloutTraceEvent(
                    request.trajectory_id,
                    request.task.task_id,
                    step_index,
                    RolloutTraceStage.REASONING_REQUEST,
                    {
                        "max_new_tokens": reasoning_request.max_new_tokens,
                        "prompt_text": reasoning_prompt.text,
                        "seed": reasoning_request.seed,
                        **(
                            {"extra_stop_token_ids": reasoning_stop_trace}
                            if reasoning_stop_ids
                            else {}
                        ),
                    },
                )
            )
            row = current_progress()
            if row is not None:
                row.begin_phase("reasoning", len(reasoning_request.input_ids))
                row.phase_metrics(
                    original_input_tokens=reasoning_input.original_tokens,
                    truncated_input_tokens=reasoning_input.removed_tokens,
                    window_protected_initial_tokens=reasoning_input.protected_initial_tokens,
                    native_thinking=(
                        request.decoding.prompt_encoder_version
                        == THINKING_ROLLOUT_PROMPT_ENCODER_VERSION
                    ),
                )
            reasoning_result = await self._generate(
                request=request,
                generation_request=reasoning_request,
                pinned=pinned,
                step_index=step_index,
            )
            if row is not None:
                row.finish_phase(
                    reasoning_result.usage.output_tokens, reasoning_result.finish_reason
                )
            reasoning = decode_reasoning_segment(
                self._generator.tokenizer,
                reasoning_result.content_token_ids,
            )
            await self._trace_sink.record(
                RolloutTraceEvent(
                    request.trajectory_id,
                    request.task.task_id,
                    step_index,
                    RolloutTraceStage.REASONING_RESULT,
                    {
                        "finish_reason": reasoning_result.finish_reason,
                        "text": reasoning.text,
                        "token_count": len(reasoning.token_ids),
                    },
                )
            )
            reasoning_draft = ReasoningDraft(
                step_index=step_index,
                prompt_text=reasoning_prompt.text,
                prompt_hash=reasoning_prompt.prompt_hash,
                text=reasoning.text,
                generated_token_ids=reasoning.token_ids,
                policy_snapshot_id=reasoning_result.policy_snapshot_id,
            )
            reasoning_finish_reasons = (
                *reasoning_finish_reasons,
                reasoning_result.finish_reason,
            )

            forward_prefix = render_forward_prefix_from_parts(
                assembled.text,
                completed_steps,
                step_index,
                reasoning.text,
            )
            action_input = encode_policy_prompt(
                self._generator.tokenizer,
                forward_prefix.text,
                initial_text=assembled.text,
                window=input_window,
                step_index=step_index,
            )
            sigma_turn = self._sigma_turn(assembled.text, completed_steps, request, action_grammars)
            action_request = RolloutGenerationRequest(
                sampling_constraint=sigma_turn.key,
                action_budget_tokens=sigma_turn.budget,
                episode_id=request.trajectory_id,
                library_version=request.library_version,
                turn_index=step_index,
                phase=GenerationPhase.ACTION,
                input_ids=action_input.ids,
                max_new_tokens=request.decoding.max_action_tokens,
                seed=derive_generation_seed(
                    base_seed=request.decoding.base_seed,
                    coordinate=request.sampling_coordinate,
                    step_index=step_index,
                    phase=GenerationPhase.ACTION,
                ),
                decoding_snapshot_id=request.decoding.snapshot_id,
                expected_policy_snapshot_id=pinned.snapshot_id,
            )
            await self._trace_sink.record(
                RolloutTraceEvent(
                    request.trajectory_id,
                    request.task.task_id,
                    step_index,
                    RolloutTraceStage.ACTION_REQUEST,
                    {
                        "max_new_tokens": action_request.max_new_tokens,
                        "prompt_text": forward_prefix.text,
                        "seed": action_request.seed,
                        **sigma_turn.trace(),
                    },
                )
            )
            if row is not None:
                row.begin_phase("action", len(action_request.input_ids))
                row.phase_metrics(
                    native_thinking=False,
                    reasoning_condition="observed-c_t-response-ended",
                    reasoning_finish_reason=reasoning_result.finish_reason,
                    original_input_tokens=action_input.original_tokens,
                    truncated_input_tokens=action_input.removed_tokens,
                    window_protected_initial_tokens=action_input.protected_initial_tokens,
                )
            action_result = await self._generate(
                request=request,
                generation_request=action_request,
                pinned=pinned,
                step_index=step_index,
            )
            if row is not None:
                row.finish_phase(action_result.usage.output_tokens, action_result.finish_reason)
            action_masks = self._replay_event(request, sigma_turn, action_result, step_index)
            action_token_ids = action_result.content_token_ids
            if not action_token_ids:
                self._reject(
                    request,
                    kind=RolloutInfrastructureKind.EMPTY_ACTION,
                    stage="decode-action",
                    step_index=step_index,
                    message="action generation produced no sampled tokens",
                )
            try:
                action_segment = decode_action_segment(
                    self._generator.tokenizer,
                    action_token_ids,
                )
            except RolloutBoundaryError:
                self._reject(
                    request,
                    kind=RolloutInfrastructureKind.EMPTY_ACTION,
                    stage="decode-action",
                    step_index=step_index,
                    message="action generation produced no model-visible text",
                )
            parse_result = self._parse_event(
                request, action_codec, sigma_turn, action_segment.text, step_index
            )
            from skillev.diagnostics.action_submission import ActionSubmissionOutcome

            submission = ActionSubmissionOutcome.observe(
                parse_result,
                finish_reason=action_result.finish_reason,
                output_tokens=action_result.usage.output_tokens,
                action_token_cap=request.decoding.max_action_tokens,
                turns_remaining=self._bounded_agent.max_turns - step_index,
            )
            await self._trace_sink.record(
                RolloutTraceEvent(
                    request.trajectory_id,
                    request.task.task_id,
                    step_index,
                    RolloutTraceStage.ACTION_RESULT,
                    {
                        "submission_outcome": submission.to_value(),
                        "finish_reason": action_result.finish_reason,
                        "raw_text": action_segment.text,
                        "token_count": len(action_segment.token_ids),
                        **({} if action_masks is None else _mask_trace(action_masks)),
                    },
                )
            )
            await self._trace_sink.record(
                RolloutTraceEvent(
                    request.trajectory_id,
                    request.task.task_id,
                    step_index,
                    RolloutTraceStage.ACTION_PARSED,
                    {
                        "action": (
                            None if parse_result.action is None else parse_result.action.to_value()
                        ),
                        "public_error_code": parse_result.public_error_code,
                        "raw_text": action_segment.text,
                        "status": parse_result.status.value,
                    },
                )
            )
            action_draft = ActionDraft(
                step_index=step_index,
                forward_prefix_text=forward_prefix.text,
                forward_prefix_hash=forward_prefix.prefix_hash,
                text=action_segment.text,
                token_ids=action_segment.token_ids,
                parse_result=parse_result,
                policy_snapshot_id=action_result.policy_snapshot_id,
            )
            action_finish_reasons = (*action_finish_reasons, action_result.finish_reason)
            executed_parse = parse_result
            if (
                writes_answer
                and parse_result.action is not None
                and parse_result.action.kind is ActionKind.COMPLETE
            ):
                executed_parse = await self._write_answer(
                    request,
                    parse_result,
                    initial_text=assembled.text,
                    previous_steps=completed_steps,
                    step_index=step_index,
                    reasoning_text=reasoning.text,
                )
            progress_stage("environment-execute")
            try:
                turn = await self._bounded_agent.execute_turn(
                    state,
                    BoundedAgentTurnRequest(
                        trajectory_id=request.trajectory_id,
                        step_index=step_index,
                        action_text=action_segment.text,
                        action_token_ids=action_segment.token_ids,
                        parse_result=executed_parse,
                        retrieved_skill_ids=assembled.contract.retrieved_skill_ids
                        if step_index >= skill_start_turn
                        else (),
                        active_skill_ids=assembled.contract.active_skill_ids
                        if step_index >= skill_start_turn
                        else (),
                    ),
                )
            except EnvironmentSkillInvocationMismatchError:
                self._reject(
                    request,
                    kind=RolloutInfrastructureKind.ENVIRONMENT_SKILL_INVOCATION_MISMATCH,
                    stage="execute-action",
                    step_index=step_index,
                    message="environment skill credit differs from the structured action",
                )
            submission = replace(
                submission,
                admitted=turn.admitted,
                executed=turn.executed,
                execution_status=turn.observation.observation_status if turn.executed else None,
            )
            if row is not None:
                row.submission_outcome(dict(submission.to_value()))
            step = materialize_trajectory_step(
                initial_text=assembled.text,
                previous_steps=completed_steps,
                draft=CompletedStepDraft(
                    reasoning=reasoning_draft,
                    action=action_draft,
                    observation=turn.observation,
                ),
                r2flow=R2FlowStepInputs(
                    legal_event_set_sha256=sigma_turn.legal.sha256,
                    action_grammar_sha256=sigma_turn.sha256,
                    action_budget_tokens=sigma_turn.budget,
                    action_stop_token_ids=tuple(action_result.stop_token_ids),
                    action_finish_reason=action_result.finish_reason,
                    budget_forced_positions=tuple(action_masks.budget_forced_positions),
                    action_masks_digest=action_masks.digest,
                    reasoning_token_ids=tuple(reasoning_result.content_token_ids),
                    reasoning_stop_token_ids=tuple(reasoning_result.stop_token_ids),
                    reasoning_finish_reason=reasoning_result.finish_reason,
                ),
            )
            from skillev.diagnostics.action_failures import (
                ACTION_FAILURE_FORMAT,
                classify_action_outcome,
            )

            observed_value = turn.observation.public_value
            observed_error = (
                observed_value.get("error") if isinstance(observed_value, dict) else None
            )
            action_diagnostics = classify_action_outcome(
                action_segment.text,
                parse_status=parse_result.status.value,
                finish_reason=action_result.finish_reason,
                action_kind=None if parse_result.action is None else parse_result.action.kind.value,
                completed=turn.state.completed,
                observation_status=step.observation_status,
                action_wire=action_codec.format_version,
                public_error_code=(
                    observed_error
                    if isinstance(observed_error, str)
                    else parse_result.public_error_code
                ),
                available_native_names=tuple(binding.name for binding in action_codec.bindings),
                output_token_count=len(action_result.content_token_ids),
            )
            if row is not None:
                row.action_outcome(action_diagnostics)
            await self._trace_sink.record(
                RolloutTraceEvent(
                    request.trajectory_id,
                    request.task.task_id,
                    step_index,
                    RolloutTraceStage.ENVIRONMENT_RESULT,
                    {
                        "submission_outcome": submission.to_value(),
                        "action_diagnostics": {
                            "format": ACTION_FAILURE_FORMAT,
                            "action_wire": action_codec.format_version,
                            "labels": list(action_diagnostics),
                            "admitted": None,
                            "executed": None,
                            "terminal_success": None,
                        },
                        "budget_usage": turn.observation.budget_usage.to_value(),
                        "invoked_skill_ids": list(turn.observation.invoked_skill_ids),
                        "observation": turn.observation.public_value,
                        "observation_status": turn.observation.observation_status,
                        **_state_trace(step.r2flow),
                    },
                )
            )
            progress_stage("step-materialized")
            legal_sets.append(sigma_turn.legal)
            prefix_token_counts.append(len(reasoning_input.ids))
            completed_steps = (*completed_steps, step)
            reasoning_token_counts = (
                *reasoning_token_counts,
                len(reasoning.token_ids),
            )
            state = turn.state
            self._emitter.emit(
                EventType.ROLLOUT_STEP_COMMITTED,
                {
                    "action_token_count": step.action_token_count,
                    "forward_prefix_hash": step.forward_prefix_hash,
                    "hindsight_prefix_hash": step.hindsight_prefix_hash,
                    "observation_status": step.observation_status,
                    "step_index": step.index,
                    "trajectory_id": request.trajectory_id,
                },
            )
        self._require_current_snapshot(request, pinned, step_index=None)
        evaluation_input: TerminalEvaluationInput
        if state.completed:
            termination = RolloutTermination.COMPLETED
            evaluation_input = SubmittedTerminalValue(state.completion_value)
        else:
            termination = RolloutTermination.HORIZON_EXHAUSTED
            evaluation_input = NoTerminalSubmission(NoSubmissionReason.HORIZON_EXHAUSTED)
        from .environment import TerminalActionEvidence

        last_step = completed_steps[-1]
        evaluation_request = TerminalEvaluationRequest(
            trajectory_id=request.trajectory_id,
            task_id=request.task.task_id,
            termination=termination,
            evaluation_input=evaluation_input,
            last_action=TerminalActionEvidence(
                last_step.index,
                last_step.action_text,
                "valid",
                last_step.observation_status,
                action_finish_reasons[-1],
            ),
            public_transcript_hash=stable_hash(
                {
                    "initial_context_hash": assembled.contract.assembled_hash,
                    "steps": [step.to_value() for step in completed_steps],
                }
            ),
        )
        await self._trace_sink.record(
            RolloutTraceEvent(
                request.trajectory_id,
                request.task.task_id,
                None,
                RolloutTraceStage.TERMINAL_REQUEST,
                {
                    "submission_produced": isinstance(evaluation_input, SubmittedTerminalValue),
                    "termination": termination.value,
                },
            )
        )
        progress_stage("terminal-evaluation")
        if self._workflow_resources is None:
            reward = await self._terminal_evaluator.evaluate(evaluation_request)
        else:
            async with self._workflow_resources.terminal_evaluations.lease():
                reward = await self._terminal_evaluator.evaluate(evaluation_request)
        await self._trace_sink.record(
            RolloutTraceEvent(
                request.trajectory_id,
                request.task.task_id,
                None,
                RolloutTraceStage.TERMINAL_RESULT,
                {
                    "native_metric_names": [reward.native_metric_name],
                    "posterior_success": reward.success,
                    "reward": reward.value,
                    "verifier_version": reward.verifier_version,
                },
            )
        )

        verifier_records = None
        verifier_inputs = None
        if self._verifier is not None:
            from skillev.verification import verification_input

            from .artifact import VerifierInputs

            verifier_inputs = VerifierInputs(
                domain=self._verifier.domain,
                legal_event_sets=tuple(legal_sets),
                forward_prefix_token_counts=tuple(prefix_token_counts),
            )
            progress_stage("verification")
            verifier_records = await self._verifier.verify(
                verification_input(
                    trajectory_id=request.trajectory_id,
                    task_id=request.task.task_id,
                    domain=self._verifier.domain,
                    query=request.task.query,
                    steps=completed_steps,
                    legal_sets=legal_sets,
                    forward_prefix_token_counts=prefix_token_counts,
                )
            )

        completed_at = self._clock()
        record = finalize_trajectory_record(
            request=request,
            assembled=assembled,
            steps=completed_steps,
            reward=reward,
            tokenizer=self._generator.tokenizer,
            created_at=completed_at,
        )
        artifact = RolloutArtifact(
            initial_context=assembled,
            record=record,
            manifest=RolloutManifest(
                trajectory_id=request.trajectory_id,
                task_id=request.task.task_id,
                policy_snapshot=pinned,
                library_version=request.library_version,
                sampling_coordinate=request.sampling_coordinate,
                decoding_snapshot_id=request.decoding.snapshot_id,
                assembler_version=self._context_assembler.assembler_version,
                action_format_version=action_codec.format_version,
                generator_backend_id=pinned.backend_id,
                termination=termination,
                reasoning_token_counts=reasoning_token_counts,
                started_at=started_at,
                completed_at=completed_at,
                prompt_encoder_version=request.decoding.prompt_encoder_version,
                reasoning_finish_reasons=reasoning_finish_reasons,
                action_finish_reasons=action_finish_reasons,
                condition_id=request.condition_id,
                state_map=phase_spec.state_map,
            ),
            action_grammars=dict(action_grammars),
            verifier_records=verifier_records,
            verifier_inputs=verifier_inputs,
        )
        self._emitter.emit(EventType.TERMINAL_REWARD_RECORDED, reward.to_value())
        self._emitter.emit(EventType.ROLLOUT_COMPLETED, artifact.to_value())
        return artifact

    def _require_sigma_setup(self, request: RolloutRequest, phase_spec: PhaseContextSpec) -> None:
        if self._bounded_agent.max_turns != phase_spec.max_turns:
            raise ValueError("the agent horizon differs from the H0 max_turns")
        if self._event_grammar is None:
            raise ValueError("sigma mode requires an event grammar runtime")
        if request.decoding.max_action_tokens < 2:
            raise ValueError("the event budget B = max_action_tokens - 1 must be positive")

    def _require_answer_writer_setup(
        self, request: RolloutRequest, phase_spec: PhaseContextSpec
    ) -> bool:
        declared = declared_completion_writer(phase_spec) is not None
        if declared != (self._answer_writer is not None):
            raise ValueError(
                "the episode has an answer writer exactly when its H0 declares executor-answer@1"
            )
        if self._answer_writer is None:
            return False
        from .answer_writer import answer_writer_domain

        answer_writer_domain(request.task)
        if request.decoding.max_action_tokens > self._answer_writer.spec.max_output_tokens:
            raise ValueError("the answer writer's output cap exceeds executor max_output_tokens")
        return True

    async def _write_answer(
        self,
        request: RolloutRequest,
        parse_result: ActionParseResult,
        *,
        initial_text: str,
        previous_steps: tuple[TrajectoryStep, ...],
        step_index: int,
        reasoning_text: str,
    ) -> ActionParseResult:
        from skillev.contracts import normalize_json

        from .answer_writer import answer_writer_domain, answer_writer_prompt, answer_writer_regex

        assert self._answer_writer is not None
        assert parse_result.action is not None
        progress_stage("answer-writer", turn_index=step_index)
        written = await self._answer_writer.write_answer(
            trajectory_id=request.trajectory_id,
            step_index=step_index,
            prompt=answer_writer_prompt(
                initial_text=initial_text,
                previous_steps=previous_steps,
                step_index=step_index,
                reasoning_text=reasoning_text,
                task=request.task,
            ),
            max_output_tokens=request.decoding.max_action_tokens,
            regex=answer_writer_regex(answer_writer_domain(request.task)),
        )
        output = written.output
        await self._trace_sink.record(
            RolloutTraceEvent(
                request.trajectory_id,
                request.task.task_id,
                step_index,
                RolloutTraceStage.ANSWER_WRITER_RESULT,
                {
                    "completion_writer": EXECUTOR_ANSWER,
                    "window": written.window.to_value(),
                    "finish_reason": output.finish,
                    "prompt_tokens": output.prompt_tokens,
                    "output_tokens": len(output.token_ids),
                    "raw_text": output.text,
                },
            )
        )
        return ActionParseResult(
            ActionParseStatus.VALID,
            replace(
                parse_result.action,
                arguments=normalize_json({"value": {"answer": output.text}}),
            ),
            None,
        )

    def _reasoning_stop_ids(
        self, request: RolloutRequest, phase_spec: PhaseContextSpec
    ) -> tuple[int, ...]:
        if request.decoding.reasoning_stop_version is None:
            return ()
        if phase_spec.reasoning_call_line is None:
            raise ValueError(
                "the decoding stop guard reasoning-stop-at-tool-call@1 requires an H0 that "
                "declares reasoning-call-line@1"
            )
        ids = tuple(self._generator.tokenizer.encode(REASONING_STOP_TEXT))
        if len(ids) != 1:
            raise ValueError("the reasoning stop guard needs <tool_call> as one tokenizer id")
        return ids

    def _require_verifier_domain(self, request: RolloutRequest) -> None:
        assert self._verifier is not None
        context = request.task.public_context
        domain = context.get("benchmark_id") if isinstance(context, dict) else None
        if domain != self._verifier.domain:
            raise ValueError("the verifier suite belongs to another domain than the task")

    def _sigma_turn(
        self,
        initial_text: str,
        completed_steps: tuple[TrajectoryStep, ...],
        request: RolloutRequest,
        action_grammars: dict[str, str],
    ) -> _SigmaTurn:
        from skillev.policy.event_grammar import event_grammar_sha256
        from skillev.scoring.quotient import legal_events_at, sigma

        assert self._event_grammar is not None
        state = sigma(initial_text, completed_steps)
        legal = legal_events_at(initial_text, state)
        budget = request.decoding.max_action_tokens - 1
        spec = legal.grammar_spec(budget=budget, stop_token_id=self._event_grammar.stop_token_id)
        key = self._event_grammar.grammar_key(spec)
        digest = event_grammar_sha256(key)
        action_grammars[digest] = key
        return _SigmaTurn(state.key(), state.rank, legal, spec, key, digest, budget)

    def _replay_event(
        self,
        request: RolloutRequest,
        turn: _SigmaTurn,
        result: RolloutGenerationResult,
        step_index: int,
    ) -> ReplayedActionMasks:
        assert self._event_grammar is not None
        if not result.content_token_ids or result.stop_token_ids != (turn.spec.stop_token_id,):
            self._reject(
                request,
                kind=RolloutInfrastructureKind.ACTION_CONSTRAINT_VIOLATION,
                stage="replay-action-mask",
                step_index=step_index,
                message="event action did not end on the grammar stop token",
            )
        try:
            return self._event_grammar.replay(
                turn.key, result.content_token_ids, result.stop_token_ids
            )
        except ValueError:
            self._reject(
                request,
                kind=RolloutInfrastructureKind.ACTION_CONSTRAINT_VIOLATION,
                stage="replay-action-mask",
                step_index=step_index,
                message="sampled action tokens lie outside the event grammar mask",
            )

    def _parse_event(
        self,
        request: RolloutRequest,
        codec: NativeToolWire,
        turn: _SigmaTurn,
        text: str,
        step_index: int,
    ) -> ActionParseResult:
        from skillev.policy.event_grammar import render_event_call
        from skillev.scoring.quotient import NonEventStepError, parse_step_event

        try:
            event = parse_step_event(turn.legal, text)
            args = dict(event.args)
            if render_event_call(turn.spec, event.u, args) != text:
                raise NonEventStepError("the action is not the canonical rendering of its event")
            if not turn.legal.allows(event.u, args):
                raise NonEventStepError("the event is not legal at this state")
            result = codec.action_from_event(event.u, args)
        except ValueError:
            self._reject(
                request,
                kind=RolloutInfrastructureKind.NON_EVENT_ACTION,
                stage="canonical-event",
                step_index=step_index,
                message="the action is not one legal canonical event of this turn",
            )
        if result.status is not ActionParseStatus.VALID:
            self._reject(
                request,
                kind=RolloutInfrastructureKind.NON_EVENT_ACTION,
                stage="canonical-event",
                step_index=step_index,
                message="the event does not map to a declared action",
            )
        return result

    def _require_current_snapshot(
        self,
        request: RolloutRequest,
        pinned: PolicySnapshot,
        *,
        step_index: int | None,
    ) -> None:
        if self._generator.snapshot() != pinned:
            self._reject(
                request,
                kind=RolloutInfrastructureKind.POLICY_SNAPSHOT_MISMATCH,
                stage="verify-policy-snapshot",
                step_index=step_index,
                message="generator policy snapshot changed during rollout",
            )

    async def _generate(
        self,
        *,
        request: RolloutRequest,
        generation_request: RolloutGenerationRequest,
        pinned: PolicySnapshot,
        step_index: int,
    ) -> RolloutGenerationResult:
        phase = generation_request.phase
        reservation = BudgetReservation(
            reservation_id=f"{request.trajectory_id}:{step_index}:{phase.value}:model",
            run_id=self._ledger.run_id,
            attempt_id=self._ledger.attempt_id,
            invocation_id=request.trajectory_id,
            maximum=(
                self._reasoning_call_maximum
                if phase is GenerationPhase.REASONING
                else self._action_call_maximum
            ),
        )
        if len(generation_request.input_ids) > reservation.maximum.input_tokens:
            raise ValueError("model input exceeds the declared per-request token allowance")
        self._ledger.reserve(reservation)
        self._emitter.emit(
            EventType.BUDGET_RESERVED,
            {
                "maximum": reservation.maximum.to_value(),
                "reservation_id": reservation.reservation_id,
            },
        )
        try:
            if self._workflow_resources is None:
                result = await self._generator.generate(generation_request)
            else:
                endpoint = getattr(self._generator, "execution_endpoint", None)
                limiter = (
                    self._workflow_resources.model_requests
                    if endpoint is None
                    else self._workflow_resources.model_limiter(endpoint(generation_request))
                )
                async with limiter.lease(
                    token_cost=len(generation_request.input_ids)
                    + generation_request.max_new_tokens,
                    role="actor",
                ):
                    result = await self._generator.generate(generation_request)
        except PolicySnapshotMismatchError:
            self._reject(
                request,
                kind=RolloutInfrastructureKind.POLICY_SNAPSHOT_MISMATCH,
                stage=f"generate-{phase.value}",
                step_index=step_index,
                message="generator rejected the exact pinned policy snapshot",
            )

        actual_usage = result.usage.add(
            BudgetVector(
                agent_turns=(1 if generation_request.phase is GenerationPhase.ACTION else 0),
            )
        )
        if result.usage.input_tokens != len(generation_request.input_ids):
            raise ValueError("generator input-token usage differs from the request")
        if result.usage.model_calls != 1:
            raise ValueError("one generation request must report one model call")
        settlement = BudgetSettlement(
            reservation_id=reservation.reservation_id,
            actual=actual_usage,
        )
        self._ledger.settle(settlement)
        self._emitter.emit(
            EventType.BUDGET_SETTLED,
            {
                "actual": actual_usage.to_value(),
                "reservation_id": reservation.reservation_id,
            },
        )
        if result.policy_snapshot_id != pinned.snapshot_id:
            self._reject(
                request,
                kind=RolloutInfrastructureKind.POLICY_SNAPSHOT_MISMATCH,
                stage=f"verify-{phase.value}-policy",
                step_index=step_index,
                message="generator returned another policy snapshot",
            )
        if result.backend_id != pinned.backend_id:
            self._reject(
                request,
                kind=RolloutInfrastructureKind.POLICY_SNAPSHOT_MISMATCH,
                stage=f"verify-{phase.value}-backend",
                step_index=step_index,
                message="generator returned another backend identity",
            )
        return result

    def _reject(
        self,
        request: RolloutRequest,
        *,
        kind: RolloutInfrastructureKind,
        stage: str,
        step_index: int | None,
        message: str,
    ) -> NoReturn:
        failure = RolloutInfrastructureFailure(
            trajectory_id=request.trajectory_id,
            task_id=request.task.task_id,
            kind=kind,
            stage=stage,
            step_index=step_index,
            public_message=message,
        )
        self._emitter.emit(EventType.ROLLOUT_REJECTED, failure.to_value())
        raise RolloutInfrastructureError(failure)


@dataclass(frozen=True, slots=True)
class _SigmaTurn:
    state_key: str
    rank: int
    legal: LegalEventSet
    spec: EventGrammarSpec
    key: str
    sha256: str
    budget: int

    def trace(self) -> dict[str, JsonValue]:
        return {
            "event_grammar_sha256": self.sha256,
            "legal_event_set_sha256": self.legal.sha256,
            "action_budget_tokens": self.budget,
            "state": {"key": self.state_key, "rank": self.rank},
        }


def _mask_trace(masks: ReplayedActionMasks) -> dict[str, JsonValue]:
    return {
        "forced_token_count": sum(masks.forced),
        "budget_forced": bool(masks.budget_forced_positions),
        "budget_forced_positions": list(masks.budget_forced_positions),
    }


def _state_trace(record: StepR2FlowRecord) -> dict[str, JsonValue]:
    return {
        "state": {
            "key": record.state_key,
            "rank": record.rank,
            "in_degree": len(record.in_edges),
            "terminal": record.terminal,
        }
    }
