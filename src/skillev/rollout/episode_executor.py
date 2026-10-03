from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol

from skillev.contracts import JsonValue, ScientificSamplingCoordinate
from skillev.contracts.reasoning_call_line import REASONING_STOP_AT_TOOL_CALL
from skillev.diagnostics.rollout_trace import RolloutTraceSink
from skillev.policy.interface import (
    ROLLOUT_PROMPT_ENCODER_VERSION,
    THINKING_ROLLOUT_PROMPT_ENCODER_VERSION,
)
from skillev.runtime import (
    BoundedAgent,
    BoundedAgentPolicy,
    BudgetLedger,
    BudgetVector,
    RuntimeEventEmitter,
    SkillLibraryState,
    StructuredAction,
)
from skillev.runtime.execution import EnvironmentObservation, RolloutEnvironmentSession
from skillev.runtime.frozen_executor import FrozenSkillExecutor
from skillev.training.config import PolicyRolloutConfig
from skillev.training.rollout_workflow import AsyncResourceLimiter, RolloutWorkflowResources

from . import (
    CanonicalInitialContextAssembler,
    DecodingSnapshot,
    RolloutArtifact,
    RolloutEngine,
    RolloutGenerator,
    RolloutRequest,
    RolloutSessionBundle,
    RolloutTask,
)
from .event_grammar_runtime import EventGrammarRuntime


class EpisodeSessionFactory(Protocol):
    def create(self, task: RolloutTask) -> RolloutSessionBundle: ...


class ResourceLimitedEnvironment:
    def __init__(self, environment: RolloutEnvironmentSession, limiter: AsyncResourceLimiter):
        self._environment, self._limiter = environment, limiter

    @property
    def environment_id(self) -> str:
        return self._environment.environment_id

    @property
    def task_family(self) -> str:
        return self._environment.task_family

    async def execute(self, action: StructuredAction, *, step_index: int) -> EnvironmentObservation:
        async with self._limiter.lease():
            return await self._environment.execute(action, step_index=step_index)

    def validate_completion(self, submission: JsonValue) -> bool:
        return self._environment.validate_completion(submission)


def episode_decoding(config: PolicyRolloutConfig, task: RolloutTask) -> DecodingSnapshot:
    thinking = config.reasoning_native_thinking
    modes = dict(config.reasoning_by_domain)
    context = task.public_context
    domain = context.get("benchmark_id") if isinstance(context, dict) else None
    if modes:
        if not isinstance(domain, str) or domain not in modes:
            raise ValueError("task has no declared domain reasoning mode")
        thinking = modes[domain]
    budget = task.budget_profile
    stop = REASONING_STOP_AT_TOOL_CALL if domain in config.reasoning_stop_domains else None
    return DecodingSnapshot.create(
        max_reasoning_tokens=config.max_reasoning_tokens
        if budget is None
        else budget.max_reasoning_tokens,
        max_action_tokens=config.max_action_tokens if budget is None else budget.max_action_tokens,
        base_seed=config.base_seed,
        prompt_encoder_version=(
            THINKING_ROLLOUT_PROMPT_ENCODER_VERSION if thinking else ROLLOUT_PROMPT_ENCODER_VERSION
        ),
        reasoning_stop_version=stop,
        action_decoding_version=config.action_decoding,
    )


async def execute_episode(
    *,
    generator: RolloutGenerator,
    sessions: EpisodeSessionFactory,
    assembler: CanonicalInitialContextAssembler,
    rollout: PolicyRolloutConfig,
    task: RolloutTask,
    trajectory_id: str,
    library: SkillLibraryState,
    coordinate: ScientificSamplingCoordinate,
    decoding: DecodingSnapshot,
    epsilon_min: float,
    condition_id: str,
    ledger: BudgetLedger,
    emitter: RuntimeEventEmitter,
    clock: Callable[[], str],
    resources: RolloutWorkflowResources,
    artifact_sink: Callable[[RolloutArtifact], Awaitable[None]] | None = None,
    trace_sink: RolloutTraceSink | None = None,
    skill_executor_factory: Callable[[str], FrozenSkillExecutor] | None = None,
    event_grammar: EventGrammarRuntime | None = None,
) -> RolloutArtifact:
    turns = rollout.max_turns if task.budget_profile is None else task.budget_profile.max_turns
    if (
        turns > rollout.max_turns
        or decoding.max_reasoning_tokens > rollout.max_reasoning_tokens
        or decoding.max_action_tokens > rollout.max_action_tokens
    ):
        raise ValueError("task rollout budget profile exceeds the declared execution cap")
    async with resources.session_setups.lease():
        bundle = await asyncio.to_thread(sessions.create, task)
    try:
        environment = bundle.environment
        if skill_executor_factory is None:
            raise ValueError("skill-md-invoke@2 requires a frozen skill executor factory")
        from .skill_executor_environment import SkillExecutorEnvironment

        executor = skill_executor_factory(trajectory_id)
        environment = SkillExecutorEnvironment(
            environment,
            bundle.retrieved_skills,
            library.current_version,
            executor,
            trajectory_id,
        )
        writes = task.action_surface is not None and task.action_surface.completion_writer
        answer_writer = executor if writes else None
        maximum = rollout.policy_call_maximum
        input_tokens = maximum.input_tokens // (2 * rollout.max_turns)
        engine = RolloutEngine(
            generator=generator,
            context_assembler=assembler,
            bounded_agent=BoundedAgent(
                environment=ResourceLimitedEnvironment(environment, resources.environment_calls),
                policy=BoundedAgentPolicy(max_turns=turns),
                ledger=ledger,
                tool_call_maximum=BudgetVector(
                    tool_calls=1,
                    wall_time_milliseconds=maximum.wall_time_milliseconds // rollout.max_turns,
                ),
                emitter=emitter,
                action_surface=task.action_surface,
            ),
            terminal_evaluator=bundle.evaluator,
            ledger=ledger,
            reasoning_call_maximum=BudgetVector(
                input_tokens=input_tokens, output_tokens=rollout.max_reasoning_tokens, model_calls=1
            ),
            action_call_maximum=BudgetVector(
                input_tokens=input_tokens,
                output_tokens=rollout.max_action_tokens,
                model_calls=1,
                agent_turns=1,
            ),
            emitter=emitter,
            clock=clock,
            workflow_resources=resources,
            trace_sink=trace_sink,
            event_grammar=event_grammar,
            verifier=bundle.verifier,
            answer_writer=answer_writer,
        )
        artifact = await engine.run(
            RolloutRequest(
                trajectory_id=trajectory_id,
                task=task,
                retrieved_skills=bundle.retrieved_skills,
                active_skill_ids=library.active_skill_ids,
                library_version=library.current_version,
                sampling_coordinate=coordinate,
                decoding=decoding,
                epsilon_min=epsilon_min,
                condition_id=condition_id,
            )
        )
        if artifact_sink is not None:
            await artifact_sink(artifact)
        return artifact
    finally:
        if bundle.cleanup is not None:
            async with resources.session_cleanups.lease():
                await bundle.cleanup()
