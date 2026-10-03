from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:
    from skillev.rollout.event_grammar_runtime import EventGrammarRuntime
    from skillev.runtime import BudgetLedger, RuntimeEventEmitter
    from skillev.runtime.executor_ledger import ExecutorCallRecord
    from skillev.runtime.frozen_executor import (
        ExecutorOutput,
        FrozenExecutorSpec,
        FrozenSkillExecutor,
    )

    from .config import PolicyRolloutConfig
    from .rollout_workflow import RolloutWorkflowResources


class FrozenBaseGenerator(Protocol):
    @property
    def tokenizer(self) -> Any: ...

    def episode_endpoint(self, episode_id: str) -> str: ...

    async def generate_frozen_base(
        self,
        *,
        episode_id: str,
        input_ids: tuple[int, ...],
        spec: FrozenExecutorSpec,
        regex: str | None = None,
    ) -> ExecutorOutput: ...


@dataclass(frozen=True, slots=True)
class R2FlowRolloutRuntime:
    skill_executor_factory: Callable[[str], FrozenSkillExecutor] | None
    event_grammar: EventGrammarRuntime | None


def xgrammar_event_runtime(tokenizer: Any) -> EventGrammarRuntime:
    from skillev.policy.event_closing import formal_tokenizer_info
    from skillev.rollout.event_grammar_runtime import XGrammarEventRuntime

    hf_tokenizer = getattr(tokenizer, "hf_tokenizer", None)
    if hf_tokenizer is None:
        raise TypeError("the event grammar needs the formal Hugging Face tokenizer")
    return XGrammarEventRuntime(tokenizer.encode, formal_tokenizer_info(hf_tokenizer))


@dataclass(slots=True)
class R2FlowRolloutBinding:
    run_root: Path
    event_grammar_factory: Callable[[Any], EventGrammarRuntime] = xgrammar_event_runtime
    _inflight: dict[str, asyncio.Future[Any]] = field(default_factory=dict)

    def bind(
        self,
        *,
        rollout: PolicyRolloutConfig,
        generator: Any,
        ledger: BudgetLedger,
        resources: RolloutWorkflowResources,
        emitter: RuntimeEventEmitter | None = None,
        executor_record_observer: Callable[[ExecutorCallRecord], None] | None = None,
    ) -> R2FlowRolloutRuntime:
        event_grammar = self.event_grammar_factory(generator.tokenizer)
        factory = self._executor_factory(
            rollout.executor, generator, ledger, resources, emitter, executor_record_observer
        )
        return R2FlowRolloutRuntime(factory, event_grammar)

    def _executor_factory(
        self,
        spec: FrozenExecutorSpec,
        generator: Any,
        ledger: BudgetLedger,
        resources: RolloutWorkflowResources,
        emitter: RuntimeEventEmitter | None,
        observer: Callable[[ExecutorCallRecord], None] | None = None,
    ) -> Callable[[str], FrozenSkillExecutor]:
        from skillev.runtime.executor_ledger import EXECUTOR_CALLS_FILE, JsonlExecutorCallSink
        from skillev.runtime.frozen_executor import (
            EXECUTOR_MEMO_FILE,
            ExecutorMemoStore,
            FrozenSkillExecutor,
        )

        if not callable(getattr(generator, "generate_frozen_base", None)) or not callable(
            getattr(generator, "episode_endpoint", None)
        ):
            raise TypeError("the frozen executor needs an adapter-free actor generator")
        memo = ExecutorMemoStore(self.run_root / EXECUTOR_MEMO_FILE)
        run_sink = JsonlExecutorCallSink(self.run_root / EXECUTOR_CALLS_FILE)
        inflight = self._inflight

        def sink(record: ExecutorCallRecord) -> None:
            run_sink(record)
            if observer is not None:
                observer(record)

        def create(trajectory_id: str) -> FrozenSkillExecutor:
            async def transport(
                input_ids: tuple[int, ...],
                call_spec: FrozenExecutorSpec,
                /,
                *,
                regex: str | None = None,
            ) -> ExecutorOutput:
                limiter = resources.model_limiter(generator.episode_endpoint(trajectory_id))
                async with limiter.lease(
                    token_cost=len(input_ids) + call_spec.max_output_tokens, role="executor"
                ):
                    output: ExecutorOutput = await (
                        generator.generate_frozen_base(
                            episode_id=trajectory_id, input_ids=input_ids, spec=call_spec
                        )
                        if regex is None
                        else generator.generate_frozen_base(
                            episode_id=trajectory_id,
                            input_ids=input_ids,
                            spec=call_spec,
                            regex=regex,
                        )
                    )
                    return output

            executor = FrozenSkillExecutor(
                spec=spec,
                tokenizer=generator.tokenizer,
                memo=memo,
                transport=transport,
                ledger=ledger,
                record_sink=sink,
                emitter=emitter,
            )
            executor._inflight = inflight
            return executor

        return create


__all__ = [
    "FrozenBaseGenerator",
    "R2FlowRolloutBinding",
    "R2FlowRolloutRuntime",
    "xgrammar_event_runtime",
]
