from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

from skillev.benchmarks.alfworld import (
    ALFWorldEnvironment,
    ALFWorldEpisode,
    ALFWorldPublicItem,
)
from skillev.contracts import SuccessRule, TerminalReward
from skillev.rollout import (
    NoTerminalSubmission,
    RolloutTask,
    TerminalEvaluationRequest,
    TerminalEvaluatorError,
)
from skillev.training import RolloutSessionBundle

from .terminal_inputs import no_submission_reward


class ALFWorldOutcomeUnavailableError(RuntimeError):
    pass


class PrivateALFWorldOutcomeView(Protocol):
    @property
    def task_id(self) -> str: ...

    @property
    def environment_id(self) -> str: ...

    @property
    def seed(self) -> int: ...

    def final_success(self) -> bool: ...


ALFWORLD_VERIFIER = "alfworld-terminal-evaluator@1"


@dataclass(frozen=True, slots=True)
class PrivateALFWorldCase:
    public: ALFWorldPublicItem
    private_task: object = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.public, ALFWorldPublicItem):
            raise TypeError("private ALFWorld case requires ALFWorldPublicItem")
        if self.private_task is None:
            raise ValueError("private ALFWorld task cannot be absent")


@dataclass(frozen=True, slots=True)
class PrivateALFWorldEpisodeSession:
    episode: ALFWorldEpisode
    outcome_view: PrivateALFWorldOutcomeView
    cleanup: Callable[[], Awaitable[None]] | None = field(default=None, repr=False)
    observed_reset_json: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.episode.task_id != self.outcome_view.task_id:
            raise ValueError("ALFWorld public and private views disagree on task identity")
        if self.episode.environment_id != self.outcome_view.environment_id:
            raise ValueError("ALFWorld public and private views disagree on environment identity")
        if self.episode.seed != self.outcome_view.seed:
            raise ValueError("ALFWorld public and private views disagree on seed")


class PrivateALFWorldEpisodeFactory(Protocol):
    def create(self, case: PrivateALFWorldCase) -> PrivateALFWorldEpisodeSession: ...


@dataclass(slots=True)
class PrivateALFWorldTerminalEvaluator:
    public: ALFWorldPublicItem
    outcome_view: PrivateALFWorldOutcomeView
    observed_reset_json: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.outcome_view.task_id != self.public.task_id:
            raise ValueError("ALFWorld evaluator belongs to another task")
        if self.outcome_view.environment_id != self.public.environment_id:
            raise ValueError("ALFWorld evaluator belongs to another environment")
        if self.outcome_view.seed != self.public.seed:
            raise ValueError("ALFWorld evaluator uses a different seed")

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        if request.task_id != self.public.task_id:
            raise TerminalEvaluatorError("terminal request reached a different ALFWorld task")
        if isinstance(request.evaluation_input, NoTerminalSubmission):
            return no_submission_reward(
                request,
                native_metric_name="alfworld-success",
                native_payload={
                    "benchmark_id": "alfworld",
                    "dataset_revision": self.public.dataset_revision,
                    "environment_snapshot_id": self.public.environment_snapshot_id,
                    "split": self.public.split,
                },
                environment_id=self.public.environment_id,
                verifier_version=ALFWORLD_VERIFIER,
            )
        try:
            success = self.outcome_view.final_success()
        except ALFWorldOutcomeUnavailableError as error:
            raise TerminalEvaluatorError("ALFWorld final outcome is unavailable") from error
        if type(success) is not bool:
            raise TerminalEvaluatorError("ALFWorld final success must be boolean")
        reward = float(success)
        return TerminalReward(
            value=reward,
            success=success,
            success_rule=SuccessRule.R_EQUALS_ONE,
            success_threshold=None,
            native_metric_name="alfworld-success",
            native_payload={
                "benchmark_id": "alfworld",
                "dataset_revision": self.public.dataset_revision,
                "environment_snapshot_id": self.public.environment_snapshot_id,
                "split": self.public.split,
            },
            environment_id=self.public.environment_id,
            verifier_version=ALFWORLD_VERIFIER,
        )


@dataclass(slots=True)
class PrivateALFWorldSessionFactory:
    cases: tuple[PrivateALFWorldCase, ...]
    episode_factory: PrivateALFWorldEpisodeFactory

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("private ALFWorld session factory requires cases")
        if any(not isinstance(case, PrivateALFWorldCase) for case in self.cases):
            raise TypeError("private ALFWorld session factory cases are invalid")
        task_ids = tuple(case.public.task_id for case in self.cases)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("private ALFWorld cases must have unique task identities")

    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        if not isinstance(task, RolloutTask):
            raise TypeError("ALFWorld session creation requires RolloutTask")
        matches = tuple(case for case in self.cases if case.public.task_id == task.task_id)
        if len(matches) != 1:
            raise ValueError("public task has no unique private ALFWorld case")
        case = matches[0]
        if task != case.public.to_rollout_task():
            raise ValueError("public ALFWorld task projection differs from its private case")
        session = self.episode_factory.create(case)
        if not isinstance(session, PrivateALFWorldEpisodeSession):
            raise TypeError("ALFWorld episode factory returned an incompatible session")
        if session.episode.task_id != case.public.task_id:
            raise ValueError("ALFWorld episode task identity differs from its private case")
        if session.episode.environment_id != case.public.environment_id:
            raise ValueError("ALFWorld episode environment differs from its private case")
        if session.episode.seed != case.public.seed:
            raise ValueError("ALFWorld episode seed differs from its private case")
        if session.episode.max_steps != case.public.max_steps:
            raise ValueError("ALFWorld episode step limit differs from its private case")
        return RolloutSessionBundle(
            environment=ALFWorldEnvironment(case.public, session.episode),
            evaluator=PrivateALFWorldTerminalEvaluator(
                case.public, session.outcome_view, session.observed_reset_json
            ),
            retrieved_skills=(),
            cleanup=session.cleanup,
        )


__all__ = [
    "ALFWorldOutcomeUnavailableError",
    "PrivateALFWorldCase",
    "PrivateALFWorldEpisodeFactory",
    "PrivateALFWorldEpisodeSession",
    "PrivateALFWorldOutcomeView",
    "PrivateALFWorldSessionFactory",
    "PrivateALFWorldTerminalEvaluator",
]
