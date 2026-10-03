from __future__ import annotations

from skillev.contracts import JsonValue
from skillev.runtime import ActionKind, FullRetrievedSkillContext, StructuredAction
from skillev.runtime.execution import EnvironmentObservation, RolloutEnvironmentSession
from skillev.runtime.frozen_executor import FrozenSkillExecutor

SKILL_EXECUTOR_RESOURCE = "skill-executor"


class SkillExecutorEnvironment:
    def __init__(
        self,
        environment: RolloutEnvironmentSession,
        skills: tuple[FullRetrievedSkillContext, ...],
        library_version: str,
        executor: FrozenSkillExecutor,
        trajectory_id: str,
    ) -> None:
        self.environment = environment
        self.skills = {skill.metadata.skill_id: skill for skill in skills}
        if len(self.skills) != len(skills) or not library_version or not trajectory_id:
            raise ValueError("executor environment needs unique skills and identities")
        self.library_version = library_version
        self.executor = executor
        self.trajectory_id = trajectory_id
        self._last = 0

    @property
    def environment_id(self) -> str:
        return self.environment.environment_id

    @property
    def task_family(self) -> str:
        return self.environment.task_family

    def validate_completion(self, submission: JsonValue) -> bool:
        return self.environment.validate_completion(submission)

    async def execute(self, action: StructuredAction, *, step_index: int) -> EnvironmentObservation:
        if action.kind is not ActionKind.SKILL:
            return await self.environment.execute(action, step_index=step_index)
        if step_index <= self._last:
            raise RuntimeError("the skill executor requires increasing steps")
        self._last = step_index
        arguments = action.arguments
        if (
            action.skill_id not in self.skills
            or action.resource_id != SKILL_EXECUTOR_RESOURCE
            or action.name != "invoke"
            or not isinstance(arguments, dict)
            or set(arguments) != {"input"}
            or type(arguments["input"]) is not str
        ):
            return EnvironmentObservation({"error": "skill_not_available"}, "schema_invalid")
        skill = self.skills[action.skill_id]
        return await self.executor.run(
            trajectory_id=self.trajectory_id,
            step_index=step_index,
            meta=skill.metadata,
            skill_md_text=skill.content,
            input_text=arguments["input"],
            library_version=self.library_version,
        )


__all__ = ["SKILL_EXECUTOR_RESOURCE", "SkillExecutorEnvironment"]
