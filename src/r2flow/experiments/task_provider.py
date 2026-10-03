from __future__ import annotations

from dataclasses import dataclass

from skillev.rollout import RolloutTask
from skillev.runtime import OrderedTaskCursorState
from skillev.training import TaskProvider


@dataclass(slots=True)
class OrderedTaskProvider(TaskProvider):
    tasks: tuple[RolloutTask, ...]
    curriculum_id: str
    cursor: int = 0

    def __post_init__(self) -> None:
        if not self.tasks or len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("the task provider requires unique ordered tasks")
        if not self.curriculum_id.strip() or not 0 <= self.cursor <= len(self.tasks):
            raise ValueError("the task provider cursor is invalid")

    def next_task(self) -> RolloutTask:
        if self.cursor >= len(self.tasks):
            raise RuntimeError("the training curriculum is exhausted")
        task = self.tasks[self.cursor]
        self.cursor += 1
        return task

    @property
    def runtime_state(self) -> OrderedTaskCursorState:
        return OrderedTaskCursorState(self.curriculum_id, self.cursor)


@dataclass(frozen=True, slots=True)
class TaskProviderFactory:
    tasks: tuple[RolloutTask, ...]
    curriculum_id: str

    def fresh(self) -> OrderedTaskProvider:
        return OrderedTaskProvider(self.tasks, self.curriculum_id)

    def from_exact_state(self, state: OrderedTaskCursorState) -> TaskProvider:
        if state.curriculum_id != self.curriculum_id:
            raise ValueError("the resume cursor belongs to another curriculum")
        return OrderedTaskProvider(self.tasks, self.curriculum_id, state.cursor)


__all__ = ["OrderedTaskProvider", "TaskProviderFactory"]
