from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from skillev.contracts import JsonValue

from .attempt_run_plan import AttemptRunCursorState
from .skill_library import SkillLibraryState

if TYPE_CHECKING:
    from skillev.evolution.detector import DetectorRuntimeState
    from skillev.training.projections import FullProjectionRuntimeState


@dataclass(frozen=True, slots=True)
class OrderedTaskCursorState:
    curriculum_id: str
    cursor: int
    format: str = "skillev-task-cursor@4"

    def __post_init__(self) -> None:
        if not self.curriculum_id.strip():
            raise ValueError("curriculum_id must be non-empty")
        if type(self.cursor) is not int or self.cursor < 0:
            raise ValueError("task cursor must be non-negative")
        if self.format != "skillev-task-cursor@4":
            raise ValueError("unsupported task cursor format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "curriculum_id": self.curriculum_id,
            "cursor": self.cursor,
            "format": self.format,
        }

    @classmethod
    def from_value(cls, value: object) -> OrderedTaskCursorState:
        if not isinstance(value, dict) or set(value) != {
            "curriculum_id",
            "cursor",
            "format",
        }:
            raise ValueError("OrderedTaskCursorState has incompatible fields")
        curriculum_id = value["curriculum_id"]
        cursor = value["cursor"]
        format_name = value["format"]
        if not isinstance(curriculum_id, str) or type(cursor) is not int:
            raise TypeError("task cursor identity has incompatible types")
        if not isinstance(format_name, str):
            raise TypeError("task cursor format must be text")
        return cls(curriculum_id=curriculum_id, cursor=cursor, format=format_name)


@dataclass(frozen=True, slots=True)
class FullRuntimeExecutionState:
    task_cursor: OrderedTaskCursorState
    run_cursor: AttemptRunCursorState
    library: SkillLibraryState
    projections: FullProjectionRuntimeState
    detector: DetectorRuntimeState
    kind: str = "full"
    format: str = "skillev-runtime-execution-state@6"
    r2flow_phase: dict[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        if self.kind != "full" or self.format != "skillev-runtime-execution-state@6":
            raise ValueError("unsupported full runtime execution state")
        if self.r2flow_phase is not None and (
            not isinstance(self.r2flow_phase, dict)
            or self.r2flow_phase.get("library_digest") != self.library.current_version
        ):
            raise ValueError("carried phase record targets another library version")
        if self.projections.kind != "full":
            raise TypeError("full runtime requires full projection state")
        if not isinstance(self.run_cursor, AttemptRunCursorState):
            raise TypeError("full runtime requires AttemptRunCursorState")
        from skillev.diagnostics import diagnostics_library_version
        from skillev.evolution.detector import detector_library_version

        expected = self.library.current_version
        if diagnostics_library_version(self.projections.diagnostics_state) != expected:
            raise ValueError("projection state targets another library version")
        if detector_library_version(self.detector) != expected:
            raise ValueError("detector state targets another library version")

    def to_value(self) -> dict[str, JsonValue]:
        value: dict[str, JsonValue] = {
            "detector": self.detector.to_value(),
            "format": self.format,
            "kind": self.kind,
            "library": self.library.to_value(),
            "projections": self.projections.to_value(),
            "run_cursor": self.run_cursor.to_value(),
            "task_cursor": self.task_cursor.to_value(),
        }
        if self.r2flow_phase is not None:
            value["r2flow_phase"] = self.r2flow_phase
        return value

    @classmethod
    def from_value(cls, value: object) -> FullRuntimeExecutionState:
        carried = value.get("r2flow_phase") if isinstance(value, dict) else None
        if isinstance(value, dict) and "r2flow_phase" in value:
            if not isinstance(carried, dict):
                raise ValueError("carried phase record must be an object")
            value = {key: item for key, item in value.items() if key != "r2flow_phase"}
        data = _require_runtime_fields(value, kind="full")
        from skillev.evolution.detector import detector_state_from_value
        from skillev.training.projections import FullProjectionRuntimeState

        return cls(
            task_cursor=OrderedTaskCursorState.from_value(data["task_cursor"]),
            run_cursor=AttemptRunCursorState.from_value(data["run_cursor"]),
            library=SkillLibraryState.from_value(data["library"]),
            projections=FullProjectionRuntimeState.from_value(data["projections"]),
            detector=detector_state_from_value(data["detector"]),
            kind="full",
            format="skillev-runtime-execution-state@6",
            r2flow_phase=carried,
        )


def _require_runtime_fields(value: object, *, kind: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "detector",
        "format",
        "kind",
        "library",
        "projections",
        "run_cursor",
        "task_cursor",
    }:
        raise ValueError("runtime execution state has incompatible fields")
    if value["kind"] != kind or value["format"] != "skillev-runtime-execution-state@6":
        raise ValueError("runtime execution state identity differs")
    return value


__all__ = [
    "FullRuntimeExecutionState",
    "OrderedTaskCursorState",
]
