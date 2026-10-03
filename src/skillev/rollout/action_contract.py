from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from skillev.contracts import JsonValue, canonical_json, normalize_json
from skillev.contracts.skill_invocation import (
    SkillInvocationAdmissionError,
    canonical_invoked_skill_ids,
)
from skillev.runtime.contracts import ActionKind, StructuredAction

from .action_surface import ActionSurface, TerminalMode, admit_arguments


@dataclass(frozen=True, slots=True)
class AdmittedAction:
    action: StructuredAction
    error: tuple[str, str] | None = None
    invoked_skill_ids: tuple[str, ...] = ()
    completion: bool = False
    submission: JsonValue = None


@dataclass(frozen=True, slots=True)
class ActionContract:
    surface_json: str | None
    retrieved_skill_ids: tuple[str, ...] = ()
    active_skill_ids: tuple[str, ...] = ()

    @classmethod
    def freeze(
        cls,
        surface: ActionSurface | None,
        *,
        retrieved_skill_ids: tuple[str, ...] = (),
        active_skill_ids: tuple[str, ...] = (),
    ) -> ActionContract:
        return cls(
            None if surface is None else canonical_json(surface.to_value()),
            retrieved_skill_ids,
            active_skill_ids,
        )

    @property
    def surface(self) -> ActionSurface | None:
        return (
            None
            if self.surface_json is None
            else ActionSurface.from_value(json.loads(self.surface_json))
        )

    def render_native_semantics(self) -> tuple[str, ...]:
        surface = self.surface
        if surface is None:
            raise ValueError("native semantics require an explicit surface")
        lines = list(surface.instructions)
        if surface.terminal_mode is TerminalMode.ENVIRONMENT:
            lines.append(
                "The environment determines termination; separate answer submission is unavailable."
            )
        if surface.dynamic_choice_fields:
            lines.append(
                "Use current public choices from: " + ", ".join(surface.dynamic_choice_fields) + "."
            )
        return tuple(lines)

    def invoked_skill_ids(self, action: StructuredAction) -> tuple[str, ...]:
        return canonical_invoked_skill_ids(
            action_kind=action.kind.value,
            action_skill_id=action.skill_id,
            retrieved_skill_ids=self.retrieved_skill_ids,
            active_skill_ids=self.active_skill_ids,
        )

    def validate(
        self,
        action: StructuredAction,
        *,
        validate_completion: Callable[[JsonValue], bool],
    ) -> AdmittedAction:
        action, error = self.admit_surface(action)
        if error is not None:
            return AdmittedAction(action, error)
        if action.kind is ActionKind.COMPLETE:
            arguments = action.arguments
            if not isinstance(arguments, dict) or "value" not in arguments:
                return AdmittedAction(action, ("schema_invalid", "invalid_completion"))
            submission = normalize_json(arguments["value"])
            if not validate_completion(submission):
                return AdmittedAction(action, ("schema_invalid", "invalid_completion"))
            return AdmittedAction(action, completion=True, submission=submission)
        try:
            skills = self.invoked_skill_ids(action)
        except SkillInvocationAdmissionError:
            return AdmittedAction(action, ("schema_invalid", "skill_not_available_in_h0"))
        return AdmittedAction(action, invoked_skill_ids=skills)

    def to_native_tools(self) -> tuple[dict[str, JsonValue], ...]:
        from .native_wire import native_bindings

        return tuple(
            binding.to_tool() for binding in native_bindings(self, public_action_semantics=True)
        )

    def to_scoring_metadata(self) -> dict[str, JsonValue]:
        return {
            "format": "raw-json-action-contract@1",
            "surface": None if self.surface_json is None else json.loads(self.surface_json),
            "retrieved_skill_ids": list(self.retrieved_skill_ids),
            "active_skill_ids": list(self.active_skill_ids),
            "probability": "raw-full-vocabulary",
        }

    def admit_surface(
        self,
        action: StructuredAction,
    ) -> tuple[StructuredAction, tuple[str, str] | None]:
        surface = self.surface
        if surface is None or action.kind is ActionKind.SKILL:
            return action, None
        if action.kind is ActionKind.COMPLETE:
            if (
                surface.terminal_mode.value != "explicit-completion"
                or surface.completion is None
                or action.name != "complete"
            ):
                return action, ("schema_invalid", "invalid_completion")
            return action, None
        resources = {tool.resource_id for tool in surface.tools}
        if action.resource_id not in resources:
            return action, ("tool_error", "unsupported_resource")
        matching_resource = tuple(
            tool for tool in surface.tools if tool.resource_id == action.resource_id
        )
        matching_tool = next(
            (tool for tool in matching_resource if tool.name == action.name),
            None,
        )
        if matching_tool is None:
            return action, ("tool_error", "unsupported_tool")
        admission = admit_arguments(action, matching_tool)
        if not admission.admitted:
            return action, ("schema_invalid", str(admission.error_code))
        normalized = cast(StructuredAction, admission.normalized_action)
        return normalized, None
