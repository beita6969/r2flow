from __future__ import annotations

import json
from dataclasses import dataclass

from skillev.contracts import JsonValue, canonical_json, normalize_json


@dataclass(frozen=True)
class EffectiveRunCondition:
    condition_id: str
    scientific_json: str
    execution_json: str

    @classmethod
    def create(
        cls, *, condition_id: str, scientific: dict[str, JsonValue], execution: dict[str, JsonValue]
    ) -> EffectiveRunCondition:
        if not condition_id.strip() or not scientific:
            raise ValueError("an effective condition requires explicit scientific settings")
        return cls(condition_id, canonical_json(scientific), canonical_json(execution))

    @property
    def scientific(self) -> dict[str, JsonValue]:
        return dict(json.loads(self.scientific_json))

    @property
    def execution(self) -> dict[str, JsonValue]:
        return dict(json.loads(self.execution_json))

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": "skillev-effective-run-condition@1",
            "condition_id": self.condition_id,
            "scientific": self.scientific,
            "execution": self.execution,
        }

    @classmethod
    def from_value(cls, value: object) -> EffectiveRunCondition:
        value = normalize_json(value)
        if (
            not isinstance(value, dict)
            or value.get("format") != "skillev-effective-run-condition@1"
        ):
            raise ValueError("unsupported effective run condition")
        identity, science, execution = (
            value["condition_id"],
            value["scientific"],
            value["execution"],
        )
        if (
            not isinstance(identity, str)
            or not isinstance(science, dict)
            or not isinstance(execution, dict)
        ):
            raise ValueError("effective run condition has incompatible fields")
        return cls.create(condition_id=identity, scientific=science, execution=execution)

    def differences(self, other: EffectiveRunCondition) -> dict[str, JsonValue]:
        def diff(left: JsonValue, right: JsonValue, path: str) -> list[JsonValue]:
            if left == right:
                return []
            if isinstance(left, dict) and isinstance(right, dict):
                rows: list[JsonValue] = []
                for key in sorted(left.keys() | right.keys()):
                    if key not in left or key not in right:
                        rows.append(
                            {
                                "field": f"{path}/{key}",
                                "before_present": key in left,
                                "after_present": key in right,
                                "before": left.get(key),
                                "after": right.get(key),
                            }
                        )
                    else:
                        rows.extend(diff(left[key], right[key], f"{path}/{key}"))
                return rows
            return [{"field": path, "before": left, "after": right}]

        return {
            "scientific": diff(self.scientific, other.scientific, "scientific"),
            "execution": diff(self.execution, other.execution, "execution"),
        }

    def require_same_science(self, other: EffectiveRunCondition) -> None:
        if self.scientific != other.scientific:
            raise ValueError(
                "scientific conditions changed; declare a new experiment, not recovery"
            )
