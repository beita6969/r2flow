from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from skillev.contracts import JsonValue

DIAGNOSTICS_FORMAT = "skillev-diagnostics@4"


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
    window_size: int = 50
    stagnation_rho: float = 0.05
    format: str = DIAGNOSTICS_FORMAT

    def __post_init__(self) -> None:
        if self.format != DIAGNOSTICS_FORMAT:
            raise ValueError(f"DiagnosticsConfig.format must be {DIAGNOSTICS_FORMAT!r}")
        if type(self.window_size) is not int or self.window_size < 1:
            raise ValueError("DiagnosticsConfig.window_size must be a positive integer")
        rho = _finite_float(
            self.stagnation_rho,
            field="DiagnosticsConfig.stagnation_rho",
        )
        if not 0.0 < rho < 1.0:
            raise ValueError("DiagnosticsConfig.stagnation_rho must lie in (0, 1)")
        object.__setattr__(self, "stagnation_rho", rho)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "stagnation_rho": self.stagnation_rho,
            "window_size": self.window_size,
        }

    @classmethod
    def from_value(cls, value: object) -> DiagnosticsConfig:
        if not isinstance(value, Mapping) or set(value) != {
            "format",
            "stagnation_rho",
            "window_size",
        }:
            raise ValueError("DiagnosticsConfig has incompatible fields")
        if value["format"] != DIAGNOSTICS_FORMAT:
            raise ValueError("DiagnosticsConfig has an incompatible format")
        window_size = value["window_size"]
        rho = value["stagnation_rho"]
        if type(window_size) is not int:
            raise ValueError("DiagnosticsConfig.window_size must be an integer")
        if isinstance(rho, bool) or not isinstance(rho, int | float):
            raise ValueError("DiagnosticsConfig.stagnation_rho must be numeric")
        return cls(window_size=window_size, stagnation_rho=float(rho))


class LibrarySegmentMismatchError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FreshDiagnosticsSegment:
    expected_library_version: str
    kind: str = "fresh"

    def __post_init__(self) -> None:
        if not isinstance(self.expected_library_version, str) or not self.expected_library_version:
            raise ValueError("expected_library_version must be non-empty")
        if self.kind != "fresh":
            raise ValueError("unsupported fresh diagnostics state")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "expected_library_version": self.expected_library_version,
            "kind": self.kind,
        }


DiagnosticsState = FreshDiagnosticsSegment


def diagnostics_state_from_value(value: object) -> DiagnosticsState:
    if not isinstance(value, Mapping):
        raise TypeError("diagnostics state must be an object")
    if value.get("kind") != "fresh" or set(value) != {"expected_library_version", "kind"}:
        raise ValueError("FreshDiagnosticsSegment has incompatible fields")
    version = value["expected_library_version"]
    if not isinstance(version, str):
        raise TypeError("expected_library_version must be text")
    return FreshDiagnosticsSegment(version)


def diagnostics_library_version(state: DiagnosticsState) -> str:
    return state.expected_library_version


def reset_diagnostics_segment(
    state: DiagnosticsState,
    *,
    old_library_version: str,
    new_library_version: str,
) -> FreshDiagnosticsSegment:
    if diagnostics_library_version(state) != old_library_version:
        raise LibrarySegmentMismatchError("diagnostics reset old library identity differs")
    if new_library_version == old_library_version:
        raise ValueError("diagnostics reset requires a new library version")
    return FreshDiagnosticsSegment(expected_library_version=new_library_version)


__all__ = [
    "DIAGNOSTICS_FORMAT",
    "DiagnosticsConfig",
    "DiagnosticsState",
    "FreshDiagnosticsSegment",
    "LibrarySegmentMismatchError",
    "diagnostics_library_version",
    "diagnostics_state_from_value",
    "reset_diagnostics_segment",
]
