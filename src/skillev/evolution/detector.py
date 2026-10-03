from __future__ import annotations

from dataclasses import dataclass

from skillev.contracts import JsonValue


@dataclass(frozen=True, slots=True)
class AwaitingDetectorSegment:
    expected_library_version: str
    kind: str = "awaiting"

    def __post_init__(self) -> None:
        if not self.expected_library_version.strip():
            raise ValueError("expected detector library version must be non-empty")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "expected_library_version": self.expected_library_version,
            "kind": self.kind,
        }


DetectorRuntimeState = AwaitingDetectorSegment


def detector_state_from_value(value: object) -> DetectorRuntimeState:
    if not isinstance(value, dict):
        raise TypeError("detector runtime state must be an object")
    if value.get("kind") != "awaiting" or set(value) != {"expected_library_version", "kind"}:
        raise ValueError("AwaitingDetectorSegment has incompatible fields")
    version = value["expected_library_version"]
    if not isinstance(version, str):
        raise TypeError("expected detector library version must be text")
    return AwaitingDetectorSegment(version)


def detector_library_version(state: DetectorRuntimeState) -> str:
    return state.expected_library_version


class LibrarySegmentDetector:
    def __init__(self, state: DetectorRuntimeState) -> None:
        self._state = state

    @classmethod
    def fresh(cls, *, library_version: str) -> LibrarySegmentDetector:
        return cls(AwaitingDetectorSegment(library_version))

    @classmethod
    def from_runtime_state(
        cls, *, expected_library_version: str, state: DetectorRuntimeState
    ) -> LibrarySegmentDetector:
        if detector_library_version(state) != expected_library_version:
            raise ValueError("detector snapshot belongs to another library")
        return cls(state)

    @property
    def state(self) -> DetectorRuntimeState:
        return self._state

    def reset_for_library(self, old_library_version: str, new_library_version: str) -> None:
        if detector_library_version(self._state) != old_library_version:
            raise ValueError("detector reset old library identity differs")
        if new_library_version == old_library_version:
            raise ValueError("detector reset requires a new library version")
        self._state = AwaitingDetectorSegment(new_library_version)


__all__ = [
    "AwaitingDetectorSegment",
    "DetectorRuntimeState",
    "LibrarySegmentDetector",
    "detector_library_version",
    "detector_state_from_value",
]
