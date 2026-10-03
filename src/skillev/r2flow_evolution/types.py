from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

from skillev.contracts import JsonValue

PHASE_EVIDENCE_FORMAT: Final = "r2flow-phase-evidence@1"
EMPTY_SLOT_BODY: Final = ""


def serialized_fields(value: Any) -> Iterator[tuple[str, Any]]:
    for item in dataclasses.fields(value):
        yield item.name, getattr(value, item.name)


class EditKind(StrEnum):
    DEFER = "defer"
    SPLIT = "split"
    REFINE = "refine"
    RETAIN = "retain"
    PRUNE = "prune"
    COMPRESS = "compress"
    GENERATE = "generate"


STRUCTURAL_EDITS: Final = frozenset(
    {EditKind.SPLIT, EditKind.REFINE, EditKind.PRUNE, EditKind.COMPRESS, EditKind.GENERATE}
)


@dataclass(frozen=True, slots=True)
class SkillSpec:
    skill_id: str
    name: str
    description: str
    body: str
    families: tuple[str, ...]
    version: int = 0
    parent_id: str | None = None

    @property
    def is_empty_slot(self) -> bool:
        return self.body == EMPTY_SLOT_BODY


@dataclass(frozen=True, slots=True)
class LibraryVersion:
    version: int
    skills: tuple[SkillSpec, ...]

    def skill(self, skill_id: str) -> SkillSpec:
        for spec in self.skills:
            if spec.skill_id == skill_id:
                return spec
        raise KeyError(skill_id)

    @property
    def skill_ids(self) -> tuple[str, ...]:
        return tuple(spec.skill_id for spec in self.skills)


@dataclass(frozen=True, slots=True)
class EdgeObs:
    step_index: int
    predecessor_key: str
    state_key: str
    in_edges: tuple[tuple[str, str], ...]
    event_label: str
    event_function: str
    skill_id: str | None
    legal_event_count: int
    edge_residual: float


@dataclass(frozen=True, slots=True)
class TrajectoryObs:
    trajectory_id: str
    query_id: str
    domain: str
    family: str
    reward: float
    reward_eta: float
    success: bool
    tokens: int
    latency_seconds: float
    edges: tuple[EdgeObs, ...]

    @property
    def terminal_key(self) -> str:
        return self.edges[-1].state_key


@dataclass(frozen=True, slots=True)
class VerifierObs:
    trajectory_id: str
    step_index: int
    skill_id: str
    z: tuple[str, ...]
    y: float
    confidence: float
    gate_eligible: bool


@dataclass(frozen=True, slots=True)
class EventVerifierObs:
    trajectory_id: str
    step_index: int
    event_class: str
    unit: str
    z: tuple[str, ...]
    y: float
    confidence: float
    eligible: bool
    verifiers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PhaseEvidence:
    phase: int
    library: LibraryVersion
    trajectories: tuple[TrajectoryObs, ...]
    verifier: tuple[VerifierObs, ...]
    eta: float
    epsilon: float
    optimizer_step: int
    event_verifier: tuple[EventVerifierObs, ...]
    format: str = PHASE_EVIDENCE_FORMAT


@dataclass(slots=True)
class PhaseState:
    phase: int = 0
    carried_counts: dict[tuple[str, tuple[str, ...]], tuple[float, float]] = field(
        default_factory=dict
    )
    cooldown: dict[str, int] = field(default_factory=dict)
    retired: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CandidateEdit:
    kind: EditKind
    skill_ids: tuple[str, ...]
    context: tuple[str, ...] | None
    rank: tuple[float, ...]
    evidence: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class AuthoredEdit:
    candidate: CandidateEdit
    added: tuple[SkillSpec, ...]
    removed: tuple[str, ...]
    author_model: str
    author_record: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class PairedOutcome:
    query_id: str
    domain: str
    success_a: float
    success_b: float
    reward_eta_a: float
    reward_eta_b: float
    tokens_a: float
    tokens_b: float
    latency_a: float
    latency_b: float


@dataclass(frozen=True, slots=True)
class GateDecision:
    accepted: tuple[AuthoredEdit, ...]
    rejected: tuple[tuple[AuthoredEdit, str], ...]
    library_after: LibraryVersion
    trace_row: dict[str, JsonValue]
