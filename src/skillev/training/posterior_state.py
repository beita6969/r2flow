from __future__ import annotations

from dataclasses import dataclass

from skillev.contracts import JsonValue, PosteriorBatchUpdate

from .evidence_context import TrajectoryEvidenceContext

POSTERIOR_PROVENANCE_FORMAT = "skillev-posterior-event-provenance@2"


@dataclass(frozen=True, slots=True)
class PosteriorEvidenceBatch:
    optimizer_step: int
    policy_snapshot_id: str
    library_version: str
    trajectory_ids: tuple[str, ...]
    posterior: PosteriorBatchUpdate
    trajectory_contexts: tuple[TrajectoryEvidenceContext, ...] = ()

    def __post_init__(self) -> None:
        if type(self.optimizer_step) is not int or self.optimizer_step < 1:
            raise ValueError("posterior evidence step must be positive")
        if not self.policy_snapshot_id or not self.library_version:
            raise ValueError("posterior evidence requires policy and library identities")
        if not self.trajectory_ids or len(set(self.trajectory_ids)) != len(self.trajectory_ids):
            raise ValueError("posterior evidence requires unique trajectory identities")
        if (
            self.trajectory_contexts
            and tuple(item.trajectory_id for item in self.trajectory_contexts)
            != self.trajectory_ids
        ):
            raise ValueError("trajectory metadata differs from the committed batch order")

    def to_value(self) -> dict[str, JsonValue]:
        value: dict[str, JsonValue] = {
            "optimizer_step": self.optimizer_step,
            "policy_snapshot_id": self.policy_snapshot_id,
            "library_version": self.library_version,
            "trajectory_ids": list(self.trajectory_ids),
            "posterior": self.posterior.to_value(),
        }
        if self.trajectory_contexts:
            value["trajectory_contexts"] = [item.to_value() for item in self.trajectory_contexts]
        return value

    @classmethod
    def from_value(cls, value: object) -> PosteriorEvidenceBatch:
        if not isinstance(value, dict):
            raise TypeError("posterior evidence batch must be an object")
        return cls(
            optimizer_step=value["optimizer_step"],
            policy_snapshot_id=value["policy_snapshot_id"],
            library_version=value["library_version"],
            trajectory_ids=tuple(value["trajectory_ids"]),
            posterior=PosteriorBatchUpdate.from_value(value["posterior"]),
            trajectory_contexts=tuple(
                TrajectoryEvidenceContext.from_value(item)
                for item in value.get("trajectory_contexts", [])
            ),
        )


@dataclass(frozen=True, slots=True)
class PosteriorEventProvenance:
    batches: tuple[PosteriorEvidenceBatch, ...]
    format: str = POSTERIOR_PROVENANCE_FORMAT

    def __post_init__(self) -> None:
        if self.format != POSTERIOR_PROVENANCE_FORMAT:
            raise ValueError("posterior recovery requires complete event evidence")
        batch_ids: set[str] = set()
        trajectory_ids: set[str] = set()
        last_step = 0
        for batch in self.batches:
            if batch.optimizer_step <= last_step or batch.posterior.batch_id in batch_ids:
                raise ValueError("posterior evidence repeats or reorders a batch")
            if trajectory_ids.intersection(batch.trajectory_ids):
                raise ValueError("posterior trajectory evidence is already committed")
            last_step = batch.optimizer_step
            batch_ids.add(batch.posterior.batch_id)
            trajectory_ids.update(batch.trajectory_ids)

    @classmethod
    def empty(cls) -> PosteriorEventProvenance:
        return cls(batches=())

    def require_new_batch(self, batch_id: str, step: int, trajectory_ids: tuple[str, ...]) -> None:
        if self.batches and step <= self.batches[-1].optimizer_step:
            raise ValueError("posterior evidence step is already committed")
        for batch in self.batches:
            if batch.posterior.batch_id == batch_id or set(batch.trajectory_ids).intersection(
                trajectory_ids
            ):
                raise ValueError("posterior batch or trajectory evidence is already committed")

    def append_batch(self, batch: PosteriorEvidenceBatch) -> PosteriorEventProvenance:
        self.require_new_batch(batch.posterior.batch_id, batch.optimizer_step, batch.trajectory_ids)
        return PosteriorEventProvenance(batches=(*self.batches, batch))

    def to_value(self) -> dict[str, JsonValue]:
        return {"batches": [batch.to_value() for batch in self.batches], "format": self.format}

    @classmethod
    def from_value(cls, value: object) -> PosteriorEventProvenance:
        if not isinstance(value, dict) or value.get("format") != POSTERIOR_PROVENANCE_FORMAT:
            raise ValueError(
                "old ID-only posterior snapshots need their original event journal; "
                "automatic migration is unsupported"
            )
        return cls(
            batches=tuple(PosteriorEvidenceBatch.from_value(batch) for batch in value["batches"])
        )
