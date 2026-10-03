from dataclasses import dataclass

from skillev.contracts import JsonValue
from skillev.evaluation.training_domains.catalog import TrainingBenchmark
from skillev.rollout import RolloutTask


@dataclass(frozen=True, slots=True)
class EvaluationSource:
    benchmark: TrainingBenchmark
    population_id: str
    source_id: str
    episode_id: str


@dataclass(frozen=True, slots=True)
class EvaluationTarget:
    target: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class EvaluationEpisodeRecord:
    episode: EvaluationSource
    input: RolloutTask
    output: EvaluationTarget

    def __post_init__(self) -> None:
        if self.input.task_id != self.episode.episode_id:
            raise ValueError("evaluation task identity differs from its frozen source slot")
        if not all((self.episode.source_id, self.episode.population_id)):
            raise ValueError("evaluation source identity must be explicit")
