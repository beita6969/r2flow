from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from skillev.contracts import JsonValue, normalize_json
from skillev.evaluation.training_domains.catalog import (
    QUESTIONS_PER_DOMAIN,
    TRAINING_STEPS,
    TrainingBenchmark,
)
from skillev.rollout import RolloutTask

TRAINING_RECORD_FORMAT = "r2flow-training-record@2"

_EVALUATORS: Mapping[TrainingBenchmark, str] = {
    TrainingBenchmark.HOTPOT_QA: "hotpotqa-official-em-f1",
    TrainingBenchmark.TRIVIA_QA: "triviaqa-official-alias-em-f1",
    TrainingBenchmark.AIME_2026: "integer-exact",
    TrainingBenchmark.HEALTHBENCH: "simple-evals-rubric",
    TrainingBenchmark.ALF_WORLD: "alfworld-success",
    TrainingBenchmark.MBPP_PLUS: "evalplus-base-plus",
}


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{field} must be non-empty text without NUL")
    return value


def _integer(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _object(value: object, *, field: str) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict):
        raise TypeError(f"{field} must be a JSON object")
    return normalized


@dataclass(frozen=True, slots=True)
class TrainingEpisode:
    benchmark: TrainingBenchmark
    population_id: str
    episode_id: str
    source_id: str
    repeat_ordinal: int
    block_position: int
    optimizer_step: int
    global_position: int

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, TrainingBenchmark):
            raise TypeError("training episode benchmark must belong to Protocol 13")
        for field in ("population_id", "episode_id", "source_id"):
            _text(getattr(self, field), field=field)
        for field in (
            "repeat_ordinal",
            "block_position",
            "optimizer_step",
            "global_position",
        ):
            _integer(getattr(self, field), field=field)
        if self.block_position >= QUESTIONS_PER_DOMAIN:
            raise ValueError("training episode lies outside its domain block")
        if not 1 <= self.optimizer_step <= TRAINING_STEPS:
            raise ValueError("training episode has an invalid optimizer step")
        if self.block_position != self.optimizer_step - 1:
            raise ValueError("training episode domain position differs from its optimizer step")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "benchmark": self.benchmark.value,
            "block_position": self.block_position,
            "episode_id": self.episode_id,
            "global_position": self.global_position,
            "optimizer_step": self.optimizer_step,
            "population_id": self.population_id,
            "repeat_ordinal": self.repeat_ordinal,
            "source_id": self.source_id,
        }

    @classmethod
    def from_value(cls, value: object) -> TrainingEpisode:
        data = _object(value, field="training episode")
        fields = {
            "benchmark",
            "block_position",
            "episode_id",
            "global_position",
            "optimizer_step",
            "population_id",
            "repeat_ordinal",
            "source_id",
        }
        if set(data) != fields:
            raise ValueError("training episode has incompatible fields")
        return cls(
            benchmark=TrainingBenchmark(_text(data["benchmark"], field="benchmark")),
            population_id=_text(data["population_id"], field="population_id"),
            episode_id=_text(data["episode_id"], field="episode_id"),
            source_id=_text(data["source_id"], field="source_id"),
            repeat_ordinal=_integer(data["repeat_ordinal"], field="repeat_ordinal"),
            block_position=_integer(data["block_position"], field="block_position"),
            optimizer_step=_integer(data["optimizer_step"], field="optimizer_step"),
            global_position=_integer(data["global_position"], field="global_position"),
        )


@dataclass(frozen=True, slots=True)
class TrainingOutput:
    evaluator_kind: str
    target: dict[str, JsonValue]

    def __post_init__(self) -> None:
        _text(self.evaluator_kind, field="evaluator kind")
        object.__setattr__(self, "target", _object(self.target, field="evaluator target"))

    def to_value(self) -> dict[str, JsonValue]:
        return {"evaluator_kind": self.evaluator_kind, "target": self.target}

    @classmethod
    def from_value(cls, value: object) -> TrainingOutput:
        data = _object(value, field="training output")
        if set(data) != {"evaluator_kind", "target"}:
            raise ValueError("training output has incompatible fields")
        return cls(
            evaluator_kind=_text(data["evaluator_kind"], field="evaluator kind"),
            target=_object(data["target"], field="evaluator target"),
        )


@dataclass(frozen=True, slots=True)
class TrainingRecord:
    episode: TrainingEpisode
    input: RolloutTask
    output: TrainingOutput
    format: str = TRAINING_RECORD_FORMAT

    def __post_init__(self) -> None:
        if self.format != TRAINING_RECORD_FORMAT:
            raise ValueError("unsupported Protocol 13 training record format")
        if not isinstance(self.episode, TrainingEpisode):
            raise TypeError("training record requires an episode")
        if not isinstance(self.input, RolloutTask):
            raise TypeError("training record input must be a RolloutTask")
        if not isinstance(self.output, TrainingOutput):
            raise TypeError("training record output must be verifier-only output")
        if self.input.task_id != self.episode.episode_id:
            raise ValueError("training record input differs from its episode ID")
        if self.output.evaluator_kind != _EVALUATORS[self.episode.benchmark]:
            raise ValueError("training record evaluator differs from its benchmark")
        context = self.input.public_context
        if not isinstance(context, dict) or context.get("benchmark_id") != (
            self.episode.benchmark.value
        ):
            raise ValueError("training record input belongs to another benchmark")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "episode": self.episode.to_value(),
            "format": self.format,
            "input": self.input.to_value(),
            "output": self.output.to_value(),
        }

    def model_input_value(self) -> dict[str, JsonValue]:
        return {
            "episode_id": self.episode.episode_id,
            "format": "r2flow-model-input@2",
            "input": self.input.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> TrainingRecord:
        data = _object(value, field="training record")
        if set(data) != {"episode", "format", "input", "output"}:
            raise ValueError("training record has incompatible fields")
        return cls(
            episode=TrainingEpisode.from_value(data["episode"]),
            input=RolloutTask.from_value(data["input"]),
            output=TrainingOutput.from_value(data["output"]),
            format=_text(data["format"], field="training record format"),
        )


__all__ = [
    "TRAINING_RECORD_FORMAT",
    "TrainingEpisode",
    "TrainingOutput",
    "TrainingRecord",
]
