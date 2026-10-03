from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from skillev.evaluation.training_domains.catalog import TrainingBenchmark

from .training_records import TrainingRecord

DOMAIN_CONDITION = "domain-balanced-6x4-seed0@1"
QUESTIONS_PER_STEP = 6
TRAJECTORIES_PER_QUESTION = 4
EFFECTIVE_BATCH_SIZE = QUESTIONS_PER_STEP * TRAJECTORIES_PER_QUESTION
TRAINING_DOMAINS = (
    TrainingBenchmark.HOTPOT_QA,
    TrainingBenchmark.TRIVIA_QA,
    TrainingBenchmark.AIME_2026,
    TrainingBenchmark.HEALTHBENCH,
    TrainingBenchmark.ALF_WORLD,
    TrainingBenchmark.MBPP_PLUS,
)


def domain_schedule_condition(domains: tuple[TrainingBenchmark, ...]) -> str:
    if domains == TRAINING_DOMAINS:
        return DOMAIN_CONDITION
    return "balanced-domain-subset-4x-seed0@1:" + ",".join(d.value for d in domains)


def load_training_sources(path: Path) -> tuple[TrainingRecord, ...]:
    with path.open(encoding="utf-8") as stream:
        records = (TrainingRecord.from_value(json.loads(line)) for line in stream if line.strip())
        return tuple(r for r in records if r.episode.benchmark in TRAINING_DOMAINS)


def training_trajectories(
    records: tuple[TrainingRecord, ...],
    *,
    steps: int,
    domain_starts: Mapping[int, tuple[TrainingBenchmark, ...]] | None = None,
) -> tuple[TrainingRecord, ...]:
    if type(steps) is not int or not 1 <= steps <= 250:
        raise ValueError("six-domain training requires 1 through 250 complete steps")
    starts = dict(domain_starts or {1: TRAINING_DOMAINS})
    if 1 not in starts or any(type(step) is not int or not 1 <= step <= steps for step in starts):
        raise ValueError("domain schedule must start at the first optimizer step")
    for domains in starts.values():
        if not domains or tuple(d for d in TRAINING_DOMAINS if d in domains) != domains:
            raise ValueError("domains must be a nonempty canonical subset without duplicates")
    required = set().union(*starts.values())
    lanes = {
        domain: tuple(record for record in records if record.episode.benchmark is domain)
        for domain in TRAINING_DOMAINS
        if domain in required
    }
    if any(not lane for lane in lanes.values()):
        raise ValueError("six-domain training requires a source lane for every allowed domain")
    cursors: Counter[TrainingBenchmark] = Counter()
    repeats: Counter[tuple[TrainingBenchmark, str]] = Counter()
    result = []
    for step in range(steps):
        active = starts[max(start for start in starts if start <= step + 1)]
        questions = []
        for slot, domain in enumerate(TRAINING_DOMAINS):
            if domain not in active:
                continue
            lane = lanes[domain]
            source = lane[cursors[domain] % len(lane)]
            cursors[domain] += 1
            key = domain, source.episode.source_id
            occurrence = repeats[key]
            repeats[key] += 1
            episode_id = f"{DOMAIN_CONDITION}/step-{step + 1:03d}/question-{slot:02d}"
            episode = replace(
                source.episode,
                episode_id=episode_id,
                repeat_ordinal=occurrence,
                block_position=step,
                optimizer_step=step + 1,
                global_position=step * QUESTIONS_PER_STEP + slot,
            )
            questions.append(
                replace(source, episode=episode, input=replace(source.input, task_id=episode_id))
            )
        for rollout in range(TRAJECTORIES_PER_QUESTION):
            for record in questions:
                task_id = f"{record.episode.episode_id}/rollout-{rollout:02d}"
                result.append(
                    replace(
                        record,
                        episode=replace(record.episode, episode_id=task_id),
                        input=replace(record.input, task_id=task_id),
                    )
                )
    return tuple(result)
