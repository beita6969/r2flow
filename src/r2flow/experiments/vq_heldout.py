from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Final

from r2flow.benchmarks.training_records import TrainingRecord
from skillev.training.vq_monitor import VQ_TASK_PREFIX, vq_query_key


VQ_HELDOUT_FORMAT: Final = "r2flow-vq-heldout@1"


def vq_task_id(record: TrainingRecord, k: int) -> str:
    return f"{VQ_TASK_PREFIX}/{record.episode.benchmark.value}/{record.episode.source_id}/r{k}"


def replicate_for_vq(records: Sequence[TrainingRecord], m_q: int) -> tuple[TrainingRecord, ...]:
    if type(m_q) is not int or m_q < 2:
        raise ValueError("V_q needs M_q >= 2 rollouts per query")
    return tuple(
        replace(
            r,
            episode=replace(r.episode, episode_id=vq_task_id(r, k)),
            input=replace(r.input, task_id=vq_task_id(r, k)),
        )
        for r in records
        for k in range(m_q)
    )


def load_vq_heldout_records(
    path: Path, domains: tuple[str, ...] | None = None
) -> tuple[TrainingRecord, ...]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format") != VQ_HELDOUT_FORMAT:
        raise ValueError("unsupported V_q held-out file")
    records = tuple(TrainingRecord.from_value(r) for r in value["heldout_records"])
    if domains is None:
        return records
    return tuple(r for r in records if r.episode.benchmark.value in domains)


__all__ = [
    "VQ_HELDOUT_FORMAT",
    "VQ_TASK_PREFIX",
    "load_vq_heldout_records",
    "replicate_for_vq",
    "vq_query_key",
    "vq_task_id",
]
