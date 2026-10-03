from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TextIO

from r2flow.benchmarks.training_records import TrainingRecord
from r2flow.benchmarks.training_schedule import TRAINING_DOMAINS

from .quality_panel import FixedQualityPanel, PanelSlot

SPLIT_FORMAT = "skillev-quality-source-split@1"
SPLIT_ALGORITHM = "canonical-source-seed0-holdout4-filtered-lane-cycle250@1"
SourceIdentity = tuple[str, str]
ProvenanceIdentity = tuple[str, str, str]


def source_identity(record: TrainingRecord) -> SourceIdentity:
    return record.episode.benchmark.value, record.episode.source_id


def _provenance_value(source: ProvenanceIdentity) -> dict[str, str]:
    return dict(zip(("benchmark_id", "population_id", "source_question_id"), source, strict=True))


@dataclass(frozen=True)
class QualitySourceSplit:
    training: tuple[TrainingRecord, ...]
    panel: tuple[TrainingRecord, ...]
    retained_occurrences: tuple[TrainingRecord, ...]
    heldout_occurrences: tuple[TrainingRecord, ...]
    final_evaluation_sources: frozenset[ProvenanceIdentity]
    summary: dict[str, object]

    def require_disjoint(self) -> None:
        training_sources = {source_identity(r) for r in self.training}
        retained_sources = {source_identity(r) for r in self.retained_occurrences}
        panel_sources = {source_identity(r) for r in self.panel}
        heldout_sources = {source_identity(r) for r in self.heldout_occurrences}
        final_sources = {(d, s) for d, _, s in self.final_evaluation_sources}
        if (training_sources | retained_sources) & (panel_sources | heldout_sources):
            raise ValueError("quality and training share a canonical source")
        if panel_sources != heldout_sources:
            raise ValueError("quality source assignment differs")
        if (training_sources | retained_sources | panel_sources | heldout_sources) & final_sources:
            raise ValueError("source pool overlaps final evaluation")
        if not training_sources <= retained_sources:
            raise ValueError("training contains a source outside the retained pool")


def plan_quality_split(
    records: tuple[TrainingRecord, ...],
    *,
    seed: int = 0,
    sources_per_domain: int = 4,
    training_occurrences_per_domain: int = 250,
    final_evaluation_sources: frozenset[ProvenanceIdentity] = frozenset(),
) -> QualitySourceSplit:
    if (type(seed), seed) != (int, 0):
        raise ValueError("quality source selection uses the single declared seed zero")
    if (type(sources_per_domain), sources_per_domain) != (int, 4):
        raise ValueError("quality source selection requires four sources per domain")
    if (type(training_occurrences_per_domain), training_occurrences_per_domain) != (int, 250):
        raise ValueError("the planned training schedule has 250 question occurrences per domain")
    domains = {domain.value for domain in TRAINING_DOMAINS}
    if {r.episode.benchmark.value for r in records} != domains:
        raise ValueError("quality splitting requires exactly the six source domains")
    excluded = {(domain, source) for domain, _, source in final_evaluation_sources}
    grouped: dict[SourceIdentity, list[TrainingRecord]] = {}
    for record in records:
        grouped.setdefault(source_identity(record), []).append(record)
    if set(grouped) & excluded:
        raise ValueError("input source pool overlaps final evaluation; no automatic replacement")
    rng = random.Random(seed)
    panel: list[TrainingRecord] = []
    selected: set[SourceIdentity] = set()
    for domain in TRAINING_DOMAINS:
        sources = sorted(key for key in grouped if key[0] == domain.value)
        if len(sources) <= sources_per_domain:
            raise ValueError("a domain lacks four independent holdouts and a remaining source")
        chosen = rng.sample(sources, sources_per_domain)
        selected.update(chosen)
        panel.extend(grouped[key][0] for key in chosen)
    if len({r.input.task_id for r in panel}) != len(panel):
        raise ValueError("original panel representative task IDs are not unique")
    retained = tuple(r for r in records if source_identity(r) not in selected)
    heldout = tuple(r for r in records if source_identity(r) in selected)
    lanes = {
        domain: tuple(r for r in retained if r.episode.benchmark is domain)
        for domain in TRAINING_DOMAINS
    }
    repeats: Counter[SourceIdentity] = Counter()
    training: list[TrainingRecord] = []
    for step in range(training_occurrences_per_domain):
        for slot, domain in enumerate(TRAINING_DOMAINS):
            lane = lanes[domain]
            original = lane[step % len(lane)]
            source = source_identity(original)
            occurrence_id = f"{SPLIT_ALGORITHM}/training/step-{step + 1:03d}/question-{slot:02d}"
            episode = replace(
                original.episode,
                episode_id=occurrence_id,
                repeat_ordinal=repeats[source],
                block_position=step,
                optimizer_step=step + 1,
                global_position=len(training),
            )
            repeats[source] += 1
            training.append(
                replace(
                    original, episode=episode, input=replace(original.input, task_id=occurrence_id)
                )
            )
    domain_summary = {}
    for domain in TRAINING_DOMAINS:
        original_lane = tuple(r for r in records if r.episode.benchmark is domain)
        after_lane = tuple(r for r in training if r.episode.benchmark is domain)
        original_counts = Counter(source_identity(r) for r in original_lane)
        after_counts = Counter(source_identity(r) for r in after_lane)
        domain_summary[domain.value] = {
            "original_occurrences": len(original_lane),
            "original_sources": len(original_counts),
            "heldout_sources": sources_per_domain,
            "heldout_occurrences": sum(source_identity(r) in selected for r in original_lane),
            "retained_sources": len(original_counts) - sources_per_domain,
            "retained_occurrences": len(lanes[domain]),
            "scheduled_training_occurrences": len(after_lane),
            "scheduled_training_sources": len(after_counts),
            "changed_source_positions_vs_original_lane_cycle": sum(
                source_identity(r) != source_identity(original_lane[i % len(original_lane)])
                for i, r in enumerate(after_lane)
            ),
            "sources": [
                {
                    "source_question_id": source[1],
                    "population_ids": sorted({r.episode.population_id for r in grouped[source]}),
                    "side": "panel" if source in selected else "training",
                    "original_occurrences": original_counts[source],
                    "scheduled_training_occurrences": after_counts[source],
                }
                for source in sorted(original_counts)
            ],
        }
    summary: dict[str, object] = {
        "format": SPLIT_FORMAT,
        "selection_algorithm": SPLIT_ALGORITHM,
        "seed": seed,
        "independence_key": ["benchmark", "canonical_source_id"],
        "population_labels_are_provenance_only": True,
        "training_order": "six-domain-step-major; original-filtered-lane cyclic order",
        "panel_order": "six-domain-major; seed-zero sample order; first occurrence per source",
        "panel_sources": len(panel),
        "training_question_occurrences": len(training),
        "training_steps": training_occurrences_per_domain,
        "effective_batch_size": 28,
        "training_trajectories_after_existing_expansion": len(training) * 4,
        "domains": domain_summary,
    }
    result = QualitySourceSplit(
        tuple(training), tuple(panel), retained, heldout, final_evaluation_sources, summary
    )
    result.require_disjoint()
    return result


def _private_file(path: Path) -> TextIO:
    return os.fdopen(
        os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8"
    )


def write_quality_split(
    split: QualitySourceSplit,
    root: Path,
    *,
    panel_id: str,
    condition_id: str,
) -> Path:
    split.require_disjoint()
    FixedQualityPanel(
        panel_id,
        condition_id,
        tuple(
            PanelSlot(
                r.input.task_id,
                r.episode.benchmark.value,
                r.episode.population_id,
                r.episode.source_id,
            )
            for r in split.panel
        ),
    )
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    for filename, records in (
        ("training.jsonl", split.training),
        ("panel-records.jsonl", split.panel),
        ("retained-occurrences.jsonl", split.retained_occurrences),
        ("heldout-occurrences.jsonl", split.heldout_occurrences),
    ):
        with _private_file(root / filename) as stream:
            for record in records:
                stream.write(
                    json.dumps(record.to_value(), ensure_ascii=False, allow_nan=False) + "\n"
                )
    with _private_file(root / "summary.json") as stream:
        json.dump(split.summary, stream, ensure_ascii=False, allow_nan=False, indent=2)
    manifest = root / "quality-panel.json"
    with _private_file(manifest) as stream:
        json.dump(
            {
                "format": SPLIT_FORMAT,
                "records": "panel-records.jsonl",
                "panel_id": panel_id,
                "condition_id": condition_id,
                "final_evaluation_sources": [
                    _provenance_value(source) for source in sorted(split.final_evaluation_sources)
                ],
                "sampling_schedule_id": SPLIT_ALGORITHM,
                "ordered_task_sequence_id": panel_id + "/six-domain-source-order@1",
            },
            stream,
            allow_nan=False,
            indent=2,
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare, but never apply, a source-disjoint private quality split."
    )
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument(
        "--final-evaluation-sources",
        type=Path,
        required=True,
        help="Explicit JSON list of benchmark/population/source coordinates; may be empty",
    )
    parser.add_argument(
        "--output", type=Path, help="New private directory; omit for count-only planning"
    )
    parser.add_argument("--panel-id", required=True)
    parser.add_argument("--condition-id", required=True)
    args = parser.parse_args()
    with args.records.open(encoding="utf-8") as stream:
        records = tuple(
            TrainingRecord.from_value(json.loads(line)) for line in stream if line.strip()
        )
    exclusions = json.loads(args.final_evaluation_sources.read_text(encoding="utf-8"))
    split = plan_quality_split(
        records,
        final_evaluation_sources=frozenset(
            (s["benchmark_id"], s["population_id"], s["source_question_id"]) for s in exclusions
        ),
    )
    if args.output is not None:
        write_quality_split(
            split, args.output, panel_id=args.panel_id, condition_id=args.condition_id
        )
    print(
        json.dumps(
            {
                "format": SPLIT_FORMAT,
                "panel_sources": len(split.panel),
                "training_question_occurrences": len(split.training),
                "written": args.output is not None,
            }
        )
    )


if __name__ == "__main__":
    main()
