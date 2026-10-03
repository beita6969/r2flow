from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

from r2flow.experiments.validation_pool import VALIDATION_POOL_ALGORITHM, VALIDATION_POOL_FORMAT
from r2flow.experiments.vq_heldout import VQ_HELDOUT_FORMAT
from skillev.training.r2flow_config import R2FLOW_HELDOUT_SPLIT
from skillev.training.r2flow_evolution_config import DEDICATED_VALIDATION_POOL
from r2flow.benchmarks.training_records import (
    TrainingEpisode,
    TrainingOutput,
    TrainingRecord,
)
from skillev.evaluation.training_domains.catalog import TrainingBenchmark
from skillev.rollout import ModelVisibleMessage, RolloutTask

DOMAINS = ("hotpotqa", "triviaqa", "aime-2026", "healthbench", "mbpp-plus", "alfworld")
PER_DOMAIN = 512
TEST = 128
VALIDATION = 16
HELDOUT = 4
STEPS = 250
DIRECT = ("hotpotqa", "triviaqa")
ALFWORLD_CONFIG = "configs/alfworld/base_config.yaml"
HEALTHBENCH_FILE = "healthbench_oss_eval.jsonl"
MBPP_FILE = "MbppPlus-v0.2.0.jsonl.gz"
ALFWORLD_TASK_TYPES = frozenset(
    {
        "pick_and_place_simple",
        "look_at_obj_in_light",
        "pick_clean_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_two_obj_and_place",
    }
)
VERSIONS = {
    "hotpotqa": "hotpotqa/hotpot_qa@1908d6afbbead072334abe2965f91bd2709910ab:distractor",
    "triviaqa": "mandarjoshi/trivia_qa@0f7faf33a3908546c6fd5b73a660e0f8ff173c2f:rc.nocontext",
    "aime-2026": "aime-1983-2026",
    "healthbench": "openai-healthbench-2025-05-07",
    "mbpp-plus": "evalplus-mbppplus-v0.2.0",
    "alfworld": "alfworld-json_2.1.1",
}
EVALUATORS = {
    "hotpotqa": "hotpotqa-official-em-f1",
    "triviaqa": "triviaqa-official-alias-em-f1",
    "aime-2026": "integer-exact",
    "healthbench": "simple-evals-rubric",
    "mbpp-plus": "evalplus-base-plus",
    "alfworld": "alfworld-success",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^0-9a-z]+", " ", text.lower()).split())


def grams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = norm(text).split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def rank(split: str, domain: str, source_id: str) -> str:
    return hashlib.sha256(f"{split}:{domain}:{source_id}".encode()).hexdigest()


def finite(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        token = "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")
        return {"format": "r2flow-nonfinite-float@1", "value": token}
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite(item) for item in value]
    return value


def item(
    source_id,
    question,
    query,
    family,
    payload,
    target,
    messages=(),
    tools=(),
    suffix=None,
    statement=None,
):
    return {
        "source_id": source_id,
        "question": question,
        "statement": statement,
        "query": query,
        "family": family,
        "payload": payload,
        "target": target,
        "messages": tuple(messages),
        "tools": tuple(tools),
        "suffix": suffix,
    }


def hotpotqa(raw: Path, names: list[str]) -> list[dict]:
    items = []
    for name in names:
        for row in read_parquet(raw / name):
            answer = row["answer"].strip()
            if not answer or "\n" in answer:
                continue
            documents = [
                (title, " ".join(sentences))
                for title, sentences in zip(
                    row["context"]["title"], row["context"]["sentences"], strict=True
                )
            ]
            passages = "\n\n".join(f"[[{title}] {text}]" for title, text in documents)
            evidence = "\n\n".join(f"[{title}] {text}" for title, text in documents)
            query = (
                f"Based on the following passages, answer the question.\n\n{passages}"
                f"\n\nQuestion: {row['question']}\n\nEvidence:\n{evidence}"
            )
            target = {
                "accepted_answers": [answer],
                "supporting_facts": {
                    "sent_id": list(row["supporting_facts"]["sent_id"]),
                    "title": list(row["supporting_facts"]["title"]),
                },
            }
            items.append(
                item(
                    f"hotpotqa:{row['id']}",
                    row["question"],
                    query,
                    "multi-hop-qa",
                    {"context_in_query": True},
                    target,
                )
            )
    return items


def triviaqa(raw: Path, name: str) -> list[dict]:
    by_id: dict[str, dict] = {}
    conflicting: set[str] = set()
    for row in read_parquet(raw / name):
        answer = row["answer"]
        answers = list(
            dict.fromkeys(a for a in [answer["value"], *answer["aliases"]] if a and a.strip())
        )
        if not answers:
            continue
        found = item(
            f"triviaqa:{row['question_id']}",
            row["question"],
            row["question"].strip(),
            "factual-qa",
            {"initial_context": "none"},
            {"accepted_answers": answers},
        )
        if by_id.setdefault(row["question_id"], found) != found:
            conflicting.add(row["question_id"])
    return [found for qid, found in sorted(by_id.items()) if qid not in conflicting]


def aime_item(source_id: str, problem: str, answer: str, slice_name: str) -> dict:
    return item(
        source_id,
        problem,
        problem.strip(),
        "integer-answer",
        {"benchmark_slice": slice_name},
        {"accepted_answers": [str(int(answer))]},
    )


def aime_history(raw: Path) -> list[dict]:
    found: dict[tuple[int, str, int], tuple[str, str]] = {}
    for row in csv.DictReader((raw / "aime_1983_2024.csv").open(encoding="utf-8")):
        part = (row.get("Part") or "").strip()
        found[(int(row["Year"]), part, int(row["Problem Number"]))] = (
            row["Question"],
            row["Answer"].strip(),
        )
    for row in read_parquet(raw / "aimo_validation_aime.parquet"):
        match = re.search(r"/(\d{4})_AIME_(I{1,2})_Problems/Problem_(\d+)", row["url"])
        found[(int(match[1]), match[2], int(match[3]))] = (
            row["problem"],
            str(row["answer"]).strip(),
        )
    for row in read_parquet(raw / "aime_2025.parquet"):
        index = int(row["problem_idx"])
        part, number = ("I", index) if index <= 15 else ("II", index - 15)
        found[(2025, part, number)] = (row["problem"], str(row["answer"]).strip())
    items = []
    for (year, part, number), (problem, answer) in sorted(found.items()):
        if year >= 2026 or not re.fullmatch(r"\d{1,3}", answer):
            continue
        label = f"{year}:{part.lower()}:{number:02d}" if part else f"{year}:{number:02d}"
        items.append(aime_item(f"aime:{label}", problem, answer, "pre-2026"))
    return items


def aime_2026(raw: Path) -> list[dict]:
    rows = read_parquet(raw / "aime_2026.parquet")
    if sorted(int(row["problem_idx"]) for row in rows) != list(range(1, 31)):
        raise SystemExit("expected the 30 AIME 2026 problems")
    return [
        aime_item(
            f"aime:2026:{int(row['problem_idx']):02d}",
            row["problem"],
            str(row["answer"]).strip(),
            "2026",
        )
        for row in sorted(rows, key=lambda row: int(row["problem_idx"]))
    ]


def healthbench(raw: Path) -> list[dict]:
    items = []
    for line in (raw / HEALTHBENCH_FILE).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        messages = [
            ModelVisibleMessage(role=m["role"], content=m["content"]) for m in row["prompt"]
        ]
        query = "\n\n".join(f"{m['role'].title()}: {m['content']}" for m in row["prompt"])
        target = {
            "grader_kind": "healthbench-qwen35-local-simple-evals",
            "prompt": row["prompt"],
            "rubrics": row["rubrics"],
        }
        items.append(
            item(
                row["prompt_id"],
                query,
                query,
                "health-dialogue",
                {"message_count": len(messages)},
                target,
                messages,
            )
        )
    if len(items) != 5000 or len({i["source_id"] for i in items}) != 5000:
        raise SystemExit("HealthBench must hold 5,000 distinct conversations")
    return items


def mbpp_statement(prompt: str) -> str:
    text = prompt.strip().strip('"').strip()
    text = text.split("\nassert ", 1)[0]
    first = re.split(r"(?<=[.?!])\s", text.strip(), maxsplit=1)[0]
    return re.sub(r"\d+", "", norm(first)).strip()


def mbpp_plus(raw: Path) -> list[dict]:
    items = []
    with gzip.open(raw / MBPP_FILE, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            target = finite(
                {
                    key: row[key]
                    for key in (
                        "assertion",
                        "atol",
                        "base_input",
                        "canonical_solution",
                        "contract",
                        "entry_point",
                        "plus_input",
                    )
                }
            )
            payload = {"language": "python", "test_suite": "evalplus-base-plus-v0.2.0"}
            text = row["prompt"].strip().strip('"').strip()
            items.append(
                item(
                    row["task_id"],
                    text,
                    row["prompt"],
                    "code-generation",
                    payload,
                    target,
                    statement=mbpp_statement(row["prompt"]),
                )
            )
    if len(items) != 378:
        raise SystemExit("MBPP+ v0.2.0 must hold 378 tasks")
    return items


def alfworld_catalog(data: Path, split: str) -> list[tuple[str, str]]:
    root = data / "json_2.1.1" / split
    if not root.is_dir():
        raise SystemExit(f"missing {root}")
    games = []
    for directory, _, names in os.walk(root):
        if "traj_data.json" not in names or "movable" in directory or "Sliced" in directory:
            continue
        traj = json.loads((Path(directory) / "traj_data.json").read_text(encoding="utf-8"))
        if traj["task_type"] not in ALFWORLD_TASK_TYPES:
            continue
        game = Path(directory) / "game.tw-pddl"
        if not game.is_file() or not json.loads(game.read_text(encoding="utf-8")).get(
            "solvable", False
        ):
            continue
        games.append((str(game), traj["task_type"]))
    return sorted(games)


def alfworld(data: Path, split: str, mode: str) -> list[dict]:
    items = []
    for index, (game, task_type) in enumerate(alfworld_catalog(data, split)):
        relative = game[game.index("json_2.1.1/") :]
        route = {
            "config_file": ALFWORLD_CONFIG,
            "game_file": relative,
            "max_steps": 50,
            "mode": mode,
            "seed": index,
        }
        items.append(
            item(
                f"alfworld:{relative[len('json_2.1.1/') :].rsplit('/', 1)[0]}",
                None,
                None,
                task_type,
                {"max_steps": 50, "observation_format": "official-text"},
                {"environment_route": route, "target_won": True},
                tools=("act",),
                suffix="official-environment",
            )
        )
    return items


def ranked(split: str, domain: str, items: list[dict]) -> list[dict]:
    return sorted(items, key=lambda i: rank(split, domain, i["source_id"]))


def unique(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    kept = []
    for found in items:
        key = norm(found["question"]) if found["question"] else found["source_id"]
        if key not in seen:
            seen.add(key)
            kept.append(found)
    return kept


def disjoint(domain: str, train: list[dict], test: list[dict]) -> tuple[list[dict], int]:
    ids = {t["source_id"] for t in test}
    texts = {norm(t["question"]) for t in test if t["question"]}
    statements = {t["statement"] for t in test if t["statement"]}
    held = [grams(t["question"]) for t in test if t["question"]] if domain == "aime-2026" else []
    kept = []
    for found in train:
        if found["source_id"] in ids or (found["question"] and norm(found["question"]) in texts):
            continue
        if found["statement"] and found["statement"] in statements:
            continue
        if held:
            mine = grams(found["question"])
            if any(len(mine & other) >= 0.5 * max(1, min(len(mine), len(other))) for other in held):
                continue
        kept.append(found)
    return kept, len(train) - len(kept)


def record(domain: str, found: dict, split: str, index: int) -> TrainingRecord:
    benchmark = TrainingBenchmark(domain)
    episode_id = f"r2flow/{domain}/{split}/{index:04d}"
    environment = f"benchmark:{domain}@{VERSIONS[domain]}"
    if found["suffix"]:
        environment += f":{found['suffix']}"
    task = RolloutTask(
        task_id=episode_id,
        environment_id=environment,
        task_family=f"{domain}/{found['family']}",
        context_id=f"{domain}:{split}",
        query=found["query"],
        available_tools=found["tools"],
        public_context={
            "benchmark_id": domain,
            "dataset_revision": VERSIONS[domain],
            "payload": found["payload"],
            "split": split,
        },
        model_visible_messages=found["messages"],
    )
    episode = TrainingEpisode(
        benchmark=benchmark,
        population_id=f"{domain}-{split}-r2flow",
        episode_id=episode_id,
        source_id=found["source_id"],
        repeat_ordinal=index // STEPS,
        block_position=index % STEPS,
        optimizer_step=index % STEPS + 1,
        global_position=index,
    )
    return TrainingRecord(
        episode=episode,
        input=task,
        output=TrainingOutput(EVALUATORS[domain], found["target"]),
    )


def write_jsonl(path: Path, records: list[TrainingRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in records:
            handle.write(
                json.dumps(value.to_value(), ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sources(rows: list[TrainingRecord]) -> list[list[str]]:
    return [[r.episode.benchmark.value, r.episode.source_id] for r in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--alfworld-data", type=Path, required=True)
    parser.add_argument("--alfworld-goals", type=Path)
    parser.add_argument("--alfworld-python", default=sys.executable)
    parser.add_argument("--alfworld-config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-domain", type=int, default=PER_DOMAIN)
    parser.add_argument("--games-out", type=Path)
    args = parser.parse_args()
    raw, out = args.raw, args.out
    builders = {
        "hotpotqa": (
            lambda: hotpotqa(raw, ["hotpotqa_train_0.parquet", "hotpotqa_train_1.parquet"]),
            lambda: hotpotqa(raw, ["hotpotqa_distractor_validation.parquet"]),
        ),
        "triviaqa": (
            lambda: triviaqa(raw, "triviaqa_rc_nocontext_train.parquet"),
            lambda: triviaqa(raw, "triviaqa_rc_nocontext_validation.parquet"),
        ),
        "aime-2026": (lambda: aime_history(raw), lambda: aime_2026(raw)),
        "healthbench": (lambda: healthbench(raw), None),
        "mbpp-plus": (lambda: mbpp_plus(raw), None),
        "alfworld": (
            lambda: alfworld(args.alfworld_data, "train", "train"),
            lambda: alfworld(args.alfworld_data, "valid_unseen", "eval_out_of_distribution"),
        ),
    }
    held_out: dict[str, list[dict]] = {}
    ood = out / "test" / "ood"
    if ood.is_dir():
        for path in sorted(ood.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("question"):
                    held_out.setdefault(row["task_type"], []).append(
                        {
                            "source_id": f"ood:{row['task_id']}",
                            "question": row["question"],
                            "statement": None,
                        }
                    )
    plan: dict[str, dict[str, list[dict]]] = {}
    report: dict[str, dict[str, int]] = {}
    for domain in DOMAINS:
        train_builder, test_builder = builders[domain]
        candidates = train_builder()
        if test_builder is None:
            test = ranked("test", domain, candidates)[:TEST]
            pool = [c for c in candidates if c["source_id"] not in {t["source_id"] for t in test}]
        else:
            test = ranked("test", domain, unique(test_builder()))
            test = test if domain == "aime-2026" else test[:TEST]
            pool = candidates
        if len(test) != (30 if domain == "aime-2026" else TEST):
            raise SystemExit(f"{domain}: only {len(test)} test items")
        distinct = unique(pool)
        kept, overlap = disjoint(domain, distinct, test + held_out.get(domain, []))
        order = ranked("train", domain, kept)
        held = order[: HELDOUT + VALIDATION]
        train = order[HELDOUT + VALIDATION :][: args.per_domain]
        if len(train) < args.per_domain:
            train = [train[i % len(train)] for i in range(args.per_domain)]
        plan[domain] = {
            "test": test,
            "heldout": held[:HELDOUT],
            "validation": held[HELDOUT:],
            "train": train,
        }
        report[domain] = {
            "candidates": len(candidates),
            "unique": len(distinct),
            "test_overlap_removed": overlap,
            "test": len(test),
            "vq_heldout": HELDOUT,
            "validation": VALIDATION,
            "train_rows": len(train),
            "train_distinct": len({t["source_id"] for t in train}),
        }
    games = [
        tuple(f["target"]["environment_route"][k] for k in ("game_file", "mode", "seed"))
        for part in plan["alfworld"].values()
        for f in part
    ]
    goals_file = args.alfworld_goals or out / "train" / "alfworld-goals.json"
    if not goals_file.is_file():
        listing = args.games_out or out / "alfworld-games.tsv"
        listing.parent.mkdir(parents=True, exist_ok=True)
        listing.write_text(
            "".join(
                f"{args.alfworld_data / g}\t{m}\t{seed}\n" for g, m, seed in dict.fromkeys(games)
            ),
            encoding="utf-8",
        )
        goals_file.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            [
                args.alfworld_python,
                str(Path(__file__).with_name("alfworld_goals.py")),
                "--config",
                str(
                    args.alfworld_config
                    or Path(__file__).parents[2] / "configs/alfworld/base_config.yaml"
                ),
                "--games",
                str(listing),
                "--out",
                str(goals_file),
            ],
            env={**os.environ, "ALFWORLD_DATA": str(args.alfworld_data)},
        )
    goals = json.loads(goals_file.read_text(encoding="utf-8"))
    for part in plan["alfworld"].values():
        for found in part:
            found["query"] = found["question"] = goals[
                found["target"]["environment_route"]["game_file"]
            ]
    train_dir, iid_dir = out / "train", out / "test" / "iid"
    train_dir.mkdir(parents=True, exist_ok=True)
    iid_dir.mkdir(parents=True, exist_ok=True)
    records = {name: [] for name in ("test", "heldout", "validation", "train")}
    for domain in DOMAINS:
        for name, rows in plan[domain].items():
            split = "test" if name == "test" else "training"
            seen: dict[str, TrainingRecord] = {}
            for index, found in enumerate(rows):
                if name == "train" and found["source_id"] in seen:
                    continue
                seen[found["source_id"]] = record(domain, found, split, index)
            records[name].extend(seen.values())
    for domain in DOMAINS:
        write_jsonl(
            iid_dir / f"{domain}.jsonl",
            [r for r in records["test"] if r.episode.benchmark.value == domain],
        )
    write_jsonl(train_dir / "training.jsonl", records["train"])
    heldout_sources = sources(records["heldout"])
    pool_sources = sources(records["validation"])
    exclusions = {
        "iid": sources(records["test"]),
        "development": pool_sources,
        "quality": heldout_sources,
    }
    ordered = [
        {
            "benchmark": benchmark,
            "source_id": source_id,
            "role": "direct-control" if benchmark in DIRECT else "procedure-applicable",
            "method_family": "public-task-family",
            "public_basis": "Seeded sha256 rank over the public training split; no outcome selection.",
        }
        for benchmark, source_id in sources(records["train"])
    ]
    write_json(
        train_dir / "data-condition.json",
        {
            "format": "r2flow-data-condition@1",
            "seed": 0,
            "source_selection": "sha256-rank-train-split-disjoint-from-iid-test@1",
            "autonomous_ttb_sources": {
                "format": "public-task-needs@1",
                "ordered_sources": ordered,
                "source_aliases": {},
                "excluded_sources": exclusions,
            },
        },
    )
    write_json(
        train_dir / "training-sources.json",
        {
            "format": "r2flow-training-sources@1",
            "training": sources(records["train"]),
            "source_aliases": {},
            "excluded_sources": {
                **exclusions,
                "vq_heldout": heldout_sources,
                "validation_pool": pool_sources,
            },
        },
    )
    vq_path = train_dir / "vq-heldout.json"
    write_json(
        vq_path,
        {
            "format": VQ_HELDOUT_FORMAT,
            "selection_algorithm": R2FLOW_HELDOUT_SPLIT,
            "existing_sources": sorted(heldout_sources),
            "extra_sources": [],
            "heldout_records": [r.to_value() for r in records["heldout"]],
            "training_records": [],
            "summary": {
                "selection_algorithm": R2FLOW_HELDOUT_SPLIT,
                "per_domain": HELDOUT,
                "domains": list(DOMAINS),
            },
        },
    )
    write_json(
        train_dir / "validation-pool.json",
        {
            "format": VALIDATION_POOL_FORMAT,
            "selection_algorithm": VALIDATION_POOL_ALGORITHM,
            "validation_query_selection": DEDICATED_VALIDATION_POOL,
            "seed": 0,
            "domains": list(DOMAINS),
            "per_domain": VALIDATION,
            "pool_sources": sorted(pool_sources),
            "vq_heldout_sha256": sha256_file(vq_path),
            "heldout_records": [r.to_value() for r in records["validation"]],
            "summary": {"per_domain": VALIDATION, "domains": list(DOMAINS)},
        },
    )
    files = sorted(p for p in (*train_dir.glob("*.json*"), *iid_dir.glob("*.jsonl")))
    summary = {
        "domains": report,
        "steps": STEPS,
        "files": {str(p.relative_to(out)): sha256_file(p) for p in files},
    }
    write_json(out / "summary.json", summary)
    for domain, counts in report.items():
        print(domain, " ".join(f"{k}={v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
