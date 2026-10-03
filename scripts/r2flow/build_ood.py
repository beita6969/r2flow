from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path

COUNT = 128
LETTERS = ("A", "B", "C", "D")
WEBSHOP_GOAL_SHUFFLE_SEED = 233
WEBSHOP_TEST_GOALS = range(0, 500)
POSED_AS = {
    "musique": "hotpotqa",
    "nq_open": "triviaqa",
    "math_hard": "aime-2026",
    "gpqa_diamond": "healthbench",
    "swe_bench_verified": "mbpp-plus",
    "webshop": "alfworld",
}
ORIGIN = {
    "musique": "dgslibisey/MuSiQue@1bd32e8b3f4f29a11f8240a9f666ef52e159c24e:validation (MuSiQue-Ans v1.0 dev)",
    "nq_open": "google-research-datasets/nq_open@5dd9790a83002ad084ddeb7c420dc716852c6f28:validation",
    "math_hard": "lighteval/MATH-Hard@69bc17a93d6754e3c3a22084a58f6ad2c24f1f07:test (Level 5)",
    "gpqa_diamond": "Idavidrein/gpqa@83022cefff930aea54f654c0b282e74b9eeda5c6:gpqa_diamond",
    "swe_bench_verified": "princeton-nlp/SWE-bench_Verified@c104f840cc67f8b6eec6f759ebc8b2693d585d4a:test",
    "webshop": "princeton-nlp/WebShop human goals, test range 0-499 (goal shuffle seed 233)",
}
MATH_SUBJECTS = {
    "Algebra": "algebra",
    "Counting & Probability": "counting_and_probability",
    "Geometry": "geometry",
    "Intermediate Algebra": "intermediate_algebra",
    "Number Theory": "number_theory",
    "Prealgebra": "prealgebra",
    "Precalculus": "precalculus",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def read_parquet(path: Path, columns: list[str] | None = None) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=columns).to_pylist()


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^0-9a-z]+", " ", text.lower()).split())


def grams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = norm(text).split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(v for v in values if v and v.strip()))


def rank(kind: str, source_id: str) -> str:
    return hashlib.sha256(f"test:{kind}:{source_id}".encode()).hexdigest()


def select(kind: str, items: list[dict]) -> list[dict]:
    ids = [i["source_id"] for i in items]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{kind}: duplicate source ids")
    chosen = sorted(items, key=lambda i: rank(kind, i["source_id"]))[:COUNT]
    if len(chosen) != COUNT:
        raise SystemExit(f"{kind}: only {len(chosen)} candidates")
    return [
        {
            "query": None,
            "answer": None,
            **i,
            "kind": kind,
            "task_type": POSED_AS[kind],
            "task_id": f"{kind}/test-{i['source_id']}",
            "split": "test",
            "origin": ORIGIN[kind],
        }
        for i in chosen
    ]


def musique(raw: Path) -> list[dict]:
    items = []
    for row in read_parquet(raw / "musique_validation.parquet"):
        if not row["answerable"]:
            continue
        paragraphs = sorted(row["paragraphs"], key=lambda p: p["idx"])
        passages = "\n\n".join(f"[[{p['title']}] {p['paragraph_text']}]" for p in paragraphs)
        query = f"Based on the following passages, answer the question.\n\n{passages}\n\nQuestion: {row['question']}"
        answers = dedupe([row["answer"], *(row["answer_aliases"] or [])])
        items.append(
            {
                "source_id": row["id"],
                "question": row["question"],
                "query": query,
                "answer": row["answer"],
                "answers": answers,
            }
        )
    return select("musique", items)


def nq_open(raw: Path) -> list[dict]:
    items = []
    for index, row in enumerate(read_parquet(raw / "nq_open_validation.parquet")):
        answers = dedupe(list(row["answer"]))
        if not answers or any("\n" in a for a in answers):
            continue
        question = row["question"].strip()
        items.append(
            {
                "source_id": str(index),
                "question": question,
                "query": question,
                "answer": answers[0],
                "answers": answers,
            }
        )
    return select("nq_open", items)


def last_boxed(text: str) -> str | None:
    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if start < 0:
        return None
    rest = text[start:]
    if rest.startswith("\\boxed "):
        return rest[len("\\boxed ") :].split("$")[0].strip()
    open_at = rest.find("{")
    if open_at < 0:
        return None
    depth = 0
    for index in range(open_at, len(rest)):
        if rest[index] == "{":
            depth += 1
        elif rest[index] == "}":
            depth -= 1
            if depth == 0:
                return rest[open_at + 1 : index].strip()
    return None


def math_hard(raw: Path) -> list[dict]:
    items = []
    counters: dict[str, int] = {}
    for row in read_parquet(raw / "math_hard_test.parquet"):
        if row["level"] != "Level 5":
            raise SystemExit("math_hard: row outside Level 5")
        subject = MATH_SUBJECTS[row["type"]]
        index = counters.get(subject, 0)
        counters[subject] = index + 1
        answer = last_boxed(row["solution"])
        if not answer:
            raise SystemExit(f"math_hard: no boxed answer in {subject}/{index}")
        problem = row["problem"].strip()
        items.append(
            {
                "source_id": f"{subject}_{index:03d}",
                "question": problem,
                "query": problem,
                "answer": answer,
                "answers": [answer],
            }
        )
    return select("math_hard", items)


def gpqa_seed(record_id: str) -> int:
    return int(hashlib.sha256(f"options:gpqa_diamond:{record_id}".encode()).hexdigest()[:8], 16)


def gpqa_order(seed: int) -> list[int]:
    return sorted(range(4), key=lambda k: hashlib.sha256(f"{seed}:{k}".encode()).hexdigest())


def gpqa(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        records = {row["Record ID"].strip(): row for row in csv.DictReader(handle)}
    if len(records) != 198:
        raise SystemExit(f"gpqa_diamond: expected 198 records, found {len(records)}")
    items = []
    for record_id, record in records.items():
        texts = [
            record["Correct Answer"],
            record["Incorrect Answer 1"],
            record["Incorrect Answer 2"],
            record["Incorrect Answer 3"],
        ]
        seed = gpqa_seed(record_id)
        order = gpqa_order(seed)
        listed = "\n".join(f"{letter}. {texts[k].strip()}" for letter, k in zip(LETTERS, order))
        question = record["Question"].strip()
        items.append(
            {
                "source_id": record_id,
                "question": question,
                "query": f"User: {question}\n\nOptions:\n{listed}",
                "answer": LETTERS[order.index(0)],
                "answers": [LETTERS[order.index(0)]],
                "option_seed": seed,
            }
        )
    return select("gpqa_diamond", items)


def swe_bench_verified(raw: Path) -> list[dict]:
    rows = read_parquet(
        raw / "swe_bench_verified_test.parquet", ["instance_id", "repo", "base_commit"]
    )
    if len(rows) != 500:
        raise SystemExit(f"swe_bench_verified: expected 500 instances, found {len(rows)}")
    return select(
        "swe_bench_verified",
        [
            {"source_id": r["instance_id"], "repo": r["repo"], "base_commit": r["base_commit"]}
            for r in rows
        ],
    )


def product_asins(path: Path):
    try:
        import ijson
    except ImportError:
        with path.open() as handle:
            for product in json.load(handle):
                yield product["asin"]
        return
    with path.open("rb") as handle:
        yield from ijson.items(handle, "item.asin")


def webshop(raw: Path) -> list[dict]:
    with (raw / "webshop_items_human_ins.json").open() as handle:
        human = json.load(handle)
    seen: set[str] = set()
    goals = []
    for asin in product_asins(raw / "webshop_items_shuffle.json"):
        if asin == "nan" or len(asin) > 10 or asin in seen:
            continue
        seen.add(asin)
        for position, instruction in enumerate(human.get(asin, [])):
            if instruction["instruction_attributes"]:
                goals.append(
                    {
                        "asin": asin,
                        "instruction_position": position,
                        "instruction": instruction["instruction"].strip("."),
                    }
                )
    random.seed(WEBSHOP_GOAL_SHUFFLE_SEED)
    random.shuffle(goals)
    if len(goals) != 12087:
        raise SystemExit(f"webshop: expected 12087 human goals, found {len(goals)}")
    return select(
        "webshop",
        [
            {
                "source_id": str(index),
                "goal_index": index,
                "goal_shuffle_seed": WEBSHOP_GOAL_SHUFFLE_SEED,
                **goals[index],
                "query": goals[index]["instruction"],
            }
            for index in WEBSHOP_TEST_GOALS
        ],
    )


def training_overlap(kind: str, rows: list[dict], training: list[dict]) -> int:
    questions = [
        t["input"]["query"].rsplit("\n\nQuestion: ", 1)[-1].split("\n\nEvidence:\n")[0]
        for t in training
    ]
    texts = {norm(q) for q in questions}
    pool = [grams(q) for q in questions] if kind in ("math_hard", "gpqa_diamond") else []
    hits = 0
    for row in rows:
        question = row.get("question")
        if not question:
            continue
        mine = grams(question)
        if norm(question) in texts or any(
            len(mine & g) >= 0.5 * max(1, min(len(mine), len(g))) for g in pool if g
        ):
            hits += 1
    return hits


def write(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )
    return sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpqa-csv", type=Path)
    parser.add_argument("--training", type=Path)
    args = parser.parse_args()
    builders = {
        "musique": lambda: musique(args.raw),
        "nq_open": lambda: nq_open(args.raw),
        "math_hard": lambda: math_hard(args.raw),
        "swe_bench_verified": lambda: swe_bench_verified(args.raw),
        "webshop": lambda: webshop(args.raw),
    }
    if args.gpqa_csv is not None:
        builders["gpqa_diamond"] = lambda: gpqa(args.gpqa_csv)
    training: dict[str, list[dict]] = {}
    if args.training is not None:
        for line in args.training.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            training.setdefault(row["episode"]["benchmark"], []).append(row)
    summary = {}
    for kind, build in builders.items():
        rows = build()
        digest = write(args.out / f"{kind}.jsonl", rows)
        if kind == "gpqa_diamond":
            ids = {
                "kind": kind,
                "task_type": POSED_AS[kind],
                "source": ORIGIN[kind],
                "csv_sha256": sha256_file(args.gpqa_csv),
                "selection": "first 128 Record IDs by ascending sha256('test:gpqa_diamond:<Record ID>')",
                "option_order": "[Correct, Incorrect 1, Incorrect 2, Incorrect 3] sorted by sha256('<option_seed>:<k>') give letters A-D",
                "items": [
                    {"record_id": r["source_id"], "option_seed": r["option_seed"]} for r in rows
                ],
            }
            (args.out / "gpqa_diamond.ids.json").write_text(
                json.dumps(ids, indent=1) + "\n", encoding="utf-8"
            )
        overlap = (
            training_overlap(kind, rows, training.get(POSED_AS[kind], [])) if training else None
        )
        summary[kind] = {
            "rows": len(rows),
            "posed_as": POSED_AS[kind],
            "sha256": digest,
            "training_overlap": overlap,
        }
        print(kind, json.dumps(summary[kind]))
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
