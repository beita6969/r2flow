from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from r2flow.benchmarks.training_records import TrainingRecord
from r2flow.source_keys import canonical_source_key
from skillev.training.r2flow_evolution_config import DEDICATED_VALIDATION_POOL

from .quality_panel_split import SourceIdentity, source_identity

VALIDATION_POOL_FORMAT: Final = "r2flow-validation-pool@1"
VALIDATION_POOL_ALGORITHM: Final = "training-allowlist-canonical-source-seed0-per-domain-sample@1"
VALIDATION_POOL_FILE: Final = "validation-pool.json"
VALIDATION_POOL_EXCLUSION_GROUP: Final = "validation_pool"


def _domain_seed(seed: int, domain: str) -> int:
    digest = hashlib.sha256(f"{VALIDATION_POOL_ALGORITHM}\0{seed}\0{domain}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class ValidationPool:
    records: tuple[TrainingRecord, ...]
    sources: frozenset[SourceIdentity]
    domains: tuple[str, ...]
    per_domain: int
    seed: int
    summary: dict[str, object]


def draw_validation_pool(
    training: Sequence[TrainingRecord],
    *,
    domains: Sequence[str],
    per_domain: int,
    excluded: frozenset[SourceIdentity],
    aliases: Mapping[str, Mapping[str, str]] | None = None,
    seed: int = 0,
) -> ValidationPool:
    if type(per_domain) is not int or per_domain < 1:
        raise ValueError("the validation pool draws at least one source per domain")
    if type(seed) is not int or seed < 0:
        raise ValueError("the validation pool seed is a non-negative integer")
    if len(set(domains)) != len(domains) or not domains:
        raise ValueError("the validation pool needs the run's distinct domains")
    grouped: dict[SourceIdentity, TrainingRecord] = {}
    for record in training:
        grouped.setdefault(canonical_source_key(source_identity(record), aliases), record)
    if set(grouped) & excluded:
        raise ValueError(
            "the training allowlist contains an excluded (V_q held-out / declared) source"
        )
    records: list[TrainingRecord] = []
    chosen_sources: set[SourceIdentity] = set()
    per_domain_summary: dict[str, object] = {}
    for domain in domains:
        candidates = sorted(s for s in grouped if s[0] == domain)
        if len(candidates) <= per_domain:
            raise ValueError(
                f"{domain}: {len(candidates)} training sources are too few for a validation "
                f"pool of {per_domain} that leaves training sources"
            )
        chosen = random.Random(_domain_seed(seed, domain)).sample(candidates, per_domain)
        chosen_sources.update(chosen)
        records.extend(grouped[s] for s in chosen)
        per_domain_summary[domain] = {
            "pool": [s[1] for s in chosen],
            "training_sources_before": len(candidates),
            "training_sources_after": len(candidates) - per_domain,
        }
    summary: dict[str, object] = {
        "format": VALIDATION_POOL_FORMAT,
        "selection_algorithm": VALIDATION_POOL_ALGORITHM,
        "validation_query_selection": DEDICATED_VALIDATION_POOL,
        "seed": seed,
        "domains": list(domains),
        "per_domain": per_domain,
        "pool_queries": len(records),
        "per_domain_sources": per_domain_summary,
    }
    return ValidationPool(
        tuple(records), frozenset(chosen_sources), tuple(domains), per_domain, seed, summary
    )


def write_validation_pool(
    pool: ValidationPool, root: Path, *, vq_heldout_sha256: str | None = None
) -> Path:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = root / VALIDATION_POOL_FILE
    value: dict[str, Any] = {
        "format": VALIDATION_POOL_FORMAT,
        "selection_algorithm": VALIDATION_POOL_ALGORITHM,
        "validation_query_selection": DEDICATED_VALIDATION_POOL,
        "seed": pool.seed,
        "domains": list(pool.domains),
        "per_domain": pool.per_domain,
        "pool_sources": sorted([list(s) for s in pool.sources]),
        "vq_heldout_sha256": vq_heldout_sha256,
        "heldout_records": [r.to_value() for r in pool.records],
        "summary": pool.summary,
    }
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
    return path


def load_validation_pool_records(
    path: Path, domains: tuple[str, ...] | None = None
) -> tuple[TrainingRecord, ...]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format") != VALIDATION_POOL_FORMAT:
        raise ValueError("unsupported validation pool file (expected r2flow-validation-pool@1)")
    records = tuple(TrainingRecord.from_value(r) for r in value["heldout_records"])
    if domains is None:
        return records
    return tuple(r for r in records if r.episode.benchmark.value in domains)


def require_pool_outside(
    path: Path,
    *,
    training: Sequence[TrainingRecord],
    vq_heldout: Path | None,
    data_condition: Mapping[str, Any] | None,
) -> tuple[TrainingRecord, ...]:
    from .vq_heldout import load_vq_heldout_records

    records = load_validation_pool_records(path)
    declaration = (data_condition or {}).get("autonomous_ttb_sources") or {}
    aliases = declaration.get("source_aliases", {}) if isinstance(declaration, dict) else {}

    def keys(rows: Sequence[TrainingRecord]) -> set[SourceIdentity]:
        return {canonical_source_key(source_identity(r), aliases) for r in rows}

    pool = keys(records)
    if not records or len(pool) != len(records):
        raise ValueError("the validation pool must hold distinct canonical sources")
    if pool & keys(training):
        raise ValueError("validation pool sources must be excluded from training")
    if vq_heldout is not None and pool & keys(load_vq_heldout_records(vq_heldout)):
        raise ValueError("the validation pool overlaps the V_q held-out set")
    return records


def declares_validation_pool(config: Any) -> bool:
    return getattr(config, "r2flow", None) is not None


def require_validation_pool_argument(config: Any, validation_pool: Path | None) -> None:
    if declares_validation_pool(config) != (validation_pool is not None):
        raise ValueError(
            "--validation-pool is required by (only) a config declaring "
            f"{DEDICATED_VALIDATION_POOL}"
        )


VALIDATION_POOL_BINDING_FILE: Final = "validation-pool-binding.json"


def bind_validation_pool(root: Path, path: Path) -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": "r2flow-validation-pool-binding@1",
        "validation_query_selection": DEDICATED_VALIDATION_POOL,
        "pool_format": VALIDATION_POOL_FORMAT,
        "sha256": sha256_file(path),
    }
    binding = root / VALIDATION_POOL_BINDING_FILE
    if binding.is_file():
        if json.loads(binding.read_text(encoding="utf-8")) != value:
            raise ValueError("--validation-pool differs from the pool this run started with")
        return value
    with os.fdopen(os.open(binding, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write("\n")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


__all__ = [
    "VALIDATION_POOL_ALGORITHM",
    "VALIDATION_POOL_BINDING_FILE",
    "VALIDATION_POOL_EXCLUSION_GROUP",
    "VALIDATION_POOL_FILE",
    "VALIDATION_POOL_FORMAT",
    "ValidationPool",
    "bind_validation_pool",
    "declares_validation_pool",
    "draw_validation_pool",
    "load_validation_pool_records",
    "require_pool_outside",
    "require_validation_pool_argument",
    "sha256_file",
    "write_validation_pool",
]
