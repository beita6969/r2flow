from collections.abc import Mapping


def native_benchmark(native: Mapping[str, object]) -> str:
    domain = native.get("benchmark_id")
    if isinstance(domain, str) and domain:
        return domain
    source = native.get("training_evidence_source")
    domain = source.get("benchmark_id") if isinstance(source, dict) else None
    return domain if isinstance(domain, str) and domain else "unknown"
