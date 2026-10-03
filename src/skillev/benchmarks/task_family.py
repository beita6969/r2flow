from __future__ import annotations


def require_benchmark_task_family(*, benchmark_id: str, task_family: str) -> str:
    if type(benchmark_id) is not str or not benchmark_id.strip() or "\x00" in benchmark_id:
        raise ValueError("benchmark_id must be non-empty text without NUL")
    if type(task_family) is not str or not task_family.strip() or "\x00" in task_family:
        raise ValueError("task_family must be non-empty text without NUL")
    prefix = f"{benchmark_id}/"
    if not task_family.startswith(prefix) or task_family == prefix:
        raise ValueError("task_family must begin with its benchmark namespace")
    return task_family


__all__ = ["require_benchmark_task_family"]
