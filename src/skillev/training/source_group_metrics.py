from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Sequence

SOURCE_GROUP_NOTE = (
    "Equal weight per (benchmark, canonical source_id), averaging repeated rollouts "
    "within source; population aliases are one group. SD is sample dispersion of "
    "source means; SE=SD/sqrt(source groups) assumes independent sources, not "
    "independent actions/rollouts. Small fixed panels provide descriptive evidence "
    "only, not a confidence interval or population guarantee. SD/SE are unknown "
    "with fewer than two groups or any missing member; heterogeneous-domain "
    "panel SE is not estimated without a declared stratified sampling design."
)
TERMINAL_NOTE = (
    "A valid terminal record has at least one committed action with explicit "
    "accepted_submission OR environment_terminal assessment, independent of task "
    "success/reward. Count once per trajectory. All committed actions must have "
    "matched assessment evidence; otherwise the record is unknown. Fully assessed "
    "horizon exhaustion without either flag is not a valid terminal record."
)


def source_group_summary(
    observations: Sequence[tuple[tuple[str, str], dict[str, float | None]]],
    *,
    allow_standard_error: bool,
) -> dict[str, float | None]:
    groups: dict[tuple[str, str], list[dict[str, float | None]]] = defaultdict(list)
    for source, values in observations:
        groups[source].append(values)
    output: dict[str, float | None] = {"source_group_count": float(len(groups))}
    names = sorted({name for _, values in observations for name in values})
    for name in names:
        means = []
        for rows in groups.values():
            member_values = [row.get(name) for row in rows]
            if all(value is not None for value in member_values):
                means.append(
                    math.fsum(v for v in member_values if v is not None) / len(member_values)
                )
        complete = len(means) == len(groups) and bool(groups)
        deviation = statistics.stdev(means) if complete and len(means) >= 2 else None
        output[name + "/source_group_observed_count"] = float(len(means))
        output[name + "/source_group_mean"] = statistics.mean(means) if complete else None
        output[name + "/source_group_sample_sd"] = deviation
        output[name + "/source_group_standard_error"] = (
            deviation / math.sqrt(len(means))
            if deviation is not None and allow_standard_error
            else None
        )
    return output
