from __future__ import annotations

import importlib
import math
from functools import wraps
from typing import Any

_TIMES = ("wait_queue_entry_time", "forward_entry_time", "prefill_finished_time")


def install_request_timing_transport() -> None:
    upstream = importlib.import_module("sglang.srt.observability.req_time_stats")
    stats = upstream.SchedulerReqTimeStats
    original = stats.__getstate__
    if getattr(original, "_skillev_timing_transport", False):
        return

    @wraps(original)
    def state(self: Any) -> dict[str, Any]:
        value = dict(original(self))
        measured = {}
        for name in _TIMES:
            timestamp = getattr(self, name, 0.0)
            if (
                isinstance(timestamp, int | float)
                and not isinstance(timestamp, bool)
                and math.isfinite(timestamp)
                and timestamp > 0
            ):
                measured[name] = timestamp
            else:
                value.pop(name, None)
        if measured:
            value.update(measured)
            value["diff_realtime_monotonic"] = upstream.global_diff_realtime_monotonic
        return value

    state._skillev_timing_transport = True
    stats.__getstate__ = state
