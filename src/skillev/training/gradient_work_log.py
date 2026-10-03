from copy import deepcopy


class GradientWorkLog:
    def __init__(self) -> None:
        self.edges: dict[tuple[int, str, int], dict[str, object]] = {}

    def observe(self, position: int, value: dict[str, object], now: float) -> None:
        direction, index = value.get("direction"), value.get("step_index")
        if not isinstance(direction, str) or type(index) is not int:
            return
        key = (position, direction, index)
        row = self.edges.setdefault(
            key,
            {
                "position": position,
                "direction": direction,
                "step_index": index,
                "host_started": now,
                "host_finished": None,
                "cuda_seconds": None,
            },
        )
        for field in ("prefix_tokens", "action_tokens", "checkpointed", "offloaded"):
            if field in value:
                row[field] = value[field]
        if value.get("stage") == "edge-backward-returned":
            row["host_finished"] = now

    def snapshot(self) -> list[dict[str, object]]:
        return [deepcopy(self.edges[key]) for key in sorted(self.edges)]
