from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from skillev.contracts import JsonValue, normalize_json
from skillev.evolution.vq_trigger import (
    ENTROPY_AT_MOST_ONE_SKILL,
    ENTROPY_H_UNDEFINED,
    ENTROPY_ONE_SKILL_PER_FAMILY,
    QueryResiduals,
    VqPoint,
    build_vq_point,
    evaluate_vq_trigger,
)

from .inflight import durable_json
from .r2flow_config import VQ_PLATEAU_TWO_CONSECUTIVE, VqTriggerConfig
from .r2flow_evolution_config import (
    DEGENERATE_ENTROPY_PHASE_CAP,
    NO_CALL_PHASE_CAP,
    PHASE_STEP_CAP,
)
from .vq_scoring import FlowTrajectoryScore

VQ_TASK_PREFIX: Final = "r2flow-vq@1"
VQ_BOUNDARY_FORMAT: Final = "r2flow-phase-boundary@1"
VQ_EVOLVE: Final = "evolve@1"
VQ_PLATEAU_REASON: Final = "vq-plateau-trigger@1"
VQ_IMPROVEMENT_TRACE: Final = "improvement-trace.jsonl"
VQ_ENTROPY_STRUCTURE_FORMAT: Final = "r2flow-vq-entropy-structure@1"
DEGENERATE_AT_MOST_ONE_SKILL: Final = ENTROPY_AT_MOST_ONE_SKILL
DEGENERATE_H_UNDEFINED: Final = ENTROPY_H_UNDEFINED
DEGENERATE_ONE_SKILL_PER_FAMILY: Final = ENTROPY_ONE_SKILL_PER_FAMILY

BoundaryCallback = Callable[..., Awaitable[None]]


def vq_query_key(task_id: str) -> str:
    prefix = VQ_TASK_PREFIX + "/"
    if not task_id.startswith(prefix):
        raise ValueError("not a V_q replica task id")
    benchmark, _, rest = task_id[len(prefix) :].partition("/")
    source, _, replica = rest.rpartition("/")
    if not benchmark or not source or not replica.startswith("r") or not replica[1:].isdigit():
        raise ValueError("malformed V_q replica task id")
    return f"{benchmark}:{source}"


def _verifier_z(artifact: Any, step: Any) -> str:
    records = getattr(artifact, "verifier_records", None)
    if not records or len(records) < step.index:
        raise ValueError("the verifier-z entropy needs a verifier record for every step")
    record = records[step.index - 1]
    if record.step_index != step.index:
        raise ValueError("verifier records are not ordered by step")
    return str(record.z.cell_key())


def skill_call_cells(artifacts: Sequence[Any]) -> dict[tuple[str, str], dict[str, int]]:
    cells: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for artifact in artifacts:
        query = vq_query_key(artifact.manifest.task_id)
        for step in artifact.record.steps:
            if not step.invoked_skill_ids:
                continue
            z = _verifier_z(artifact, step)
            for skill in step.invoked_skill_ids:
                cells[(query, z)][skill] += 1
    return {key: dict(calls) for key, calls in cells.items()}


def called_families(cells: Mapping[tuple[str, str], Mapping[str, int]]) -> list[str | None]:
    from skillev.evolution.task_features import transferable_family

    families: set[str | None] = set()
    for (query, _), calls in cells.items():
        if sum(n for n in calls.values() if n > 0) > 0:
            families.add(transferable_family(query.partition(":")[0]))
    return sorted(families, key=lambda family: (family is None, family or ""))


def one_visible_skill_per_called_family(structure: object) -> bool:
    families = structure.get("called_families") if isinstance(structure, dict) else None
    visible = structure.get("visible_skills_by_family") if isinstance(structure, dict) else None
    return (
        isinstance(families, list)
        and isinstance(visible, dict)
        and bool(families)
        and all(family is not None and visible.get(family) == 1 for family in families)
    )


def build_point_from_scores(
    *,
    optimizer_step: int,
    library_version: str,
    artifacts: Sequence[Any],
    scores: Sequence[FlowTrajectoryScore],
    expected_queries: frozenset[str],
    active_skills: int,
    v_min: float,
) -> VqPoint:
    deltas: dict[str, list[float]] = {query: [] for query in expected_queries}
    for score in scores:
        query = vq_query_key(score.task_id)
        if query not in deltas:
            raise ValueError("a scored rollout is outside the declared held-out set")
        deltas[query].append(score.delta_0T)
    return build_vq_point(
        optimizer_step=optimizer_step,
        library_version=library_version,
        queries=[QueryResiduals(q, tuple(v)) for q, v in deltas.items()],
        cell_calls=skill_call_cells(artifacts),
        active_skills=active_skills,
        v_min=v_min,
    )


def vq_point_from_value(value: dict[str, Any]) -> VqPoint:
    return VqPoint(
        optimizer_step=int(value["optimizer_step"]),
        library_version=str(value["library_version"]),
        status=value["status"],
        queries=tuple(
            QueryResiduals(q["query_key"], tuple(float(d) for d in q["deltas"]))
            for q in value["queries"]
        ),
        log_v=value["log_v"],
        se_log_v=value["se_log_v"],
        tau2=value["tau2"],
        v=value["v"],
        h_norm=value["h_norm"],
        calls_total=int(value["calls_total"]),
        active_skills=int(value["active_skills"]),
        format=value["format"],
    )


def mirror_directory_additive(source: Path, destination: Path) -> None:
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        if path.suffix in (".pending", ".tmp"):
            continue
        target = destination / path.relative_to(source)
        if target.is_file() and target.read_bytes() == path.read_bytes():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".mirror-tmp")
        shutil.copyfile(path, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)


def _value(data: object) -> dict[str, JsonValue]:
    value = normalize_json(data)
    assert isinstance(value, dict)
    return value


class VqPlateauMonitor:
    def __init__(
        self,
        root: Path,
        config: VqTriggerConfig,
        *,
        expected_queries: frozenset[str],
        collect: Callable[[Path, int, str], Awaitable[Sequence[Any]]],
        score: Callable[[Sequence[Any]], Awaitable[Sequence[FlowTrajectoryScore]]],
        active_skills: Callable[[], int],
        library_version: Callable[[], str],
        on_boundary: BoundaryCallback,
        visible_skills_by_family: Callable[[], Mapping[str, int]],
        mirror: Callable[[Path], None] | None = None,
        no_call_cap_steps: int | None = None,
        phase_cap_steps: int | None = None,
        phase_start: Callable[[], int] | None = None,
        phase_cap_from_step: int = 1,
        degenerate_entropy_cap: str | None = None,
    ) -> None:
        if no_call_cap_steps is not None and (
            type(no_call_cap_steps) is not int or no_call_cap_steps < 1
        ):
            raise ValueError("the no-call phase cap must be a positive number of steps")
        if (phase_cap_steps is None and degenerate_entropy_cap is None) != (phase_start is None):
            raise ValueError("phase-step-cap@1 needs both the cap and the phase start")
        if degenerate_entropy_cap is not None and (
            degenerate_entropy_cap != DEGENERATE_ENTROPY_PHASE_CAP
            or no_call_cap_steps is None
            or phase_start is None
        ):
            raise ValueError(
                f"{DEGENERATE_ENTROPY_PHASE_CAP} extends no-call-phase-cap@1 and needs the "
                "phase start"
            )
        if phase_cap_steps is not None and (
            type(phase_cap_steps) is not int or phase_cap_steps < 1
        ):
            raise ValueError("the per-phase step cap must be a positive number of steps")
        if type(phase_cap_from_step) is not int or phase_cap_from_step < 0:
            raise ValueError("the per-phase step cap starts at a nonnegative optimizer step")
        self.on_boundary = on_boundary
        self.no_call_cap_steps = no_call_cap_steps
        self.phase_cap_steps = phase_cap_steps
        self.phase_start = phase_start
        self.phase_cap_from_step = phase_cap_from_step
        self.degenerate_entropy_cap = degenerate_entropy_cap
        self.visible_skills_by_family = visible_skills_by_family
        if not expected_queries:
            raise ValueError("the V_q monitor needs the declared held-out queries")
        self.root, self.config = root, config
        self.expected_queries = expected_queries
        self.collect, self.score = collect, score
        self.active_skills, self.library_version = active_skills, library_version
        self.mirror = mirror
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        saved = root / "trigger.json"
        declared = _value({"trigger": config.to_value(), "queries": sorted(expected_queries)})
        if saved.exists():
            if json.loads(saved.read_text(encoding="utf-8")) != declared:
                raise ValueError("V_q trigger or held-out set changed during the run")
        else:
            durable_json(saved, declared)

    def _path(self, kind: str, step: int) -> Path:
        return self.root / f"{kind}-{step:08d}.json"

    def _entropy_structure(self, step: int) -> object:
        path = self._path("entropy", step)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def entropy_degenerate_steps(self, series: Sequence[VqPoint]) -> frozenset[int]:
        return frozenset(
            point.optimizer_step
            for point in series
            if point.status == "complete"
            and one_visible_skill_per_called_family(self._entropy_structure(point.optimizer_step))
        )

    def series(self) -> tuple[VqPoint, ...]:
        return tuple(
            vq_point_from_value(json.loads(p.read_text(encoding="utf-8")))
            for p in sorted(self.root.glob("point-*.json"))
        )

    async def _measure(self, step: int, snapshot_id: str) -> VqPoint:
        attempt = self._path("attempt", step)
        library = self.library_version()
        gap = VqPoint(step, library, "gap", (), None, None, None, None, None, 0, 0)
        if attempt.exists():
            durable_json(attempt, {**json.loads(attempt.read_text()), "status": "interrupted"})
            return gap
        durable_json(
            attempt,
            {"status": "collecting", "policy_step": step, "policy_snapshot_id": snapshot_id},
        )
        directory = self.root / "collections" / f"point-{step:08d}"
        try:
            artifacts = tuple(await self.collect(directory, step, snapshot_id))
            scores = tuple(await self.score(artifacts))
            point = build_point_from_scores(
                optimizer_step=step,
                library_version=library,
                artifacts=artifacts,
                scores=scores,
                expected_queries=self.expected_queries,
                active_skills=self.active_skills(),
                v_min=self.config.v_min,
            )
            cells = skill_call_cells(artifacts)
            durable_json(
                self._path("entropy", step),
                _value(
                    {
                        "format": VQ_ENTROPY_STRUCTURE_FORMAT,
                        "optimizer_step": step,
                        "library_version": library,
                        "called_cells": len(cells),
                        "called_families": called_families(cells),
                        "visible_skills_by_family": dict(
                            sorted(self.visible_skills_by_family().items())
                        ),
                    }
                ),
            )
        except (OSError, TimeoutError, RuntimeError) as error:
            durable_json(
                attempt,
                {
                    "status": "infrastructure-failure",
                    "error_type": type(error).__name__,
                    "policy_step": step,
                    "policy_snapshot_id": snapshot_id,
                },
            )
            return gap
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "flow.jsonl").open("w", encoding="utf-8") as stream:
            for score in scores:
                stream.write(json.dumps(score.to_value(), sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        durable_json(attempt, {"status": "complete", "policy_step": step, "rollouts": len(scores)})
        return point

    async def check(self, *, policy_step: int, policy_snapshot_id: str) -> None:
        cfg = self.config
        if policy_step < cfg.baseline_step or policy_step % cfg.cadence_steps:
            return
        point_path = self._path("point", policy_step)
        if not point_path.exists():
            point = await self._measure(policy_step, policy_snapshot_id)
            durable_json(point_path, point.to_value())
        point = vq_point_from_value(json.loads(point_path.read_text(encoding="utf-8")))
        series = self.series()
        decision = evaluate_vq_trigger(
            series, cfg, entropy_degenerate_steps=self.entropy_degenerate_steps(series)
        )
        diagnostics: dict[str, Any] = {
            "point_status": point.status,
            "v_q": point.v,
            "log_v": point.log_v,
            "se_log_v": point.se_log_v,
            "tau2": point.tau2,
            "h_norm": point.h_norm,
            "calls_total": point.calls_total,
            "active_skills": point.active_skills,
            "rho_w": decision.rho,
            "slope": decision.slope,
            "slope_half_width": decision.half_width,
            "epsilon_b": decision.epsilon_b,
            "delta_h": decision.delta_h,
            "entropy_vacuous": decision.entropy_vacuous,
        }
        durable_json(
            self._path("decision", policy_step),
            _value(
                {
                    "decision": decision.to_value(),
                    "policy_step": policy_step,
                    "policy_snapshot_id": policy_snapshot_id,
                    "diagnostics": diagnostics,
                }
            ),
        )
        await self._evolve(policy_step, policy_snapshot_id, point, decision)
        if self.mirror is not None:
            self.mirror(self.root)

    def no_call_cap_reached(self, segment: str) -> bool:
        if self.no_call_cap_steps is None:
            return False
        points = [p for p in self.series() if p.library_version == segment]
        complete = [p for p in points if p.status == "complete"]
        if not complete or any(p.calls_total > 0 for p in points):
            return False
        steps = [p.optimizer_step for p in complete]
        return max(steps) - min(steps) >= self.no_call_cap_steps

    def degenerate_entropy_state(
        self, policy_step: int, segment: str, point: VqPoint
    ) -> dict[str, JsonValue] | None:
        if self.degenerate_entropy_cap is None or self.phase_start is None:
            return None
        if segment != self.library_version():
            path = self._path("boundary", policy_step)
            saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            recorded = saved.get("degenerate_entropy")
            if saved.get("library_version") != segment or not isinstance(recorded, dict):
                return None
            return _value(recorded)
        start = self.phase_start()
        if type(start) is not int or not 0 <= start <= policy_step:
            raise ValueError("the phase start must be an optimizer step at or before the point")
        structure = self._entropy_structure(policy_step)
        families = structure.get("called_families") if isinstance(structure, dict) else None
        visible = structure.get("visible_skills_by_family") if isinstance(structure, dict) else None
        why: str | None = None
        if point.status == "complete":
            if point.active_skills <= 1:
                why = DEGENERATE_AT_MOST_ONE_SKILL
            elif point.h_norm is None:
                why = DEGENERATE_H_UNDEFINED
            elif one_visible_skill_per_called_family(structure):
                why = DEGENERATE_ONE_SKILL_PER_FAMILY
        phase_steps = policy_step - start
        return _value(
            {
                "rule": DEGENERATE_ENTROPY_PHASE_CAP,
                "max_phase_steps_no_calls": self.no_call_cap_steps,
                "phase_start_step": start,
                "phase_steps": phase_steps,
                "point_status": point.status,
                "visible_active_skills": point.active_skills,
                "h_norm": point.h_norm,
                "called_families": families,
                "visible_skills_by_family": visible,
                "degenerate": why,
                "reached": why is not None
                and self.no_call_cap_steps is not None
                and phase_steps >= self.no_call_cap_steps,
            }
        )

    def phase_cap_state(self, policy_step: int, segment: str) -> dict[str, JsonValue] | None:
        if (
            self.phase_cap_steps is None
            or self.phase_start is None
            or policy_step < self.phase_cap_from_step
        ):
            return None
        if segment != self.library_version():
            path = self._path("boundary", policy_step)
            saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            recorded = saved.get("phase_step_cap")
            if saved.get("library_version") != segment or not isinstance(recorded, dict):
                return None
            return _value(recorded)
        start = self.phase_start()
        if type(start) is not int or not 0 <= start <= policy_step:
            raise ValueError("the phase start must be an optimizer step at or before the point")
        return {
            "rule": PHASE_STEP_CAP,
            "max_phase_steps": self.phase_cap_steps,
            "phase_start_step": start,
            "phase_steps": policy_step - start,
            "effective_from_step": self.phase_cap_from_step,
            "reached": policy_step - start >= self.phase_cap_steps,
        }

    def plateau_confirmation_state(
        self, policy_step: int, segment: str, decision: Any
    ) -> dict[str, JsonValue] | None:
        if self.config.plateau_confirmation is None:
            return None
        assert self.config.plateau_confirmation == VQ_PLATEAU_TWO_CONSECUTIVE

        def fired_at(step: int, value: Mapping[str, Any]) -> bool:
            window = value.get("window_steps")
            return (
                value.get("status") == "boundary"
                and value.get("library_version") == segment
                and isinstance(window, list)
                and bool(window)
                and window[-1] == step
            )

        fired = fired_at(policy_step, decision.to_value())
        previous = policy_step - self.config.cadence_steps
        path = self._path("decision", previous)
        recorded = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        earlier = recorded.get("decision") if isinstance(recorded, dict) else None
        previous_fired = isinstance(earlier, dict) and fired_at(previous, earlier)
        return _value(
            {
                "rule": VQ_PLATEAU_TWO_CONSECUTIVE,
                "fired": fired,
                "previous_step": previous,
                "previous_fired": previous_fired,
                "confirmed": fired and previous_fired,
            }
        )

    async def _evolve(
        self, policy_step: int, policy_snapshot_id: str, point: VqPoint, decision: Any
    ) -> None:
        segment = point.library_version
        cap = self.phase_cap_state(policy_step, segment)
        degenerate = self.degenerate_entropy_state(policy_step, segment, point)
        confirmation = self.plateau_confirmation_state(policy_step, segment, decision)
        reason: str | None = None
        if (
            decision.status == "boundary"
            and decision.library_version == segment
            and (confirmation is None or confirmation["confirmed"] is True)
        ):
            reason = VQ_PLATEAU_REASON
        elif self.no_call_cap_reached(segment):
            reason = NO_CALL_PHASE_CAP
        elif degenerate is not None and degenerate["reached"]:
            reason = DEGENERATE_ENTROPY_PHASE_CAP
        elif cap is not None and cap["reached"]:
            reason = PHASE_STEP_CAP
        decision_path = self._path("decision", policy_step)
        recorded = json.loads(decision_path.read_text(encoding="utf-8"))
        capped = {} if cap is None else {"phase_step_cap": cap}
        if degenerate is not None:
            capped["degenerate_entropy"] = degenerate
        if confirmation is not None:
            capped["plateau_confirmation"] = confirmation
        recorded = {**recorded, **capped}
        durable_json(decision_path, _value({**recorded, "boundary_reason": reason}))
        if reason is None or segment != self.library_version():
            return
        boundary = self._path("boundary", policy_step)
        if not boundary.exists():
            durable_json(
                boundary,
                _value(
                    {
                        "format": VQ_BOUNDARY_FORMAT,
                        "action": VQ_EVOLVE,
                        "reason": reason,
                        **capped,
                        "library_version": segment,
                        "optimizer_step": policy_step,
                        "policy_snapshot_id": policy_snapshot_id,
                        "decision": decision.to_value(),
                    }
                ),
            )
        await self.on_boundary(
            optimizer_step=policy_step,
            policy_snapshot_id=policy_snapshot_id,
            reason=reason,
            diagnostics=_value({**recorded, "boundary_reason": reason}),
        )


__all__ = [
    "VQ_BOUNDARY_FORMAT",
    "VQ_ENTROPY_STRUCTURE_FORMAT",
    "VQ_EVOLVE",
    "VQ_IMPROVEMENT_TRACE",
    "VQ_PLATEAU_REASON",
    "VqPlateauMonitor",
    "build_point_from_scores",
    "called_families",
    "mirror_directory_additive",
    "skill_call_cells",
    "vq_point_from_value",
    "vq_query_key",
]
