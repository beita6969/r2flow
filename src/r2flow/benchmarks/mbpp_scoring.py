from __future__ import annotations

import ast
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MBPP_REQUEST_FORMAT = "skillev-mbpp-request@2"
MBPP_VERDICT_FORMAT = "skillev-mbpp-verdict@2"


@dataclass(frozen=True, slots=True)
class MBPPScorerProfile:
    profile_id: str = "evalplus-26d6d00-native-time-defaults-raw-ipc@2"
    source_revision: str = "26d6d00"
    condition: str = "native-time-defaults"
    min_time_limit: float = 4.0
    gt_time_limit_factor: float = 4.0
    maximum_memory_bytes: int = 4 * 1024**3
    per_lane_timeout_seconds: int = 60
    outer_timeout_seconds: float = 900.0
    maximum_concurrency: int = 8
    result_transport: str = "single-writer-raw@1"

    def __post_init__(self) -> None:
        if not self.profile_id or not self.source_revision:
            raise ValueError("EvalPlus source and profile identities are required")
        if self.condition not in {"native-time-defaults", "custom-time-limits"}:
            raise ValueError("EvalPlus timing condition must be explicit")
        if self.source_revision != "26d6d00":
            raise ValueError("this scoring contract uses the shared fixed EvalPlus source")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (
                self.min_time_limit,
                self.gt_time_limit_factor,
                self.maximum_memory_bytes,
                self.outer_timeout_seconds,
            )
        ):
            raise ValueError("EvalPlus time and memory allowances must be finite and positive")
        if self.condition == "native-time-defaults" and (
            self.min_time_limit != 4.0 or self.gt_time_limit_factor != 4.0
        ):
            raise ValueError("non-native timing requires an explicitly custom condition")
        if self.maximum_concurrency < 1:
            raise ValueError("EvalPlus concurrency must be positive")
        if self.result_transport not in {"synchronized", "single-writer-raw@1"}:
            raise ValueError("unsupported EvalPlus result transport")
        if self.per_lane_timeout_seconds != 60:
            raise ValueError("the fixed EvalPlus source supports its native 60-second lane cap")
        if self.outer_timeout_seconds < 2 * (self.per_lane_timeout_seconds + 3) + 30:
            raise ValueError("outer timeout cannot cut off two native lanes and reference startup")

    def to_value(self) -> dict[str, Any]:
        return {**asdict(self), "lane_execution": "independent-base-and-plus", "fast_check": False}


def resolve_mbpp_profile(settings: dict[str, Any]) -> MBPPScorerProfile:
    raw = dict(settings.get("profile", {}))
    if raw:
        raw.setdefault("result_transport", "synchronized")
    if raw.pop("lane_execution", "independent-base-and-plus") != "independent-base-and-plus":
        raise ValueError("both native scoring lanes must run independently")
    if raw.pop("fast_check", False) is not False:
        raise ValueError("this scoring condition requires all native checks")
    profile = MBPPScorerProfile(**raw)
    if (
        settings.get("timeout_seconds", profile.outer_timeout_seconds)
        != profile.outer_timeout_seconds
    ):
        raise ValueError("outer timeout differs from the single frozen scorer profile")
    source = Path(settings["source_root"]) / "evalplus" / "config.py"
    defaults = {
        item.targets[0].id: ast.literal_eval(item.value)
        for item in ast.parse(source.read_text()).body
        if isinstance(item, ast.Assign)
        and isinstance(item.targets[0], ast.Name)
        and item.targets[0].id in {"DEFAULT_MIN_TIME_LIMIT", "DEFAULT_GT_TIME_LIMIT_FACTOR"}
    }
    if profile.condition == "native-time-defaults" and (
        profile.min_time_limit != defaults["DEFAULT_MIN_TIME_LIMIT"]
        or profile.gt_time_limit_factor != defaults["DEFAULT_GT_TIME_LIMIT_FACTOR"]
    ):
        raise ValueError("native timing profile differs from the selected source defaults")
    return profile


class MBPPScoringInfrastructureError(RuntimeError):
    pass


def mbpp_request(
    *,
    profile: MBPPScorerProfile,
    private_target: dict[str, Any],
    prompt: str,
    source_task_id: str,
    submission: str,
    task_id: str,
) -> dict[str, Any]:
    return {
        "format": MBPP_REQUEST_FORMAT,
        "operation": "evaluate-mbpp-plus",
        "scorer_profile": profile.to_value(),
        "private_target": private_target,
        "prompt": prompt,
        "source_task_id": source_task_id,
        "submission": submission,
        "task_id": task_id,
    }


def decode_mbpp_verdict(value: object, profile: MBPPScorerProfile) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MBPPScoringInfrastructureError("EvalPlus response is not an object")
    if value.get("infrastructure_error"):
        raise MBPPScoringInfrastructureError(
            "EvalPlus infrastructure failure at "
            + str(value.get("error_stage"))
            + ": "
            + str(value.get("error_type"))
        )
    if (
        value.get("format") != MBPP_VERDICT_FORMAT
        or value.get("scorer_profile") != profile.to_value()
    ):
        raise MBPPScoringInfrastructureError("EvalPlus verdict differs from the frozen condition")
    if any(type(value.get(key)) is not bool for key in ("base_passed", "plus_passed")):
        raise MBPPScoringInfrastructureError("EvalPlus requires native boolean verdicts")
    expected = "passed" if value["base_passed"] and value["plus_passed"] else "failed"
    if value.get("status") != expected:
        raise MBPPScoringInfrastructureError("EvalPlus overall status disagrees with its lanes")
    lanes, syntax = value.get("lanes"), value.get("syntax")
    if not isinstance(lanes, dict) or not isinstance(syntax, dict):
        raise MBPPScoringInfrastructureError("EvalPlus native diagnostics are absent")
    for key in ("base", "plus"):
        lane = lanes.get(key)
        if not isinstance(lane, dict) or lane.get("native_status") not in {
            "pass",
            "fail",
            "timeout",
        }:
            raise MBPPScoringInfrastructureError("EvalPlus native lane is absent")
        if (lane["native_status"] == "pass") != value[key + "_passed"]:
            raise MBPPScoringInfrastructureError("EvalPlus native lane disagrees with boolean")
    return value


def mbpp_failure_kind(result: dict[str, Any]) -> str | None:
    if result.get("base_passed") is True and result.get("plus_passed") is True:
        return None
    if result.get("syntax", {}).get("status") == "invalid":
        return "syntax-error"
    lanes = result.get("lanes", {})
    for name in ("base", "plus"):
        lane = lanes.get(name, {})
        if lane.get("native_status") == "timeout":
            return f"{name}-timeout"
        if lane.get("native_status") == "fail" and lane.get("completed_inputs") == 0:
            return f"{name}-execution-failure"
    return "base-test-failure" if not result["base_passed"] else "plus-test-failure"
