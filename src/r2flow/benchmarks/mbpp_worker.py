from __future__ import annotations

import argparse
import ast
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Any


def _restore(value: object) -> object:
    if isinstance(value, dict):
        if set(value) == {"format", "value"} and value.get("format") == (
            "r2flow-nonfinite-float@1"
        ):
            values = {
                "nan": float("nan"),
                "+inf": float("inf"),
                "-inf": float("-inf"),
            }
            return values[str(value["value"])]
        return {key: _restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore(item) for item in value]
    return value


def _emit(value: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(value, separators=(",", ":"), sort_keys=True))
    sys.stdout.flush()


def _configure_result_transport(native_eval: Any, mode: str) -> None:
    if mode == "synchronized":
        return
    if mode != "single-writer-raw@1":
        raise ValueError("unsupported EvalPlus result transport")
    native_eval.Value = multiprocessing.RawValue
    native_eval.Array = multiprocessing.RawArray


def main() -> None:
    stage = "imports"
    try:
        from evalplus import eval as native_eval
        from evalplus.data.mbpp import mbpp_deserialize_inputs
        from evalplus.eval._special_oracle import (
            MBPP_OUTPUT_NOT_NONE_TASKS,
        )
        from evalplus.gen.util import trusted_exec

        stage = "request-validation"
        request = json.loads(sys.stdin.buffer.read())
        if set(request) != {
            "format",
            "scorer_profile",
            "operation",
            "private_target",
            "prompt",
            "source_task_id",
            "submission",
            "task_id",
        }:
            raise ValueError("request fields differ")
        if (
            request["format"] != "skillev-mbpp-request@2"
            or request["operation"] != "evaluate-mbpp-plus"
        ):
            raise ValueError("operation differs")
        target = _restore(request["private_target"])
        prompt = request["prompt"]
        source_task_id = request["source_task_id"]
        submission = request["submission"]
        if (
            not isinstance(target, dict)
            or not isinstance(prompt, str)
            or not isinstance(source_task_id, str)
            or not isinstance(submission, str)
        ):
            raise TypeError("request types differ")
        required = {
            "assertion",
            "atol",
            "base_input",
            "canonical_solution",
            "contract",
            "entry_point",
            "plus_input",
        }
        if set(target) != required:
            raise ValueError("private target fields differ")
        entry_point = target["entry_point"]
        canonical = target["canonical_solution"]
        base_inputs = mbpp_deserialize_inputs(source_task_id, target["base_input"])
        plus_inputs = mbpp_deserialize_inputs(source_task_id, target["plus_input"])
        atol = target["atol"]
        if not isinstance(entry_point, str) or not isinstance(canonical, str):
            raise TypeError("target source fields differ")
        output_not_none = entry_point in MBPP_OUTPUT_NOT_NONE_TASKS
        profile = request["scorer_profile"]
        if not isinstance(profile, dict) or profile.get("source_revision") != "26d6d00":
            raise ValueError("the shared fixed EvalPlus profile is required")
        if (
            profile.get("lane_execution") != "independent-base-and-plus"
            or profile.get("fast_check") is not False
        ):
            raise ValueError("independent complete native lanes are required")
        minimum = float(profile["min_time_limit"])
        factor = float(profile["gt_time_limit_factor"])
        os.environ["EVALPLUS_MAX_MEMORY_BYTES"] = str(profile["maximum_memory_bytes"])
        os.environ.pop("EVALPLUS_TIMEOUT_PER_TASK", None)
        _configure_result_transport(native_eval, profile["result_transport"])
        success = native_eval.PASS
        lanes: dict[str, object] = {}
        try:
            ast.parse(submission)
            syntax: dict[str, object] = {"status": "valid"}
        except SyntaxError as error:
            syntax = {"status": "invalid", "error_type": type(error).__name__, "line": error.lineno}

        def evaluate(inputs: Any, *, lane: str) -> bool:
            nonlocal stage
            started = time.monotonic()
            stage = f"{lane}-reference"
            expected, reference_times = trusted_exec(
                prompt + canonical,
                inputs,
                entry_point,
                record_time=True,
                output_not_none=output_not_none,
            )
            stage = f"{lane}-candidate"
            reference_seconds = time.monotonic() - started
            candidate_started = time.monotonic()
            verdict = native_eval.untrusted_check(
                "mbpp",
                submission,
                inputs,
                entry_point,
                expected=expected,
                atol=atol,
                ref_time=reference_times,
                fast_check=False,
                min_time_limit=minimum,
                gt_time_limit_factor=factor,
            )
            lanes[lane] = {
                "native_status": verdict[0],
                "details": [bool(item) for item in verdict[1]],
                "completed_inputs": len(verdict[1]),
                "planned_inputs": len(inputs),
                "reference_seconds": reference_seconds,
                "candidate_seconds": time.monotonic() - candidate_started,
            }
            return bool(verdict[0] == success)

        base_passed = evaluate(base_inputs, lane="base")
        plus_passed = evaluate(plus_inputs, lane="plus")
        _emit(
            {
                "format": "skillev-mbpp-verdict@2",
                "base_passed": base_passed,
                "plus_passed": plus_passed,
                "status": "passed" if base_passed and plus_passed else "failed",
                "lanes": lanes,
                "syntax": syntax,
                "scorer_profile": profile,
            }
        )
    except BaseException as error:
        missing_module = getattr(error, "name", None)
        _emit(
            {
                "error_module": missing_module if isinstance(missing_module, str) else None,
                "error_stage": stage,
                "error_type": type(error).__name__,
                "infrastructure_error": True,
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-request official EvalPlus MBPP+ verifier.")
    parser.add_argument("--official-source-root", type=Path)
    arguments = parser.parse_args()
    if arguments.official_source_root is not None:
        root = arguments.official_source_root
        if not root.is_absolute() or not (root / "evalplus" / "config.py").is_file():
            raise ValueError("EvalPlus source root must be an explicit deployed source directory")
        sys.path.insert(0, str(root))
    main()
