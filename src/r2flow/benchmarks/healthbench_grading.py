from __future__ import annotations

import importlib
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from skillev.evaluation.external_judge_policy import (
    HEALTHBENCH_REFUSAL_RULE,
    healthbench_refusals_exceed_limit,
)
from skillev.evaluation.healthbench_judge_profile import judge_profile
from skillev.evaluation.healthbench_official import HealthBenchExternalJudgeProfile

from .healthbench_grader_usage import IncompleteNativeGradingError, MeteredCompletions


def _load_official(source_root: Path) -> tuple[Any, Any, type[Any]]:
    if not (source_root / "healthbench_eval.py").is_file():
        raise ValueError("pinned simple-evals HealthBench source is absent")
    sys.path.insert(0, str(source_root.parent))
    package = source_root.name
    module = importlib.import_module(f"{package}.healthbench_eval")
    types_module = importlib.import_module(f"{package}.types")
    return module.HealthBenchEval, module.RubricItem, types_module.SamplerResponse


def resolve_healthbench_profile(settings: dict[str, Any]) -> HealthBenchExternalJudgeProfile:
    value = settings.get("effective_profile")
    if value is None:
        return judge_profile()
    profile = HealthBenchExternalJudgeProfile(**value)
    if profile != judge_profile():
        raise ValueError("native HealthBench requires a declared external API profile")
    return profile


def grade_health(
    candidate: str,
    target: dict[str, Any],
    settings: dict[str, Any],
    *,
    diagnostics: dict[str, Any] | None = None,
    record_requests: Callable[[list[dict[str, Any]]], None] | None = None,
    recovery: Any = None,
    replay: Any = None,
) -> tuple[float, int, dict[str, float]]:
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

    from skillev.evaluation.healthbench_official import (
        ExternalHealthBenchRubricSampler,
        ExternalJudgeTransport,
        RubricAttemptRegistry,
        TransportRetryPolicy,
    )

    profile = resolve_healthbench_profile(settings)
    health_type, rubric_type, response_type = _load_official(Path(settings["official_source"]))
    from r2flow.benchmarks.healthbench_api import BoundedAPICompletions, make_client

    gate: Any = None
    classifier: Callable[[BaseException], str] | None = None
    if recovery is not None:
        from skillev.evaluation.healthbench_judge_recovery import CallGate, failure_class

        gate, classifier = CallGate(), failure_class
    client = make_client() if recovery is None else make_client(recovery=recovery)
    delegate = BoundedAPICompletions(
        client.chat.completions,
        capacity_timeout_seconds=None if recovery is None else recovery.capacity_timeout_seconds,
        gate=gate,
    )
    meter = MeteredCompletions(
        delegate,
        retain_evidence=diagnostics is not None,
        record_evidence=record_requests,
        replay=replay,
        record_spool_request_ids=recovery is not None,
        gate=gate,
        failure_classifier=classifier,
    )
    attempts = RubricAttemptRegistry()
    started = time.monotonic()
    grades: list[dict[str, Any]] = []
    rubrics: list[Any] | None = None
    completed = False
    refused: set[int] = set()
    try:
        sampler = ExternalHealthBenchRubricSampler(
            ExternalJudgeTransport(
                client=cast(Any, SimpleNamespace(chat=SimpleNamespace(completions=meter))),
                model=profile.model,
                max_completion_tokens=profile.max_completion_tokens,
                reasoning_effort=profile.reasoning_effort,
                retry_policy=TransportRetryPolicy(
                    maximum_attempts=profile.maximum_attempts,
                    request_timeout_seconds=profile.request_timeout_seconds,
                    total_deadline_seconds=300,
                ),
                transient_error_types=(
                    APIConnectionError,
                    APITimeoutError,
                    InternalServerError,
                    RateLimitError,
                ),
            ),
            response_type,
            attempts,
        )
        evaluator = object.__new__(health_type)
        evaluator.grader_model = sampler
        evaluator.length_adjustment_center = None
        evaluator.length_adjustment_penalty_per_500_chars = None
        rubrics = [rubric_type.from_dict(row) for row in target["rubrics"]]
        from r2flow.benchmarks.healthbench_ledger import official_criterion_messages

        criterion_messages = official_criterion_messages(
            health_type.grade_sample,
            target["prompt"],
            candidate,
            rubrics,
        )
        meter.bind_criterion_messages(criterion_messages)
        _, _, grades = evaluator.grade_sample(
            prompt=target["prompt"],
            response_text=candidate,
            example_tags=[],
            rubric_items=rubrics,
        )
        if len(grades) != len(rubrics) or any(
            type(row.get("criteria_met")) is not bool for row in grades
        ):
            raise RuntimeError("HealthBench rubric grading is incomplete")
        refused = _refused_criteria(grades, criterion_messages, attempts)
        if healthbench_refusals_exceed_limit(len(refused), len(rubrics)):
            raise RuntimeError("HealthBench judge refused more than a third of the criteria")
        kept = [row for index, row in enumerate(grades) if index not in refused]
        positive = sum(float(row["points"]) for row in kept if row["points"] > 0)
        if positive <= 0:
            raise ValueError("HealthBench rubric has no positive denominator")
        earned = sum(float(row["points"]) for row in kept if row["criteria_met"])
        negative = sum(row["points"] < 0 and row["criteria_met"] for row in kept)
        cost = meter.snapshot()
        cost["semantic_repairs"] = float(sampler.semantic_attempts.semantic_repair_count)
        cost["wall_seconds"] = time.monotonic() - started
        completed = True
        return earned / positive, negative, cost
    except Exception as error:
        if gate is not None:
            gate.close_and_drain(recovery.drain_seconds(profile.request_timeout_seconds))
        cost = meter.snapshot()
        cost["semantic_repairs"] = float(attempts.semantic_repair_count)
        cost["wall_seconds"] = time.monotonic() - started
        raise IncompleteNativeGradingError(cost) from error
    finally:
        if diagnostics is not None:
            diagnostics.update(
                status="completed" if completed else "incomplete-grading",
                effective_profile=asdict(profile),
                model_route=profile.model,
                rubric_grades=grades,
                criterion_results=[
                    {"criterion_index": index, **row} for index, row in enumerate(grades)
                ],
                criterion_count=None if rubrics is None else len(rubrics),
                candidate_answer=candidate,
                native_raw_score=None if not completed else earned / positive,
                learning_reward=None if not completed else min(1.0, max(0.0, earned / positive)),
                binary_success=None
                if not completed
                else earned / positive >= 0.60 and negative == 0,
                negative_criterion_count=None if not completed else negative,
                requests=meter.evidence(),
                **({} if gate is None else {"quiescent": gate.idle()}),
            )
            diagnostics["criterion_results"] = [
                {**row, "status": "refused" if row["criterion_index"] in refused else "graded"}
                for row in diagnostics["criterion_results"]
            ]
            diagnostics.update(
                refusal_rule=HEALTHBENCH_REFUSAL_RULE,
                refused_count=len(refused),
                refused_criterion_indices=sorted(refused),
            )
        client.close()


def _refused_criteria(
    grades: list[dict[str, Any]],
    criterion_messages: list[list[dict[str, str]]] | None,
    attempts: Any,
) -> set[int]:
    from skillev.evaluation.healthbench_official import (
        REFUSED_CRITERION_EXPLANATION,
        semantic_messages_key,
    )

    marked = {
        index
        for index, row in enumerate(grades)
        if row.get("explanation") == REFUSED_CRITERION_EXPLANATION
    }
    keys = attempts.refused_message_keys
    if criterion_messages is not None and marked != {
        index
        for index, messages in enumerate(criterion_messages)
        if semantic_messages_key(messages) in keys
    }:
        raise RuntimeError("HealthBench refused criteria disagree with the sampler record")
    if not criterion_messages and marked and not keys:
        raise RuntimeError("HealthBench refusal marker without a sampler refusal")
    return marked
