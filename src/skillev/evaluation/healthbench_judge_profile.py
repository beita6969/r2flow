from dataclasses import asdict
from typing import cast

from skillev.contracts import JsonValue

from .external_judge_policy import (
    EXTERNAL_JUDGE_API_BASE,
    EXTERNAL_JUDGE_EFFORT,
    EXTERNAL_JUDGE_KEY_ENV,
    EXTERNAL_JUDGE_MAX_TOKENS,
    EXTERNAL_JUDGE_MODEL,
    EXTERNAL_JUDGE_PROVIDER,
    HEALTHBENCH_JUDGE_PROFILE,
    HEALTHBENCH_REFUSAL_RULE,
    HEALTHBENCH_SEMANTIC_CALLS,
)
from .healthbench_official import HealthBenchExternalJudgeProfile

METRIC = "judge-medium-api-rubric-score"
VERIFIER = "healthbench-simple-evals-judge-medium-external@4"


def judge_profile() -> HealthBenchExternalJudgeProfile:
    return HealthBenchExternalJudgeProfile(
        profile_id=HEALTHBENCH_JUDGE_PROFILE,
        backend="openai-chat-completions",
        model=EXTERNAL_JUDGE_MODEL,
        rubric_source_repository="https://github.com/openai/simple-evals",
        rubric_source_revision="652c89d0ca9df547706735883097e9537d40dc47",
        rubric_source_path="healthbench_eval.py",
        endpoint_environment="R2FLOW_JUDGE_API_BASE",
        api_key_environment=EXTERNAL_JUDGE_KEY_ENV,
        call_mode="per-rubric",
        response_format="json-object",
        max_completion_tokens=EXTERNAL_JUDGE_MAX_TOKENS,
        reasoning_effort=EXTERNAL_JUDGE_EFFORT,
        temperature=None,
        top_p=None,
        request_timeout_seconds=120.0,
        maximum_attempts=1,
    )


def healthbench_condition() -> dict[str, JsonValue]:
    return {
        **cast(dict[str, JsonValue], asdict(judge_profile())),
        "provider": EXTERNAL_JUDGE_PROVIDER,
        "upstream_api": "responses",
        "endpoint": EXTERNAL_JUDGE_API_BASE,
        "metric": METRIC,
        "verifier": VERIFIER,
        "semantic_calls_per_rubric": HEALTHBENCH_SEMANTIC_CALLS,
        "semantic_repair": "simple-evals-unparseable-grader-output-retry@1",
        "refusal_rule": HEALTHBENCH_REFUSAL_RULE,
        "ttb_reward": "clip(raw-rubric-score,0,1)",
        "success": "raw-rubric-score>=0.60-and-no-negative-rubric",
    }
