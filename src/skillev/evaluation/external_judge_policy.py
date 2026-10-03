import os

EXTERNAL_JUDGE_MODEL = os.environ.get("R2FLOW_JUDGE_MODEL", "")
EXTERNAL_JUDGE_EFFORT = os.environ.get("R2FLOW_JUDGE_REASONING_EFFORT", "medium")
EXTERNAL_JUDGE_API_BASE = os.environ.get("R2FLOW_JUDGE_API_BASE", "")
EXTERNAL_JUDGE_PROVIDER = "external"
EXTERNAL_JUDGE_KEY_ENV = "R2FLOW_JUDGE_API_KEY"
EXTERNAL_JUDGE_MAX_TOKENS = 8000

HEALTHBENCH_JUDGE_PROFILE = "healthbench-judge-medium-external-per-rubric@4"
HEALTHBENCH_SEMANTIC_CALLS = 3
HEALTHBENCH_REFUSAL_RULE = "exclude-refused-criteria-at-most-one-third@1"
HEALTHBENCH_MAX_REFUSED_FRACTION = (1, 3)


def healthbench_refusals_exceed_limit(refused_count: int, criterion_count: int) -> bool:
    numerator, denominator = HEALTHBENCH_MAX_REFUSED_FRACTION
    return refused_count * denominator > criterion_count * numerator
