from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Final

from skillev.contracts import JsonValue
from skillev.contracts.answer_writer import EXECUTOR_ANSWER
from skillev.contracts.canonical import stable_hash
from skillev.contracts.wikipedia_search import (
    PASSAGE_TEXT_CAP,
    PASSAGES_PER_QUERY,
    WIKIPEDIA_PASSAGE_FORMAT,
    WIKIPEDIA_QUERY_POLICY,
    WIKIPEDIA_RANKING,
    WIKIPEDIA_SEARCH_PROFILE,
    WIKIPEDIA_SEARCH_RESOURCE,
    WIKIPEDIA_SEARCH_TOOL_NAME,
)
from skillev.rollout import RolloutTask
from skillev.rollout.action_surface import (
    ActionSurface,
    ArgumentFieldSpec,
    ArgumentType,
    CompletionSpec,
    RolloutBudgetProfile,
    TerminalMode,
    ToolActionSpec,
)
from skillev.task_semantic_guidance import (
    HOTPOT_DISTRACTOR_INPUT,
    TRAINING_PUBLIC_INPUT,
    TRIVIA_WIKIPEDIA_INPUT,
    WRITER_SUBMISSION_INSTRUCTION,
)

from .passage_corpus import (
    PASSAGE_TOOL_NAME,
    PASSAGE_TOOL_RESOURCE,
    PassageCorpus,
    distractor_context_query,
    hotpot_open_passage_projection,
)

R2FLOW_TOOL_SET: Final = "r2flow-tool-set@5"
HOTPOTQA_DISTRACTOR = (
    f"{HOTPOT_DISTRACTOR_INPUT} (question + all documents in full in H0, official order; "
    "open_passage kept)"
)
TRIVIAQA_WIKIPEDIA = (
    f"{WIKIPEDIA_SEARCH_PROFILE} top-{PASSAGES_PER_QUERY} {WIKIPEDIA_QUERY_POLICY} "
    f"({WIKIPEDIA_SEARCH_TOOL_NAME}; pinned DPR Wikipedia passages, atlas_wikipedia_fts5)"
)
TOOL_REGIMES: Final[dict[str, dict[str, JsonValue]]] = {
    "hotpotqa": {
        "retrieval": HOTPOTQA_DISTRACTOR,
        "tools": ["open_passage", "invoke_skill", "submit_answer"],
        "completion_writer": EXECUTOR_ANSWER,
    },
    "triviaqa": {
        "retrieval": TRIVIAQA_WIKIPEDIA,
        "tools": [WIKIPEDIA_SEARCH_TOOL_NAME, "invoke_skill", "submit_answer"],
        "completion_writer": EXECUTOR_ANSWER,
    },
    "aime-2026": {
        "retrieval": "none",
        "tools": ["invoke_skill", "submit_answer"],
        "completion_writer": EXECUTOR_ANSWER,
    },
    "mbpp-plus": {
        "retrieval": "none",
        "tools": ["invoke_skill", "submit_answer"],
        "completion_writer": EXECUTOR_ANSWER,
    },
    "healthbench": {
        "retrieval": "none",
        "tools": ["invoke_skill", "submit_answer"],
        "completion_writer": EXECUTOR_ANSWER,
    },
    "alfworld": {"retrieval": "none", "tools": ["act", "invoke_skill"]},
}
_PROFILES = {
    "hotpotqa": RolloutBudgetProfile("r2flow-open-passage", 14, 1024, 1024),
    "triviaqa": RolloutBudgetProfile("r2flow-static", 8, 1024, 1024),
    "aime-2026": RolloutBudgetProfile("r2flow-static", 8, 1024, 1024),
    "healthbench": RolloutBudgetProfile("r2flow-long-answer", 8, 1024, 1536),
    "mbpp-plus": RolloutBudgetProfile("r2flow-code", 8, 1024, 2048),
}


def r2flow_action_contract(
    benchmark: str, *, corpus: PassageCorpus | None = None, max_steps: int | None = None
) -> tuple[ActionSurface, RolloutBudgetProfile]:
    if benchmark not in TOOL_REGIMES:
        raise ValueError(f"{benchmark} is outside {R2FLOW_TOOL_SET}")
    if (corpus is not None) != (benchmark == "hotpotqa"):
        raise ValueError("exactly HotpotQA carries a per-question passage corpus")
    if benchmark == "alfworld":
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("ALFWorld action contract requires max_steps")
        return (
            ActionSurface(
                terminal_mode=TerminalMode.ENVIRONMENT,
                tools=(
                    ToolActionSpec(
                        "alfworld",
                        "act",
                        {"command": ArgumentFieldSpec(ArgumentType.STRING, True)},
                        {"command": "look"},
                    ),
                ),
                dynamic_choice_fields=("admissible_commands",),
            ),
            RolloutBudgetProfile("r2flow-embodied", max_steps, 1024, 1024),
        )
    if max_steps is not None:
        raise ValueError("only ALFWorld declares native max_steps")
    tools: tuple[ToolActionSpec, ...] = ()
    if benchmark == "triviaqa":
        tools = (
            ToolActionSpec(
                WIKIPEDIA_SEARCH_RESOURCE,
                WIKIPEDIA_SEARCH_TOOL_NAME,
                {"query": ArgumentFieldSpec(ArgumentType.STRING, True)},
                {"query": "SEARCH_QUERY"},
            ),
        )
    if corpus is not None:
        titles = corpus.titles()
        tools = (
            ToolActionSpec(
                PASSAGE_TOOL_RESOURCE,
                PASSAGE_TOOL_NAME,
                {"title": ArgumentFieldSpec(ArgumentType.STRING, True, choices=titles)},
                {"title": titles[0]},
            ),
        )
    return (
        ActionSurface(
            terminal_mode=TerminalMode.EXPLICIT_COMPLETION,
            tools=tools,
            completion=CompletionSpec({}, {}),
            instructions=(WRITER_SUBMISSION_INSTRUCTION,),
            completion_writer=EXECUTOR_ANSWER,
        ),
        _PROFILES[benchmark],
    )


def r2flow_hotpot_task(task: RolloutTask) -> tuple[RolloutTask, PassageCorpus]:
    question, corpus = hotpot_open_passage_projection(task.query)
    surface, profile = r2flow_action_contract("hotpotqa", corpus=corpus)
    context = task.public_context if isinstance(task.public_context, dict) else {}
    return (
        replace(
            task,
            query=distractor_context_query(question, corpus),
            public_context={
                **context,
                "input_profile": HOTPOT_DISTRACTOR_INPUT,
                "payload": {
                    "input_profile": HOTPOT_DISTRACTOR_INPUT,
                    "document_count": len(corpus.documents),
                },
            },
            action_surface=surface,
            budget_profile=profile,
        ),
        corpus,
    )


def r2flow_trivia_task(task: RolloutTask) -> RolloutTask:
    context = task.public_context if isinstance(task.public_context, dict) else {}
    profile = context.get("input_profile", TRAINING_PUBLIC_INPUT)
    if (
        context.get("benchmark_id") != "triviaqa"
        or context.get("payload") != {"initial_context": "none"}
        or profile != TRAINING_PUBLIC_INPUT
        or task.model_visible_messages
        or not isinstance(task.query, str)
        or not task.query.strip()
    ):
        raise ValueError("TriviaQA corpus_search requires a question-only training-format input")
    surface, profile_budget = r2flow_action_contract("triviaqa")
    return replace(
        task,
        public_context={
            **context,
            "input_profile": TRIVIA_WIKIPEDIA_INPUT,
            "payload": {
                "input_profile": TRIVIA_WIKIPEDIA_INPUT,
                "passages_per_query": PASSAGES_PER_QUERY,
            },
        },
        action_surface=surface,
        budget_profile=profile_budget,
    )


def wikipedia_corpus_identity(
    *, sha256: str, size_bytes: int, passages: int, corpus_id: str
) -> dict[str, JsonValue]:
    if (
        type(sha256) is not str
        or len(sha256) != 64
        or any(c not in "0123456789abcdef" for c in sha256)
        or type(size_bytes) is not int
        or size_bytes < 1
        or type(passages) is not int
        or passages < 1
        or type(corpus_id) is not str
        or not corpus_id.strip()
    ):
        raise ValueError("a Wikipedia corpus identity pins sha256, size, passages and corpus id")
    return {
        "corpus_id": corpus_id,
        "passage_format": WIKIPEDIA_PASSAGE_FORMAT,
        "passage_text_cap": PASSAGE_TEXT_CAP,
        "passages": passages,
        "passages_per_query": PASSAGES_PER_QUERY,
        "profile": WIKIPEDIA_SEARCH_PROFILE,
        "query_policy": WIKIPEDIA_QUERY_POLICY,
        "ranking": WIKIPEDIA_RANKING,
        "sha256": sha256,
        "size_bytes": size_bytes,
    }


def retrieval_corpus_hash(
    corpora: dict[str, PassageCorpus], *, wikipedia: Mapping[str, JsonValue] | None = None
) -> str:
    hotpot = stable_hash(
        sorted([source, corpus.corpus_hash()] for source, corpus in corpora.items())
    )
    if wikipedia is None:
        return hotpot
    return stable_hash({"hotpotqa": hotpot, "triviaqa": dict(wikipedia)})


__all__ = [
    "HOTPOTQA_DISTRACTOR",
    "R2FLOW_TOOL_SET",
    "TOOL_REGIMES",
    "TRIVIAQA_WIKIPEDIA",
    "r2flow_action_contract",
    "r2flow_hotpot_task",
    "r2flow_trivia_task",
    "retrieval_corpus_hash",
    "wikipedia_corpus_identity",
]
