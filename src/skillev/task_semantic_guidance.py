TASK_SEMANTIC_GUIDANCE = "public-task-semantics@17"
HEALTHBENCH_AUDITED_TASK_SENTENCE = (
    "Write the assistant's reply to the user's latest message; the earlier messages are the "
    "conversation so far. The deliverable is one conversational reply to the user."
)
HOTPOT_DISTRACTOR_INPUT = "hotpotqa-distractor-open-passage@1"
HOTPOT_AUDITED_ANSWER_FORMAT = (
    "For a factoid question, answer with the shortest span copied from the document wording, "
    "not a sentence; for a yes/no question, answer yes or no."
)
ALFWORLD_AUDITED_TASK_SEMANTICS = (
    "Follow the task stated by the current environment. Select one admissible command from "
    "the current public action surface. The environment determines task completion. The "
    "command list describes what is currently executable, not what has already happened. "
    "'go to' navigates; 'open' and 'close' operate a container. 'take ... from ...' "
    "picks up an object and 'move ... to ...' places a held object. 'inventory' reports held "
    "objects. 'look' reports the current view; it does not pick up objects or use appliances. "
    "'clean', 'heat', and 'cool' perform the named operation using the specified appliance. "
    "'use' operates an available object. Use the exact object identifiers and commands "
    "supplied by the environment. Only actual execution feedback changes the known state; an "
    "imagined action, observation, or success in reasoning does not. Cleaned, heated, and "
    "cooled are object states, not different object names. The admissible list is local to "
    "the current state. Different object types named by the simulator are distinct "
    "categories, even when everyday language treats them as similar. An object of a different "
    "listed type is not a substitute for the requested type. The operations have separate "
    "meanings: 'clean' with a sinkbasin sets the cleaned state; 'heat' with a microwave sets "
    "the hot state; 'cool' with a fridge sets the cool state. Cleaning does not set the hot or "
    "cool state. Heating removes the cool state, and cooling removes the hot state. These "
    "meanings do not prescribe an action sequence."
)
AIME_AUDITED_TASK_SEMANTICS = "Solve the problem and state your final integer from 0 through 999."
MBPP_PLUS_TASK_SEMANTICS = (
    "Implement the requested Python function using the function name and "
    "calling convention shown in the public examples. The examples are "
    "part of the specification, including the expected return values. "
    "Return executable Python source."
)
TRIVIA_WIKIPEDIA_INPUT = "triviaqa-wikipedia-search@1"
TRAINING_PUBLIC_INPUT = "training-public-source-bridge@1"


def validate_task_semantic_guidance(version: str) -> None:
    if version != TASK_SEMANTIC_GUIDANCE:
        raise ValueError("unsupported public task semantic guidance")


def public_task_semantics(benchmark: str, *, input_profile: str) -> str:
    if benchmark == "hotpotqa":
        if input_profile != HOTPOT_DISTRACTOR_INPUT:
            raise ValueError("R2 Flow HotpotQA requires a declared HotpotQA input profile")
        return (
            "Answer the question. The documents are shown in full after the question, each "
            "headed by its title in brackets; open_passage returns one document's full "
            "text. " + HOTPOT_AUDITED_ANSWER_FORMAT
        )
    if benchmark == "triviaqa" and input_profile == TRIVIA_WIKIPEDIA_INPUT:
        return (
            "Answer the question. corpus_search returns the five passages of a frozen English "
            "Wikipedia collection that best match the words of a query (English stopwords are "
            "ignored; the remaining words are joined by OR and ranked by BM25); a passage may "
            "or may not contain the answer. Return the short answer itself, rather than a "
            "sentence restating the question."
        )
    if input_profile != TRAINING_PUBLIC_INPUT:
        raise ValueError("R2 Flow task semantics require the training input profile")
    if benchmark == "triviaqa":
        return (
            "Answer the question from your own knowledge; no documents are available. "
            "Return the short answer itself, rather than a sentence restating the question."
        )
    if benchmark == "mbpp-plus":
        return MBPP_PLUS_TASK_SEMANTICS
    if benchmark == "alfworld":
        return ALFWORLD_AUDITED_TASK_SEMANTICS
    if benchmark == "aime-2026":
        return AIME_AUDITED_TASK_SEMANTICS
    if benchmark == "healthbench":
        return HEALTHBENCH_AUDITED_TASK_SENTENCE
    raise ValueError("benchmark is outside the R2 Flow task semantics")


def phase_deliverable(benchmark: str | None) -> str | None:
    if benchmark is None:
        return None
    return {
        "healthbench": "conversation-reply@2",
        "aime-2026": "integer-answer@2",
        "alfworld": "environment-event@2",
    }.get(benchmark)


TRAINING_SUBMISSION_INSTRUCTION = "Put the final response in the answer parameter."
WRITER_SUBMISSION_INSTRUCTION = (
    "submit_answer() takes no parameters: it ends the episode, and the frozen answer writer "
    "writes the final response from the task, the listed call results and the current "
    "reasoning draft."
)
_WRITER_DELIVERABLES = {
    "conversation-reply@2": "conversation-reply@3",
    "integer-answer@2": "integer-answer@3",
}


def writer_deliverable(deliverable: str | None) -> str | None:
    if deliverable is None:
        return None
    if deliverable not in _WRITER_DELIVERABLES:
        raise ValueError("the answer writer declares the public-task-semantics@17 phase notes")
    return _WRITER_DELIVERABLES[deliverable]
