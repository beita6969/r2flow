from __future__ import annotations

import hashlib
import heapq
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from skillev.contracts import JsonValue
from skillev.contracts.answer_writer import written_answer
from skillev.contracts.skill_call_budget import SKILL_FUNCTION
from skillev.contracts.state_map import SIGMA_TRACE_QUOTIENT, SKILL_CONSUMPTION_DEPENDENCY
from skillev.contracts.wikipedia_search import SEARCH_RESULTS_STATUS, render_search_results
from skillev.policy.event_grammar import (
    AFTER_NAME,
    CALL_CLOSE,
    ENUM_CLOSE,
    OPEN,
    EventParseError,
    key_json,
    parse_event_call,
)
from skillev.rollout.legal_events import (
    ACT_FUNCTION,
    LegalEventSet,
    environment_from_observation,
    horizon,
    initial_environment,
    legal_event_set,
)

ARTIFACT_REFERENCE_VERSION = "artifact-reference@1"
CANONICAL_OUTPUT_VERSION = "canonical-output@1"
STATE_KEY_VERSION = "state-key@1"
DERIVED_FACTS_VERSION = "execution-status@1"
CLASS_RANK: Mapping[str, int] = {
    "open_passage": 0,
    "corpus_search": 0,
    "act": 1,
    "invoke_skill": 2,
    "submit_answer": 3,
}
SUBMIT_FUNCTION = "submit_answer"
INVOKE_FUNCTION = "invoke_skill"
OPEN_PASSAGE_FUNCTION = "open_passage"
CORPUS_SEARCH_FUNCTION = "corpus_search"
SHINGLE_WORDS = 8
_PARSE_BUDGET = 1 << 30
_PARSE_STOP = 0


class NonEventStepError(ValueError):
    pass


def _sha(value: object) -> str:
    return hashlib.sha256(key_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CanonicalEvent:
    u: str
    args: tuple[tuple[str, str], ...]

    @property
    def label(self) -> str:
        return key_json({"u": self.u, "args": dict(self.args)})

    @property
    def class_rank(self) -> int:
        return CLASS_RANK.get(self.u, 2)

    def arg(self, name: str) -> str:
        return dict(self.args)[name]

    def free_text(self) -> str | None:
        if self.u == CORPUS_SEARCH_FUNCTION:
            return dict(self.args).get("query")
        return dict(self.args).get("input") if self.u == INVOKE_FUNCTION else None

    def call_text(self) -> str:
        parts = [OPEN, self.u, AFTER_NAME]
        for name, value in self.args:
            parts.extend((f"<parameter={name}>\n", value, ENUM_CLOSE))
        parts.append(CALL_CLOSE)
        return "".join(parts)


@dataclass(frozen=True, slots=True)
class CanonicalOutput:
    status: str
    handle: str | None
    content: str
    fields: tuple[tuple[str, JsonValue], ...]
    env: Mapping[str, JsonValue] | None
    terminal: bool

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "status": self.status,
            "handle": self.handle,
            "content": self.content,
            "fields": [[name, value] for name, value in self.fields],
            "env": None if self.env is None else dict(self.env),
            "terminal": self.terminal,
        }


def canonical_output(event: CanonicalEvent, observation_text: str, status: str) -> CanonicalOutput:
    try:
        value = json.loads(observation_text)
    except json.JSONDecodeError as error:
        raise ValueError("observation text must be canonical JSON") from error
    if not isinstance(value, dict):
        raise ValueError("observation must be a JSON object")
    error_code = value.get("error")
    error_fields: tuple[tuple[str, JsonValue], ...] = (
        (("error", error_code),) if isinstance(error_code, str) else ()
    )
    if event.u == OPEN_PASSAGE_FUNCTION:
        if status == "success" and value.get("status") == "passage-opened":
            title, text = value.get("title"), value.get("text")
            if not isinstance(title, str) or not isinstance(text, str):
                raise ValueError("passage observation lacks title/text")
            return CanonicalOutput(status, title, text, (), None, False)
        return CanonicalOutput(status, None, "", error_fields, None, False)
    if event.u == CORPUS_SEARCH_FUNCTION:
        if status == "success" and value.get("status") == SEARCH_RESULTS_STATUS:
            passages = value.get("passages")
            if not isinstance(passages, list) or not all(isinstance(p, dict) for p in passages):
                raise ValueError("corpus_search observation lacks passages")
            return CanonicalOutput(status, None, render_search_results(passages), (), None, False)
        return CanonicalOutput(status, None, "", error_fields, None, False)
    if event.u == INVOKE_FUNCTION:
        if value.get("status") == "skill-executed":
            output = value.get("output")
            if not isinstance(output, str):
                raise ValueError("executor observation lacks output")
            fields: list[tuple[str, JsonValue]] = [
                (name, value[name])
                for name in ("skill_id", "version", "output_truncated")
                if name in value
            ]
            return CanonicalOutput(status, None, output, (*fields, *error_fields), None, False)
        return CanonicalOutput(status, None, "", error_fields, None, False)
    if event.u == ACT_FUNCTION:
        env = environment_from_observation(value) if status == "success" else None
        if env is None:
            return CanonicalOutput(status, None, "", error_fields, None, False)
        return CanonicalOutput(status, None, str(env["text"]), (), env, bool(env["terminal"]))
    if event.u == SUBMIT_FUNCTION:
        answer = written_answer(value) if not event.args and status == "success" else None
        if answer is not None:
            return CanonicalOutput(status, None, answer, (), None, True)
        accepted = status == "success" and value == {"status": "accepted_for_evaluation"}
        return CanonicalOutput(status, None, "", error_fields, None, accepted)
    raise ValueError(f"canonical-output@1 does not declare function {event.u!r}")


@dataclass(frozen=True, slots=True)
class EventInstance:
    event: CanonicalEvent
    occurrence: int
    output: CanonicalOutput

    @property
    def label(self) -> str:
        return self.event.label

    @property
    def instance_id(self) -> str:
        return _sha({"label": self.event.label, "occ": self.occurrence})

    def priority(self) -> tuple[int, str, int]:
        return (self.event.class_rank, self.event.label, self.occurrence)


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())


def references(consumer: EventInstance, producer: EventInstance) -> bool:
    text = consumer.event.free_text()
    if text is None:
        return False
    words = _words(text)
    joined = " " + " ".join(words) + " "
    handle = producer.output.handle
    if handle is not None:
        handle_words = _words(handle)
        if handle_words and " " + " ".join(handle_words) + " " in joined:
            return True
    content = _words(producer.output.content)
    if len(content) < SHINGLE_WORDS or len(words) < SHINGLE_WORDS:
        return False
    grams = {
        tuple(words[index : index + SHINGLE_WORDS])
        for index in range(len(words) - SHINGLE_WORDS + 1)
    }
    return any(
        tuple(content[index : index + SHINGLE_WORDS]) in grams
        for index in range(len(content) - SHINGLE_WORDS + 1)
    )


def consumes(consumer: EventInstance, producer: EventInstance) -> bool:
    return consumer.event.free_text() is not None and producer.event.u in (
        INVOKE_FUNCTION,
        OPEN_PASSAGE_FUNCTION,
        CORPUS_SEARCH_FUNCTION,
        ACT_FUNCTION,
    )


def depends(a: EventInstance, b: EventInstance) -> bool:
    return (
        a.label == b.label
        or SUBMIT_FUNCTION in (a.event.u, b.event.u)
        or a.output.terminal
        or b.output.terminal
        or (a.event.u == ACT_FUNCTION and b.event.u == ACT_FUNCTION)
        or references(a, b)
        or references(b, a)
        or consumes(a, b)
        or consumes(b, a)
    )


@dataclass(frozen=True, slots=True)
class SharedState:
    root_key: str
    horizon: int
    root_env: Mapping[str, JsonValue] | None
    events: tuple[EventInstance, ...]
    edges: tuple[tuple[int, int], ...]
    omega: Mapping[str, JsonValue]
    facts: tuple[tuple[str, str, bool], ...]
    terminal: str | None

    @property
    def rank(self) -> int:
        return len(self.events)

    @property
    def env(self) -> Mapping[str, JsonValue] | None:
        env = self.omega["env"]
        return env if isinstance(env, dict) else None

    def maximal(self) -> tuple[int, ...]:
        sources = {source for source, _ in self.edges}
        return tuple(index for index in range(self.rank) if index not in sources)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "v": STATE_KEY_VERSION,
            "sigma": SIGMA_TRACE_QUOTIENT,
            "dep": SKILL_CONSUMPTION_DEPENDENCY,
            "ref": ARTIFACT_REFERENCE_VERSION,
            "out": CANONICAL_OUTPUT_VERSION,
            "facts_version": DERIVED_FACTS_VERSION,
            "root": self.root_key,
            "horizon": self.horizon,
            "events": [[item.label, item.occurrence] for item in self.events],
            "edges": [list(edge) for edge in self.edges],
            "artifacts": [_sha(item.output.to_value()) for item in self.events],
            "facts": [list(fact) for fact in self.facts],
            "omega": dict(self.omega),
            "rank": self.rank,
            "terminal": self.terminal,
        }

    def key(self) -> str:
        return _sha(self.to_value())


def build_trace(
    *,
    root_key: str,
    horizon: int,
    root_env: Mapping[str, JsonValue] | None,
    instances: Sequence[EventInstance],
) -> SharedState:
    count = len(instances)
    successors: list[list[int]] = [[] for _ in range(count)]
    indegree = [0] * count
    for later in range(count):
        for earlier in range(later):
            if depends(instances[earlier], instances[later]):
                successors[earlier].append(later)
                indegree[later] += 1
    heap = [(instances[i].priority(), i) for i in range(count) if indegree[i] == 0]
    heapq.heapify(heap)
    order: list[int] = []
    while heap:
        _, index = heapq.heappop(heap)
        order.append(index)
        for nxt in successors[index]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                heapq.heappush(heap, (instances[nxt].priority(), nxt))
    position = {index: rank for rank, index in enumerate(order)}
    reach = [0] * count
    direct = [0] * count
    for index in range(count):
        for nxt in successors[index]:
            direct[position[index]] |= 1 << position[nxt]
    for source in reversed(range(count)):
        closure = direct[source]
        bits = direct[source]
        while bits:
            low = bits & -bits
            closure |= reach[low.bit_length() - 1]
            bits ^= low
        reach[source] = closure
    edges: list[tuple[int, int]] = []
    for source in range(count):
        for target in range(source + 1, count):
            if not reach[source] >> target & 1:
                continue
            through = any(
                reach[source] >> middle & 1 and reach[middle] >> target & 1
                for middle in range(source + 1, target)
            )
            if not through:
                edges.append((source, target))
    events = tuple(instances[index] for index in order)
    env: Mapping[str, JsonValue] | None = root_env
    for item in events:
        if item.event.u == ACT_FUNCTION and item.output.env is not None:
            env = item.output.env
    terminal: str | None = None
    if any(item.event.u == SUBMIT_FUNCTION and item.output.terminal for item in events):
        terminal = "submitted"
    elif any(item.event.u == ACT_FUNCTION and item.output.terminal for item in events):
        terminal = "environment-terminal"
    elif count >= horizon:
        terminal = "horizon"
    facts = tuple(
        sorted(
            (item.instance_id, DERIVED_FACTS_VERSION, item.output.status == "success")
            for item in events
        )
    )
    return SharedState(
        root_key=root_key,
        horizon=horizon,
        root_env=root_env,
        events=events,
        edges=tuple(edges),
        omega={
            "rank": count,
            "turns_remaining": horizon - count,
            "env": None if env is None else dict(env),
        },
        facts=facts,
        terminal=terminal,
    )


class StepLike(Protocol):
    @property
    def action_text(self) -> str: ...

    @property
    def observation_text(self) -> str: ...

    @property
    def observation_status(self) -> str: ...


@dataclass(frozen=True, slots=True)
class StepParts:
    action_text: str
    observation_text: str
    observation_status: str


def state_map_of(initial_text: str) -> str | None:
    from skillev.policy.phase_context import PhaseContextSpec

    spec, _ = PhaseContextSpec.split(initial_text)
    return None if spec is None else spec.state_map


def root_key(initial_text: str) -> str:
    return _sha({"h0": initial_text})


def legal_events_at(initial_text: str, state: SharedState) -> LegalEventSet:
    return legal_event_set(
        initial_text,
        rank=state.rank,
        omega_env=state.env,
        terminal=state.terminal is not None,
        skill_calls=skill_calls(state),
    )


def skill_calls(state: SharedState) -> int:
    return sum(1 for item in state.events if item.event.u == SKILL_FUNCTION)


def parse_step_event(legal: LegalEventSet, action_text: str) -> CanonicalEvent:
    if not legal.functions:
        raise NonEventStepError("no event is legal at a terminal state")
    spec = legal.grammar_spec(budget=_PARSE_BUDGET, stop_token_id=_PARSE_STOP)
    try:
        name, args = parse_event_call(spec, action_text)
    except EventParseError as error:
        raise NonEventStepError(f"recorded action is not a legal event: {error}") from None
    return CanonicalEvent(name, tuple(args.items()))


def sigma(initial_text: str, steps: Sequence[StepLike]) -> SharedState:
    if state_map_of(initial_text) != SIGMA_TRACE_QUOTIENT:
        raise ValueError("sigma requires a declared state map")
    total = horizon(initial_text)
    root_env = initial_environment(initial_text)
    env: Mapping[str, JsonValue] | None = root_env
    occurrences: dict[str, int] = {}
    instances: list[EventInstance] = []
    terminal = False
    calls = 0
    for rank, step in enumerate(steps):
        legal = legal_event_set(
            initial_text, rank=rank, omega_env=env, terminal=terminal, skill_calls=calls
        )
        event = parse_step_event(legal, step.action_text)
        calls += event.u == SKILL_FUNCTION
        output = canonical_output(event, step.observation_text, step.observation_status)
        occurrence = occurrences.get(event.label, 0)
        occurrences[event.label] = occurrence + 1
        instances.append(EventInstance(event, occurrence, output))
        if event.u == ACT_FUNCTION and output.env is not None:
            env = output.env
        terminal = terminal or output.terminal
    return build_trace(
        root_key=root_key(initial_text),
        horizon=total,
        root_env=root_env,
        instances=instances,
    )


def sigma_from_parts(
    initial_text: str,
    previous_steps: Sequence[StepLike],
    action_text: str,
    observation_text: str,
    observation_status: str,
) -> SharedState:
    return sigma(
        initial_text,
        (*previous_steps, StepParts(action_text, observation_text, observation_status)),
    )


def is_legal(initial_text: str, state: SharedState, event: CanonicalEvent) -> bool:
    return legal_events_at(initial_text, state).allows(event.u, dict(event.args))


@dataclass(frozen=True, slots=True)
class InEdge:
    predecessor: SharedState
    event_index: int
    event: EventInstance

    @property
    def predecessor_key(self) -> str:
        return self.predecessor.key()

    @property
    def label(self) -> str:
        return self.event.label

    @property
    def label_hash(self) -> str:
        return _sha({"label": self.event.label})

    def call_text(self) -> str:
        return self.event.event.call_text()


def predecessor(state: SharedState, index: int) -> SharedState:
    rest = [item for position, item in enumerate(state.events) if position != index]
    return build_trace(
        root_key=state.root_key,
        horizon=state.horizon,
        root_env=state.root_env,
        instances=rest,
    )


def in_edges(initial_text: str, state: SharedState) -> tuple[InEdge, ...]:
    if root_key(initial_text) != state.root_key:
        raise ValueError("state belongs to another H0")
    edges: dict[tuple[str, str], InEdge] = {}
    for index in state.maximal():
        item = state.events[index]
        pred = predecessor(state, index)
        if pred.terminal is not None or not is_legal(initial_text, pred, item.event):
            continue
        edges.setdefault((pred.key(), item.label), InEdge(pred, index, item))
    return tuple(sorted(edges.values(), key=lambda edge: edge.event.priority()))


def actual_in_edge_index(
    initial_text: str,
    previous: SharedState,
    current: SharedState,
    event: CanonicalEvent,
    edges: Sequence[InEdge] | None = None,
) -> int:
    candidates = in_edges(initial_text, current) if edges is None else edges
    matches = [
        position
        for position, edge in enumerate(candidates)
        if edge.label == event.label and edge.predecessor_key == previous.key()
    ]
    if len(matches) != 1:
        raise ValueError("the actual edge is not exactly one in-edge of its state")
    return matches[0]


def assert_actual_in_edge(initial_text: str, steps: Sequence[StepLike], t: int) -> int:
    if type(t) is not int or not 1 <= t <= len(steps):
        raise ValueError("t must index a recorded step (one-based)")
    previous = sigma(initial_text, steps[: t - 1])
    current = sigma(initial_text, steps[:t])
    event = parse_step_event(legal_events_at(initial_text, previous), steps[t - 1].action_text)
    return actual_in_edge_index(initial_text, previous, current, event)


__all__ = [
    "ARTIFACT_REFERENCE_VERSION",
    "CANONICAL_OUTPUT_VERSION",
    "CLASS_RANK",
    "DERIVED_FACTS_VERSION",
    "SIGMA_TRACE_QUOTIENT",
    "STATE_KEY_VERSION",
    "CanonicalEvent",
    "CanonicalOutput",
    "EventInstance",
    "InEdge",
    "NonEventStepError",
    "SharedState",
    "StepParts",
    "actual_in_edge_index",
    "assert_actual_in_edge",
    "build_trace",
    "canonical_output",
    "depends",
    "in_edges",
    "is_legal",
    "legal_events_at",
    "parse_step_event",
    "predecessor",
    "references",
    "root_key",
    "sigma",
    "sigma_from_parts",
    "state_map_of",
]
