from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Protocol, cast

import yaml

from skillev.contracts import JsonValue
from skillev.runtime.skill_md import TRANSFERABLE_FAMILIES, normalise_body

from .author_memory import (
    PreviousDraft,
    drafts_for_families,
    memory_record,
    render_previous_drafts,
)
from .evidence import (
    REDACTED,
    SKILL_FUNCTION,
    SOLVER_FAILING_QUERIES,
    SOLVER_ROLLOUTS_PER_FAILING_QUERY,
    SOLVER_SUCCESSES,
    clip_middle,
    public_needles,
)
from .skill_md import (
    DEFAULT_TOKEN_CAP,
    DESCRIPTION_MAX_CHARS,
    SKILL_MD_VALIDATION_VERSION,
    TOKEN_ESTIMATE_RULE,
    forbidden_needle_patterns,
    forbidden_patterns,
    needle_sha256,
    redact_forbidden,
    validate_skill_md,
)
from .types import (
    STRUCTURAL_EDITS,
    AuthoredEdit,
    CandidateEdit,
    EditKind,
    LibraryVersion,
    PhaseEvidence,
    SkillSpec,
    VerifierObs,
)

AUTHOR_PROMPT_VERSION: Final = "r2flow-author-prompt@5"
AUTHOR_RECORD_FORMAT: Final = "r2flow-author-record@1"
MATERIAL_RULE: Final = "author-material=solver-evidence@1"
SKILL_SEPARATOR: Final = "=== NEXT SKILL ==="
SUBMISSION_FUNCTIONS: Final = frozenset({"submit_answer"})
REDACTION_MARKER: Final = "[REMOVED]"
DEFAULT_MAX_OUTPUT_TOKENS: Final = 16000
REPLY_ECHO_CHARS: Final = 8000
SELECTION_RULE: Final = "author-material-selection=distinct-query@1"
MATERIAL_CHAR_BUDGET: Final = 60000
MAX_TRAJECTORIES: Final = SOLVER_FAILING_QUERIES * SOLVER_ROLLOUTS_PER_FAILING_QUERY + (
    SOLVER_SUCCESSES
)
TAIL_STEPS: Final = 12
EARLIER_TARGET_STEPS: Final = 3
FIELD_CHARS: Final = 2000
COMPACT_CHARS: Final = 400
USER_AGENT: Final = "r2flow-author/1.0"
_HIDDEN_EVIDENCE_KEYS: Final = frozenset(
    {
        "trajectory_ids",
        "query_ids",
        "task_ids",
        "domains",
        "verifier_evidence",
        "native_outcomes",
    }
)
BOOTSTRAP_RULE: Final = "empty-slot-bootstrap@1"
BOOTSTRAP_TEMPLATE: Final = "refine-empty-slot-bootstrap@1"
_Z_LABELS: Final = ("context class", "previous-step failure mode", "prompt-length bucket", "turn")


class AuthorClientError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True, slots=True)
class AuthorResponse:
    text: str
    response_id: str
    model: str
    usage: dict[str, JsonValue]
    status: str | None = None
    reasoning_sent: bool = False


class AuthorClient(Protocol):
    @property
    def model(self) -> str: ...

    def complete(self, prompt: str, *, max_output_tokens: int) -> AuthorResponse: ...


@dataclass(slots=True)
class FakeAuthorClient:
    replies: Sequence[str] | Callable[[str], str]
    model: str = "fake-author"
    prompts: list[str] = field(default_factory=list)

    def complete(self, prompt: str, *, max_output_tokens: int) -> AuthorResponse:
        self.prompts.append(prompt)
        index = len(self.prompts)
        if callable(self.replies):
            text = self.replies(prompt)
        elif index <= len(self.replies):
            text = self.replies[index - 1]
        else:
            raise AuthorClientError("fake author has no scripted reply left")
        usage: dict[str, JsonValue] = {
            "input_tokens": len(prompt.split()),
            "output_tokens": len(text.split()),
            "total_tokens": len(prompt.split()) + len(text.split()),
        }
        return AuthorResponse(
            text=text,
            response_id=f"fake-{index}",
            model=self.model,
            usage=usage,
            status="completed",
        )


_SECRET_RES: Final = (
    re.compile(r"(?i)bearer\s+\S+"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"),
)


def _scrub(text: str) -> str:
    for pattern in _SECRET_RES:
        text = pattern.sub("[redacted]", text)
    return text[:300]


def _build_request(
    url: str, data: bytes, key_file: Path, user_agent: str
) -> urllib.request.Request:
    try:
        key = key_file.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise AuthorClientError(
            f"cannot read the author key file ({type(error).__name__})"
        ) from None
    if not key or any(c.isspace() for c in key):
        raise AuthorClientError("the author key file does not hold one token")
    return urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": user_agent,
        },
    )


def _is_timeout(error: BaseException) -> bool:
    if isinstance(error, TimeoutError):
        return True
    return isinstance(error, urllib.error.URLError) and isinstance(error.reason, TimeoutError)


def _stderr_line(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


class _RejectedError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(status, detail)
        self.status = status
        self.detail = detail


class GatewayAuthorClient:
    def __init__(
        self,
        base_file: str | Path,
        key_file: str | Path,
        model: str,
        timeout_s: float = 600.0,
        *,
        reasoning_effort: str | None = "high",
        max_retries: int = 3,
        max_rate_limited_retries: int = 8,
        backoff_s: float = 2.0,
        max_backoff_s: float = 60.0,
        urlopen: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        user_agent: str = USER_AGENT,
        retry_log: Callable[[str], None] | None = None,
    ) -> None:
        if not model or type(model) is not str:
            raise ValueError("author model must be non-empty text")
        if timeout_s <= 0 or max_retries < 0 or max_rate_limited_retries < 0:
            raise ValueError("timeout_s must be positive and the retry budgets non-negative")
        self._base_file = Path(base_file)
        self._key_file = Path(key_file)
        self._model = model
        self.timeout_s = float(timeout_s)
        self._reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.max_rate_limited_retries = max_rate_limited_retries
        self.backoff_s = backoff_s
        self.max_backoff_s = max_backoff_s
        self._retry_log = retry_log or _stderr_line
        self._urlopen = urlopen or urllib.request.urlopen
        self._sleep = sleep
        self.user_agent = user_agent

    def __repr__(self) -> str:
        return f"GatewayAuthorClient(model={self._model!r}, base_file={str(self._base_file)!r})"

    @property
    def model(self) -> str:
        return self._model

    def _url(self) -> str:
        try:
            base = self._base_file.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise AuthorClientError(
                f"cannot read the author base file ({type(error).__name__})"
            ) from None
        if not re.match(r"^https?://\S+$", base):
            raise AuthorClientError("the author base file does not hold one http(s) URL")
        return base.rstrip("/") + "/responses"

    def _post(self, url: str, payload: dict[str, JsonValue]) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        attempt = timeouts = throttled = 0
        while True:
            request = _build_request(url, data, self._key_file, self.user_agent)
            wait = min(self.backoff_s * (2**attempt), self.max_backoff_s)
            try:
                with self._urlopen(request, timeout=self.timeout_s) as response:
                    raw = response.read()
            except urllib.error.HTTPError as error:
                status = int(error.code)
                try:
                    detail = _scrub(error.read(2000).decode("utf-8", "replace"))
                except Exception:
                    detail = ""
                if status != 429 and status < 500:
                    raise _RejectedError(status, detail) from None
                retry_after = error.headers.get("Retry-After") if error.headers else None
                if retry_after is not None and str(retry_after).strip().isdigit():
                    wait = max(wait, min(float(str(retry_after).strip()), 120.0))
                failure = f"HTTP {status}: {detail}"
                label = f"HTTP {status}"
                throttled += 1
            except Exception as error:
                if not _is_timeout(error):
                    raise AuthorClientError(
                        f"author request failed ({type(error).__name__})"
                    ) from None
                failure = label = "timeout"
                timeouts += 1
            else:
                try:
                    body = json.loads(raw)
                except ValueError:
                    raise AuthorClientError("author response is not JSON") from None
                if not isinstance(body, dict):
                    raise AuthorClientError("author response is not a JSON object")
                return body
            if timeouts > self.max_retries or throttled > self.max_rate_limited_retries:
                raise AuthorClientError(
                    f"author request failed after {attempt + 1} attempts ({failure})"
                )
            wait = min(wait, 120.0)
            self._retry_log(
                f"[author-retry] model={self._model} failure={label} "
                f"attempt={attempt + 1} wait={wait:.0f}s"
            )
            self._sleep(wait)
            attempt += 1

    def complete(self, prompt: str, *, max_output_tokens: int) -> AuthorResponse:
        if type(prompt) is not str or not prompt:
            raise ValueError("author prompt must be non-empty text")
        url = self._url()
        payload: dict[str, JsonValue] = {
            "model": self._model,
            "input": prompt,
            "max_output_tokens": int(max_output_tokens),
        }
        if self._reasoning_effort is not None:
            payload["reasoning"] = {"effort": self._reasoning_effort}
        try:
            body = self._post(url, payload)
        except _RejectedError as rejected:
            if "reasoning" in payload and "reasoning" in rejected.detail.lower():
                self._reasoning_effort = None
                del payload["reasoning"]
                try:
                    body = self._post(url, payload)
                except _RejectedError as again:
                    raise AuthorClientError(
                        f"author request rejected: HTTP {again.status}: {again.detail}",
                        status=again.status,
                    ) from None
            else:
                raise AuthorClientError(
                    f"author request rejected: HTTP {rejected.status}: {rejected.detail}",
                    status=rejected.status,
                ) from None
        if body.get("error"):
            raise AuthorClientError("author response carries an error object")
        usage = body.get("usage")
        return AuthorResponse(
            text=_response_text(body),
            response_id=str(body.get("id") or ""),
            model=str(body.get("model") or self._model),
            usage=cast(dict[str, JsonValue], usage) if isinstance(usage, dict) else {},
            status=str(body["status"]) if body.get("status") is not None else None,
            reasoning_sent="reasoning" in payload,
        )


def _response_text(body: Mapping[str, Any]) -> str:
    texts: list[str] = []
    output = body.get("output")
    for item in output if isinstance(output, list) else []:
        if not isinstance(item, dict) or item.get("type") not in (None, "message"):
            continue
        content = item.get("content")
        for part in content if isinstance(content, list) else []:
            if (
                isinstance(part, dict)
                and isinstance(part.get("text"), str)
                and part.get("type") in (None, "output_text", "text")
            ):
                texts.append(part["text"])
    if not texts and isinstance(body.get("output_text"), str):
        texts.append(body["output_text"])
    return "".join(texts)


@dataclass(frozen=True, slots=True)
class _Step:
    step_index: int
    function: str
    arguments: str
    skill_id: str | None
    executor_output: str
    observation: str
    reasoning: str
    verifier: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class _Trajectory:
    trajectory_id: str
    family: str
    cluster: tuple[str, ...]
    query_text: str
    reward: float
    success: bool
    steps: tuple[_Step, ...]
    order: tuple[int, int] = (0, 0)
    diagnostics: tuple[str, ...] = ()
    termination: str = ""
    repeated: tuple[tuple[str, int], ...] = ()
    public_sha: frozenset[str] = frozenset()


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _number(value: object, default: float = 0.0) -> float:
    return (
        float(value) if isinstance(value, int | float) and not isinstance(value, bool) else default
    )


def _parse_material(
    entries: Sequence[Mapping[str, Any]], evidence: PhaseEvidence
) -> list[_Trajectory]:
    edge_skills = {
        (trajectory.trajectory_id, edge.step_index): edge.skill_id
        for trajectory in evidence.trajectories
        for edge in trajectory.edges
    }
    if not entries:
        return [
            _Trajectory(
                t.trajectory_id,
                t.family,
                (),
                "",
                t.reward,
                t.success,
                tuple(
                    _Step(e.step_index, e.event_function, "", e.skill_id, "", "", "", None)
                    for e in t.edges
                ),
            )
            for t in evidence.trajectories
        ]
    parsed = []
    for entry in entries:
        if not isinstance(entry, Mapping) or type(entry.get("trajectory_id")) is not str:
            raise ValueError("every material entry needs a text trajectory_id")
        tid = entry["trajectory_id"]
        steps = []
        for number, raw in enumerate(entry.get("steps") or (), start=1):
            if not isinstance(raw, Mapping):
                raise ValueError(f"material steps of {tid} must be objects")
            index = raw.get("step_index")
            index = index if type(index) is int else number
            skill = raw.get("skill_id")
            verifier = raw.get("verifier")
            steps.append(
                _Step(
                    step_index=index,
                    function=_text(raw.get("function")) or "?",
                    arguments=_text(raw.get("arguments")),
                    skill_id=skill if type(skill) is str else edge_skills.get((tid, index)),
                    executor_output=_text(raw.get("executor_output")),
                    observation=_text(raw.get("observation")),
                    reasoning=_text(raw.get("reasoning") or raw.get("reasoning_excerpt")),
                    verifier=verifier if isinstance(verifier, Mapping) else None,
                )
            )
        cluster = entry.get("cluster")
        diagnostics = entry.get("diagnostics")
        episode = entry.get("episode")
        episode = episode if isinstance(episode, Mapping) else {}
        repeated = episode.get("repeated_actions")
        public_sha = entry.get("public_forbidden_sha256")
        parsed.append(
            _Trajectory(
                trajectory_id=tid,
                family=_text(entry.get("family")),
                cluster=tuple(_text(c) for c in cluster)
                if isinstance(cluster, list | tuple)
                else (),
                query_text=_text(entry.get("query_text")),
                reward=_number(entry.get("reward")),
                success=entry.get("success") is True,
                steps=tuple(steps),
                order=(
                    int(_number(entry.get("optimizer_step"))),
                    int(_number(entry.get("position"))),
                ),
                diagnostics=tuple(d for d in diagnostics if type(d) is str)
                if isinstance(diagnostics, list)
                else (),
                termination=_text(episode.get("termination")),
                repeated=tuple(
                    (row[0], row[1])
                    for row in repeated
                    if isinstance(row, list | tuple)
                    and len(row) == 2
                    and type(row[0]) is str
                    and type(row[1]) is int
                )
                if isinstance(repeated, list)
                else (),
                public_sha=frozenset(s for s in public_sha if type(s) is str)
                if isinstance(public_sha, list)
                else frozenset(),
            )
        )
    return parsed


class _Redactor:
    def __init__(self, patterns: tuple[re.Pattern[str], ...]) -> None:
        self.patterns = patterns
        self.count = 0

    def __call__(self, text: str) -> str:
        redacted, hits = redact_forbidden(text, self.patterns, marker=REDACTION_MARKER)
        self.count += hits
        return redacted


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " [...]"


def _z_text(z: Sequence[str] | None) -> str:
    if not z:
        return "(none)"
    if len(z) == len(_Z_LABELS):
        return ", ".join(f"{label}={value}" for label, value in zip(_Z_LABELS, z, strict=True))
    return ", ".join(str(value) for value in z)


def _in_cluster(z: Sequence[str], context: Sequence[str] | None) -> bool:
    return bool(context) and tuple(z[: len(context or ())]) == tuple(context or ())


def _cluster_match(cluster: Sequence[str], context: Sequence[str] | None) -> bool:
    if not cluster or not context:
        return False
    n = min(len(cluster), len(context))
    return tuple(cluster[:n]) == tuple(context[:n]) or cluster[-1] == context[0]


def _verifier_index(evidence: PhaseEvidence) -> dict[tuple[str, int], list[VerifierObs]]:
    index: dict[tuple[str, int], list[VerifierObs]] = {}
    for record in evidence.verifier:
        index.setdefault((record.trajectory_id, record.step_index), []).append(record)
    return index


def _ranked_relevant(
    candidate: CandidateEdit,
    evidence: PhaseEvidence,
    pool: Sequence[_Trajectory],
    targets: tuple[str, ...],
    families: tuple[str, ...],
) -> list[tuple[int, _Trajectory]]:
    named = candidate.evidence.get("trajectory_ids")
    named_ids = set(cast(list[str], named)) if isinstance(named, list) else set()
    context = candidate.context
    weak = {
        record.trajectory_id
        for record in evidence.verifier
        if (
            targets and record.skill_id in targets and context and tuple(record.z) == tuple(context)
        )
        or (not targets and _in_cluster(record.z, context))
    }

    def tier(trajectory: _Trajectory) -> int | None:
        if trajectory.trajectory_id in named_ids:
            return 0
        if trajectory.trajectory_id in weak:
            return 1
        if targets and any(step.skill_id in targets for step in trajectory.steps):
            return 2
        if _cluster_match(trajectory.cluster, context):
            return 3
        if trajectory.family in families or (
            trajectory.cluster and trajectory.cluster[-1] in families
        ):
            return 4
        return None

    return sorted(
        ((rank, trajectory) for trajectory in pool if (rank := tier(trajectory)) is not None),
        key=lambda item: (item[0], item[1].trajectory_id),
    )


def _step_verifier_text(step: _Step, records: Sequence[VerifierObs]) -> str:
    if records:
        return " | verifier: " + "; ".join(
            f"y={record.y:.2f} (context: {_z_text(record.z)})" for record in records
        )
    if step.verifier is not None and isinstance(step.verifier.get("y"), int | float):
        text = f" | verifier: y={float(step.verifier['y']):.2f}"
        confidence = step.verifier.get("confidence")
        if isinstance(confidence, int | float):
            text += f" (confidence {float(confidence):.2f})"
        return text
    return ""


def _verdict_counts(
    trajectory: _Trajectory, verifier: Mapping[tuple[str, int], list[VerifierObs]]
) -> tuple[int, int]:
    outcomes: list[float] = []
    for step in trajectory.steps:
        records = verifier.get((trajectory.trajectory_id, step.step_index), [])
        if records:
            outcomes.extend(record.y for record in records)
        elif step.verifier is not None and isinstance(step.verifier.get("y"), int | float):
            outcomes.append(float(step.verifier["y"]))
    return sum(1 for y in outcomes if y >= 0.5), sum(1 for y in outcomes if y < 0.5)


@dataclass(frozen=True, slots=True)
class _Group:
    key: str
    failures: tuple[_Trajectory, ...]
    success: _Trajectory | None

    @property
    def members(self) -> tuple[_Trajectory, ...]:
        return self.failures + ((self.success,) if self.success is not None else ())


def _query_key(trajectory: _Trajectory) -> str:
    return " ".join(trajectory.query_text.split()) or trajectory.trajectory_id


def _recency(trajectory: _Trajectory) -> tuple[int, int, str]:
    return (-trajectory.order[0], -trajectory.order[1], trajectory.trajectory_id)


def _distinct_query_groups(ranked: Sequence[tuple[int, _Trajectory]]) -> list[_Group]:
    rows_by_key: dict[str, list[tuple[int, _Trajectory]]] = {}
    for rank, trajectory in ranked:
        rows_by_key.setdefault(_query_key(trajectory), []).append((rank, trajectory))
    failing: list[tuple[tuple[int, tuple[int, int, str]], str, tuple[_Trajectory, ...]]] = []
    wins: dict[str, tuple[int, _Trajectory]] = {}
    for key, rows in rows_by_key.items():
        fails = sorted((t for _, t in rows if not t.success), key=_recency)
        if fails:
            best = min(rank for rank, t in rows if not t.success)
            kept = tuple(fails[:SOLVER_ROLLOUTS_PER_FAILING_QUERY])
            failing.append(((best, _recency(fails[0])), key, kept))
        won = [(rank, t) for rank, t in rows if t.success]
        if won:
            wins[key] = (min(rank for rank, _ in won), min((t for _, t in won), key=_recency))
    failing.sort(key=lambda item: (item[0], item[1]))
    failing = failing[:SOLVER_FAILING_QUERIES]
    contrast = [key for _, key, _ in failing if key in wins][:SOLVER_SUCCESSES]
    chosen = {key for _, key, _ in failing}
    others = sorted(
        (key for key in wins if key not in chosen),
        key=lambda key: (wins[key][0], _recency(wins[key][1]), key),
    )
    groups = [
        _Group(key, fails, wins[key][1] if key in contrast else None) for _, key, fails in failing
    ]
    extra = others[: SOLVER_SUCCESSES - len(contrast)]
    groups.extend(_Group(key, (), wins[key][1]) for key in extra)
    return groups


def _priority(groups: Sequence[_Group]) -> list[_Trajectory]:
    first = [group.failures[0] for group in groups if group.failures]
    contrast: list[_Trajectory] = []
    other: list[_Trajectory] = []
    for group in groups:
        if group.success is not None:
            (contrast if group.failures else other).append(group.success)
    later = [trajectory for group in groups for trajectory in group.failures[1:]]
    return first + contrast + other + later


def _public_patterns(
    needles: Mapping[str, re.Pattern[str]], digests: Mapping[str, str], trajectory: _Trajectory
) -> tuple[re.Pattern[str], ...]:
    public = "\n".join(
        [
            trajectory.query_text,
            *(step.observation for step in trajectory.steps if step.function != SKILL_FUNCTION),
        ]
    )
    exempt = public_needles(needles, public)
    return tuple(
        needles[needle]
        for needle in sorted(needles)
        if needle not in exempt and digests[needle] not in trajectory.public_sha
    )


def _trajectory_steps(trajectory: _Trajectory, targets: tuple[str, ...]) -> tuple[list[_Step], int]:
    steps = list(trajectory.steps)
    tail = TAIL_STEPS * (2 if trajectory.success else 1)
    if len(steps) <= tail:
        return steps, 0
    earlier = [
        step for step in steps[:-tail] if step.skill_id is not None and step.skill_id in targets
    ][:EARLIER_TARGET_STEPS]
    return earlier + steps[-tail:], len(steps) - len(earlier) - tail


def _shown(text: str, redact: _Redactor, limit: int = FIELD_CHARS) -> str:
    return clip_middle(redact(text.strip()), limit)


def _render_query(trajectory: _Trajectory, redact: _Redactor) -> str:
    if not trajectory.query_text.strip():
        return "Task (public text):\n(not provided)"
    return "Task (public text):\n" + _shown(trajectory.query_text, redact, 2 * FIELD_CHARS)


def _render_rollout(
    trajectory: _Trajectory,
    verifier: Mapping[tuple[str, int], list[VerifierObs]],
    targets: tuple[str, ...],
    redact: _Redactor,
) -> str:
    accepted, rejected = _verdict_counts(trajectory, verifier)
    lines = [
        f"verifier checks: {accepted} accepted, {rejected} rejected, {len(trajectory.steps)} steps"
    ]
    if trajectory.termination:
        lines.append("Episode end: " + redact(trajectory.termination))
    if trajectory.repeated:
        lines.append(
            "Repeated actions: "
            + "; ".join(f"{redact(action)} (x{count})" for action, count in trajectory.repeated)
        )
    kept, omitted = _trajectory_steps(trajectory, targets)
    lines.append(f"Steps ({omitted} earlier steps omitted):" if omitted else "Steps:")
    for step in kept:
        lines.extend(_step_lines(trajectory, step, verifier, redact))
    return "\n".join(lines)


def _step_lines(
    trajectory: _Trajectory,
    step: _Step,
    verifier: Mapping[tuple[str, int], list[VerifierObs]],
    redact: _Redactor,
) -> list[str]:
    submission = step.function in SUBMISSION_FUNCTIONS
    if submission and trajectory.success:
        arguments = "[withheld: final answer]"
    else:
        arguments = _shown(step.arguments, redact) if step.arguments.strip() else ""
    observation = (
        _shown(step.observation, redact) if step.observation.strip() and not submission else ""
    )
    head = f"- step {step.step_index}: {step.function}"
    if (
        step.skill_id is None
        and not step.reasoning.strip()
        and not step.executor_output.strip()
        and "\n" not in arguments + observation
        and len(arguments) + len(observation) <= COMPACT_CHARS
    ):
        line = head + (f" {arguments}" if arguments else "")
        return [line + (f" -> {observation}" if observation else "")]
    if step.skill_id is not None:
        head += f" skill={step.skill_id}" + _step_verifier_text(
            step, verifier.get((trajectory.trajectory_id, step.step_index), [])
        )
    lines = [head]
    if step.reasoning.strip():
        lines.append("  Solver reasoning (excerpt): " + _shown(step.reasoning, redact))
    if arguments:
        lines.append("  Call arguments: " + arguments)
    if step.executor_output.strip():
        lines.append("  Executor output: " + _shown(step.executor_output, redact))
    if observation:
        lines.append("  Observation: " + observation)
    return lines


def _render_material(
    groups: Sequence[_Group],
    evidence: PhaseEvidence,
    targets: tuple[str, ...],
    redact: _Redactor,
    budget: int,
    needles: Mapping[str, re.Pattern[str]],
) -> tuple[list[str], list[str]]:
    verifier = _verifier_index(evidence)
    digests = {needle: needle_sha256(needle) for needle in needles}
    group_of = {member.trajectory_id: group for group in groups for member in group.members}
    headers: dict[str, str] = {}
    blocks: dict[str, str] = {}
    used = 0
    for trajectory in _priority(groups):
        if len(blocks) >= MAX_TRAJECTORIES:
            break
        key = group_of[trajectory.trajectory_id].key
        local = _Redactor(_public_patterns(needles, digests, trajectory))
        header = headers.get(key) or _render_query(trajectory, local)
        block = _render_rollout(trajectory, verifier, targets, local)
        cost = len(block) + 16 + (0 if key in headers else len(header) + 16)
        if used + cost > budget:
            if blocks:
                continue
            block = block[: max(budget - len(header), 0)].rstrip() + "\n[... rollout truncated]"
            cost = budget
        headers.setdefault(key, header)
        blocks[trajectory.trajectory_id] = block
        redact.count += local.count
        used += cost
    rendered: list[str] = []
    chosen: list[str] = []
    for group in groups:
        shown = [member for member in group.members if member.trajectory_id in blocks]
        if not shown:
            continue
        parts = [f"## Task Q{len(rendered) + 1}\n{headers[group.key]}"]
        for member in shown:
            chosen.append(member.trajectory_id)
            parts.append(f"### T{len(chosen)}: {blocks[member.trajectory_id]}")
        rendered.append("\n\n".join(parts))
    return rendered, chosen


def _verifier_summary(
    evidence: PhaseEvidence, skill_ids: tuple[str, ...], context: Sequence[str] | None
) -> str:
    cells: dict[tuple[str, tuple[str, ...]], list[VerifierObs]] = {}
    for record in evidence.verifier:
        if skill_ids and record.skill_id not in skill_ids:
            continue
        if not skill_ids and not _in_cluster(record.z, context):
            continue
        cells.setdefault((record.skill_id, tuple(record.z)), []).append(record)
    if not cells:
        return "(no verifier records)"
    lines = []
    for (skill_id, z), records in sorted(cells.items()):
        weight = sum(record.confidence for record in records)
        mean = (
            sum(record.confidence * record.y for record in records) / weight
            if weight > 0
            else float("nan")
        )
        mean_text = f"{mean:.2f}" if weight > 0 else "n/a"
        lines.append(
            f"- {skill_id} | {_z_text(z)} | n={len(records)} | confidence-weighted mean y="
            f"{mean_text}"
        )
    return "\n".join(lines)


_PREAMBLE: Final = f"""You write SKILL.md files for the skill library of an AI agent \
(the solver) that works on tasks step by step.

# How a skill is used
- The full SKILL.md body is shown to the solver model inside its prompt (in its list of \
available skills) before it starts a task of the skill's families. The solver reads the \
procedure and applies it in its own reasoning; this is how a skill changes the solver's behaviour \
and its answers.
- The solver is a mid-sized model. It follows CONCRETE checklists, exact command forms and \
explicit lists of what to include far better than abstract principles. A skill such as "be \
careful and thorough" does nothing; "before you commit, list every requirement the task states \
and check your answer against each one" changes behaviour.

# How the final answer is written
- The solver does not write the final answer itself. When it decides to answer, a separate \
frozen answer-writer model, which never sees the skill, writes the final answer from the task, \
the solver's completed actions with their results, and the solver's last reasoning draft. A \
procedure reaches the answer only through that draft: end the procedure with the solver stating \
its final answer explicitly and completely, in the exact form the task requires, in its last \
reasoning draft.

# What to write
1. Fix what the verifiers flagged. Read the verifier outcomes first: they show which \
invocations the independent checks accepted or rejected, and in which context. Each step of \
your procedure must address a failure pattern that recurs across the rejected invocations, by \
telling the solver exactly what to ADD or DO. Never advise behaviour that the evidence shows \
the checks reject (for example, do not ask for more searching when invocations were rejected \
for their format).
2. Be concrete and executable, in the task's own vocabulary: exact command templates with \
<placeholders> (for example "<verb> <object> <argument>"), the parts a complete answer must \
contain, the exact final-answer shape the task requires. Short worked mini-examples are \
encouraged if they are fictional and marked as fictional.
3. Stay general across tasks of this kind. The environment's general vocabulary (command verbs, \
kinds of objects, places and tools, the parts of a typical answer) is concrete and welcome; \
but never hard-code answers, entities, names, numbers, formulas, code or special cases of the \
training tasks below. A rule must also help on an unseen task of the same kind.
4. Use the evidence: compare the flagged and the accepted rollouts of the same task; the solver's \
reasoning shows where it went wrong; repeated actions and the episode end show loops. \
Generalise only from patterns that recur across tasks, never from a single case.
5. Content, in this order: when the procedure applies; a short numbered procedure of concrete \
actions; a final checklist the solver runs before it commits (each item a concrete thing to \
verify); a reminder that the final answer keeps exactly the output format the task requires.
6. Wording rules, checked automatically: address the solver as "you"; never use the words \
Supervisor, caller or orchestrator; do not name the agent's functions by their identifiers \
(describe the action instead, e.g. search, open a passage, issue an environment command); do \
not name benchmarks, datasets, task ids or URLs; no answer wrappers and no fixed answer ranges.
7. Body layout, required by the parser:
INPUT: <the task situation the procedure applies to>
OUTPUT: <what applying the procedure yields>

Procedure:
1. <concrete action>
2. <concrete action>

Final checklist:
- <concrete thing to verify before you commit>
8. Size: keep the body under 350 words; the whole file must stay under {DEFAULT_TOKEN_CAP} tokens \
(punctuation and <placeholders> count too).
9. The frontmatter has exactly two keys: name (a short lowercase slug of letters, digits and \
hyphens) and description (one line of at most {DESCRIPTION_MAX_CHARS} characters saying what the \
procedure does and when it applies).
"""


def author_preamble() -> str:
    return _PREAMBLE


def author_prompt_version() -> str:
    return AUTHOR_PROMPT_VERSION


def _families_text(families: Sequence[str]) -> str:
    return ", ".join(families) if families else "(none)"


def _existing_skill_text(spec: SkillSpec) -> str:
    body = spec.body if not spec.is_empty_slot else "(empty: no procedure yet)\n"
    return (
        f"## Skill {spec.skill_id} (families: {_families_text(spec.families)})\n"
        f"description: {spec.description or '(none)'}\n"
        f"body:\n<<<\n{body.rstrip()}\n>>>"
    )


@dataclass(frozen=True, slots=True)
class _Plan:
    template: str
    task: str
    targets: tuple[SkillSpec, ...]
    draft_families: tuple[tuple[str, ...], ...]
    removed: tuple[str, ...]
    parent_id: str | None
    related: tuple[SkillSpec, ...] = ()


class _PlanError(ValueError):
    pass


def _transferable(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({value for value in values if value in TRANSFERABLE_FAMILIES}))


def _generate_families(candidate: CandidateEdit) -> tuple[str, ...]:
    raw = candidate.evidence.get("families")
    if isinstance(raw, list) and raw and all(type(item) is str for item in raw):
        families = _transferable(cast(list[str], raw))
        if families:
            return families
    single = candidate.evidence.get("family")
    if type(single) is str and single in TRANSFERABLE_FAMILIES:
        return (single,)
    context_class = candidate.evidence.get("context_class")
    if type(context_class) is str and context_class in TRANSFERABLE_FAMILIES:
        return (context_class,)
    for item in candidate.context or ():
        if item in TRANSFERABLE_FAMILIES:
            return (item,)
    raise _PlanError("generate candidate names no transferable family")


def _plan(candidate: CandidateEdit, library: LibraryVersion) -> _Plan:
    kind = candidate.kind
    try:
        targets = tuple(library.skill(skill_id) for skill_id in candidate.skill_ids)
    except KeyError as error:
        raise _PlanError(f"target skill {error.args[0]} is not in the library") from None
    z = _z_text(candidate.context)
    if kind is EditKind.REFINE:
        if len(targets) != 1:
            raise _PlanError("refine needs exactly one target skill")
        (target,) = targets
        if target.is_empty_slot and candidate.evidence.get("rule") == BOOTSTRAP_RULE:
            task = (
                f"Skill slot {target.skill_id} (families: {_families_text(target.families)}) "
                "has no instructions yet and is therefore not offered to the Supervisor. Write "
                "the slot's first procedure: a reusable procedure that the Supervisor can call "
                "on an input text it writes itself (the executor sees only that input) so that "
                "such invocations pass the independent checks. Infer from the material where "
                "invocations were rejected or went wrong. Reply with one SKILL.md file."
            )
            template = BOOTSTRAP_TEMPLATE
        elif target.is_empty_slot:
            task = (
                f"Skill slot {target.skill_id} (families: {_families_text(target.families)}) "
                "has no instructions yet: when the Supervisor calls it, the executor sees only "
                "the Supervisor's input. Write its first procedure. Infer from the material "
                "what the Supervisor tried to use this skill for and where the invocations or "
                "the tasks failed, and write a general procedure that makes such invocations "
                "reliable. Reply with one SKILL.md file."
            )
            template = "refine-empty-slot@1"
        else:
            task = (
                f"Revise skill {target.skill_id} (families: {_families_text(target.families)}). "
                "Its verifier reliability is adequate overall but confirmed low in the context "
                f"[{z}]. Keep what works and change the procedure so that it also handles that "
                "context; the revised skill replaces the current one. Reply with one SKILL.md "
                "file."
            )
            template = "refine@1"
        return _Plan(
            template, task, targets, (target.families,), (target.skill_id,), target.skill_id
        )
    if kind is EditKind.SPLIT:
        if len(targets) != 1 or targets[0].is_empty_slot:
            raise _PlanError("split needs exactly one non-empty target skill")
        (target,) = targets
        head = candidate.context[0] if candidate.context else ""
        if len(target.families) >= 2 and head in target.families:
            first: tuple[str, ...] = (head,)
            second = tuple(f for f in target.families if f != head)
        else:
            first = second = target.families
        task = (
            f"Skill {target.skill_id} behaves very differently across contexts: its verifier "
            f"reliability in the context [{z}] differs strongly from its other contexts. "
            "Replace it with two context-specialised skills: skill 1 for invocations in that "
            f"context (families: {_families_text(first)}) and skill 2 for the remaining "
            f"contexts (families: {_families_text(second)}). Each description must say when "
            "to call it rather than the other. Reply with two SKILL.md files, skill 1 first, "
            f'separated by a line containing only "{SKILL_SEPARATOR}".'
        )
        return _Plan("split@1", task, targets, (first, second), (target.skill_id,), target.skill_id)
    if kind is EditKind.COMPRESS:
        if len(targets) != 2 or targets[0].skill_id == targets[1].skill_id:
            raise _PlanError("compress needs a pair of distinct skills")
        a, b = targets
        families = tuple(sorted(set(a.families) | set(b.families)))
        task = (
            f"Skills {a.skill_id} and {b.skill_id} are redundant: they are called in "
            "near-equivalent contexts with near-equivalent outputs and verifier outcomes. "
            "Merge them into one skill that covers all their uses without losing coverage; "
            "the merged skill replaces both. Reply with one SKILL.md file."
        )
        return _Plan("compress@1", task, targets, (families,), (a.skill_id, b.skill_id), a.skill_id)
    if kind is EditKind.GENERATE:
        if targets:
            raise _PlanError("generate takes no target skill")
        families = _generate_families(candidate)
        related = tuple(spec for spec in library.skills if set(spec.families) & set(families))
        task = (
            f"Steps in the context cluster [{z}] (families: {_families_text(families)}) are "
            "often rejected by the independent verifiers and no existing skill is reliable "
            "there. Write one new skill that the "
            "Supervisor can call in that situation to make those steps verifiably better. It "
            "must not duplicate the existing skills listed below. Reply with one SKILL.md file."
        )
        return _Plan("generate@1", task, (), (families,), (), None, related)
    raise _PlanError(f"{kind.value} is not authored")


_GENERAL_CHECK: Final = "turn each cause into a general check"
_CONCRETE_FIX: Final = "turn each recurring cause of rejected invocations into a concrete instruction or checklist item"


def _concrete_plan(plan: _Plan) -> _Plan:
    solver = _task_plan(plan)
    if solver.template in (BOOTSTRAP_TEMPLATE, "refine-empty-slot@1"):
        return replace(solver, task=solver.task.replace(_GENERAL_CHECK, _CONCRETE_FIX))
    return solver


def _task_plan(plan: _Plan) -> _Plan:
    families = _families_text(plan.targets[0].families) if plan.targets else ""
    if plan.template == BOOTSTRAP_TEMPLATE:
        task = (
            f"Skill slot {plan.targets[0].skill_id} (families: {families}) has no procedure "
            "yet, so the solver works on these tasks without one. Write the slot's first "
            "procedure: a general procedure the solver applies in its own reasoning on tasks "
            "of these families so that its steps pass the independent checks more reliably. "
            "Infer from the evidence why the checks rejected steps (the verifier outcomes, the "
            "solver's reasoning, flagged versus accepted rollouts of the same task) and turn "
            "each cause into a general check. Reply with one SKILL.md file."
        )
    elif plan.template == "refine-empty-slot@1":
        task = (
            f"Skill slot {plan.targets[0].skill_id} (families: {families}) has no procedure "
            "yet: when the solver ran it, the executor saw only the solver's input. Write its "
            "first procedure: a general procedure the solver applies in its own reasoning on "
            "tasks of these families (and that still works when run on the executor). Infer "
            "from the evidence where the invocations were rejected or went wrong and turn each cause "
            "into a general check. Reply with one SKILL.md file."
        )
    elif plan.template == "split@1":
        task = plan.task.replace("say when to call it rather", "say when it applies rather")
    elif plan.template == "generate@1":
        task = plan.task.replace(
            "Write one new skill that the Supervisor can call in that situation",
            "Write one new skill: a general procedure the solver applies in that situation "
            "(it may also run it on the executor)",
        )
    else:
        return plan
    return replace(plan, task=task)


def _evidence_text(candidate: CandidateEdit) -> str:
    shown = {
        key: value
        for key, value in sorted(candidate.evidence.items())
        if key not in _HIDDEN_EVIDENCE_KEYS
    }
    return json.dumps(
        {
            "kind": candidate.kind.value,
            "context": list(candidate.context) if candidate.context is not None else None,
            "predicates": shown,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _render_prompt(
    candidate: CandidateEdit,
    plan: _Plan,
    evidence: PhaseEvidence,
    pool: Sequence[_Trajectory],
    redact: _Redactor,
    material_char_budget: int,
    *,
    needles: Mapping[str, re.Pattern[str]],
    previous: Sequence[PreviousDraft] = (),
) -> tuple[str, list[str]]:
    target_ids = tuple(spec.skill_id for spec in plan.targets)
    families = tuple(sorted({f for draft in plan.draft_families for f in draft}))
    groups = _distinct_query_groups(
        _ranked_relevant(candidate, evidence, pool, target_ids, families)
    )
    rendered, chosen = _render_material(
        groups, evidence, target_ids, redact, material_char_budget, needles
    )
    sections = [author_preamble(), "# Your task\n" + plan.task]
    if plan.targets:
        sections.append(
            "# Current skill(s)\n"
            + "\n\n".join(redact(_existing_skill_text(spec)) for spec in plan.targets)
        )
    if plan.related:
        sections.append(
            "# Existing skills of these families\n"
            + "\n".join(
                f"- {spec.skill_id}: " + redact(spec.description or "(empty slot)")
                for spec in plan.related
            )
        )
    history = render_previous_drafts(previous, redact, current_ids=target_ids)
    if history is not None:
        sections.append(history)
    sections.append(
        "# Evidence\nDecision predicates that selected this edit (JSON):\n"
        + redact(_evidence_text(candidate))
        + "\n\nVerifier outcomes (y = 1: the independent verifier accepted the invocation):\n"
        + _verifier_summary(evidence, target_ids, candidate.context)
    )
    sections.append(
        "# Training evidence: distinct tasks of these families (public material; gold "
        "answers, reference solutions, hidden tests and rubric texts are never shown, "
        f"{REDACTED} and {REDACTION_MARKER} mark withheld text)\n"
        + (
            f"{len(rendered)} tasks, {len(chosen)} rollouts; each task's public text is "
            "shown once, followed by its rollouts.\n\n" + "\n\n".join(rendered)
            if rendered
            else "(no trajectory material)"
        )
    )
    count = len(plan.draft_families)
    shape = (
        "the SKILL.md file only"
        if count == 1
        else f'the {count} SKILL.md files, separated by a line containing only "{SKILL_SEPARATOR}"'
    )
    sections.append(
        f"# Reply format\nReply with {shape}, and nothing else:\n"
        "---\nname: <slug>\ndescription: <one line>\n---\n\n<body>"
    )
    return "\n\n".join(section.rstrip() for section in sections) + "\n", chosen


@dataclass(frozen=True, slots=True)
class _Draft:
    name: str
    description: str
    body: str


_FENCE_RE: Final = re.compile(r"^\s*```[\w-]*\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


def _unfence(text: str) -> str:
    match = _FENCE_RE.match(text)
    return match.group(1) if match else text


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def _split_drafts(text: str, expected: int) -> tuple[list[_Draft], list[str]]:
    parts = re.split(
        r"(?m)^[ \t]*" + re.escape(SKILL_SEPARATOR) + r"[ \t]*$", _unfence(text.strip())
    )
    parts = [part for part in parts if part.strip()]
    errors: list[str] = []
    if len(parts) != expected:
        errors.append(
            f"layout: expected {expected} SKILL.md file(s), found {len(parts)}"
            + (f' (separate them with a line "{SKILL_SEPARATOR}")' if expected > 1 else "")
        )
    drafts: list[_Draft] = []
    for number, part in enumerate(parts[:expected], start=1):
        lines = _unfence(part.strip()).split("\n")
        if lines[0].strip() != "---":
            errors.append(f"skill {number}: must start with a '---' frontmatter line")
            continue
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
        if end is None:
            errors.append(f"skill {number}: the frontmatter is not closed by '---'")
            continue
        try:
            front = yaml.safe_load("\n".join(lines[1:end]))
        except yaml.YAMLError:
            errors.append(f"skill {number}: the frontmatter is not valid YAML")
            continue
        if not isinstance(front, dict):
            errors.append(f"skill {number}: the frontmatter must be 'name:' and 'description:'")
            continue
        name, description = front.get("name"), front.get("description")
        if type(name) is not str or not name.strip():
            errors.append(f"skill {number}: frontmatter name is missing")
            continue
        if type(description) is not str or not description.strip():
            errors.append(f"skill {number}: frontmatter description is missing")
            continue
        body = normalise_body("\n".join(lines[end + 1 :]))
        drafts.append(_Draft(_slug(name), " ".join(description.split()), body))
    return drafts, errors


def _body_sha12(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]


def authored_skill_id(families: Sequence[str], body: str) -> str:
    if not families:
        raise ValueError("an authored skill needs a family")
    return f"skill-{sorted(families)[0]}-{_body_sha12(body)}"


def _drafts_to_specs(
    text: str,
    plan: _Plan,
    library: LibraryVersion,
    *,
    forbidden_strings: Collection[str],
    token_cap: int,
) -> tuple[tuple[SkillSpec, ...], list[str]]:
    drafts, errors = _split_drafts(text, len(plan.draft_families))
    specs: list[SkillSpec] = []
    for number, (draft, families) in enumerate(
        zip(drafts, plan.draft_families, strict=False), start=1
    ):
        family_tuple = tuple(sorted(set(families)))
        spec = SkillSpec(
            skill_id=authored_skill_id(family_tuple, draft.body),
            name=draft.name,
            description=draft.description,
            body=draft.body,
            families=family_tuple,
            version=library.version + 1,
            parent_id=plan.parent_id,
        )
        problems = validate_skill_md(spec, forbidden_strings=forbidden_strings, token_cap=token_cap)
        if spec.skill_id in library.skill_ids:
            problems.append("duplicate: the body is identical to an existing library skill")
        errors.extend(f"skill {number}: {problem}" for problem in problems)
        specs.append(spec)
    if len(specs) == 2 and specs[0].body == specs[1].body:
        errors.append("split: the two skills must differ")
    return tuple(specs), errors


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _call_record(prompt: str, response: AuthorResponse) -> dict[str, JsonValue]:
    return {
        "prompt_sha256": _sha256(prompt),
        "response_id": response.response_id,
        "model": response.model,
        "usage": response.usage,
        "status": response.status,
        "reasoning_sent": response.reasoning_sent,
    }


def _usage_total(calls: Sequence[Mapping[str, JsonValue]]) -> dict[str, JsonValue]:
    total: dict[str, JsonValue] = {}
    for call in calls:
        usage = call.get("usage")
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if type(value) is int:
                previous = total.get(key, 0)
                total[key] = (previous if type(previous) is int else 0) + value
    return total


class _AuthoringFailedError(Exception):
    def __init__(self, reason: str, record: dict[str, JsonValue]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.record = record


def _repair_prompt(prompt: str, reply: str, errors: Sequence[str], redact: _Redactor) -> str:
    echoed = redact(_clip(reply, REPLY_ECHO_CHARS)) if reply.strip() else "(empty reply)"
    return (
        prompt
        + "\n# Your previous reply\n<<<\n"
        + echoed
        + "\n>>>\n\n# It failed these checks\n"
        + "\n".join(f"- {error}" for error in errors)
        + f"\n\nReturn the corrected file(s) only, in the reply format above. {REDACTION_MARKER} "
        "in your previous reply marks text that is not allowed in a skill.\n"
    )


def _author_one(
    order: int,
    candidate: CandidateEdit,
    plan: _Plan,
    evidence: PhaseEvidence,
    library: LibraryVersion,
    client: AuthorClient,
    pool: Sequence[_Trajectory],
    *,
    forbidden_strings: Collection[str],
    max_output_tokens: int,
    token_cap: int,
    material_char_budget: int,
    previous_drafts: Sequence[PreviousDraft] = (),
) -> AuthoredEdit:
    redact = _Redactor(forbidden_patterns(forbidden_strings))
    families = {family for draft in plan.draft_families for family in draft}
    shown = drafts_for_families(previous_drafts, families)
    prompt, trajectory_ids = _render_prompt(
        candidate,
        plan,
        evidence,
        pool,
        redact,
        material_char_budget,
        needles=forbidden_needle_patterns(forbidden_strings),
        previous=shown,
    )
    material_redactions = redact.count
    record: dict[str, JsonValue] = {
        "format": AUTHOR_RECORD_FORMAT,
        "prompt_version": author_prompt_version(),
        "template": plan.template,
        "candidate_order": order,
        "prompt_sha256": _sha256(prompt),
        "material": {
            "rule": MATERIAL_RULE,
            "selection": SELECTION_RULE,
            "trajectory_ids": list(trajectory_ids),
            "redactions": material_redactions,
            "char_budget": material_char_budget,
        },
        "validation_rule": SKILL_MD_VALIDATION_VERSION,
        "token_rule": TOKEN_ESTIMATE_RULE,
        "token_cap": token_cap,
        "parents": [spec.skill_id for spec in plan.targets],
        "memory": memory_record(shown),
    }
    calls: list[dict[str, JsonValue]] = []
    attempts: list[JsonValue] = []

    def finish(response: AuthorResponse | None, repair_used: bool) -> None:
        record["calls"] = cast(list[JsonValue], calls)
        record["validation"] = attempts
        record["repair_used"] = repair_used
        record["usage"] = _usage_total(calls)
        record["response_id"] = response.response_id if response else None
        record["model"] = response.model if response else client.model

    try:
        response = client.complete(prompt, max_output_tokens=max_output_tokens)
    except AuthorClientError as error:
        finish(None, False)
        raise _AuthoringFailedError(f"author call failed: {error}", record) from None
    calls.append(_call_record(prompt, response))
    specs, errors = _drafts_to_specs(
        response.text,
        plan,
        library,
        forbidden_strings=forbidden_strings,
        token_cap=token_cap,
    )
    attempts.append({"attempt": 1, "errors": list(errors)})
    repair_used = False
    if errors:
        repair_used = True
        repair = _repair_prompt(prompt, response.text, errors, redact)
        try:
            response = client.complete(repair, max_output_tokens=max_output_tokens)
        except AuthorClientError as error:
            finish(None, True)
            raise _AuthoringFailedError(f"author repair call failed: {error}", record) from None
        calls.append(_call_record(repair, response))
        specs, errors = _drafts_to_specs(
            response.text,
            plan,
            library,
            forbidden_strings=forbidden_strings,
            token_cap=token_cap,
        )
        attempts.append({"attempt": 2, "errors": list(errors)})
    finish(response, repair_used)
    if errors:
        raise _AuthoringFailedError("draft failed validation after one repair", record)
    return AuthoredEdit(
        candidate=candidate,
        added=specs,
        removed=plan.removed,
        author_model=response.model,
        author_record=record,
    )


def author_edits(
    candidates: Sequence[CandidateEdit],
    evidence: PhaseEvidence,
    library: LibraryVersion,
    client: AuthorClient,
    *,
    max_edits: int,
    forbidden_strings: Collection[str],
    material: Sequence[Mapping[str, Any]] = (),
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    token_cap: int = DEFAULT_TOKEN_CAP,
    max_failures: int = 2,
    failures: list[dict[str, JsonValue]] | None = None,
    material_char_budget: int = MATERIAL_CHAR_BUDGET,
    previous_drafts: Sequence[PreviousDraft] = (),
) -> list[AuthoredEdit]:
    if type(max_edits) is not int or max_edits < 0:
        raise ValueError("max_edits must be a non-negative integer")
    if library.version != evidence.library.version:
        raise ValueError("evidence was collected on another library version")
    pool = _parse_material(material, evidence)
    authored: list[AuthoredEdit] = []
    failed = 0
    for order, candidate in enumerate(candidates):
        if len(authored) >= max_edits or failed >= max_failures:
            break
        if candidate.kind not in STRUCTURAL_EDITS:
            continue
        if candidate.evidence.get("verifier_eligible") is not True:
            _report(failures, order, candidate, "skipped: no verifier-only eligibility (C.1 (i))")
            continue
        if candidate.kind is EditKind.PRUNE:
            if len(candidate.skill_ids) != 1 or candidate.skill_ids[0] not in library.skill_ids:
                _report(failures, order, candidate, "skipped: prune target is not in the library")
                continue
            authored.append(
                AuthoredEdit(
                    candidate=candidate,
                    added=(),
                    removed=candidate.skill_ids,
                    author_model="none",
                    author_record={
                        "format": AUTHOR_RECORD_FORMAT,
                        "template": "prune-no-author@1",
                        "candidate_order": order,
                        "repair_used": False,
                        "validation": [],
                    },
                )
            )
            continue
        try:
            plan = _concrete_plan(_plan(candidate, library))
        except _PlanError as error:
            _report(failures, order, candidate, f"skipped: {error}")
            continue
        try:
            authored.append(
                _author_one(
                    order,
                    candidate,
                    plan,
                    evidence,
                    library,
                    client,
                    pool,
                    forbidden_strings=forbidden_strings,
                    max_output_tokens=max_output_tokens,
                    token_cap=token_cap,
                    material_char_budget=material_char_budget,
                    previous_drafts=previous_drafts,
                )
            )
        except _AuthoringFailedError as failure:
            failed += 1
            _report(failures, order, candidate, failure.reason, failure.record)
    return authored


def _report(
    sink: list[dict[str, JsonValue]] | None,
    order: int,
    candidate: CandidateEdit,
    reason: str,
    record: dict[str, JsonValue] | None = None,
) -> None:
    if sink is not None:
        sink.append(
            {
                "candidate_order": order,
                "kind": candidate.kind.value,
                "skills": list(candidate.skill_ids),
                "reason": reason,
                "record": record,
            }
        )


__all__ = [
    "AUTHOR_PROMPT_VERSION",
    "AUTHOR_RECORD_FORMAT",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "MATERIAL_CHAR_BUDGET",
    "MATERIAL_RULE",
    "SELECTION_RULE",
    "SKILL_SEPARATOR",
    "SUBMISSION_FUNCTIONS",
    "USER_AGENT",
    "AuthorClient",
    "AuthorClientError",
    "AuthorResponse",
    "FakeAuthorClient",
    "GatewayAuthorClient",
    "author_edits",
    "author_preamble",
    "author_prompt_version",
    "authored_skill_id",
]
