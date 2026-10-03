from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import cast

from skillev.contracts import (
    JsonValue,
    ScientificSamplingCoordinate,
    StepR2FlowRecord,
    TerminalReward,
    TrajectoryRecord,
    TrajectoryStep,
    build_trajectory_record,
    normalize_json,
    validate_sha256,
)
from skillev.contracts.action_wire import EVENT_GRAMMAR_BOUNDARY
from skillev.contracts.state_map import SIGMA_TRACE_QUOTIENT
from skillev.contracts.ttb_common import require_iso_timestamp
from skillev.contracts.verifier_record import VerifierRecord
from skillev.policy.interface import encode_rollout_prompt
from skillev.runtime.execution import ActionParseResult, EnvironmentObservation
from skillev.scoring import render_forward_prefix_from_parts

from .context import AssembledInitialContext
from .generator import RolloutTokenizerProtocol
from .legal_events import LegalEventSet
from .types import INITIAL_CONTEXT_PROFILE, PolicySnapshot, RolloutRequest, RolloutTermination


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be non-empty text")
    return value


def _exact_object(
    value: object,
    *,
    label: str,
    expected: set[str],
) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or normalized != value:
        raise ValueError(f"{label} must be a normalized JSON object")
    if set(normalized) != expected:
        raise ValueError(f"{label} has an incompatible field set")
    return normalized


def _wire_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be an integer")
    return value


@dataclass(frozen=True, slots=True)
class ReasoningDraft:
    step_index: int
    prompt_text: str
    prompt_hash: str
    text: str
    generated_token_ids: tuple[int, ...]
    policy_snapshot_id: str

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index < 1:
            raise ValueError("reasoning draft step_index must be positive")
        _text(self.prompt_text, field="reasoning prompt_text")
        validate_sha256(self.prompt_hash)
        if not isinstance(self.text, str):
            raise ValueError("reasoning text must be text")
        if not isinstance(self.generated_token_ids, tuple) or any(
            type(item) is not int or item < 0 for item in self.generated_token_ids
        ):
            raise ValueError("reasoning generated_token_ids must be a token tuple")
        _text(self.policy_snapshot_id, field="reasoning policy_snapshot_id")


@dataclass(frozen=True, slots=True)
class ActionDraft:
    step_index: int
    forward_prefix_text: str
    forward_prefix_hash: str
    text: str
    token_ids: tuple[int, ...]
    parse_result: ActionParseResult
    policy_snapshot_id: str

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index < 1:
            raise ValueError("action draft step_index must be positive")
        _text(self.forward_prefix_text, field="action forward_prefix_text")
        validate_sha256(self.forward_prefix_hash)
        _text(self.text, field="action text")
        if not isinstance(self.token_ids, tuple) or not self.token_ids:
            raise ValueError("action token_ids must be a non-empty tuple")
        if any(type(item) is not int or item < 0 for item in self.token_ids):
            raise ValueError("action token_ids must contain non-negative integers")
        if not isinstance(self.parse_result, ActionParseResult):
            raise ValueError("parse_result must be an ActionParseResult")
        _text(self.policy_snapshot_id, field="action policy_snapshot_id")


@dataclass(frozen=True, slots=True)
class CompletedStepDraft:
    reasoning: ReasoningDraft
    action: ActionDraft
    observation: EnvironmentObservation

    def __post_init__(self) -> None:
        if self.reasoning.step_index != self.action.step_index:
            raise ValueError("reasoning and action drafts must have the same step index")
        if self.reasoning.policy_snapshot_id != self.action.policy_snapshot_id:
            raise ValueError("reasoning and action drafts must use the same policy snapshot")
        if not isinstance(self.observation, EnvironmentObservation):
            raise ValueError("observation must be an EnvironmentObservation")
        _text(self.observation.observation_text, field="observation_text")


@dataclass(frozen=True, slots=True)
class R2FlowStepInputs:
    legal_event_set_sha256: str
    action_grammar_sha256: str
    action_budget_tokens: int
    action_stop_token_ids: tuple[int, ...]
    action_finish_reason: str
    budget_forced_positions: tuple[int, ...]
    action_masks_digest: str
    reasoning_token_ids: tuple[int, ...]
    reasoning_stop_token_ids: tuple[int, ...]
    reasoning_finish_reason: str


def _sigma_step(
    *,
    initial_text: str,
    previous_steps: tuple[TrajectoryStep, ...],
    draft: CompletedStepDraft,
    inputs: R2FlowStepInputs,
    state_map: str,
) -> tuple[str, StepR2FlowRecord]:
    from skillev.policy.event_grammar import EVENT_GRAMMAR_VERSION, EVENT_IDENTITY, key_json
    from skillev.scoring.quotient import (
        StepParts,
        actual_in_edge_index,
        in_edges,
        legal_events_at,
        parse_step_event,
        sigma,
    )
    from skillev.scoring.rendering import render_in_edge_hindsight_prefix

    previous = sigma(initial_text, previous_steps)
    legal = legal_events_at(initial_text, previous)
    if legal.sha256 != inputs.legal_event_set_sha256:
        raise ValueError("the action grammar was built from another legal event set")
    event = parse_step_event(legal, draft.action.text)
    parts = StepParts(
        draft.action.text,
        draft.observation.observation_text,
        draft.observation.observation_status,
    )
    current = sigma(initial_text, (*previous_steps, parts))
    edges = in_edges(initial_text, current)
    actual = actual_in_edge_index(initial_text, previous, current, event, edges)
    hindsight = render_in_edge_hindsight_prefix(initial_text, (*previous_steps, parts))
    reasoning_stopped = bool(inputs.reasoning_stop_token_ids)
    record = StepR2FlowRecord(
        state_map=state_map,
        event_label=event.label,
        event_function=event.u,
        event_args_sha256=hashlib.sha256(key_json(dict(event.args)).encode("utf-8")).hexdigest(),
        event_identity=EVENT_IDENTITY,
        state_key=current.key(),
        predecessor_key=previous.key(),
        rank=current.rank,
        in_edges=tuple((edge.predecessor_key, edge.label_hash) for edge in edges),
        actual_in_edge=actual,
        legal_event_set_sha256=legal.sha256,
        action_grammar_version=EVENT_GRAMMAR_VERSION,
        action_grammar_sha256=inputs.action_grammar_sha256,
        action_budget_tokens=inputs.action_budget_tokens,
        action_stop_token_ids=inputs.action_stop_token_ids,
        action_finish_reason=inputs.action_finish_reason,
        budget_forced_positions=inputs.budget_forced_positions,
        action_masks_digest=inputs.action_masks_digest,
        reasoning_token_ids=inputs.reasoning_token_ids,
        reasoning_token_ids_include_stop=reasoning_stopped,
        reasoning_stop_token_ids=inputs.reasoning_stop_token_ids,
        reasoning_finish_reason=inputs.reasoning_finish_reason,
        terminal=current.terminal,
    )
    return hindsight.prefix_hash, record


def materialize_trajectory_step(
    *,
    initial_text: str,
    previous_steps: tuple[TrajectoryStep, ...],
    draft: CompletedStepDraft,
    r2flow: R2FlowStepInputs,
) -> TrajectoryStep:
    from skillev.scoring.quotient import state_map_of

    state_map = state_map_of(initial_text)
    if state_map is None:
        raise ValueError("step-r2flow inputs require a declared state map")
    expected_step_index = len(previous_steps) + 1
    if draft.reasoning.step_index != expected_step_index:
        raise ValueError("draft step index does not follow the trajectory prefix")
    forward = render_forward_prefix_from_parts(
        initial_text,
        previous_steps,
        expected_step_index,
        draft.reasoning.text,
    )
    if (
        forward.text != draft.action.forward_prefix_text
        or forward.prefix_hash != draft.action.forward_prefix_hash
    ):
        raise ValueError("action draft does not match the canonical forward prefix")
    hindsight_hash, record = _sigma_step(
        initial_text=initial_text,
        previous_steps=previous_steps,
        draft=draft,
        inputs=r2flow,
        state_map=state_map,
    )
    return TrajectoryStep(
        index=expected_step_index,
        reasoning_text=draft.reasoning.text,
        action_text=draft.action.text,
        action_token_ids=draft.action.token_ids,
        action_token_count=len(draft.action.token_ids),
        observation_text=draft.observation.observation_text,
        observation_status=draft.observation.observation_status,
        invoked_skill_ids=draft.observation.invoked_skill_ids,
        forward_prefix_hash=forward.prefix_hash,
        hindsight_prefix_hash=hindsight_hash,
        r2flow=record,
    )


@dataclass(frozen=True, slots=True)
class RolloutManifest:
    trajectory_id: str
    task_id: str
    policy_snapshot: PolicySnapshot
    library_version: str
    sampling_coordinate: ScientificSamplingCoordinate
    decoding_snapshot_id: str
    assembler_version: str
    action_format_version: str
    generator_backend_id: str
    termination: RolloutTermination
    reasoning_token_counts: tuple[int, ...]
    started_at: str
    completed_at: str
    prompt_encoder_version: str
    reasoning_finish_reasons: tuple[str, ...]
    action_finish_reasons: tuple[str, ...]
    condition_id: str
    state_map: str

    def __post_init__(self) -> None:
        if self.state_map != SIGMA_TRACE_QUOTIENT:
            raise ValueError("unsupported manifest state map")
        for field, value in (
            ("trajectory_id", self.trajectory_id),
            ("task_id", self.task_id),
            ("library_version", self.library_version),
            ("decoding_snapshot_id", self.decoding_snapshot_id),
            ("assembler_version", self.assembler_version),
            ("action_format_version", self.action_format_version),
            ("generator_backend_id", self.generator_backend_id),
            ("prompt_encoder_version", self.prompt_encoder_version),
            ("condition_id", self.condition_id),
        ):
            _text(value, field=field)
        if not isinstance(self.policy_snapshot, PolicySnapshot):
            raise ValueError("policy_snapshot must be a PolicySnapshot")
        if not isinstance(self.sampling_coordinate, ScientificSamplingCoordinate):
            raise ValueError("manifest requires a scientific sampling coordinate")
        if self.sampling_coordinate.task_id != self.task_id:
            raise ValueError("manifest sampling coordinate belongs to another task")
        if self.generator_backend_id != self.policy_snapshot.backend_id:
            raise ValueError("manifest backend does not match the policy snapshot")
        if not isinstance(self.termination, RolloutTermination):
            raise ValueError("termination must be a RolloutTermination")
        if not isinstance(self.reasoning_token_counts, tuple) or not self.reasoning_token_counts:
            raise ValueError("reasoning_token_counts must be a non-empty tuple")
        if any(type(count) is not int or count < 0 for count in self.reasoning_token_counts):
            raise ValueError("reasoning token counts must be non-negative integers")
        for label, reasons in (
            ("reasoning_finish_reasons", self.reasoning_finish_reasons),
            ("action_finish_reasons", self.action_finish_reasons),
        ):
            if not isinstance(reasons, tuple) or any(
                not isinstance(reason, str) or not reason for reason in reasons
            ):
                raise ValueError(f"{label} must be a tuple of non-empty text")
            if reasons and len(reasons) != len(self.reasoning_token_counts):
                raise ValueError(f"{label} must align with the rollout horizon")
        require_iso_timestamp(self.started_at, field="started_at", location="rollout manifest")
        require_iso_timestamp(self.completed_at, field="completed_at", location="rollout manifest")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "action_format_version": self.action_format_version,
            "action_boundary_version": EVENT_GRAMMAR_BOUNDARY,
            "action_finish_reasons": list(self.action_finish_reasons),
            "assembler_version": self.assembler_version,
            "completed_at": self.completed_at,
            "decoding_snapshot_id": self.decoding_snapshot_id,
            "generator_backend_id": self.generator_backend_id,
            "library_version": self.library_version,
            "policy_snapshot": self.policy_snapshot.to_value(),
            "prompt_encoder_version": self.prompt_encoder_version,
            "reasoning_finish_reasons": list(self.reasoning_finish_reasons),
            "reasoning_token_counts": list(self.reasoning_token_counts),
            "sampling_coordinate": self.sampling_coordinate.to_value(),
            "started_at": self.started_at,
            "task_id": self.task_id,
            "termination": self.termination.value,
            "trajectory_id": self.trajectory_id,
            "condition_id": self.condition_id,
            "initial_context_profile": INITIAL_CONTEXT_PROFILE,
            "state_map": self.state_map,
        }

    @classmethod
    def from_value(cls, value: object) -> RolloutManifest:
        if not isinstance(value, dict):
            raise ValueError("rollout manifest must be a JSON object")
        normalized = normalize_json(value)
        if not isinstance(normalized, dict) or normalized != value:
            raise ValueError("rollout manifest must be a normalized JSON object")
        if set(normalized) != {
            "action_format_version",
            "assembler_version",
            "completed_at",
            "decoding_snapshot_id",
            "generator_backend_id",
            "library_version",
            "policy_snapshot",
            "reasoning_token_counts",
            "sampling_coordinate",
            "started_at",
            "task_id",
            "termination",
            "trajectory_id",
            "action_boundary_version",
            "action_finish_reasons",
            "prompt_encoder_version",
            "reasoning_finish_reasons",
            "condition_id",
            "initial_context_profile",
            "state_map",
        }:
            raise ValueError("rollout manifest has an incompatible field set")
        if (
            normalized["action_boundary_version"] != EVENT_GRAMMAR_BOUNDARY
            or normalized["initial_context_profile"] != INITIAL_CONTEXT_PROFILE
        ):
            raise ValueError("unsupported manifest action boundary or context profile")
        counts = normalized["reasoning_token_counts"]
        if type(counts) is not list:
            raise ValueError("reasoning_token_counts must be an array")
        reasoning_counts = tuple(
            _wire_int(item, field="reasoning_token_counts item") for item in counts
        )
        reasoning_reasons_value = normalized["reasoning_finish_reasons"]
        action_reasons_value = normalized["action_finish_reasons"]
        if type(reasoning_reasons_value) is not list or type(action_reasons_value) is not list:
            raise ValueError("manifest finish reasons must be arrays")
        reasoning_reasons = tuple(
            _text(item, field="reasoning_finish_reasons item") for item in reasoning_reasons_value
        )
        action_reasons = tuple(
            _text(item, field="action_finish_reasons item") for item in action_reasons_value
        )
        termination = _text(normalized["termination"], field="termination")
        return cls(
            trajectory_id=_text(normalized["trajectory_id"], field="trajectory_id"),
            task_id=_text(normalized["task_id"], field="task_id"),
            policy_snapshot=PolicySnapshot.from_value(normalized["policy_snapshot"]),
            library_version=_text(normalized["library_version"], field="library_version"),
            sampling_coordinate=ScientificSamplingCoordinate.from_value(
                normalized["sampling_coordinate"]
            ),
            decoding_snapshot_id=_text(
                normalized["decoding_snapshot_id"],
                field="decoding_snapshot_id",
            ),
            assembler_version=_text(
                normalized["assembler_version"],
                field="assembler_version",
            ),
            action_format_version=_text(
                normalized["action_format_version"],
                field="action_format_version",
            ),
            generator_backend_id=_text(
                normalized["generator_backend_id"],
                field="generator_backend_id",
            ),
            termination=RolloutTermination(termination),
            reasoning_token_counts=reasoning_counts,
            started_at=_text(normalized["started_at"], field="started_at"),
            completed_at=_text(normalized["completed_at"], field="completed_at"),
            prompt_encoder_version=_text(
                normalized["prompt_encoder_version"], field="prompt_encoder_version"
            ),
            reasoning_finish_reasons=reasoning_reasons,
            action_finish_reasons=action_reasons,
            condition_id=_text(normalized["condition_id"], field="condition_id"),
            state_map=_text(normalized["state_map"], field="state_map"),
        )


@dataclass(frozen=True, slots=True)
class VerifierInputs:
    domain: str
    legal_event_sets: tuple[LegalEventSet, ...]
    forward_prefix_token_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        _text(self.domain, field="verifier domain")
        if len(self.legal_event_sets) != len(self.forward_prefix_token_counts):
            raise ValueError("one legal set and one prefix token count per step")
        if any(type(count) is not int or count < 0 for count in self.forward_prefix_token_counts):
            raise ValueError("prefix token counts must be non-negative integers")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "domain": self.domain,
            "forward_prefix_token_counts": list(self.forward_prefix_token_counts),
            "legal_event_sets": [legal.to_value() for legal in self.legal_event_sets],
        }

    @classmethod
    def from_value(cls, value: object) -> VerifierInputs:
        raw = _exact_object(
            value,
            label="verifier inputs",
            expected={"domain", "forward_prefix_token_counts", "legal_event_sets"},
        )
        counts = raw["forward_prefix_token_counts"]
        sets = raw["legal_event_sets"]
        if not isinstance(counts, list) or not isinstance(sets, list):
            raise ValueError("verifier inputs must hold lists")
        return cls(
            domain=_text(raw["domain"], field="verifier domain"),
            legal_event_sets=tuple(LegalEventSet.from_value(item) for item in sets),
            forward_prefix_token_counts=tuple(
                _wire_int(item, field="forward_prefix_token_count") for item in counts
            ),
        )


@dataclass(frozen=True, slots=True)
class RolloutArtifact:
    initial_context: AssembledInitialContext
    record: TrajectoryRecord
    manifest: RolloutManifest
    action_grammars: dict[str, str] | None = None
    verifier_records: tuple[VerifierRecord, ...] | None = None
    verifier_inputs: VerifierInputs | None = None

    def __post_init__(self) -> None:
        self._validate_verification()
        sigma_steps = [step for step in self.record.steps if step.r2flow is not None]
        if (self.action_grammars is not None) != bool(sigma_steps):
            raise ValueError("action grammars accompany exactly the sigma-mode steps")
        if self.action_grammars is not None:
            for digest, key in self.action_grammars.items():
                if hashlib.sha256(key.encode("utf-8")).hexdigest() != digest:
                    raise ValueError("action grammar table digest mismatch")
            for step in sigma_steps:
                assert step.r2flow is not None
                if step.r2flow.action_grammar_sha256 not in self.action_grammars:
                    raise ValueError("a sigma-mode step's action grammar key is not recorded")
        if self.manifest.trajectory_id != self.record.trajectory_id:
            raise ValueError("manifest and record trajectory IDs do not match")
        context_task_id = self.initial_context.contract.meta.get("task_id")
        if not isinstance(context_task_id, str) or self.manifest.task_id != context_task_id:
            raise ValueError("manifest and initial context task IDs do not match")
        context_library_version = self.initial_context.contract.meta.get("library_version")
        if (
            not isinstance(context_library_version, str)
            or self.manifest.library_version != context_library_version
        ):
            raise ValueError("manifest and initial context library versions do not match")
        if self.initial_context.contract != self.record.initial_context:
            raise ValueError("artifact initial-context commitments do not match")
        if self.manifest.decoding_snapshot_id != self.record.decoding_snapshot_id:
            raise ValueError("manifest and record decoding snapshots do not match")
        if self.manifest.policy_snapshot.tokenizer_id != self.record.tokenizer_id:
            raise ValueError("manifest and record tokenizer identities do not match")
        if self.manifest.assembler_version != self.record.initial_context.assembler_version:
            raise ValueError("manifest and initial context assembler versions do not match")
        if len(self.manifest.reasoning_token_counts) != self.record.horizon:
            raise ValueError("reasoning token counts do not align with record horizon")

    def _validate_verification(self) -> None:
        if (self.verifier_records is None) != (self.verifier_inputs is None):
            raise ValueError("verifier records and verifier inputs accompany each other")
        if self.verifier_records is None or self.verifier_inputs is None:
            return
        steps = self.record.steps
        horizon = len(steps)
        if len(self.verifier_inputs.legal_event_sets) != horizon:
            raise ValueError("verifier inputs do not cover the committed steps")
        for step, legal in zip(steps, self.verifier_inputs.legal_event_sets, strict=True):
            if step.r2flow is None or step.r2flow.legal_event_set_sha256 != legal.sha256:
                raise ValueError("a verifier legal set differs from the step's grammar legal set")
        if [record.step_index for record in self.verifier_records] != list(range(1, horizon + 1)):
            raise ValueError("verifier records must cover steps 1..T in order")
        for record in self.verifier_records:
            if (
                record.trajectory_id != self.record.trajectory_id
                or record.task_id != self.manifest.task_id
                or record.domain != self.verifier_inputs.domain
            ):
                raise ValueError("a verifier record belongs to another trajectory, task or domain")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            **(
                {
                    "verifier_inputs": cast(JsonValue, self.verifier_inputs.to_value()),
                    "verifier_records": cast(
                        JsonValue, [item.to_value() for item in self.verifier_records]
                    ),
                }
                if self.verifier_records is not None and self.verifier_inputs is not None
                else {}
            ),
            **(
                {"action_grammars": cast(JsonValue, dict(sorted(self.action_grammars.items())))}
                if self.action_grammars is not None
                else {}
            ),
            "initial_context": self.initial_context.to_value(),
            "manifest": self.manifest.to_value(),
            "record": self.record.to_value(),
        }

    @classmethod
    def from_value(
        cls,
        value: object,
        *,
        tokenizer: RolloutTokenizerProtocol,
    ) -> RolloutArtifact:
        normalized = _exact_object(
            value,
            label="rollout artifact",
            expected={"initial_context", "manifest", "record"}
            | (
                {"action_grammars"}
                if isinstance(value, dict) and "action_grammars" in value
                else set()
            )
            | (
                {"verifier_inputs", "verifier_records"}
                if isinstance(value, dict)
                and ("verifier_records" in value or "verifier_inputs" in value)
                else set()
            ),
        )
        raw_records = normalized.get("verifier_records")
        if raw_records is not None and not isinstance(raw_records, list):
            raise ValueError("verifier records must be a list")
        grammars = normalized.get("action_grammars")
        if grammars is not None and (
            not isinstance(grammars, dict)
            or any(not isinstance(key, str) for key in grammars.values())
        ):
            raise ValueError("action grammars must map digests to key text")
        raw_record = TrajectoryRecord.from_value(normalized["record"])
        initial_context = AssembledInitialContext.from_value(normalized["initial_context"])
        if (
            len(encode_rollout_prompt(tokenizer, initial_context.text))
            != initial_context.contract.assembled_token_count
        ):
            raise ValueError("artifact initial-context token count is invalid")
        admitted = build_trajectory_record(
            tokenizer=tokenizer,
            trajectory_id=raw_record.trajectory_id,
            environment_id=raw_record.environment_id,
            task_family=raw_record.task_family,
            initial_context=raw_record.initial_context,
            steps=raw_record.steps,
            horizon=raw_record.horizon,
            reward=raw_record.reward,
            shifted_reward=raw_record.shifted_reward,
            epsilon_min=raw_record.epsilon_min,
            tokenizer_id=raw_record.tokenizer_id,
            decoding_snapshot_id=raw_record.decoding_snapshot_id,
            created_at=raw_record.created_at,
        )
        return cls(
            initial_context=initial_context,
            record=admitted,
            manifest=RolloutManifest.from_value(normalized["manifest"]),
            action_grammars=None
            if grammars is None
            else {digest: cast(str, key) for digest, key in grammars.items()},
            verifier_records=None
            if raw_records is None
            else tuple(VerifierRecord.from_value(item) for item in raw_records),
            verifier_inputs=None
            if "verifier_inputs" not in normalized
            else VerifierInputs.from_value(normalized["verifier_inputs"]),
        )


def finalize_trajectory_record(
    *,
    request: RolloutRequest,
    assembled: AssembledInitialContext,
    steps: tuple[TrajectoryStep, ...],
    reward: TerminalReward,
    tokenizer: RolloutTokenizerProtocol,
    created_at: str,
) -> TrajectoryRecord:
    if reward.environment_id != request.task.environment_id:
        raise ValueError("terminal reward belongs to another environment")
    if assembled.contract.query != request.task.query:
        raise ValueError("assembled context belongs to another task query")
    if (
        len(encode_rollout_prompt(tokenizer, assembled.text))
        != assembled.contract.assembled_token_count
    ):
        raise ValueError("assembled context token count changed before finalization")
    return build_trajectory_record(
        tokenizer=tokenizer,
        trajectory_id=request.trajectory_id,
        environment_id=request.task.environment_id,
        task_family=request.task.task_family,
        initial_context=assembled.contract,
        steps=steps,
        horizon=len(steps),
        reward=reward,
        shifted_reward=reward.value + request.epsilon_min,
        epsilon_min=request.epsilon_min,
        tokenizer_id=tokenizer.tokenizer_id,
        decoding_snapshot_id=request.decoding.snapshot_id,
        created_at=created_at,
    )
