from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from r2flow.benchmarks.training_schedule import TRAINING_DOMAINS
from skillev.contracts import JsonValue
from skillev.evaluation.healthbench_judge_profile import healthbench_condition
from skillev.policy import QwenMultimodalBackboneConfig
from skillev.rollout import RolloutTask
from skillev.runtime import SkillLibraryState
from skillev.training.run_condition import EffectiveRunCondition

from .bayesian_training_config import (
    R2FLOW_EVOLVE_FORMAL,
    R2FLOW_FORMATS,
    R2FLOW_ONLY_FIELDS,
    BayesianFormalConfig,
)


if TYPE_CHECKING:
    from skillev.application import SKILLEVApplication


def require_clean_initial_application(
    application: SKILLEVApplication,
    *,
    preparation: Path,
    checkpoint_directory: Path,
    root: Path,
    initial_skill_profile: str = "empty-skill-slots@1",
    domains: tuple[str, ...] | None = None,
) -> None:
    import torch

    from skillev.experiments._evolution_preflight_seed import planned_seed_documents
    from skillev.policy.checkpoint import read_policy_checkpoint_state
    from skillev.training.fresh_state import (
        FreshNamespaces,
        inspect_fresh_state,
        require_fresh_state,
        save_fresh_state_report,
    )

    parameters = torch.load(
        preparation.parent / "initial_named_parameters.pt", map_location="cpu", weights_only=True
    )
    if not isinstance(parameters, dict) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in parameters.items()
    ):
        raise ValueError("fresh preparation requires its original named trainable tensors")
    report = inspect_fresh_state(
        application,
        initial_library=SkillLibraryState.from_seed_documents(
            planned_seed_documents(initial_skill_profile, domains)
        ),
        namespaces=FreshNamespaces(
            request_journals=(root / "requests.sqlite3",),
            evidence_directories=(root / "inflight", root / "evidence"),
        ),
        preparation_state=read_policy_checkpoint_state(checkpoint_directory),
        initial_parameters=parameters,
    )
    save_fresh_state_report(root / "fresh-application-start.json", report)
    require_fresh_state(report)


def resolved_input_profiles(tasks: tuple[RolloutTask, ...]) -> dict[str, str]:
    from skillev.task_semantic_guidance import TRAINING_PUBLIC_INPUT

    profiles: dict[str, str] = {}
    for task in tasks:
        context = task.public_context
        if not isinstance(context, dict):
            raise ValueError("fresh task requires declared public source semantics")
        domain, profile = (
            context.get("benchmark_id"),
            context.get("input_profile", TRAINING_PUBLIC_INPUT),
        )
        if not isinstance(domain, str) or not isinstance(profile, str):
            raise ValueError("fresh task requires a domain and public input profile")
        if domain in profiles and profiles[domain] != profile:
            raise ValueError("one domain cannot silently mix public input profiles")
        profiles[domain] = profile
    return profiles


def load_fresh_config(path: Path) -> BayesianFormalConfig:
    raw = yaml.safe_load(path.read_text())
    if isinstance(raw, dict) and raw.get("format") in R2FLOW_FORMATS:
        return _load_r2flow_fresh_config(path, raw)
    if isinstance(raw, dict) and set(R2FLOW_ONLY_FIELDS) & raw.keys():
        raise ValueError("R2 Flow controls require formal condition @7")
    names = (
        {item.name for item in fields(BayesianFormalConfig)}
        - set(R2FLOW_ONLY_FIELDS)
        - {
            "initial_skill_profile",
            "action_tokens_by_domain",
            "learning_protocol",
            "domains",
        }
    )
    if not isinstance(raw, dict) or names - raw.keys():
        raise ValueError("fresh restart must explicitly resolve every configuration control")
    config = BayesianFormalConfig.load(path)
    require_fresh_interface(config)
    return config


def _load_r2flow_fresh_config(path: Path, raw: dict[str, object]) -> BayesianFormalConfig:
    names = {item.name for item in fields(BayesianFormalConfig)}
    names -= {
        "turns_by_domain",
        "final_turn_completion",
        "skill_call_budget_by_domain",
        "gradient_group_clip",
        "skill_visibility",
        "reasoning_call_line_domains",
        "reasoning_stop_domains",
        "healthbench_judge_recovery",
        "thinking_on_domains",
        "training_source_exclusions",
        "action_decoding",
    }
    domains = raw.get("domains")
    if isinstance(domains, list) and "healthbench" not in domains:
        if "healthbench_judge" in raw:
            raise ValueError("an R2 Flow run without HealthBench declares no HealthBench judge")
        names.discard("healthbench_judge")
    if names - raw.keys():
        raise ValueError("fresh restart must explicitly resolve every configuration control")
    config = BayesianFormalConfig.load(path)
    require_fresh_interface(config)
    return config


def require_fresh_interface(config: BayesianFormalConfig) -> None:
    if config.format in R2FLOW_FORMATS:
        _require_r2flow_interface(config)
        return
    raise ValueError("fresh restart requires an R2 Flow format")


def _require_r2flow_interface(config: BayesianFormalConfig) -> None:
    from skillev.benchmarks.r2flow_tool_set import R2FLOW_TOOL_SET
    from skillev.contracts.action_wire import NATIVE_EVENT_CALL_WIRE
    from skillev.contracts.skill_exposure import SKILL_EXPOSURE
    from skillev.evaluation.external_judge_policy import HEALTHBENCH_JUDGE_PROFILE
    from skillev.policy.interface import INPUT_WINDOW_VERSION
    from skillev.task_semantic_guidance import TASK_SEMANTIC_GUIDANCE
    from skillev.training.config import POLICY_ROLLOUT_CONFIG_FORMAT

    r2flow = config.r2flow
    prompt_flags = (
        config.phase_context
        and config.reasoning_tool_catalog
        and config.token_budget_notice
        and config.public_action_semantics
        and config.task_semantic_guidance == TASK_SEMANTIC_GUIDANCE
    )
    if (
        r2flow is None
        or not prompt_flags
        or config.action_wire != NATIVE_EVENT_CALL_WIRE
        or config.input_window != INPUT_WINDOW_VERSION
        or config.skill_exposure != SKILL_EXPOSURE
        or config.tool_set != R2FLOW_TOOL_SET
        or (
            "healthbench" in config.domains
            and config.healthbench_judge != HEALTHBENCH_JUDGE_PROFILE
        )
        or config.sampling_config.to_value()["format"] != POLICY_ROLLOUT_CONFIG_FORMAT
        or r2flow.flow_recording != "r2flow-flow-record@1"
    ):
        raise ValueError("fresh R2 Flow run requires the declared v27 identity")
    if not set(config.domains) <= set(dict(config.reasoning_tokens_by_domain)):
        raise ValueError("fresh restart must explicitly declare every active reasoning budget")
    if config.format == R2FLOW_EVOLVE_FORMAL:
        declared = set(config.thinking_off_domains) | set(config.thinking_on_domains)
        undeclared = sorted(set(config.domains) - declared)
        if undeclared:
            raise ValueError(
                "fresh @8 run must explicitly declare every training domain's thinking mode "
                f"(thinking_off_domains / thinking_on_domains): {undeclared}"
            )


def resolve_effective_run_condition(
    *,
    config: BayesianFormalConfig,
    backbone: QwenMultimodalBackboneConfig,
    initial_library: SkillLibraryState,
    data_condition: dict[str, JsonValue],
    input_profiles: dict[str, str],
    scorer_contracts: dict[str, JsonValue],
    execution: dict[str, JsonValue],
    condition_id: str,
) -> EffectiveRunCondition:
    require_fresh_interface(config)
    config.require_backbone(backbone)
    if not data_condition or not isinstance(data_condition.get("ordered_selected_sources"), list):
        raise ValueError("fresh run requires its actual ordered source and split declaration")
    known_domains = {d.value for d in TRAINING_DOMAINS}
    if not set(config.domains) <= set(input_profiles) <= known_domains or not all(
        input_profiles.values()
    ):
        raise ValueError("every domain requires an explicit public input profile")
    if not set(config.domains) <= set(scorer_contracts) <= known_domains or not all(
        scorer_contracts.values()
    ):
        raise ValueError("every domain requires its actual terminal scorer contract")
    if "healthbench" in config.domains and scorer_contracts["healthbench"] != (
        healthbench_condition()
    ):
        raise ValueError("HealthBench runtime judge differs from the frozen condition")
    declared = config.expanded_value()
    for name in ("performance_profile", "planning_hours", "target_steps_per_hour"):
        declared.pop(name)
    return EffectiveRunCondition.create(
        condition_id=condition_id,
        scientific={
            "formal": declared,
            "resolved_domains": config.schedule_summary(),
            "input_profiles": {d: input_profiles[d] for d in config.domains},
            "terminal_scorers": {d: scorer_contracts[d] for d in config.domains},
            "data_condition": data_condition,
            "initial_library": initial_library.to_value(),
            "initial_model": {
                "base_model": config.model,
                "base_dtype": config.base_dtype,
                "revision": backbone.revision,
                "tokenizer_id": backbone.tokenizer_id,
                "lora_target_modules": list(backbone.lora_target_modules),
                "z_initialization": backbone.z_initialization.to_value(),
                "seed": config.seed,
            },
        },
        execution=execution,
    )
