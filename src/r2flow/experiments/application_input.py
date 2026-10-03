from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, cast

from skillev.application import TerminalComponents
from skillev.application_config import ApplicationConfig
from skillev.contracts import JsonValue, normalize_json
from skillev.evolution import PhiBudgetAuthority, SkillAuthoringAuthority
from skillev.experiments import FormalExperimentSpec, FormalMethod
from skillev.policy import (
    PrivateInitialCheckpointBinding,
    QwenBackboneConfig,
    QwenDeploymentConfig,
    QwenMultimodalBackboneConfig,
)
from skillev.runtime import (
    AttemptRunCursorState,
    BudgetLedger,
    BudgetVector,
    RuntimeSnapshotIdentity,
    SkillDocument,
)
from skillev.runtime.attempt_run_plan import ExactAttemptRunPlan
from skillev.training import PrivateCheckpointStorageBinding

RUN_APPLICATION_INPUT_FORMAT: Final = "r2flow-application-input@2"
SKILLFLOW_UPSTREAM_REVISION: Final = "74be52bb6bd9f0e9e68dacb72636b75649197983"
SKILLFLOW_PARITY_CONTRACT: Final = "skillflow-upstream-parity@1"


def _object(value: object, *, fields: set[str], label: str) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or set(normalized) != fields:
        raise ValueError(f"{label} has incompatible fields")
    return normalized


class ApplicationInputKind(StrEnum):
    EXACT_SKILLFLOW = "exact-skillflow"
    BAYESIAN_IMPROVE = "bayesian-improve"


@dataclass(frozen=True, slots=True)
class ExactSkillFlowApplicationContract:
    upstream_revision: str
    parity_contract: str
    official_configuration_projection: JsonValue
    kind: ApplicationInputKind = ApplicationInputKind.EXACT_SKILLFLOW

    def __post_init__(self) -> None:
        if self.upstream_revision != SKILLFLOW_UPSTREAM_REVISION:
            raise ValueError("exact SkillFlow input has another upstream revision")
        if self.parity_contract != SKILLFLOW_PARITY_CONTRACT:
            raise ValueError("exact SkillFlow input has another parity contract")
        projection = normalize_json(self.official_configuration_projection)
        if not isinstance(projection, dict) or not projection:
            raise ValueError("exact SkillFlow configuration projection is unavailable")
        object.__setattr__(self, "official_configuration_projection", projection)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "kind": self.kind.value,
            "official_configuration_projection": self.official_configuration_projection,
            "parity_contract": self.parity_contract,
            "upstream_revision": self.upstream_revision,
        }


@dataclass(frozen=True, slots=True)
class BayesianImproveApplicationContract:
    method: FormalMethod
    calibration_enabled: bool
    method_authority: str = "idea.tex"
    kind: ApplicationInputKind = ApplicationInputKind.BAYESIAN_IMPROVE

    def __post_init__(self) -> None:
        if self.method not in {
            FormalMethod.BAYESIAN_IMPROVE_FULL,
            FormalMethod.BAYESIAN_IMPROVE_NO_CALIBRATION,
        }:
            raise ValueError("BayesianImprove input cannot tag the SkillFlow baseline")
        expected = self.method is FormalMethod.BAYESIAN_IMPROVE_FULL
        if self.calibration_enabled is not expected:
            raise ValueError("BayesianImprove input calibration flag differs from its method")
        if self.method_authority != "idea.tex":
            raise ValueError("BayesianImprove input must retain idea.tex authority")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "calibration_enabled": self.calibration_enabled,
            "kind": self.kind.value,
            "method": self.method.value,
            "method_authority": self.method_authority,
        }


MethodApplicationContract = ExactSkillFlowApplicationContract | BayesianImproveApplicationContract


def require_method_contract(
    method: FormalMethod,
    contract: MethodApplicationContract,
) -> None:
    if method is FormalMethod.SKILLFLOW_BASELINE:
        if not isinstance(contract, ExactSkillFlowApplicationContract):
            raise TypeError("SkillFlow baseline requires the exact upstream input tag")
        return
    if not isinstance(contract, BayesianImproveApplicationContract):
        raise TypeError("BayesianImprove requires its method-specific input tag")
    if contract.method is not method:
        raise ValueError("BayesianImprove method contract differs from the attempt")


def _method_contract_from_value(value: object) -> MethodApplicationContract:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or type(normalized.get("kind")) is not str:
        raise ValueError("Protocol 10 method application contract is untagged")
    kind = ApplicationInputKind(normalized["kind"])
    if kind is ApplicationInputKind.EXACT_SKILLFLOW:
        data = _object(
            normalized,
            fields={
                "kind",
                "official_configuration_projection",
                "parity_contract",
                "upstream_revision",
            },
            label="exact SkillFlow application contract",
        )
        revision = data["upstream_revision"]
        parity = data["parity_contract"]
        if type(revision) is not str or type(parity) is not str:
            raise TypeError("exact SkillFlow application identities must be text")
        return ExactSkillFlowApplicationContract(
            upstream_revision=revision,
            parity_contract=parity,
            official_configuration_projection=data["official_configuration_projection"],
        )
    data = _object(
        normalized,
        fields={"calibration_enabled", "kind", "method", "method_authority"},
        label="BayesianImprove application contract",
    )
    method = data["method"]
    authority = data["method_authority"]
    enabled = data["calibration_enabled"]
    if type(method) is not str or type(authority) is not str or type(enabled) is not bool:
        raise TypeError("BayesianImprove application contract fields have invalid types")
    return BayesianImproveApplicationContract(
        method=FormalMethod(method),
        calibration_enabled=enabled,
        method_authority=authority,
    )


@dataclass(frozen=True, slots=True)
class ApplicationIdentity:
    method: FormalMethod
    application_config: ApplicationConfig
    run_plan: ExactAttemptRunPlan
    initial_run_cursor: AttemptRunCursorState
    snapshot_identity: RuntimeSnapshotIdentity
    phase_checkpoint_cycle_ordinals: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.method, FormalMethod):
            raise TypeError("Protocol 10 application identity requires a formal method")
        if not isinstance(self.application_config, ApplicationConfig):
            raise TypeError("Protocol 10 application identity requires application config")
        if not isinstance(self.run_plan, ExactAttemptRunPlan):
            raise TypeError("Protocol 10 application identity requires the frozen run plan")
        self.initial_run_cursor.require_plan(self.run_plan)
        if not isinstance(self.snapshot_identity, RuntimeSnapshotIdentity):
            raise TypeError("Protocol 10 application identity requires snapshot identity")
        if self.snapshot_identity.application_config_hash != self.application_config.content_hash:
            raise ValueError("Protocol 10 snapshot and application config differ")
        if self.snapshot_identity.run_plan_hash != self.run_plan.content_hash:
            raise ValueError("Protocol 10 snapshot and run plan differ")
        if any(
            type(ordinal) is not int or not 1 <= ordinal <= self.run_plan.maximum_cycles
            for ordinal in self.phase_checkpoint_cycle_ordinals
        ):
            raise ValueError("Protocol 10 phase checkpoint ordinals are invalid")

    @property
    def method_identity_hash(self) -> str:
        return self.snapshot_identity.method_identity_hash

    @property
    def sampling_schedule_hash(self) -> str:
        return self.snapshot_identity.sampling_schedule_hash

    @property
    def ordered_task_sequence_hash(self) -> str:
        return self.snapshot_identity.ordered_task_sequence_hash

    def runtime_snapshot_identity(self) -> RuntimeSnapshotIdentity:
        return self.snapshot_identity


@dataclass(frozen=True, slots=True)
class ApplicationInput:
    method: FormalMethod
    method_contract: MethodApplicationContract
    backbone: QwenDeploymentConfig
    backbone_kind: str
    application: ApplicationConfig
    checkpoint_storage: PrivateCheckpointStorageBinding
    initial_checkpoint: PrivateInitialCheckpointBinding
    seed_documents: tuple[SkillDocument, ...]
    attempt_budget: BudgetVector
    phi_per_cycle_maximum: BudgetVector
    authoring_authority: SkillAuthoringAuthority
    snapshot_identity: RuntimeSnapshotIdentity
    phase_checkpoint_cycle_ordinals: tuple[int, ...]
    format: str = RUN_APPLICATION_INPUT_FORMAT

    def __post_init__(self) -> None:
        if self.format != RUN_APPLICATION_INPUT_FORMAT:
            raise ValueError("Protocol 10 application input format is unsupported")
        self.require_method(self.method)
        if not isinstance(self.backbone, QwenBackboneConfig | QwenMultimodalBackboneConfig):
            raise TypeError("Protocol 10 application input requires a Qwen deployment")
        expected_kind = (
            "qwen-multimodal"
            if isinstance(self.backbone, QwenMultimodalBackboneConfig)
            else "qwen-causal"
        )
        if self.backbone_kind != expected_kind:
            raise ValueError("Protocol 10 backbone kind and deployment differ")
        if not isinstance(self.application, ApplicationConfig):
            raise TypeError("Protocol 10 application input requires application config")
        if not isinstance(self.checkpoint_storage, PrivateCheckpointStorageBinding):
            raise TypeError("Protocol 10 application input requires checkpoint storage")
        if not isinstance(self.initial_checkpoint, PrivateInitialCheckpointBinding):
            raise TypeError("Protocol 10 application input requires initial checkpoint")
        if not self.seed_documents or any(
            not isinstance(document, SkillDocument) for document in self.seed_documents
        ):
            raise TypeError("Protocol 10 application input requires a seed library")
        if not isinstance(self.attempt_budget, BudgetVector):
            raise TypeError("Protocol 10 application input requires an attempt budget")
        if not isinstance(self.phi_per_cycle_maximum, BudgetVector):
            raise TypeError("Protocol 10 application input requires a Phi budget")
        if not isinstance(self.authoring_authority, SkillAuthoringAuthority):
            raise TypeError("Protocol 10 application input requires authoring authority")
        if not isinstance(self.snapshot_identity, RuntimeSnapshotIdentity):
            raise TypeError("Protocol 10 application input requires snapshot identity")

    @classmethod
    def from_value(cls, value: object) -> ApplicationInput:
        data = _object(
            value,
            fields={
                "application",
                "attempt_budget",
                "authoring_authority",
                "backbone",
                "backbone_kind",
                "checkpoint_storage",
                "format",
                "initial_checkpoint",
                "method",
                "method_contract",
                "phase_checkpoint_cycle_ordinals",
                "phi_per_cycle_maximum",
                "seed_documents",
                "snapshot_identity",
            },
            label="Protocol 10 application input",
        )
        documents = data["seed_documents"]
        ordinals = data["phase_checkpoint_cycle_ordinals"]
        backbone = data["backbone"]
        if not isinstance(documents, list) or not isinstance(ordinals, list):
            raise TypeError("Protocol 10 documents and phase ordinals must be arrays")
        if any(type(ordinal) is not int for ordinal in ordinals):
            raise TypeError("Protocol 10 phase ordinals must be integers")
        if not isinstance(backbone, dict):
            raise TypeError("Protocol 10 backbone must be an object")
        kind = data["backbone_kind"]
        if type(kind) is not str:
            raise TypeError("Protocol 10 backbone kind must be text")
        if kind == "qwen-causal":
            deployment: QwenDeploymentConfig = QwenBackboneConfig.from_value(backbone)
        elif kind == "qwen-multimodal":
            deployment = QwenMultimodalBackboneConfig.from_value(backbone)
        else:
            raise ValueError("Protocol 10 Qwen deployment kind is unsupported")
        format_value = data["format"]
        method_value = data["method"]
        if type(format_value) is not str or type(method_value) is not str:
            raise TypeError("Protocol 10 application format and method must be text")
        return cls(
            method=FormalMethod(method_value),
            method_contract=_method_contract_from_value(data["method_contract"]),
            backbone=deployment,
            backbone_kind=kind,
            application=ApplicationConfig.from_value(data["application"]),
            checkpoint_storage=PrivateCheckpointStorageBinding.from_value(
                data["checkpoint_storage"]
            ),
            initial_checkpoint=PrivateInitialCheckpointBinding.from_value(
                data["initial_checkpoint"]
            ),
            seed_documents=tuple(SkillDocument.from_value(item) for item in documents),
            attempt_budget=BudgetVector.from_value(data["attempt_budget"]),
            phi_per_cycle_maximum=BudgetVector.from_value(data["phi_per_cycle_maximum"]),
            authoring_authority=SkillAuthoringAuthority.from_value(data["authoring_authority"]),
            snapshot_identity=RuntimeSnapshotIdentity.from_value(data["snapshot_identity"]),
            phase_checkpoint_cycle_ordinals=tuple(cast(list[int], ordinals)),
            format=format_value,
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "application": self.application.to_value(),
            "attempt_budget": self.attempt_budget.to_value(),
            "authoring_authority": self.authoring_authority.to_value(),
            "backbone": self.backbone.to_value(),
            "backbone_kind": self.backbone_kind,
            "checkpoint_storage": self.checkpoint_storage.to_value(),
            "format": self.format,
            "initial_checkpoint": self.initial_checkpoint.to_value(),
            "method": self.method.value,
            "method_contract": self.method_contract.to_value(),
            "phase_checkpoint_cycle_ordinals": list(self.phase_checkpoint_cycle_ordinals),
            "phi_per_cycle_maximum": self.phi_per_cycle_maximum.to_value(),
            "seed_documents": [document.to_value() for document in self.seed_documents],
            "snapshot_identity": self.snapshot_identity.to_value(),
        }

    def require_method(self, method: FormalMethod) -> None:
        if method is not self.method:
            raise ValueError("Protocol 10 method and tagged application input differ")
        require_method_contract(method, self.method_contract)

    @classmethod
    def read(cls, path: Path) -> ApplicationInput:
        if not path.is_absolute() or not path.is_file():
            raise ValueError("Protocol 10 application input must be an absolute file")
        return cls.from_value(json.loads(path.read_text(encoding="utf-8")))

    def identity(
        self,
        *,
        method: FormalMethod,
        experiment: FormalExperimentSpec,
    ) -> ApplicationIdentity:
        self.require_method(method)
        if self.application.trainer.execution.batch_size != experiment.batch_size:
            raise ValueError("Protocol 10 application batch size differs from the experiment")
        return ApplicationIdentity(
            method=method,
            application_config=self.application,
            run_plan=experiment.run_plan,
            initial_run_cursor=AttemptRunCursorState.fresh(experiment.run_plan),
            snapshot_identity=self.snapshot_identity,
            phase_checkpoint_cycle_ordinals=self.phase_checkpoint_cycle_ordinals,
        )

    def terminal_components(self, *, run_id: str, attempt_id: str) -> TerminalComponents:
        return TerminalComponents(
            ledger=BudgetLedger(run_id=run_id, attempt_id=attempt_id, cap=self.attempt_budget),
            authoring_authority=self.authoring_authority,
            phi_budget=PhiBudgetAuthority(self.phi_per_cycle_maximum),
        )


__all__ = [
    "RUN_APPLICATION_INPUT_FORMAT",
    "SKILLFLOW_PARITY_CONTRACT",
    "SKILLFLOW_UPSTREAM_REVISION",
    "BayesianImproveApplicationContract",
    "ExactSkillFlowApplicationContract",
    "ApplicationIdentity",
    "ApplicationInput",
    "ApplicationInputKind",
    "require_method_contract",
]
