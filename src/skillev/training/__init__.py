from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skillev.rollout import RolloutSessionBundle, UnskilledRolloutSessionBundle

    from .checkpoint import (
        FilesystemTrainingCheckpointStore,
        TrainingCheckpointSnapshot,
        TrainingCheckpointStore,
    )
    from .config import (
        CheckpointConfig,
        OptimizerConfig,
        PolicyRolloutConfig,
        PrivateCheckpointStorageBinding,
        TrainerConfig,
        TrainingExecutionConfig,
        TTBMethodConfig,
        conservative_rollout_maximum,
    )
    from .distributed_ttb import (
        DistributedTTBError,
        DistributedTTBGradientCoordinator,
        DistributedTTBOOMKind,
        DistributedTTBPartitionableOOMError,
        DistributedTTBSingleArtifactOOMError,
        DistributedTTBTopology,
        DistributedTTBWorkerOOMError,
        classify_distributed_ttb_oom,
        initialize_distributed_ttb,
        partition_training_batch,
        serve_distributed_ttb_worker,
        training_artifact_token_cost,
    )
    from .gradient_worker import (
        AggregatedGradientResult,
        CostBalancedGradientWorkerPool,
        GradientRequest,
        GradientResult,
        GradientTensorSummary,
        GradientWorkerClient,
        GradientWorkerOOMError,
        GradientWorkItem,
        SealedGradientBatch,
        aggregate_gradient_results,
        partition_items_by_token_cost,
    )
    from .loop import ComponentGradientNorms, TrainingLoop
    from .planning import (
        AppliedTrainingState,
        BatchReadyTrainingState,
        CollectedTrainingBatch,
        FixedAttemptBudgetPlan,
        IdleTrainingState,
        PlannedRollout,
        TrainingBatchPlan,
        TrainingRuntimeState,
        TrainingStepExecutionContext,
    )
    from .projections import (
        POSTERIOR_PROVENANCE_FORMAT,
        FlowOnlyProjectionRuntimeState,
        FullProjectionRuntimeState,
        MethodProjectionPipeline,
        OnlineDiagnosticsProjection,
        PosteriorEventProvenance,
        ProjectionBatchRuntimeState,
        ProjectionRuntimeState,
        ProjectionTransition,
        TrainingProjectionPipeline,
        TrainingStepSource,
        projection_state_from_value,
    )
    from .rollout_workflow import (
        ROLLOUT_WORKFLOW_BINDING_FORMAT,
        AsyncResourceLimiter,
        ResourceTiming,
        RolloutBatchPerformanceReport,
        RolloutBatchWorkflow,
        RolloutWorkflowBinding,
        RolloutWorkflowResources,
    )
    from .runtime_components import (
        RolloutBatchCollector,
        RolloutSessionFactory,
        SkillLibraryView,
        TaskProvider,
        TTBOptimizerKernel,
    )
    from .step_math import TTBStepPreparer

_EXPORTS: dict[str, tuple[str, str]] = {
    **{
        name: ("skillev.training.config", name)
        for name in (
            "CheckpointConfig",
            "PrivateCheckpointStorageBinding",
            "OptimizerConfig",
            "PolicyRolloutConfig",
            "TrainerConfig",
            "TrainingExecutionConfig",
            "TTBMethodConfig",
            "conservative_rollout_maximum",
        )
    },
    **{
        name: ("skillev.training.planning", name)
        for name in (
            "AppliedTrainingState",
            "BatchReadyTrainingState",
            "CollectedTrainingBatch",
            "FixedAttemptBudgetPlan",
            "IdleTrainingState",
            "PlannedRollout",
            "TrainingBatchPlan",
            "TrainingStepExecutionContext",
            "TrainingRuntimeState",
        )
    },
    **{
        name: ("skillev.training.projections", name)
        for name in (
            "FlowOnlyProjectionRuntimeState",
            "FullProjectionRuntimeState",
            "MethodProjectionPipeline",
            "OnlineDiagnosticsProjection",
            "POSTERIOR_PROVENANCE_FORMAT",
            "PosteriorEventProvenance",
            "ProjectionBatchRuntimeState",
            "ProjectionRuntimeState",
            "ProjectionTransition",
            "TrainingProjectionPipeline",
            "TrainingStepSource",
            "projection_state_from_value",
        )
    },
    "RolloutSessionBundle": ("skillev.rollout", "RolloutSessionBundle"),
    "UnskilledRolloutSessionBundle": ("skillev.rollout", "UnskilledRolloutSessionBundle"),
    **{
        name: ("skillev.training.distributed_ttb", name)
        for name in (
            "DistributedTTBError",
            "DistributedTTBGradientCoordinator",
            "DistributedTTBOOMKind",
            "DistributedTTBPartitionableOOMError",
            "DistributedTTBSingleArtifactOOMError",
            "DistributedTTBTopology",
            "DistributedTTBWorkerOOMError",
            "classify_distributed_ttb_oom",
            "initialize_distributed_ttb",
            "partition_training_batch",
            "serve_distributed_ttb_worker",
            "training_artifact_token_cost",
        )
    },
    **{
        name: ("skillev.training.checkpoint", name)
        for name in (
            "FilesystemTrainingCheckpointStore",
            "TrainingCheckpointSnapshot",
            "TrainingCheckpointStore",
        )
    },
    **{
        name: ("skillev.training.loop", name) for name in ("ComponentGradientNorms", "TrainingLoop")
    },
    "TTBStepPreparer": ("skillev.training.step_math", "TTBStepPreparer"),
    **{
        name: ("skillev.training.runtime_components", name)
        for name in (
            "RolloutBatchCollector",
            "RolloutSessionFactory",
            "SkillLibraryView",
            "TaskProvider",
            "TTBOptimizerKernel",
        )
    },
    **{
        name: ("skillev.training.rollout_workflow", name)
        for name in (
            "AsyncResourceLimiter",
            "ResourceTiming",
            "RolloutBatchPerformanceReport",
            "ROLLOUT_WORKFLOW_BINDING_FORMAT",
            "RolloutBatchWorkflow",
            "RolloutWorkflowBinding",
            "RolloutWorkflowResources",
        )
    },
    **{
        name: ("skillev.training.gradient_worker", name)
        for name in (
            "AggregatedGradientResult",
            "CostBalancedGradientWorkerPool",
            "GradientRequest",
            "GradientResult",
            "GradientTensorSummary",
            "GradientWorkerClient",
            "GradientWorkerOOMError",
            "GradientWorkItem",
            "SealedGradientBatch",
            "aggregate_gradient_results",
            "partition_items_by_token_cost",
        )
    },
}

__all__ = [
    "POSTERIOR_PROVENANCE_FORMAT",
    "ROLLOUT_WORKFLOW_BINDING_FORMAT",
    "AggregatedGradientResult",
    "AppliedTrainingState",
    "AsyncResourceLimiter",
    "BatchReadyTrainingState",
    "CheckpointConfig",
    "CollectedTrainingBatch",
    "ComponentGradientNorms",
    "CostBalancedGradientWorkerPool",
    "DistributedTTBError",
    "DistributedTTBGradientCoordinator",
    "DistributedTTBOOMKind",
    "DistributedTTBPartitionableOOMError",
    "DistributedTTBSingleArtifactOOMError",
    "DistributedTTBTopology",
    "DistributedTTBWorkerOOMError",
    "FilesystemTrainingCheckpointStore",
    "FixedAttemptBudgetPlan",
    "FlowOnlyProjectionRuntimeState",
    "FullProjectionRuntimeState",
    "GradientRequest",
    "GradientResult",
    "GradientTensorSummary",
    "GradientWorkItem",
    "GradientWorkerClient",
    "GradientWorkerOOMError",
    "IdleTrainingState",
    "MethodProjectionPipeline",
    "OnlineDiagnosticsProjection",
    "OptimizerConfig",
    "PlannedRollout",
    "PolicyRolloutConfig",
    "PosteriorEventProvenance",
    "PrivateCheckpointStorageBinding",
    "ProjectionBatchRuntimeState",
    "ProjectionRuntimeState",
    "ProjectionTransition",
    "ResourceTiming",
    "RolloutBatchCollector",
    "RolloutBatchPerformanceReport",
    "RolloutBatchWorkflow",
    "RolloutSessionBundle",
    "RolloutSessionFactory",
    "RolloutWorkflowBinding",
    "RolloutWorkflowResources",
    "SealedGradientBatch",
    "SkillLibraryView",
    "TTBMethodConfig",
    "TTBOptimizerKernel",
    "TTBStepPreparer",
    "TaskProvider",
    "TrainerConfig",
    "TrainingBatchPlan",
    "TrainingCheckpointSnapshot",
    "TrainingCheckpointStore",
    "TrainingExecutionConfig",
    "TrainingLoop",
    "TrainingProjectionPipeline",
    "TrainingRuntimeState",
    "TrainingStepExecutionContext",
    "TrainingStepSource",
    "UnskilledRolloutSessionBundle",
    "aggregate_gradient_results",
    "classify_distributed_ttb_oom",
    "conservative_rollout_maximum",
    "initialize_distributed_ttb",
    "partition_items_by_token_cost",
    "partition_training_batch",
    "projection_state_from_value",
    "serve_distributed_ttb_worker",
    "training_artifact_token_cost",
]


def __getattr__(name: str) -> object:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
