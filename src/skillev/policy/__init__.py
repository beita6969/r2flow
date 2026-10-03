from typing import TYPE_CHECKING

from .artifact_identity import (
    ARTIFACT_FILE_IDENTITY_FORMAT,
    BASE_MODEL_ARTIFACT_IDENTITY_FORMAT,
    ArtifactFileIdentity,
    BaseModelArtifactIdentity,
)
from .config import (
    QWEN35_GATED_DELTA_KERNEL_PACKAGE,
    QWEN35_GATED_DELTA_KERNEL_VERSION,
    TEACHER_FORCED_ACTIVATION_OFFLOAD_MIN_TOKENS,
    QwenBackboneConfig,
    QwenDeploymentConfig,
    QwenMultimodalBackboneConfig,
    public_qwen_deployment_hash,
    qwen_backend_class,
    qwen_dtype_conversion_policy,
)
from .interface import (
    ROLLOUT_PROMPT_ENCODER_VERSION,
    ROLLOUT_SOURCE_MESSAGES_BEGIN,
    ROLLOUT_SOURCE_MESSAGES_END,
    AdapterRole,
    GenerationResult,
    PolicyBackbone,
    PolicyGenerationRequest,
    PolicyParameterGroups,
    PolicyScoringMemoryError,
    PolicyTokenizerProtocol,
    RolloutPromptTokenizerProtocol,
    encode_rollout_prompt,
    rollout_chat_messages,
)
from .tokenizer_identity import (
    PUBLIC_TOKENIZER_IDENTITY_FORMAT,
    TOKENIZER_ARTIFACT_IDENTITY_FORMAT,
    PublicTokenizerIdentity,
    PublicTokenizerKind,
    TokenizerArtifactIdentity,
)
from .trainable_state import (
    PRIVATE_INITIAL_CHECKPOINT_BINDING_FORMAT,
    TRAINABLE_STATE_IDENTITY_FORMAT,
    PrivateInitialCheckpointBinding,
    TrainableStateIdentity,
)

if TYPE_CHECKING:
    from .hf_backbone import (
        QwenMultimodalPolicyBackbone,
        QwenPolicyBackbone,
        build_qwen_policy_backbone,
    )
    from .tokenizer import QwenTokenizerAdapter, qwen_tokenizer_artifact_identity

__all__ = [
    "ARTIFACT_FILE_IDENTITY_FORMAT",
    "BASE_MODEL_ARTIFACT_IDENTITY_FORMAT",
    "PRIVATE_INITIAL_CHECKPOINT_BINDING_FORMAT",
    "PUBLIC_TOKENIZER_IDENTITY_FORMAT",
    "QWEN35_GATED_DELTA_KERNEL_PACKAGE",
    "QWEN35_GATED_DELTA_KERNEL_VERSION",
    "ROLLOUT_PROMPT_ENCODER_VERSION",
    "ROLLOUT_SOURCE_MESSAGES_BEGIN",
    "ROLLOUT_SOURCE_MESSAGES_END",
    "TEACHER_FORCED_ACTIVATION_OFFLOAD_MIN_TOKENS",
    "TOKENIZER_ARTIFACT_IDENTITY_FORMAT",
    "TRAINABLE_STATE_IDENTITY_FORMAT",
    "AdapterRole",
    "ArtifactFileIdentity",
    "PolicyTokenizerProtocol",
    "BaseModelArtifactIdentity",
    "GenerationResult",
    "PolicyBackbone",
    "PolicyGenerationRequest",
    "PolicyParameterGroups",
    "PolicyScoringMemoryError",
    "PrivateInitialCheckpointBinding",
    "PublicTokenizerIdentity",
    "PublicTokenizerKind",
    "QwenBackboneConfig",
    "QwenDeploymentConfig",
    "QwenMultimodalBackboneConfig",
    "QwenMultimodalPolicyBackbone",
    "QwenPolicyBackbone",
    "QwenTokenizerAdapter",
    "RolloutPromptTokenizerProtocol",
    "TokenizerArtifactIdentity",
    "TrainableStateIdentity",
    "build_qwen_policy_backbone",
    "encode_rollout_prompt",
    "public_qwen_deployment_hash",
    "qwen_backend_class",
    "qwen_dtype_conversion_policy",
    "qwen_tokenizer_artifact_identity",
    "rollout_chat_messages",
]


def __getattr__(name: str) -> object:
    if name == "QwenPolicyBackbone":
        from .hf_backbone import QwenPolicyBackbone

        return QwenPolicyBackbone
    if name == "QwenMultimodalPolicyBackbone":
        from .hf_backbone import QwenMultimodalPolicyBackbone

        return QwenMultimodalPolicyBackbone
    if name == "QwenTokenizerAdapter":
        from .tokenizer import QwenTokenizerAdapter

        return QwenTokenizerAdapter
    if name == "build_qwen_policy_backbone":
        from .hf_backbone import build_qwen_policy_backbone

        return build_qwen_policy_backbone
    if name == "qwen_tokenizer_artifact_identity":
        from .tokenizer import qwen_tokenizer_artifact_identity

        return qwen_tokenizer_artifact_identity
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
