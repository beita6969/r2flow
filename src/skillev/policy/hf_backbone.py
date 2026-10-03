from __future__ import annotations

import copy
import importlib.metadata
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    PreTrainedModel,
)
from transformers.cache_utils import Cache, DynamicCache

from .artifact_identity import BaseModelArtifactIdentity
from .checkpoint import (
    CHECKPOINT_FORMAT,
    PolicyCheckpointState,
    load_policy_checkpoint,
    save_policy_checkpoint,
    trainable_state_identity,
)
from .config import (
    QWEN35_GATED_DELTA_KERNEL_PACKAGE,
    QWEN35_GATED_DELTA_KERNEL_VERSION,
    QwenBackboneConfig,
    QwenDeploymentConfig,
    QwenMultimodalBackboneConfig,
    public_qwen_deployment_hash,
    qwen_backend_class,
    qwen_dtype_conversion_policy,
)
from .flow_head import (
    FLOW_OFFSET_FILE,
    FLOW_OFFSET_FORMAT,
    FLOW_OFFSET_PARAMETER,
    FlowHeadSpec,
    psi_reset_seed,
)
from .interface import (
    AdapterRole,
    PolicyTokenizerProtocol,
    GenerationResult,
    PolicyGenerationRequest,
    PolicyParameterGroups,
)
from .tokenizer import QwenTokenizerAdapter
from .trainable_state import TrainableStateIdentity
from .versions import TrainableVersions
from .z_initialization import ZInitializationSpec

if TYPE_CHECKING:
    from skillev.training.performance_config import TrainingPerformanceConfig

    from .scoring_execution import TeacherForcingConfig
    from .teacher_forcing import ScoringSequence, SequenceScores


def require_qwen35_gated_delta_kernel(model: nn.Module) -> None:
    try:
        installed = importlib.metadata.version(QWEN35_GATED_DELTA_KERNEL_PACKAGE)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("checkpointed CUDA Qwen3.5 requires flash-linear-attention") from error
    if installed != QWEN35_GATED_DELTA_KERNEL_VERSION:
        raise RuntimeError("checkpointed CUDA Qwen3.5 requires the pinned FLA version")
    delta_modules = tuple(
        module for module in model.modules() if type(module).__name__ == "Qwen3_5GatedDeltaNet"
    )
    if not delta_modules:
        raise RuntimeError("Qwen3.5 model has no gated-delta modules")
    if any(
        not getattr(
            getattr(module, "chunk_gated_delta_rule", None),
            "__module__",
            "",
        ).startswith("fla.")
        for module in delta_modules
    ):
        raise RuntimeError("Qwen3.5 did not select the pinned FLA gated-delta kernel")


_POLICY_EPISODE_UNSTABLE_TAIL_TOKENS: Final = 64
_DTYPES: Final = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _validate_ids(ids: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(ids, tuple) or not ids:
        raise ValueError(f"{field} must be a non-empty tuple of token ids")
    if any(type(token_id) is not int or token_id < 0 for token_id in ids):
        raise ValueError(f"{field} must contain non-negative integer token ids")
    return ids


class _QwenPolicyBackboneBase:
    def __init__(
        self,
        config: QwenBackboneConfig,
        *,
        model_loader: Callable[..., PreTrainedModel],
        backend_name: str,
        performance: TrainingPerformanceConfig | None = None,
    ) -> None:
        self._config = config
        self._scoring_role: AdapterRole | None = None
        self.performance_config = performance
        from .scoring_execution import TeacherForcingConfig
        from .teacher_forcing import TeacherForcingExecutor

        self._teacher_forcing = TeacherForcingExecutor(
            TeacherForcingConfig() if performance is None else performance.teacher_forcing
        )
        self._z_feature_cache: OrderedDict[tuple[int, ...], torch.Tensor] = OrderedDict()
        self._z_feature_cache_entries = (
            0 if performance is None else performance.z_feature_cache_entries
        )
        self._z_feature_hits = 0
        self._z_feature_misses = 0
        self._z_feature_seconds = 0.0
        dtype = _DTYPES[config.torch_dtype]
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA Qwen backend requested but CUDA is unavailable")
        device = torch.device(config.device)
        artifact_before = self._measure_base_model_artifact(config)
        if artifact_before is not None and artifact_before != config.base_model_artifact:
            raise ValueError("Qwen base files differ from the pinned artifact before load")
        loaded = model_loader(
            config.base_model_path,
            attn_implementation=config.attention_implementation,
            dtype=dtype,
            local_files_only=True,
            revision=config.revision,
            trust_remote_code=False,
        )
        artifact_after = self._measure_base_model_artifact(config)
        if artifact_before is not None and (
            artifact_after != artifact_before or artifact_after != config.base_model_artifact
        ):
            raise ValueError("Qwen base files changed while the model was loading")
        loaded_module = cast(nn.Module, loaded)
        loaded_module.to(device=device, dtype=dtype)
        base_model = cast(PreTrainedModel, loaded_module)
        base_model.requires_grad_(False)
        base_model.eval()
        if config.device == "cuda" and config.teacher_forced_gradient_checkpointing:
            require_qwen35_gated_delta_kernel(base_model)
            if performance is not None and performance.deterministic_gradients:
                from .fla_execution import configure_qwen35_fla_kernels

                configure_qwen35_fla_kernels(performance.fla_profile)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_target_modules),
            bias="none",
        )
        peft_model = get_peft_model(
            base_model,
            lora_config,
            adapter_name=AdapterRole.FORWARD_POLICY.value,
        )
        if not isinstance(peft_model, PeftModel):
            raise RuntimeError("PEFT did not construct a PeftModel")
        peft_model.add_adapter(
            AdapterRole.BACKWARD_POLICY.value,
            lora_config,
        )
        peft_model.set_requires_grad([role.value for role in AdapterRole], True)
        peft_model.eval()
        self._model = peft_model
        self._input_device = device
        if config.teacher_forced_gradient_checkpointing:
            nonzero_dropout = tuple(
                module.p
                for module in peft_model.modules()
                if isinstance(module, nn.Dropout) and module.p != 0.0
            )
            if nonzero_dropout:
                raise ValueError(
                    "gradient-checkpointed teacher forcing requires a zero-dropout model"
                )
            peft_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            peft_model.enable_input_require_grads()

        self._tokenizer: PolicyTokenizerProtocol = QwenTokenizerAdapter.from_config(config)
        self._z_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.z_hidden_width),
            nn.GELU(),
            nn.Linear(config.z_hidden_width, 1),
        ).to(device=self._input_device, dtype=torch.float32)
        self._initialize_z_module(self._z_head, seed=config.z_initialization.initial_seed)

        self._forward_parameters = self._adapter_parameters(AdapterRole.FORWARD_POLICY)
        self._backward_parameters = self._adapter_parameters(AdapterRole.BACKWARD_POLICY)
        self._z_parameters = tuple(self._z_head.parameters())
        self._flow_offsets: nn.Parameter | None = None
        self._flow_offset_domains: tuple[str, ...] = ()
        self._flow_offsets_initialization: dict[str, Any] | None = None
        self._psi_head: nn.Sequential = config.flow_head.build(config.hidden_size).to(
            device=self._input_device, dtype=torch.float32
        )
        config.flow_head.initialize(
            self._psi_head, seed=psi_reset_seed(config.flow_head.initial_seed)
        )
        self._psi_parameters: tuple[nn.Parameter, ...] = tuple(self._psi_head.parameters())
        if set(map(id, self._psi_parameters)) & set(
            map(
                id,
                (*self._forward_parameters, *self._backward_parameters, *self._z_parameters),
            )
        ):
            raise RuntimeError("policy parameter groups overlap")
        if not self._forward_parameters or not self._backward_parameters or not self._z_parameters:
            raise RuntimeError("policy parameter groups were not constructed")
        if (
            set(map(id, self._forward_parameters)) & set(map(id, self._backward_parameters))
            or set(map(id, self._forward_parameters)) & set(map(id, self._z_parameters))
            or set(map(id, self._backward_parameters)) & set(map(id, self._z_parameters))
        ):
            raise RuntimeError("policy parameter groups overlap")

        self._public_deployment_hash = public_qwen_deployment_hash(
            config,
            backend_kind=(
                "qwen-multimodal"
                if isinstance(config, QwenMultimodalBackboneConfig)
                else "qwen-causal"
            ),
        )
        self._backbone_id = self._public_deployment_hash
        self._adapter_versions = dict.fromkeys(
            AdapterRole,
            f"{self._backbone_id}@0",
        )
        self._z_version = f"{self._backbone_id}@0"
        self._initial_trainable_state_hash = self.trainable_state_identity.content_hash
        self._policy_episode_id: str | None = None
        self._policy_episode_tokens: tuple[int, ...] = ()
        self._policy_episode_past: Cache | None = None
        self._last_policy_prefill_token_count = 0
        self._last_policy_reused_token_count = 0
        self._policy_episode_total_prefill_token_count = 0
        self._policy_episode_total_reused_token_count = 0
        self._last_completed_policy_episode_prefill_token_count = 0
        self._last_completed_policy_episode_reused_token_count = 0

    @staticmethod
    def _measure_base_model_artifact(
        config: QwenBackboneConfig,
    ) -> BaseModelArtifactIdentity | None:
        if config.base_model_artifact is None:
            return None
        return BaseModelArtifactIdentity.from_directory(
            directory=Path(config.base_model_path),
            backend_class=qwen_backend_class(config),
            upstream_revision=config.revision,
            dtype_conversion_policy=qwen_dtype_conversion_policy(config),
        )

    @property
    def backbone_id(self) -> str:
        return self._backbone_id

    @property
    def tokenizer(self) -> PolicyTokenizerProtocol:
        return self._tokenizer

    def adapter_version(self, role: AdapterRole) -> str:
        if not isinstance(role, AdapterRole):
            raise ValueError("role must be an AdapterRole")
        return self._adapter_versions[role]

    @property
    def z_version(self) -> str:
        return self._z_version

    @property
    def trainable_state_identity(self) -> TrainableStateIdentity:
        return trainable_state_identity(
            model=self._model,
            z_head=self._z_head,
            backbone_deployment_hash=self._public_deployment_hash,
            psi_head=self._psi_head,
        )

    @property
    def initial_trainable_state_hash(self) -> str:
        return self._initial_trainable_state_hash

    def bind_initial_trainable_state(self, expected: TrainableStateIdentity) -> None:
        if not isinstance(expected, TrainableStateIdentity):
            raise TypeError("initial trainable state must be TrainableStateIdentity")
        if expected.backbone_deployment_hash != self._public_deployment_hash:
            raise ValueError("initial trainable state targets another deployment")
        if self.trainable_state_identity != expected:
            raise ValueError("loaded trainable tensors differ from formal initial state")
        self._initial_trainable_state_hash = expected.content_hash

    def mark_policy_update(self, optimizer_step: int) -> None:
        if type(optimizer_step) is not int or optimizer_step < 0:
            raise ValueError("optimizer_step must be a non-negative integer")
        self._adapter_versions = {
            role: f"{version.rpartition('@')[0]}@{optimizer_step}"
            for role, version in self._adapter_versions.items()
        }
        self._z_version = f"{self._z_version.rpartition('@')[0]}@{optimizer_step}"
        self._clear_policy_episode()

    def begin_policy_episode(self, episode_id: str) -> None:
        if type(episode_id) is not str or not episode_id:
            raise ValueError("episode_id must be non-empty text")
        self._clear_policy_episode()
        self._policy_episode_id = episode_id
        self._policy_episode_total_prefill_token_count = 0
        self._policy_episode_total_reused_token_count = 0

    def synchronize_trainable_versions(self, versions: TrainableVersions) -> None:
        self._require_idle_scoring()
        if not isinstance(versions, TrainableVersions):
            raise TypeError("parameter synchronization requires trainable component versions")
        self._adapter_versions = {
            AdapterRole.FORWARD_POLICY: versions.forward,
            AdapterRole.BACKWARD_POLICY: versions.backward,
        }
        self._z_version = versions.z
        self._clear_policy_episode()

    def end_policy_episode(self, episode_id: str) -> None:
        if episode_id != self._policy_episode_id:
            raise ValueError("policy episode identity mismatch")
        self._last_completed_policy_episode_prefill_token_count = (
            self._policy_episode_total_prefill_token_count
        )
        self._last_completed_policy_episode_reused_token_count = (
            self._policy_episode_total_reused_token_count
        )
        self._clear_policy_episode()

    def _clear_policy_episode(self) -> None:
        self._policy_episode_id = None
        self._policy_episode_tokens = ()
        self._policy_episode_past = None

    def _adapter_parameters(self, role: AdapterRole) -> tuple[nn.Parameter, ...]:
        parameters: list[nn.Parameter] = []
        seen: set[int] = set()
        for module in self._model.modules():
            if not isinstance(module, LoraLayer):
                continue
            for registry in (module.lora_A, module.lora_B):
                if role.value not in registry:
                    continue
                for parameter in registry[role.value].parameters():
                    if id(parameter) not in seen:
                        seen.add(id(parameter))
                        parameters.append(parameter)
        return tuple(parameters)

    def _activate_adapter(self, role: AdapterRole) -> None:
        if not isinstance(role, AdapterRole):
            raise ValueError("role must be an AdapterRole")
        self._require_idle_scoring()
        self._model.set_adapter(role.value, inference_mode=False)
        self._model.set_requires_grad([item.value for item in AdapterRole], True)
        self._model.eval()

    def _require_idle_scoring(self) -> None:
        if self._scoring_role is not None:
            raise RuntimeError("cannot mutate policy state during a scoring/backward interval")

    @contextmanager
    def scoring_session(self, role: AdapterRole) -> Iterator[None]:
        if self._scoring_role is not None:
            raise RuntimeError("scoring sessions cannot overlap")
        previous_adapter = self._model.active_adapter
        previous_training = self._model.training
        self._activate_adapter(role)
        self._scoring_role = role
        try:
            with self._teacher_forcing.session(self._model):
                yield
        finally:
            self._scoring_role = None
            self._model.set_adapter(previous_adapter, inference_mode=False)
            self._model.set_requires_grad([item.value for item in AdapterRole], True)
            self._model.train(previous_training)

    @property
    def scoring_checkpoint_enabled(self) -> bool:
        return self._config.teacher_forced_gradient_checkpointing

    def _ids_tensor(self, ids: tuple[int, ...], *, field: str) -> torch.Tensor:
        validated = _validate_ids(ids, field=field)
        return torch.tensor([validated], dtype=torch.long, device=self._input_device)

    def score(
        self,
        prefix_ids: tuple[int, ...],
        action_ids: tuple[int, ...],
        role: AdapterRole,
    ) -> torch.Tensor:
        return self.score_edge_microbatch(((prefix_ids, action_ids),), role)[0]

    def score_edge_microbatch(
        self,
        edges: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
        role: AdapterRole,
    ) -> tuple[torch.Tensor, ...]:
        validated = tuple(
            (_validate_ids(prefix, field="prefix_ids"), _validate_ids(action, field="action_ids"))
            for prefix, action in edges
        )
        if self._scoring_role is None:
            self._activate_adapter(role)
        elif self._scoring_role is not role:
            raise RuntimeError("scoring adapter differs from the active interval")
        return self._teacher_forcing.score(
            model=self._model,
            device=self._input_device,
            edges=validated,
            role=role,
            checkpoint_enabled=self._config.teacher_forced_gradient_checkpointing,
        )

    def score_sequence(self, sequence: ScoringSequence, role: AdapterRole) -> SequenceScores:
        if self._scoring_role is None:
            self._activate_adapter(role)
        elif self._scoring_role is not role:
            raise RuntimeError("scoring adapter differs from the active interval")
        return self._teacher_forcing.score_sequence(
            model=self._model,
            device=self._input_device,
            sequence=sequence,
            role=role,
            checkpoint_enabled=self._config.teacher_forced_gradient_checkpointing,
        )

    def score_reference_sequence(self, sequence: ScoringSequence) -> SequenceScores:
        role = self._scoring_role or AdapterRole.BACKWARD_POLICY
        with self._model.disable_adapter(), torch.no_grad():
            scores = self._teacher_forcing.score_sequence(
                model=self._model,
                device=self._input_device,
                sequence=sequence,
                role=role,
                checkpoint_enabled=False,
            )
        if self._teacher_forcing.metrics:
            self._teacher_forcing.metrics[-1]["reference"] = True
        return scores

    @property
    def teacher_forcing_config(self) -> TeacherForcingConfig:
        return self._teacher_forcing.config

    def drain_scoring_metrics(self) -> list[dict[str, Any]]:
        metrics, self._teacher_forcing.metrics = self._teacher_forcing.metrics, []
        return metrics

    def finish_edge_backward_profile(self) -> None:
        self._teacher_forcing.finish_backward_profile()

    def generate_policy(self, request: PolicyGenerationRequest) -> GenerationResult:
        if not isinstance(request, PolicyGenerationRequest):
            raise TypeError("policy generation requires PolicyGenerationRequest")
        self._activate_adapter(AdapterRole.FORWARD_POLICY)
        return self._generate_policy_locked(request)

    def _generation_state(
        self,
        input_ids: tuple[int, ...],
        *,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Generator, frozenset[int]]:
        ids = self._ids_tensor(input_ids, field="input_ids")
        attention_mask = torch.ones_like(ids)
        generator_device = "cuda" if self._input_device.type == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(seed)
        return ids, attention_mask, generator, frozenset(self._config.eos_token_ids)

    @staticmethod
    def _append_attention_slot(attention_mask: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                attention_mask,
                torch.ones(
                    (1, 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ),
            dim=1,
        )

    def _generate_policy_locked(
        self,
        request: PolicyGenerationRequest,
    ) -> GenerationResult:
        input_ids, _, generator, eos_ids = self._generation_state(
            request.input_ids,
            seed=request.seed,
        )
        content: list[int] = []
        reused_tokens = 0
        working_cache: Cache | None = None
        published_cache: Cache | None = None
        published_tokens: tuple[int, ...] = ()
        if self._policy_episode_id is not None and self._policy_episode_past is not None:
            cached = self._policy_episode_tokens
            if cached and request.input_ids[: len(cached)] != cached:
                raise RuntimeError("policy episode cache has no reusable canonical prompt prefix")
            reused_tokens = len(cached)
            if reused_tokens >= len(request.input_ids):
                raise RuntimeError("policy episode prompts must grow monotonically")
            working_cache = self._policy_episode_past
        elif self._policy_episode_id is not None:
            working_cache = DynamicCache(config=self._model.config)
        self._last_policy_reused_token_count = reused_tokens
        self._last_policy_prefill_token_count = len(request.input_ids) - reused_tokens
        if self._policy_episode_id is not None:
            self._policy_episode_total_reused_token_count += reused_tokens
            self._policy_episode_total_prefill_token_count += len(request.input_ids) - reused_tokens

        if working_cache is None:
            attention_mask = torch.ones_like(input_ids)
            next_input = input_ids
        else:
            stable_length = max(
                reused_tokens,
                len(request.input_ids) - _POLICY_EPISODE_UNSTABLE_TAIL_TOKENS,
            )
            if stable_length > reused_tokens:
                stable_input = self._ids_tensor(
                    request.input_ids[reused_tokens:stable_length],
                    field="stable policy episode suffix",
                )
                stable_attention = torch.ones(
                    (1, stable_length),
                    dtype=torch.long,
                    device=self._input_device,
                )
                with torch.no_grad():
                    stable_output = self._model(
                        input_ids=stable_input,
                        attention_mask=stable_attention,
                        past_key_values=working_cache,
                        use_cache=True,
                        return_dict=True,
                    )
                if not isinstance(stable_output.past_key_values, Cache):
                    raise RuntimeError("fixed Qwen generation did not return a Transformers cache")
                working_cache = stable_output.past_key_values
            published_cache = copy.deepcopy(working_cache)
            published_tokens = request.input_ids[:stable_length]
            next_input = self._ids_tensor(
                request.input_ids[stable_length:],
                field="unstable policy episode tail",
            )
            attention_mask = torch.ones(
                (1, len(request.input_ids)),
                dtype=torch.long,
                device=self._input_device,
            )
        with torch.no_grad():
            for _ in range(request.max_new_tokens):
                model_output = self._model(
                    input_ids=next_input,
                    attention_mask=attention_mask,
                    past_key_values=working_cache,
                    use_cache=True,
                    return_dict=True,
                )
                if not isinstance(model_output.past_key_values, Cache):
                    raise RuntimeError("fixed Qwen generation did not return a Transformers cache")
                working_cache = model_output.past_key_values
                next_token = self._sample_policy_token(
                    model_output.logits[:, -1, :].to(dtype=torch.float32),
                    generator=generator,
                )
                token_id = int(next_token.item())
                if token_id in eos_ids:
                    self._publish_policy_episode_cache(published_tokens, published_cache)
                    return GenerationResult(
                        content_token_ids=tuple(content),
                        stop_token_ids=(token_id,),
                        finish_reason="stop",
                    )
                content.append(token_id)
                next_input = next_token.reshape(1, 1)
                attention_mask = self._append_attention_slot(attention_mask)
        self._publish_policy_episode_cache(published_tokens, published_cache)
        return GenerationResult(
            content_token_ids=tuple(content),
            stop_token_ids=(),
            finish_reason="length",
        )

    def _publish_policy_episode_cache(
        self,
        processed_tokens: tuple[int, ...],
        past_key_values: Cache | None,
    ) -> None:
        if self._policy_episode_id is None or past_key_values is None:
            return
        self._policy_episode_tokens = processed_tokens
        self._policy_episode_past = past_key_values

    @staticmethod
    def _sample_policy_token(
        logits: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        probabilities = torch.softmax(logits.to(dtype=torch.float32), dim=-1)
        return torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator,
        ).squeeze(-1)

    def configure_performance(self, performance: TrainingPerformanceConfig) -> None:
        self._require_idle_scoring()
        performance.configure_process()
        self.performance_config = performance
        from .teacher_forcing import TeacherForcingExecutor

        if self._teacher_forcing.offload.live_pinned_bytes:
            raise RuntimeError("cannot replace execution configuration with a live scoring graph")
        self._teacher_forcing = TeacherForcingExecutor(performance.teacher_forcing)
        if self._input_device.type == "cuda" and performance.deterministic_gradients:
            from .fla_execution import configure_qwen35_fla_kernels

            configure_qwen35_fla_kernels(performance.fla_profile)
        self.clear_z_feature_cache()
        self._z_feature_cache_entries = performance.z_feature_cache_entries

    def clear_z_feature_cache(self) -> None:
        self._z_feature_cache.clear()

    @property
    def z_feature_cache_metrics(self) -> dict[str, int | float]:
        return {
            "hits": self._z_feature_hits,
            "misses": self._z_feature_misses,
            "entries": len(self._z_feature_cache),
            "forward_seconds": self._z_feature_seconds,
            "feature_bytes": sum(
                t.numel() * t.element_size() for t in self._z_feature_cache.values()
            ),
        }

    def _frozen_query_features(self, query_ids: tuple[int, ...]) -> torch.Tensor:
        key = _validate_ids(query_ids, field="query_ids")
        cached = self._z_feature_cache.get(key)
        if cached is not None:
            self._z_feature_cache.move_to_end(key)
            self._z_feature_hits += 1
            return cached
        started = time.perf_counter()
        self._z_feature_misses += 1
        input_ids = self._ids_tensor(key, field="query_ids")
        self._model.eval()
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "output_hidden_states": True,
            "use_cache": False,
            "return_dict": True,
        }
        if self._model.config.model_type in {"qwen3_5", "qwen3_5_text"}:
            kwargs["logits_to_keep"] = 1
        with self._model.disable_adapter(), torch.no_grad():
            outputs = self._model(**kwargs)
            features = (
                outputs.hidden_states[-1]
                .mean(dim=1)
                .to(
                    device=self._input_device,
                    dtype=torch.float32,
                )
                .detach()
                .clone()
            )
        if self._z_feature_cache_entries and not features.is_inference():
            self._z_feature_cache[key] = features
            while len(self._z_feature_cache) > self._z_feature_cache_entries:
                self._z_feature_cache.popitem(last=False)
        self._z_feature_seconds += time.perf_counter() - started
        return cast(torch.Tensor, features)

    def z_value(self, query_ids: tuple[int, ...]) -> torch.Tensor:
        self._require_idle_scoring()
        return cast(torch.Tensor, self._z_head(self._frozen_query_features(query_ids)).squeeze())

    @property
    def z_initialization_spec(self) -> ZInitializationSpec:
        return self._config.z_initialization

    def _initialize_z_module(self, module: nn.Module, *, seed: int) -> None:
        self._config.z_initialization.initialize(module, seed=seed)

    def reset_z(self, seed: int) -> str:
        self._require_idle_scoring()
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("Z reset seed must be an unsigned 64-bit integer")
        self._initialize_z_module(self._z_head, seed=seed)
        self._config.flow_head.initialize(self._psi_head, seed=psi_reset_seed(seed))
        self._z_version = f"z-reset-{seed:016x}@0"
        return self._z_version

    @property
    def flow_head_spec(self) -> FlowHeadSpec:
        return self._config.flow_head

    def psi_values(self, features: torch.Tensor) -> torch.Tensor:
        self._require_idle_scoring()
        if features.ndim != 2 or features.shape[1] != self._config.hidden_size:
            raise ValueError("flow-head features must be [n, hidden_size]")
        if features.requires_grad:
            raise ValueError("flow-head features must be detached")
        values = self._psi_head(features.to(device=self._input_device, dtype=torch.float32))
        return cast(torch.Tensor, values.squeeze(-1))

    def enable_flow_offsets(self, domains: tuple[str, ...]) -> None:
        if not domains or len(set(domains)) != len(domains):
            raise ValueError("flow offsets need unique domains")
        if self._flow_offsets is not None:
            if tuple(domains) != self._flow_offset_domains:
                raise ValueError("flow offsets are already enabled for other domains")
            return
        self._flow_offsets = nn.Parameter(
            torch.zeros(len(domains), dtype=torch.float32, device=self._input_device)
        )
        self._flow_offset_domains = tuple(domains)
        self._psi_parameters = (*self._psi_parameters, self._flow_offsets)

    @property
    def flow_offset_parameter(self) -> nn.Parameter | None:
        return self._flow_offsets

    @property
    def flow_offset_domains(self) -> tuple[str, ...]:
        return self._flow_offset_domains

    @property
    def flow_offsets_initialization(self) -> dict[str, Any] | None:
        return self._flow_offsets_initialization

    def flow_offset(self, domain: str) -> torch.Tensor:
        if self._flow_offsets is None:
            raise RuntimeError("flow offsets are not enabled")
        if domain not in self._flow_offset_domains:
            raise ValueError(f"no flow offset is declared for domain {domain!r}")
        return self._flow_offsets[self._flow_offset_domains.index(domain)]

    def initialize_flow_offsets(self, values: dict[str, float], record: dict[str, Any]) -> None:
        if self._flow_offsets is None:
            raise RuntimeError("flow offsets are not enabled")
        if self._flow_offsets_initialization is not None:
            raise RuntimeError("flow offsets are initialised once and never re-initialised")
        if set(values) - set(self._flow_offset_domains):
            raise ValueError("flow offset initialisation names an undeclared domain")
        with torch.no_grad():
            for domain, value in values.items():
                self._flow_offsets[self._flow_offset_domains.index(domain)] = float(value)
        self._flow_offsets_initialization = dict(record)

    def _load_flow_offsets(self, directory: Path) -> None:
        import json

        assert self._flow_offsets is not None
        value = json.loads((directory / FLOW_OFFSET_FILE).read_text(encoding="utf-8"))
        if (
            value.get("format") != FLOW_OFFSET_FORMAT
            or tuple(value.get("domains", ())) != self._flow_offset_domains
            or len(value.get("values", ())) != len(self._flow_offset_domains)
        ):
            raise ValueError("checkpoint flow offsets do not match this backbone")
        with torch.no_grad():
            self._flow_offsets.copy_(
                torch.tensor(value["values"], dtype=torch.float32, device=self._input_device)
            )
        self._flow_offsets_initialization = value.get("initialization")

    def parameter_groups(self) -> PolicyParameterGroups:
        return PolicyParameterGroups(
            forward=self._forward_parameters,
            backward=self._backward_parameters,
            z_head=self._z_parameters,
            psi_head=self._psi_parameters,
        )

    def named_trainable_parameters(self) -> dict[str, nn.Parameter]:
        trainable = set(map(id, (*self._forward_parameters, *self._backward_parameters)))
        named = {
            f"policy.{name}": parameter
            for name, parameter in self._model.named_parameters()
            if id(parameter) in trainable
        }
        named.update(
            {f"z_head.{name}": parameter for name, parameter in self._z_head.named_parameters()}
        )
        named.update(
            {f"psi_head.{name}": parameter for name, parameter in self._psi_head.named_parameters()}
        )
        if self._flow_offsets is not None:
            named[FLOW_OFFSET_PARAMETER] = self._flow_offsets
        return named

    def save_checkpoint(self, directory: str) -> None:
        if not isinstance(directory, str) or not directory.strip():
            raise ValueError("directory must be non-empty text")
        optimizer_step = int(self._adapter_versions[AdapterRole.FORWARD_POLICY].rpartition("@")[2])
        save_policy_checkpoint(
            directory=Path(directory),
            model=self._model,
            z_head=self._z_head,
            state=PolicyCheckpointState(
                backbone_id=self._backbone_id,
                forward_version=self._adapter_versions[AdapterRole.FORWARD_POLICY],
                backward_version=self._adapter_versions[AdapterRole.BACKWARD_POLICY],
                z_version=self._z_version,
                optimizer_step=optimizer_step,
                trainable_state=self.trainable_state_identity,
                format=CHECKPOINT_FORMAT,
            ),
            psi_head=self._psi_head,
        )
        if self._flow_offsets is not None:
            import json

            value = {
                "format": FLOW_OFFSET_FORMAT,
                "domains": list(self._flow_offset_domains),
                "values": [float(v) for v in self._flow_offsets.detach().cpu().tolist()],
                "initialization": self._flow_offsets_initialization,
            }
            (Path(directory) / FLOW_OFFSET_FILE).write_text(
                json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
            )

    def load_checkpoint(self, directory: str) -> None:
        self._require_idle_scoring()
        if not isinstance(directory, str) or not directory.strip():
            raise ValueError("directory must be non-empty text")
        state = load_policy_checkpoint(
            directory=Path(directory),
            model=self._model,
            z_head=self._z_head,
            device=self._input_device,
            psi_head=self._psi_head,
        )
        if self._flow_offsets is not None:
            self._load_flow_offsets(Path(directory))
        if state.backbone_id != self._backbone_id:
            raise ValueError("checkpoint targets another Qwen backbone")
        if state.trainable_state.backbone_deployment_hash != self._public_deployment_hash:
            raise ValueError("checkpoint targets another public Qwen deployment")
        self.synchronize_trainable_versions(
            TrainableVersions(state.forward_version, state.backward_version, state.z_version)
        )
        self._model.set_requires_grad([role.value for role in AdapterRole], True)
        self._model.eval()


class QwenPolicyBackbone(_QwenPolicyBackboneBase):
    def __init__(
        self, config: QwenBackboneConfig, *, performance: TrainingPerformanceConfig | None = None
    ) -> None:
        if type(config) is not QwenBackboneConfig:
            raise TypeError("QwenPolicyBackbone requires QwenBackboneConfig")
        super().__init__(
            config,
            performance=performance,
            model_loader=AutoModelForCausalLM.from_pretrained,
            backend_name="qwen-causal-policy-backbone@3",
        )


class QwenMultimodalPolicyBackbone(_QwenPolicyBackboneBase):
    def __init__(
        self,
        config: QwenMultimodalBackboneConfig,
        *,
        performance: TrainingPerformanceConfig | None = None,
    ) -> None:
        if not isinstance(config, QwenMultimodalBackboneConfig):
            raise TypeError("QwenMultimodalPolicyBackbone requires QwenMultimodalBackboneConfig")
        super().__init__(
            config,
            performance=performance,
            model_loader=AutoModelForMultimodalLM.from_pretrained,
            backend_name="qwen-multimodal-policy-backbone@3",
        )


def build_qwen_policy_backbone(
    config: QwenDeploymentConfig,
    *,
    performance: TrainingPerformanceConfig | None = None,
) -> QwenPolicyBackbone | QwenMultimodalPolicyBackbone:
    if performance is not None:
        performance.configure_process()
    if type(config) is QwenBackboneConfig:
        return QwenPolicyBackbone(config, performance=performance)
    if type(config) is QwenMultimodalBackboneConfig:
        return QwenMultimodalPolicyBackbone(config, performance=performance)
    raise TypeError("unsupported Qwen deployment config type")


__all__ = [
    "QwenMultimodalPolicyBackbone",
    "QwenPolicyBackbone",
    "build_qwen_policy_backbone",
]
