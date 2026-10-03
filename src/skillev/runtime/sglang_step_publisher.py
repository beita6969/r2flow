from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from skillev.contracts import canonical_json
from skillev.policy import PolicyBackbone

from .sglang_gateway import AdapterGeneration, PreparedAdapterSwap, SGLangGateway
from .sglang_pool import SGLangActorPool

_EXPORT_MARKER = "publication.json"


class StepAdapterPublisher(Protocol):
    def restore(
        self,
        *,
        optimizer_step: int,
        policy_snapshot_id: str,
    ) -> AdapterGeneration: ...

    def prepare(
        self,
        *,
        optimizer_step: int,
        policy_snapshot_id: str,
    ) -> object: ...

    def commit(self, prepared: object) -> AdapterGeneration: ...

    def rollback(self, prepared: object) -> None: ...


@dataclass(frozen=True, slots=True)
class PreparedStepAdapter:
    optimizer_step: int
    policy_snapshot_id: str
    export_directory: Path
    gateway_swap: PreparedAdapterSwap


@dataclass(slots=True)
class SGLangStepAdapterPublisher:
    backbone: PolicyBackbone
    gateway: SGLangGateway
    export_root: Path
    adapter_namespace: str
    keep_recent: int = 3

    def __post_init__(self) -> None:
        self.export_root = self.export_root.resolve()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", self.adapter_namespace):
            raise ValueError("adapter namespace contains unsupported characters")
        if type(self.keep_recent) is not int or self.keep_recent < 1:
            raise ValueError("adapter retention must keep at least one export")

    def prepare(
        self,
        *,
        optimizer_step: int,
        policy_snapshot_id: str,
    ) -> PreparedStepAdapter:
        if type(optimizer_step) is not int or optimizer_step < 0:
            raise ValueError("adapter publication step must be non-negative")
        if not policy_snapshot_id.strip():
            raise ValueError("adapter publication requires a policy snapshot")
        export = self._export(
            optimizer_step=optimizer_step,
            policy_snapshot_id=policy_snapshot_id,
        )
        swap = self.gateway.prepare_supervisor_adapter(
            adapter_path=str(export / "policy" / "forward_adapter"),
            adapter_revision=self._adapter_revision(optimizer_step, policy_snapshot_id),
        )
        return PreparedStepAdapter(
            optimizer_step=optimizer_step,
            policy_snapshot_id=policy_snapshot_id,
            export_directory=export,
            gateway_swap=swap,
        )

    def restore(
        self,
        *,
        optimizer_step: int,
        policy_snapshot_id: str,
    ) -> AdapterGeneration:
        if type(optimizer_step) is not int or optimizer_step < 0:
            raise ValueError("adapter publication step must be non-negative")
        if not policy_snapshot_id.strip():
            raise ValueError("adapter publication requires a policy snapshot")
        export = self._export(
            optimizer_step=optimizer_step,
            policy_snapshot_id=policy_snapshot_id,
        )
        generation = self.gateway.restore_supervisor_adapter(
            adapter_path=str(export / "policy" / "forward_adapter"),
            adapter_revision=self._adapter_revision(optimizer_step, policy_snapshot_id),
        )
        if isinstance(self.gateway, SGLangActorPool):
            self.gateway.bind_policy_snapshot(policy_snapshot_id)
        self._retain_recent()
        return generation

    def commit(self, prepared: object) -> AdapterGeneration:
        publication = self._require_prepared(prepared)
        self._retain_recent()
        generation = self.gateway.commit_supervisor_adapter(publication.gateway_swap)
        if isinstance(self.gateway, SGLangActorPool):
            self.gateway.bind_policy_snapshot(publication.policy_snapshot_id)
        return generation

    def rollback(self, prepared: object) -> None:
        publication = self._require_prepared(prepared)
        self.gateway.rollback_supervisor_adapter(publication.gateway_swap)

    def _export(self, *, optimizer_step: int, policy_snapshot_id: str) -> Path:
        self.export_root.mkdir(parents=True, exist_ok=True)
        name = f"{self.adapter_namespace}-step-{optimizer_step:08d}"
        final = self.export_root / name
        marker_value = {
            "format": "skillev-sglang-adapter-publication@1",
            "optimizer_step": optimizer_step,
            "policy_snapshot_id": policy_snapshot_id,
        }
        if final.exists():
            marker = final / _EXPORT_MARKER
            existing = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else None
            if existing != marker_value:
                raise FileExistsError("adapter export identity differs from the existing step")
            self._require_adapter_files(final)
            return final
        with tempfile.TemporaryDirectory(
            prefix=f".{name}.staging-{os.getpid()}-",
            dir=self.export_root,
        ) as staging_text:
            staging = Path(staging_text)
            self.backbone.save_checkpoint(str(staging / "policy"))
            self._require_adapter_files(staging)
            (staging / _EXPORT_MARKER).write_text(
                canonical_json(marker_value) + "\n",
                encoding="utf-8",
            )
            os.replace(staging, final)
        return final

    @staticmethod
    def _require_adapter_files(root: Path) -> None:
        adapter = root / "policy" / "forward_adapter"
        required = (adapter / "adapter_config.json", adapter / "adapter_model.safetensors")
        if any(not path.is_file() for path in required):
            raise RuntimeError("forward adapter export is incomplete")

    def _retain_recent(self) -> None:
        exports = sorted(
            path
            for path in self.export_root.iterdir()
            if path.is_dir()
            and path.name.startswith(f"{self.adapter_namespace}-step-")
            and path.name.removeprefix(f"{self.adapter_namespace}-step-").isdigit()
        )
        for path in exports[: -self.keep_recent]:
            shutil.rmtree(path)

    def _adapter_revision(self, optimizer_step: int, policy_snapshot_id: str) -> str:
        return (
            f"{self.adapter_namespace}-step-{optimizer_step:08d}-"
            f"{_snapshot_label(policy_snapshot_id)}"
        )

    @staticmethod
    def _require_prepared(prepared: object) -> PreparedStepAdapter:
        if not isinstance(prepared, PreparedStepAdapter):
            raise TypeError("adapter publication requires a prepared step")
        return prepared


def _snapshot_label(policy_snapshot_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", policy_snapshot_id).strip("-._")
    if not normalized:
        raise ValueError("policy snapshot identity has no adapter-safe characters")
    return normalized[-24:]


__all__ = [
    "PreparedStepAdapter",
    "SGLangStepAdapterPublisher",
    "StepAdapterPublisher",
]
