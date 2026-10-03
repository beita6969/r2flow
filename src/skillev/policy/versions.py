from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interface import PolicyBackbone


@dataclass(frozen=True, slots=True)
class TrainableVersions:
    forward: str
    backward: str
    z: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.forward, self.backward, self.z)
        ):
            raise ValueError("trainable component versions must be nonempty text")

    @classmethod
    def from_backbone(cls, backbone: PolicyBackbone) -> TrainableVersions:
        from .interface import AdapterRole

        return cls(
            backbone.adapter_version(AdapterRole.FORWARD_POLICY),
            backbone.adapter_version(AdapterRole.BACKWARD_POLICY),
            backbone.z_version,
        )
