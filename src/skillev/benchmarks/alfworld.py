from __future__ import annotations

from ._embodied import (
    EmbodiedCommand,
    EmbodiedEpisode,
    EmbodiedPublicItem,
    EmbodiedPublicStep,
    EmbodiedTextEnvironment,
    OrderedEmbodiedTaskProvider,
)

ALFWORLD_BENCHMARK_ID = "alfworld"
ALFWORLD_RESOURCE_ID = "alfworld"

ALFWorldCommand = EmbodiedCommand
ALFWorldEpisode = EmbodiedEpisode
ALFWorldPublicStep = EmbodiedPublicStep


class ALFWorldPublicItem(EmbodiedPublicItem):
    __slots__ = ()
    BENCHMARK_ID = ALFWORLD_BENCHMARK_ID
    RESOURCE_ID = ALFWORLD_RESOURCE_ID


class OrderedALFWorldTaskProvider(OrderedEmbodiedTaskProvider):
    __slots__ = ()
    ITEM_TYPE = ALFWorldPublicItem


class ALFWorldEnvironment(EmbodiedTextEnvironment):
    __slots__ = ()
    ITEM_TYPE = ALFWorldPublicItem


__all__ = [
    "ALFWORLD_BENCHMARK_ID",
    "ALFWORLD_RESOURCE_ID",
    "ALFWorldCommand",
    "ALFWorldEnvironment",
    "ALFWorldEpisode",
    "ALFWorldPublicItem",
    "ALFWorldPublicStep",
    "OrderedALFWorldTaskProvider",
]
