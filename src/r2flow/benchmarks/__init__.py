from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .alfworld import (
        ALFWorldOutcomeUnavailableError,
        PrivateALFWorldCase,
        PrivateALFWorldEpisodeFactory,
        PrivateALFWorldEpisodeSession,
        PrivateALFWorldOutcomeView,
        PrivateALFWorldSessionFactory,
        PrivateALFWorldTerminalEvaluator,
    )
    from .alfworld_official import (
        OfficialALFWorldEpisodeFactory,
        OfficialALFWorldResetResult,
        OfficialALFWorldStepResult,
        OfficialALFWorldTask,
        OfficialALFWorldTextEnv,
        OfficialALFWorldTextEnvFactory,
    )
    from .official_process import (
        ALFWorldGameDeployment,
        OfficialALFWorldProcessFactory,
        OfficialEnvironmentInfrastructureError,
        PinnedOfficialProcess,
    )
    from .private_workers import (
        AsyncSubprocessWorkerTransport,
        InMemoryWorkerTransport,
        PrivateJSONWorker,
        PrivateWorkerTransport,
        WorkerError,
    )
    from .static import (
        PrivateRetrievalBenchmarkSessionFactory,
        PrivateStaticBenchmarkCase,
        PrivateStaticBenchmarkEvaluator,
        PrivateStaticBenchmarkSessionFactory,
        PrivateStaticTarget,
        StaticScoringRule,
    )

_EXPORT_MODULES: dict[str, str] = {
    "ALFWorldGameDeployment": ".official_process",
    "ALFWorldOutcomeUnavailableError": ".alfworld",
    "OfficialALFWorldEpisodeFactory": ".alfworld_official",
    "OfficialALFWorldProcessFactory": ".official_process",
    "OfficialALFWorldResetResult": ".alfworld_official",
    "OfficialALFWorldStepResult": ".alfworld_official",
    "OfficialALFWorldTask": ".alfworld_official",
    "OfficialALFWorldTextEnv": ".alfworld_official",
    "OfficialALFWorldTextEnvFactory": ".alfworld_official",
    "OfficialEnvironmentInfrastructureError": ".official_process",
    "PinnedOfficialProcess": ".official_process",
    "AsyncSubprocessWorkerTransport": ".private_workers",
    "InMemoryWorkerTransport": ".private_workers",
    "PrivateJSONWorker": ".private_workers",
    "PrivateWorkerTransport": ".private_workers",
    "WorkerError": ".private_workers",
    "PrivateALFWorldCase": ".alfworld",
    "PrivateALFWorldEpisodeFactory": ".alfworld",
    "PrivateALFWorldEpisodeSession": ".alfworld",
    "PrivateALFWorldOutcomeView": ".alfworld",
    "PrivateALFWorldSessionFactory": ".alfworld",
    "PrivateALFWorldTerminalEvaluator": ".alfworld",
    "PrivateRetrievalBenchmarkSessionFactory": ".static",
    "PrivateStaticBenchmarkCase": ".static",
    "PrivateStaticBenchmarkEvaluator": ".static",
    "PrivateStaticBenchmarkSessionFactory": ".static",
    "PrivateStaticTarget": ".static",
    "StaticScoringRule": ".static",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "ALFWorldGameDeployment",
    "ALFWorldOutcomeUnavailableError",
    "AsyncSubprocessWorkerTransport",
    "InMemoryWorkerTransport",
    "OfficialALFWorldEpisodeFactory",
    "OfficialALFWorldProcessFactory",
    "OfficialALFWorldResetResult",
    "OfficialALFWorldStepResult",
    "OfficialALFWorldTask",
    "OfficialALFWorldTextEnv",
    "OfficialALFWorldTextEnvFactory",
    "OfficialEnvironmentInfrastructureError",
    "PinnedOfficialProcess",
    "PrivateALFWorldCase",
    "PrivateALFWorldEpisodeFactory",
    "PrivateALFWorldEpisodeSession",
    "PrivateALFWorldOutcomeView",
    "PrivateALFWorldSessionFactory",
    "PrivateALFWorldTerminalEvaluator",
    "PrivateJSONWorker",
    "PrivateRetrievalBenchmarkSessionFactory",
    "PrivateStaticBenchmarkCase",
    "PrivateStaticBenchmarkEvaluator",
    "PrivateStaticBenchmarkSessionFactory",
    "PrivateStaticTarget",
    "PrivateWorkerTransport",
    "WorkerError",
    "StaticScoringRule",
]
