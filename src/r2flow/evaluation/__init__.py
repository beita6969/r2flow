from __future__ import annotations

from importlib import import_module

_MODULES: dict[str, tuple[str, ...]] = {
    "native_metrics": (
        "NativeMetricValue",
        "public_native_metric_values",
    ),
}
_EXPORTS = {
    name: (f"r2flow.evaluation.{module}", name)
    for module, names in _MODULES.items()
    for name in names
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> object:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
