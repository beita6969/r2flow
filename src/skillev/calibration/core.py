from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from skillev.contracts import JsonValue

CALIBRATION_FORMAT: Final = "skillev-calibration@3"


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    alpha_0: float = 1.0
    beta_0: float = 1.0
    default_k: float = 1.0
    format: str = CALIBRATION_FORMAT

    def __post_init__(self) -> None:
        if self.format != CALIBRATION_FORMAT:
            raise ValueError(f"CalibrationConfig.format must be {CALIBRATION_FORMAT!r}")
        alpha_0 = _finite_number(self.alpha_0, field="CalibrationConfig.alpha_0")
        beta_0 = _finite_number(self.beta_0, field="CalibrationConfig.beta_0")
        default_k = _finite_number(
            self.default_k,
            field="CalibrationConfig.default_k",
        )
        if alpha_0 <= 0.0 or beta_0 <= 0.0:
            raise ValueError("CalibrationConfig priors must be positive")
        if default_k < 0.0:
            raise ValueError("CalibrationConfig.default_k must be non-negative")
        object.__setattr__(self, "alpha_0", alpha_0)
        object.__setattr__(self, "beta_0", beta_0)
        object.__setattr__(self, "default_k", default_k)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "alpha_0": self.alpha_0,
            "beta_0": self.beta_0,
            "default_k": self.default_k,
            "format": self.format,
        }

    @classmethod
    def from_value(cls, value: object) -> CalibrationConfig:
        if not isinstance(value, Mapping) or set(value) != {
            "alpha_0",
            "beta_0",
            "default_k",
            "format",
        }:
            raise ValueError("CalibrationConfig has incompatible fields")
        if value["format"] != CALIBRATION_FORMAT:
            raise ValueError("CalibrationConfig has an incompatible format")
        numeric: dict[str, float] = {}
        for name in ("alpha_0", "beta_0", "default_k"):
            item = value[name]
            if isinstance(item, bool) or not isinstance(item, int | float):
                raise ValueError(f"CalibrationConfig.{name} must be numeric")
            numeric[name] = float(item)
        return cls(
            alpha_0=numeric["alpha_0"],
            beta_0=numeric["beta_0"],
            default_k=numeric["default_k"],
        )


__all__ = ["CALIBRATION_FORMAT", "CalibrationConfig"]
