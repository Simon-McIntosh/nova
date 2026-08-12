"""Partial regression after removing shared control waveforms.

The Frisch--Waugh--Lovell construction projects both a target and a candidate
regressor out of the same control span, then performs a through-origin fit on
the two residuals.  This is useful when the candidate waveform is correlated
with level drives whose direct response must not be credited to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class PartialRegressionError(ValueError):
    """Raised when a partial coefficient is not identified by the arrays."""


@dataclass(frozen=True)
class PartialFit:
    """A partial slope and the residual target variance it explains."""

    slope: float
    variance_explained: float
    residual: float
    signal: float
    regressor_power: float
    sample_count: int
    control_rank: int


def _vectors(
    target: np.ndarray,
    regressor: np.ndarray,
    controls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.asarray(target, dtype=float)
    regressor = np.asarray(regressor, dtype=float)
    controls = np.asarray(controls, dtype=float)
    if target.ndim != 1 or regressor.ndim != 1:
        raise PartialRegressionError("target and regressor must be one-dimensional")
    if controls.ndim == 1:
        controls = controls[:, None]
    if controls.ndim != 2:
        raise PartialRegressionError("controls must be a samples-by-controls array")
    if target.shape != regressor.shape or controls.shape[0] != target.size:
        raise PartialRegressionError(
            "target, regressor and controls must share samples"
        )
    finite = np.isfinite(target) & np.isfinite(regressor)
    if controls.shape[1]:
        finite &= np.all(np.isfinite(controls), axis=1)
    return target[finite], regressor[finite], controls[finite]


def residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Return ``values`` after projecting out the columns of ``controls``."""

    values = np.asarray(values, dtype=float)
    controls = np.asarray(controls, dtype=float)
    if controls.ndim == 1:
        controls = controls[:, None]
    if values.ndim != 1 or controls.ndim != 2 or controls.shape[0] != values.size:
        raise PartialRegressionError("values and controls must share one sample axis")
    if controls.shape[1] == 0:
        return values.copy()
    coefficients, _, _, _ = np.linalg.lstsq(controls, values, rcond=None)
    return values - controls @ coefficients


def partial_fit(
    target: np.ndarray,
    regressor: np.ndarray,
    controls: np.ndarray,
    *,
    intercept: bool = True,
) -> PartialFit:
    """Fit a regressor after partialling the same controls from both arrays.

    Non-finite samples are removed jointly.  An intercept is included in the
    control span by default so a standing offset cannot become a differential
    coefficient.
    """

    target, regressor, controls = _vectors(target, regressor, controls)
    if intercept:
        controls = np.column_stack((np.ones(target.size), controls))
    rank = int(np.linalg.matrix_rank(controls)) if controls.shape[1] else 0
    if target.size <= rank + 1:
        raise PartialRegressionError("too few independent samples after controls")
    residual_target = residualize(target, controls)
    residual_regressor = residualize(regressor, controls)
    power = float(residual_regressor @ residual_regressor)
    if not np.isfinite(power) or power <= np.finfo(float).eps:
        raise PartialRegressionError("regressor has no variation outside the controls")
    slope = float(residual_regressor @ residual_target) / power
    error = residual_target - slope * residual_regressor
    signal = float(residual_target @ residual_target)
    residual = float(error @ error)
    explained = 1.0 - residual / signal if signal > 0.0 else 0.0
    return PartialFit(
        slope=slope,
        variance_explained=float(np.clip(explained, 0.0, 1.0)),
        residual=float(np.sqrt(residual / target.size)),
        signal=float(np.sqrt(signal / target.size)),
        regressor_power=power,
        sample_count=target.size,
        control_rank=rank,
    )
