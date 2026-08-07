"""Fit what a channel reads against what it should read, and pool the answer.

A gain is one number, which makes it the easiest quantity in the calibration ladder
to produce and the easiest to produce wrongly.  The fits here are all least squares
through the origin -- a scale, with no intercept, because an intercept is an offset
and the offset has its own kind and its own stage in the read path.  What separates
them is what they guard against.

A scale only means something when the thing being scaled has the right shape.  The
best scale onto a waveform of the wrong shape is a projection, and reporting it as a
gain writes a shape error into a number a consumer reads as an amplitude.  Every
scalar fit therefore returns the variance it explains beside the slope, and the
screens that use it refuse a fit that leaves most of the channel's variance behind.

A scale fitted where several drives contribute is not that drive's scale.  A drive
must carry a stated share of the predicted power before its scale is recorded, and
the prediction itself must retain a stated share of the summed power of its parts:
two drives cancelling at a channel make the prediction small and every drive's own
power a large multiple of it, so a leverage test alone admits exactly the pulse where
the scale is a ratio of two nearly cancelling numbers.

A pulse is one measurement, not ten thousand.  A waveform's samples are correlated,
so a standard error taken from the sample count shrinks by two orders of magnitude
and makes every channel look significantly different from every other.  Every
quantity is estimated per pulse first and pooled across pulses, and its error is the
scatter of the per-pulse values.

Scale and orientation cannot be separated within one pulse that drove one circuit.
A channel sharing its position with one measuring the other component can be fitted
for both at once, but the two columns carry the same waveform on such a pulse and any
scale trades against any angle for the same prediction.  The fit is therefore carried
as normal equations and solved over every pulse at once, where different circuits
present different ratios of the two components and the columns stop being collinear.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

MINIMUM_SAMPLES = 200
"""Samples a pulse must give a channel before its scale is estimated."""

MINIMUM_FITS = 3
"""Pulses a pooled scale needs before it carries a standard error.

Two values have a spread that is one number's distance from another, which is not a
scatter, and reporting it as one invites a consumer to read it as a bound.
"""

MINIMUM_LEVERAGE = 0.5
"""Share of a channel's predicted power one drive must carry to own the scale."""

MINIMUM_COHERENCE = 0.5
"""Share of the drives' summed power the prediction itself must retain.

Below this the drives are cancelling at the channel, so each one's power is a large
multiple of what the channel actually sees and the leverage test passes for the wrong
reason.
"""

MINIMUM_SHAPE_AGREEMENT = 0.5
"""Variance of the channel the fitted scale must explain to be called a gain."""

MAXIMUM_COLLINEARITY = 0.95
"""Correlation between two joint-fit columns that still separates them."""

MAXIMUM_CONDITION = 1.0e4
"""Condition number past which a pooled two-parameter solve is not read.

Four orders is where the smaller singular value stops carrying more signal than
double-precision noise on the larger one.
"""


class GainError(ValueError):
    """Raised when a scale cannot be estimated from what was supplied."""


@dataclass(frozen=True)
class ScalarFit:
    """A slope through the origin and how much of the target it accounts for."""

    slope: float
    variance_explained: float
    residual: float
    signal: float
    sample_count: int

    @property
    def agrees(self) -> bool:
        """Return whether the scaled shape accounts for most of the target."""

        return self.variance_explained >= MINIMUM_SHAPE_AGREEMENT


@dataclass(frozen=True)
class LineFit:
    """A straight line through a window, and the scatter left about it."""

    slope: float
    intercept: float
    scatter: float
    sample_count: int


@dataclass(frozen=True)
class DriveGain:
    """One channel's scale for one drive on one pulse."""

    channel: str
    drive: str
    slope: float
    leverage: float
    shape_agreement: float
    residual: float
    signal: float
    sample_count: int


@dataclass(frozen=True)
class PooledGain:
    """One channel's scale pooled over the pulses that measured it."""

    channel: str
    drive: str
    slope: float
    standard_error: float
    fit_count: int

    @property
    def identified(self) -> bool:
        """Return whether enough pulses back the scale to bound it."""

        return self.fit_count >= MINIMUM_FITS


@dataclass(frozen=True)
class NormalSystem:
    """One pulse's contribution to a joint scale-and-orientation solve.

    Carried as normal equations rather than as a fitted pair because one pulse
    usually cannot separate the two parameters and a set of pulses can.
    """

    channel: str
    partner: str
    gram: tuple[float, float, float]
    moment: tuple[float, float]
    observed_square: float
    sample_count: int

    @property
    def matrix(self) -> np.ndarray:
        """Return the symmetric two-by-two normal matrix."""

        first, cross, second = self.gram
        return np.asarray([[first, cross], [cross, second]], dtype=float)


@dataclass(frozen=True)
class AxisFit:
    """A channel's scale and orientation from one pulse, and whether it separated."""

    channel: str
    partner: str
    gain: float
    tilt: float
    residual: float
    signal: float
    collinearity: float
    sample_count: int

    @property
    def separable(self) -> bool:
        """Return whether this pulse alone can divide scale from orientation."""

        return abs(self.collinearity) < MAXIMUM_COLLINEARITY


@dataclass(frozen=True)
class PooledAxisFit:
    """A channel's scale and orientation solved over every pulse at once.

    ``condition`` is the honest statement of whether the pulses separated the two
    parameters at all.  The errors are a jackknife over pulses -- each left out in
    turn -- which is what a standard error means when the samples inside a pulse are
    correlated.
    """

    channel: str
    partner: str
    gain: float
    tilt: float
    gain_error: float
    tilt_error: float
    condition: float
    residual: float
    signal: float
    fit_count: int

    @property
    def identified(self) -> bool:
        """Return whether the pooled solve separated scale from orientation."""

        return (
            self.fit_count >= MINIMUM_FITS
            and math.isfinite(self.condition)
            and self.condition < MAXIMUM_CONDITION
        )


def baseline_offset(
    signal: np.ndarray | Sequence[float],
    mask: np.ndarray | Sequence[bool],
) -> float:
    """Return a channel's standing level, measured over a quiet window.

    The window is the caller's to choose and is normally the span before anything was
    driven.  Only the finite samples inside it count, because a channel with gaps in
    its quiet window still has a zero and dropping the channel over them would leave
    the zero unmeasurable on exactly the pulses whose excitation is cleanest.
    """

    values = np.asarray(signal, dtype=float)
    quiet = np.asarray(mask, dtype=bool) & np.isfinite(values)
    if not quiet.any():
        raise GainError(
            "the quiet window holds no finite sample, so the channel's standing level "
            "is not measurable on this pulse"
        )
    return float(np.mean(values[quiet]))


def through_origin_fit(
    predictor: np.ndarray | Sequence[float],
    target: np.ndarray | Sequence[float],
    *,
    mask: np.ndarray | Sequence[bool] | None = None,
) -> ScalarFit:
    """Return the slope of the target on the predictor through the origin.

    Through the origin because the quantity wanted is a scale.  A fitted intercept
    would absorb whatever standing offset survives the baseline subtraction and hide
    it inside a number reported as a gain, and the offset has its own correction kind
    precisely so that it does not end up there.

    ``variance_explained`` is measured against the target's own variance about its
    mean, so a slope that reproduces the target's shape scores near one however large
    the scale is.
    """

    x = np.asarray(predictor, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.shape != y.shape:
        raise GainError(f"{x.size} predictor samples against {y.size} target samples")
    keep = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    x, y = x[keep], y[keep]
    power = float(x @ x)
    if power <= 0.0:
        raise GainError(
            "the predictor carries no power over the admitted samples, so a scale on "
            "it is not defined"
        )
    slope = float(x @ y / power)
    residual = y - slope * x
    total = float(np.sum((y - y.mean()) ** 2))
    return ScalarFit(
        slope=slope,
        variance_explained=(
            math.nan
            if total <= 0.0
            else float(1.0 - float(residual @ residual) / total)
        ),
        residual=float(np.sqrt(np.mean(residual**2))),
        signal=float(np.sqrt(np.mean(y**2))),
        sample_count=int(y.size),
    )


def drift_fit(
    time: np.ndarray | Sequence[float],
    signal: np.ndarray | Sequence[float],
    *,
    mask: np.ndarray | Sequence[bool] | None = None,
) -> LineFit:
    """Fit a straight line through a window and return the scatter about it.

    An integrator's zero moves, so a channel's noise floor taken about the window's
    mean measures the drift as if it were noise.  Taking it about a fitted line
    separates the two: the slope is the drift rate the read path removes as a ramp,
    and the scatter is what is left once it is gone.
    """

    axis = np.asarray(time, dtype=float)
    values = np.asarray(signal, dtype=float)
    if axis.shape != values.shape:
        raise GainError(f"{axis.size} times against {values.size} samples")
    keep = np.isfinite(axis) & np.isfinite(values)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    if int(keep.sum()) < 2:
        raise GainError("a line needs two finite samples")
    slope, intercept = np.polyfit(axis[keep], values[keep], 1)
    residual = values[keep] - (slope * axis[keep] + intercept)
    return LineFit(
        slope=float(slope),
        intercept=float(intercept),
        scatter=float(np.std(residual)),
        sample_count=int(keep.sum()),
    )


def drive_gains(
    observed: np.ndarray | Sequence[float],
    drive: np.ndarray,
    response: np.ndarray | Sequence[float],
    names: Sequence[str],
    *,
    channel: str = "",
    mask: np.ndarray | Sequence[bool] | None = None,
    minimum_samples: int = MINIMUM_SAMPLES,
    minimum_leverage: float = MINIMUM_LEVERAGE,
    minimum_coherence: float = MINIMUM_COHERENCE,
    minimum_shape_agreement: float = MINIMUM_SHAPE_AGREEMENT,
) -> tuple[DriveGain, ...]:
    """Return the scale each drive that dominates this pulse gives one channel.

    ``observed`` is baseline-free; ``response`` is the channel's row of the forward
    model, one entry per drive column.  A drive is reported only where it carries
    ``minimum_leverage`` of the predicted power, where the prediction retains
    ``minimum_coherence`` of its parts' summed power, and where the fitted scale
    explains ``minimum_shape_agreement`` of the channel's variance.  A pulse failing
    the coherence test reports nothing at all, because there the failure is a property
    of the pulse rather than of any one drive.
    """

    y_all = np.asarray(observed, dtype=float)
    columns = np.asarray(drive, dtype=float) * np.asarray(response, dtype=float)
    if columns.shape[1] != len(names):
        raise GainError(f"{columns.shape[1]} drive columns against {len(names)} names")
    prediction = columns.sum(axis=1)
    keep = np.isfinite(y_all) & np.isfinite(prediction)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    if int(keep.sum()) < minimum_samples:
        return ()
    y = y_all[keep]
    observed_square = float(y @ y)
    total = float(prediction[keep] @ prediction[keep])
    parts = float(np.sum(columns[keep, :] ** 2))
    if total <= 0.0 or observed_square <= 0.0:
        return ()
    if parts > 0.0 and total < minimum_coherence * parts:
        return ()

    gains = []
    for index, name in enumerate(names):
        partial = columns[keep, index]
        power = float(partial @ partial)
        if power <= 0.0 or power / total < minimum_leverage:
            continue
        slope = float(partial @ y / power)
        residual = y - slope * partial
        explained = 1.0 - float(residual @ residual) / observed_square
        if explained < minimum_shape_agreement:
            continue
        gains.append(
            DriveGain(
                channel=channel,
                drive=name,
                slope=slope,
                leverage=power / total,
                shape_agreement=float(explained),
                residual=float(np.sqrt(np.mean(residual**2))),
                signal=float(np.sqrt(np.mean(y**2))),
                sample_count=int(y.size),
            )
        )
    return tuple(gains)


def pool_scalar_gains(
    values: Sequence[float],
    *,
    channel: str = "",
    drive: str = "",
) -> PooledGain:
    """Pool per-pulse scales into one scale and the standard error of its mean.

    The mean is unweighted.  Weighting by each pulse's own sample count or residual
    would let one long or one quiet pulse decide the answer, and the spread across
    pulses is the very thing the error is supposed to measure.
    """

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return PooledGain(channel, drive, math.nan, math.inf, 0)
    if array.size == 1:
        return PooledGain(channel, drive, float(array[0]), math.inf, 1)
    return PooledGain(
        channel=channel,
        drive=drive,
        slope=float(array.mean()),
        standard_error=float(array.std(ddof=1) / math.sqrt(array.size)),
        fit_count=int(array.size),
    )


def axis_fit(
    observed: np.ndarray | Sequence[float],
    prediction: np.ndarray | Sequence[float],
    partner: np.ndarray | Sequence[float],
    *,
    channel: str = "",
    partner_channel: str = "",
    mask: np.ndarray | Sequence[bool] | None = None,
    minimum_samples: int = MINIMUM_SAMPLES,
) -> tuple[AxisFit, NormalSystem] | None:
    """Fit a channel's scale and orientation jointly against a co-located channel.

    The orientation term is fitted against a *measurement* rather than against the
    model.  Where a channel shares its position with one measuring the other
    component,

        observed = a * predicted_own_component + b * measured_other_component

    recovers ``gain = hypot(a, b)`` and ``tilt = atan2(b, a)`` while reading the other
    component off the instrument that measures it.  Fitting the second column from the
    model instead would let a misdescribed field be absorbed into the angle, which is
    the error the angle is meant to be separated from.

    Returns the pulse's own pair beside the normal equations it contributes, because
    on a pulse that drove one circuit the two columns carry the same waveform and only
    the pooled solve can divide them.
    """

    y_all = np.asarray(observed, dtype=float)
    own = np.asarray(prediction, dtype=float)
    other = np.asarray(partner, dtype=float)
    keep = np.isfinite(y_all) & np.isfinite(own) & np.isfinite(other)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    if int(keep.sum()) < minimum_samples:
        return None
    y = y_all[keep]
    first, second = own[keep], other[keep]
    design = np.column_stack([first, second])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    axial, cross = float(coefficients[0]), float(coefficients[1])
    joint = y - design @ coefficients
    fit = AxisFit(
        channel=channel,
        partner=partner_channel,
        gain=float(math.hypot(axial, cross)),
        tilt=float(math.atan2(cross, axial)),
        residual=float(np.sqrt(np.mean(joint**2))),
        signal=float(np.sqrt(np.mean(y**2))),
        collinearity=_collinearity(first, second),
        sample_count=int(y.size),
    )
    system = NormalSystem(
        channel=channel,
        partner=partner_channel,
        gram=(float(first @ first), float(first @ second), float(second @ second)),
        moment=(float(first @ y), float(second @ y)),
        observed_square=float(y @ y),
        sample_count=int(y.size),
    )
    return fit, system


def _collinearity(first: np.ndarray, second: np.ndarray) -> float:
    """Return the correlation of two design columns, zero where one is constant."""

    if np.std(first) <= 0.0 or np.std(second) <= 0.0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _solve(matrix: np.ndarray, moment: np.ndarray) -> tuple[float, float] | None:
    """Return the two-parameter solution, or None where the matrix is singular."""

    try:
        solution = np.linalg.solve(matrix, moment)
    except np.linalg.LinAlgError:
        return None
    return float(solution[0]), float(solution[1])


def pool_normal_systems(systems: Iterable[NormalSystem]) -> tuple[PooledAxisFit, ...]:
    """Solve each channel's scale and orientation over every pulse that constrains it.

    Summing normal equations is what makes the pooled solve possible: each drive
    presents the channel with a different ratio of the two components, so columns that
    were collinear within one pulse stop being collinear once pulses driving different
    circuits are added.  The condition number of the summed matrix is reported rather
    than being used to regularise, because a solve the pulses did not separate should
    come back visibly unseparated and not quietly damped.
    """

    grouped: dict[str, list[NormalSystem]] = {}
    for row in systems:
        grouped.setdefault(row.channel, []).append(row)

    fits = []
    for channel, rows in sorted(grouped.items()):
        matrix = sum((row.matrix for row in rows), np.zeros((2, 2)))
        moment = np.asarray(
            [sum(row.moment[0] for row in rows), sum(row.moment[1] for row in rows)],
            dtype=float,
        )
        observed = float(sum(row.observed_square for row in rows))
        samples = int(sum(row.sample_count for row in rows))
        solution = _solve(matrix, moment)
        if solution is None or samples == 0:
            continue
        coefficients = np.asarray(solution, dtype=float)
        residual_square = max(
            0.0,
            observed
            - 2.0 * float(coefficients @ moment)
            + float(coefficients @ matrix @ coefficients),
        )
        gain_error, tilt_error = _jackknife(rows)
        fits.append(
            PooledAxisFit(
                channel=channel,
                partner=rows[0].partner,
                gain=float(math.hypot(*solution)),
                tilt=float(math.atan2(solution[1], solution[0])),
                gain_error=gain_error,
                tilt_error=tilt_error,
                condition=float(np.linalg.cond(matrix)),
                residual=float(math.sqrt(residual_square / samples)),
                signal=float(math.sqrt(observed / samples)),
                fit_count=len(rows),
            )
        )
    return tuple(fits)


def _jackknife(rows: Sequence[NormalSystem]) -> tuple[float, float]:
    """Return the leave-one-pulse-out spread of the scale and the orientation."""

    if len(rows) < 2:
        return math.inf, math.inf
    pairs = []
    for index in range(len(rows)):
        kept = list(rows[:index]) + list(rows[index + 1 :])
        matrix = sum((row.matrix for row in kept), np.zeros((2, 2)))
        moment = np.asarray(
            [sum(row.moment[0] for row in kept), sum(row.moment[1] for row in kept)],
            dtype=float,
        )
        solution = _solve(matrix, moment)
        if solution is not None:
            pairs.append((math.hypot(*solution), math.atan2(solution[1], solution[0])))
    if not pairs:
        return math.inf, math.inf
    scale = math.sqrt(len(rows) - 1)
    gains = np.asarray([row[0] for row in pairs], dtype=float)
    tilts = np.asarray([row[1] for row in pairs], dtype=float)
    return float(gains.std() * scale), float(tilts.std() * scale)


def score_axis_correction(
    observed: np.ndarray | Sequence[float],
    prediction: np.ndarray | Sequence[float],
    partner: np.ndarray | Sequence[float],
    correction: tuple[float, float],
    *,
    mask: np.ndarray | Sequence[bool] | None = None,
) -> tuple[float, float]:
    """Score a scale-and-orientation pair the fit never saw, against doing nothing.

    Returns the corrected residual beside the uncorrected one.  The pair comes in
    rather than being fitted here, so what the two numbers compare is a prediction
    against the description as it stands -- the only form in which a calibration can be
    challenged rather than merely reported.
    """

    y_all = np.asarray(observed, dtype=float)
    own = np.asarray(prediction, dtype=float)
    other = np.asarray(partner, dtype=float)
    keep = np.isfinite(y_all) & np.isfinite(own) & np.isfinite(other)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    if not keep.any():
        raise GainError("no admitted sample is finite on all three inputs")
    gain, tilt = correction
    reference = y_all[keep] - own[keep]
    corrected = y_all[keep] - (
        gain * math.cos(tilt) * own[keep] + gain * math.sin(tilt) * other[keep]
    )
    return (
        float(np.sqrt(np.mean(corrected**2))),
        float(np.sqrt(np.mean(reference**2))),
    )
