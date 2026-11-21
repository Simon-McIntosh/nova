"""The arithmetic that turns a fitted conductor error into an applied current.

Both the driver that builds an arm and the summarizer that reads it need the
same three quantities: the measured ampere-turns a frame carries, the fitted
equivalent error per conductor, and the current the solve was actually handed.
Keeping that arithmetic in one place makes the two agree by construction, and
lets it be checked against a hand-computed case without a solve.
"""

from __future__ import annotations

from typing import Iterable, Mapping


def current_at(source_time, column, time_ms: float, applied_multiplier: float) -> float:
    """Interpolate one corpus column to a time, in ampere-turns.

    The column is in kiloamps, so the interpolation is scaled by a thousand and
    by the conductor's applied turn multiplier.
    """

    import numpy as np

    source_time = np.asarray(source_time, dtype=float)
    column = np.asarray(column, dtype=float)
    valid = np.isfinite(source_time) & np.isfinite(column)
    if np.count_nonzero(valid) < 2:
        raise ValueError("a corpus current column needs two finite samples")
    kiloamp = np.interp(time_ms, source_time[valid], column[valid])
    return 1000.0 * float(kiloamp) * float(applied_multiplier)


def equivalent_error(names: Iterable[str], correction: Mapping[str, float]) -> list:
    """The fitted equivalent error per conductor, in ampere-turns.

    A conductor the fit does not name carries no error, so an incomplete fit
    leaves the rest of the vector exactly as measured rather than dropping it.
    """

    return [float(correction.get(name, 0.0)) for name in names]


def applied_currents(
    names: Iterable[str], measured: Iterable[float], correction: Mapping[str, float]
) -> list:
    """The measured current with the fitted equivalent error removed."""

    names = list(names)
    measured = [float(item) for item in measured]
    error = equivalent_error(names, correction)
    if len(measured) != len(names):
        raise ValueError("measured must carry one current per conductor")
    return [item - amount for item, amount in zip(measured, error)]


def column_shift(error_a_turn: float, applied_multiplier: float) -> float:
    """The corpus-column shift, in the column's own kiloamp unit.

    A corpus column is read as kiloamps and scaled by the conductor's applied
    turn multiplier to reach ampere-turns, so removing an ampere-turn error
    shifts the column by the negated error over ``1000 * multiplier``.
    """

    if applied_multiplier == 0.0:
        raise ValueError("a conductor with no applied multiplier carries no column")
    return -float(error_a_turn) / (1000.0 * float(applied_multiplier))


def relative_correction(measured: Iterable[float], error: Iterable[float]) -> list:
    """Each conductor's error as a fraction of its own measured current.

    A conductor carrying no measured current has no meaningful fraction, and a
    division there would report an infinite correction, so it is reported as
    zero, and the caller reads the ampere-turns beside it instead.
    """

    measured = [float(item) for item in measured]
    error = [float(item) for item in error]
    if len(measured) != len(error):
        raise ValueError("measured and error must have one entry per conductor")
    return [
        0.0 if item == 0.0 else amount / item for item, amount in zip(measured, error)
    ]
