"""Solve the sensors for the currents, and say which currents they cannot see.

Every fit that regresses a measurement on a published drive channel inherits what
the drive map asserts that channel means.  An inversion does not: it takes the
described conductor positions, builds each conductor's sensor response per unit
current, and asks the measurement alone what was flowing,

    I(t) = argmin || W (G I - B(t)) ||

with ``W`` the inverse of each sensor's own measured noise floor.  Divided by a
recorded conductor current the answer returns units per recorded ampere -- exactly the
quantity a drive map asserts -- so the wiring is readable off the ratio rather than
assumed by the design.

Because the solve runs per time sample and there are more sensors than conductors,
what can be resolved is a property of ``G`` alone: of the geometry and the sensor
set, and not of any one pulse's excitation.  That is what makes
:func:`identifiability` a permanent statement about the machine rather than a
statement about a cohort, and it is why it is computed and reported before any
measurement is read.

A small singular value is not a numerical nuisance to be damped.  It names a
combination of conductors the sensors do not distinguish, and its right singular
vector says which ones.  Damping it returns a confident number for a quantity nothing
measured, so nothing here regularises: the spectrum is reported, the truncation ladder
shows how an answer moves as directions are dropped, and a ratio that only appears at
full rank is thereby visible as one the sensors barely carry.

A conductor whose current is measured and published is a known drive, not an unknown.
Solving for it alongside the others leaves the conductor-minus-its-own-neighbour
degeneracy in the design, so :func:`subtract_known_drives` removes its field from the
measurement first; solving it anyway is the arm that demonstrates why.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from nova.calibrate.gain import ScalarFit, through_origin_fit

DOMINANT_THRESHOLD = 0.15
"""Weight in a singular vector below which a conductor is not named as dominant.

A mode mixing thirteen conductors at a twentieth each names nothing; the cut keeps
the report to the conductors that actually make the direction.
"""

DOMINANT_COUNT = 4
"""Conductors named per mode, largest weight first."""

HELD_OUT_STRIDE = 3
"""Every nth sensor withheld, so the residual measures reconstruction not fit.

Withholding by stride over the sensor ordering rather than at random keeps the
held-out set reproducible and spreads it over the array instead of concentrating it
where a random draw happened to fall.
"""

MINIMUM_EXCITED_SAMPLES = 100
"""Samples a conductor must be driven over before its ratio is reported."""


class InversionError(ValueError):
    """Raised when an inversion cannot be posed from what was supplied."""


@dataclass(frozen=True)
class SpectralMode:
    """One direction of the design, and which conductors make it."""

    index: int
    singular_value: float
    relative: float
    dominant: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class Spectrum:
    """What a sensor set can and cannot separate, before any measurement is read."""

    columns: tuple[str, ...]
    singular_values: np.ndarray
    modes: tuple[SpectralMode, ...]

    @property
    def condition_number(self) -> float:
        """Return the ratio of the largest singular value to the smallest."""

        values = self.singular_values
        if values.size == 0 or values[-1] <= 0.0:
            return np.inf
        return float(values[0] / values[-1])

    def unresolved(self, relative: float) -> tuple[SpectralMode, ...]:
        """Return the modes the sensors carry below a share of the strongest."""

        return tuple(mode for mode in self.modes if mode.relative < relative)


@dataclass(frozen=True)
class CurrentSolution:
    """The currents one window of samples implies, and how well they account for it."""

    columns: tuple[str, ...]
    currents: np.ndarray
    residual: float
    signal: float
    held_out_residual: float
    held_out_signal: float
    held_out_count: int
    sensor_count: int
    sample_count: int
    truncated: Mapping[int, np.ndarray]

    def current(self, column: str) -> np.ndarray:
        """Return the solved current history of one conductor."""

        if column not in self.columns:
            raise InversionError(
                f"{column!r} is not a column of this solve; it carries "
                f"{list(self.columns)}"
            )
        return self.currents[self.columns.index(column)]


def whiten(
    design: np.ndarray, floors: Sequence[float] | Mapping[str, float], **kwargs
) -> np.ndarray:
    """Scale each sensor row by the inverse of its own measured noise floor.

    A sensor with no measured floor is given zero weight rather than unit weight: an
    unmeasured floor is not a good one, and admitting the row at unity would let the
    least-characterised sensor in the array set the solve.  Passing ``rows`` names the
    channels when the floors arrive as a mapping.
    """

    matrix = np.asarray(design, dtype=float)
    if isinstance(floors, Mapping):
        rows = kwargs.pop("rows", None)
        if rows is None:
            raise InversionError(
                "floors given as a mapping need the row channel names to be ordered "
                "against the design"
            )
        values = np.asarray(
            [float(floors.get(name, np.inf)) for name in rows], dtype=float
        )
    else:
        values = np.asarray(floors, dtype=float)
    if kwargs:
        raise InversionError(f"unexpected arguments {sorted(kwargs)}")
    if values.size != matrix.shape[0]:
        raise InversionError(
            f"{values.size} floors against a design of {matrix.shape[0]} rows"
        )
    with np.errstate(divide="ignore"):
        weight = np.where(values > 0.0, 1.0 / values, 0.0)
    return matrix * np.nan_to_num(weight, posinf=0.0)[:, None]


def identifiability(
    design: np.ndarray,
    columns: Sequence[str],
    *,
    dominant_threshold: float = DOMINANT_THRESHOLD,
    dominant_count: int = DOMINANT_COUNT,
) -> Spectrum:
    """Report the directions the sensor set resolves, strongest first.

    The design should already be whitened, because the spectrum of an unwhitened
    design measures the sensors' units as much as their geometry.
    """

    matrix = np.asarray(design, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(columns):
        raise InversionError(
            f"a design of shape {matrix.shape} does not match {len(columns)} columns"
        )
    _, values, vectors = np.linalg.svd(matrix, full_matrices=False)
    largest = float(values[0]) if values.size and values[0] > 0.0 else np.nan
    modes = []
    for index, value in enumerate(values):
        vector = vectors[index]
        order = np.argsort(-np.abs(vector))[:dominant_count]
        modes.append(
            SpectralMode(
                index=int(index),
                singular_value=float(value),
                relative=float(value / largest),
                dominant=tuple(
                    (columns[int(slot)], float(vector[int(slot)]))
                    for slot in order
                    if abs(vector[int(slot)]) > dominant_threshold
                ),
            )
        )
    return Spectrum(
        columns=tuple(columns),
        singular_values=np.asarray(values, dtype=float),
        modes=tuple(modes),
    )


def subtract_known_drives(
    observed: np.ndarray,
    design: np.ndarray,
    columns: Sequence[str],
    known: Mapping[str, np.ndarray | Sequence[float]],
) -> np.ndarray:
    """Remove the field of every conductor whose current was measured.

    A published current is a statement, and leaving its conductor in the design as an
    unknown lets the solve give it whatever current best absorbs a neighbour's error.
    Removing its field first is what publishing the current means.
    """

    values = np.asarray(observed, dtype=float)
    matrix = np.asarray(design, dtype=float)
    order = list(columns)
    for name, current in known.items():
        if name not in order:
            raise InversionError(
                f"{name!r} is a known drive with no column in the design, so its "
                "field cannot be removed from the measurement"
            )
        history = np.asarray(current, dtype=float)
        if history.size != values.shape[1]:
            raise InversionError(
                f"{name!r} supplies {history.size} samples against {values.shape[1]}"
            )
        values = values - np.outer(matrix[:, order.index(name)], history)
    return values


def solve_currents(
    design: np.ndarray,
    observed: np.ndarray,
    columns: Sequence[str],
    *,
    weights: Sequence[float] | None = None,
    held_out_stride: int = HELD_OUT_STRIDE,
    ranks: Iterable[int] = (),
) -> CurrentSolution:
    """Solve every sample of a window for the currents the sensors imply.

    ``observed`` is sensors by samples.  Weighting is applied to both the design and
    the measurement so the solve minimises a residual in units of each sensor's own
    floor, which is what makes one array's sensors comparable with another's.

    The held-out residual is the number worth reading.  A fit residual falls whenever
    a column is added, so it says how flexible the design is; withholding sensors and
    predicting them says whether the currents are real.
    """

    matrix = np.asarray(design, dtype=float)
    measured = np.asarray(observed, dtype=float)
    if matrix.shape[0] != measured.shape[0]:
        raise InversionError(
            f"a design of {matrix.shape[0]} sensors against {measured.shape[0]} rows "
            "of measurement"
        )
    if matrix.shape[1] != len(columns):
        raise InversionError(
            f"a design of {matrix.shape[1]} columns against {len(columns)} names"
        )
    if weights is not None:
        weight = np.asarray(weights, dtype=float)[:, None]
        matrix, measured = matrix * weight, measured * weight

    solution, *_ = np.linalg.lstsq(matrix, measured, rcond=None)
    residual = measured - matrix @ solution

    keep = np.arange(matrix.shape[0]) % held_out_stride != 0
    held = ~keep
    if held.any() and keep.sum() >= matrix.shape[1]:
        partial, *_ = np.linalg.lstsq(matrix[keep], measured[keep], rcond=None)
        held_residual = float(
            np.sqrt(np.mean((measured[held] - matrix[held] @ partial) ** 2))
        )
        held_signal = float(np.sqrt(np.mean(measured[held] ** 2)))
    else:
        held_residual = held_signal = np.nan

    return CurrentSolution(
        columns=tuple(columns),
        currents=np.asarray(solution, dtype=float),
        residual=float(np.sqrt(np.mean(residual**2))),
        signal=float(np.sqrt(np.mean(measured**2))),
        held_out_residual=held_residual,
        held_out_signal=held_signal,
        held_out_count=int(held.sum()),
        sensor_count=int(matrix.shape[0]),
        sample_count=int(measured.shape[1]),
        truncated=truncation_ladder(matrix, measured, ranks),
    )


def truncation_ladder(
    design: np.ndarray, observed: np.ndarray, ranks: Iterable[int]
) -> dict[int, np.ndarray]:
    """Solve the same window with a bounded number of directions kept.

    An answer that only appears once the weakest directions are admitted is being
    carried by a combination the sensors barely see.  Reporting the ladder is how that
    becomes visible without choosing a regularisation for the reader.
    """

    matrix = np.asarray(design, dtype=float)
    measured = np.asarray(observed, dtype=float)
    wanted = sorted({int(rank) for rank in ranks})
    if not wanted:
        return {}
    left, values, right = np.linalg.svd(matrix, full_matrices=False)
    ladder: dict[int, np.ndarray] = {}
    for rank in wanted:
        if rank < 1 or rank > values.size:
            continue
        if np.any(values[:rank] <= 0.0):
            continue
        inverse = right[:rank].T @ np.diag(1.0 / values[:rank]) @ left[:, :rank].T
        ladder[rank] = np.asarray(inverse @ measured, dtype=float)
    return ladder


def conductor_ratio(
    recorded: np.ndarray | Sequence[float],
    inverted: np.ndarray | Sequence[float],
    *,
    excited: float,
    minimum_samples: int = MINIMUM_EXCITED_SAMPLES,
) -> ScalarFit | None:
    """Return solved current per recorded ampere, over the samples that were driven.

    Restricted to samples where the recorded current exceeds ``excited`` because a
    ratio taken through zero is a quotient of two noise floors.  The fit is through
    the origin: a conductor carrying no current produces no field, and an intercept
    would let a neighbour's standing field be read as this conductor's wiring.

    Returns None where the conductor was not driven over enough samples to constrain
    the ratio, which is a pulse that says nothing about this conductor rather than an
    error.
    """

    current = np.asarray(recorded, dtype=float)
    solved = np.asarray(inverted, dtype=float)
    if current.shape != solved.shape:
        raise InversionError(
            f"{current.size} recorded samples against {solved.size} solved samples"
        )
    driven = np.isfinite(current) & np.isfinite(solved) & (np.abs(current) > excited)
    if int(driven.sum()) < minimum_samples:
        return None
    return through_origin_fit(current, solved, mask=driven)
