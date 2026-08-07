"""Ask where a residual field came from, by scanning candidate sources over a plane.

A pooled coupling is a vector over sensors: what each one reads per unit of drive
that the description does not predict.  Such a vector says a fault exists; it does
not say where.  This module turns it into a position by asking, at every point of the
poloidal plane, how much of the vector a single circular filament there would
explain.

Everything the described conductors can produce has to come out first.  A residual
lying inside the span of the response columns is not evidence of an undescribed
source -- it is evidence that some described conductor's strength is slightly wrong,
which the response can absorb in any proportion the fit likes.  Projecting the
residual off that span leaves only what no combination of described conductors
produces, and the same projection is applied to every candidate field so the two are
compared on the same footing.

The score is a squared cosine: the fraction of the surviving power a candidate
accounts for, which is scale-free and so cannot be won by a candidate that is merely
large.  The amplitude that goes with it is reported separately, because the two
answer different questions -- whether a source of that shape is there, and how much
current it would need to be carrying.

Symmetry is deliberately not imposed.  A source that is up-down symmetric should
emerge as two peaks in the map rather than being built into the scan, because a scan
that assumes the symmetry cannot be evidence for it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import ellipe, ellipk

VACUUM_PERMEABILITY = 4.0e-7 * np.pi
"""Magnetic constant, in henry per metre."""


class LocalizationError(ValueError):
    """Raised when a scan cannot be posed from what was supplied."""


@dataclass(frozen=True)
class SpanProjector:
    """An orthonormal basis for a response's column span.

    Held as a basis rather than re-derived per candidate because a scan evaluates
    thousands of candidates against one span, and the projection is the same
    subtraction every time.
    """

    basis: np.ndarray

    def residual(self, vector: np.ndarray | Sequence[float]) -> np.ndarray:
        """Return the part of a vector no column of the response can produce."""

        values = np.asarray(vector, dtype=float)
        if values.shape[0] != self.basis.shape[0]:
            raise LocalizationError(
                f"a vector of {values.shape[0]} entries against a span over "
                f"{self.basis.shape[0]} sensors"
            )
        return values - self.basis @ (self.basis.T @ values)


@dataclass(frozen=True)
class ScanResult:
    """What a single filament at each point of a grid would explain."""

    radius: np.ndarray
    height: np.ndarray
    score: np.ndarray
    current: np.ndarray

    @property
    def peak(self) -> FilamentPeak:
        """Return the best-scoring point of the grid."""

        if not np.isfinite(self.score).any():
            raise LocalizationError("no grid point scored finitely")
        index = np.unravel_index(
            int(np.nanargmax(np.where(np.isfinite(self.score), self.score, -np.inf))),
            self.score.shape,
        )
        return FilamentPeak(
            radius=float(self.radius[index[1]]),
            height=float(self.height[index[0]]),
            score=float(self.score[index]),
            current=float(self.current[index]),
        )


@dataclass(frozen=True)
class FilamentPeak:
    """One point of a scan, and what a filament there would have to carry."""

    radius: float
    height: float
    score: float
    current: float


def loop_field(
    r: np.ndarray | Sequence[float],
    z: np.ndarray | Sequence[float],
    radius: float,
    height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the field of a circular filament, in tesla per ampere.

    The closed form in complete elliptic integrals, evaluated at points ``(r, z)`` for
    a loop of the given radius sitting at the given height.  Exact for a filament,
    which is what the scan wants: a candidate source is being asked where it is, and
    giving it a cross-section would add parameters the sensors cannot separate from
    position.

    Points on the filament itself are returned as zero rather than as an infinity, so
    a grid that happens to pass through a sensor does not poison the scan around it.
    """

    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    if radius <= 0.0:
        raise LocalizationError(
            "a filament of zero or negative radius encloses nothing"
        )
    offset = z - height
    far = (radius + r) ** 2 + offset**2
    modulus = np.clip(4.0 * radius * r / far, 0.0, 1.0 - 1.0e-12)
    complete_first, complete_second = ellipk(modulus), ellipe(modulus)
    near = (radius - r) ** 2 + offset**2
    base = VACUUM_PERMEABILITY / (2.0 * np.pi * np.sqrt(far))
    with np.errstate(divide="ignore", invalid="ignore"):
        radial = (
            base
            * (offset / np.where(r == 0.0, np.inf, r))
            * (
                -complete_first
                + complete_second * (radius**2 + r**2 + offset**2) / near
            )
        )
        axial = base * (
            complete_first + complete_second * (radius**2 - r**2 - offset**2) / near
        )
    return (
        np.nan_to_num(radial, posinf=0.0, neginf=0.0),
        np.nan_to_num(axial, posinf=0.0, neginf=0.0),
    )


def axial_projection(
    radial: np.ndarray,
    axial: np.ndarray,
    radial_cosine: np.ndarray | Sequence[float],
    axial_sine: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Return the field component each sensor is sensitive to.

    A poloidal field sensor reads one projection of the field at its position, so a
    candidate must be reduced to that projection before it can be compared with what
    the sensor recorded.  The two direction cosines come from the sensor description
    and are the caller's to supply: which way a sensor points is machine knowledge.
    """

    return np.asarray(radial) * np.asarray(radial_cosine, dtype=float) + np.asarray(
        axial
    ) * np.asarray(axial_sine, dtype=float)


def span_projector(response: np.ndarray) -> SpanProjector:
    """Return an orthonormal basis for the described response's column span."""

    matrix = np.asarray(response, dtype=float)
    if matrix.ndim != 2:
        raise LocalizationError("a response span needs a two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise LocalizationError(
            "the response carries non-finite entries, so its span is not defined; "
            "restrict the sensor set before taking it"
        )
    basis, _ = np.linalg.qr(matrix)
    return SpanProjector(basis=np.asarray(basis, dtype=float))


def filament_scan(
    target: np.ndarray | Sequence[float],
    sensor_r: np.ndarray | Sequence[float],
    sensor_z: np.ndarray | Sequence[float],
    radial_cosine: np.ndarray | Sequence[float],
    axial_sine: np.ndarray | Sequence[float],
    *,
    projector: SpanProjector,
    radius: np.ndarray | Sequence[float],
    height: np.ndarray | Sequence[float],
) -> ScanResult:
    """Score a single circular filament at every point of a grid.

    ``target`` is the surviving residual -- already projected off the response span --
    over the same sensors, in the same order, as the positions and direction cosines.
    Each candidate is projected off the same span before it is compared, because a
    candidate whose field the described conductors could reproduce explains nothing
    that needed explaining.

    The score is the squared cosine between candidate and target, so it is bounded in
    ``[0, 1]`` and reads directly as the fraction of surviving power the candidate
    accounts for.  ``current`` is the amplitude that fraction is attained at, in
    amperes per unit of whatever the target is per.
    """

    residual = np.asarray(target, dtype=float)
    positions = (
        np.asarray(sensor_r, dtype=float),
        np.asarray(sensor_z, dtype=float),
    )
    if not all(axis.shape == residual.shape for axis in positions):
        raise LocalizationError(
            f"a target over {residual.size} sensors against positions of "
            f"{positions[0].size}"
        )
    if residual.shape[0] != projector.basis.shape[0]:
        raise LocalizationError(
            f"a target over {residual.shape[0]} sensors against a span over "
            f"{projector.basis.shape[0]}"
        )
    grid_r = np.asarray(radius, dtype=float)
    grid_z = np.asarray(height, dtype=float)
    target_power = float(residual @ residual)
    if target_power <= 0.0:
        raise LocalizationError(
            "the target carries no power, so nothing can explain it"
        )

    score = np.zeros((grid_z.size, grid_r.size))
    current = np.zeros((grid_z.size, grid_r.size))
    for row, z0 in enumerate(grid_z):
        for column, r0 in enumerate(grid_r):
            field = projector.residual(
                axial_projection(
                    *loop_field(positions[0], positions[1], float(r0), float(z0)),
                    radial_cosine,
                    axial_sine,
                )
            )
            power = float(field @ field)
            if power <= 0.0:
                continue
            overlap = float(field @ residual)
            score[row, column] = overlap**2 / (power * target_power)
            current[row, column] = overlap / power
    return ScanResult(radius=grid_r, height=grid_z, score=score, current=current)


def surviving_fraction(
    vector: np.ndarray | Sequence[float], residual: np.ndarray | Sequence[float]
) -> float:
    """Return the share of a vector's power that the span could not absorb."""

    full = np.asarray(vector, dtype=float)
    left = np.asarray(residual, dtype=float)
    power = float(full @ full)
    if power <= 0.0:
        raise LocalizationError("a vector with no power has no surviving share")
    return float(left @ left) / power
