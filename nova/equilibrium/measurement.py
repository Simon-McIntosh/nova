"""External magnetics geometry, per-slice measurements and the whitened solve.

Every magnetics-driven reconstruction on the spine reads the same two objects —
a fixed :class:`Magnetics` geometry and one :class:`SliceMeasurement` per time
slice — and inverts through the same whitened, column-normalised least squares.
Keeping one row order for the sensor geometry and the measurement vectors is
what makes every design matrix row-aligned with the data.

All quantities are raw SI: flux loops read the total poloidal flux
:math:`\\Phi = 2 \\pi R A_\\phi` [Wb], field probes read the poloidal field [T]
projected on their own orientation.
"""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class Magnetics:
    """External magnetic diagnostic geometry, in measurement row order.

    ``angle`` is the poloidal orientation of a field probe [deg]; flux-loop rows
    ignore it, so a facility that leaves it unset can pass zeros.
    """

    r: np.ndarray
    z: np.ndarray
    angle: np.ndarray
    flux_loop: np.ndarray

    @property
    def number(self) -> int:
        """Return the sensor count."""
        return int(np.asarray(self.r).size)

    def project(self, psi: np.ndarray, br: np.ndarray, bz: np.ndarray) -> np.ndarray:
        """Return each sensor's reading of a per-ampere field: flux or field."""
        angle = np.deg2rad(np.asarray(self.angle, dtype=np.float64))
        flux_loop = np.asarray(self.flux_loop, dtype=bool)
        if psi.ndim == 2:  # a column block: one row per sensor
            angle = angle[:, None]
            flux_loop = flux_loop[:, None]
        return np.where(flux_loop, psi, br * np.cos(angle) + bz * np.sin(angle))


@dataclass(frozen=True)
class SliceMeasurement:
    """One time slice of external magnetics, in :class:`Magnetics` row order.

    ``measured`` may carry NaN on absent channels; those rows must be masked
    out. ``vacuum`` is the known-conductor prediction on the same rows, so
    ``measured - vacuum`` is the plasma signature a read sees. ``scale`` is the
    per-row measurement scale the whitening divides by, which puts flux-loop and
    field-probe rows on comparable terms.
    """

    measured: np.ndarray
    vacuum: np.ndarray
    mask: np.ndarray
    scale: np.ndarray
    plasma_current: float
    """Rogowski total plasma current [A] — a trusted absolute measurement."""

    vacuum_flux: np.ndarray | None = field(default=None, repr=False)
    """Known-conductor flux on the reconstruction grid [Wb], for a boundary push."""

    conductor_current: np.ndarray | None = field(default=None, repr=False)
    """Measured currents [A], in the fitted conductor response column order."""

    conductor_scale: np.ndarray | None = field(default=None, repr=False)
    """Positive current measurement uncertainties [A], in that same order."""

    @cached_property
    def weight(self) -> np.ndarray:
        """Return the whitening weight, zero on untrusted rows."""
        keep = np.asarray(self.mask, dtype=bool)
        weight = np.zeros(keep.size)
        weight[keep] = 1.0 / np.maximum(
            np.asarray(self.scale, dtype=np.float64)[keep], 1e-12
        )
        return weight

    @cached_property
    def signature(self) -> np.ndarray:
        """Return the plasma sensor signature ``measured - vacuum``.

        Absent channels carry NaN and are masked out by a zero weight, but
        ``NaN * 0`` would still poison the least squares, so they are zeroed
        here first.
        """
        return np.nan_to_num(
            np.asarray(self.measured, dtype=np.float64)
        ) - np.nan_to_num(np.asarray(self.vacuum, dtype=np.float64))


def whitened_solve(
    response: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    ridge: float,
    *,
    penalty: np.ndarray | None = None,
    anchor: tuple[np.ndarray, float, float] | None = None,
    diagnostics: bool = False,
    rcond: float | None = None,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Solve a whitened, column-normalised ridge least-squares system.

    Whitening by the per-row measurement scale puts every channel on comparable
    terms; normalising the columns means the ridge is a dimensionless numerical
    floor rather than a prior biasing the fit toward small amplitudes.

    ``penalty`` scales the ridge per column, which is how a graded (Sobolev)
    penalty damps high-order modes far harder than the low-order shape. ``anchor``
    is an optional extra ``(row, target, weight)`` equation — a gauge tie such as
    the poloidal-circulation current constraint — appended to the whitened system;
    with ``anchor=None`` the solve is exactly the sensor-only solve.

    Returns the coefficients and their covariance, both in the raw (un-normalised)
    coefficient frame. With ``diagnostics=True``, also return the singular values
    of the whitened, column-normalised data response and its null directions in
    the raw coefficient frame. The spectrum excludes the regulariser, which
    cannot make an unobserved direction data-determined. ``rcond`` is the
    relative singular-value cutoff used to identify those directions.
    """
    weighted = response * weight[:, None]
    rhs = target * weight
    if anchor is not None:
        row, anchor_target, anchor_weight = anchor
        anchor_row = float(anchor_weight) * np.asarray(row, dtype=np.float64)
        weighted = np.vstack([weighted, anchor_row[None, :]])
        rhs = np.concatenate([rhs, [float(anchor_weight) * float(anchor_target)]])

    column_norm = np.linalg.norm(weighted, axis=0)
    column_norm = np.where(column_norm > 0.0, column_norm, 1.0)
    normalised = weighted / column_norm[None, :]
    count = normalised.shape[1]
    if count == 0:
        result = np.zeros(0), np.zeros((0, 0))
        return (*result, np.zeros(0), np.zeros((0, 0))) if diagnostics else result
    if penalty is None:
        gram = normalised.T @ normalised + ridge * np.eye(count)
    else:
        gram = normalised.T @ normalised + np.diag(
            ridge * np.asarray(penalty, dtype=np.float64)[:count]
        )
    if ridge == 0.0:
        coefficients = np.linalg.lstsq(normalised, rhs, rcond=rcond)[0] / column_norm
    else:
        coefficients = np.linalg.solve(gram, normalised.T @ rhs) / column_norm
    try:
        covariance = np.linalg.pinv(gram) / np.outer(column_norm, column_norm)
    except np.linalg.LinAlgError:  # degenerate response
        covariance = np.full((count, count), np.nan)
    if not diagnostics:
        return coefficients, covariance
    _, spectrum, right = np.linalg.svd(normalised, full_matrices=True)
    cutoff = (
        np.finfo(normalised.dtype).eps * max(normalised.shape)
        if rcond is None
        else rcond
    )
    rank = (
        int(np.count_nonzero(spectrum > cutoff * spectrum[0])) if spectrum.size else 0
    )
    null = right[rank:] / column_norm[None, :]
    if null.size:
        null /= np.linalg.norm(null, axis=1, keepdims=True)
    return coefficients, covariance, spectrum, null
