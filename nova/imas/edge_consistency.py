"""Edge consistency between a state's flux functions and its flux map.

A reference equilibrium is only usable at the edge if the current its flux
functions describe agrees with the current its flux map implies.  This module
measures that gap in three forms on one structured grid:

``whole_slice_relative``
    ``RMS(j_phi_profile - j_phi_grid) / max|j_phi_grid|`` over every scored
    cell of the slice.  This is the figure the MAST parity chain already
    reports, so a row here reads directly against a parity-chain row.

``band_absolute``
    ``RMS(j_phi_profile - j_phi_grid)`` in A/m^2 over the cells of the edge
    band, a fixed normalised-flux width inside the boundary.

``band_relative``
    ``RMS((j_phi_profile - j_phi_grid) / j_phi_profile)`` over the same band
    cells.  This is the edge criterion.  The band forms never borrow the
    slice-wide denominator, so a well-conditioned slice cannot flatter them.

Conventions
-----------
The flux map is the total poloidal flux ``Phi = 2 pi psi`` in Wb, the quantity
nova's internal COCOS 17 chain carries and the one ``TOTAL_FLUX_FACTOR``
carries it between.  ``j_phi_grid`` applies the same structured-grid
Grad--Shafranov operator and the same factor the parity chain uses, so the two
readings are commensurate by construction.  ``j_phi_profile`` is formed by
:func:`nova.equilibrium.convention.toroidal_current_density`, the single
statement of the COCOS 17 sign chain, from the flux functions and the local
major radius.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import numpy as np

from nova.equilibrium.convention import toroidal_current_density
from nova.equilibrium.map_extraction import apply_delta_star
from nova.jax.config import configure_dtypes

__all__ = [
    "EDGE_BAND_RELATIVE_THRESHOLD",
    "EDGE_BAND_WIDTH",
    "EdgeConsistency",
    "edge_consistency",
    "grid_current_density",
]

#: Width of the edge band in normalised poloidal flux, measured inwards from
#: the boundary: the band is the set of cells with
#: ``1 - EDGE_BAND_WIDTH <= psi_norm <= 1``.  A tenth of the normalised flux
#: span is wide enough to hold hundreds of cells on a 65 by 65 IDS grid and
#: narrow enough that the band is the edge rather than the plasma.
EDGE_BAND_WIDTH = 0.10

#: Bound on :attr:`EdgeConsistency.band_relative` for a state whose flux
#: functions and flux map are mutually consistent.
#:
#: The basis is the analytic lower-single-null state measured at the 65 by 65
#: grid the IDS profile reads carry.  Its own second-order discretisation
#: error sets the floor at ``5.80e-4`` on that grid, and a one per cent
#: perturbation of FF' over the whole plasma moves the band residual to
#: ``2.33e-3``.  The bound sits about 1.7 times above the floor and about 2.3
#: times below the perturbation, so it separates a discretised-but-consistent
#: state from a state whose flux functions and flux map disagree.  It is fixed
#: before any IDS row is read and is deliberately not a scorecard field.
EDGE_BAND_RELATIVE_THRESHOLD = 1.0e-3


@dataclass(frozen=True)
class EdgeConsistency:
    """The three forms of the flux-function/flux-map residual.

    ``band_cell_count`` is the number of cells the band forms scored and
    ``scored_cell_count`` the number the whole-slice form scored.  A form
    whose cell set is empty is ``nan`` rather than a misleading zero.
    """

    whole_slice_relative: float
    band_absolute: float
    band_relative: float
    band_cell_count: int
    scored_cell_count: int


def grid_current_density(
    flux: np.ndarray, grid_r: np.ndarray, grid_z: np.ndarray
) -> np.ndarray:
    """Return the toroidal current density a total-flux map implies.

    ``flux`` is the total poloidal flux ``Phi`` in Wb on a uniform structured
    ``(Z, R)`` grid, and the result is returned on that same grid.  The
    operator is :func:`nova.equilibrium.map_extraction.apply_delta_star`, the
    repository's one structured-grid Grad--Shafranov read: the same centred
    ``Delta*`` and the same ``-Delta*Phi / (TOTAL_FLUX_FACTOR mu_0 R)``
    recovery.  That receipt carries its map ``(radius, height)`` and this one
    is ``(Z, R)``, so the map is transposed into the operator and its current
    density transposed back.  Cells without a complete stencil are ``nan``, so
    a caller's mask and the operator's own validity mask compose without a
    second boundary rule.
    """

    flux = np.asarray(flux, dtype=float)
    grid_r = np.asarray(grid_r, dtype=float)
    grid_z = np.asarray(grid_z, dtype=float)
    if flux.shape != (grid_z.size, grid_r.size):
        raise ValueError("flux does not match the profile grid")
    receipt = apply_delta_star(grid_r, grid_z, flux.T)
    return np.transpose(np.asarray(receipt.toroidal_current_density, dtype=float))


def edge_consistency(
    flux: np.ndarray,
    grid_r: np.ndarray,
    grid_z: np.ndarray,
    psi_axis: float,
    psi_boundary: float,
    p_prime: Callable[[np.ndarray], np.ndarray],
    ff_prime: Callable[[np.ndarray], np.ndarray],
    *,
    inside: np.ndarray | None = None,
    band_width: float = EDGE_BAND_WIDTH,
) -> EdgeConsistency:
    """Measure the flux-function/flux-map gap of one COCOS 17 state.

    ``flux`` is the total poloidal flux on the uniform ``(Z, R)`` grid named by
    ``grid_r`` and ``grid_z``.  ``psi_axis`` and ``psi_boundary`` bound the
    plasma and normalise the flux.  ``p_prime`` and ``ff_prime`` are the flux
    functions, called with the normalised flux array and returning values on
    the same grid.  ``inside`` optionally restricts the whole-slice form to a
    caller's cell set; when it is ``None`` the plasma interior
    ``0 <= psi_norm <= 1`` is the scored set, which is what the edge band is
    measured inside of in either case.
    """

    flux = np.asarray(flux, dtype=float)
    # The grid operator is assembled in jnp, so the working precision is set on
    # entry rather than inherited: at single precision the cancellation inside
    # Delta* leaves a round-off floor above the residual this measure reports.
    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("edge_consistency needs double precision arithmetic")
    span = float(psi_boundary) - float(psi_axis)
    if span == 0.0:
        raise ValueError("axis and boundary flux must differ")
    psi_norm = (flux - float(psi_axis)) / span
    if inside is None:
        interior = (psi_norm >= 0.0) & (psi_norm <= 1.0)
    else:
        interior = np.asarray(inside, dtype=bool)
        if interior.shape != flux.shape:
            raise ValueError("inside does not match the profile grid")
    current_density = grid_current_density(flux, grid_r, grid_z)
    radius = np.asarray(grid_r, dtype=float)[np.newaxis, :]
    profile_current = toroidal_current_density(
        radius,
        np.asarray(p_prime(psi_norm), dtype=float),
        np.asarray(ff_prime(psi_norm), dtype=float),
    )

    difference = np.where(
        np.isfinite(current_density) & np.isfinite(profile_current),
        profile_current - current_density,
        np.nan,
    )
    finite = np.isfinite(difference)
    scored = finite & interior
    reference_scale = (
        float(np.max(np.abs(current_density[scored]))) if scored.any() else float("nan")
    )
    whole_slice_relative = _relative_rms(difference[scored], reference_scale)

    band = finite & (psi_norm >= 1.0 - band_width) & (psi_norm <= 1.0)
    band_absolute = (
        float(np.sqrt(np.mean(np.square(difference[band]))))
        if band.any()
        else float("nan")
    )
    band_relative = _relative_rms(difference[band], profile_current[band])

    return EdgeConsistency(
        whole_slice_relative=whole_slice_relative,
        band_absolute=band_absolute,
        band_relative=band_relative,
        band_cell_count=int(np.count_nonzero(band)),
        scored_cell_count=int(np.count_nonzero(scored)),
    )


def _relative_rms(difference: np.ndarray, denominator) -> float:
    """Return ``RMS(difference / denominator)`` or ``nan`` where undefined."""

    difference = np.asarray(difference, dtype=float)
    denominator = np.broadcast_to(
        np.asarray(denominator, dtype=float), difference.shape
    )
    safe = np.isfinite(difference) & np.isfinite(denominator) & (denominator != 0.0)
    if not safe.any():
        return float("nan")
    ratio = np.where(safe, difference / np.where(safe, denominator, 1.0), 0.0)
    return float(np.sqrt(np.mean(np.square(ratio))))
