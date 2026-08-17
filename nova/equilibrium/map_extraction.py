r"""Deterministic response kernels for a labelled equilibrium flux map.

The functions in this module only read a supplied map.  They do not compare
it with a measurement, optimise a state, or update a profile.  Total poloidal
flux follows :mod:`nova.equilibrium.convention`, so the gridded
Grad--Shafranov relation is

.. math::
    \Delta^\star \Phi = -2\pi\mu_0Rj_\phi
      = 4\pi^2\left(\mu_0R^2p' + FF'\right).

The surface separation follows the affine current projection used for EFIT
maps: ``R * j_phi`` onto ``R**2`` and a constant on each normalised-flux
shell, followed by the sign and total-flux conversion pinned above.  Its
receipt exposes the projection residual, the scaled design condition, and
uncertainty inflation where a surface collapses onto the magnetic axis or
where ``|grad(psi_N)|`` approaches zero.  Those values qualify a produced
response; they are never used here to alter or fit the supplied map.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.constants import mu_0

from nova.equilibrium.conservation import FluxLattice, delta_star
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR

__all__ = [
    "ChordSamplingReceipt",
    "MapCurrentReceipt",
    "SurfaceExtractionReceipt",
    "VacuumRegionReceipt",
    "apply_delta_star",
    "extract_flux_functions",
    "sample_chord_psi_norm",
    "vacuum_region_receipt",
]


@dataclass(frozen=True)
class MapCurrentReceipt:
    """Grad--Shafranov operator and current read from one flux map.

    Nodes outside the production operator's complete-stencil interior are
    deliberately marked invalid and carry ``NaN``.  Publishing a one-sided
    boundary estimate would mix a different operator into the same receipt.
    """

    radius: NDArray[np.float64]
    height: NDArray[np.float64]
    flux: NDArray[np.float64]
    delta_star_flux: NDArray[np.float64]
    toroidal_current_density: NDArray[np.float64]
    valid: NDArray[np.bool_]
    radial_step: float
    vertical_step: float
    finite: bool


@dataclass(frozen=True)
class SurfaceExtractionReceipt:
    """Per-surface flux functions and their numerical qualification."""

    psi_norm: NDArray[np.float64]
    p_prime: NDArray[np.float64]
    ff_prime: NDArray[np.float64]
    p_prime_uncertainty: NDArray[np.float64]
    ff_prime_uncertainty: NDArray[np.float64]
    projection_rms: NDArray[np.float64]
    condition_number: NDArray[np.float64]
    uncertainty_inflation: NDArray[np.float64]
    minimum_gradient: NDArray[np.float64]
    sample_count: NDArray[np.int64]
    reliable: NDArray[np.bool_]
    current: MapCurrentReceipt


@dataclass(frozen=True)
class VacuumRegionReceipt:
    """Qualification of ``Delta-star(flux) approximately zero`` in vacuum."""

    sample_count: int
    rms_delta_star: float
    max_abs_delta_star: float
    reference_rms_delta_star: float
    relative_rms: float
    relative_tolerance: float
    passed: bool


@dataclass(frozen=True)
class ChordSamplingReceipt:
    """Normalised flux sampled at supplied chord coordinates."""

    coordinates: NDArray[np.float64]
    psi_norm: NDArray[np.float64]
    inside_grid: NDArray[np.bool_]
    finite: bool
    axis_flux: float
    boundary_flux: float


def _uniform_axis(values: ArrayLike, name: str) -> tuple[NDArray[np.float64], float]:
    axis = np.asarray(values, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 3:
        raise ValueError(f"{name} must be one dimensional with at least three nodes")
    difference = np.diff(axis)
    if np.any(difference <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    if not np.allclose(difference, difference[0], rtol=1.0e-12, atol=0.0):
        raise ValueError(f"{name} must be uniformly spaced")
    return axis, float(difference[0])


def _map(values: ArrayLike, shape: tuple[int, int], name: str) -> NDArray[np.float64]:
    field = np.asarray(values, dtype=np.float64)
    if field.shape == (shape[0] * shape[1],):
        field = field.reshape(shape)
    if field.shape != shape:
        raise ValueError(f"{name} must have shape {shape} or flatten to that shape")
    return field


def apply_delta_star(
    radius: ArrayLike,
    height: ArrayLike,
    flux: ArrayLike,
) -> MapCurrentReceipt:
    """Apply the centred Grad--Shafranov operator to a structured flux map.

    ``flux`` may be lattice-shaped or flattened in C order.  The returned
    current density is derived only from Ampere's law; no source profile is
    required or inferred by this operation.
    """

    radius_axis, radial_step = _uniform_axis(radius, "radius")
    height_axis, vertical_step = _uniform_axis(height, "height")
    shape = (radius_axis.size, height_axis.size)
    flux_map = _map(flux, shape, "flux")
    mesh = FluxLattice(radius_axis, height_axis)
    radius_map = mesh.node_radius.reshape(shape)
    delta_star_flux = np.asarray(
        delta_star(mesh, flux_map.reshape(-1)), dtype=np.float64
    ).reshape(shape)
    valid = np.array(mesh.interior(), dtype=bool, copy=True).reshape(shape)
    valid &= np.isfinite(delta_star_flux)
    current_density = np.full(shape, np.nan, dtype=np.float64)
    current_density[valid] = -delta_star_flux[valid] / (
        TOTAL_FLUX_FACTOR * mu_0 * radius_map[valid]
    )
    finite = bool(
        np.all(np.isfinite(flux_map))
        and np.all(np.isfinite(delta_star_flux[valid]))
        and np.all(np.isfinite(current_density[valid]))
    )
    return MapCurrentReceipt(
        radius=radius_axis,
        height=height_axis,
        flux=flux_map,
        delta_star_flux=delta_star_flux,
        toroidal_current_density=current_density,
        valid=valid,
        radial_step=radial_step,
        vertical_step=vertical_step,
        finite=finite,
    )


def _surface_boundaries(centres: NDArray[np.float64]) -> NDArray[np.float64]:
    boundary = np.empty(centres.size + 1, dtype=np.float64)
    boundary[1:-1] = 0.5 * (centres[1:] + centres[:-1])
    boundary[0] = max(0.0, centres[0] - 0.5 * (centres[1] - centres[0]))
    boundary[-1] = min(1.0, centres[-1] + 0.5 * (centres[-1] - centres[-2]))
    return boundary


def extract_flux_functions(
    radius: ArrayLike,
    height: ArrayLike,
    flux: ArrayLike,
    psi_norm: ArrayLike,
    *,
    surfaces: ArrayLike | None = None,
    plasma_mask: ArrayLike | None = None,
    min_samples: int = 8,
    maximum_condition: float = 50.0,
    maximum_inflation: float = 100.0,
) -> SurfaceExtractionReceipt:
    """Separate ``p_prime`` and ``FF_prime`` on normalised-flux shells.

    On each shell, ``R * j_phi`` is projected onto ``[R**2, 1]``.  Those
    coefficients are derivatives in the input map's per-radian convention;
    division by ``-2 pi`` (and multiplication of the intercept by ``mu_0``)
    publishes Nova's negated-total-flux ``p_prime`` and ``FF_prime``.  This is
    a deterministic read of the supplied map, not an optimisation against
    observations.  The two-column design is scaled before its condition
    number is evaluated, so the receipt describes geometric separability
    rather than the different SI units of its columns.

    ``surfaces`` defaults to 19 shell centres from 0.05 through 0.95.  A
    caller may include points closer to the axis or separatrix; unresolved
    shells remain in the fixed-shape receipt with ``reliable=False`` and
    infinite uncertainty rather than disappearing.
    """

    if min_samples < 3:
        raise ValueError("min_samples must be at least three")
    current = apply_delta_star(radius, height, flux)
    shape = current.flux.shape
    normalised = _map(psi_norm, shape, "psi_norm")
    if surfaces is None:
        surface_coordinate = np.linspace(0.05, 0.95, 19)
    else:
        surface_coordinate = np.asarray(surfaces, dtype=np.float64)
    if (
        surface_coordinate.ndim != 1
        or surface_coordinate.size < 2
        or np.any(np.diff(surface_coordinate) <= 0.0)
        or surface_coordinate[0] < 0.0
        or surface_coordinate[-1] > 1.0
    ):
        raise ValueError("surfaces must be increasing normalised flux in [0, 1]")
    if plasma_mask is None:
        selected = (normalised >= 0.0) & (normalised <= 1.0)
    else:
        selected = _map(plasma_mask, shape, "plasma_mask").astype(bool)
    selected &= current.valid & np.isfinite(normalised)

    mesh = FluxLattice(current.radius, current.height)
    radial_gradient, vertical_gradient = (
        np.asarray(component, dtype=np.float64).reshape(shape)
        for component in mesh.gradient(normalised.reshape(-1))
    )
    gradient = np.hypot(radial_gradient, vertical_gradient)
    selected &= np.isfinite(gradient)
    positive_gradient = gradient[selected & (gradient > 0.0)]
    gradient_reference = (
        float(np.median(positive_gradient)) if positive_gradient.size else 1.0
    )
    gradient_floor = max(
        np.finfo(np.float64).eps * gradient_reference,
        np.finfo(np.float64).tiny,
    )

    count = surface_coordinate.size
    p_prime = np.full(count, np.nan)
    ff_prime = np.full(count, np.nan)
    p_uncertainty = np.full(count, np.inf)
    ff_uncertainty = np.full(count, np.inf)
    projection_rms = np.full(count, np.nan)
    condition_number = np.full(count, np.inf)
    uncertainty_inflation = np.full(count, np.inf)
    minimum_gradient = np.full(count, np.nan)
    sample_count = np.zeros(count, dtype=np.int64)
    reliable = np.zeros(count, dtype=bool)

    boundaries = _surface_boundaries(surface_coordinate)
    radius_map = np.broadcast_to(current.radius[:, None], shape)
    current_density = current.toroidal_current_density
    response = radius_map * current_density
    response_scale = max(
        float(np.nanmax(np.abs(response[selected]))) if np.any(selected) else 0.0,
        1.0,
    )
    numerical_floor = np.sqrt(np.finfo(np.float64).eps) * response_scale

    for index, centre in enumerate(surface_coordinate):
        upper_comparison = normalised <= boundaries[index + 1]
        if index + 1 < count:
            upper_comparison = normalised < boundaries[index + 1]
        shell = selected & (normalised >= boundaries[index]) & upper_comparison
        sample_count[index] = int(np.count_nonzero(shell))
        if sample_count[index] < min_samples:
            continue

        design = np.column_stack([radius_map[shell] ** 2, np.ones(sample_count[index])])
        target = response[shell]
        column_scale = np.linalg.norm(design, axis=0)
        scaled_design = design / column_scale
        singular_values = np.linalg.svd(scaled_design, compute_uv=False)
        if singular_values[-1] <= np.finfo(np.float64).eps:
            continue
        condition_number[index] = float(singular_values[0] / singular_values[-1])
        scaled_coefficients = np.linalg.lstsq(scaled_design, target, rcond=None)[0]
        coefficients = scaled_coefficients / column_scale
        p_prime[index] = -coefficients[0] / TOTAL_FLUX_FACTOR
        ff_prime[index] = -mu_0 * coefficients[1] / TOTAL_FLUX_FACTOR

        residual = target - design @ coefficients
        projection_rms[index] = float(np.sqrt(np.mean(residual**2)))
        variance = max(projection_rms[index], numerical_floor) ** 2
        scaled_covariance = variance * np.linalg.pinv(scaled_design.T @ scaled_design)
        covariance = scaled_covariance / np.outer(column_scale, column_scale)

        minimum_gradient[index] = float(np.min(gradient[shell]))
        gradient_inflation = gradient_reference / max(
            minimum_gradient[index], gradient_floor
        )
        shell_width = boundaries[index + 1] - boundaries[index]
        axis_inflation = np.sqrt(1.0 / max(centre, shell_width))
        uncertainty_inflation[index] = float(
            max(1.0, gradient_inflation, axis_inflation)
        )
        p_uncertainty[index] = (
            np.sqrt(max(float(covariance[0, 0]), 0.0))
            * uncertainty_inflation[index]
            / TOTAL_FLUX_FACTOR
        )
        ff_uncertainty[index] = (
            mu_0
            * np.sqrt(max(float(covariance[1, 1]), 0.0))
            * uncertainty_inflation[index]
            / TOTAL_FLUX_FACTOR
        )
        reliable[index] = bool(
            np.isfinite(p_prime[index])
            and np.isfinite(ff_prime[index])
            and condition_number[index] <= maximum_condition
            and uncertainty_inflation[index] <= maximum_inflation
        )

    return SurfaceExtractionReceipt(
        psi_norm=surface_coordinate,
        p_prime=p_prime,
        ff_prime=ff_prime,
        p_prime_uncertainty=p_uncertainty,
        ff_prime_uncertainty=ff_uncertainty,
        projection_rms=projection_rms,
        condition_number=condition_number,
        uncertainty_inflation=uncertainty_inflation,
        minimum_gradient=minimum_gradient,
        sample_count=sample_count,
        reliable=reliable,
        current=current,
    )


def vacuum_region_receipt(
    current: MapCurrentReceipt,
    vacuum_mask: ArrayLike,
    *,
    relative_tolerance: float,
) -> VacuumRegionReceipt:
    """Return the vacuum-region elliptic-residual receipt for one map."""

    if relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive")
    vacuum = _map(vacuum_mask, current.flux.shape, "vacuum_mask").astype(bool)
    selected = vacuum & current.valid
    sample_count = int(np.count_nonzero(selected))
    if sample_count == 0:
        raise ValueError("vacuum_mask selects no nodes with a complete stencil")
    values = current.delta_star_flux[selected]
    rms = float(np.sqrt(np.mean(values**2)))
    maximum = float(np.max(np.abs(values)))
    reference_values = current.delta_star_flux[current.valid]
    reference = float(np.sqrt(np.mean(reference_values**2)))
    scale_floor = (
        np.finfo(np.float64).eps
        * max(float(np.max(np.abs(current.flux))), 1.0)
        / min(current.radial_step, current.vertical_step) ** 2
    )
    reference = max(reference, scale_floor)
    relative = rms / reference
    return VacuumRegionReceipt(
        sample_count=sample_count,
        rms_delta_star=rms,
        max_abs_delta_star=maximum,
        reference_rms_delta_star=reference,
        relative_rms=relative,
        relative_tolerance=float(relative_tolerance),
        passed=bool(np.isfinite(relative) and relative <= relative_tolerance),
    )


def sample_chord_psi_norm(
    radius: ArrayLike,
    height: ArrayLike,
    flux: ArrayLike,
    coordinates: ArrayLike,
    *,
    axis_flux: float,
    boundary_flux: float,
) -> ChordSamplingReceipt:
    """Bilinearly sample ``psi_N`` at supplied chord coordinates.

    ``coordinates`` has final dimension ``(R, Z)`` and may retain any leading
    chord and point dimensions.  Points outside the grid are receipt-visible
    through ``inside_grid=False`` and carry ``NaN`` rather than an extrapolated
    label.
    """

    radius_axis, _ = _uniform_axis(radius, "radius")
    height_axis, _ = _uniform_axis(height, "height")
    flux_map = _map(flux, (radius_axis.size, height_axis.size), "flux")
    point = np.asarray(coordinates, dtype=np.float64)
    if point.ndim < 1 or point.shape[-1] != 2:
        raise ValueError("coordinates must have final dimension (R, Z)")
    span = float(boundary_flux) - float(axis_flux)
    if not np.isfinite(span) or span == 0.0:
        raise ValueError("axis_flux and boundary_flux must define a finite span")

    flat = point.reshape(-1, 2)
    inside = (
        np.isfinite(flat).all(axis=1)
        & (flat[:, 0] >= radius_axis[0])
        & (flat[:, 0] <= radius_axis[-1])
        & (flat[:, 1] >= height_axis[0])
        & (flat[:, 1] <= height_axis[-1])
    )
    sampled = np.full(flat.shape[0], np.nan, dtype=np.float64)
    if np.any(inside):
        kept = flat[inside]
        radial_index = np.searchsorted(radius_axis, kept[:, 0], side="right") - 1
        vertical_index = np.searchsorted(height_axis, kept[:, 1], side="right") - 1
        radial_index = np.clip(radial_index, 0, radius_axis.size - 2)
        vertical_index = np.clip(vertical_index, 0, height_axis.size - 2)
        radial_fraction = (kept[:, 0] - radius_axis[radial_index]) / (
            radius_axis[radial_index + 1] - radius_axis[radial_index]
        )
        vertical_fraction = (kept[:, 1] - height_axis[vertical_index]) / (
            height_axis[vertical_index + 1] - height_axis[vertical_index]
        )
        lower_left = flux_map[radial_index, vertical_index]
        upper_left = flux_map[radial_index + 1, vertical_index]
        lower_right = flux_map[radial_index, vertical_index + 1]
        upper_right = flux_map[radial_index + 1, vertical_index + 1]
        interpolated = (
            (1.0 - radial_fraction) * (1.0 - vertical_fraction) * lower_left
            + radial_fraction * (1.0 - vertical_fraction) * upper_left
            + (1.0 - radial_fraction) * vertical_fraction * lower_right
            + radial_fraction * vertical_fraction * upper_right
        )
        sampled[inside] = (interpolated - float(axis_flux)) / span

    output_shape = point.shape[:-1]
    psi_norm = sampled.reshape(output_shape)
    inside_grid = inside.reshape(output_shape)
    return ChordSamplingReceipt(
        coordinates=point,
        psi_norm=psi_norm,
        inside_grid=inside_grid,
        finite=bool(np.all(np.isfinite(psi_norm[inside_grid]))),
        axis_flux=float(axis_flux),
        boundary_flux=float(boundary_flux),
    )
