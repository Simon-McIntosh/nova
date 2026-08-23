r"""Deterministic observation kernels over declared geometry and fields.

The Thomson kernel consumes explicit profile-coordinate and profile-value
arrays.  It does not define a persistent plasma state: the caller owns those
arrays and their provenance.  Chord coordinates are mapped to normalised flux
by :func:`nova.equilibrium.map_extraction.sample_chord_psi_norm`, after which
the two supplied profiles are linearly interpolated.

Virtual flux loops and poloidal probes compose the point and loop factories
owned by :mod:`nova.biot.biot`.  Their numerical-error receipts are measured
as the absolute difference between the dedicated sensor factory and the point
factory evaluated at the identical coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from nova.biot.biotframe import Target
from nova.equilibrium.map_extraction import sample_chord_psi_norm

if TYPE_CHECKING:
    from nova.biot.biot import Biot

__all__ = [
    "InterpolationSupportReceipt",
    "ObservationKernelReceipt",
    "ThomsonSignals",
    "VirtualMagneticSignals",
    "synthesize_thomson",
    "virtual_flux_loops",
    "virtual_poloidal_probes",
]

NOVA_COCOS = 17


@dataclass(frozen=True)
class InterpolationSupportReceipt:
    """Coordinate support used by one deterministic observation read."""

    method: str
    coordinate_minimum: float
    coordinate_maximum: float
    supported: jax.Array | NDArray[np.bool_]


@dataclass(frozen=True)
class ObservationKernelReceipt:
    """Units, convention, interpolation support, and numerical error."""

    kernel: str
    output_names: tuple[str, ...]
    units: tuple[str, ...]
    cocos: int
    interpolation_support: InterpolationSupportReceipt
    numerical_error_bound: jax.Array | NDArray[np.float64]


@dataclass(frozen=True)
class ThomsonSignals:
    """Electron temperature and density sampled along declared chords."""

    psi_norm: jax.Array
    electron_temperature: jax.Array
    electron_density: jax.Array
    receipt: ObservationKernelReceipt


@dataclass(frozen=True)
class VirtualMagneticSignals:
    """One virtual magnetic signal family and its qualification receipt."""

    values: NDArray[np.float64]
    receipt: ObservationKernelReceipt


def _require_nova_cocos(cocos: int) -> int:
    if int(cocos) != NOVA_COCOS:
        raise ValueError(
            f"observation kernels require Nova COCOS {NOVA_COCOS}, got {cocos}"
        )
    return NOVA_COCOS


def _profile_support(values: ArrayLike) -> NDArray[np.float64]:
    support = np.asarray(values, dtype=np.float64)
    if (
        support.ndim != 1
        or support.size < 2
        or not np.all(np.isfinite(support))
        or np.any(np.diff(support) <= 0.0)
    ):
        raise ValueError(
            "profile_psi_norm must be a finite, strictly increasing one-dimensional "
            "array with at least two points"
        )
    return support


def _profile_values(values: ArrayLike, size: int, name: str) -> jax.Array:
    profile = jnp.asarray(values, dtype=jnp.float64)
    if profile.shape != (size,):
        raise ValueError(f"{name} must have one value per profile_psi_norm point")
    return profile


def _map_interpolation_error(flux: jax.Array, flux_span: jax.Array) -> jax.Array:
    """Return the resolved-grid bilinear interpolation error envelope."""

    radial_curvature = jnp.max(jnp.abs(jnp.diff(flux, n=2, axis=0)))
    vertical_curvature = jnp.max(jnp.abs(jnp.diff(flux, n=2, axis=1)))
    return (radial_curvature + vertical_curvature) / (8.0 * jnp.abs(flux_span))


def _profile_interpolation_error(
    support: NDArray[np.float64], values: jax.Array
) -> jax.Array:
    """Return the piecewise-linear error envelope resolved by profile curvature."""

    interval = jnp.asarray(np.diff(support), dtype=values.dtype)
    if support.size == 2:
        return jnp.asarray(0.0, dtype=values.dtype)
    slope = jnp.diff(values) / interval
    curvature_spacing = 0.5 * (interval[:-1] + interval[1:])
    curvature = jnp.diff(slope) / curvature_spacing
    return jnp.max(jnp.abs(curvature)) * jnp.max(interval) ** 2 / 8.0


def _profile_signal_error(
    support: NDArray[np.float64],
    values: jax.Array,
    map_error: jax.Array,
) -> jax.Array:
    interval = jnp.asarray(np.diff(support), dtype=values.dtype)
    maximum_slope = jnp.max(jnp.abs(jnp.diff(values) / interval))
    return _profile_interpolation_error(support, values) + maximum_slope * map_error


def synthesize_thomson(
    radius: ArrayLike,
    height: ArrayLike,
    flux: ArrayLike,
    profile_psi_norm: ArrayLike,
    electron_temperature: ArrayLike,
    electron_density: ArrayLike,
    chord_coordinates: ArrayLike,
    *,
    axis_flux: float,
    boundary_flux: float,
    temperature_unit: str = "eV",
    density_unit: str = "m^-3",
    cocos: int = NOVA_COCOS,
) -> ThomsonSignals:
    """Return deterministic Thomson signals at supplied chord coordinates.

    Differentiation is supported through the flux map and both profile-value
    arrays.  The profile coordinate and grid axes are static interpolation
    supports; no persistent state object is introduced.
    """

    convention = _require_nova_cocos(cocos)
    support = _profile_support(profile_psi_norm)
    temperature = _profile_values(
        electron_temperature, support.size, "electron_temperature"
    )
    density = _profile_values(electron_density, support.size, "electron_density")
    chord = sample_chord_psi_norm(
        radius,
        height,
        flux,
        chord_coordinates,
        axis_flux=axis_flux,
        boundary_flux=boundary_flux,
    )
    psi_norm = jnp.asarray(chord.psi_norm)
    lower = float(support[0])
    upper = float(support[-1])
    supported = (
        jnp.asarray(chord.inside_grid)
        & jnp.isfinite(psi_norm)
        & (psi_norm >= lower)
        & (psi_norm <= upper)
    )
    support_array = jnp.asarray(support, dtype=temperature.dtype)
    temperature_signal = jnp.where(
        supported,
        jnp.interp(psi_norm, support_array, temperature),
        jnp.nan,
    )
    density_signal = jnp.where(
        supported,
        jnp.interp(psi_norm, support_array, density),
        jnp.nan,
    )

    flux_map = jnp.asarray(flux, dtype=temperature.dtype).reshape(
        (np.asarray(radius).size, np.asarray(height).size)
    )
    flux_span = jnp.asarray(boundary_flux) - jnp.asarray(axis_flux)
    map_error = _map_interpolation_error(flux_map, flux_span)
    error_bound = jnp.stack(
        [
            _profile_signal_error(support, temperature, map_error),
            _profile_signal_error(support, density, map_error),
        ]
    )
    receipt = ObservationKernelReceipt(
        kernel="thomson",
        output_names=("electron_temperature", "electron_density"),
        units=(str(temperature_unit), str(density_unit)),
        cocos=convention,
        interpolation_support=InterpolationSupportReceipt(
            method="bilinear flux map followed by linear flux-profile interpolation",
            coordinate_minimum=lower,
            coordinate_maximum=upper,
            supported=supported,
        ),
        numerical_error_bound=error_bound,
    )
    return ThomsonSignals(
        psi_norm=psi_norm,
        electron_temperature=temperature_signal,
        electron_density=density_signal,
        receipt=receipt,
    )


def _sensor_coordinates(coordinates: ArrayLike) -> NDArray[np.float64]:
    point = np.asarray(coordinates, dtype=np.float64)
    if point.ndim != 2 or point.shape[1] != 2 or not np.all(np.isfinite(point)):
        raise ValueError("sensor coordinates must be a finite (sensor, R-Z) array")
    if np.any(point[:, 0] <= 0.0):
        raise ValueError("sensor major radius must be strictly positive")
    return point


def _analytic_support(
    coordinates: NDArray[np.float64], method: str
) -> InterpolationSupportReceipt:
    return InterpolationSupportReceipt(
        method=method,
        coordinate_minimum=float(np.min(coordinates[:, 0])),
        coordinate_maximum=float(np.max(coordinates[:, 0])),
        supported=np.ones(coordinates.shape[0], dtype=bool),
    )


def _factory_agreement_bound(
    values: NDArray[np.float64], reference: NDArray[np.float64]
) -> NDArray[np.float64]:
    scale = np.maximum(np.maximum(np.abs(values), np.abs(reference)), 1.0)
    roundoff = 8.0 * np.finfo(np.float64).eps * scale
    return np.abs(values - reference) + roundoff


def virtual_flux_loops(
    biot: Biot,
    coordinates: ArrayLike,
    *,
    cocos: int = NOVA_COCOS,
) -> VirtualMagneticSignals:
    """Evaluate total poloidal flux through declared axisymmetric loops."""

    convention = _require_nova_cocos(cocos)
    point = _sensor_coordinates(coordinates)
    biot.point.solve(point)
    point_reference = np.asarray(biot.point.psi, dtype=np.float64).copy()

    loop = biot.poloidal_flux_loop
    loop.target = Target()
    for radius, height in point:
        loop.insert(radius, height)
    loop.solve()
    values = np.asarray(loop.psi, dtype=np.float64).copy()
    error_bound = _factory_agreement_bound(values, point_reference)
    return VirtualMagneticSignals(
        values=values,
        receipt=ObservationKernelReceipt(
            kernel="virtual_flux_loop",
            output_names=("total_poloidal_flux",),
            units=("Wb",),
            cocos=convention,
            interpolation_support=_analytic_support(
                point, "axisymmetric loop factory; no spatial interpolation"
            ),
            numerical_error_bound=error_bound,
        ),
    )


def virtual_poloidal_probes(
    biot: Biot,
    coordinates: ArrayLike,
    orientation: ArrayLike,
    *,
    cocos: int = NOVA_COCOS,
) -> VirtualMagneticSignals:
    """Evaluate poloidal field projected on declared unit directions."""

    convention = _require_nova_cocos(cocos)
    point = _sensor_coordinates(coordinates)
    direction = np.asarray(orientation, dtype=np.float64)
    if direction.shape != point.shape or not np.all(np.isfinite(direction)):
        raise ValueError("probe orientation must be a finite (sensor, R-Z) array")
    length = np.linalg.norm(direction, axis=1)
    if not np.allclose(length, 1.0, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("each probe orientation must be a unit vector")

    biot.point.solve(point)
    point_reference = (
        np.asarray(biot.point.br, dtype=np.float64) * direction[:, 0]
        + np.asarray(biot.point.bz, dtype=np.float64) * direction[:, 1]
    )
    biot.probe.solve(point)
    values = (
        np.asarray(biot.probe.br, dtype=np.float64) * direction[:, 0]
        + np.asarray(biot.probe.bz, dtype=np.float64) * direction[:, 1]
    )
    error_bound = _factory_agreement_bound(values, point_reference)
    return VirtualMagneticSignals(
        values=values,
        receipt=ObservationKernelReceipt(
            kernel="virtual_poloidal_probe",
            output_names=("projected_poloidal_field",),
            units=("T",),
            cocos=convention,
            interpolation_support=_analytic_support(
                point, "point probe factory; no spatial interpolation"
            ),
            numerical_error_bound=error_bound,
        ),
    )
