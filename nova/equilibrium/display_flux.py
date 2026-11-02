"""Evaluate a display flux map on a fine grid from a state's solved currents.

The flux a state produces is a matrix-vector product: the conductor currents and
the plasma-cell uniform, radial and vertical current moments contract against
the coupling kernels at whatever points are asked for.  A panel therefore does
not have to draw the lattice the solve was carried on; it can evaluate the same
field wherever it wants the contour to be read.

The coupling is supplied by the caller as :class:`DisplayCoupling`, a pair of
callables over target points.  Every producer already owns one -- the same
``hybrid_greens`` columns and circuit response rows it builds its lattice target
from -- so this module adds no kernel of its own and never interpolates the
lattice.

The lattice spline stays the solve's authority.  The map equals the state at the
lattice nodes only when the state is a converged fixed point, so the builder
reports the largest node mismatch it sees, and the saddle it polishes on its own
grid beside the saddle of the lattice spline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time

import jax.numpy as jnp
import numpy as np

from nova.linalg.tensor_spline import fit_tensor_spline

#: Display spacing [m] the plan fixes as the coarsest a panel may be drawn at.
DEFAULT_SPACING = 0.010


@dataclass(frozen=True)
class DisplayCoupling:
    """Target-point coupling of one machine: flux blocks and external response.

    ``plasma`` returns the uniform, radial and vertical moment columns and
    ``external`` the flux the conductors and the prescribed-current response
    carry, both for the ``(n, 2)`` ``(radius, height)`` points they are handed.
    """

    plasma: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]
    external: Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class DisplayFluxMap:
    """One state's field evaluated on its own rectangular display grid."""

    radius: np.ndarray
    height: np.ndarray
    flux: np.ndarray
    saddle_rz: np.ndarray
    saddle_flux: float
    lattice_saddle_rz: np.ndarray
    saddle_offset_mm: float
    max_node_mismatch_wb: float
    reproduction_relative_error: float
    spacing_m: float
    build_seconds: float

    @property
    def shape(self) -> tuple[int, int]:
        """Return the map extent as ``(radial, vertical)``."""
        return self.flux.shape


def _grid(points: np.ndarray, coupling: DisplayCoupling, chunk: int):
    """Evaluate the coupling over ``points`` in bounded blocks."""
    uniform = np.empty((points.shape[0], 0), dtype=np.float64)
    radial = uniform
    vertical = uniform
    external = np.empty(points.shape[0], dtype=np.float64)
    for start in range(0, points.shape[0], chunk):
        block = points[start : start + chunk]
        block_uniform, block_radial, block_vertical = coupling.plasma(block)
        external[start : start + chunk] = np.asarray(
            coupling.external(block), dtype=np.float64
        )
        if uniform.shape[1] == 0:
            uniform = np.empty((points.shape[0], np.shape(block_uniform)[1]))
            radial = np.empty_like(uniform)
            vertical = np.empty_like(uniform)
        stop = start + block.shape[0]
        uniform[start:stop] = np.asarray(block_uniform, dtype=np.float64)
        radial[start:stop] = np.asarray(block_radial, dtype=np.float64)
        vertical[start:stop] = np.asarray(block_vertical, dtype=np.float64)
    return external, uniform, radial, vertical


def _moment_flux(coupling: DisplayCoupling, points: np.ndarray, moments, chunk: int):
    """Return the conductor-plus-plasma flux at ``points``."""
    external, uniform, radial, vertical = _grid(points, coupling, chunk)
    flux = external.copy()
    for block, values in zip(
        (uniform, radial, vertical),
        moments,
        strict=True,
    ):
        if block.shape[1] == 0:
            continue
        flux += block @ np.asarray(values, dtype=np.float64)
    return flux


def _evaluation(surface, point):
    """Return the spline's value and gradient at ``point``."""
    values = surface.evaluate(jnp.asarray(point[0]), jnp.asarray(point[1]))
    gradient = np.asarray(
        [values.radial_derivative, values.vertical_derivative], dtype=np.float64
    ).reshape(2)
    return float(np.asarray(values.value).reshape(())), gradient


def _polished_saddle(values, radius, height, xpoint_rz):
    """Return the map's own stationary saddle and the flux it carries.

    The saddle is the stationary point of the map's own tensor spline nearest
    where the caller says the read found one.  It is polished by Newton steps on
    the spline gradient from the mid-cell point of the cell whose two central
    differences change sign, which is the same stationary point the map carries
    at its own resolution rather than one carried over from another grid.
    """
    surface = fit_tensor_spline(radius, height, values)
    host = np.asarray(values, dtype=np.float64)
    radial_step = float(radius[1] - radius[0])
    vertical_step = float(height[1] - height[0])
    radial_slope = np.gradient(host, radial_step, axis=1)
    vertical_slope = np.gradient(host, vertical_step, axis=0)
    sign_r = np.sign(radial_slope[1:-1, 1:-1] * radial_slope[1:-1, 2:])
    sign_z = np.sign(vertical_slope[1:-1, 1:-1] * vertical_slope[2:, 1:-1])
    row, column = np.nonzero((sign_r < 0.0) & (sign_z < 0.0))
    if row.size == 0:
        return np.full(2, np.nan), float("nan")
    candidate = np.c_[
        np.asarray(radius)[column + 1] + 0.5 * radial_step,
        np.asarray(height)[row + 1] + 0.5 * vertical_step,
    ]
    nearest = int(
        np.argmin(np.linalg.norm(candidate - np.asarray(xpoint_rz)[None, :], axis=1))
    )
    point = candidate[nearest]
    for _step in range(20):
        _value, gradient = _evaluation(surface, point)
        if np.linalg.norm(gradient) < 1.0e-9:
            break
        offset = 1.0e-4 * np.maximum(np.abs(point), 1.0e-3)
        hessian = np.column_stack(
            (
                (_evaluation(surface, point + [offset[0], 0.0])[1] - gradient)
                / offset[0],
                (_evaluation(surface, point + [0.0, offset[1]])[1] - gradient)
                / offset[1],
            )
        )
        try:
            point = point - np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(point)):
            point = candidate[nearest]
            break
    return point, _evaluation(surface, point)[0]


def build_display_flux_map(
    profile,
    state,
    *,
    coupling: DisplayCoupling,
    moments=None,
    spacing: float = DEFAULT_SPACING,
    extent: tuple[float, float, float, float] | None = None,
    current=None,
    prescribed_current=None,
    target_current=None,
    requested_class=None,
    xpoint_rz=None,
    chunk: int = 4096,
) -> DisplayFluxMap:
    """Return one state's field on a fine display grid.

    The map is evaluated from the state's own conductor and plasma currents at
    every display point.  ``reproduction_relative_error`` is the largest
    deviation of that evaluation from the operator's own ``raster_image`` at the
    lattice nodes, and ``max_node_mismatch_wb`` the largest deviation from the
    state itself: zero for a converged fixed point, the fixed-point residual
    otherwise.
    """
    started = time.perf_counter()
    if spacing > DEFAULT_SPACING:
        raise ValueError(
            f"display spacing {spacing} m is coarser than the {DEFAULT_SPACING} m floor"
        )
    lattice = profile.lattice
    operator = profile.operator
    if moments is None:
        if target_current is None:
            moments = operator.current_moments_and_observation(state, requested_class)[
                0
            ]
        else:
            moments = operator.normalised_current_moments_and_observation(
                state, target_current, requested_class
            )[0]
    nodewise = (moments.cell_current, moments.radial_moment, moments.vertical_moment)

    radius_nodes = np.asarray(lattice.radius, dtype=np.float64)
    height_nodes = np.asarray(lattice.height, dtype=np.float64)
    bounds = extent or (
        float(radius_nodes[0]),
        float(radius_nodes[-1]),
        float(height_nodes[0]),
        float(height_nodes[-1]),
    )
    radius = np.arange(bounds[0], bounds[1] + 0.5 * spacing, spacing)
    height = np.arange(bounds[2], bounds[3] + 0.5 * spacing, spacing)
    radius_grid, height_grid = np.meshgrid(radius, height, indexing="ij")
    points = np.c_[radius_grid.ravel(), height_grid.ravel()]
    flux = _moment_flux(coupling, points, nodewise, chunk).reshape(
        radius.size, height.size
    )

    node_flux = _moment_flux(coupling, np.asarray(lattice.coordinate), nodewise, chunk)
    raster = np.asarray(
        operator.raster_image(
            moments, current=current, prescribed_current=prescribed_current
        )[: lattice.node_count],
        dtype=np.float64,
    )
    state_nodes = np.asarray(state, dtype=np.float64)[: lattice.node_count]
    scale = max(float(np.max(np.abs(raster))), 1.0e-30)
    reproduction = float(np.max(np.abs(node_flux - raster))) / scale
    mismatch = float(np.max(np.abs(node_flux - state_nodes)))

    if xpoint_rz is None:
        _masks, topology = operator.read(state, requested_class)
        xpoint_rz = topology.x_point
    xpoint_rz = np.asarray(xpoint_rz, dtype=np.float64).reshape(-1)[:2]

    saddle_rz, saddle_flux = _polished_saddle(
        np.asarray(flux.T), radius, height, xpoint_rz
    )
    lattice_values = jnp.asarray(np.asarray(state)[: lattice.node_count]).reshape(
        lattice.shape
    )
    lattice_saddle_rz, _lattice_flux = _polished_saddle(
        np.asarray(lattice_values).T, radius_nodes, height_nodes, xpoint_rz
    )
    return DisplayFluxMap(
        radius=radius,
        height=height,
        flux=flux,
        saddle_rz=np.asarray(saddle_rz, dtype=np.float64),
        saddle_flux=saddle_flux,
        lattice_saddle_rz=np.asarray(lattice_saddle_rz, dtype=np.float64),
        saddle_offset_mm=float(
            1.0e3 * np.linalg.norm(np.asarray(saddle_rz) - lattice_saddle_rz)
        ),
        max_node_mismatch_wb=mismatch,
        reproduction_relative_error=reproduction,
        spacing_m=float(spacing),
        build_seconds=time.perf_counter() - started,
    )


__all__ = [
    "DEFAULT_SPACING",
    "DisplayCoupling",
    "DisplayFluxMap",
    "build_display_flux_map",
]
