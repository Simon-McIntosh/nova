"""Flux linked by a scenario's prescribed plasma current distribution."""

from __future__ import annotations

import numpy as np
from shapely import contains_xy
from shapely.geometry import Polygon

from nova.biot.greens import greens_psi, hybrid_greens
from nova.circuit.conductor import ConductorSet
from nova.circuit.linkage import section_grid, section_points
from nova.imas.profiles import PrescribedCurrent


def plasma_flux_at_points(
    current: PrescribedCurrent,
    radius: np.ndarray,
    height: np.ndarray,
    *,
    vertical_offset: float = 0.0,
) -> np.ndarray:
    """Total poloidal flux [Wb] at arbitrary circular filament locations."""
    target_r, target_z = np.broadcast_arrays(
        np.asarray(radius, float), np.asarray(height, float)
    )
    if np.any(target_r <= 0):
        raise ValueError("target radius must be positive")
    source_r, source_z = np.meshgrid(current.radius, current.height, indexing="ij")
    live = np.flatnonzero(current.cell_current.ravel())
    source_r = source_r.ravel()[live]
    source_z = source_z.ravel()[live] + vertical_offset
    amps = current.cell_current.ravel()[live]
    result = np.zeros(target_r.size)
    flat_r, flat_z = target_r.ravel(), target_z.ravel()
    for start in range(0, len(live), 256):
        stop = min(start + 256, len(live))
        with np.errstate(divide="ignore", invalid="ignore"):
            kernel = greens_psi(
                flat_r[:, None],
                flat_z[:, None],
                source_r[None, start:stop],
                source_z[None, start:stop],
            )
        singular_row, singular_col = np.where(~np.isfinite(kernel))
        for row, col in zip(singular_row, singular_col):
            source = start + col
            psi, _, _ = hybrid_greens(
                np.array([flat_r[row]]),
                np.array([flat_z[row]]),
                source_r[source],
                source_z[source],
                float(np.mean(np.diff(current.radius))),
                float(np.mean(np.diff(current.height))),
            )
            kernel[row, col] = psi[0]
        result += kernel @ amps[start:stop]
    return result.reshape(target_r.shape)


def plasma_linked_flux(
    current: PrescribedCurrent,
    conductors: ConductorSet,
    *,
    section_spacing: float = 0.05,
    section_n_max: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Section-averaged plasma flux [Wb] per circuit id, including turn shares."""
    if section_spacing <= 0 or section_n_max < 1:
        raise ValueError("section quadrature spacing and cap must be positive")
    circuits = conductors.circuits
    result = np.zeros(len(circuits))
    polygons = conductors.polygon_by_circuit()
    rectangular = [int(circuit) for circuit in circuits if int(circuit) not in polygons]
    if rectangular:
        point_r, point_z, weight, owner = section_grid(
            conductors,
            conductors.rows(rectangular),
            section_spacing,
            section_n_max,
        )
        point_flux = plasma_flux_at_points(current, point_r, point_z)
        rectangular_flux = np.bincount(
            owner, weights=weight * point_flux, minlength=len(rectangular)
        )
        result[np.isin(circuits, rectangular)] = rectangular_flux
    for index, circuit in enumerate(circuits):
        polygon = polygons.get(int(circuit))
        if polygon is None:
            continue
        vertices = np.asarray(polygon.vertices, float)
        extent = np.ptp(vertices, axis=0)
        mid = np.mean([vertices.min(axis=0), vertices.max(axis=0)], axis=0)
        point_r, point_z, weight = section_points(
            mid[0], mid[1], extent[0], extent[1], section_spacing, section_n_max
        )
        inside = contains_xy(Polygon(vertices), point_r, point_z)
        if not np.any(inside):
            raise ValueError(f"polygon circuit {circuit} has no quadrature points")
        weight = weight[inside] / weight[inside].sum()
        result[index] = polygon.current_share * np.dot(
            weight, plasma_flux_at_points(current, point_r[inside], point_z[inside])
        )
    return circuits, result


def plasma_linked_flux_history(
    currents: list[PrescribedCurrent], conductors: ConductorSet
) -> tuple[np.ndarray, np.ndarray]:
    """Plasma flux [time, circuit] for a fixed conductor geometry."""
    linked = [plasma_linked_flux(current, conductors) for current in currents]
    if not linked:
        return conductors.circuits, np.empty((0, len(conductors.circuits)))
    return linked[0][0], np.stack([flux for _, flux in linked])
