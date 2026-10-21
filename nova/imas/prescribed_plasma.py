"""Flux linked by a scenario's prescribed plasma current distribution."""

from __future__ import annotations

import numpy as np

from nova.biot.greens import greens_psi, hybrid_greens
from nova.biot.polygon import polygon_greens
from nova.circuit.conductor import ConductorSet, PolygonSection
from nova.circuit.linkage import section_grid
from nova.imas.profiles import PrescribedCurrent


def polygon_area(vertices: np.ndarray) -> float:
    """Enclosed area [m^2] of a polygon-section outline, orientation-free."""
    corners = np.asarray(vertices, float)
    radius, height = corners[:, 0], corners[:, 1]
    return 0.5 * abs(
        float(np.dot(radius, np.roll(height, -1)) - np.dot(height, np.roll(radius, -1)))
    )


def plasma_cells(
    current: PrescribedCurrent, *, vertical_offset: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The (r, z, current) of every plasma cell carrying current [A]."""
    source_r, source_z = np.meshgrid(current.radius, current.height, indexing="ij")
    live = np.flatnonzero(current.cell_current.ravel())
    return (
        source_r.ravel()[live],
        source_z.ravel()[live] + vertical_offset,
        current.cell_current.ravel()[live],
    )


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
    source_r, source_z, amps = plasma_cells(current, vertical_offset=vertical_offset)
    result = np.zeros(target_r.size)
    flat_r, flat_z = target_r.ravel(), target_z.ravel()
    for start in range(0, amps.size, 256):
        stop = min(start + 256, amps.size)
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


def polygon_quadrature(
    vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interior quadrature points of a polygon section, area-weighted.

    The section is fanned into triangles from its first corner and each triangle
    is split four ways, so every point is the centroid of a sub-triangle that
    lies inside the section by construction.  A section thinner than any fixed
    grid spacing is therefore sampled like any other; a bounding-box grid cannot
    do this, because its points are placed on the box and a thin enough section
    contains none of them.  Weights are sub-triangle area fractions, so they sum
    to one whatever the section's shape and orientation.

    Reciprocity through :func:`nova.biot.polygon.polygon_greens` averages the
    same section without sampling, and was measured too slow to use here: that
    kernel costs about 2.1 s per section against 4.6 s for EVERY one of the
    machine's shaped sections at their sampled points, so a plant build would
    take tens of minutes rather than seconds.
    """
    corners = np.asarray(vertices, float)
    if polygon_area(corners) <= 0.0:
        raise ValueError("polygon section encloses no area")
    apex = corners[0]
    centres: list[np.ndarray] = []
    areas: list[float] = []
    for index in range(1, corners.shape[0] - 1):
        left, right = corners[index], corners[index + 1]
        mid_left, mid_right = 0.5 * (apex + left), 0.5 * (apex + right)
        mid_edge = 0.5 * (left + right)
        for corner_a, corner_b, corner_c in (
            (apex, mid_left, mid_right),
            (mid_left, left, mid_edge),
            (mid_right, mid_edge, right),
            (mid_left, mid_edge, mid_right),
        ):
            centres.append((corner_a + corner_b + corner_c) / 3.0)
            areas.append(0.5 * abs(float(np.cross(corner_b - corner_a, corner_c - corner_a))))
    total = float(np.sum(areas))
    if total <= 0.0:
        raise ValueError("polygon section encloses no area")
    points = np.array(centres)
    return points[:, 0], points[:, 1], np.array(areas) / total


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
    if polygons:
        point_r, point_z, weight, owner = _polygon_point_grid(polygons, circuits)
        point_flux = plasma_flux_at_points(current, point_r, point_z)
        result += np.bincount(
            owner, weights=weight * point_flux, minlength=len(circuits)
        )
    return circuits, result


def _polygon_point_grid(
    polygons: dict[int, PolygonSection], circuits: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every shaped circuit's quadrature points in one array, with owners.

    The points of every section are gathered so the plasma flux is evaluated for
    all of them in a single call; the weight carries each section's turn share
    and its area fraction together.
    """
    point_r: list[np.ndarray] = []
    point_z: list[np.ndarray] = []
    weight: list[np.ndarray] = []
    owner: list[np.ndarray] = []
    for index, circuit in enumerate(circuits):
        polygon = polygons.get(int(circuit))
        if polygon is None:
            continue
        sub_r, sub_z, sub_weight = polygon_quadrature(
            np.asarray(polygon.vertices, float)
        )
        point_r.append(sub_r)
        point_z.append(sub_z)
        weight.append(float(polygon.current_share) * sub_weight)
        owner.append(np.full(sub_r.size, index, dtype=np.int64))
    return (
        np.concatenate(point_r),
        np.concatenate(point_z),
        np.concatenate(weight),
        np.concatenate(owner),
    )


def polygon_plasma_flux(
    current: PrescribedCurrent,
    polygon: PolygonSection,
) -> float:
    """Plasma flux [Wb] averaged over one polygon circuit's own cross-section."""
    point_r, point_z, weight = polygon_quadrature(
        np.asarray(polygon.vertices, float)
    )
    flux = plasma_flux_at_points(current, point_r, point_z)
    return polygon.current_share * float(np.dot(weight, flux))


def plasma_linked_flux_history(
    currents: list[PrescribedCurrent], conductors: ConductorSet
) -> tuple[np.ndarray, np.ndarray]:
    """Plasma flux [time, circuit] for a fixed conductor geometry."""
    linked = [plasma_linked_flux(current, conductors) for current in currents]
    if not linked:
        return conductors.circuits, np.empty((0, len(conductors.circuits)))
    return linked[0][0], np.stack([flux for _, flux in linked])
