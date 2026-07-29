"""Distance-banded coupling for finite arcs with polygon cross-sections.

The expensive finite-section reduction is needed near the swept conductor.  Far
from it, the section is represented by a circular-arc filament carrying the
section's first three area moments.  The split is based on distance to the
finite swept volume: within the angular span it is the poloidal contour distance;
outside either end it also carries the chord to the nearest end plane.

The equivalent filament may sit at the area centroid or at the section's
root-mean-square radius.  A general third-order expansion carries the non-zero
first moment introduced by the latter placement, so comparing the two placements
does not silently compare different approximation orders.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss

from nova.biot.greens import MU0, second_moments, section_centroid, third_moments
from nova.biot.polygonarc import polygon_arc_greens

ARC_FAR_LIMIT = 32.0
"""Base exact/filament seam in section radii of swept-volume distance."""

_FILAMENT_NODES = 128
_MOMENT_STEP = 5.0e-2


@lru_cache(maxsize=16)
def _arc_rule(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the Gauss-Legendre rule on the unit interval."""
    point, weight = leggauss(nodes)
    return 0.5 * (point + 1.0), 0.5 * weight


def arc_filament_greens(
    target_r: np.ndarray,
    target_z: np.ndarray,
    target_phi: np.ndarray,
    source_r: float,
    source_z: float,
    start: float,
    end: float,
    *,
    nodes: int = _FILAMENT_NODES,
) -> tuple[np.ndarray, ...]:
    """Return ``(A_r, A_phi, B_r, B_phi, B_z)`` for a circular arc filament.

    The fixed one-dimensional rule is independent of the finite section and is
    used only in its far field.  It integrates the Biot-Savart line expressions
    directly in the target's cylindrical basis.
    """
    radius = np.abs(np.asarray(target_r, dtype=np.float64))
    shape = radius.shape
    height = np.broadcast_to(np.asarray(target_z, dtype=np.float64), shape)
    azimuth = np.broadcast_to(np.asarray(target_phi, dtype=np.float64), shape)
    point, weight = _arc_rule(nodes)
    theta = start + (end - start) * point
    delta = azimuth[..., None] - theta
    cosine = np.cos(delta)
    sine = np.sin(delta)
    dz = height[..., None] - source_z
    distance2 = (
        radius[..., None] ** 2
        + source_r**2
        - 2.0 * radius[..., None] * source_r * cosine
        + dz**2
    )
    distance = np.sqrt(distance2)
    factor = MU0 / (4.0 * np.pi) * (end - start)
    weighted = weight / distance
    field_weighted = weight / (distance2 * distance)
    ar = factor * source_r * np.sum(sine * weighted, axis=-1)
    aphi = factor * source_r * np.sum(cosine * weighted, axis=-1)
    br = factor * source_r * np.sum(dz * cosine * field_weighted, axis=-1)
    bphi = -factor * source_r * np.sum(dz * sine * field_weighted, axis=-1)
    bz = (
        factor
        * source_r
        * np.sum((source_r - radius[..., None] * cosine) * field_weighted, axis=-1)
    )
    return tuple(value.reshape(shape) for value in (ar, aphi, br, bphi, bz))


def rms_radius(vertices: np.ndarray) -> float:
    """Return ``sqrt(mean(r**2))`` over the polygon area."""
    centre = section_centroid(vertices)
    radial_moment, _, _ = second_moments(vertices)
    return float(np.sqrt(centre[0] ** 2 + radial_moment))


def arc_far_limit(vertices: np.ndarray) -> float:
    """Return the section's exact/filament seam in its own bounding radii.

    The third-order filament leaves a fourth-order shape residual.  Near-isotropic
    sections clear the measured envelope at :data:`ARC_FAR_LIMIT`; a high-aspect
    plate carries a larger directional residual and widens the seam by the
    square-root scaling measured across the arc's thin-plate acceptance geometry.
    """
    extent = np.ptp(np.asarray(vertices, dtype=np.float64), axis=0)
    positive = extent[extent > 0.0]
    aspect = float(np.max(positive) / np.min(positive))
    return ARC_FAR_LIMIT * max(1.0, np.sqrt(aspect / 4.0))


def arc_moment_filament(
    target_r: np.ndarray,
    target_z: np.ndarray,
    target_phi: np.ndarray,
    vertices: np.ndarray,
    start: float,
    end: float,
    *,
    placement: str = "centroid",
    corrected: bool = True,
    nodes: int = _FILAMENT_NODES,
    order: int = 3,
) -> tuple[np.ndarray, ...]:
    """Return a finite-arc filament carrying the section's area moments.

    ``placement`` is ``"centroid"`` or ``"rms"``.  With ``corrected=False`` the
    chosen filament is returned bare.  Otherwise the source-position Taylor
    expansion carries the first moment and the complete second- and third-moment
    tensors about that placement.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    centre = section_centroid(vertices)
    if placement == "centroid":
        source = centre
    elif placement == "rms":
        source = np.array([rms_radius(vertices), centre[1]])
    else:
        raise ValueError("placement must be 'centroid' or 'rms'")

    evaluated: dict[tuple[int, int], np.ndarray] = {}

    def at(dr: int, dz: int, step: float) -> np.ndarray:
        if (dr, dz) not in evaluated:
            evaluated[dr, dz] = np.stack(
                arc_filament_greens(
                    target_r,
                    target_z,
                    target_phi,
                    source[0] + dr * step,
                    source[1] + dz * step,
                    start,
                    end,
                    nodes=nodes,
                )
            )
        return evaluated[dr, dz]

    extent = vertices - centre
    section_radius = float(np.max(np.hypot(extent[:, 0], extent[:, 1])))
    step = _MOMENT_STEP * section_radius
    value = at(0, 0, step)
    if not corrected:
        return tuple(value)

    radial_moment, vertical_moment, cross_moment = second_moments(vertices)
    displacement = centre - source
    radial_about = radial_moment + displacement[0] ** 2
    vertical_about = vertical_moment + displacement[1] ** 2
    cross_about = cross_moment + displacement[0] * displacement[1]
    radial_gradient = (at(1, 0, step) - at(-1, 0, step)) / (2.0 * step)
    vertical_gradient = (at(0, 1, step) - at(0, -1, step)) / (2.0 * step)
    radial_curvature = (at(1, 0, step) - 2.0 * value + at(-1, 0, step)) / step**2
    vertical_curvature = (at(0, 1, step) - 2.0 * value + at(0, -1, step)) / step**2
    cross_curvature = (
        at(1, 1, step) - at(1, -1, step) - at(-1, 1, step) + at(-1, -1, step)
    ) / (4.0 * step**2)
    corrected_value = (
        value
        + displacement[0] * radial_gradient
        + displacement[1] * vertical_gradient
        + 0.5 * radial_about * radial_curvature
        + cross_about * cross_curvature
        + 0.5 * vertical_about * vertical_curvature
    )
    if order >= 3:
        (
            radial_third,
            radial_radial_vertical,
            radial_vertical_vertical,
            vertical_third,
        ) = third_moments(vertices)
        dr, dz = displacement
        radial_third += 3.0 * dr * radial_moment + dr**3
        radial_radial_vertical += (
            dz * radial_moment + 2.0 * dr * cross_moment + dr * dr * dz
        )
        radial_vertical_vertical += (
            dr * vertical_moment + 2.0 * dz * cross_moment + dr * dz * dz
        )
        vertical_third += 3.0 * dz * vertical_moment + dz**3
        if (
            max(
                abs(radial_third),
                abs(radial_radial_vertical),
                abs(radial_vertical_vertical),
                abs(vertical_third),
            )
            > 1.0e-12 * section_radius**3
        ):
            third_correction = (
                radial_third
                * (
                    at(2, 0, step)
                    - 2.0 * at(1, 0, step)
                    + 2.0 * at(-1, 0, step)
                    - at(-2, 0, step)
                )
                + vertical_third
                * (
                    at(0, 2, step)
                    - 2.0 * at(0, 1, step)
                    + 2.0 * at(0, -1, step)
                    - at(0, -2, step)
                )
                + 3.0
                * radial_radial_vertical
                * (
                    at(1, 1, step)
                    - 2.0 * at(0, 1, step)
                    + at(-1, 1, step)
                    - at(1, -1, step)
                    + 2.0 * at(0, -1, step)
                    - at(-1, -1, step)
                )
                + 3.0
                * radial_vertical_vertical
                * (
                    at(1, 1, step)
                    - 2.0 * at(1, 0, step)
                    + at(1, -1, step)
                    - at(-1, 1, step)
                    + 2.0 * at(-1, 0, step)
                    - at(-1, -1, step)
                )
            )
            corrected_value += third_correction / (12.0 * step**3)
    return tuple(corrected_value)


def _section_distance(
    target_r: np.ndarray, target_z: np.ndarray, vertices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return contour distance, nearest contour radius, and inside-polygon mask."""
    target_r = np.asarray(target_r, dtype=np.float64)
    target_z = np.asarray(target_z, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    flat_r, flat_z = target_r.ravel(), target_z.ravel()
    start = vertices[:, None, :]
    edge = (np.roll(vertices, -1, axis=0) - vertices)[:, None, :]
    offset = np.stack([flat_r, flat_z], axis=-1)[None, :, :] - start
    length2 = np.sum(edge * edge, axis=-1)
    reach = np.clip(
        np.sum(offset * edge, axis=-1) / np.where(length2 > 0.0, length2, 1.0),
        0.0,
        1.0,
    )
    nearest = start + reach[..., None] * edge
    gap = offset - reach[..., None] * edge
    distance2 = np.sum(gap * gap, axis=-1)
    edge_index = np.argmin(distance2, axis=0)
    column = np.arange(flat_r.size)
    contour = np.sqrt(distance2[edge_index, column])
    nearest_r = nearest[edge_index, column, 0]

    r0, z0 = vertices.T
    r1, z1 = np.roll(vertices, -1, axis=0).T
    crosses = (z0[:, None] > flat_z) != (z1[:, None] > flat_z)
    crossing_r = (r1 - r0)[:, None] * (flat_z - z0[:, None]) / np.where(
        z1[:, None] != z0[:, None], (z1 - z0)[:, None], 1.0
    ) + r0[:, None]
    inside = np.count_nonzero(crosses & (flat_r < crossing_r), axis=0) % 2 == 1
    shape = target_r.shape
    return contour.reshape(shape), nearest_r.reshape(shape), inside.reshape(shape)


def arc_contour_distance(
    target_r: np.ndarray,
    target_z: np.ndarray,
    target_phi: np.ndarray,
    vertices: np.ndarray,
    start: float,
    end: float,
) -> np.ndarray:
    """Return distance to the finite swept section, including either end [m].

    Outside the angular span, the poloidal gap to the section and the azimuthal
    chord to the nearer end are orthogonal contributions.  The chord uses the
    closest poloidal point's radius; for a target whose poloidal projection lies
    inside the section it uses the target radius and the poloidal term is zero.
    """
    target_r = np.abs(np.asarray(target_r, dtype=np.float64))
    target_z = np.broadcast_to(np.asarray(target_z, dtype=np.float64), target_r.shape)
    target_phi = np.broadcast_to(
        np.asarray(target_phi, dtype=np.float64), target_r.shape
    )
    contour, nearest_r, inside_section = _section_distance(target_r, target_z, vertices)
    sweep = end - start
    span = min(abs(sweep), 2.0 * np.pi)
    direction = 1.0 if sweep >= 0.0 else -1.0
    relative = np.remainder(direction * (target_phi - start), 2.0 * np.pi)
    within = relative <= span
    start_gap = np.abs(
        np.arctan2(np.sin(target_phi - start), np.cos(target_phi - start))
    )
    end_gap = np.abs(np.arctan2(np.sin(target_phi - end), np.cos(target_phi - end)))
    angle_gap = np.minimum(start_gap, end_gap)
    poloidal = np.where(inside_section, 0.0, contour)
    source_r = np.where(inside_section, target_r, nearest_r)
    chord = np.sqrt(
        np.maximum(2.0 * target_r * source_r * (1.0 - np.cos(angle_gap)), 0.0)
    )
    outside = np.hypot(poloidal, chord)
    return np.where(within, contour, outside)


def arc_band(
    target_r: np.ndarray,
    target_z: np.ndarray,
    target_phi: np.ndarray,
    vertices: np.ndarray,
    start: float,
    end: float,
    *,
    far_limit: float | None = None,
) -> np.ndarray:
    """Return zero for exact pairs and one for moment-filament pairs."""
    if far_limit is None:
        far_limit = arc_far_limit(vertices)
    centre = section_centroid(vertices)
    offset = np.asarray(vertices, dtype=np.float64) - centre
    radius = float(np.max(np.hypot(offset[:, 0], offset[:, 1])))
    distance = arc_contour_distance(
        target_r, target_z, target_phi, vertices, start, end
    )
    return (distance >= far_limit * radius).astype(np.int_)


def banded_arc_greens(
    target_r: np.ndarray,
    target_z: np.ndarray,
    target_phi: np.ndarray,
    vertices: np.ndarray,
    start: float,
    end: float,
    *,
    far_limit: float | None = None,
    placement: str = "centroid",
) -> tuple[np.ndarray, ...]:
    """Return each finite-arc pair through the exact or moment-filament route."""
    target_r = np.asarray(target_r, dtype=np.float64)
    target_z = np.broadcast_to(np.asarray(target_z, dtype=np.float64), target_r.shape)
    target_phi = np.broadcast_to(
        np.asarray(target_phi, dtype=np.float64), target_r.shape
    )
    assignment = arc_band(
        target_r,
        target_z,
        target_phi,
        vertices,
        start,
        end,
        far_limit=far_limit,
    )
    rows = np.empty((5,) + target_r.shape)
    near = assignment == 0
    if near.any():
        rows[:, near] = np.stack(
            polygon_arc_greens(
                target_r[near],
                target_z[near],
                target_phi[near],
                vertices,
                start,
                end,
            )
        )
    far = assignment == 1
    if far.any():
        rows[:, far] = np.stack(
            arc_moment_filament(
                target_r[far],
                target_z[far],
                target_phi[far],
                vertices,
                start,
                end,
                placement=placement,
            )
        )
    return tuple(rows)


__all__ = [
    "ARC_FAR_LIMIT",
    "arc_band",
    "arc_contour_distance",
    "arc_far_limit",
    "arc_filament_greens",
    "arc_moment_filament",
    "banded_arc_greens",
    "rms_radius",
]
