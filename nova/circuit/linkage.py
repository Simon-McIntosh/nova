"""Two-section flux linkage between conductor circuits.

The mutual inductance of two conductors is the flux one links per ampere of the
other.  Linking centroid-to-centroid is exact only when the flux is uniform
across both cross-sections; the larger vessel elements and thin coil-case walls
violate that badly enough to matter, so the linkage here integrates BOTH
sections: the finite-area axisymmetric kernel
(:func:`nova.biot.greens.hybrid_greens`, or
:func:`nova.biot.polygon.polygon_greens` for a shaped section) integrates the
source cross-section exactly, and the observer side is averaged over a midpoint
sub-grid of the target cross-section.

The sub-gridding criterion is MACHINE-AGNOSTIC: sub-cells are sized at a
fraction of the conductor set's OWN median section scale ``median(sqrt(dr dz))``,
capped per dimension, so the rule transfers to another machine unchanged instead
of baking one machine's conductor sizes into a metre-level cell size.  Small
sections stay unsubdivided -- they see uniform flux, and centroid linking is
exact enough there.

The analytic-source / quadrature-observer estimate is symmetric only up to the
observer quadrature error, so a full linkage matrix is symmetrised before use;
the physical inductance matrix is symmetric by reciprocity.
"""

from __future__ import annotations

import numpy as np

from nova.biot.greens import hybrid_greens
from nova.biot.polygon import polygon_greens
from nova.circuit.conductor import ConductorSet

#: floor on the section extents fed to the source-side flux integration [m] --
#: a guard against a zero-thickness section, NEVER a conducting-area model (a
#: thin shell's resistance must use its TRUE thickness)
SECTION_FLOOR = 0.01

#: floor on the section extents used to PLACE observer sub-points [m]; the true
#: extents position the points, so a 3 mm wall's observers stay inside the wall
POSITION_FLOOR = 1e-4


def median_section_scale(dr: np.ndarray, dz: np.ndarray) -> float:
    """Median cross-section scale ``sqrt(dr dz)`` of a conductor set [m].

    The machine-intrinsic length that normalises the observer sub-gridding: a
    fixed metre-level cell size would bake one machine's conductor sizes into
    the linkage accuracy, so the subdivision criterion is expressed as a
    FRACTION of this scale and transfers across machines unchanged.
    """
    scale = np.sqrt(
        np.maximum(np.abs(dr), POSITION_FLOOR) * np.maximum(np.abs(dz), POSITION_FLOOR)
    )
    return float(np.median(scale))


def section_points(
    r: float, z: float, dr: float, dz: float, delta: float, n_max: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Midpoint sub-grid of one rectangular cross-section, equal-area weighted.

    A section smaller than ``delta`` in a dimension stays unsubdivided there; a
    larger one is split so sub-cells are at most ``delta`` across, capped at
    ``n_max`` per dimension.  Weights are the uniform current shares ``1/n``.
    The TRUE section extents place the points, so a thin wall's observer points
    stay within the wall (the source-side extent floor guards only the flux
    integration and must never smear the observer positions).
    """
    width = max(abs(dr), POSITION_FLOOR)
    height = max(abs(dz), POSITION_FLOOR)
    n_r = max(1, min(int(np.ceil(width / delta)), n_max))
    n_z = max(1, min(int(np.ceil(height / delta)), n_max))
    r_offsets = width * ((np.arange(n_r) + 0.5) / n_r - 0.5)
    z_offsets = height * ((np.arange(n_z) + 0.5) / n_z - 0.5)
    mesh_r, mesh_z = np.meshgrid(r + r_offsets, z + z_offsets)
    return (
        mesh_r.ravel(),
        mesh_z.ravel(),
        np.full(mesh_r.size, 1.0 / mesh_r.size),
    )


def section_grid(
    conductors: ConductorSet,
    groups: list[np.ndarray],
    delta: float,
    n_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenated cross-section sub-points of every filament group.

    Returns ``(r, z, weight, owner)`` -- the weight folds each filament's
    current share with its section quadrature weight; ``owner`` maps each point
    back to its group index, for a :func:`numpy.bincount` reduction.
    """
    point_r: list[np.ndarray] = []
    point_z: list[np.ndarray] = []
    weight: list[np.ndarray] = []
    owner: list[np.ndarray] = []
    for index, rows in enumerate(groups):
        for row in rows:
            sub_r, sub_z, sub_w = section_points(
                float(conductors.r[row]),
                float(conductors.z[row]),
                float(conductors.dr[row]),
                float(conductors.dz[row]),
                delta,
                n_max,
            )
            point_r.append(sub_r)
            point_z.append(sub_z)
            weight.append(float(conductors.current_share[row]) * sub_w)
            owner.append(np.full(sub_r.size, index, dtype=np.int64))
    return (
        np.concatenate(point_r),
        np.concatenate(point_z),
        np.concatenate(weight),
        np.concatenate(owner),
    )


def linked_flux_columns(
    conductors: ConductorSet,
    source_rows: np.ndarray,
    point_r: np.ndarray,
    point_z: np.ndarray,
    weight: np.ndarray,
    owner: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """Flux linkage [Wb/A] of one source group into every observer group.

    The finite-area kernel integrates the SOURCE cross-section exactly; the
    observer side is the section-averaged flux over each filament's sub-grid.
    """
    column = np.zeros(n_groups)
    for row in source_rows:
        psi, _b_r, _b_z = hybrid_greens(
            point_r,
            point_z,
            float(conductors.r[row]),
            float(conductors.z[row]),
            max(abs(float(conductors.dr[row])), SECTION_FLOOR),
            max(abs(float(conductors.dz[row])), SECTION_FLOOR),
        )
        column += float(conductors.current_share[row]) * np.bincount(
            owner, weights=weight * psi, minlength=n_groups
        )
    return column


def polygon_flux_columns(
    section, point_r, point_z, weight, owner, n_groups: int
) -> np.ndarray:
    """Flux linkage [Wb/A] of a shaped source section into every observer group.

    The exact polygon-section kernel replaces the rectangular bounding box: same
    area, shaped mutual and self linkage.  The observer sub-grid stays the
    axis-aligned box -- the flux another circuit links across these small
    sections varies negligibly over the shear.
    """
    psi, _b_r, _b_z = polygon_greens(point_r, point_z, section.vertices)
    return section.current_share * np.bincount(
        owner, weights=weight * psi, minlength=n_groups
    )


def circuit_linkage_matrix(
    conductors: ConductorSet,
    circuits,
    *,
    section_scale_frac: float = 1.0,
    section_n_max: int = 6,
) -> np.ndarray:
    """Symmetric two-section flux linkage matrix of a circuit set [Wb/A].

    ``lmat[i, j]`` is the flux circuit ``i`` links per ampere of circuit ``j``,
    self terms included (the kernel is smooth inside conductors, so the self
    term needs no special casing).  The analytic-source / quadrature-observer
    estimate is symmetrised because the physical inductance matrix is.
    """
    circuits = list(circuits)
    groups = conductors.rows(circuits)
    n_circuits = len(groups)
    delta = section_scale_frac * median_section_scale(conductors.dr, conductors.dz)
    point_r, point_z, weight, owner = section_grid(
        conductors, groups, delta, section_n_max
    )
    polygon = conductors.polygon_by_circuit()
    lmat = np.zeros((n_circuits, n_circuits))
    for index, (circuit, rows) in enumerate(zip(circuits, groups, strict=True)):
        section = polygon.get(int(circuit))
        if section is not None:
            lmat[:, index] = polygon_flux_columns(
                section, point_r, point_z, weight, owner, n_circuits
            )
        else:
            lmat[:, index] = linked_flux_columns(
                conductors, rows, point_r, point_z, weight, owner, n_circuits
            )
    return 0.5 * (lmat + lmat.T)


def guard_positive_definite(lmat: np.ndarray, floor_frac: float = 1e-4) -> np.ndarray:
    """Clip a linkage matrix's spectrum to keep it positive definite.

    The observer quadrature can push a near-degenerate mode of an otherwise SPD
    matrix marginally negative; the physical inductance matrix cannot be, so the
    eigenvalues are floored at ``floor_frac`` of the largest before the
    generalised eigenproblem sees them.
    """
    values, vectors = np.linalg.eigh(lmat)
    floor = floor_frac * values.max()
    return (vectors * np.clip(values, floor, None)) @ vectors.T


def ring_resistance(
    conductors: ConductorSet, circuits, resistivity: float
) -> np.ndarray:
    """Nominal toroidal-ring resistance of each circuit [Ohm].

    ``2 pi r rho / (dr dz)`` per filament at its TRUE cross-section, combined
    with the ``current_share**2`` weights of parallel paths at fixed shares.
    The source-side section floor is a flux-integration guard and must NEVER
    inflate a thin shell's conducting area -- a 3 mm case wall carries 3.3x the
    resistance the floored area would give.
    """
    return np.array(
        [
            np.sum(
                2.0
                * np.pi
                * conductors.r[rows]
                * resistivity
                / np.maximum(np.abs(conductors.dr[rows] * conductors.dz[rows]), 1e-8)
                * conductors.current_share[rows] ** 2
            )
            for rows in conductors.rows(circuits)
        ]
    )


def _merged_channel_column(
    conductors: ConductorSet,
    circuits: list[int],
    point_r: np.ndarray,
    point_z: np.ndarray,
    weight: np.ndarray,
    owner: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """Flux column of one drive channel, averaging its redundant circuits."""
    columns = [
        linked_flux_columns(
            conductors,
            np.flatnonzero(conductors.circuit == int(circuit)),
            point_r,
            point_z,
            weight,
            owner,
            n_groups,
        )
        for circuit in sorted(circuits)
    ]
    return np.mean(np.asarray(columns), axis=0)


def channel_flux_linkage(
    conductors: ConductorSet,
    observer_circuits,
    channel_circuits: dict[str, list[int]],
    *,
    channel_gain: dict[str, float] | None = None,
    section_scale_frac: float = 1.0,
    section_n_max: int = 6,
) -> tuple[list[str], np.ndarray]:
    """Flux linked by each observer circuit per ampere of each drive channel.

    ``channel_circuits`` maps a measured drive channel name to the circuits it
    energises; redundant same-channel circuits are AVERAGED, matching how the
    coupling generator merges them into one channel.  ``channel_gain`` scales
    individual channels on the source side (a measured channel whose response
    differs from its nominal turn count).  Returns ``(channels, m)`` with
    ``m`` shaped ``(n_observers, n_channels)`` [Wb/A], channels sorted.
    """
    groups = conductors.rows(list(observer_circuits))
    delta = section_scale_frac * median_section_scale(conductors.dr, conductors.dz)
    point_r, point_z, weight, owner = section_grid(
        conductors, groups, delta, section_n_max
    )
    channels = sorted(channel_circuits)
    gain = channel_gain or {}
    linkage = np.column_stack(
        [
            _merged_channel_column(
                conductors,
                channel_circuits[channel],
                point_r,
                point_z,
                weight,
                owner,
                len(groups),
            )
            * gain.get(channel, 1.0)
            for channel in channels
        ]
    )
    return channels, linkage


def drive_linkage(
    conductors: ConductorSet,
    channel_circuits: dict[str, list[int]],
    *,
    channel_gain: dict[str, float] | None = None,
    section_scale_frac: float = 1.0,
    section_n_max: int = 6,
) -> tuple[list[str], np.ndarray]:
    """Flux linkage among the measured drive channels, self terms included.

    Returns ``(channels, lam)`` with ``lam[i, j]`` the flux channel ``i``'s
    circuits link per ampere(-turn) of channel ``j`` [Wb/A] -- the same
    two-section linkage as the passive system.  A same-channel merge that
    AVERAGES on the source side must also average on the observer side, so a
    channel carrying several redundant circuits is normalised by their count.
    The flux a winding links from every measured drive is the inductive part of
    its terminal voltage, which is what a galvanic case-wiring hypothesis reads.
    """
    channels = sorted(channel_circuits)
    groups = [
        np.concatenate(
            [
                np.flatnonzero(conductors.circuit == int(circuit))
                for circuit in sorted(channel_circuits[channel])
            ]
        )
        for channel in channels
    ]
    delta = section_scale_frac * median_section_scale(conductors.dr, conductors.dz)
    point_r, point_z, weight, owner = section_grid(
        conductors, groups, delta, section_n_max
    )
    gain = channel_gain or {}
    lam = np.column_stack(
        [
            _merged_channel_column(
                conductors,
                channel_circuits[channel],
                point_r,
                point_z,
                weight,
                owner,
                len(channels),
            )
            * gain.get(channel, 1.0)
            for channel in channels
        ]
    )
    merge_count = np.array([len(channel_circuits[ch]) for ch in channels])
    return channels, lam / merge_count[:, np.newaxis]


def sensor_grid_couplings(
    conductors: ConductorSet,
    circuits,
    sensors,
    grid_r: np.ndarray,
    grid_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-ampere sensor signatures and grid flux columns of a circuit set.

    Returns ``(a_sensor (n_sensors, n_circuits), g_grid (n_points, n_circuits))``
    -- what each sensor channel reads and what flux each grid point sees per
    ampere of circuit current, on the finite-area kernel throughout.  Pure
    geometry: build once per machine description and reuse across slices.  A
    circuit carrying a section-shape override uses the exact shaped field, the
    same substitution the linkage matrix applies.
    """
    grid_r = np.asarray(grid_r, dtype=np.float64)
    grid_z = np.asarray(grid_z, dtype=np.float64)
    polygon = conductors.polygon_by_circuit()
    sensor_columns, grid_columns = [], []
    for circuit in circuits:
        section = polygon.get(int(circuit))
        if section is not None:
            psi_s, b_r, b_z = polygon_greens(sensors.r, sensors.z, section.vertices)
            sensor_columns.append(
                section.current_share * sensors.project(psi_s, b_r, b_z)
            )
            grid_columns.append(
                section.current_share
                * polygon_greens(grid_r, grid_z, section.vertices)[0]
            )
            continue
        sensor_column = np.zeros(sensors.n_sensors)
        grid_column = np.zeros(grid_r.size)
        for row in np.flatnonzero(conductors.circuit == int(circuit)):
            width = max(abs(float(conductors.dr[row])), SECTION_FLOOR)
            height = max(abs(float(conductors.dz[row])), SECTION_FLOOR)
            share = float(conductors.current_share[row])
            centre = (float(conductors.r[row]), float(conductors.z[row]))
            psi_s, b_r, b_z = hybrid_greens(
                sensors.r, sensors.z, *centre, width, height
            )
            sensor_column += share * sensors.project(psi_s, b_r, b_z)
            grid_column += (
                share * hybrid_greens(grid_r, grid_z, *centre, width, height)[0]
            )
        sensor_columns.append(sensor_column)
        grid_columns.append(grid_column)
    return np.column_stack(sensor_columns), np.column_stack(grid_columns)


__all__ = [
    "POSITION_FLOOR",
    "SECTION_FLOOR",
    "channel_flux_linkage",
    "circuit_linkage_matrix",
    "drive_linkage",
    "guard_positive_definite",
    "linked_flux_columns",
    "median_section_scale",
    "polygon_flux_columns",
    "ring_resistance",
    "section_grid",
    "section_points",
    "sensor_grid_couplings",
]
