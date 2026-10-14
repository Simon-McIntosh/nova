"""Columnar conductor, section and sensor descriptions for the circuit tier.

A conductor set is a flat table of axisymmetric rectangular-section filaments
grouped into CIRCUITS.  A circuit is one electrical path: every filament in it
carries the same current up to its own fixed ``current_share`` (the turn
multiplier for a wound circuit, the parallel-path share for a subdivided
shell).  All geometry is raw SI; the radial coordinate is the section centroid's
major radius.

The circuit membership and the drive-channel wiring are INPUTS here.  Deciding
which conductors form a circuit, and which circuits are driven rather than
inferred, is a machine-description concern that belongs upstream -- this module
only carries the result, so the circuit models stay machine-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PolygonSection:
    """Exact section shape overriding one circuit's rectangular bounding box.

    A wired conductor whose true cross-section is a parallelogram (sheared
    crowns, angled arms) links flux differently from the axis-aligned box that
    bounds it.  ``vertices`` is the (n, 2) array of ``(r, z)`` corners in either
    orientation with no repeated closing vertex; ``current_share`` scales the
    whole section as the filament shares do.  The section AREA is preserved by
    construction, so ring resistance and the size scale are unaffected -- only
    the linkage is reshaped.
    """

    circuit: int
    vertices: np.ndarray
    current_share: float = 1.0


@dataclass(frozen=True)
class ConductorSet:
    """Rectangular-section toroidal filaments grouped into circuits.

    All arrays are ``(n_filaments,)``: ``r``/``z`` the section centroid, ``dr``/
    ``dz`` its radial/vertical extents [m], ``current_share`` the filament's
    fixed share of its circuit current, and ``circuit`` the integer circuit it
    belongs to.  ``polygon_sections`` optionally overrides individual circuits'
    section shape (see :class:`PolygonSection`).
    """

    r: np.ndarray
    z: np.ndarray
    dr: np.ndarray
    dz: np.ndarray
    current_share: np.ndarray
    circuit: np.ndarray
    polygon_sections: tuple[PolygonSection, ...] = field(default_factory=tuple)

    def __post_init__(self):
        """Coerce the columns to float64 / int64 arrays and check they align."""
        for name in ("r", "z", "dr", "dz", "current_share"):
            object.__setattr__(
                self, name, np.asarray(getattr(self, name), dtype=np.float64)
            )
        object.__setattr__(self, "circuit", np.asarray(self.circuit, dtype=np.int64))
        sizes = {getattr(self, name).size for name in self._columns}
        if len(sizes) != 1:
            raise ValueError(f"conductor columns disagree in length: {sizes}")
        object.__setattr__(self, "polygon_sections", tuple(self.polygon_sections))

    _columns = ("r", "z", "dr", "dz", "current_share", "circuit")

    @property
    def n_filaments(self) -> int:
        """Return the number of filaments in the set."""
        return int(self.r.size)

    @property
    def circuits(self) -> np.ndarray:
        """Return the sorted unique circuit ids."""
        return np.unique(self.circuit)

    def rows(self, circuits) -> list[np.ndarray]:
        """Return the filament row indices of each circuit, in the given order."""
        return [np.flatnonzero(self.circuit == int(c)) for c in circuits]

    def polygon_by_circuit(self) -> dict[int, PolygonSection]:
        """Return the section-shape override of each circuit that has one."""
        return {int(ps.circuit): ps for ps in self.polygon_sections}

    def section_scale(self, circuits) -> np.ndarray:
        """Conducting cross-section scale ``sqrt(sum|dr dz|)`` per circuit [m].

        The geometric size of a circuit, used to normalise the adjacency
        neighbour rule -- a dimensionless comparison that transfers across
        machines rather than a metre-level threshold.
        """
        return np.array(
            [
                np.sqrt(np.sum(np.abs(self.dr[rows] * self.dz[rows])))
                for rows in self.rows(circuits)
            ]
        )

    def centroids(self, circuits) -> tuple[np.ndarray, np.ndarray]:
        """Current-share-weighted ``(r, z)`` centroid of each circuit [m]."""
        cr, cz = [], []
        for rows in self.rows(circuits):
            weight = np.abs(self.current_share[rows])
            total = max(float(weight.sum()), 1e-30)
            cr.append(float(np.sum(weight * self.r[rows]) / total))
            cz.append(float(np.sum(weight * self.z[rows]) / total))
        return np.array(cr), np.array(cz)


@dataclass(frozen=True)
class ThicknessSubdivision:
    """Independent skin-current layers and their unsplit parent circuits."""

    conductors: ConductorSet
    circuits: np.ndarray
    parent: np.ndarray
    resistance_factor: np.ndarray


def subdivide_passive_thickness(
    conductors: ConductorSet,
    passive_circuits,
    thickness_m: np.ndarray,
    max_layer_thickness_m: float,
) -> ThicknessSubdivision:
    """Split each passive element normal to its wall into toroidal rings.

    Each strip is an independent current path.  For equal strips its resistance
    is the source ring resistance times the strip count; its parent mapping
    lets the caller carry source measured resistances without replacing them
    with a nominal material value.  Polygon walls retain their true shape.
    """
    if not np.isfinite(max_layer_thickness_m) or max_layer_thickness_m <= 0:
        raise ValueError("maximum layer thickness must be positive")
    passive = np.asarray(list(passive_circuits), dtype=np.int64)
    thickness = np.asarray(thickness_m, dtype=np.float64)
    if (
        thickness.shape != passive.shape
        or np.any(~np.isfinite(thickness))
        or np.any(thickness <= 0)
    ):
        raise ValueError("one positive thickness is required per passive circuit")
    if len(set(passive)) != len(passive):
        raise ValueError("passive circuits must be unique")
    rows = conductors.rows(passive)
    if any(len(group) != 1 for group in rows):
        raise ValueError("each passive circuit must have one source element")
    polygon = conductors.polygon_by_circuit()
    columns = {name: [] for name in conductors._columns}
    polygons = [
        section
        for section in conductors.polygon_sections
        if section.circuit not in passive
    ]
    for row in range(conductors.n_filaments):
        if int(conductors.circuit[row]) not in passive:
            for name in columns:
                columns[name].append(getattr(conductors, name)[row])
    next_circuit = int(np.max(conductors.circuit)) + 1
    circuits, parents, factors = [], [], []
    for parent, source_rows, wall in zip(passive, rows, thickness, strict=True):
        row = int(source_rows[0])
        count = int(np.ceil(wall / max_layer_thickness_m))
        source_polygon = polygon.get(int(parent))
        for layer in range(count):
            circuit = next_circuit
            next_circuit += 1
            values = {name: getattr(conductors, name)[row] for name in columns}
            values["circuit"] = circuit
            values["current_share"] = 1.0
            if source_polygon is not None:
                vertices = source_polygon.vertices
                if vertices.shape != (4, 2):
                    raise ValueError("passive wall polygon must have four corners")
                low, high = layer / count, (layer + 1) / count
                first = vertices[0] * (1 - low) + vertices[3] * low
                second = vertices[1] * (1 - low) + vertices[2] * low
                third = vertices[1] * (1 - high) + vertices[2] * high
                fourth = vertices[0] * (1 - high) + vertices[3] * high
                stripe = np.array([first, second, third, fourth])
                extent = np.ptp(stripe, axis=0)
                values.update(
                    r=float(np.mean(stripe[:, 0])),
                    z=float(np.mean(stripe[:, 1])),
                    dr=float(extent[0]),
                    dz=float(extent[1]),
                )
                polygons.append(PolygonSection(circuit, stripe))
            elif abs(float(values["dr"])) <= abs(float(values["dz"])):
                values["dr"] = float(values["dr"]) / count
                values["r"] = (
                    float(values["r"]) + (layer + 0.5 - count / 2) * values["dr"]
                )
            else:
                values["dz"] = float(values["dz"]) / count
                values["z"] = (
                    float(values["z"]) + (layer + 0.5 - count / 2) * values["dz"]
                )
            for name in columns:
                columns[name].append(values[name])
            circuits.append(circuit)
            parents.append(parent)
            factors.append(count)
    return ThicknessSubdivision(
        conductors=ConductorSet(**columns, polygon_sections=tuple(polygons)),
        circuits=np.asarray(circuits, dtype=np.int64),
        parent=np.asarray(parents, dtype=np.int64),
        resistance_factor=np.asarray(factors, dtype=np.float64),
    )


@dataclass(frozen=True)
class SensorSet:
    """Magnetic sensor positions and orientations for per-ampere signatures.

    ``r``/``z`` are the measurement points [m]; ``angle`` the probe's measuring
    direction in the poloidal plane [rad], measured from the ``+r`` axis so the
    reading is ``B_r cos(angle) + B_z sin(angle)``; ``is_flux`` selects the
    channels that read total poloidal flux [Wb] instead of a field component,
    for which ``angle`` is ignored.
    """

    r: np.ndarray
    z: np.ndarray
    angle: np.ndarray
    is_flux: np.ndarray

    def __post_init__(self):
        """Coerce the columns and check they align."""
        for name in ("r", "z", "angle"):
            object.__setattr__(
                self, name, np.asarray(getattr(self, name), dtype=np.float64)
            )
        object.__setattr__(self, "is_flux", np.asarray(self.is_flux, dtype=bool))
        sizes = {getattr(self, name).size for name in ("r", "z", "angle", "is_flux")}
        if len(sizes) != 1:
            raise ValueError(f"sensor columns disagree in length: {sizes}")

    @property
    def n_sensors(self) -> int:
        """Return the number of sensor channels."""
        return int(self.r.size)

    def project(self, psi: np.ndarray, b_r: np.ndarray, b_z: np.ndarray) -> np.ndarray:
        """Return each channel's reading from per-sensor ``(psi, b_r, b_z)``."""
        return np.where(
            self.is_flux,
            psi,
            b_r * np.cos(self.angle) + b_z * np.sin(self.angle),
        )


__all__ = [
    "ConductorSet",
    "PolygonSection",
    "SensorSet",
    "ThicknessSubdivision",
    "subdivide_passive_thickness",
]
