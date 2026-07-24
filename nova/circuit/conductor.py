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


__all__ = ["ConductorSet", "PolygonSection", "SensorSet"]
