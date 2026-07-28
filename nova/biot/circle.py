"""Biot-Savart coupling for complete circular filaments of finite section.

A point filament ring is log-singular on its own location, so a target that
approaches one -- a neighbouring coil turn, a plasma cell's neighbour, or the cell
itself on the diagonal of an all-to-all matrix -- has asked the filament a question
it cannot answer. :attr:`nova.biot.constants.Constants.K` returns the divergence
there, which is the right answer for a filament and not an available one for an
operator whose every diagonal entry is exactly that configuration.

What the diagonal IS
--------------------
Not a filament value at all. Each source element carries its own cross-section
polygon, and a target inside that section is an ORDINARY INTERIOR POINT for a
finite section: flux and both field components are bounded and smooth there with
nothing to approximate. Two integrals are involved and they are different
quantities, so this element names both:

* the SINGLE integral -- the source's current spread uniformly over its true
  section, evaluated at one target POINT. Exact in closed form
  (:mod:`nova.biot.polygonanalytic`), finite on the section contour and on a
  vertex.
* the DOUBLE integral -- that single integral averaged over the TARGET element's
  own section. This is the quantity an inductance operator wants: the flux a
  conductor links is the mean over the area its current occupies, and the force it
  feels is set by the mean field there. The single integral at a point is the
  midpoint rule for it, and on the coincident term the two differ by the whole gap
  between a section's arithmetic mean logarithmic distance and its geometric mean
  distance -- measured +7.1% of the self flux for a coil filament of unit aspect
  ratio, and it does not vanish as the section shrinks. See
  :mod:`nova.biot.sectionaverage`.

**The diagonal is the double integral**, and it is a correction rather than a
restatement: it converges onto the published geometric-mean-distance ring
inductance ``mu_0 R (ln(8 R/GMD) - 2)`` as that formula's own ``(a/R)^2`` error
allows -- 1.3e-06 relative at ``w/R = 0.005`` for a square section against the
tabulated ``GMD = 0.447049 w`` -- while carrying no shape assumption, so it is
equally right for a slender rectangle, a hexagon and a wall-clipped cell, and it
is defined for the field components, where a mean distance means nothing.

A target that declares no section of its own -- a grid point, a field probe -- is a
point, and takes the single integral. That is a property of what the target IS and
not of where it sits, so no entry of the operator jumps as a probe crosses a
conductor boundary.

The bands
---------
Both treatments are banded on the source section's own bounding radius about its
area centroid, and outside a band the point filament at the section's
root-mean-square radius stands as before. Neither band closes to zero error: for a
FULL RING the finite-section correction is set by the major radius rather than by
the distance to the target, so a bare filament does not converge to a section at
any standoff and both seams flatten onto a floor of order ``(a/R0)^2`` -- 1.7e-05
of the self flux for a coil filament, 7.8e-06 for a plasma cell. The bands are
therefore placed where the term they carry has decayed onto that floor, and the
seam is the floor:

* ``section_band`` (4 radii) -- the single integral, one kernel evaluation a pair.
  Measured deviation from the double integral at the seam: 3.4e-04 of the self flux
  for a coil filament, 5.4e-05 for a plasma cell, against floors of 1.7e-05 and
  7.8e-06.
* ``average_band`` (1.5 radii) -- the target-section average, which costs a
  quadrature over the target section instead of one evaluation and so is the band
  COST sets. It is placed inside a hexagonal tiling's first-neighbour separation of
  ``sqrt(3)`` circumradii, which leaves the diagonal averaged and a plasma cell's
  own neighbours on the single integral. What that leaves on the table, measured
  against the double integral as a fraction of the self flux: 1.5e-03 at 1.5 radii
  and 8.6e-04 at 2 for a coil filament, 8.8e-04 and 9.1e-05 for a plasma cell, and
  about 3e-04 at the neighbour separation itself. Widening it to 2 radii multiplies
  a hexagonally-tiled plasma-grid build by about seven, which is why it is not the
  default; a caller who needs the wider band can raise it.

A principled error-BOUNDED cutoff remains open, exactly as
:attr:`nova.biot.polysection.PolySection.standoff` records; these bands are placed
on measurement, and the measurements are in :mod:`tests.test_biotcircle`.

The corner-level convention is NOT restated here. It is contracted in
:mod:`nova.biot.constants` -- ``gamma`` held at one where a target is level with a
source corner, the bounded product assigned the mean of its two one-sided limits so
the reduction stays continuous across its own corner planes -- and both kernels
this element routes through inherit it.
"""

from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import numpy as np

from nova.biot.constants import Constants
from nova.biot.greens import section_centroid
from nova.biot.matrix import Matrix
from nova.biot.polygonanalytic import polygon_analytic_greens
from nova.biot.sectionaverage import section_nodes


def _polygon(poly) -> np.ndarray:
    """Return a frame polygon's ``(n, 2)`` r-z corners, closing vertex dropped."""
    points = np.asarray(poly.points, dtype=np.float64)[:, [0, 2]]
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    return points


def _sections(frame) -> list[np.ndarray] | None:
    """Return each element's section polygon, or ``None`` where the frame has none.

    A frame that carries no ``poly`` column, or carries one an element left unset,
    describes points rather than conductors and is treated as such.
    """
    if "poly" not in frame.columns.to_list():
        return None
    poly = np.asarray(frame["poly"])
    if any(not hasattr(entry, "points") for entry in poly):
        return None
    return [_polygon(entry) for entry in poly]


@dataclass
class Circle(Constants, Matrix):
    """
    Extend base class.

    Compute interaction for complete circular filaments of finite cross-section.

    """

    attrs: dict[str, str] = field(
        default_factory=lambda: {
            "rs": "rms",
            "zs": "z",
            "x": "x",
            "y": "y",
            "z": "z",
        }
    )

    axisymmetric: ClassVar[bool] = True
    name: ClassVar[str] = "circle"  # element name

    section_band: ClassVar[float] = 4.0
    """Source-section radii within which the source is its true section.

    Beyond it the point filament at the section's root-mean-square radius stands.
    See the module docstring for the seam this leaves and what sets the value.
    """

    average_band: ClassVar[float] = 1.5
    """Source-section radii within which the target's own section is averaged over.

    Applies only where the target frame declares a section; a bare point target is
    a point at any distance. Never wider than ``section_band``: averaging a
    filament source over a target section would mix a treatment the band edge has
    already left behind.
    """

    def __post_init__(self):
        """Load intergration constants."""
        super().__post_init__()
        self.data["r"] = np.linalg.norm([self["x"], self["y"]], axis=0)
        for attr in ["rs", "zs", "r", "z"]:
            setattr(self, attr, self.data[attr])

    @cached_property
    def _filament(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(psi, Br, Bz)`` per ampere from the point filament ring.

        Evaluated on every pair, including the ones a finite section then claims, so
        the divergence a coincident target drives the modulus complement to is
        allowed to appear and is overwritten rather than avoided. The vertical row
        takes its second-kind weight from
        :attr:`nova.biot.constants.Constants.axial_weight`, the arrangement that
        does not form ``2 r - b k^2`` as a difference.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            aphi = (
                1
                / (2 * np.pi)
                * self.a
                / self.r
                * ((1 - self.k2 / 2) * self.K - self.E)
            )
            psi = 2 * np.pi * self.mu_0 * self.r * aphi
            br = (
                self.mu_0
                / (2 * np.pi)
                * self.gamma
                * (self.K - (2 - self.k2) / (2 * self.ck2) * self.E)
                / (self.a * self.r)
            )
            bz = (
                self.mu_0
                / (2 * np.pi)
                * (self.r * self.K - self.axial_weight / (2 * self.ck2) * self.E)
                / (self.a * self.r)
            )
        return psi, br, bz

    @cached_property
    def _coupling(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(psi, Br, Bz)`` per ampere, shape ``(target, source)``.

        The filament on every pair, then the finite section on the pairs the bands
        claim. A pair inside ``average_band`` is a double integral, one inside
        ``section_band`` alone is a single integral, and no pair is both: the two
        sets are disjoint by construction, so a pair is evaluated by exactly one
        treatment.

        The source column's own polygon is handed the point targets and every
        target-section quadrature node in ONE kernel call. The closed form holds its
        corner parts live and amortises them across a call, so a per-pair or
        per-treatment call would pay that build repeatedly; the weighted means are
        formed from the same nodes :mod:`nova.biot.sectionaverage` supplies, and
        :mod:`tests.test_biotcircle` holds the diagonal this produces against
        :func:`nova.biot.sectionaverage.averaged_greens` so the two cannot drift.
        """
        coupling = tuple(value.copy() for value in self._filament)
        source = _sections(self.source)
        if source is None:
            return coupling
        target = _sections(self.target)
        for column, vertices in enumerate(source):
            centre = section_centroid(vertices)
            radius = float(np.max(np.hypot(*(vertices - centre).T)))
            target_r = self.r[:, column]
            target_z = self.z[:, column]
            distance = np.hypot(target_r - centre[0], target_z - centre[1])
            inside = distance < self.section_band * radius
            averaged = (
                inside & (distance < self.average_band * radius)
                if target is not None
                else np.zeros(inside.shape, dtype=bool)
            )
            rows = np.flatnonzero(inside & ~averaged)
            section = [section_nodes(target[row]) for row in np.flatnonzero(averaged)]
            node = [target_r[rows], *(point[:, 0] for point, _ in section)]
            height = [target_z[rows], *(point[:, 1] for point, _ in section)]
            if sum(len(part) for part in node) == 0:
                continue
            evaluated = polygon_analytic_greens(
                np.concatenate(node), np.concatenate(height), vertices
            )
            for value, got in zip(coupling, evaluated):
                value[rows, column] = got[: rows.size]
                start = rows.size
                for row, (_, weight) in zip(np.flatnonzero(averaged), section):
                    stop = start + weight.size
                    value[row, column] = weight @ got[start:stop] / weight.sum()
                    start = stop
        return coupling

    @cached_property
    def Psi(self):
        """Return the total poloidal flux array [Wb/A]."""
        return self._coupling[0]

    @cached_property
    def Aphi(self):
        """Return the toroidal vector potential array [Wb/(m.A)]."""
        return self.Psi / (2 * np.pi * self.mu_0 * self.r)

    @cached_property
    def Br(self):
        """Return the radial field array [T/A]."""
        return self._coupling[1]

    @cached_property
    def Bz(self):
        """Return the vertical field array [T/A]."""
        return self._coupling[2]


if __name__ == "__main__":
    from nova.frame.coilset import CoilSet

    coilset = CoilSet(dcoil=-100, dplasma=-150)
    coilset.coil.insert(
        5, 0.5, 0.01, 0.8, section="r", turn="r", nturn=300, segment="circle"
    )
    coilset.coil.insert(
        5.1, 0.5 + 0.4, 0.2, 0.01, section="r", turn="r", nturn=300, segment="circle"
    )
    coilset.coil.insert(
        5.1, 0.5 - 0.4, 0.2, 0.01, section="r", turn="r", nturn=300, segment="circle"
    )
    coilset.coil.insert(
        5.2, 0.5, 0.01, 0.8, section="r", turn="r", nturn=300, segment="circle"
    )
    coilset.saloc["Ic"] = 5e3

    coilset.grid.solve(2000, 1)
    coilset.grid.plot("psi", colors="C1")
    coilset.plot()
