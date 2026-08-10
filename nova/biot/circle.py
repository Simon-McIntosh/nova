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

What decides that the element evaluates that quantity correctly is neither of the
two candidates above but a third thing: a brute-force four-dimensional quadrature
over source section x target section, assembled from the elliptic integrals directly
and sharing no code with the closed form or with the target-side rule. Against its
Richardson limit the shipped rule lands within 3e-06 of the self term on a coil
filament, 2e-07 on a plasma cell and 2e-04 on a section ten times taller than it is
wide (:mod:`tests.test_biotsectionaverage`). What that settles is the RULE, and only
the rule.

What the current is assumed to do
---------------------------------
Every quantity above is the uniform-current limit: the element spreads an element's
current at CONSTANT DENSITY over the whole of its section polygon. There is no
jacket, no insulation, no cooling channel, no void and no turn structure inside a
section -- an ``nturn`` count scales the result and never subdivides the area.

That is a modelling choice and not a numerical detail, because it is not neutral in
sign. Concentrating the same current into discrete sub-conductors inside the same
outline lowers each conductor's own geometric mean distance and so RAISES the self
inductance; smearing it over the gross outline is therefore the LOWEST self
inductance that outline can carry. A real wound coil -- cable-in-conduit turns in
steel jackets, insulated from each other -- sits above this element's value by an
amount set by the winding pack, and none of the convergence figures quoted here
bear on that gap. A comparison against a machine inductance table is therefore a
comparison between two models, reported as a difference with neither side the
reference. Against the tabulated ITER PF1 / CS3U / CS2U self terms this element
reads LOW -- by 4.1e-03 H on PF1 and 4.7e-04 H on the two CS coils -- which is the
direction winding granularity predicts on both.

A first-order estimate of that effect, ``N mu_0 R ln(GMD_cell / GMD_conductor)`` for
one conductor per turn cell, closes the CS gap and overshoots it: 6.0e-04 to 8.2e-04
H for a 36 mm to 30 mm round conductor in the 51.9 mm turn cell, against 4.7e-04
observed. On PF1 the same estimate gives about 1.0e-03 against 4.1e-03 observed, and
reproducing that gap by granularity alone would need a 2.5 mm conductor in a 61.6 mm
cell, which is not a conductor. So granularity plausibly carries the CS difference
and does NOT carry PF1's, and what PF1's difference is remains open. Do not treat
either side of it as correct, and do not tune a quadrature rule towards the table.

A target that declares no section of its own -- a grid point, a field probe -- is a
point, and takes the single integral. That is a property of what the target IS and
not of where it sits, so no entry of the operator jumps as a probe crosses a
conductor boundary.

The bands
---------
Both treatments are banded on the source section's own bounding radius about its
area centroid, and outside a band the pair is the point filament at the section's
root-mean-square radius. Neither band closes to zero error: for a
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
* ``average_band`` (2 radii) -- the target-section average, which costs a quadrature
  over the target section instead of a single evaluation. Two radii is where two
  sections of equal bounding radius stop being able to TOUCH, so the band is every
  pair whose sections can overlap or meet -- which is the configuration in which the
  kernel's curvature over the target section is set by the source section rather than
  by the major radius, and the only one in which the midpoint rule fails at leading
  order. Measured deviation of the point target from the double integral, as a
  fraction of the self flux: 8.6e-04 at the seam for a coil filament and 9.1e-05 for
  a plasma cell, against 1.5e-03 and 8.8e-04 one half radius inside it. Two adjacent
  slender sections are what need the band that wide, and their price is paid on the
  MUTUAL term rather than on the self term: two undiscretised ITER CS sections sit
  1.94 radii apart, and their reduced mutual inductance misses the uniform-current
  double integral over both sections by 3.4e-02 H at a 1.5 band against 8.8e-06 at
  2 -- and by the same 8.8e-06 at 4, so two radii buys all of it and nothing above
  it is worth paying for.

The cost of the wider band is a build a half longer at worst, not the sevenfold its
pair count suggests. Measured at 1.5 radii against 2, as averaged pairs per source
column and as assembly seconds, on both populations this element is built over:

* an ITER PF and CS stack -- 4.0 to 6.3 pairs a column and 4.48 s to 5.25 s at
  ``dcoil = 0.25`` (142 elements), 4.6 to 7.8 and 32.8 s to 34.5 s at 0.1 (866),
  so the finer the discretisation the less the widening costs;
* a hexagonal tiling, where the shell structure is sharpest -- 1.6 to 6.4 pairs a
  column, the diagonal alone to the diagonal plus the first neighbour shell at
  ``sqrt(3)`` circumradii, and 12.6 s to 18.5 s at 206 cells, 20.7 s to 32.5 s at
  380.

The pairs the wider band converts were already inside ``section_band`` and already
paying a kernel evaluation, and the closed form holds its corner parts live across
one call, so the extra nodes cost far less than their count.

A principled error-bounded cutoff remains open. Polygon-section standoff is therefore
an explicit immutable study policy rather than a mutable product default; these
Circle bands remain measurement-based and are pinned in :mod:`tests.test_biotcircle`.

Which mechanism serves which path
---------------------------------
The target-side average runs only where the target frame declares a section, so the
two lanes that build an inductance operator reach the same quantity by different
routes and neither is a fallback for the other:

* a target frame WITH sections -- an all-to-all coil matrix, where source and
  target are one frame -- takes the double integral inside ``average_band``. Its
  own discretisation is then free: summing the double integral over a tiling of
  sub-sections reproduces the whole section's double integral identically, because
  the pair sum IS the area integral split up, so the reduced value does not depend on
  ``dcoil`` at all once both bands cover the coil.
* a target frame WITHOUT sections -- a grid point or field probe -- remains a point
  at any distance. Inductance instead expands strictly positive nodes over the actual
  material in every existing dcoil cell, contracts those nodes to that physical
  target cell, and only then applies turns and parent reduction. Measured on the ITER
  PF1/CS3U pair, the fixed-order target rule converges with the existing dcoil cells
  towards the independent uniform-current double integral.

Neither lane covers a tiled plasma grid, whose cells default to
``segment="polysection"``: exact on the source side for every pair and with no band,
but evaluated at the target cell's centre, so the plasma self term still carries the
whole point-target gap -- measured at +5.7% of the cell's own flux in
:mod:`benchmarks.plasma_coupling_accuracy`.

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
from nova.biot.greens import section_centroid, traced_filament_greens
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


def _inside_closed_band(
    distance: np.ndarray, radius: float, scale: float
) -> np.ndarray:
    """Return pairs inside a physically closed standoff band.

    Touching equal sections sit exactly on the averaging boundary. Independent
    centroid arithmetic can move the two directions of that pair by a few ulps, so
    include a roundoff-sized guard rather than making reciprocity depend on polygon
    vertex order.
    """
    boundary = scale * radius
    tolerance = 8.0 * np.finfo(np.float64).eps * np.maximum(distance, boundary)
    return distance <= boundary + tolerance


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

    average_band: ClassVar[float] = 2.0
    """Source-section radii within which the target's own section is averaged over.

    Two radii is the separation at which two sections of equal bounding radius stop
    being able to touch, so the band is every pair whose sections can overlap or meet.
    Applies only where the target frame declares a section; a bare point target is a
    point at any distance. Never wider than ``section_band``: averaging a filament
    source over a target section would mix a treatment the band edge has already left
    behind. See the module docstring for what the seam costs.
    """

    def __post_init__(self):
        """Load intergration constants."""
        super().__post_init__()
        self.data["r"] = np.linalg.norm([self["x"], self["y"]], axis=0)
        for attr in ["rs", "zs", "r", "z"]:
            setattr(self, attr, self.data[attr])
        source_radius = np.asarray(self.source["rms"], dtype=np.float64)
        if not np.all(np.isfinite(source_radius) & (source_radius > 0.0)):
            raise ValueError("circle sources require a finite positive radius")

    @staticmethod
    def point_greens(target_r, target_z, source_r, source_z):
        """Return canonical point-filament rows for already-partitioned pairs."""
        return traced_filament_greens(np, target_r, target_z, source_r, source_z)

    @cached_property
    def _coupling(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(psi, Br, Bz)`` per ampere, shape ``(target, source)``.

        Point, axis, single-integral and double-integral pairs are partitioned before
        any kernel is evaluated. A pair inside ``average_band`` is a double integral,
        one inside ``section_band`` alone is a single integral, and no pair is both.

        The source column's own polygon is handed the point targets and every
        target-section quadrature node in ONE kernel call. The closed form holds its
        corner parts live and amortises them across a call, so a per-pair or
        per-treatment call would pay that build repeatedly; the weighted means are
        formed from the same nodes :mod:`nova.biot.sectionaverage` supplies, and
        :mod:`tests.test_biotcircle` holds the diagonal this produces against
        :func:`nova.biot.sectionaverage.averaged_greens` so the two cannot drift.
        """
        coupling = tuple(np.empty(self.shape, dtype=np.float64) for _ in range(3))
        source = _sections(self.source)
        target = _sections(self.target)
        for column in range(self.shape[1]):
            target_r = self.r[:, column]
            target_z = self.z[:, column]
            vertices = None if source is None else source[column]
            if vertices is None:
                inside = np.zeros(target_r.shape, dtype=bool)
                averaged = np.zeros(target_r.shape, dtype=bool)
            else:
                centre = section_centroid(vertices)
                radius = float(np.max(np.hypot(*(vertices - centre).T)))
                distance = np.hypot(target_r - centre[0], target_z - centre[1])
                inside = _inside_closed_band(distance, radius, self.section_band)
                averaged = (
                    inside & _inside_closed_band(distance, radius, self.average_band)
                    if target is not None
                    else np.zeros(inside.shape, dtype=bool)
                )

            axis = target_r == 0.0
            ordinary = ~inside & ~axis
            if ordinary.any():
                evaluated = self.point_greens(
                    target_r[ordinary],
                    target_z[ordinary],
                    self.rs[ordinary, column],
                    self.zs[ordinary, column],
                )
                for value, got in zip(coupling, evaluated, strict=True):
                    value[ordinary, column] = got

            axis_point = axis & ~inside
            if axis_point.any():
                source_r = self.rs[axis_point, column]
                dz = target_z[axis_point] - self.zs[axis_point, column]
                coupling[0][axis_point, column] = 0.0
                coupling[1][axis_point, column] = 0.0
                coupling[2][axis_point, column] = (
                    self.mu_0 * source_r**2 / (2.0 * (source_r**2 + dz**2) ** 1.5)
                )

            if vertices is None or not inside.any():
                continue
            rows = np.flatnonzero(inside & ~averaged)
            section = [section_nodes(target[row]) for row in np.flatnonzero(averaged)]
            node = [target_r[rows], *(point[:, 0] for point, _ in section)]
            height = [target_z[rows], *(point[:, 1] for point, _ in section)]
            if sum(len(part) for part in node) == 0:
                continue
            evaluated = polygon_analytic_greens(
                np.concatenate(node), np.concatenate(height), vertices
            )
            for value, got in zip(coupling, evaluated, strict=True):
                value[rows, column] = got[: rows.size]
                start = rows.size
                for row, (_, weight) in zip(np.flatnonzero(averaged), section):
                    stop = start + weight.size
                    value[row, column] = weight @ got[start:stop] / weight.sum()
                    start = stop
            finite_axis = axis & inside
            coupling[0][finite_axis, column] = 0.0
            coupling[1][finite_axis, column] = 0.0
        return coupling

    @cached_property
    def Psi(self):
        """Return the total poloidal flux array [Wb/A]."""
        return self._coupling[0]

    @cached_property
    def Aphi(self):
        """Return the toroidal vector potential array [Wb/(m.A)]."""
        potential = np.zeros_like(self.Psi)
        np.divide(
            self.Psi,
            2 * np.pi * self.mu_0 * self.r,
            out=potential,
            where=self.r != 0.0,
        )
        return potential

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
