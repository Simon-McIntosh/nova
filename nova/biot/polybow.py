"""Biot-Savart calculation for polygon cross-section arc segments.

:class:`nova.biot.bow.Bow` thickens a circular-arc filament into a RECTANGULAR
cross-section through Urankar Part IV and a fixed-node zeta quadrature.  This
element is its peer for an arbitrary polygon, evaluated in closed form by
:func:`nova.biot.polygonarc.polygon_arc_greens` -- the finite-sweep counterpart
of the ring :class:`nova.biot.polysection.PolySection` already carries.

Why a peer of ``Bow`` rather than a subclass of it
--------------------------------------------------
The five rows the frame consumes -- ``A_r``, ``A_phi``, ``B_r``, ``B_phi`` and
``B_z`` in the target's own cylindrical basis, rotated into the source's local
cartesian frame and thence to global -- are :class:`nova.biot.arc.Arc`'s, not
``Bow``'s.  ``Bow`` adds exactly two things on top: a ``__post_init__`` that
stacks the four corners of a rectangle onto every geometry array, and the row
expressions that consume that stack.  A polygon section needs neither, so
subclassing ``Bow`` would mean inheriting a base whose whole body has to be
disabled.  Extending ``Arc`` instead inherits the frame plumbing and nothing
else, which is what the two elements genuinely share.

What the section is, and where it comes from
--------------------------------------------
The corners are the frame's ``poly`` column -- the single section of record, the
same column :class:`nova.biot.polysection.PolySection` reads for a ring -- taken
as its ``(r, z)`` corners with collinear runs collapsed.  Not rebuilt from the
section DESCRIPTOR, its name and its ``width`` and ``height``, for two reasons no
descriptor can answer.  A class that rebuilds its own section can drift from the
one that is plotted, meshed and integrated for volume, and there is then no way to
tell which of the two is the conductor.  And a descriptor reaches only the five
named shapes: a free-form polygon, which is the whole capability this element
exists for, is expressible in no ``(width, height)`` pair at all.

For that to be worth reading, the column has to be exact.  It is built from the
sweep's own float64 corner loops rather than from the vtk mesh those loops build
(:func:`nova.geometry.section.poloidal_footprint`), because VTK's points default
to single precision and a mesh round trip lands a corner authored at 2.97 on
2.96999979 -- 7.9e-09 relative on the rows, four orders above this reduction's own
3.5e-12, which would peg the element at the accuracy of the one it replaces.

Hollow sections
---------------
A ``skin`` or ``box`` section is an annulus, which a single corner list cannot
carry, and it is reached by superposition rather than refused: the frame inserts
the outer boundary at current density ``+j`` and the interior boundary as a core at
``-j``, linked, so the material between them carries ``j`` and the core cancels.
Both members are solid sections this element already evaluates, and the frame's own
link machinery sums the pair -- see :meth:`nova.frame.winding.Winding.insert`.

Quantities are per ampere of total conductor current at uniform current density
along the sweep, in nova's own frame convention -- the vector potential without
the ``mu0`` of Urankar eq 3a, which :class:`nova.biot.matrix.Matrix` restores on
the field alone.
"""

from dataclasses import dataclass
from functools import cached_property
import math
from typing import ClassVar

import numpy as np

from nova.biot.arc import Arc, arctan2
from nova.biot.matrix import Matrix
from nova.biot.polygonarc import polygon_arc_greens
from nova.geometry.section import collapse_collinear


def section_corners(poly) -> np.ndarray:
    """Return a frame polygon's ``(n, 2)`` r-z corners, ready for the reduction.

    The stored polygon carries its corners as ``(x, y, z)`` with the toroidal
    coordinate at zero, closes on a repeat of its first, and -- being a projection
    -- can carry a corner part way along a straight edge.  The reduction pays per
    corner, and a corner is what an arc's cost tracks, so a collinear run is
    collapsed rather than evaluated: a swept hexagon reaches the kernel with six
    corners where the projection's own union of stations has sixteen.  Collapsed
    again here rather than trusted to the column, so a polygon reaching the frame by
    another route cannot smuggle a redundant corner into the reduction's cost.
    """
    return collapse_collinear(np.asarray(poly.points, dtype=np.float64)[:, [0, 2]])


def signed_section_area(vertices: np.ndarray) -> float:
    """Return a translation-stable signed area for a planar corner list."""
    vertices = np.asarray(vertices, dtype=np.float64)
    local = vertices - vertices[0]
    rolled = np.roll(local, -1, axis=0)
    cross = local[:, 0] * rolled[:, 1] - rolled[:, 0] * local[:, 1]
    return 0.5 * math.fsum(cross.tolist())


def section_area(vertices: np.ndarray) -> float:
    """Return the area a closed corner list encloses.

    The area the REDUCTION normalised by, measured on the same corners it was
    handed rather than read from a column, so the two cannot disagree about which
    polygon the per-ampere convention refers to. Coordinates are recentered before
    the shoelace products so a small section keeps its area after a large rigid
    translation.
    """
    return abs(signed_section_area(vertices))


@dataclass
class PolyBow(Arc, Matrix):
    """
    Extend filament Arc base class.

    Compute interaction for polygon cross-section arc segments.

    """

    axisymmetric: ClassVar[bool] = False
    name: ClassVar[str] = "polybow"
    filament_centerline_limits: ClassVar[bool] = False

    nodes: ClassVar[int | None] = None
    """Residual quadrature nodes per panel; ``None`` uses the kernel's own count.

    The closed form leaves two smooth ``arsinh`` integrals per corner and nothing
    else, and their node count is fixed by the acceptance gate in
    :mod:`tests.test_biotpolygonarc`.  Raise it only for a section far more
    elongated than the gate's thin plate; there is nothing to gain by lowering it,
    because the residuals are what the evaluation spends its time on and a coarser
    rule moves the answer before it saves anything worth having.
    """

    @cached_property
    def _sweep(self) -> np.ndarray:
        """Return each source arc's end azimuth in its own local frame [rad].

        The local frame puts the arc's start at azimuth zero and its axis on
        ``z``, so one angle fixes the sweep.  Unwrapped onto ``[0, 2 pi)`` by the
        same operator :attr:`nova.biot.arc.Arc.alpha` uses, so an arc that passes
        the branch cut is one sweep rather than two.
        """
        return arctan2(self("source", "y2"), self("source", "x2"))[0]

    @cached_property
    def _section_vertices(self) -> list[np.ndarray]:
        """Return each source's cross-section corners as an ``(n, 2)`` r-z array.

        The reduction works in the arc's OWN frame -- origin on the arc's plane,
        vertical axis along the arc's -- where the ``poly`` column is in the global
        poloidal one.  For an arc whose axis is the vertical, which is the arc for
        which a poloidal projection IS the section, the two frames differ by one
        translation along that axis, and the source carries both images of its own
        start point to measure it with: :attr:`nova.biot.arc.Arc.zs` is that point's
        local elevation and ``z1`` its global one.  The radius needs no such
        correction -- it is measured from the same axis in both.
        """
        elevation = np.asarray(self.zs)[0] - np.asarray(self.source["z1"], dtype=float)
        vertices = []
        for column, poly in enumerate(np.asarray(self.source["poly"])):
            corners = section_corners(poly)
            corners[:, 1] += elevation[column]
            vertices.append(corners)
        return vertices

    @cached_property
    def _rows(self) -> np.ndarray:
        """Return the five rows per ampere, shape ``(5, target, source)``.

        One kernel call per source, because a section and a sweep belong to a
        source and the reduction is organised by the section's corners.  Divided
        by ``mu0`` on the way out: the reduction carries Urankar eq 3a's factor
        and nova's frame convention does not, restoring it on the field alone.

        Scaled from the kernel's own per-ampere convention -- uniform density over
        the section it integrated -- to the density the FRAME's ``area`` implies,
        which is the same convention ``Bow`` works in.  The ratio is one for a solid
        section, whose area column is its own polygon's; it is what makes a hollow
        section's linked pair sum, because both members then carry the annulus as
        their density denominator and the ``-1`` factor is all the core needs.
        """
        radius = np.asarray(self.r)
        elevation = np.asarray(self.z)
        azimuth = np.asarray(self._phi)
        area = np.asarray(self.source["area"], dtype=float)
        rows = np.empty((5,) + radius.shape)
        kernel = {} if self.nodes is None else {"nodes": self.nodes}
        for column, vertices in enumerate(self._section_vertices):
            rows[:, :, column] = (
                section_area(vertices)
                / area[column]
                * np.stack(
                    polygon_arc_greens(
                        radius[:, column],
                        elevation[:, column],
                        azimuth[:, column],
                        vertices,
                        0.0,
                        self._sweep[column],
                        **kernel,
                    )
                )
            )
        return rows / self.mu_0

    @cached_property
    def _Ar_hat(self):
        """Return local radial vector potential intergration coefficents."""
        return self._rows[0]

    @cached_property
    def _Aphi_hat(self):
        """Return local toroidal vector potential intergration coefficents."""
        return self._rows[1]

    @cached_property
    def _Br_hat(self):
        """Return local radial magnetic field intergration coefficents."""
        return self._rows[2]

    @cached_property
    def _Bphi_hat(self):
        """Return local toroidal magnetic field intergration coefficents."""
        return self._rows[3]

    @cached_property
    def _Bz_hat(self):
        """Return local vertical magnetic field intergration coefficents."""
        return self._rows[4]

    def _intergrate(self, data):
        """Return intergral quantity, which the reduction has already formed.

        The arc's two limits and the ``1/(4 pi)`` are folded inside the closed
        form -- an edge's two limits are differenced against each other there,
        before anything else is added to them, because each is of order the
        squared major radius where the answer is of order the section's own
        scale.  So nothing is left to do here, where ``Arc`` and ``Bow`` still
        have their limit pair to take.
        """
        return data
