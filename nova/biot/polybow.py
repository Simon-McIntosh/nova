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
The corners are built from the frame's own section descriptor -- its name, its
``width`` and its ``height`` about the arc's own ``(r, z)`` -- through
:class:`nova.geometry.polygen.PolyGen`, in the same convention ``Bow`` builds
its rectangle from.  Not from the frame's ``poly`` column, which for a swept
element is the projection of the 3-D volume onto the poloidal plane: its axes
come back transposed against ``width`` and ``height``, and it is stored at
single precision.  ``poly`` IS the section for an axisymmetric ring, which is
why ``PolySection`` reads it and this element does not.

Quantities are per ampere of total conductor current at uniform current density
along the sweep, in nova's own frame convention -- the vector potential without
the ``mu0`` of Urankar eq 3a, which :class:`nova.biot.matrix.Matrix` restores on
the field alone.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

import numpy as np

from nova.biot.arc import Arc, arctan2
from nova.biot.matrix import Matrix
from nova.biot.polygonarc import polygon_arc_greens
from nova.geometry.polygen import PolyGen


def _distinct(points: np.ndarray) -> np.ndarray:
    """Return a closed ring's corners with every repeat of its neighbour dropped.

    A generated section comes back as a shapely ring, so its first corner is
    repeated at the end -- and a generator that sweeps an angle closes on a corner
    that is its own first to within round-off rather than exactly, which leaves the
    ring one corner longer still.  Both are edges of zero length, and the tolerance
    is taken against the SECTION's own extent so a thin plate keeps corners a
    coordinate-absolute one would merge.
    """
    scale = max(float(np.max(np.ptp(points, axis=0))), np.finfo(float).tiny)
    gap = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
    return points[gap > 1e-9 * scale]


@dataclass
class PolyBow(Arc, Matrix):
    """
    Extend filament Arc base class.

    Compute interaction for polygon cross-section arc segments.

    """

    axisymmetric: ClassVar[bool] = False
    name: ClassVar[str] = "polybow"

    nodes: ClassVar[int | None] = None
    """Residual quadrature nodes per panel; ``None`` uses the kernel's own count.

    The closed form leaves two smooth ``arsinh`` integrals per corner and nothing
    else, and their node count is fixed by the acceptance gate in
    :mod:`tests.test_biotpolygonarc`.  Raise it only for a section far more
    elongated than the gate's thin plate; there is nothing to gain by lowering it,
    because the residuals are what the evaluation spends its time on and a coarser
    rule moves the answer before it saves anything worth having.
    """

    solid_sections: ClassVar[frozenset[str]] = frozenset(
        {"disc", "ellipse", "hexagon", "rectangle", "square"}
    )
    """Section names this element can build from ``(width, height)``.

    Every :class:`~nova.geometry.polygen.PolyGen` generator whose third and fourth
    arguments are the section's own two dimensions, which is what the frame stores.
    ``skin`` and ``box`` are excluded because theirs is a HOLLOWNESS factor rather
    than a height, and because the section they describe has an interior boundary
    that a single corner list cannot carry; ``polygon``, ``shell`` and ``outline``
    because a swept element keeps no corner list of its own to read.  A ``disc``
    or an ``ellipse`` is admitted and costs what its corner count implies -- the
    generator resolves a quadrant in sixteen segments, so sixty-four corners
    against a hexagon's six.
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
        """Return each source's cross-section corners as an ``(n, 2)`` r-z array."""
        section = [PolyGen(str(name)).shape for name in self.source["section"]]
        if unsupported := set(section) - self.solid_sections:
            raise NotImplementedError(
                f"cross-section {sorted(unsupported)} cannot be built from its "
                f"width and height alone; {self.name} takes "
                f"{sorted(self.solid_sections)}"
            )
        width = np.asarray(self.source["width"], dtype=float)
        height = np.asarray(self.source["height"], dtype=float)
        radius = np.asarray(self.rs)[0]
        elevation = np.asarray(self.zs)[0]
        vertices = []
        for column, name in enumerate(section):
            shape = PolyGen(name)(
                radius[column], elevation[column], width[column], height[column]
            )
            vertices.append(_distinct(np.asarray(shape.exterior.coords, float)))
        return vertices

    @cached_property
    def _rows(self) -> np.ndarray:
        """Return the five rows per ampere, shape ``(5, target, source)``.

        One kernel call per source, because a section and a sweep belong to a
        source and the reduction is organised by the section's corners.  Divided
        by ``mu0`` on the way out: the reduction carries Urankar eq 3a's factor
        and nova's frame convention does not, restoring it on the field alone.
        """
        radius = np.asarray(self.r)
        elevation = np.asarray(self.z)
        azimuth = np.asarray(self._phi)
        rows = np.empty((5,) + radius.shape)
        kernel = {} if self.nodes is None else {"nodes": self.nodes}
        for column, vertices in enumerate(self._section_vertices):
            rows[:, :, column] = polygon_arc_greens(
                radius[:, column],
                elevation[:, column],
                azimuth[:, column],
                vertices,
                0.0,
                self._sweep[column],
                **kernel,
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
