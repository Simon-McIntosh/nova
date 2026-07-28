"""Closed-form potential and field of a straight polygon-section conductor.

:class:`nova.biot.beam.Beam` thickens a straight filament into a RECTANGULAR
cross-section, in closed form and in the segment's own cartesian frame.  This
element is its peer for an arbitrary simple polygon, and it is the straight-path
counterpart of the swept :class:`nova.biot.polybow.PolyBow`.

Why a peer of ``Beam`` rather than a subclass of it
--------------------------------------------------
``Beam``'s body IS its rectangle: four of its six theta coefficients are ratios
whose denominators are the corner offsets of an axis-aligned box, and the corner
sum is a ``(2, 2, 2)`` einsum over the two ends of each of the three coordinate
ranges the box spans.  A polygon has no such product structure -- its corners are
a list, not a tensor -- so subclassing would mean inheriting a base whose whole
body has to be disabled.  What the two elements genuinely share is the frame
plumbing, which is :class:`nova.biot.matrix.Matrix`'s, so the two are siblings on
it.  The same argument the arc pair settles in :mod:`nova.biot.polybow`.

The reduction
-------------
Current flows along the local ``z`` axis at uniform density over the section
``S``, so only ``A_z`` is nonzero and, with ``u = x' - x`` and ``v = y' - y``
in-plane and ``w = z' - z`` axial::

    4 pi area A_z = integral_S da' [ arsinh(w2 / rho) - arsinh(w1 / rho) ]

with ``rho = hypot(u, v)`` -- the axial integral is elementary, and what is left
is the section average of the filament kernel :class:`nova.biot.line.Line`
evaluates.  The area integral goes to the boundary by the divergence theorem,
once for the potential and once for the field, and both times the vector field it
needs is radial in the section plane:

* for the FIELD, ``arsinh(w / rho)`` is itself the potential of the two
  transverse derivatives -- ``d/du arsinh(w/rho) = -w u / (rho^2 r)`` is the
  filament's own ``B``, so with ``V = (arsinh, 0)`` and ``(0, arsinh)`` the field
  rows are contour integrals of ``arsinh`` weighted by the outward normal;
* for the POTENTIAL, a radial field ``h(rho) (u, v)/rho`` has divergence
  ``(1/rho) d(rho h)/drho``, and ``integral rho arsinh(w/rho) drho`` is
  elementary by parts, so ``h = (rho/2) arsinh(w/rho) + w r / (2 rho)``.

That second ``h`` carries a ``1/rho`` singularity at the target's own transverse
position, whose flux is a delta -- an interior-point indicator, discontinuous
where the answer is smooth.  The constant of integration is free and kills it:
taking ``w (r - |w|) / (2 rho) = w rho / (2 (r + |w|))`` instead leaves ``h``
vanishing at the origin, so ONE expression covers a target outside the section,
on its boundary and inside it, with no inside test anywhere.  This is what makes
a target inside the conductor -- the reason a finite-section element exists at
all -- an ordinary evaluation rather than a case.

Each edge then contributes single integrals along its own length, in the
perpendicular-foot parameter ``t`` with ``rho^2 = d^2 + t^2`` for the edge's
signed distance ``d``, and both are elementary::

    integral arsinh(w / rho) dt = t arsinh(w / rho) + w arsinh(t / sigma)
                                  - sign(w) |d| arctan(t |w| / (|d| r))
    d integral w / (r + |w|) dt  = d w arsinh(t / sigma)
                                  - w |w| arctan(t d rho^2
                                                 / ((r + |w|)(d^2 r + t^2 |w|)))

with ``sigma = hypot(d, w)`` and ``r = hypot(rho, w)``.  The second line is the
sum of two arctangents each carrying a ``1/|d|`` that the edge's own ``d`` weight
cannot cancel; combined by the difference identity -- valid unconditionally here,
because the two arguments share a sign so ``1 + AB`` is never below one -- the
pole becomes a factor of ``d`` in the numerator and the row is regular on an edge
whose extended line passes through the target.  ``arsinh`` and ``arctan`` are the
whole transcendental family: a straight path has no curvature, so there is no
toroidal integration and nothing elliptic appears.

There is no modulus approaching unity anywhere and no confluence of parameters to
manage, so the evaluation is round-off throughout, including on every degeneracy
a field grid aligned with the section reaches by construction -- an edge parallel
to either axis, a target projecting onto an edge or exactly onto a corner, a
target on the axis, and a target in the plane of an end face.  ``Beam``'s own
``theta`` denominators vanish when a target is aligned axially with a section
corner, where these rows are finite.

What the section is, and where it comes from
--------------------------------------------
The corners are the frame's ``poly`` column, the single section of record, for
the reasons :mod:`nova.biot.polybow` sets out in full: a class that rebuilds its
own section from the ``width`` and ``height`` descriptor can drift from the one
that is plotted, meshed and integrated for volume, and a descriptor reaches only
the named shapes where a free-form polygon -- the whole capability this element
exists for -- is expressible in no ``(width, height)`` pair at all.

That column is a POLOIDAL PROJECTION of the section, and for a straight segment
the projection is not the identity the way it is for an arc swept about the
vertical.  Current crosses the plane normal to the path, which the local frame
spans with its ``x`` and ``y`` axes, and the projection maps that plane onto
``(r, z)`` by a linear map the frame's own axes measure::

    (dr, dz) = M (dx, dy),  M = [[ex . rhat, ey . rhat], [ex . zhat, ey . zhat]]

evaluated at the segment's end point, which is where the swept footprint sits.
``M`` is a signed identity exactly when the section plane contains both the
radial and the vertical direction -- the arc case, where a translation is all
``PolyBow`` needs -- and for a chord of a circular path it is the cosine of half
the chord's own turn, which inverting recovers.  A section read straight off the
column instead would come back narrow by that cosine, which is a percent for a
coarsely-resolved path.

Quantities are per ampere of total conductor current at uniform current density,
in nova's own frame convention -- the vector potential without ``mu0``, which
:class:`nova.biot.matrix.Matrix` restores on the field alone -- and normalised by
the frame's ``area`` column exactly as ``Beam`` is.  That is what makes a hollow
section's linked pair sum: a ``skin`` or a ``box`` is an annulus, which one corner
list cannot carry, and the frame inserts its outer boundary at ``+j`` and its
interior boundary as a core at ``-j``, both of them solid sections this element
already evaluates.
"""

from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import numpy as np

from nova.biot.matrix import Matrix
from nova.biot.polybow import section_area, section_corners

_ROWS = ("A_z", "B_x", "B_y")
"""The rows the reduction forms, in the segment's own local cartesian frame.

``A_x``, ``A_y`` and ``B_z`` are identically zero for a current along the local
``z`` axis, and the frame's own rotation is what carries these three to global.
"""


def _arcsinh(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Return ``arsinh(numerator / denominator)``, zero where it is unreachable.

    Both places the ratio is formed, the denominator vanishes only where the
    factor multiplying the ``arsinh`` vanishes too -- the perpendicular-foot
    parameter for a target projecting onto a corner, the axial offset for a target
    in the plane of an end face -- so the product's limit is zero and returning
    zero for the factor is exact rather than a floor.  Written as a masked ratio
    rather than as a limit of the product so no infinity is formed at all.
    """
    live = denominator > 0.0
    return np.where(live, np.arcsinh(numerator / np.where(live, denominator, 1.0)), 0.0)


def _counterclockwise(vertices: np.ndarray) -> np.ndarray:
    """Return a closed corner list wound counterclockwise.

    The divergence theorem needs the OUTWARD normal, and the normal is taken as
    the edge tangent turned a quarter turn -- which points outward for one winding
    and inward for the other, flipping the sign of every row.  Fixing the winding
    here is what lets the reduction take the normal from the tangent unconditionally.
    """
    corners = np.asarray(vertices, dtype=np.float64)
    rolled = np.roll(corners, -1, axis=0)
    signed = float(np.sum(corners[:, 0] * rolled[:, 1] - rolled[:, 0] * corners[:, 1]))
    return corners if signed > 0 else corners[::-1].copy()


def polygon_beam_greens(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    vertices: np.ndarray,
    z1: np.ndarray,
    z2: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Return ``(A_z, B_x, B_y)`` per ampere, in closed form.

    A prism of arbitrary simple-polygon cross-section, given as the ``(n, 2)``
    array of its corners in the ``x-y`` plane, carrying current along ``z`` from
    ``z1`` to ``z2`` at uniform density over the section.  Every argument is in
    the prism's OWN cartesian frame and every row is per ampere of total conductor
    current, in nova's convention -- neither the potential nor the field carries
    ``mu0``.

    The corners locate the section absolutely rather than relative to the axis, so
    a target inside the conductor is a target inside the corner loop, and the
    axial limits are the only thing measured from anywhere else.  ``z1`` and ``z2``
    broadcast against the targets, which is what lets one call carry a whole
    target cloud against one source.

    Edge ``i`` runs from corner ``i`` to corner ``i + 1`` and the loop closes on
    its own first corner; a corner part way along a straight edge costs an
    evaluation and contributes nothing, so collapse the collinear runs before
    calling (:func:`nova.geometry.section.collapse_collinear`).
    """
    shape = np.broadcast_shapes(np.shape(x), np.shape(y), np.shape(z))
    target_x, target_y, target_z = (
        np.broadcast_to(np.asarray(value, dtype=np.float64), shape).ravel()
        for value in (x, y, z)
    )
    corners = _counterclockwise(vertices)
    area = section_area(corners)
    rolled = np.roll(corners, -1, axis=0)
    edge = rolled - corners
    tangent = edge / np.linalg.norm(edge, axis=1)[:, np.newaxis]
    normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=-1)

    start_x = corners[:, 0][:, np.newaxis] - target_x[np.newaxis]
    start_y = corners[:, 1][:, np.newaxis] - target_y[np.newaxis]
    end_x = rolled[:, 0][:, np.newaxis] - target_x[np.newaxis]
    end_y = rolled[:, 1][:, np.newaxis] - target_y[np.newaxis]
    # the edge's signed distance from the target, and its two endpoints in the
    # perpendicular-foot parameter, so that rho**2 is d**2 + t**2 by construction
    distance = (
        start_x * normal[:, 0][:, np.newaxis] + start_y * normal[:, 1][:, np.newaxis]
    )[np.newaxis, np.newaxis]
    foot = np.stack(
        [
            start_x * tangent[:, 0][:, np.newaxis]
            + start_y * tangent[:, 1][:, np.newaxis],
            end_x * tangent[:, 0][:, np.newaxis] + end_y * tangent[:, 1][:, np.newaxis],
        ]
    )[np.newaxis]
    axial = (
        np.stack(
            [
                np.broadcast_to(np.asarray(limit, dtype=np.float64), shape).ravel()
                for limit in (z1, z2)
            ]
        )
        - target_z[np.newaxis]
    )[:, np.newaxis, np.newaxis]

    axial_magnitude = np.abs(axial)
    distance_magnitude = np.abs(distance)
    rho = np.sqrt(distance**2 + foot**2)
    radius = np.sqrt(distance**2 + foot**2 + axial**2)
    sigma = np.sqrt(distance**2 + axial**2)
    contour = (
        foot * _arcsinh(axial, rho)
        + axial * _arcsinh(foot, sigma)
        - np.sign(axial)
        * distance_magnitude
        * np.arctan2(foot * axial_magnitude, distance_magnitude * radius)
    )
    # the regularised radial term, its two arctangents already combined so the
    # pole an edge's extended line puts at d = 0 is a factor of d instead
    weighted = distance * axial * _arcsinh(foot, sigma) - axial * axial_magnitude * (
        np.arctan2(
            foot * distance * rho**2,
            (radius + axial_magnitude)
            * (distance**2 * radius + foot**2 * axial_magnitude),
        )
    )

    sign = np.array([[1.0, -1.0], [-1.0, 1.0]])[:, :, np.newaxis, np.newaxis]
    outward = normal[np.newaxis, np.newaxis, :, :, np.newaxis]
    axes = (0, 1, 2)
    scale = 1.0 / (4 * np.pi * area)
    return tuple(
        (scale * row).reshape(shape)
        for row in (
            0.5 * np.sum(sign * (distance * contour + weighted), axis=axes),
            -np.sum(sign * outward[..., 1, :] * contour, axis=axes),
            np.sum(sign * outward[..., 0, :] * contour, axis=axes),
        )
    )


@dataclass
class PolyBeam(Matrix):
    """
    Extend Biot base class.

    Compute interaction for polygon cross-section straight segments.

    """

    axisymmetric: ClassVar[bool] = False
    name: ClassVar[str] = "polybeam"

    attrs: dict[str, str] = field(default_factory=lambda: {"dl": "dl"})

    @cached_property
    def _end_point(self) -> np.ndarray:
        """Return each source's global end point, ``(n, 3)``.

        The reference the section's projection is measured from and mapped at.
        """
        return np.column_stack(
            [
                np.asarray(self.source[coord], dtype=float)
                for coord in ("x2", "y2", "z2")
            ]
        )

    @cached_property
    def _projection(self) -> np.ndarray:
        """Return each source's section-plane to poloidal-plane map, ``(n, 2, 2)``.

        Built at the segment's END point, which is where a swept footprint sits:
        both end stations of a chord whose endpoints share a radius project onto
        the same ring, and that ring is the column.  The map's columns are the
        local ``x`` and ``y`` axes resolved on the radial and vertical directions
        there, so it is the signed identity whenever the section plane contains
        both -- the arc case -- and carries the chord's own tilt otherwise.
        """
        axes = np.asarray(self.source.space.coordinate_axes, dtype=float)
        point = self._end_point
        radial = np.zeros_like(point)
        radial[:, :2] = point[:, :2]
        norm = np.linalg.norm(radial, axis=1)
        if np.any(norm == 0.0):
            raise ValueError(
                "a straight polygon-section segment ending on the vertical axis "
                "has no poloidal projection of its own section plane"
            )
        radial /= norm[:, np.newaxis]
        vertical = np.zeros_like(point)
        vertical[:, 2] = 1.0
        projection = np.stack(
            [
                np.einsum("ni,nij->nj", direction, axes[:, :, :2])
                for direction in (radial, vertical)
            ],
            axis=1,
        )
        determinant = np.linalg.det(projection)
        if np.any(np.abs(determinant) < 1e-09):
            raise ValueError(
                "a straight polygon-section segment whose section plane projects "
                "onto a line carries no section in the frame's poloidal column"
            )
        return projection

    @cached_property
    def _section_vertices(self) -> list[np.ndarray]:
        """Return each source's section corners in its own local ``x-y`` plane.

        The column's corners are offsets from the end point's own projection,
        mapped through :attr:`_projection` into the plane current crosses and
        placed on the axis -- whose local ``x`` and ``y`` a straight segment holds
        constant along its whole length, the axis being the local ``z``.
        """
        point = self._end_point
        reference = np.stack(
            [np.linalg.norm(point[:, :2], axis=1), point[:, 2]], axis=-1
        )
        axis = np.stack(
            [np.asarray(self("source", coord))[0] for coord in ("x2", "y2")], axis=-1
        )
        vertices = []
        for column, poly in enumerate(np.asarray(self.source["poly"])):
            offset = section_corners(poly) - reference[column]
            vertices.append(
                axis[column] + np.linalg.solve(self._projection[column], offset.T).T
            )
        return vertices

    @cached_property
    def _rows(self) -> np.ndarray:
        """Return the three local rows per ampere, shape ``(3, target, source)``.

        One kernel call per source, because a section belongs to a source and the
        reduction is organised by its corners.  Scaled from the kernel's own
        per-ampere convention -- uniform density over the polygon it integrated --
        to the density the FRAME's ``area`` column implies, which is the convention
        ``Beam`` works in.  The ratio is one for a solid section, whose column is
        its own polygon's area; it is what makes a hollow section's linked pair sum,
        because both members then carry the annulus as their density denominator and
        the ``-1`` factor is all the core needs.
        """
        local = [np.asarray(self("target", coord)) for coord in ("x", "y", "z")]
        limits = [np.asarray(self("source", coord)) for coord in ("z1", "z2")]
        area = np.asarray(self.source["area"], dtype=float)
        rows = np.empty((len(_ROWS),) + self.shape)
        for column, vertices in enumerate(self._section_vertices):
            rows[:, :, column] = (
                section_area(vertices)
                / area[column]
                * np.stack(
                    polygon_beam_greens(
                        local[0][:, column],
                        local[1][:, column],
                        local[2][:, column],
                        vertices,
                        limits[0][:, column],
                        limits[1][:, column],
                    )
                )
            )
        return rows

    @property
    def _Ax_hat(self):
        """Return local x-coord vector potential intergration coefficents."""
        return np.zeros(self.shape)

    @property
    def _Ay_hat(self):
        """Return local y-coord vector potential intergration coefficents."""
        return np.zeros(self.shape)

    @cached_property
    def _Az_hat(self):
        """Return local z-coord vector potential intergration coefficents."""
        return self._rows[0]

    @cached_property
    def _Bx_hat(self):
        """Return local x-coord magnetic field intergration coefficents."""
        return self._rows[1]

    @cached_property
    def _By_hat(self):
        """Return local y-coord magnetic field intergration coefficents."""
        return self._rows[2]

    @property
    def _Bz_hat(self):
        """Return local z-coord magnetic field intergration coefficents."""
        return np.zeros(self.shape)

    def _intergrate(self, data):
        """Return intergral quantity, which the reduction has already formed.

        The axial limits, the corner sum and the ``1/(4 pi area)`` are all folded
        inside the closed form, where ``Beam`` still has its ``(2, 2, 2)`` corner
        stack to contract.
        """
        return data
