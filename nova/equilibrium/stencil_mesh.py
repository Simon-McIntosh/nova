r"""Differential receipts on an unstructured cell mesh from neighbour rings.

:class:`~nova.equilibrium.conservation.FluxLattice` reads its derivatives by
central differences, which needs a tensor-product raster of one cell size. The
plasma mesh the package actually ships is not one: cells are hexagons on a
half-offset tiling, trimmed where the first wall cuts them, so rows are offset
by half a pitch, the domain stops at a curved boundary and a clipped cell
carries both a smaller area and a displaced centroid.

This class carries the same differential contract on that mesh. The geometry it
needs is what the grid solve already produces — cell centroids, cell areas, and
the centre-first neighbour rings the tessellation recovers from a Delaunay
triangulation of those centroids, packed exactly as
:func:`nova.geometry.hexstencil.hex_stencil` packs a structured one, so the
null search and the derivative operator read the same rings.

Derivatives come from a least-squares quadratic fitted on each ring. Six
neighbours and the centre give seven samples for the six coefficients of

.. math::
    f \simeq c_0 + c_1 u + c_2 v + c_3 u^2 + c_4 u v + c_5 v^2,

so the fit is overdetermined by one and the derivative at the centre is a fixed
linear functional of the ring values. The coordinates are centred on the ring
centre and scaled by the ring half-width before the fit, the normalisation
:class:`~nova.biot.null.Null2D` already applies, which is what keeps the design
matrix conditioned at a metre-scale major radius: on a regular hexagonal ring
its condition number is 5.5, and a quarter-pitch centroid displacement — far
more than first-wall clipping produces — leaves it under twenty.

A hexagonal ring determines the full quadratic rather than the Laplacian alone.
Restricted to the six ring points the quadratic basis spans the angular modes
:math:`1, \cos\theta, \sin\theta, \cos 2\theta, \sin 2\theta` and the centre
supplies the sixth degree of freedom, so :math:`\partial^2/\partial R^2` and
:math:`\partial^2/\partial Z^2` are separately resolved and not just their sum.
The first mode the ring cannot see is :math:`\cos 3\theta`, which is where the
truncation error lives; both the gradient and the elliptic operator converge at
second order in the pitch.

Cells without a ring — the hull of the tessellation, and any cell a caller
withheld — carry no derivative. They are reported as zero and excluded by
:meth:`StencilMesh.interior`, so a receipt never reads a value the mesh could
not form, exactly as the lattice border is trimmed before a residual is
reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.conservation import STENCIL_MARGIN

__all__ = ["RING_CONDITION_LIMIT", "StencilMesh", "ring_condition"]

#: Largest normalised design-matrix condition number a ring may carry. A
#: regular hexagonal ring sits at 5.5 and irregular ones climb slowly, so the
#: limit is far above any tiling and still catches the failure it exists for: a
#: cluster whose points are collinear, coincident or otherwise unable to
#: determine a quadratic, which pinv would answer with a plausible-looking
#: least-norm fit rather than an error.
RING_CONDITION_LIMIT = 1.0e3

#: Coefficients of the fitted quadratic, in design-matrix column order.
_VALUE, _RADIAL, _VERTICAL, _RADIAL_CURVATURE, _CROSS, _VERTICAL_CURVATURE = range(6)


def _quadratic_design(local: np.ndarray) -> np.ndarray:
    """Return the quadratic design matrix of normalised ring coordinates."""
    radial, vertical = local[..., 0], local[..., 1]
    return np.stack(
        [
            np.ones_like(radial),
            radial,
            vertical,
            radial**2,
            radial * vertical,
            vertical**2,
        ],
        axis=-1,
    )


def _normalised_ring(coordinate: np.ndarray, stencil: np.ndarray):
    """Return every ring centred on its own centre and scaled to unit width."""
    cluster = coordinate[stencil]
    offset = cluster - cluster[:, :1]
    scale = np.max(np.abs(offset), axis=1)
    if np.any(scale <= 0.0):
        raise ValueError("every ring must span both coordinate axes")
    return offset / scale[:, np.newaxis, :], scale, cluster


def ring_condition(coordinate, stencil) -> np.ndarray:
    """Return the conditioning of the quadratic fit on every ring.

    A caller that assembles rings from a tessellation reads this to decide
    which ones to hand over. :class:`StencilMesh` REFUSES a ring it cannot fit
    rather than answering with a least-norm one, so the selection has to be
    made deliberately, and this is what it is made on: the shape of the
    neighbourhood, measured after the centring and scaling the fit applies,
    and therefore independent of the major radius the ring sits at.
    """
    local, _scale, _cluster = _normalised_ring(
        np.ascontiguousarray(coordinate, dtype=np.float64), np.asarray(stencil)
    )
    return np.linalg.cond(_quadratic_design(local))


@dataclass(frozen=True)
class StencilMesh:
    """Cell mesh whose derivatives are fitted on centre-first neighbour rings.

    ``coordinate`` holds the ``(radius, height)`` centroid of every cell,
    ``area`` its poloidal cross-section, and ``stencil`` the neighbour rings:
    one row per cell that carries a derivative, its own index in column zero
    and its neighbours after it. A cell may appear in any number of rings but
    may centre only one.
    """

    coordinate: np.ndarray
    stencil: np.ndarray
    area: np.ndarray
    radial_weight: np.ndarray = field(init=False, repr=False)
    vertical_weight: np.ndarray = field(init=False, repr=False)
    elliptic_weight: np.ndarray = field(init=False, repr=False)
    ring_condition: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        """Validate the mesh and fit the derivative weights of every ring."""
        coordinate = np.ascontiguousarray(self.coordinate, dtype=np.float64)
        if coordinate.ndim != 2 or coordinate.shape[1] != 2:
            raise ValueError("mesh coordinates must have shape (cells, 2)")
        if np.any(coordinate[:, 0] <= 0.0):
            raise ValueError("cell radius must be strictly positive")
        area = np.ascontiguousarray(self.area, dtype=np.float64)
        if area.shape != (coordinate.shape[0],):
            raise ValueError("one area is needed per cell")
        stencil = np.ascontiguousarray(self.stencil, dtype=np.intp)
        if stencil.ndim != 2 or stencil.shape[1] < 6:
            raise ValueError(
                "a quadratic fit needs rings of at least six cells, centre first"
            )
        if stencil.size and (stencil.min() < 0 or stencil.max() >= len(coordinate)):
            raise ValueError("a ring indexes a cell the mesh does not carry")
        centre = stencil[:, 0]
        if len(np.unique(centre)) != len(centre):
            raise ValueError("a cell may centre at most one ring")
        object.__setattr__(self, "coordinate", coordinate)
        object.__setattr__(self, "area", area)
        object.__setattr__(self, "stencil", stencil)
        self._fit_rings()

    def _fit_rings(self):
        """Solve the normalised quadratic fit of every ring for its weights."""
        local, scale, cluster = _normalised_ring(self.coordinate, self.stencil)
        design = _quadratic_design(local)
        condition = np.linalg.cond(design)
        if np.any(condition > RING_CONDITION_LIMIT):
            worst = int(np.argmax(condition))
            raise ValueError(
                f"ring {worst} centred on cell {self.stencil[worst, 0]} cannot "
                f"determine a quadratic (condition {condition[worst]:.3e})"
            )
        inverse = np.linalg.pinv(design)
        radial = inverse[:, _RADIAL] / scale[:, :1]
        vertical = inverse[:, _VERTICAL] / scale[:, 1:]
        curvature = (
            2.0 * inverse[:, _RADIAL_CURVATURE] / scale[:, :1] ** 2
            + 2.0 * inverse[:, _VERTICAL_CURVATURE] / scale[:, 1:] ** 2
        )
        object.__setattr__(self, "radial_weight", radial)
        object.__setattr__(self, "vertical_weight", vertical)
        object.__setattr__(
            self, "elliptic_weight", curvature - radial / cluster[:, :1, 0]
        )
        object.__setattr__(self, "ring_condition", condition)

    @property
    def node_count(self) -> int:
        """Return the cell count."""
        return len(self.coordinate)

    @property
    def node_radius(self) -> np.ndarray:
        """Return the major radius [m] of every cell centroid."""
        return self.coordinate[:, 0]

    @property
    def cell_area(self) -> np.ndarray:
        """Return the poloidal cross-section [m^2] of every cell."""
        return self.area

    @property
    def centre(self) -> np.ndarray:
        """Return the cells that carry a derivative, in ring order."""
        return self.stencil[:, 0]

    def _scatter(self, ring_value) -> jax.Array:
        """Return one per-ring value placed on its centre cell, zero elsewhere."""
        return (
            jnp.zeros(self.node_count, dtype=ring_value.dtype)
            .at[self.centre]
            .set(ring_value)
        )

    def _apply(self, weight: np.ndarray, values) -> jax.Array:
        """Return one fitted derivative of the ring values."""
        return jnp.sum(jnp.asarray(weight, dtype=values.dtype) * values, axis=1)

    def gradient(self, field) -> tuple[jax.Array, jax.Array]:
        """Return the radial and vertical derivative of one cell field."""
        values = jnp.asarray(field)[self.stencil]
        return (
            self._scatter(self._apply(self.radial_weight, values)),
            self._scatter(self._apply(self.vertical_weight, values)),
        )

    def delta_star(self, flux) -> jax.Array:
        """Return the elliptic operator value [Wb/m^2] of one flux map.

        The radial, vertical and curvature weights of a ring act on the same
        seven values, so the whole operator is carried by one weight vector
        rather than by three fits of the same quadratic.
        """
        values = jnp.asarray(flux)[self.stencil]
        return self._scatter(self._apply(self.elliptic_weight, values))

    def erode(self, mask, margin: int) -> jax.Array:
        """Return a cell mask shrunk by ``margin`` ring steps."""
        eroded = jnp.asarray(mask, dtype=bool)
        for _ in range(margin):
            eroded = self._scatter(jnp.all(eroded[self.stencil], axis=1))
        return eroded

    def interior(self, margin: int = STENCIL_MARGIN) -> jax.Array:
        """Return the mask of cells whose ring neighbourhood is complete."""
        return self.erode(jnp.ones(self.node_count, dtype=bool), margin)
