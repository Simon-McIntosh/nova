"""Biot-Savart coupling for toroidal conductors of polygonal cross-section.

A point-filament ring is log-singular at its own location, so any target that
approaches a conductor — a neighbouring plasma cell, or the cell itself on the
diagonal of the plasma-plasma matrix — inherits a spurious near-field spike.
Spreading the current uniformly over the true cross-section removes the
singularity: the flux and field stay finite and smooth through the conductor.
:func:`nova.biot.polygon.polygon_greens` does that for an arbitrary polygon,
which is what a hexagonal (or wall-clipped) plasma cell needs.

Near and far
------------
The exact polygon kernel integrates over the section boundary at every target,
so it costs roughly four orders of magnitude more per target-source pair than
the point-filament form. Beyond a few section sizes it also buys nothing: the
finite-area correction there is the constant second-moment term, a few tenths of
a percent in flux and far below any measurement it is compared against. This
element therefore evaluates the exact kernel only inside a standoff band of
``standoff`` section radii and the point-filament form outside it — the same
near/far contract :func:`nova.biot.greens.hybrid_greens` already applies to
rectangular sections, generalised to a polygon. On a plasma grid the band holds
a few percent of the target-source pairs, which is what makes the exact
treatment affordable where it is physically real.

Three bands
-----------
The two-way near/far split above is a study knob, not the shipped default,
because a bare point filament does not converge to a finite section for a full
ring at any standoff. The measured alternative is the three-band scheme in
:mod:`nova.biot.bandedcoupling` — converged rule, reduced rule, moment-corrected
filament, binned by distance to the section contour — which holds every component
to one part in a million of its peak. It is available here through ``banded`` and
is off by default: exact everywhere remains the shipped lane and the reference
the banded one is measured against.

Quantities are per ampere of total conductor current, in raw SI: total poloidal
flux :math:`\\Phi = 2 \\pi R A_\\phi` [Wb] and field components [T].
"""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

import numpy as np

from nova.biot.bandedcoupling import banded_greens
from nova.biot.greens import greens_bz_br, greens_psi, section_centroid
from nova.biot.matrix import Matrix
from nova.biot.polygon import polygon_greens


@dataclass
class PolySection(Matrix):
    """Couple complete toroidal conductors of arbitrary polygonal section.

    The section vertices come from each source element's own polygon, so a
    regular hexagonal plasma cell and a cell clipped by the first wall are
    handled by the same code path with no shape assumption.
    """

    axisymmetric: ClassVar[bool] = True
    name: ClassVar[str] = "polysection"

    standoff: ClassVar[float | None] = None
    """Standoff band, in section radii, within which the exact kernel is used.

    The default ``None`` (or ``inf``) means *exact everywhere*: every
    target-source pair goes through the polygon kernel, with no point-filament
    far field and no seam at all. That is the physically unimpeachable setting
    and the only one shipped as a default — a finite band needs a principled,
    error-bounded cutoff, and none of the geometry-derived candidates measured
    so far qualifies (a few section radii keeps the seam small, ~5e-4 in flux,
    but where the finite-section correction stops mattering has to come from
    the section's second-moment error bound, not from a budget). The
    operator-assembly review owns that choice.

    A finite value is for scoped studies via :meth:`configured`. Measured
    guidance for such sweeps: the exact kernel costs about four orders of
    magnitude more per pair than the point form; two radii is the useful floor
    (it still covers a cell and its first ring of neighbours — in a hexagonal
    tiling of circumradius ``a`` the nearest centres sit at ``sqrt(3) a`` —
    and on a 234-cell mesh delivers the identical self-term correction for a
    quarter of the cost of three radii). Below two radii, near-neighbour pairs
    fall back to the bare point kernel just where
    :class:`nova.biot.circle.OffsetFilaments` would be applying its
    coincident-filament offset, and the two paths stop agreeing.
    """

    quadrature: ClassVar[tuple[int, int] | None] = None
    """Boundary quadrature as ``(n_panels, n_nodes)``; ``None`` uses the kernel's
    own rule.

    Lowering it is a false economy: at ``(8, 24)`` the flux still holds a few
    parts in ten million but ``B_Z`` degrades to ~1e-02 relative against the
    closed-form rectangle oracle, which :mod:`tests.test_biotpolygon` pins at
    ``rtol=1e-10``. Raise it if a section is far more elongated than a plasma
    cell; otherwise leave it alone.
    """

    banded: ClassVar[bool] = False
    """Route pairs through :mod:`nova.biot.bandedcoupling` instead of the exact rule.

    The three-band scheme evaluates the converged rule only where the finite
    section is physically resolved, a reduced rule out to the section's own far
    seam, and a moment-corrected filament beyond it — about one percent of the
    exact-everywhere quadrature node count on a plasma grid, with every component
    held to one part in a million of its peak against the exact lane.

    The default is ``False``: exact everywhere stays the shipped lane and the
    reference the banded one is measured against. Flipping the plasma-coupling
    default is a separate decision from having the scheme available.
    """

    @classmethod
    @contextmanager
    def configured(cls, *, standoff=..., quadrature=..., banded=...):
        """Apply a temporary configuration for the duration of a solve.

        The element is built inside :class:`nova.biot.solve.Solve`, so there is
        no per-call argument to thread a configuration through; this scopes a
        change instead of leaving the class mutated::

            with PolySection.configured(banded=True):  # three-band scheme
                coilset.plasmagrid.solve()
        """
        previous = (cls.standoff, cls.quadrature, cls.banded)
        if standoff is not ...:
            cls.standoff = standoff
        if quadrature is not ...:
            cls.quadrature = quadrature
        if banded is not ...:
            cls.banded = banded
        try:
            yield cls
        finally:
            cls.standoff, cls.quadrature, cls.banded = previous

    @staticmethod
    def section_radius(vertices: np.ndarray) -> float:
        """Return the section's bounding radius about its area centroid [m]."""
        vertices = np.asarray(vertices, dtype=np.float64)
        centre = section_centroid(vertices)
        return float(np.max(np.hypot(*(vertices - centre).T)))

    @classmethod
    def near_band(
        cls, target_r: np.ndarray, target_z: np.ndarray, vertices: np.ndarray
    ) -> np.ndarray:
        """Return the mask of targets inside the section's standoff band.

        Every target is inside the band when the standoff is ``None`` or
        infinite, which is how *exact everywhere* is expressed.
        """
        target_r = np.asarray(target_r, dtype=np.float64)
        if cls.standoff is None or not np.isfinite(cls.standoff):
            return np.ones(target_r.shape, dtype=bool)
        vertices = np.asarray(vertices, dtype=np.float64)
        centre = section_centroid(vertices)
        distance = np.hypot(
            target_r - centre[0],
            np.asarray(target_z, dtype=np.float64) - centre[1],
        )
        return distance < cls.standoff * cls.section_radius(vertices)

    @classmethod
    def section_greens(
        cls, target_r: np.ndarray, target_z: np.ndarray, vertices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(psi, Br, Bz)`` per ampere: exact near the section, point far.

        The returned arrays are shaped like ``target_r``. With ``banded`` set, the
        three-band scheme handles every pair instead and neither ``standoff`` nor
        ``quadrature`` applies — the bands carry their own measured rules.
        """
        target_r = np.asarray(target_r, dtype=np.float64)
        target_z = np.asarray(target_z, dtype=np.float64)
        if cls.banded:
            return banded_greens(target_r, target_z, vertices)
        centre = section_centroid(vertices)
        psi = greens_psi(target_r, target_z, centre[0], centre[1])
        bz, br = greens_bz_br(target_r, target_z, centre[0], centre[1])
        near = cls.near_band(target_r, target_z, vertices)
        if near.any():
            psi, br, bz = psi.copy(), br.copy(), bz.copy()
            rule = (
                {}
                if cls.quadrature is None
                else dict(zip(("n_panels", "n_nodes"), cls.quadrature))
            )
            psi[near], br[near], bz[near] = polygon_greens(
                target_r[near], target_z[near], vertices, **rule
            )
        return psi, br, bz

    @cached_property
    def _section_vertices(self) -> list[np.ndarray]:
        """Return each source element's section vertices as ``(n, 2)`` r-z arrays."""
        vertices = []
        for poly in np.asarray(self.source["poly"]):
            points = np.asarray(poly.points, dtype=np.float64)[:, [0, 2]]
            if len(points) > 1 and np.allclose(points[0], points[-1]):
                points = points[:-1]  # drop the repeated closing vertex
            vertices.append(points)
        return vertices

    @cached_property
    def _coupling(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the ``(psi, Br, Bz)`` matrices, shape ``(target, source)``."""
        target_r = np.asarray(self.target("r"))
        target_z = np.asarray(self.target("z"))
        psi = np.empty(target_r.shape)
        br = np.empty(target_r.shape)
        bz = np.empty(target_r.shape)
        for column, vertices in enumerate(self._section_vertices):
            psi[:, column], br[:, column], bz[:, column] = self.section_greens(
                target_r[:, column], target_z[:, column], vertices
            )
        return psi, br, bz

    @cached_property
    def Psi(self):
        """Return the total poloidal flux array [Wb/A]."""
        return self._coupling[0]

    @cached_property
    def Aphi(self):
        """Return the toroidal vector potential array [Wb/(m.A)]."""
        return self.Psi / (2 * np.pi * self.mu_0 * self.target("r"))

    @cached_property
    def Br(self):
        """Return the radial field array [T/A]."""
        return self._coupling[1]

    @cached_property
    def Bz(self):
        """Return the vertical field array [T/A]."""
        return self._coupling[2]
