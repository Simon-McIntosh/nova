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

Two exact kernels
-----------------
Independently of how pairs are binned, the *exact* treatment itself comes two
ways. :func:`nova.biot.polygon.polygon_greens` reduces the section integral to a
contour sum and does the remaining angular integral by quadrature;
:func:`nova.biot.polygonanalytic.polygon_analytic_greens` does that integral in
closed form as well, leaving only two smooth ``arsinh`` residuals per corner.
``closed_form`` selects the second, and it composes with either binning — exact
everywhere becomes closed-form everywhere, and the three-band scheme's near band
takes the closed form where the quadrature's own singularity is.

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
from nova.biot.polygonanalytic import polygon_analytic_greens


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

    closed_form: ClassVar[bool] = True
    """Take the exact kernel from :mod:`nova.biot.polygonanalytic` in closed form.

    The angular integral the boundary quadrature does numerically is done
    analytically instead, leaving two smooth ``arsinh`` residuals per corner —
    the same physics through a different evaluation, and on the measured evidence
    a strictly better one on both counts, which is why it is the default.

    Cheaper: a hexagonal plasma cell costs 171 µs/pair against the ``(16, 48)``
    rule's 858 (5.0×), because a corner is evaluated once for both its edges
    rather than 768 quadrature nodes being spent per pair. Shape-dependent, as
    that implies, where the quadrature is not: the cost tracks the corner count,
    so a wall-clipped cell costs more than a regular one and a grid gets cheaper
    per pair as it refines and the clipped fraction falls.

    More accurate, and most where it matters: it holds 1e-10 of local magnitude
    on all three components across the whole acceptance gate, and it is finite and
    accurate ON the contour and ON a vertex, where a boundary quadrature is
    integrating through its own singularity. Measured on a real 179-cell plasma
    grid, the worst off-diagonal pair is a neighbour's centre 0.001 contour radii
    outside the source cell, where the quadrature is 2.9e-03 out on B_Z; refining
    it to 1024 panels brings it to 2.1e-12 OF THE CLOSED FORM's value, so the
    closed form is the value and the quadrature was the error.

    Independent of ``standoff`` and ``banded`` in MEANING — they choose how pairs
    are binned, this chooses what the exact treatment is, so with neither set it is
    closed-form everywhere and with ``banded`` it serves the near band.
    ``quadrature`` has no meaning for it: the closed form's own residual node count
    is fixed by its acceptance gate. Set it ``False`` to get the boundary
    quadrature back, which is what the closed form's equivalence gate is measured
    against.

    Independent in meaning, NOT in cost, which is the one thing measuring it
    disproved. The closed form holds three corner parts live and amortises them
    across a call, so its rate falls 38× between 8 and 4096 pairs in one call
    (6532 → 170 µs/pair) where the quadrature — which builds one angular rule and
    reuses it — falls only 1.3× (1084 → 851). **The two cross at 64 pairs per
    call.** An exact-everywhere column hands the kernel every pair at once and the
    closed form wins five-fold; the three-band scheme hands its near band about
    thirteen, an order below the crossing, and there the quadrature is 2.3×
    cheaper. So a banded build pays about 1.9× for the closed form's near-contour
    accuracy, and a caller minimising a banded build should turn this off. It is
    not switched automatically on batch width: which kernel ran would then depend
    on how a caller happened to group its pairs, and a stored operator's values
    have to be reproducible from its geometry alone.
    """

    @classmethod
    @contextmanager
    def configured(cls, *, standoff=..., quadrature=..., banded=..., closed_form=...):
        """Apply a temporary configuration for the duration of a solve.

        The element is built inside :class:`nova.biot.solve.Solve`, so there is
        no per-call argument to thread a configuration through; this scopes a
        change instead of leaving the class mutated::

            with PolySection.configured(banded=True):  # three-band scheme
                coilset.plasmagrid.solve()

            with PolySection.configured(closed_form=True):  # exact, in closed form
                coilset.plasmagrid.solve()
        """
        previous = (cls.standoff, cls.quadrature, cls.banded, cls.closed_form)
        if standoff is not ...:
            cls.standoff = standoff
        if quadrature is not ...:
            cls.quadrature = quadrature
        if banded is not ...:
            cls.banded = banded
        if closed_form is not ...:
            cls.closed_form = closed_form
        try:
            yield cls
        finally:
            cls.standoff, cls.quadrature, cls.banded, cls.closed_form = previous

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
        three-band scheme handles every pair instead and ``standoff`` does not
        apply — the bands carry their own measured rules. ``closed_form`` selects
        which exact kernel serves either arrangement, and ``quadrature`` applies
        only to the boundary-quadrature one.
        """
        target_r = np.asarray(target_r, dtype=np.float64)
        target_z = np.asarray(target_z, dtype=np.float64)
        if cls.banded:
            return banded_greens(
                target_r, target_z, vertices, closed_form=cls.closed_form
            )
        centre = section_centroid(vertices)
        psi = greens_psi(target_r, target_z, centre[0], centre[1])
        bz, br = greens_bz_br(target_r, target_z, centre[0], centre[1])
        near = cls.near_band(target_r, target_z, vertices)
        if near.any():
            psi, br, bz = psi.copy(), br.copy(), bz.copy()
            psi[near], br[near], bz[near] = cls.exact_greens(
                target_r[near], target_z[near], vertices
            )
        return psi, br, bz

    @classmethod
    def exact_greens(
        cls, target_r: np.ndarray, target_z: np.ndarray, vertices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(psi, Br, Bz)`` from the configured exact kernel.

        The one place the two evaluations are chosen between, so the standoff
        band, the banded scheme's near band and a direct call cannot disagree
        about which one is in force.
        """
        if cls.closed_form:
            return polygon_analytic_greens(target_r, target_z, vertices)
        rule = (
            {}
            if cls.quadrature is None
            else dict(zip(("n_panels", "n_nodes"), cls.quadrature))
        )
        return polygon_greens(target_r, target_z, vertices, **rule)

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
