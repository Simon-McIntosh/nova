"""Distance-banded coupling for toroidal conductors of polygonal cross-section.

The exact polygon kernel (:func:`nova.biot.polygon.polygon_greens`) is
affordable everywhere -- a 2000-cell plasma build measures at about seven
minutes on sixteen cores -- but it spends the same 768 quadrature nodes on a
pair thirty section radii apart as on a pair inside the conductor. That is
wasted work rather than accuracy: the phi integrand is analytic off the section
boundary, so its convergence is spectral and the number of nodes a pair needs
falls steeply with distance. This module bins pairs by distance and gives each
bin a treatment that is already converged there.

The bands
---------
Distance is measured to the section CONTOUR, not to its centroid. What sets a
pair's difficulty is how close the target comes to the boundary the kernel
integrates over -- a target one section radius from the centroid is against the
boundary in one direction and well clear of it in another, and the centroid
distance cannot tell the two apart.

======  =====================================  ===============================
band    contour distance / section radius      treatment
======  =====================================  ===============================
near    below ``NEAR_LIMIT``                   the exact kernel -- nothing
                                               approximated
mid     ``NEAR_LIMIT`` to the far seam         ``MID_RULE`` -- reduced rule
far     beyond the far seam                    moment-corrected filament
======  =====================================  ===============================

Which exact kernel serves the near band is the caller's choice: the ``NEAR_RULE``
boundary quadrature, or the closed form of :mod:`nova.biot.polygonanalytic`
through ``closed_form=True``. Here, unlike on the exact-everywhere lane, that is a
genuine trade rather than a free win.

The closed form is the more accurate one, and by a margin that lands exactly in
this band: measured on a real plasma grid, inside a quarter of a contour radius
the shipped ``(16, 48)`` rule is 4.4e-03 out on B_R and 4.3e-02 on B_Z of local
magnitude, where the closed form is at 1.6e-13 and 2.6e-07 against a 64x96
oracle. Refining the rule through (24,64), (32,80), (48,96), (64,96) to (96,128)
takes B_Z from 4.35e-02 to 1.35e-10, converging ONTO the closed form -- so the
quadrature is the error, and it is the error where a plasma matrix's diagonal
lives.

It is also the DEARER one here, which the exact-everywhere lane does not predict.
The closed form amortises up to three live corner parts across a call, so its rate
improves 38-fold between 8 and 4096 pairs in one call while the quadrature's
improves 1.3-fold; the two cross at 64 pairs. A near band holds about thirteen
pairs of a whole column, an order below that crossing, so on the near-band call
itself the quadrature is 4.1x cheaper -- 984.2 us/pair against 4023.4 -- which
dilutes to 2.3x over a whole banded column (13.6 against 31.5) and 1.9x over a
real 560-cell build (19.2 s against 36.4). Which one to prefer depends on whether
that band's accuracy or the build's cost is the binding constraint, so this module
takes it as an argument and states the numbers rather than choosing.

Note for anyone reading a banded rate off an idealised section: do not. A real
grid's wall cuts the boundary ring into three- to twelve-corner cells, 120 of a
560-cell grid's carrying enough skew for the wide 16-radius far seam, so its mid
band holds 12.1% of pairs where an idealised hexagon holds 2.6 -- and at the mid
rule's 268 us/pair against the far filament's 0.7 that fraction is most of a
banded column's cost. A rate projected from the hexagon alone is optimistic by
about 4.5x (13.6 projected against 61.1 measured on a real build).

The limits are measured, not budgeted: each is the distance beyond which EVERY
component -- flux, radial field and vertical field separately -- holds to one
part in a million of its own peak against the converged rule. Per component
matters: at the reduced rule the flux error and the vertical-field error differ
by more than two orders of magnitude, so a flux-only bound passes a rule that
has lost two digits of B_Z.

The far seam is the section's own, from :func:`mid_limit`, because the far
model's accuracy depends on the section's symmetry: a regular hexagonal plasma
cell has no third moments and is fourth-order accurate from ``MID_LIMIT``, while
a cell clipped by the first wall keeps a third-order residual and needs
``SKEWED_MID_LIMIT``.

The far field is a MOMENT-corrected filament and not a bare one. A bare centroid
filament does not converge to a finite section for a full toroidal ring at any
distance -- the curvature its second moment multiplies is set by the major
radius, so the relative error flattens onto an ``(a / R0)^2`` floor. See
:func:`nova.biot.greens.moment_filament`.

Fixed shapes
------------
The rule is fixed WITHIN each band rather than varied per pair. A batched or
sharded assembly needs every kernel call in a tile to have one identical array
shape; three bands give three shapes, and binning by geometry -- rather than by a
per-pair error estimate -- keeps the seams reproducible for a stored operator.

Quantities and signs match :func:`nova.biot.polygon.polygon_greens`: per ampere
of total conductor current, total poloidal flux psi [Wb/A] and field components
[T/A].
"""

from __future__ import annotations

import numpy as np

from nova.biot.greens import moment_filament, section_centroid, third_moments
from nova.biot.polygon import polygon_greens
from nova.biot.polygonanalytic import polygon_analytic_greens

NEAR_LIMIT = 2.2
"""Near/mid seam, in section radii of contour distance."""

MID_LIMIT = 6.8
"""Mid/far seam for a section with no surviving skew, in section radii."""

SKEWED_MID_LIMIT = 16.0
"""Mid/far seam for a section whose third moments survive, in section radii.

A section symmetric about its own centroid -- a regular hexagonal plasma cell --
has no third moments, so the corrected filament's leading residual there is
FOURTH order and clears one part in a million from ``MID_LIMIT``. A cell clipped
by the first wall keeps a third-order residual: measured on one, the worst
component sits at 3.7e-05 of the local magnitude at the unskewed seam and needs
this distance instead. Both numbers are measured on the section shape they
describe; one budget applied to both would be wrong in one direction or the
other.
"""

SKEW_TOLERANCE = 1.0e-3
"""Normalised third moment above which a section takes the wider far seam.

``max|m_ijk| / a^3`` for section radius ``a``. The far-field residual is
proportional to the skew, and the measured wall-clipped cell sits at 4.2e-03 with
a residual of 3.7e-06 of local magnitude at ``MID_LIMIT`` once the skew term is
carried; a section an order of magnitude less skew than that is inside one part
in a million there, which is where this sits.
"""

NEAR_RULE = (16, 48)
"""``(n_panels, n_nodes)`` for the near band -- the converged rule."""

MID_RULE = (8, 24)
"""``(n_panels, n_nodes)`` for the mid band."""

FILAMENT_EVALUATIONS = 13
"""Green's-function evaluations the far band spends per pair, worst case.

Five for a section symmetric about its own axes, as a regular hexagonal plasma
cell is -- its cross and third moments vanish. Thirteen once both survive. Carried
here as the batch-shape figure, the padded worst case, rather than as the count a
given section takes.
"""


def section_radius(vertices: np.ndarray) -> float:
    """Return the section's bounding radius about its area centroid [m]."""
    vertices = np.asarray(vertices, dtype=np.float64)
    offset = vertices - section_centroid(vertices)
    return float(np.max(np.hypot(*offset.T)))


def section_skew(vertices: np.ndarray) -> float:
    """Return the section's largest third moment, normalised by its radius cubed.

    Dimensionless, so it compares a millimetre section with a metre one, and zero
    to round-off for any section symmetric about its own centroid.
    """
    return max(abs(moment) for moment in third_moments(vertices)) / (
        section_radius(vertices) ** 3
    )


def mid_limit(vertices: np.ndarray) -> float:
    """Return the section's own mid/far seam, in section radii of contour distance.

    Derived from the section's skew rather than fixed: what sets the seam is where
    the corrected filament's residual falls under the per-component bound, and
    that residual is third order for a skewed section and fourth order for a
    symmetric one.
    """
    return MID_LIMIT if section_skew(vertices) <= SKEW_TOLERANCE else SKEWED_MID_LIMIT


def contour_distance(
    target_r: np.ndarray, target_z: np.ndarray, vertices: np.ndarray
) -> np.ndarray:
    """Return each target's distance to the section boundary [m], like ``target_r``.

    Unsigned: a target inside the conductor reports its distance to the nearest
    edge, which is below one section radius and so always lands in the near band.
    Computed as the minimum over edges of the point-to-segment distance, which is
    exact for a polygon and needs no inside test.
    """
    target_r = np.asarray(target_r, dtype=np.float64)
    target_z = np.asarray(target_z, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    start = vertices[:, None, :]
    edge = (np.roll(vertices, -1, axis=0) - vertices)[:, None, :]
    offset = np.stack([target_r.ravel(), target_z.ravel()], axis=-1)[None, :, :] - start
    length2 = np.sum(edge * edge, axis=-1)
    reach = np.clip(
        np.sum(offset * edge, axis=-1) / np.where(length2 > 0.0, length2, 1.0), 0.0, 1.0
    )
    gap = offset - reach[..., None] * edge
    return np.min(np.hypot(gap[..., 0], gap[..., 1]), axis=0).reshape(target_r.shape)


def band(
    target_r: np.ndarray,
    target_z: np.ndarray,
    vertices: np.ndarray,
    *,
    near_limit: float = NEAR_LIMIT,
    far_limit: float | None = None,
) -> np.ndarray:
    """Return each target's band index: 0 near, 1 mid, 2 far, like ``target_r``.

    A pure function of the contour distance and the section's own geometry, so the
    seams sit at fixed places and a stored operator's band assignment is
    reproducible. ``far_limit`` defaults to the section's own :func:`mid_limit`.
    """
    if far_limit is None:
        far_limit = mid_limit(vertices)
    offset = contour_distance(target_r, target_z, vertices) / section_radius(vertices)
    return (offset >= near_limit).astype(np.int_) + (offset >= far_limit)


def quadrature_nodes(assignment: np.ndarray, *, closed_form: bool = False) -> int:
    """Return the phi-quadrature nodes a band assignment spends in total.

    The far band spends none: it is a handful of point Green's-function
    evaluations, some three orders of magnitude below one node of the converged
    rule, and counting them as nodes would misstate the saving in the direction
    of flattering the scheme. With ``closed_form`` the near band spends none
    either -- its angular integral is done in closed form, and what it does spend
    is a per-corner reduction that no node count describes. Compare cost per pair
    for that route, not nodes.
    """
    assignment = np.asarray(assignment)
    near = 0 if closed_form else int(np.count_nonzero(assignment == 0))
    mid = int(np.count_nonzero(assignment == 1))
    return near * NEAR_RULE[0] * NEAR_RULE[1] + mid * MID_RULE[0] * MID_RULE[1]


def near_greens(
    target_r: np.ndarray,
    target_z: np.ndarray,
    vertices: np.ndarray,
    *,
    closed_form: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the near band's ``(psi, B_R, B_Z)`` through the chosen exact kernel.

    Both routes are exact in the sense the band needs -- neither drops a term of
    the finite-section physics. They differ in how the angular integral is done,
    and therefore in what limits their accuracy: the quadrature by how well 768
    nodes resolve an integrand that is singular as the target reaches the
    boundary, the closed form by round-off in a reduction that has no integrand
    left there at all.
    """
    if closed_form:
        return polygon_analytic_greens(target_r, target_z, vertices)
    return polygon_greens(
        target_r, target_z, vertices, n_panels=NEAR_RULE[0], n_nodes=NEAR_RULE[1]
    )


def banded_greens(
    target_r: np.ndarray,
    target_z: np.ndarray,
    vertices: np.ndarray,
    *,
    near_limit: float = NEAR_LIMIT,
    far_limit: float | None = None,
    closed_form: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(psi, B_R, B_Z)`` per ampere, each pair through its band's treatment.

    ``vertices`` -- ``(n, 2)`` of the section's ``(r, z)`` corners, either
    orientation, no repeated closing vertex. Returned arrays are shaped like
    ``target_r``. ``closed_form`` selects the near band's exact kernel; see
    :func:`near_greens`.
    """
    target_r = np.asarray(target_r, dtype=np.float64)
    target_z = np.asarray(target_z, dtype=np.float64)
    assignment = band(
        target_r, target_z, vertices, near_limit=near_limit, far_limit=far_limit
    )
    psi = np.empty(target_r.shape)
    br = np.empty(target_r.shape)
    bz = np.empty(target_r.shape)
    near = assignment == 0
    if near.any():
        psi[near], br[near], bz[near] = near_greens(
            target_r[near], target_z[near], vertices, closed_form=closed_form
        )
    inside = assignment == 1
    if inside.any():
        psi[inside], br[inside], bz[inside] = polygon_greens(
            target_r[inside],
            target_z[inside],
            vertices,
            n_panels=MID_RULE[0],
            n_nodes=MID_RULE[1],
        )
    far = assignment == 2
    if far.any():
        psi[far], br[far], bz[far] = moment_filament(
            target_r[far], target_z[far], vertices
        )
    return psi, br, bz


__all__ = [
    "NEAR_LIMIT",
    "MID_LIMIT",
    "SKEWED_MID_LIMIT",
    "SKEW_TOLERANCE",
    "NEAR_RULE",
    "MID_RULE",
    "FILAMENT_EVALUATIONS",
    "section_radius",
    "section_skew",
    "mid_limit",
    "contour_distance",
    "band",
    "quadrature_nodes",
    "near_greens",
    "banded_greens",
]
