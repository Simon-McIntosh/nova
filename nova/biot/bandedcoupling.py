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
near    below ``NEAR_LIMIT``                   ``NEAR_RULE`` -- exact, nothing
                                               approximated
mid     ``NEAR_LIMIT`` to the far seam         ``MID_RULE`` -- reduced rule
far     beyond the far seam                    moment-corrected filament
======  =====================================  ===============================

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


def quadrature_nodes(assignment: np.ndarray) -> int:
    """Return the phi-quadrature nodes a band assignment spends in total.

    The far band spends none: it is a handful of point Green's-function
    evaluations, some three orders of magnitude below one node of the converged
    rule, and counting them as nodes would misstate the saving in the direction
    of flattering the scheme.
    """
    assignment = np.asarray(assignment)
    near = int(np.count_nonzero(assignment == 0))
    mid = int(np.count_nonzero(assignment == 1))
    return near * NEAR_RULE[0] * NEAR_RULE[1] + mid * MID_RULE[0] * MID_RULE[1]


def banded_greens(
    target_r: np.ndarray,
    target_z: np.ndarray,
    vertices: np.ndarray,
    *,
    near_limit: float = NEAR_LIMIT,
    far_limit: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(psi, B_R, B_Z)`` per ampere, each pair through its band's treatment.

    ``vertices`` -- ``(n, 2)`` of the section's ``(r, z)`` corners, either
    orientation, no repeated closing vertex. Returned arrays are shaped like
    ``target_r``.
    """
    target_r = np.asarray(target_r, dtype=np.float64)
    target_z = np.asarray(target_z, dtype=np.float64)
    assignment = band(
        target_r, target_z, vertices, near_limit=near_limit, far_limit=far_limit
    )
    psi = np.empty(target_r.shape)
    br = np.empty(target_r.shape)
    bz = np.empty(target_r.shape)
    for index, rule in enumerate((NEAR_RULE, MID_RULE)):
        inside = assignment == index
        if inside.any():
            psi[inside], br[inside], bz[inside] = polygon_greens(
                target_r[inside],
                target_z[inside],
                vertices,
                n_panels=rule[0],
                n_nodes=rule[1],
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
    "banded_greens",
]
