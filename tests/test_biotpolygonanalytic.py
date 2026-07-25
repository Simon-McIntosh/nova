"""The acceptance gate for any polygon-section evaluation, quadrature or closed form.

Urankar's Part V performs the toroidal integration analytically, leaving only two
smooth ``arsinh`` integrals in the flux component. The kernel that ships today
instead integrates the whole edge integrand numerically. Both are supposed to
compute the same thing, so neither should be trusted against itself: this module
holds ONE oracle and ONE set of targets, and every candidate evaluation is held
to it.

The oracle is the boundary quadrature at a rule far above the shipped default
(64 x 96 against 16 x 48). That is a legitimate reference because the phi
integrand is analytic off the section boundary and the quadrature converges
spectrally, so raising the rule raises accuracy until it saturates at round-off
-- which the first test here demonstrates rather than assumes.

Distance is measured to the section BOUNDARY, not to its centroid. For a polygon
those are different questions: a target one section radius from the centroid is
a centimetre outside a vertex and touching a face, and it is proximity to the
CONTOUR that the edge integrand is sensitive to. Measuring from the centroid
mixes converged and unconverged targets into the same band and makes the whole
gate unreadable.

The tolerance is therefore banded rather than uniform, and both the standoff and
the tolerances are measured rather than assumed. Worst deviation over all four
sections, against a 96 x 128 rule:

    contour distance     oracle 64 x 96      shipped 16 x 48
    >= 0.0 radii            3.9e-11             2.6e-04
    >= 0.1 radii            9.5e-12             1.8e-06
    >= 0.2 radii            9.8e-12             5.6e-08
    >= 0.5 radii            1.1e-11             4.6e-11
    >= 1.0 radii            1.3e-11             5.1e-12

So the oracle is converged to about 1e-11 everywhere including on the contour,
which is what lets it serve as the reference at all; the shipped quadrature needs
half a section radius of standoff to reach 1e-10 and loses six orders on the
contour itself. Beyond the standoff the gate is 1e-10; inside it the band is
1e-03. A closed form should BEAT the near-contour band -- it has no quadrature
there to converge -- and tightening that constant is how a future transcription
demonstrates it was worth having.

``CANDIDATES`` is the extension point. Adding a closed-form evaluation means
adding one entry and inheriting the whole gate: every band, all directions,
targets inside the conductor, four section shapes, and the two structural
reductions (a horizontal edge must contribute nothing, and a section with no
sloped edge must reproduce the rectangle kernel we already trust).
"""

import numpy as np
import pytest

from nova.biot.greens import cylinder_greens
from nova.biot.polygon import polygon_greens

COMPONENTS = ("psi", "Br", "Bz")
ORACLE_RULE = dict(n_panels=64, n_nodes=96)

# Distance to the section contour, in section radii, beyond which the boundary
# quadrature is converged and both formulations must agree to round-off.
CONTOUR_STANDOFF = 0.5
GATE = 1e-10
NEAR_CONTOUR_GATE = 1e-3

R0 = 6.2


def hexagon(r0=R0, z0=0.0, radius=0.06):
    """Return the plasma cell section: a regular hexagon."""
    angle = np.pi / 6 + np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    return np.column_stack([r0 + radius * np.cos(angle), z0 + radius * np.sin(angle)])


def rectangle(r0=R0, z0=0.0, width=0.1, height=0.08):
    """Return an axis-aligned rectangle -- every edge has zero or infinite slope."""
    return np.array(
        [
            [r0 - width / 2, z0 - height / 2],
            [r0 + width / 2, z0 - height / 2],
            [r0 + width / 2, z0 + height / 2],
            [r0 - width / 2, z0 + height / 2],
        ]
    )


def trapezium(r0=2.0, z0=0.1):
    """Return a slanted, non-symmetric quadrilateral -- every edge slope differs."""
    return np.array(
        [
            [r0, z0],
            [r0 + 0.25, z0 - 0.05],
            [r0 + 0.3, z0 + 0.2],
            [r0 - 0.05, z0 + 0.12],
        ]
    )


def thin_plate(r0=3.0):
    """Return a high aspect-ratio parallelogram -- the hardest conditioning."""
    return np.array([[r0, 0.0], [r0 + 0.4, 0.06], [r0 + 0.4, 0.075], [r0, 0.015]])


SECTIONS = {
    "hexagon": hexagon(),
    "rectangle": rectangle(),
    "trapezium": trapezium(),
    "thin_plate": thin_plate(),
}


def section_radius(vertices):
    """Return the section's bounding radius about its centroid."""
    vertices = np.asarray(vertices, float)
    centre = vertices.mean(axis=0)
    return float(np.max(np.hypot(*(vertices - centre).T)))


def gate_targets(vertices, directions=16):
    """Return targets spanning inside, near, mid and far, in every direction.

    Offsets are in section radii from the centroid and start inside the
    conductor, because the interior is the whole reason a finite-area kernel
    exists. Directions matter as much as distance: a polygon's field is not
    isotropic, and a term transcribed with the wrong sign on a ``cos 2 alpha``
    shows up towards a vertex long before it shows up towards a face.
    """
    vertices = np.asarray(vertices, float)
    centre = vertices.mean(axis=0)
    radius = section_radius(vertices)
    scales = np.array([0.3, 0.7, 1.05, 1.4, 2.0, 3.0, 6.0, 12.0, 30.0])
    angle = np.linspace(0.0, 2.0 * np.pi, directions, endpoint=False)
    offset = np.repeat(scales * radius, directions)
    bearing = np.tile(angle, scales.size)
    return centre[0] + offset * np.cos(bearing), centre[1] + offset * np.sin(bearing)


def contour_distance(target_r, target_z, vertices):
    """Return each target's distance to the section contour, in section radii.

    Signed magnitude is not wanted: a target just inside a face and one just
    outside it are equally hard for the edge integrand, so the unsigned distance
    to the polygon boundary is the quantity that predicts convergence.
    """
    from shapely.geometry import Point, Polygon

    boundary = Polygon(np.asarray(vertices, float)).boundary
    radius = section_radius(vertices)
    return (
        np.array(
            [
                boundary.distance(Point(r, z))
                for r, z in zip(np.ravel(target_r), np.ravel(target_z))
            ]
        )
        / radius
    )


def converged(target_r, target_z, vertices):
    """Return the mask of targets far enough from the contour to gate at 1e-10."""
    return contour_distance(target_r, target_z, vertices) >= CONTOUR_STANDOFF


def oracle(target_r, target_z, vertices):
    """Return the reference ``(psi, Br, Bz)``: boundary quadrature, high order."""
    return polygon_greens(target_r, target_z, vertices, **ORACLE_RULE)


def shipped(target_r, target_z, vertices):
    """Return the shipped evaluation, at the kernel's own default rule."""
    return polygon_greens(target_r, target_z, vertices)


# Every entry is held to the full gate below. A closed-form Urankar Part V
# evaluation joins by adding one line here.
CANDIDATES = {"boundary_quadrature": shipped}


def worst_relative(got, want):
    """Return the max deviation, relative to the component's own peak."""
    scale = np.max(np.abs(want))
    return float(np.max(np.abs(got - want)) / scale)


def test_the_oracle_is_converged_so_it_can_be_used_as_one():
    """Raising the rule past the oracle must not move it.

    Without this the gate is circular: a reference that is still converging
    would let a candidate pass by matching the reference's error rather than the
    integral's value.
    """
    vertices = hexagon()
    target_r, target_z = gate_targets(vertices)
    keep = converged(target_r, target_z, vertices)
    assert keep.sum() > 0.5 * keep.size, "target set is dominated by the contour"
    finer = polygon_greens(target_r, target_z, vertices, n_panels=96, n_nodes=128)
    # the oracle is claimed converged EVERYWHERE, contour included -- that is
    # what distinguishes it from the rule under test
    for name, coarse, fine in zip(
        COMPONENTS, oracle(target_r, target_z, vertices), finer
    ):
        assert worst_relative(coarse, fine) < GATE, name


@pytest.mark.parametrize("candidate", sorted(CANDIDATES))
@pytest.mark.parametrize("section", sorted(SECTIONS))
def test_a_candidate_matches_the_oracle_everywhere(candidate, section):
    """Per component, over near, mid, far and interior targets, all directions."""
    vertices = SECTIONS[section]
    target_r, target_z = gate_targets(vertices)
    keep = converged(target_r, target_z, vertices)
    reference = oracle(target_r, target_z, vertices)
    computed = CANDIDATES[candidate](target_r, target_z, vertices)
    for name, got, want in zip(COMPONENTS, computed, reference):
        deviation = worst_relative(got[keep], want[keep])
        assert deviation <= GATE, f"{candidate}/{section}/{name}: {deviation:.3e}"
        near = worst_relative(got[~keep], want[~keep])
        assert near <= NEAR_CONTOUR_GATE, (
            f"{candidate}/{section}/{name} near contour: {near:.3e}"
        )


@pytest.mark.parametrize("candidate", sorted(CANDIDATES))
def test_a_candidate_holds_the_gate_band_by_band(candidate):
    """Reported per distance band, so a failure says WHERE it went wrong.

    A single worst-case number over all targets hides whether a term is wrong
    in the near field, the far field, or only inside the conductor -- which is
    the first thing a transcription needs to know.
    """
    vertices = hexagon()
    target_r, target_z = gate_targets(vertices)
    distance = contour_distance(target_r, target_z, vertices)
    reference = oracle(target_r, target_z, vertices)
    computed = CANDIDATES[candidate](target_r, target_z, vertices)
    bands = {
        "contour": (distance < CONTOUR_STANDOFF, NEAR_CONTOUR_GATE),
        "near": ((distance >= CONTOUR_STANDOFF) & (distance < 2.0), GATE),
        "mid": ((distance >= 2.0) & (distance < 11.0), GATE),
        "far": (distance >= 11.0, GATE),
    }
    failures = {}
    for label, (mask, tolerance) in bands.items():
        assert mask.any(), f"band {label} is empty"
        for name, got, want in zip(COMPONENTS, computed, reference):
            deviation = worst_relative(got[mask], want[mask])
            if deviation > tolerance:
                failures[f"{label}/{name}"] = deviation
    assert not failures, f"{candidate}: {failures}"


@pytest.mark.parametrize("candidate", sorted(CANDIDATES))
def test_a_horizontal_edge_contributes_nothing(candidate):
    """Paper eq 7a. Splitting a horizontal edge in two must change nothing.

    A closed form divides by the edge's ``dz`` to form its slope, so this is
    where a transcription either handles the degenerate edge or produces a
    silent infinity.
    """
    vertices = rectangle()
    r_low, r_high = vertices[0, 0], vertices[1, 0]
    z_low, z_high = vertices[0, 1], vertices[2, 1]
    split = np.array(
        [
            [r_low, z_low],
            [0.5 * (r_low + r_high), z_low],  # extra vertex on the bottom edge
            [r_high, z_low],
            [r_high, z_high],
            [r_low, z_high],
        ]
    )
    target_r, target_z = gate_targets(vertices)
    keep = converged(target_r, target_z, vertices)
    plain = CANDIDATES[candidate](target_r, target_z, vertices)
    divided = CANDIDATES[candidate](target_r, target_z, split)
    for name, got, want in zip(COMPONENTS, divided, plain):
        assert np.all(np.isfinite(got)), name
        assert worst_relative(got[keep], want[keep]) <= GATE, name


@pytest.mark.parametrize("candidate", sorted(CANDIDATES))
def test_a_section_with_no_sloped_edge_reproduces_the_rectangle_kernel(candidate):
    """The ``b1 = 0`` limit, against the rectangle implementation already trusted.

    ``cylinder_greens`` is the reference here rather than a fresh transcription
    of Part III, because it is what the rest of the suite pins. Targets on the
    section's horizontal faces are excluded: there its corner antiderivative
    carries a ``sign(z_corner - z_target)`` dead-band that collapses when the
    difference is zero, and it returns a discontinuous flux -- one such target
    returns a negative psi where its neighbours are smooth and positive. That is
    a defect in the rectangle kernel, not a disagreement about the polygon.
    """
    width, height = 0.1, 0.08
    vertices = rectangle(width=width, height=height)
    target_r, target_z = gate_targets(vertices, directions=24)
    # stay clear of the contour, where the rectangle kernel's dead-band lives
    keep = contour_distance(target_r, target_z, vertices) >= CONTOUR_STANDOFF
    reference = cylinder_greens(target_r[keep], target_z[keep], R0, 0.0, width, height)
    computed = CANDIDATES[candidate](target_r, target_z, vertices)
    # the rectangle kernel's own 785-point midpoint zeta quadrature limits the
    # agreement to about 1e-03 in B_R, so this is a structural check at the
    # reference's accuracy, not at the 1e-10 gate
    for name, got, want in zip(COMPONENTS, computed, reference):
        assert worst_relative(got[keep], want) < 2e-3, name


# --------------------------------------------------------------------------
# The closed form. Urankar performs the toroidal integration analytically, so per
# edge only two smooth ``arsinh`` integrals remain numerical and everything else
# reduces to complete elliptic integrals. It is held here to two different
# standards, because two different things can go wrong. Whether the REDUCTION is
# right is settled per edge, against a converged quadrature of the very integral
# it replaces, at targets where nothing is near-degenerate. What ACCURACY it then
# delivers over a whole section is a separate, measured question -- and the
# answer depends on the section's aspect ratio, not on the target's distance,
# which is why the closed form does not join CANDIDATES above.


def alpha_quadrature(target_r, target_z, edge, which, nodes=600):
    """Return the edge integral the closed form replaces, by direct quadrature.

    ``W(u) = 4 integral_0^(pi/2) da (-cos 2a) g(u, a)`` with ``phi = pi - 2a``:
    the same edge antiderivative ``g`` the shipped kernel integrates over ``phi``,
    written in the closed form's own angle so the two are the same integral and
    not merely the same flux. Deliberately naive -- literal transcription of
    ``g``, a single high-order rule, no machinery shared with the reduction.
    """
    ra, za, rb, zb = edge
    b1 = (rb - ra) / (zb - za)
    a02 = 1.0 + b1 * b1
    node, weight = np.polynomial.legendre.leggauss(nodes)
    alpha = 0.25 * np.pi * (node + 1.0)
    r = np.atleast_1d(target_r)[:, None]
    z = np.atleast_1d(target_z)[:, None]
    angle = alpha[None, :]
    cos_phi = -np.cos(2.0 * angle)
    sin_phi = np.sin(2.0 * angle)
    sin_two_phi = np.sin(2.0 * (np.pi - 2.0 * angle))
    r1 = ra - b1 * (za - z)
    u = (zb - z) if which else (za - z)
    offset = (r1 + b1 * u) - r * cos_phi
    plane_offset = r1 - r * cos_phi
    g_squared = u * u + (r * sin_phi) ** 2
    b_squared = plane_offset**2 + a02 * (r * sin_phi) ** 2
    distance = np.sqrt(g_squared + offset**2)
    gamma = u + b1 * offset
    integrand = (
        gamma * distance / (2.0 * a02)
        + u * r * cos_phi * np.arcsinh(offset / np.sqrt(g_squared))
        + (b_squared + 2.0 * a02 * r * cos_phi * plane_offset)
        / (2.0 * a02 * np.sqrt(a02))
        * np.arcsinh(gamma / np.sqrt(b_squared))
        - 0.5
        * r
        * r
        * sin_two_phi
        * np.arctan((u * offset - b1 * g_squared) / (r * sin_phi * distance))
    )
    return 4.0 * (integrand @ (0.25 * np.pi * weight * -np.cos(2.0 * alpha)))


# Edges spanning the slopes a polygon presents, including the exactly vertical
# one: ``b1 = 0`` collapses the reduction's quadratic in x to a linear factor,
# sending one of its two roots to infinity, and is the case a formulation built
# only for sloped edges gets wrong outright rather than by a few digits.
REDUCTION_EDGES = {
    "sloped": (2.0, 0.1, 2.25, 0.05),
    "steep": (2.0, 0.1, 2.05, 0.6),
    "vertical": (2.2, -0.3, 2.2, 0.3),
    "reversed": (2.4, 0.5, 2.0, 0.1),
}


@pytest.mark.parametrize("limit", [0, 1])
@pytest.mark.parametrize("edge", sorted(REDUCTION_EDGES))
def test_the_closed_form_reproduces_the_edge_integral_it_replaces(edge, limit):
    """Per edge, per limit: the reduction against a quadrature of the same integral.

    This is the transcription check, and it is deliberately run where the
    reduction's small quantities are not small: targets a good fraction of the
    major radius away from the edge, so the elliptic modulus is clear of 1 and no
    pole sits near an end of the integration range. Anything wrong in the four
    integrations by parts, the arctangent boundary term, or the pole bookkeeping
    shows up here at full size, uncontaminated by conditioning.
    """
    from nova.biot.polygonanalytic import _edge_flux

    target_r = np.array([2.6, 3.0, 1.6, 4.0, 0.9])
    target_z = np.array([0.4, -0.45, 0.35, 0.2, -0.7])
    vertices = np.asarray(REDUCTION_EDGES[edge], dtype=float)
    reference = alpha_quadrature(target_r, target_z, vertices, limit)
    computed = _edge_flux(target_r, target_z, vertices, limit, 48)
    # measured worst over these edges and targets is 1.2e-09, set by the
    # arctangent term, whose four-pole reduction is the deepest of the four
    assert worst_relative(computed, reference) <= 1e-08


def scaled_hexagon(r0, radius):
    """Return a regular hexagon of the given bounding radius, centred at r0."""
    angle = np.pi / 6 + np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    return np.column_stack([r0 + radius * np.cos(angle), radius * np.sin(angle)])


def worst_deviation(vertices, standoff=CONTOUR_STANDOFF):
    """Return the closed form's worst deviation from the oracle, beyond standoff."""
    from nova.biot.polygonanalytic import polygon_analytic_flux

    target_r, target_z = gate_targets(vertices)
    reference = oracle(target_r, target_z, vertices)[0]
    computed = polygon_analytic_flux(target_r, target_z, vertices)
    keep = contour_distance(target_r, target_z, vertices) >= standoff
    assert np.all(np.isfinite(computed)), "closed form returned a non-finite flux"
    return worst_relative(computed[keep], reference[keep])


# Measured worst deviation beyond half a section radius of the contour, for a
# hexagon at r0 = 3 as its bounding radius shrinks. Two of the reduction's small
# quantities -- the complement of the elliptic modulus, and the gap between the
# G^2 pole and the end of the integration range -- both fall as (radius/r0)^2 and
# become CONFLUENT, so the loss goes as the fourth power of the aspect ratio: a
# factor of three off the radius costs two orders of magnitude. The worst target
# is a near one every time, not a far one.
#
# A slender section at large major radius is therefore the hard case for the
# closed form and the easy case for the boundary quadrature, which is the reverse
# of what the cost comparison alone suggests. It is also the case a tokamak coil
# pack presents, so this table -- not the cost -- is what decides where the closed
# form can be used.
ASPECT_ACCURACY = {1.0: 1e-07, 0.3: 1e-05, 0.1: 1e-02}


@pytest.mark.parametrize("radius", sorted(ASPECT_ACCURACY))
def test_the_closed_form_tracks_the_quadrature_over_a_whole_section(radius):
    """The conditioning limit is measured, so it is asserted in both directions.

    An upper bound alone would let a regression pass unnoticed inside a loose
    tolerance. Bounding below as well pins the loss to the aspect ratio it is
    understood to come from, so a reformulation that resolves the confluence FAILS
    this test and has to update the table -- which is the point of recording it.
    """
    tolerance = ASPECT_ACCURACY[radius]
    deviation = worst_deviation(scaled_hexagon(3.0, radius))
    assert deviation <= tolerance, f"radius {radius}: {deviation:.3e}"
    assert deviation > 1e-3 * tolerance, (
        f"radius {radius}: {deviation:.3e} beats the recorded conditioning"
    )


def test_the_closed_form_is_unusable_at_a_thin_section_aspect_ratio():
    """The end of the trend, stated as a fact rather than left to be discovered.

    At a hundredth of the major radius the confluence has consumed every digit
    and the closed form returns a number of the wrong order. The shipped
    quadrature is at its most accurate here, so this is not a gap that needs
    covering -- but it is a gap, and a caller choosing between the two needs it
    written down rather than inferred from the table above.
    """
    assert worst_deviation(scaled_hexagon(3.0, 0.03)) > 1.0


def test_the_closed_form_reproduces_the_rectangle_kernel_for_vertical_edges():
    """The ``b1 = 0`` limit, against the rectangle implementation already trusted.

    Both of the rectangle's contributing edges are exactly vertical, so this is
    the collapsed-quadratic case end to end rather than one edge at a time. The
    tolerance is the rectangle kernel's own 785-point zeta quadrature, not the
    closed form's.
    """
    from nova.biot.polygonanalytic import polygon_analytic_flux

    width, height = 1.0, 0.8
    vertices = rectangle(r0=3.0, width=width, height=height)
    target_r, target_z = gate_targets(vertices, directions=24)
    keep = contour_distance(target_r, target_z, vertices) >= CONTOUR_STANDOFF
    computed = polygon_analytic_flux(target_r, target_z, vertices)
    reference = cylinder_greens(
        target_r[keep], target_z[keep], 3.0, 0.0, width, height
    )[0]
    assert worst_relative(computed[keep], reference) < 2e-3


def test_a_horizontal_edge_contributes_nothing_to_the_closed_form():
    """Paper eq 7a. The reduction divides by the edge's dz to form its slope.

    Splitting a horizontal edge in two adds an edge whose contribution must be
    exactly nothing, so the two evaluations agree to round-off rather than to the
    conditioning tolerance -- both sides are the same closed form.
    """
    from nova.biot.polygonanalytic import polygon_analytic_flux

    vertices = rectangle(r0=3.0, width=1.0, height=0.8)
    r_low, r_high = vertices[0, 0], vertices[1, 0]
    z_low, z_high = vertices[0, 1], vertices[2, 1]
    split = np.array(
        [
            [r_low, z_low],
            [0.5 * (r_low + r_high), z_low],
            [r_high, z_low],
            [r_high, z_high],
            [r_low, z_high],
        ]
    )
    target_r, target_z = gate_targets(vertices)
    keep = converged(target_r, target_z, vertices)
    plain = polygon_analytic_flux(target_r, target_z, vertices)
    divided = polygon_analytic_flux(target_r, target_z, split)
    assert np.all(np.isfinite(divided))
    assert worst_relative(divided[keep], plain[keep]) <= 1e-12


def test_a_target_level_with_an_edge_end_stays_finite():
    """``u = 0`` puts one G^2 root at each end of the integration range.

    The factorisation the reduction runs on does not exist there, so the
    evaluation is floored off zero. That keeps it finite -- which a grid whose
    rows line up with the section's corners needs -- and this test says so
    explicitly, because the ACCURACY at that target is whatever the confluent
    limit allows and is not claimed anywhere.

    A target coinciding with a VERTEX is excluded, not overlooked: there ``u = 0``
    and ``r = r'`` together drive the modulus to 1, where the complete integral of
    the first kind itself diverges, and no amount of care in the reduction
    recovers it. The shipped quadrature is finite there and this is not.
    """
    from nova.biot.polygonanalytic import polygon_analytic_flux

    vertices = scaled_hexagon(3.0, 1.0)
    level = np.unique(vertices[:, 1])
    target_r = np.repeat(np.array([1.8, 2.6, 4.6, 6.0]), level.size)
    target_z = np.tile(level, 4)
    assert np.all(np.isfinite(polygon_analytic_flux(target_r, target_z, vertices)))
