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
# answer depends on the section's aspect ratio and on how far the targets reach,
# not on either alone.
#
# It does not join CANDIDATES above. Per edge and per limit the reduction now
# reproduces its own integral to 1e-14, but assembling a section differences that
# integral over each edge's two limits and then over the edges, and the edge
# antiderivative is of order the squared major radius while the flux is not: three
# of the four sections miss the 1e-10 far band on the FLUX by 2x to 12x, purely on
# round-off through that cancellation. BOTH FIELD COMPONENTS now pass the far band on
# every section, worst 3.4e-11. SECTION_ACCURACY below records the distance from the
# gate rather than describing it.


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
    # the arctangent term is the deepest of the four and sets what is measured
    # here; splitting it onto one denominator at a time took the worst over these
    # edges and targets from 1.2e-09 to well inside the bound below
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
# hexagon at r0 = 3 as its bounding radius shrinks. Both tables below are the
# closed form's own envelope, not a gate: the oracle is converged to 1e-12 or
# better on every one of these target sets, checked by raising its rule to
# 128 x 192, so what they record is the closed form.
#
# The envelope is not monotone in the aspect ratio, and the two ends of it fail for
# DIFFERENT reasons. Both are round-off amplified by a cancellation, and neither is
# the pole confluence any more -- that one is cured.
#
#   Slender end. Each denominator carries a root a distance of order the squared
#   aspect ratio past one end of the integration range, so the moments there are
#   large and the numerator's value at that end is small. Taking each root in the
#   basis that vanishes at ITS end makes that value a leading coefficient instead
#   of an alternating sum, which is what removed the old fourth-power trend
#   (5.5e-02 at radius 0.03, 6.8e-01 at 0.01). Giving every complete integral the
#   modulus COMPLEMENT rather than the float parameter took what remained down
#   another order and a half at the slender end -- 4.0e-09 to 2.5e-10 at radius 0.03
#   and 2.0e-07 to 7.1e-09 at 0.01 -- because a slender section is exactly where the
#   complement is small and a float parameter cannot carry it. What is left is the
#   ordinary round-off of a contraction whose terms exceed their sum, and it grows as
#   the section thins because the moments do.
#
#   Fat end. The targets run to thirty section radii, so at radius 1.0 they reach
#   ten major radii, and the edge antiderivative's arsinh weight is of order the
#   squared major radius while the flux the two limits differ by is not. The
#   difference over an edge's two limits and then over the edges cancels four
#   decades at radius 1.0 -- which is why the fat entry is WORSE than the middle of
#   the table rather than better.
#
# Both are amplifications of round-off rather than of a formulation error, so the
# way past them is a reduction that forms the difference between an edge's two
# limits before the large antiderivative is assembled, not a better pole split.
ASPECT_ACCURACY = {1.0: 3e-09, 0.3: 2e-10, 0.1: 8e-11, 0.03: 5e-10, 0.01: 2e-08}


@pytest.mark.parametrize("radius", sorted(ASPECT_ACCURACY))
def test_the_closed_form_tracks_the_quadrature_over_a_whole_section(radius):
    """The conditioning limit is measured, so it is asserted in both directions.

    An upper bound alone would let a regression pass unnoticed inside a loose
    tolerance. Bounding below as well pins the loss to the mechanism it is
    understood to come from, so a reformulation that improves on it FAILS this test
    and has to update the table -- which is the point of recording it.
    """
    tolerance = ASPECT_ACCURACY[radius]
    deviation = worst_deviation(scaled_hexagon(3.0, radius))
    assert deviation <= tolerance, f"radius {radius}: {deviation:.3e}"
    assert deviation > 1e-3 * tolerance, (
        f"radius {radius}: {deviation:.3e} beats the recorded conditioning"
    )


# The same envelope over the four gate sections, worst component, so the distance
# from the 1e-10 gate is recorded rather than described. The closed form does not
# join CANDIDATES on these numbers, but the margin is now small: three of the four
# sections still miss the far band, by 2x on the hexagon, 2.4x on the trapezium and
# 12x on the thin plate, and every one of those misses is the flux rather than the
# field. Giving every complete integral the modulus complement moved all eight
# entries -- most of an order at the far band, two orders at the near one, where the
# targets sit close to the contour and so close to an edge's end.
SECTION_ACCURACY = {
    "hexagon": (3e-10, 4e-10),
    "rectangle": (4e-12, 7e-11),
    "thin_plate": (2e-09, 3e-10),
    "trapezium": (4e-10, 2e-11),
}


@pytest.mark.parametrize("section", sorted(SECTION_ACCURACY))
def test_the_closed_form_holds_its_recorded_envelope_over_the_gate_sections(section):
    """Per section, per band, worst component, bounded both ways."""
    from nova.biot.polygonanalytic import polygon_analytic_greens

    vertices = SECTIONS[section]
    target_r, target_z = gate_targets(vertices)
    distance = contour_distance(target_r, target_z, vertices)
    reference = oracle(target_r, target_z, vertices)
    computed = polygon_analytic_greens(target_r, target_z, vertices)
    for band, mask, tolerance in (
        ("far", distance >= CONTOUR_STANDOFF, SECTION_ACCURACY[section][0]),
        ("near", distance < CONTOUR_STANDOFF, SECTION_ACCURACY[section][1]),
    ):
        worst = max(
            worst_relative(got[mask], want[mask])
            for got, want in zip(computed, reference)
        )
        assert worst <= tolerance, f"{section}/{band}: {worst:.3e}"
        assert worst > 1e-2 * tolerance, (
            f"{section}/{band}: {worst:.3e} beats the recorded envelope"
        )
        # the near-contour band is where the closed form should be strongest,
        # having no quadrature there to converge, and it is: every section holds
        # it three orders inside the band the boundary quadrature is given
        assert worst <= 1e-2 * NEAR_CONTOUR_GATE or band == "far"


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


def test_a_target_level_with_an_edge_end_is_evaluated_rather_than_approached():
    """``u = 0`` drives BOTH of G^2's roots onto the ends of the range.

    A grid whose rows line up with the section's corners produces this on every
    row, so it is not exotic. It used to be floored off zero, which kept the
    evaluation finite at about 1e-08; taking each root in the basis that vanishes
    at its own end carries it exactly instead, because each shift goes to zero
    together with the numerator's leading coefficient there, and the whole
    configuration comes out at round-off.

    A target coinciding with a VERTEX adds ``r' = r`` on top, which drives the
    modulus to one as well; that is the harder degeneracy and it has its own
    tests below.
    """
    from nova.biot.polygonanalytic import polygon_analytic_greens

    vertices = scaled_hexagon(3.0, 1.0)
    level = np.unique(vertices[:, 1])
    target_r = np.repeat(np.array([1.8, 2.6, 4.6, 6.0]), level.size)
    target_z = np.tile(level, 4)
    reference = polygon_greens(target_r, target_z, vertices, n_panels=128, n_nodes=192)
    computed = polygon_analytic_greens(target_r, target_z, vertices)
    for name, got, want in zip(COMPONENTS, computed, reference):
        assert np.all(np.isfinite(got)), name
        assert worst_relative(got, want) <= 1e-11, name


# --------------------------------------------------------------------------
# A target ON A VERTEX. Two edges meet there, and for each of them the vertex is
# one of its two limits, so at that limit u = 0 (the target is level with the edge
# end) AND r' = r AND r1 = r together. Then the elliptic modulus k^2 = 4 r r'/a^2
# reaches ONE, where K diverges, and both of the ring denominator's pole shifts and
# one of the plane denominator's vanish at the same time.
#
# The flux and the field of a finite section are BOUNDED at its own vertex -- the
# section's own corner carries no more singularity than its faces do -- so every
# divergence here has to cancel against a vanishing coefficient. These sections are
# destined for evaluation ACROSS themselves inside the plasma bundle, where a
# target lands on a vertex by accident of grid alignment rather than by contrivance,
# so a formulation that blows up there cannot be used at all.
#
# The band is the near-contour one: a vertex is ON the contour, where the boundary
# quadrature oracle is converged to about 1e-11 and the gate is 1e-03.


def on_the_vertices(vertices):
    """Return targets landing exactly on each of the section's corners."""
    corners = np.asarray(vertices, float)
    return corners[:, 0].copy(), corners[:, 1].copy()


def about_the_vertices(vertices, offset, directions=8):
    """Return targets a fixed distance off each corner, in every direction."""
    corners = np.asarray(vertices, float)
    bearing = np.linspace(0.0, 2.0 * np.pi, directions, endpoint=False)
    return (
        (corners[:, 0][:, None] + offset * np.cos(bearing)).ravel(),
        (corners[:, 1][:, None] + offset * np.sin(bearing)).ravel(),
    )


# Worst deviation from the boundary quadrature with a target on EVERY corner of the
# section, per component, against a 384 x 384 rule. The flux column is the closed
# form's own: the quadrature's psi is converged to 1e-12 there. The field column is
# NOT -- the quadrature forms its field by differentiating its antiderivative and at
# a corner that still moves 1.6e-06 between 384 x 384 and 768 x 512 -- so what those
# entries bound is the pair, and the closed form's own field error is somewhere under
# them.
#
#     section        psi          Br          Bz
#     hexagon      1.9e-07     2.3e-06     2.2e-06
#     rectangle    4.3e-13     2.5e-06     2.0e-06
#     thin_plate   2.2e-07     3.6e-06     1.9e-06
#     trapezium    2.4e-08     3.0e-07     2.1e-07
#
# The flux entries are the arsinh residual quadrature's, not the reduction's: raising
# its node count from the default 64 to 256 takes the hexagon from 1.9e-07 to 8.9e-11
# and the thin plate to 3.8e-10, and at the corner itself the reduction reproduces
# its own edge integral to 1e-10 against a 40-digit quadrature of it, per edge and per
# limit. What limits the section is one panel whose boundary layer collapses onto the
# range end there; ``_LAYER_FLOOR`` records that trade-off.
#
# The rectangle is exact because a vertical edge's whole integrand vanishes at its own
# endpoint -- every term carries the edge slope or ``u``. Approaching such an endpoint
# rather than landing on it is the WORST-conditioned direction the closed form has,
# for the same reason: the arctangent's boundary term is of order ``r^2`` and the
# limit it must produce is of order ``u r``, so the cancellation is one part in
# ``r/u`` and a vertical edge a picometre from its own end keeps a digit or two of
# its own contribution. It does not reach the section, because that contribution is
# proportional to ``u`` and so vanishing: the neighbourhood entries below hold at
# 1e-12 of a section radius.


@pytest.mark.parametrize("section", sorted(SECTIONS))
def test_a_target_on_a_vertex_is_evaluated_rather_than_diverging(section):
    """Every corner of every gate section, all three components.

    The trapezium and the thin plate both join edges of very different slope at
    every corner, and the thin plate joins a steep one to an exactly vertical one,
    whose ``B^2`` is linear rather than quadratic in the range variable -- so if
    the cancellation were per-edge rather than structural these would show it.
    """
    from nova.biot.polygonanalytic import polygon_analytic_greens

    vertices = SECTIONS[section]
    target_r, target_z = on_the_vertices(vertices)
    reference = polygon_greens(target_r, target_z, vertices, n_panels=128, n_nodes=192)
    computed = polygon_analytic_greens(target_r, target_z, vertices)
    for name, got, want in zip(COMPONENTS, computed, reference):
        assert np.all(np.isfinite(got)), name
        assert worst_relative(got, want) <= NEAR_CONTOUR_GATE, name


@pytest.mark.parametrize("offset", [1e-12, 1e-9, 1e-6])
@pytest.mark.parametrize("section", sorted(SECTIONS))
def test_the_neighbourhood_of_a_vertex_is_evaluated_too(section, offset):
    """A hair off the corner, in every direction, at three decades of standoff.

    Patching the exact point would leave the configuration a real target grid
    actually produces -- a corner missed by a rounding error -- still broken, so
    the neighbourhood is gated at the same band as the corner itself. It holds the
    band by three to four orders at every one of these standoffs, and by the same
    margin as the corner itself, which is what says the corner is a limit rather
    than a special case: worst 2.2e-06 anywhere in the table above's neighbourhood.
    """
    from nova.biot.polygonanalytic import polygon_analytic_greens

    vertices = SECTIONS[section]
    target_r, target_z = about_the_vertices(vertices, offset)
    reference = polygon_greens(target_r, target_z, vertices, n_panels=128, n_nodes=192)
    computed = polygon_analytic_greens(target_r, target_z, vertices)
    for name, got, want in zip(COMPONENTS, computed, reference):
        assert np.all(np.isfinite(got)), name
        assert worst_relative(got, want) <= NEAR_CONTOUR_GATE, name


# How fast the evaluation closes on its own on-vertex value as the standoff shrinks,
# and the level it bottoms out at. A section's field is Lipschitz at its corner up to
# a logarithm -- the field's GRADIENT is what carries the corner's log singularity --
# so the modulus is ``s log(1/s)``, and over the ten decades swept below a linear
# bound with this constant covers that. The floor is where the closed form's own
# round-off at a corner sits, from the table above.
VERTEX_MODULUS = 1e3
VERTEX_FLOOR = 3e-9


@pytest.mark.parametrize("section", sorted(SECTIONS))
def test_the_evaluation_is_continuous_through_a_vertex(section):
    """Sweep a target along a line THROUGH each corner and watch it close up.

    A value that is right AT the corner while its neighbourhood is not is a patch,
    not a limit, and no comparison against the boundary quadrature can tell the two
    apart at a corner -- the quadrature's own field is unconverged there. So this
    asks the closed form alone: approached from either side along five bearings
    through every corner, over ten decades of standoff, its value must CONVERGE on
    the value it returns AT the corner, at the rate a bounded field allows. A
    discontinuity shows up as a deviation that stops falling; the measured one falls
    by very nearly a decade per decade all the way down, from 1e-01 at a thousandth
    of a section radius to 5e-10 at 1e-12 of one.
    """
    from nova.biot.polygonanalytic import polygon_analytic_greens

    vertices = np.asarray(SECTIONS[section], float)
    radius = section_radius(vertices)
    standoff = np.logspace(-3, -12, 10)
    for corner in vertices:
        for bearing in np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False):
            # the corner itself first, so each component's own value there is the
            # one the rest of the sweep is held to
            walk = np.concatenate([[0.0], standoff, -standoff]) * radius
            target_r = corner[0] + walk * np.cos(bearing)
            target_z = corner[1] + walk * np.sin(bearing)
            for name, got in zip(
                COMPONENTS, polygon_analytic_greens(target_r, target_z, vertices)
            ):
                assert np.all(np.isfinite(got)), name
                deviation = np.abs(got[1:] - got[0]) / np.max(np.abs(got))
                bound = VERTEX_MODULUS * np.tile(standoff, 2) + VERTEX_FLOOR
                assert np.all(deviation <= bound), (
                    f"{section}/{name}: {np.max(deviation / bound):.3e} of the bound"
                )


# --------------------------------------------------------------------------
# The field. Paper eq 11b gives H_l as its own contour integrand rather than as a
# derivative of the potential, built from the same four transcendentals: D,
# arsinh beta1, arsinh beta2 and arctan beta3, with different polynomial weights.
# So the field costs no new reduction machinery -- only new weights through the
# same integrations by parts, which is why it is worth transcribing rather than
# differentiating the reduced flux (that would need the derivatives of K, E and
# Pi with respect to both modulus and characteristic).
#
# The azimuthal component is the structural check. Eq 11b gives it the SAME
# bracket as the radial one, weighted by sin phi instead of cos phi; the bracket
# is even about phi = pi and sin phi is odd, so it integrates to nothing over the
# full turn. Axisymmetry is reproduced by the algebra rather than assumed, the
# same way cn K = 0 kills the radial vector potential.
#
# Eq 11b is TRANSCRIBED here as well as in the kernel, so a per-edge check against
# it cannot see a mis-transcription -- both sides carry the same one. Maxwell can,
# and did: the last test in this module holds the field to the gradient of the
# flux, and it is what caught the z row's rational term being quadratic in the edge
# slope where it should be linear. That error is invisible on a rectangle (slope
# zero) and on a 45-degree edge (slope one), the only two slopes at which the two
# forms agree.


def alpha_field_quadrature(target_r, target_z, edge, which, component, nodes=300):
    """Return eq 11b's edge field integral by direct quadrature in alpha.

    ``phi = pi - 2a`` maps one full turn onto ``a`` in ``[-pi/2, pi/2]``, and the
    integral is taken over exactly that -- not over the quarter range doubled --
    so an ODD integrand comes back as zero rather than as a quarter-range
    fragment. That is what lets the same reference serve both the two components
    the implementation computes and the azimuthal one it omits. ``h_l`` is the
    bracket of eq 11b transcribed literally; no machinery is shared with the
    reduction under test.

    The two halves are integrated SEPARATELY, at ``nodes`` each. One rule across
    the whole range is not converged, and not by a little: ``a = 0`` is ``phi =
    pi``, where ``sin phi`` changes sign and ``beta3`` runs to infinity, so
    ``arctan beta3`` jumps by ``pi``. The product with ``sin phi`` is continuous
    but has an ``|a|`` kink, and a single Gauss rule sees a discontinuous
    derivative in its interior: at 600 nodes the z component is wrong in the fifth
    decimal. The r component has no arctangent, but ``a = 0`` is also where its
    bracket varies on the scale of the target's offset from the edge end, and a
    rule that reaches the feature only through its interior nodes resolves it far
    worse than one that clusters at it -- 1e-6 against 1e-14 here. Splitting keeps
    the node sets mirror images, so an odd integrand still cancels exactly.
    """
    ra, za, rb, zb = edge
    b1 = (rb - ra) / (zb - za)
    a02 = 1.0 + b1 * b1
    a03 = a02 * np.sqrt(a02)
    node, weight = np.polynomial.legendre.leggauss(nodes)
    r = np.atleast_1d(target_r)[:, None]
    z = np.atleast_1d(target_z)[:, None]
    r1 = ra - b1 * (za - z)
    u = (zb - z) if which else (za - z)
    total = 0.0
    for half in (-1.0, 1.0):
        angle = 0.25 * np.pi * half * (node + 1.0)[None, :]
        cos_phi = -np.cos(2.0 * angle)
        sin_phi = np.sin(2.0 * angle)
        offset = (r1 + b1 * u) - r * cos_phi
        plane_offset = r1 - r * cos_phi
        g_squared = u * u + (r * sin_phi) ** 2
        b_squared = plane_offset**2 + a02 * (r * sin_phi) ** 2
        distance = np.sqrt(g_squared + offset**2)
        first = np.arcsinh(offset / np.sqrt(g_squared))
        second = np.arcsinh((u + b1 * offset) / np.sqrt(b_squared))
        third = np.arctan((u * offset - b1 * g_squared) / (r * sin_phi * distance))
        bracket = (
            distance / a02
            + r * cos_phi * first
            - b1 / a03 * (r1 + b1 * b1 * r * cos_phi) * second
        )
        if component == "r":
            integrand = cos_phi * bracket
        elif component == "phi":
            integrand = sin_phi * bracket
        else:
            integrand = (
                u * first
                + (b1 * b1 * r1 - (2.0 * a02 - 1.0) * r * cos_phi) / a03 * second
                - r * sin_phi * third
                - b1 / a02 * distance
            )
        total = total + integrand @ (0.25 * np.pi * weight)
    return 2.0 * total


FIELD_TARGET_R = np.array([2.6, 3.0, 1.6, 4.0, 0.9])
FIELD_TARGET_Z = np.array([0.4, -0.45, 0.35, 0.2, -0.7])


@pytest.mark.parametrize("component", ["r", "z"])
@pytest.mark.parametrize("edge", sorted(REDUCTION_EDGES))
def test_the_edge_field_reference_is_converged_so_it_can_be_used_as_one(
    edge, component
):
    """Raising the rule past the reference must not move it.

    The same guard the boundary-quadrature oracle carries at the top of this
    module, and for the same reason: without it a candidate can pass by matching
    the reference's error instead of the integral's value -- or, as happened here,
    a correct reduction can be rejected because the reference is the wrong side.
    Both are silent, because a Gauss rule reports no residual.
    """
    vertices = np.asarray(REDUCTION_EDGES[edge], dtype=float)
    for limit in (0, 1):
        coarse = alpha_field_quadrature(
            FIELD_TARGET_R, FIELD_TARGET_Z, vertices, limit, component
        )
        fine = alpha_field_quadrature(
            FIELD_TARGET_R, FIELD_TARGET_Z, vertices, limit, component, nodes=450
        )
        assert worst_relative(coarse, fine) < 1e-12, limit


@pytest.mark.parametrize("component", ["r", "z"])
@pytest.mark.parametrize("limit", [0, 1])
@pytest.mark.parametrize("edge", sorted(REDUCTION_EDGES))
def test_the_closed_form_reproduces_the_edge_field_integral(edge, limit, component):
    """Per edge, per limit, per component: eq 11b's reduction against its integral.

    Same methodology and the same non-degenerate targets as the flux check above,
    for the same reason: this isolates the transcription of the weights and the
    integrations by parts from the conditioning of the reduction they feed.
    """
    from nova.biot.polygonanalytic import _edge_field

    vertices = np.asarray(REDUCTION_EDGES[edge], dtype=float)
    reference = alpha_field_quadrature(
        FIELD_TARGET_R, FIELD_TARGET_Z, vertices, limit, component
    )
    computed = _edge_field(FIELD_TARGET_R, FIELD_TARGET_Z, vertices, limit, 48)[
        0 if component == "r" else 1
    ]
    assert worst_relative(computed, reference) <= 1e-08


@pytest.mark.parametrize("edge", sorted(REDUCTION_EDGES))
def test_the_azimuthal_field_integrand_cancels_over_the_full_turn(edge):
    """Eq 11b's phi component is its r bracket weighted by sin phi, and vanishes.

    Not asserted about the implementation -- which never forms it -- but about the
    quadrature of the paper's own expression, which is what licenses the
    implementation to omit it. If this failed, an axisymmetric ring would carry a
    toroidal field and the reduction would be built on a false premise.
    """
    vertices = np.asarray(REDUCTION_EDGES[edge], dtype=float)
    target_r = np.array([2.6, 3.0, 1.6, 4.0])
    target_z = np.array([0.4, -0.45, 0.35, 0.2])
    for limit in (0, 1):
        azimuthal = alpha_field_quadrature(target_r, target_z, vertices, limit, "phi")
        radial = alpha_field_quadrature(target_r, target_z, vertices, limit, "r")
        assert np.max(np.abs(azimuthal)) <= 1e-12 * np.max(np.abs(radial))


def worst_field_deviation(vertices, standoff=CONTOUR_STANDOFF):
    """Return the closed form's worst field deviation from the oracle, per component.

    Targets where the ORACLE returns a non-finite field are dropped: it forms its
    field by dividing a flux gradient by ``2 pi r`` without guarding the axis, and
    a target set of thirty section radii about ``r0 = 3`` lands exactly on ``r = 0``
    at two of the radii below. That is a defect in ``polygon_greens``, not a
    disagreement about the polygon, and there is no reference to compare against
    there -- which is itself worth knowing, because the closed form IS finite on
    the axis and a separate test pins it.
    """
    from nova.biot.polygonanalytic import polygon_analytic_greens

    target_r, target_z = gate_targets(vertices)
    reference = oracle(target_r, target_z, vertices)
    computed = polygon_analytic_greens(target_r, target_z, vertices)
    keep = contour_distance(target_r, target_z, vertices) >= standoff
    out = {}
    for name, got, want in zip(COMPONENTS, computed, reference):
        assert np.all(np.isfinite(got)), f"{name} returned a non-finite value"
        usable = keep & np.isfinite(want)
        out[name] = worst_relative(got[usable], want[usable])
    return out


# The field's envelope against section aspect ratio, measured the same way as the
# flux's. The confluence is a property of the reduction's pole structure, not of
# which weight sits on top of it, so the field tracks the flux at the slender end
# -- and the point of measuring rather than assuming is that the field's weights
# are one order LOWER in the arctangent term and one HIGHER in the radial D term.
#
# At the fat end it does NOT track, and by three decades: the flux's edge sum
# cancels four decades there because its arsinh weight is of order the squared
# major radius, while the field's is of order the major radius itself. The field is
# the better-conditioned of the two over the whole table, which is the reverse of
# what differentiating the reduced flux would have given.
#
# The slender end moved by nearly two orders when every complete integral was given
# the modulus complement -- 2.0e-09 to 2.4e-11 at radius 0.03 and 3.0e-08 to 3.6e-10
# at 0.01 -- so the field tracks the flux there more closely than it did, both being
# limited by the same contraction round-off rather than by the modulus.
FIELD_ASPECT_ACCURACY = {1.0: 2e-12, 0.3: 2e-12, 0.1: 4e-12, 0.03: 5e-11, 0.01: 8e-10}


@pytest.mark.parametrize("radius", sorted(FIELD_ASPECT_ACCURACY))
def test_the_closed_form_field_tracks_the_quadrature_over_a_whole_section(radius):
    """Both field components, bounded above and below, as for the flux."""
    tolerance = FIELD_ASPECT_ACCURACY[radius]
    deviation = worst_field_deviation(scaled_hexagon(3.0, radius))
    for name in ("Br", "Bz"):
        assert deviation[name] <= tolerance, f"{name} radius {radius}: {deviation}"
    assert max(deviation["Br"], deviation["Bz"]) > 1e-3 * tolerance, (
        f"radius {radius}: {deviation} beats the recorded conditioning"
    )


def test_the_closed_form_field_reproduces_the_rectangle_kernel():
    """``b1 = 0`` for every contributing edge, against the trusted rectangle field.

    The tolerance is the rectangle kernel's own 785-point zeta quadrature, which
    the gate above already records as limiting B_R agreement to about 1e-03.
    """
    from nova.biot.polygonanalytic import polygon_analytic_greens

    width, height = 1.0, 0.8
    vertices = rectangle(r0=3.0, width=width, height=height)
    target_r, target_z = gate_targets(vertices, directions=24)
    keep = contour_distance(target_r, target_z, vertices) >= CONTOUR_STANDOFF
    computed = polygon_analytic_greens(target_r, target_z, vertices)
    reference = cylinder_greens(target_r[keep], target_z[keep], 3.0, 0.0, width, height)
    for name, got, want in zip(COMPONENTS, computed, reference):
        assert worst_relative(got[keep], want) < 2e-3, name


def test_a_horizontal_edge_contributes_nothing_to_the_closed_form_field():
    """The field's own version of paper eq 7a."""
    from nova.biot.polygonanalytic import polygon_analytic_greens

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
    plain = polygon_analytic_greens(target_r, target_z, vertices)
    divided = polygon_analytic_greens(target_r, target_z, split)
    for name, got, want in zip(COMPONENTS, divided, plain):
        assert np.all(np.isfinite(got)), name
        assert worst_relative(got[keep], want[keep]) <= 1e-12, name


def test_the_field_is_clean_on_the_axis():
    """B_R is odd in r and B_Z even, so on the axis B_R must vanish exactly.

    The shipped kernel forms its field by dividing the flux gradient by 2 pi r and
    returns a non-finite value on the axis. This module carries the field's own
    integrand, which needs no such division, so it should be finite there -- and
    the parity it inherits from the reduction pins B_R to zero rather than merely
    small.
    """
    from nova.biot.polygonanalytic import polygon_analytic_greens

    vertices = scaled_hexagon(3.0, 1.0)
    target_z = np.array([-0.8, -0.2, 0.0, 0.3, 0.9])
    axis = np.zeros_like(target_z)
    _, radial, vertical = polygon_analytic_greens(axis, target_z, vertices)
    assert np.all(np.isfinite(radial)) and np.all(np.isfinite(vertical))
    assert np.max(np.abs(radial)) <= 1e-14 * np.max(np.abs(vertical))
    # and the parity either side of the axis, which is what makes that exact
    offset = np.full_like(target_z, 0.4)
    _, left, below = polygon_analytic_greens(-offset, target_z, vertices)
    _, right, above = polygon_analytic_greens(offset, target_z, vertices)
    np.testing.assert_allclose(left, -right, rtol=1e-11)
    np.testing.assert_allclose(below, above, rtol=1e-11)


@pytest.mark.parametrize("section", sorted(SECTIONS))
def test_the_closed_form_field_is_the_gradient_of_its_own_flux(section):
    """``B_R = -(1/2 pi r) dpsi/dz`` and ``B_Z = (1/2 pi r) dpsi/dr``.

    The independent check on eq 11b's TRANSCRIPTION, which the per-edge tests
    above cannot make: they hold the reduction to a quadrature of eq 11b, and both
    sides read eq 11b the same way, so a wrong weight satisfies both. This one
    compares two different rows of the paper against each other through Maxwell,
    and it is what found the z row's rational term.

    Deliberately not against the shipped kernel, which forms its own field by
    differentiating its own flux and so would confirm nothing about eq 11b. The
    stencil is a five-point central difference on the closed form's flux, at
    targets two section radii clear of the contour where the flux is analytic.

    The tolerance is the STENCIL's, measured: the ``h^4`` truncation and the flux
    envelope amplified by ``1/12h`` cross at 1e-7 on the hexagon, the most slender
    of the four sections and so the one whose flux carries the most noise, and at
    1e-8 or better on the other three. Three decades of resolution below that is
    ample for what this test exists to catch -- the term it found was wrong by 170
    per cent on the trapezium and by a factor of six on a steep edge.
    """
    from nova.biot.polygonanalytic import polygon_analytic_flux, polygon_analytic_greens

    vertices = SECTIONS[section]
    radius = section_radius(vertices)
    centre = np.asarray(vertices, float).mean(axis=0)
    bearing = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    scale = np.repeat(np.array([2.0, 4.0, 9.0]) * radius, bearing.size)
    target_r = centre[0] + scale * np.cos(np.tile(bearing, 3))
    target_z = centre[1] + scale * np.sin(np.tile(bearing, 3))

    step = 0.02 * radius

    def derivative(along_r):
        total = 0.0
        # five-point central difference: (-f(2h) + 8f(h) - 8f(-h) + f(-2h))/(12h)
        for shift, coefficient in ((-2, 1.0), (-1, -8.0), (1, 8.0), (2, -1.0)):
            walk = shift * step
            total = total + coefficient * polygon_analytic_flux(
                target_r + (walk if along_r else 0.0),
                target_z + (0.0 if along_r else walk),
                vertices,
            )
        return total / (12.0 * step)

    two_pi_r = 2.0 * np.pi * target_r
    _, radial, vertical = polygon_analytic_greens(target_r, target_z, vertices)
    assert worst_relative(radial, -derivative(along_r=False) / two_pi_r) <= 1e-06
    assert worst_relative(vertical, derivative(along_r=True) / two_pi_r) <= 1e-06
