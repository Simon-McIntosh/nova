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
from nova.biot.polygonanalytic import polygon_analytic_flux, polygon_analytic_greens

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


# Every entry is held to the full gate below: every band, all directions, targets
# inside the conductor, four section shapes, and the two structural reductions.
CANDIDATES = {
    "boundary_quadrature": shipped,
    "closed_form": polygon_analytic_greens,
}


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
# It is a CANDIDATE, held to the whole gate above. Getting there took the two
# post-reduction cancellations out rather than the pole treatment, which was
# already right: assembling a section differences each edge's antiderivative over
# its two limits and then over the edges, and the antiderivative is of order the
# squared major radius while the flux is not, so a per-limit value accurate to a
# few hundred ulp arrives at the section as 1e-9. Carrying every numerator in the
# harmonic basis rather than in powers of a range variable put each per-limit value
# back at a few ulp, and removing the residual quadratures' logarithm in closed
# form took the near-contour band and the corner with it. The tables below record
# where that leaves the envelope, in both directions.


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
# The envelope is nearly flat in the aspect ratio now, and what shape is left comes
# from the section sum's own cancellation rather than from the reduction. Three
# effects took it there, in the order they were found, and each is worth the entry
# it earned:
#
#   The pole confluence. Each denominator carries a root a distance of order the
#   squared aspect ratio past one end of the integration range, so the moments
#   there are large and the numerator's value at that end is small. Taking each
#   root in the range variable that vanishes at ITS end makes that value a leading
#   coefficient instead of an alternating sum, which removed a fourth-power trend
#   (5.5e-02 at radius 0.03, 6.8e-01 at 0.01).
#
#   The modulus complement. Giving every complete integral ``k'^2`` from the
#   geometry rather than a float parameter took the slender end down another order
#   and a half -- 4.0e-09 to 2.5e-10 at radius 0.03 -- because a slender section is
#   exactly where the complement is small.
#
#   The numerator basis. The reduced numerators reach degree six and the monomial
#   basis on the unit range is conditioned at four decades by then, so the
#   contraction was forming each per-limit value out of terms that exceeded it by
#   as much. In the harmonic basis the coefficients are bounded by the numerator's
#   own size and nothing cancels: two more orders here, and it is what carries the
#   FAT end, where the targets reach ten major radii and the section sum differences
#   an antiderivative of order the squared major radius down to a flux that is not.
ASPECT_ACCURACY = {1.0: 6e-12, 0.3: 3e-12, 0.1: 3e-12, 0.03: 9e-12, 0.01: 8e-11}


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


# The same envelope over the four gate sections, worst component, recorded in both
# directions so an improvement forces the table to be updated rather than passing
# unnoticed inside a loose bound. Every entry is now inside the 1e-10 gate the
# CANDIDATES sweep applies, which is why the closed form is registered there; the
# margin is 1.7x on the thin plate's far band and one to two orders everywhere else.
#
# The two changes that got there moved different entries. The harmonic numerator
# basis carries the FAR band, where the section sum's cancellation is worst: 48x on
# the hexagon, 33x on the thin plate, 62x on the trapezium. Removing the residual
# quadratures' logarithm in closed form carries the NEAR band, where the targets sit
# on the contour and so at an edge's own end -- and it is the near band, not the far,
# that the rectangle's remaining 3.9e-11 sits in, on its B_R rather than its flux.
SECTION_ACCURACY = {
    "hexagon": (9e-12, 9e-12),
    "rectangle": (3e-12, 6e-11),
    "thin_plate": (9e-11, 9e-12),
    "trapezium": (1e-11, 6e-12),
}


@pytest.mark.parametrize("section", sorted(SECTION_ACCURACY))
def test_the_closed_form_holds_its_recorded_envelope_over_the_gate_sections(section):
    """Per section, per band, worst component, bounded both ways."""

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


# --------------------------------------------------------------------------
# THE CORNER. A limit of an edge is one of the polygon's corners, and most of the
# reduction sees only that corner: ``u`` and the corner's own radius fix the ring
# modulus and its complement, and with them the harmonic moment stack, the ``G^2``
# split and the first of the two residual ``arsinh`` integrals. The edge's SLOPE
# reaches only ``B^2``, the arctangent's numerator and the derivative blocks. Every
# corner of a closed section is the end of two edges, so the corner part is formed
# once and read from both -- which is only sound if the two edges are handed the
# same corner to the last bit.
#
# One term goes further. The ``arsinh beta1`` contribution is a function of the
# corner alone in ALL THREE components, and a section sums each edge as its lower
# limit less its upper, so around a closed chain every corner carries it twice with
# opposite signs and it cancels. It survives only where a horizontal edge is
# dropped, which breaks the chain at that edge's two ends, so it is accumulated
# against the signed number of live edges meeting at each corner. For a hexagonal
# plasma cell that count is zero everywhere and the term -- with the residual
# quadrature that feeds it -- is never formed at all.
#
# Measured on the gate target sets, ONE corner's term is 14 to 1900 times the
# assembled answer, so this is a cancellation of one to three decades rather than a
# small term being neglected; carried per limit instead it leaves 7e-15 of the
# answer behind, three orders inside the recorded envelopes, which is why the tables
# above did not move when it was dropped. Both halves are asserted, because both
# fail silently: that the term really does cancel where it is dropped, and that it
# really is kept where it does not.


def corner_terms(target_r, target_z, vertices):
    """Return the ``arsinh beta1`` contribution at each corner, per component."""
    from nova.biot.polygonanalytic import _NODES, _Vertex

    r = np.abs(np.asarray(target_r, float)).ravel()
    z = np.broadcast_to(target_z, np.shape(target_r)).ravel()
    return [
        _Vertex(r, z, corner[0], corner[1], _NODES, residual=True).arsinh_terms()
        for corner in np.asarray(vertices, float)
    ]


def live_edge_count(vertices):
    """Return the signed number of live edges leaving each corner.

    Edge ``i`` runs from corner ``i`` to corner ``i + 1`` and is summed as its lower
    limit less its upper, so a corner carries ``+1`` for the edge that starts there
    and ``-1`` for the one that ends there.
    """
    from nova.biot.polygon import pack_section

    live = pack_section(vertices)[1] != 0.0
    return live.astype(int) - np.roll(live, 1).astype(int)


def component_scale(target_r, vertices):
    """Return each component's factor on the raw edge sum, from the assembly."""
    from nova.biot.polygon import pack_section

    norm = pack_section(vertices)[2]
    return (
        0.5 * norm * np.abs(target_r),
        norm / (4.0 * np.pi) * np.sign(target_r),
        norm / (4.0 * np.pi) * np.ones_like(target_r),
    )


@pytest.mark.parametrize("section", sorted(SECTIONS))
def test_the_two_edges_at_a_corner_are_handed_the_same_corner(section):
    """Bit-identical endpoints, or the shared corner part is an approximation.

    ``pack_section`` gives each edge its own copy of the two corners it spans, and
    the reduction's corner part -- the modulus and its complement, the moment stack,
    the ring split, the first residual -- is formed once from one of them and read
    by both edges. If the two copies ever disagreed in the last bit nothing else in
    this module would notice: every tolerance here is far above one ulp of a corner
    coordinate, and the term that cancels exactly would stop cancelling.
    """
    from nova.biot.polygon import pack_section

    edges, weights, _ = pack_section(SECTIONS[section])
    for index in range(len(edges)):
        if not (weights[index] and weights[index - 1]):
            continue  # a dropped horizontal edge carries a placeholder, not a corner
        np.testing.assert_array_equal(edges[index][:2], edges[index - 1][2:])


# The three gate sections with no horizontal edge; the rectangle is the broken chain
# and is asserted the other way below.
UNBROKEN = ["hexagon", "thin_plate", "trapezium"]


@pytest.mark.parametrize("section", UNBROKEN)
def test_the_corner_term_cancels_around_an_unbroken_chain(section):
    """Formed per limit it cancels to round-off; per corner it is not formed at all.

    Two assertions, and the second is what licenses the first. Summed the way the
    edge loop would have summed it -- each corner once with either sign -- the term
    leaves only round-off of the assembled answer, so omitting it is an identity and
    not an approximation. And one corner's term ALONE is orders larger than that
    answer, so the identity is a cancellation of one to three decades rather than a
    statement about a term that was negligible anyway.
    """
    vertices = np.asarray(SECTIONS[section], float)
    assert not np.any(live_edge_count(vertices)), "section has a horizontal edge"
    target_r, target_z = gate_targets(vertices)
    terms = corner_terms(target_r, target_z, vertices)
    assembled = polygon_analytic_greens(target_r, target_z, vertices)
    per_limit = [np.zeros_like(np.abs(target_r)) for _ in COMPONENTS]
    for index in range(len(vertices)):
        upper = terms[(index + 1) % len(vertices)]
        for slot, high, low in zip(per_limit, upper, terms[index]):
            slot -= high - low
    for name, factor, left, got, single in zip(
        COMPONENTS, component_scale(target_r, vertices), per_limit, assembled, terms[0]
    ):
        answer = np.max(np.abs(got))
        assert np.max(np.abs(factor * left)) <= 2e-14 * answer, name
        assert np.max(np.abs(factor * single)) > 5.0 * answer, name


def test_the_corner_term_survives_where_a_horizontal_edge_breaks_the_chain():
    """A rectangle drops two edges, so the term is load-bearing at four corners.

    The signed count is the whole bookkeeping, so it is asserted directly, and then
    the magnitude of what it keeps: the surviving sum is 6 per cent of the answer in
    ``B_Z`` and the size of the answer itself in the flux, so a formulation that
    dropped the term wherever it dropped the quadrature would be wrong by that much
    -- and the recorded envelope above, which the rectangle holds at 3e-12, is what
    says the shipped assembly does not.
    """
    vertices = np.asarray(rectangle(), float)
    count = live_edge_count(vertices)
    assert list(count) == [-1, 1, -1, 1]
    target_r, target_z = gate_targets(vertices)
    terms = corner_terms(target_r, target_z, vertices)
    assembled = polygon_analytic_greens(target_r, target_z, vertices)
    kept = [np.zeros_like(np.abs(target_r)) for _ in COMPONENTS]
    for index in range(len(vertices)):
        for slot, term in zip(kept, terms[index]):
            slot += count[index] * term
    for name, factor, keep, got in zip(
        COMPONENTS, component_scale(target_r, vertices), kept, assembled
    ):
        assert np.max(np.abs(factor * keep)) > 1e-2 * np.max(np.abs(got)), name


# What the reorganisation is worth, counted rather than timed, so a regression shows
# up here as well as in the benchmark. Per section: the moment stacks the build
# forms, and the graded residual quadratures.
#
#   section       corners   live edges   moment stacks   residuals   was
#   hexagon           6          6             6             12       12 / 24
#   trapezium         4          4             4              8        8 / 16
#   thin_plate        4          4             4              8        8 / 16
#   rectangle         4          2             4              8        4 /  8
#
# The rectangle is the case that gains NOTHING, and that is the honest shape of the
# win: each of its four corners belongs to exactly one live edge, so there is no
# sharing to exploit, and its chain is broken at every corner, so the first residual
# survives everywhere. A hexagonal plasma cell is the opposite -- every corner shared
# and the chain closed -- and it halves both counts.
BUILD_COUNTS = {
    "hexagon": (6, 12),
    "trapezium": (4, 8),
    "thin_plate": (4, 8),
    "rectangle": (4, 8),
}


@pytest.mark.parametrize("section", sorted(BUILD_COUNTS))
def test_the_corner_work_is_done_once_per_corner(section, monkeypatch):
    """Count the calls, so the sharing cannot quietly stop happening.

    Every accuracy test in this module passes just as well with the corner part
    formed twice -- it is the same arithmetic -- so nothing else here can tell
    whether the reorganisation is in effect.
    """
    import nova.biot.polygonanalytic as reduction

    counts = dict.fromkeys(("moments", "residuals"), 0)

    def counted(key, function):
        def wrapper(*args, **kwargs):
            counts[key] += 1
            return function(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(
        reduction, "harmonic_moments", counted("moments", reduction.harmonic_moments)
    )
    monkeypatch.setattr(
        reduction, "graded_residual", counted("residuals", reduction.graded_residual)
    )
    polygon_analytic_flux(np.array([6.5]), np.array([0.1]), SECTIONS[section])
    moments, residuals = BUILD_COUNTS[section]
    assert counts == {"moments": moments, "residuals": residuals}


def test_a_target_level_with_an_edge_end_is_evaluated_rather_than_approached():
    """``u = 0`` drives BOTH of G^2's roots onto the ends of the range.

    A grid whose rows line up with the section's corners produces this on every
    row, so it is not exotic. Flooring the coordinate away from zero leaves an
    error of about 1e-08; taking each root in the basis that vanishes at its own
    end carries it exactly instead, because each shift goes to zero together with
    the numerator's leading coefficient there, and the whole configuration comes
    out at round-off.

    A target coinciding with a VERTEX adds ``r' = r`` on top, which drives the
    modulus to one as well; that is the harder degeneracy and it has its own
    tests below.
    """

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
# section, per component, against a 128 x 192 rule. The flux column is the closed
# form's own: the quadrature's psi is converged to 1e-12 there. The field column is
# NOT -- the quadrature forms its field by differentiating its antiderivative and at
# a corner that still moves 1.6e-06 between 384 x 384 and 768 x 512 -- so what those
# entries bound is the pair, and the closed form's own field error is somewhere under
# them.
#
#     section        psi          Br          Bz
#     hexagon      3.2e-12     2.7e-05     2.6e-05
#     rectangle    3.5e-13     2.7e-05     2.4e-05
#     thin_plate   3.3e-12     4.7e-05     2.3e-05
#     trapezium    2.1e-13     3.9e-06     2.4e-06
#
# The flux column is the reduction's. A corner makes both residual integrands
# genuinely log-singular at the range end, and no amount of panel grading resolves
# a logarithm. Removing it in closed form holds these entries five orders below the
# residual-quadrature values on three of the four sections at no node cost.
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
# bound with this constant covers that.
#
# The floor is NOT the on-vertex value's accuracy, which is now 3e-12: it is the
# accuracy of the neighbourhood a picometre out, where the offsets that set the
# residual quadratures' panel grading are 1e-13 of a section radius and the panel
# reaches only ``_LAYER_FLOOR``. What is left unresolved there is the width of that
# layer times its own logarithm, and the measured floor tracks it. Approaching a
# corner is therefore less accurate than landing on it, while remaining five orders
# inside the band either way.
VERTEX_MODULUS = 1e3
VERTEX_FLOOR = 1e-7


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
# It stays the better-conditioned of the two over the whole table, by one to two
# orders, which is the reverse of what differentiating the reduced flux would have
# given: the flux's arsinh weight is of order the squared major radius where the
# field's is of order the major radius itself, so the section sum differences a
# smaller quantity down to the same answer.
FIELD_ASPECT_ACCURACY = {1.0: 3e-13, 0.3: 3e-13, 0.1: 2e-12, 0.03: 4e-12, 0.01: 9e-12}


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


# The packed driver is the same reduction with its Python control flow replaced by
# arithmetic, so that a whole tile of pairs -- each against its own section -- goes
# through one call with no branch on a value. Nothing about the ANSWER may change.


def pair_geometry(target_r, target_z, sections):
    """Return the packed geometry of every (target, section) pair, flattened.

    ``pad_batch`` gives a batch of sections one fixed edge count; the pair list then
    repeats each target across the sections and each section across the targets, so
    a single packed call covers the whole outer product.
    """
    from nova.biot.polygon import pad_batch

    edge, weight, norm = pad_batch(sections)
    pairs = np.arange(np.size(target_r) * len(sections))
    rows, columns = np.divmod(pairs, len(sections))
    return (
        np.asarray(target_r, float).ravel()[rows],
        np.asarray(target_z, float).ravel()[rows],
        edge[:, :, columns],
        weight[:, columns],
        norm[columns],
    )


@pytest.mark.parametrize("section", sorted(SECTIONS))
def test_the_packed_driver_reproduces_the_host_driver(section):
    """Same reduction, no shortcuts: the two must agree to round-off.

    The host driver skips a dead edge with ``continue``, forms a corner's term only
    where the chain is broken, and forms a pole family only where some column's root
    is far enough out to need one. None of those decisions can be made from a traced
    value, so the packed driver makes all three with arithmetic instead -- a zero
    weight, a signed live-edge count, and a selection inside the pole route. The
    difference is cost, not arithmetic, and this pins that.
    """
    from nova.biot.polygonanalytic import packed_analytic_greens

    vertices = SECTIONS[section]
    target_r, target_z = gate_targets(vertices)
    host = polygon_analytic_greens(target_r, target_z, vertices)
    packed = packed_analytic_greens(np, *pair_geometry(target_r, target_z, [vertices]))
    for name, got, want in zip(COMPONENTS, packed, host):
        assert worst_relative(np.asarray(got), want) < 1e-11, name


def test_the_packed_driver_takes_a_batch_of_unlike_sections_in_one_call():
    """One call, one shape, a different section per pair -- which is the point.

    A hexagonal plasma cell, a cell clipped by the first wall and a rectangle have
    six, five and four live edges; padded to a common count they go through the same
    call, and each pair's answer must equal what the host driver gives for that
    section alone. The rectangle is the case that exercises the placeholder: its two
    horizontal edges carry no corner of their own, so the driver has to take those
    corners from the edges that END there.
    """
    from nova.biot.polygonanalytic import packed_analytic_greens

    clipped = np.delete(hexagon(), 2, axis=0)
    sections = [hexagon(), clipped, rectangle(), thin_plate()]
    target_r = np.array([R0 + 0.04, R0 - 0.02, R0, 3.02])
    target_z = np.array([0.01, -0.03, 0.05, 0.008])
    packed = [
        np.asarray(component)
        for component in packed_analytic_greens(
            np, *pair_geometry(target_r, target_z, sections)
        )
    ]
    count = len(sections)
    for index, vertices in enumerate(sections):
        host = polygon_analytic_greens(target_r, target_z, vertices)
        for component, name in enumerate(COMPONENTS):
            got = packed[component][index::count][index]
            want = host[component][index]
            assert abs(got - want) <= 1e-11 * max(abs(want), 1e-30), (name, index)
