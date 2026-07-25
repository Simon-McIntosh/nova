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
