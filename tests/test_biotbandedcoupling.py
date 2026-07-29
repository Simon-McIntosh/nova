"""Contract for the three-band polygon-section coupling scheme.

The exact polygon kernel is affordable everywhere (a 2000-cell plasma build
measures at ~7 min on 16 cores) but it spends the same 768 quadrature nodes on a
pair 30 section radii apart as on a pair inside the conductor, where the
integrand is analytic and a far smaller rule is already converged. The scheme
bins each target-source pair by its distance to the section CONTOUR and gives
each band a fixed-shape treatment:

* inside the near limit, the exact kernel -- nothing is approximated;
* out to the section's far seam, a reduced rule whose spectral error has fallen
  below the per-component bound;
* beyond it, a centroid filament carrying the section's own moments, a handful of
  Green's-function evaluations rather than a boundary quadrature.

The near band has two exact kernels to choose between -- the converged boundary
quadrature, or the closed form of :mod:`nova.biot.polygonanalytic` -- and every
per-component bound below is measured for BOTH, because the scheme's seams are
pinned against the exact lane and swapping that lane's evaluation must not move
one. What the closed form buys is measured separately: on the contour itself a
boundary quadrature is integrating through its own singularity, so the two
disagree there by parts in a thousand, and refining the quadrature moves it
towards the closed form rather than away.

Both a regular hexagon and a wall-clipped cell are measured throughout, because
the far seam is not the same for the two: a section symmetric about its centroid
has no third moments and its filament is fourth-order accurate, while a clipped
one keeps a third-order residual and needs a wider mid band.

Per component, not on the flux alone: at the reduced rule the flux error and the
vertical-field error differ by more than two orders of magnitude, so a
flux-only bound would pass a rule that has lost two digits of B_Z.

Every bound below is measured against the converged rule over a target sweep
that crosses all three bands, and the seam jumps are measured by evaluating both
of a seam's models at the same point, which isolates the switch from the radial
variation of the field.
"""

from __future__ import annotations

import numpy as np
import pytest

from nova.biot.bandedcoupling import (
    MID_LIMIT,
    MID_RULE,
    NEAR_LIMIT,
    NEAR_RULE,
    SKEW_TOLERANCE,
    SKEWED_MID_LIMIT,
    band,
    banded_greens,
    contour_distance,
    mid_limit,
    near_greens,
    quadrature_nodes,
    section_radius,
    section_skew,
)
from nova.biot.greens import moment_filament
from nova.biot.polygon import polygon_greens
from nova.biot.polygonanalytic import polygon_analytic_greens

R0, Z0 = 6.2, 0.0
CELL_RADIUS = 0.06
COMPONENTS = ("psi", "br", "bz")


def hexagon(r0=R0, z0=Z0, radius=CELL_RADIUS):
    """Return the plasma cell section, a regular hexagon of circumradius ``radius``."""
    angle = np.pi / 6 + np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    return np.column_stack([r0 + radius * np.cos(angle), z0 + radius * np.sin(angle)])


def clipped_cell(r0=R0, z0=Z0, radius=CELL_RADIUS):
    """Return a hexagon with one corner cut by a straight wall.

    Simple and mildly asymmetric, as a plasma cell clipped by the first wall is:
    its third moments do not vanish, where a regular hexagon's do by symmetry.
    """
    corner = list(hexagon(r0, z0, radius))
    return np.array(
        corner[:2]
        + [[r0 - 0.35 * radius, z0 + 0.75 * radius]]
        + [[r0 - 0.95 * radius, z0 + 0.30 * radius]]
        + corner[3:]
    )


def ray_targets(offsets, count=16):
    """Return targets on rings at each CENTROID offset in section radii.

    Sampling every direction rather than one ray matters: neither the reduced
    rule's error nor the quadrupole residual is isotropic, and a band has to hold
    in the worst direction, which for a hexagon is towards a vertex.
    """
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    offset = np.asarray(offsets, dtype=float)[:, None] * CELL_RADIUS
    return (
        (R0 + offset * np.cos(angle)).ravel(),
        (Z0 + offset * np.sin(angle)).ravel(),
    )


SECTIONS = {"hexagon": hexagon, "wall-clipped": clipped_cell}


@pytest.fixture(scope="module", params=[False, True], ids=["quadrature", "closed-form"])
def route(request):
    """Return which exact kernel serves the near band for this run.

    Not a study knob here: the scheme's mid and far bounds are what pin its seams,
    and they are stated against the exact lane, so they have to be measured for
    whichever evaluation that lane uses.
    """
    return request.param


@pytest.fixture(scope="module", params=list(SECTIONS))
def sweep(request, route):
    """Return the measured sweep: bands, the converged reference and the scheme.

    Both section shapes matter and for different reasons: the regular hexagon is
    the shipped plasma cell, and the wall-clipped one is the shape whose surviving
    odd moments push its own far-field seam outwards.
    """
    vertices = SECTIONS[request.param]()
    target_r, target_z = ray_targets(np.geomspace(1.02, 40.0, 26))
    exact = polygon_greens(target_r, target_z, vertices, n_panels=64, n_nodes=96)
    scheme = banded_greens(target_r, target_z, vertices, closed_form=route)
    return {
        "label": request.param,
        "route": route,
        "vertices": vertices,
        "band": band(target_r, target_z, vertices),
        "exact": dict(zip(COMPONENTS, exact)),
        "scheme": dict(zip(COMPONENTS, scheme)),
        "peak": {
            name: float(np.max(np.abs(value))) for name, value in zip(COMPONENTS, exact)
        },
    }


def worst_in_band(sweep, index):
    """Return the worst per-component error in one band, relative to the peak."""
    inside = sweep["band"] == index
    assert inside.any()
    return {
        name: float(
            np.max(np.abs(sweep["scheme"][name][inside] - sweep["exact"][name][inside]))
            / sweep["peak"][name]
        )
        for name in COMPONENTS
    }


# --- the band geometry ------------------------------------------------------


def test_the_distance_is_measured_to_the_contour_not_the_centroid():
    """A vertex direction and a face direction are not the same distance out."""
    vertices = hexagon()
    radius = section_radius(vertices)
    # the flat-top hexagon has a vertex on the +z axis and a face on the +r axis
    apex = np.array([R0]), np.array([Z0 + 2 * radius])
    flat = np.array([R0 + 2 * radius]), np.array([Z0])
    towards_vertex = contour_distance(*apex, vertices)
    towards_face = contour_distance(*flat, vertices)
    assert float(towards_vertex[0]) == pytest.approx(radius, rel=1e-9)
    assert float(towards_face[0]) > float(towards_vertex[0])
    # the centroid itself is one apothem from the nearest face, not zero from it
    centre = contour_distance(np.array([R0]), np.array([Z0]), vertices)
    assert float(centre[0]) == pytest.approx(radius * np.sqrt(3) / 2, rel=1e-9)


@pytest.mark.parametrize("label", list(SECTIONS))
def test_the_bands_are_fixed_by_the_geometry_and_the_measured_limits(label):
    """Assignment is a deterministic function of the contour distance alone."""
    vertices = SECTIONS[label]()
    radius = section_radius(vertices)
    target_r, target_z = ray_targets(np.linspace(0.0, 60.0, 81), count=13)
    assignment = band(target_r, target_z, vertices)
    distance = contour_distance(target_r, target_z, vertices) / radius

    assert set(np.unique(assignment)) == {0, 1, 2}
    expected = (distance >= NEAR_LIMIT).astype(int) + (distance >= mid_limit(vertices))
    np.testing.assert_array_equal(assignment, expected)
    # monotone in contour distance and reproducible call to call
    order = np.argsort(distance)
    assert np.all(np.diff(assignment[order]) >= 0)
    np.testing.assert_array_equal(assignment, band(target_r, target_z, vertices))


def test_the_far_seam_moves_out_for_a_section_whose_skew_survives():
    """A section's own moments set where its far field is good enough.

    A regular hexagon is symmetric about its centroid, so its odd moments vanish
    and the corrected filament is fourth-order accurate from the locked seam. A
    wall-clipped cell keeps a third-order residual whose magnitude is proportional
    to that skew, so its seam sits further out -- geometry-derived either way, not
    one budget applied to both.
    """
    assert section_skew(hexagon()) < SKEW_TOLERANCE
    assert section_skew(clipped_cell()) > SKEW_TOLERANCE
    assert mid_limit(hexagon()) == MID_LIMIT
    assert mid_limit(clipped_cell()) == SKEWED_MID_LIMIT
    assert SKEWED_MID_LIMIT > MID_LIMIT


def test_a_wall_clipped_section_is_banded_without_a_shape_assumption():
    """The scheme reads the section's own polygon, so a clipped cell is ordinary."""
    vertices = clipped_cell()
    target_r, target_z = ray_targets(np.geomspace(1.02, 40.0, 12), count=8)
    assignment = band(target_r, target_z, vertices)
    assert set(np.unique(assignment)) == {0, 1, 2}
    for component in banded_greens(target_r, target_z, vertices):
        assert np.all(np.isfinite(component))


# --- the per-component bounds ------------------------------------------------


def test_the_near_band_is_the_exact_kernel_untouched(sweep):
    """Inside the near limit nothing is approximated: it is the exact kernel itself.

    Bit-identity, not a tolerance, and against whichever of the two exact
    evaluations the run configured -- the near band adds nothing of its own to
    either, which is what makes the scheme's error entirely the mid and far
    models'.
    """
    vertices, route = sweep["vertices"], sweep["route"]
    target_r, target_z = ray_targets(np.geomspace(1.02, 3.0, 6), count=8)
    inside = band(target_r, target_z, vertices) == 0
    assert inside.any()
    reference = (
        polygon_analytic_greens(target_r[inside], target_z[inside], vertices)
        if route
        else polygon_greens(
            target_r[inside],
            target_z[inside],
            vertices,
            n_panels=NEAR_RULE[0],
            n_nodes=NEAR_RULE[1],
        )
    )
    scheme = banded_greens(target_r, target_z, vertices, closed_form=route)
    for got, expected in zip(scheme, reference):
        np.testing.assert_array_equal(got[inside], expected)


@pytest.mark.parametrize("label", list(SECTIONS))
def test_the_two_exact_kernels_agree_where_both_are_converged(label):
    """Swapping the near band's evaluation is a change of route, not of physics.

    Half a contour radius out and beyond, the quadrature's integrand is analytic
    enough for 768 nodes and the two agree at the closed form's own recorded
    envelope, which is what lets either serve the band without moving a seam.
    Measured over the outer three quarters of the near band, worst of the two
    sections and of the three components: 9.5e-12 on the flux, 2.6e-11 on the
    field, both relative to the band's peak.
    """
    vertices = SECTIONS[label]()
    target_r, target_z = ray_targets(np.geomspace(1.02, 2.6, 24), count=16)
    outside = (
        contour_distance(target_r, target_z, vertices) / section_radius(vertices) >= 0.5
    )
    assert outside.any()
    quadrature = near_greens(target_r, target_z, vertices, closed_form=False)
    closed = near_greens(target_r, target_z, vertices, closed_form=True)
    for name, one, other in zip(COMPONENTS, quadrature, closed):
        scale = float(np.max(np.abs(one)))
        assert np.max(np.abs(one[outside] - other[outside])) / scale <= 1e-10, name


@pytest.mark.parametrize("label", list(SECTIONS))
def test_against_the_contour_the_quadrature_loses_four_orders_on_the_field(label):
    """Where the closed form earns the near band, quantified.

    Inside a quarter of a contour radius the quadrature's integrand is within a
    hair of its own singularity, and 768 nodes resolve a near-kink rather than an
    analytic function. Measured, worst component, relative to the band's peak:
    1.9e-04 on B_R and 3.6e-06 (hexagon) / 1.8e-05 (clipped) on B_Z, against
    2.8e-08 on the flux -- so a flux-only comparison hides it by four orders, and
    it is the FIELD that a plasma-plasma diagonal is built from.
    """
    vertices = SECTIONS[label]()
    target_r, target_z = ray_targets(np.geomspace(1.02, 2.6, 24), count=16)
    against = (
        contour_distance(target_r, target_z, vertices) / section_radius(vertices) < 0.25
    )
    assert against.any()
    quadrature = near_greens(target_r, target_z, vertices, closed_form=False)
    closed = near_greens(target_r, target_z, vertices, closed_form=True)
    gap = {}
    for name, one, other in zip(COMPONENTS, quadrature, closed):
        scale = float(np.max(np.abs(one)))
        gap[name] = float(np.max(np.abs(one[against] - other[against]))) / scale
    assert gap["br"] >= 1e-4
    assert gap["bz"] >= 1e-6
    assert gap["psi"] < gap["br"] / 1000.0  # what a flux-only comparison hides


@pytest.mark.parametrize("label", list(SECTIONS))
def test_on_the_contour_the_quadrature_converges_towards_the_closed_form(label):
    """Which of the two disagreeing values is the wrong one.

    Comparing them cannot say -- either could be off -- so it is settled by
    REFINING the quadrature. On the contour its integrand has a genuine kink, so
    it is first-order in the PANEL count there rather than spectral, and measured
    at a fixed 96 nodes it is exactly that: 9.7e-04 of peak at 16 panels, 4.9e-04
    at 32, 2.4e-04 at 64, 6.1e-05 at 256 -- every ratio the panel ratio to three
    figures. The shipped ``(16, 48)`` rule sits at 3.9e-03. It converges TO the
    closed form, so the closed form is the value and the quadrature is the error.
    """
    vertices = SECTIONS[label]()
    # midpoints of the section's own edges: on the contour, and away from a corner
    following = np.roll(vertices, -1, axis=0)
    target_r, target_z = (0.5 * (vertices + following)).T.copy()
    closed = polygon_analytic_greens(target_r, target_z, vertices)
    scale = [float(np.max(np.abs(value))) for value in closed]

    panels = (16, 32, 64, 256)
    gap = []
    for count in panels:
        rule = polygon_greens(target_r, target_z, vertices, n_panels=count, n_nodes=96)
        gap.append(
            max(
                float(np.max(np.abs(one - other))) / peak
                for one, other, peak in zip(rule, closed, scale)
            )
        )
    assert gap[0] > 5e-4
    # first order in the panel count, which is what a kink in the integrand gives
    for before, after, coarse, fine in zip(gap, gap[1:], panels, panels[1:]):
        assert before / after == pytest.approx(fine / coarse, rel=0.2)


def test_the_mid_band_holds_every_component(sweep):
    """The reduced rule is converged past the bound everywhere in its band."""
    for name, value in worst_in_band(sweep, 1).items():
        assert value <= 1e-6, name


def test_the_far_band_holds_every_component(sweep):
    """The corrected filament is inside the bound everywhere in its band."""
    for name, value in worst_in_band(sweep, 2).items():
        assert value <= 1e-6, name


def test_no_component_is_left_unfinite_anywhere(sweep):
    """Including targets inside the conductor and on the contour itself."""
    for name in COMPONENTS:
        assert np.all(np.isfinite(sweep["scheme"][name])), name


# --- the seams ---------------------------------------------------------------


def at_contour_distance(vertices, offset, angle):
    """Return the target on a ray from the centroid at a given contour offset.

    Bisected rather than solved: the contour distance along a ray is monotone but
    piecewise, with a break where the nearest edge changes.
    """
    from nova.biot.greens import section_centroid

    centre = section_centroid(vertices)
    radius = section_radius(vertices)
    direction = np.array([np.cos(angle), np.sin(angle)])
    low, high = 0.0, 4.0 * (offset + 2.0) * radius

    def distance(reach):
        point = centre + reach * direction
        return float(contour_distance(point[:1], point[1:], vertices)[0])

    for _ in range(80):
        middle = 0.5 * (low + high)
        if distance(middle) < offset * radius:
            low = middle
        else:
            high = middle
    point = centre + 0.5 * (low + high) * direction
    return point[:1], point[1:]


@pytest.mark.parametrize(
    "closed_form", [False, True], ids=["quadrature", "closed-form"]
)
@pytest.mark.parametrize("label", list(SECTIONS))
@pytest.mark.parametrize("seam,inner,outer", [("near", 0, 1), ("mid", 1, 2)])
def test_a_seam_does_not_step_the_field(label, seam, inner, outer, closed_form):
    """Both of a seam's models, at the same point on the seam, agree to the bound.

    Evaluating at one point isolates the switch: any difference is the seam jump
    itself rather than the field's own variation across the boundary. Measured for
    both near-band kernels, since replacing one of a seam's two models is exactly
    the way a seam could open up.
    """
    vertices = SECTIONS[label]()
    offset = NEAR_LIMIT if seam == "near" else mid_limit(vertices)
    peak = {
        name: float(np.max(np.abs(value)))
        for name, value in zip(
            COMPONENTS,
            polygon_greens(*ray_targets(np.geomspace(1.02, 40.0, 26)), vertices),
        )
    }

    def model(side, target_r, target_z):
        if side == 0:
            return near_greens(target_r, target_z, vertices, closed_form=closed_form)
        if side == 1:
            return polygon_greens(
                target_r,
                target_z,
                vertices,
                n_panels=MID_RULE[0],
                n_nodes=MID_RULE[1],
            )
        return moment_filament(target_r, target_z, vertices)

    for angle in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
        target_r, target_z = at_contour_distance(vertices, offset, angle)
        models = [model(side, target_r, target_z) for side in (inner, outer)]
        for name, before, after in zip(COMPONENTS, *models):
            jump = float(np.abs(before - after)[0]) / peak[name]
            assert np.isfinite(jump)
            assert jump <= 1e-6, (label, name, seam, angle, jump)


# --- the cost the scheme buys ------------------------------------------------


def hex_lattice(cells=2000, radius=CELL_RADIUS):
    """Return the centres of a hexagonal tiling, as a plasma grid lays them out."""
    pitch = np.sqrt(3.0) * radius
    reach = int(np.ceil(np.sqrt(cells / np.pi)))
    span = np.arange(-reach, reach + 1)
    row, column = np.meshgrid(span, span)
    centre_r = R0 + pitch * (column + 0.5 * (row % 2))
    centre_z = Z0 + pitch * np.sqrt(3.0) / 2.0 * row
    keep = np.hypot(centre_r - R0, centre_z - Z0) <= reach * pitch
    return centre_r[keep], centre_z[keep]


@pytest.mark.parametrize("label,bound", [("hexagon", 0.03), ("wall-clipped", 0.05)])
def test_the_scheme_evaluates_a_small_fraction_of_the_exact_node_count(label, bound):
    """On a plasma-grid-like target cloud the banded node count is a few percent.

    This is the whole point of the scheme: the exact treatment is kept where the
    finite section is physically resolved and the far field, which is almost every
    pair, drops to filament cost. A section whose skew pushes its far seam out
    keeps more pairs on the reduced rule and so costs more, still by a small
    multiple of nothing.
    """
    vertices = SECTIONS[label]()
    target_r, target_z = hex_lattice()
    assert target_r.size > 1500
    assignment = band(target_r, target_z, vertices)
    exact = target_r.size * NEAR_RULE[0] * NEAR_RULE[1]
    assert quadrature_nodes(assignment) / exact <= bound
    # with the closed form on the near band those nodes are gone too, and what is
    # left is the mid band alone -- the near band's cost stops being a node count
    closed = quadrature_nodes(assignment, closed_form=True)
    assert 0 < closed < quadrature_nodes(assignment)
    assert closed == np.count_nonzero(assignment == 1) * MID_RULE[0] * MID_RULE[1]
