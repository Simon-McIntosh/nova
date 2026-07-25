"""Contract for the three-band polygon-section coupling scheme.

The exact polygon kernel is affordable everywhere (a 2000-cell plasma build
measures at ~7 min on 16 cores) but it spends the same 768 quadrature nodes on a
pair 30 section radii apart as on a pair inside the conductor, where the
integrand is analytic and a far smaller rule is already converged. The scheme
bins each target-source pair by its distance to the section CONTOUR and gives
each band a fixed-shape treatment:

* inside the near limit, the converged rule -- nothing is approximated;
* out to the section's far seam, a reduced rule whose spectral error has fallen
  below the per-component bound;
* beyond it, a centroid filament carrying the section's own moments, a handful of
  Green's-function evaluations rather than a boundary quadrature.

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
    quadrature_nodes,
    section_radius,
    section_skew,
)
from nova.biot.greens import moment_filament
from nova.biot.polygon import polygon_greens

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


@pytest.fixture(scope="module", params=list(SECTIONS))
def sweep(request):
    """Return the measured sweep: bands, the converged reference and the scheme.

    Both section shapes matter and for different reasons: the regular hexagon is
    the shipped plasma cell, and the wall-clipped one is the shape whose surviving
    odd moments push its own far-field seam outwards.
    """
    vertices = SECTIONS[request.param]()
    target_r, target_z = ray_targets(np.geomspace(1.02, 40.0, 26))
    exact = polygon_greens(target_r, target_z, vertices, n_panels=64, n_nodes=96)
    return {
        "label": request.param,
        "vertices": vertices,
        "band": band(target_r, target_z, vertices),
        "exact": dict(zip(COMPONENTS, exact)),
        "scheme": dict(zip(COMPONENTS, banded_greens(target_r, target_z, vertices))),
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


def test_the_near_band_is_the_converged_kernel_untouched(sweep):
    """Inside the near limit nothing is approximated: it is the exact rule itself."""
    vertices = sweep["vertices"]
    target_r, target_z = ray_targets(np.geomspace(1.02, 3.0, 6), count=8)
    inside = band(target_r, target_z, vertices) == 0
    assert inside.any()
    reference = polygon_greens(
        target_r[inside],
        target_z[inside],
        vertices,
        n_panels=NEAR_RULE[0],
        n_nodes=NEAR_RULE[1],
    )
    scheme = banded_greens(target_r, target_z, vertices)
    for got, expected in zip(scheme, reference):
        np.testing.assert_array_equal(got[inside], expected)


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


@pytest.mark.parametrize("label", list(SECTIONS))
@pytest.mark.parametrize("seam,inner,outer", [("near", 0, 1), ("mid", 1, 2)])
def test_a_seam_does_not_step_the_field(label, seam, inner, outer):
    """Both of a seam's models, at the same point on the seam, agree to the bound.

    Evaluating at one point isolates the switch: any difference is the seam jump
    itself rather than the field's own variation across the boundary.
    """
    vertices = SECTIONS[label]()
    offset = NEAR_LIMIT if seam == "near" else mid_limit(vertices)
    rules = {0: NEAR_RULE, 1: MID_RULE}
    peak = {
        name: float(np.max(np.abs(value)))
        for name, value in zip(
            COMPONENTS,
            polygon_greens(*ray_targets(np.geomspace(1.02, 40.0, 26)), vertices),
        )
    }
    for angle in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
        target_r, target_z = at_contour_distance(vertices, offset, angle)
        models = []
        for side in (inner, outer):
            if side in rules:
                models.append(
                    polygon_greens(
                        target_r,
                        target_z,
                        vertices,
                        n_panels=rules[side][0],
                        n_nodes=rules[side][1],
                    )
                )
            else:
                models.append(moment_filament(target_r, target_z, vertices))
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
