"""Accuracy and geometry contract for finite-arc banding."""

import numpy as np

from nova.biot.arcbandedcoupling import (
    ARC_FAR_LIMIT,
    arc_band,
    arc_contour_distance,
    arc_far_limit,
    arc_filament_greens,
    arc_moment_filament,
    banded_arc_greens,
    rms_radius,
)
from nova.biot.greens import greens_bz_br, greens_psi, second_moments, section_centroid
from nova.biot.polygonarc import polygon_arc_greens

RADIUS = 6.2
SECTION_RADIUS = 0.06
START, END = 0.4, 2.1


def hexagon():
    """Return a representative symmetric section."""
    angle = np.pi / 6.0 + np.arange(6) * np.pi / 3.0
    return np.column_stack(
        [
            RADIUS + SECTION_RADIUS * np.cos(angle),
            SECTION_RADIUS * np.sin(angle),
        ]
    )


def thin_plate():
    """Return the elongated parallelogram from the finite-arc accuracy gate."""
    return np.array(
        [
            [RADIUS - 0.2, 0.0],
            [RADIUS + 0.2, 0.03],
            [RADIUS + 0.2, 0.0375],
            [RADIUS - 0.2, 0.0075],
        ]
    )


def test_a_full_arc_filament_reduces_to_the_ring():
    target_r = np.array([4.0, 5.0, 7.0])
    target_z = np.array([0.2, -0.4, 1.0])
    rows = arc_filament_greens(
        target_r,
        target_z,
        np.zeros(3),
        RADIUS,
        0.1,
        0.0,
        2.0 * np.pi,
    )
    psi = greens_psi(target_r, target_z, RADIUS, 0.1)
    bz, br = greens_bz_br(target_r, target_z, RADIUS, 0.1)
    np.testing.assert_allclose(2.0 * np.pi * target_r * rows[1], psi, rtol=2e-12)
    np.testing.assert_allclose(rows[2], br, rtol=2e-12)
    np.testing.assert_allclose(rows[4], bz, rtol=2e-12)
    np.testing.assert_allclose(rows[0], 0.0, atol=2e-22)
    np.testing.assert_allclose(rows[3], 0.0, atol=2e-22)


def test_the_rms_radius_carries_the_radial_second_moment():
    vertices = hexagon()
    centre = section_centroid(vertices)
    radial_moment, _, _ = second_moments(vertices)
    assert np.isclose(rms_radius(vertices) ** 2, centre[0] ** 2 + radial_moment)


def test_the_distance_carries_the_chord_beyond_an_arc_end():
    vertices = hexagon()
    delta = 0.03
    distance = arc_contour_distance(
        np.array([RADIUS]),
        np.array([0.0]),
        np.array([START - delta]),
        vertices,
        START,
        END,
    )
    expected = 2.0 * RADIUS * np.sin(delta / 2.0)
    np.testing.assert_allclose(distance, expected, rtol=2e-13)

    within = arc_contour_distance(
        np.array([RADIUS + 2.0 * SECTION_RADIUS]),
        np.array([0.0]),
        np.array([0.5 * (START + END)]),
        vertices,
        START,
        END,
    )
    radial_extent = SECTION_RADIUS * np.cos(np.pi / 6.0)
    np.testing.assert_allclose(within, 2.0 * SECTION_RADIUS - radial_extent, rtol=2e-13)


def target_routes(levels):
    """Return nonsymmetric target rays within the sweep and beyond both ends."""
    levels = np.asarray(levels)
    return (
        (
            RADIUS + levels * SECTION_RADIUS,
            np.full_like(levels, 0.35 * SECTION_RADIUS),
            np.full_like(levels, 0.5 * (START + END) + 0.17),
        ),
        (
            np.full_like(levels, RADIUS + 0.27 * SECTION_RADIUS),
            np.full_like(levels, 0.31 * SECTION_RADIUS),
            START - levels * SECTION_RADIUS / RADIUS,
        ),
        (
            np.full_like(levels, RADIUS - 0.23 * SECTION_RADIUS),
            np.full_like(levels, -0.29 * SECTION_RADIUS),
            END + levels * SECTION_RADIUS / RADIUS,
        ),
    )


def relative_envelope(got, exact):
    """Return each pair's worst error, scaled by each row's peak."""
    scale = np.max(np.abs(exact), axis=1)[:, None]
    return np.max(np.abs(got - exact) / scale, axis=0)


def test_the_centroid_moment_filament_holds_through_the_off_end_seam():
    vertices = hexagon()
    levels = np.array([32.0, 40.0, 48.0])
    for target_r, target_z, target_phi in target_routes(levels):
        exact = np.stack(
            polygon_arc_greens(target_r, target_z, target_phi, vertices, START, END)
        )
        centroid = np.stack(
            arc_moment_filament(
                target_r,
                target_z,
                target_phi,
                vertices,
                START,
                END,
                placement="centroid",
            )
        )
        rms_corrected = np.stack(
            arc_moment_filament(
                target_r,
                target_z,
                target_phi,
                vertices,
                START,
                END,
                placement="rms",
            )
        )
        rms_bare = np.stack(
            arc_moment_filament(
                target_r,
                target_z,
                target_phi,
                vertices,
                START,
                END,
                placement="rms",
                corrected=False,
            )
        )
        centroid_error = relative_envelope(centroid, exact)
        rms_error = relative_envelope(rms_corrected, exact)
        bare_error = relative_envelope(rms_bare, exact)
        assert np.max(centroid_error) < 1.0e-6
        assert np.max(rms_error) < 1.0e-6
        assert np.max(bare_error) > 50.0 * np.max(centroid_error)


def test_an_elongated_section_widens_its_own_far_seam():
    vertices = thin_plate()
    far_limit = arc_far_limit(vertices)
    assert far_limit > ARC_FAR_LIMIT
    centre = section_centroid(vertices)
    offset = vertices - centre
    radius = float(np.max(np.hypot(offset[:, 0], offset[:, 1])))
    levels = np.geomspace(4.0, 1.4 * far_limit, 18)
    target_r = centre[0] + levels * radius
    target_z = np.full_like(levels, centre[1] + 0.35 * radius)
    target_phi = np.full_like(levels, 0.5 * (START + END) + 0.17)
    exact = np.stack(
        polygon_arc_greens(target_r, target_z, target_phi, vertices, START, END)
    )
    got = np.stack(
        arc_moment_filament(target_r, target_z, target_phi, vertices, START, END)
    )
    envelope = relative_envelope(got, exact)
    assert np.max(envelope[levels >= far_limit]) < 1.0e-6


def test_the_banded_route_is_exact_inside_and_moment_corrected_outside():
    vertices = hexagon()
    levels = np.array([2.0, 36.0, 48.0])
    target_r, target_z, target_phi = target_routes(levels)[1]
    assignment = arc_band(target_r, target_z, target_phi, vertices, START, END)
    np.testing.assert_array_equal(assignment, [0, 1, 1])
    exact = np.stack(
        polygon_arc_greens(target_r, target_z, target_phi, vertices, START, END)
    )
    got = np.stack(
        banded_arc_greens(target_r, target_z, target_phi, vertices, START, END)
    )
    np.testing.assert_allclose(got[:, 0], exact[:, 0], rtol=3e-11, atol=2e-19)
    assert np.max(relative_envelope(got[:, 1:], exact[:, 1:])) < 1.0e-6
