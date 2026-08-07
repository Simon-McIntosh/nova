"""Whether a planted source is found where it was planted, and the field is the field.

The scan rests entirely on the filament field being right, so that is checked first
and against two things that do not share its algebra.  On the axis the loop field has
an elementary closed form with no elliptic integrals in it at all, and off the axis a
direct numerical Biot-Savart integration around the loop agrees to the accuracy of the
quadrature.  An error in the elliptic form would have to be reproduced by both to
survive.

The scan itself is then tested by planting: put a filament somewhere on the grid, read
what the sensors would see, project off a described response span, and require the
peak to come back at the point it was planted at, carrying the current it was planted
with.  Adding described conductors that partly absorb the source is what makes it a
test of the projection rather than of the arithmetic -- what survives is a smaller and
differently shaped vector, and the peak has to survive with it.

Symmetry gets its own case because the scan is supposed to be evidence for it.  A
source planted above the midplane must come back above the midplane, with its mirror
scoring lower: a scan that imposed the symmetry could not tell the two apart, and then
finding a symmetric answer would mean nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from nova.calibrate.localize import (
    VACUUM_PERMEABILITY,
    LocalizationError,
    axial_projection,
    filament_scan,
    loop_field,
    span_projector,
    surviving_fraction,
)

SENSOR_COUNT = 40
"""Sensors ringing the plane, enough to over-determine a handful of conductors."""


def sensors():
    """Return sensor positions and the axis each one is sensitive to."""

    angle = np.linspace(0.0, 2.0 * np.pi, SENSOR_COUNT, endpoint=False)
    radius, height = 1.45 + 0.35 * np.cos(angle), 0.9 * np.sin(angle)
    axis = np.linspace(0.0, np.pi, SENSOR_COUNT)
    return radius, height, np.cos(axis), np.sin(axis)


def biot_savart_loop(point_r, point_z, radius, height, steps=200_000):
    """Return a loop's field at one point by direct numerical integration.

    The line integral around the winding, with no elliptic integrals anywhere in it,
    which is what makes it an independent check of the closed form.
    """

    phi = np.linspace(0.0, 2.0 * np.pi, steps, endpoint=False)
    step = 2.0 * np.pi / steps
    source = np.stack(
        [radius * np.cos(phi), radius * np.sin(phi), np.full(phi.size, height)]
    )
    element = np.stack(
        [-radius * np.sin(phi) * step, radius * np.cos(phi) * step, np.zeros(phi.size)]
    )
    separation = np.asarray([point_r, 0.0, point_z])[:, None] - source
    distance = np.linalg.norm(separation, axis=0)
    field = np.cross(element, separation, axis=0) / distance**3
    total = VACUUM_PERMEABILITY / (4.0 * np.pi) * field.sum(axis=1)
    return float(total[0]), float(total[2])


@pytest.mark.parametrize("height", [0.0, 0.4, -1.1])
def test_the_field_on_the_axis_matches_its_elementary_closed_form(height):
    radius, offsets = 0.8, np.asarray([0.0, 0.3, -0.75, 2.0])
    radial, axial = loop_field(np.zeros_like(offsets), offsets, radius, height)
    expected = (
        VACUUM_PERMEABILITY
        * radius**2
        / (2.0 * (radius**2 + (offsets - height) ** 2) ** 1.5)
    )
    assert np.allclose(radial, 0.0, atol=1e-18)
    assert np.allclose(axial, expected, rtol=1e-12)


@pytest.mark.parametrize(
    ("point_r", "point_z"), [(1.3, 0.2), (0.4, -0.9), (2.2, 1.4)]
)
def test_the_field_off_the_axis_matches_a_direct_line_integral(point_r, point_z):
    radius, height = 1.0, 0.25
    radial, axial = loop_field(
        np.asarray([point_r]), np.asarray([point_z]), radius, height
    )
    expected_r, expected_z = biot_savart_loop(point_r, point_z, radius, height)
    assert float(radial[0]) == pytest.approx(expected_r, rel=1e-6)
    assert float(axial[0]) == pytest.approx(expected_z, rel=1e-6)


def test_a_loop_of_no_radius_is_refused():
    with pytest.raises(LocalizationError, match="encloses nothing"):
        loop_field(np.asarray([1.0]), np.asarray([0.0]), 0.0, 0.0)


def test_a_sensor_sitting_on_the_winding_returns_a_number_not_an_infinity():
    radial, axial = loop_field(np.asarray([1.0]), np.asarray([0.0]), 1.0, 0.0)
    assert np.isfinite(radial).all() and np.isfinite(axial).all()


def test_what_the_span_leaves_is_orthogonal_to_every_described_conductor():
    generator = np.random.default_rng(5)
    response = generator.normal(size=(SENSOR_COUNT, 6))
    projector = span_projector(response)
    residual = projector.residual(generator.normal(size=SENSOR_COUNT))
    assert np.allclose(response.T @ residual, 0.0, atol=1e-12)


def test_a_vector_inside_the_span_leaves_nothing_behind():
    generator = np.random.default_rng(6)
    response = generator.normal(size=(SENSOR_COUNT, 6))
    inside = response @ generator.normal(size=6)
    residual = span_projector(response).residual(inside)
    assert surviving_fraction(inside, residual) == pytest.approx(0.0, abs=1e-24)


def test_a_response_carrying_holes_has_no_span():
    response = np.zeros((SENSOR_COUNT, 3))
    response[2, 1] = np.nan
    with pytest.raises(LocalizationError, match="non-finite"):
        span_projector(response)


def described_response(radius, height, cosine, sine):
    """Return the sensor response of four conductors the description carries."""

    conductors = ((1.5, -1.1), (1.5, 1.1), (1.65, -0.5), (1.65, 0.5))
    return np.column_stack(
        [
            axial_projection(*loop_field(radius, height, r0, z0), cosine, sine)
            for r0, z0 in conductors
        ]
    )


def test_a_planted_filament_is_found_where_it_was_planted():
    radius, height, cosine, sine = sensors()
    response = described_response(radius, height, cosine, sine)
    projector = span_projector(response)
    grid_r = np.linspace(0.15, 2.4, 46)
    grid_z = np.linspace(-2.4, 2.4, 65)
    planted_r, planted_z, current = float(grid_r[12]), float(grid_z[20]), 0.031
    reading = current * axial_projection(
        *loop_field(radius, height, planted_r, planted_z), cosine, sine
    )
    result = filament_scan(
        projector.residual(reading),
        radius,
        height,
        cosine,
        sine,
        projector=projector,
        radius=grid_r,
        height=grid_z,
    )
    peak = result.peak
    assert peak.radius == pytest.approx(planted_r)
    assert peak.height == pytest.approx(planted_z)
    assert peak.score == pytest.approx(1.0, abs=1e-9)
    assert peak.current == pytest.approx(current, rel=1e-9)


def test_the_scan_finds_the_source_through_conductors_that_partly_absorb_it():
    radius, height, cosine, sine = sensors()
    response = described_response(radius, height, cosine, sine)
    projector = span_projector(response)
    grid_r = np.linspace(0.15, 2.4, 46)
    grid_z = np.linspace(-2.4, 2.4, 65)
    planted_r, planted_z = float(grid_r[30]), float(grid_z[44])
    reading = 0.02 * axial_projection(
        *loop_field(radius, height, planted_r, planted_z), cosine, sine
    ) + response @ np.asarray([1.0e3, -0.7e3, 0.3e3, 0.9e3])
    survived = projector.residual(reading)
    assert surviving_fraction(reading, survived) < 0.5
    peak = filament_scan(
        survived,
        radius,
        height,
        cosine,
        sine,
        projector=projector,
        radius=grid_r,
        height=grid_z,
    ).peak
    assert peak.radius == pytest.approx(planted_r)
    assert peak.height == pytest.approx(planted_z)


def test_the_scan_does_not_impose_up_down_symmetry():
    radius, height, cosine, sine = sensors()
    response = described_response(radius, height, cosine, sine)
    projector = span_projector(response)
    grid_r = np.linspace(0.15, 2.4, 46)
    grid_z = np.linspace(-2.4, 2.4, 65)
    planted_r, planted_z = float(grid_r[20]), float(grid_z[48])
    reading = 0.05 * axial_projection(
        *loop_field(radius, height, planted_r, planted_z), cosine, sine
    )
    result = filament_scan(
        projector.residual(reading),
        radius,
        height,
        cosine,
        sine,
        projector=projector,
        radius=grid_r,
        height=grid_z,
    )
    mirrored = int(np.argmin(np.abs(grid_z + planted_z)))
    assert result.peak.height == pytest.approx(planted_z)
    assert result.score[mirrored, 20] < 0.5 * result.peak.score


def test_a_target_carrying_no_power_cannot_be_explained():
    radius, height, cosine, sine = sensors()
    projector = span_projector(described_response(radius, height, cosine, sine))
    with pytest.raises(LocalizationError, match="carries no power"):
        filament_scan(
            np.zeros(SENSOR_COUNT),
            radius,
            height,
            cosine,
            sine,
            projector=projector,
            radius=np.linspace(0.5, 2.0, 4),
            height=np.linspace(-1.0, 1.0, 4),
        )
