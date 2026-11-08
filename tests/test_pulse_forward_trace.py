"""The traced plasma region of a solved map, and the wall the solve samples."""

import numpy as np
import pytest

from nova.equilibrium.wall_mask import inside_polygon
from nova.media.poloidal import boundary_wall_gap
from nova.media.sources.pulse_forward import REFINE, _material_on, trace_slice

WALL = np.array(
    [[0.3, -1.0], [1.5, -1.0], [1.5, 1.0], [0.3, 1.0], [0.3, -1.0]], dtype=float
)
RADIUS = np.linspace(0.2, 1.6, 57)
HEIGHT = np.linspace(-1.1, 1.1, 89)
PITCH = float(np.diff(RADIUS).mean())


def _grid():
    return np.meshgrid(RADIUS, HEIGHT, indexing="ij")


def test_densified_wall_samples_the_column_midplane():
    """A long straight wall run gains samples between its end vertices."""
    from scripts.media.mast_pulse_forward import densify_wall

    column = np.array([[0.195, -1.084], [0.195, 1.084], [1.0, 1.084], [0.195, -1.084]])
    dense = densify_wall(column, 0.01)
    on_column = dense[np.isclose(dense[:, 0], 0.195)]
    assert np.min(np.abs(on_column[:, 1])) <= 0.005
    assert np.max(np.diff(np.sort(on_column[:, 1]))) <= 0.01 + 1e-12
    # The vertex-only wall, the negative control, has no sample near Z = 0.
    assert np.min(np.abs(column[:-1, 1][np.isclose(column[:-1, 0], 0.195)])) > 1.0


def test_limited_boundary_runs_along_the_wall_it_touches():
    """A level that would extend past the wall is cut by it, not floated inside."""
    rr, zz = _grid()
    axis = np.array([0.6, 0.0])
    flux = (rr - axis[0]) ** 2 + zz**2
    material = _material_on(RADIUS, HEIGHT, WALL, REFINE)
    # The boundary level reaches R = 0.25, past the inboard wall at R = 0.3.
    boundary, surfaces = trace_slice(RADIUS, HEIGHT, flux, material, axis, 0.0, 0.35**2)
    assert boundary_wall_gap(boundary, WALL) < PITCH / REFINE
    assert np.min(boundary[:, 0]) >= 0.3 - PITCH / REFINE
    assert len(surfaces) > 0
    # Negative control: the bare contour at the same level leaves the wall.
    angle = np.linspace(0, 2 * np.pi, 200)
    bare = axis + 0.35 * np.column_stack([np.cos(angle), np.sin(angle)])
    assert np.min(bare[:, 0]) < 0.3 - PITCH


def test_diverted_boundary_excludes_the_basin_beyond_the_saddle():
    """The region below the saddle level that the axis cannot reach is left out."""
    rr, zz = _grid()
    width = 0.08
    flux = -np.exp(-((rr - 0.9) ** 2 + (zz - 0.35) ** 2) / width) - np.exp(
        -((rr - 0.9) ** 2 + (zz + 0.55) ** 2) / width
    )
    axis = np.array([0.9, 0.35])
    axis_flux = float(np.min(flux[:, HEIGHT > 0]))
    column = flux[np.argmin(np.abs(RADIUS - 0.9))]
    between = (HEIGHT > -0.55) & (HEIGHT < 0.35)
    saddle_flux = float(np.max(column[between]))
    saddle_z = float(HEIGHT[between][np.argmax(column[between])])
    material = _material_on(RADIUS, HEIGHT, WALL, REFINE)
    boundary, _ = trace_slice(
        RADIUS, HEIGHT, flux, material, axis, axis_flux, saddle_flux
    )
    assert np.min(boundary[:, 1]) > -0.55
    distance = np.hypot(boundary[:, 0] - 0.9, boundary[:, 1] - saddle_z)
    assert np.min(distance) < 2 * PITCH
    lower = inside_polygon(np.array([0.9]), np.array([-0.55]), *boundary.T)[0]
    assert not lower


def test_an_axis_outside_any_sub_boundary_region_is_refused():
    """The trace refuses rather than drawing a boundary around nothing."""
    rr, zz = _grid()
    flux = (rr - 0.6) ** 2 + zz**2
    material = _material_on(RADIUS, HEIGHT, WALL, REFINE)
    with pytest.raises(ValueError, match="sub-boundary region"):
        trace_slice(RADIUS, HEIGHT, flux, material, np.array([1.4, 0.9]), 0.0, 0.01)


def test_thomson_screen_drops_samples_below_twice_their_uncertainty():
    """Only a value at least twice its own sigma is a resolved measurement."""
    from nova.media.sources.mast_thomson import ThomsonString

    positions = np.column_stack([np.array([0.3, 0.6, 0.9, 1.2]), np.zeros(4)])
    string = ThomsonString(
        name="core",
        positions=positions,
        time=np.array([0.1]),
        temperature=np.array([[100.0, 2000.0, 300.0, np.nan]]),
        density=np.ones((1, 4)),
        temperature_error=np.array([[10.0, 16000.0, 150.0, 1.0]]),
        density_error=np.zeros((1, 4)),
    )
    radius, values, sigma = string.resolved(0.1, "temperature")
    assert radius.tolist() == [0.3, 0.9]
    assert values.tolist() == [100.0, 300.0]
    assert sigma.tolist() == [10.0, 150.0]
    bare = ThomsonString(
        name="core",
        positions=positions,
        time=np.array([0.1]),
        temperature=np.ones((1, 4)),
        density=np.ones((1, 4)),
    )
    with pytest.raises(ValueError, match="uncertainty"):
        bare.resolved(0.1, "temperature")


def test_thomson_edge_is_the_fitted_separatrix_temperature_crossing():
    """The measured edge is where the fitted outboard step passes 40 eV."""
    from nova.diagnostics.thomson_edge import (
        SEPARATRIX_TEMPERATURE,
        _profile,
        outboard_edge,
    )

    radius = np.linspace(0.25, 1.47, 140)
    truth = _profile(radius, 1.30, 0.06, 600.0, 5.0, 0.0, 1.0)
    rng = np.random.default_rng(3)
    measured = truth * (1.0 + 0.05 * rng.standard_normal(radius.size))
    sigma = 0.05 * truth + 1.0
    fit = outboard_edge(radius, measured, sigma)
    fine = np.linspace(1.0, 1.47, 20001)
    expected = fine[
        np.argmax(
            _profile(fine, 1.30, 0.06, 600.0, 5.0, 0.0, 1.0) < SEPARATRIX_TEMPERATURE
        )
    ]
    assert fit is not None
    assert abs(fit.edge - expected) < 0.01
    # Negative control: a flat profile has no step and reports no edge.
    flat = np.full(radius.size, 300.0)
    assert (
        outboard_edge(
            radius, flat * (1 + 0.01 * rng.standard_normal(radius.size)), 0.01 * flat
        )
        is None
    )
