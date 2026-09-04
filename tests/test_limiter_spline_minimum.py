"""Pins for the tensor-spline minimum restricted to reachable wall pieces."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.equilibrium import connectivity_boundary as cb
    from nova.equilibrium.wall_mask import inside_polygon
    from nova.jax.config import configure_dtypes


_MAST_OPERANDS = (
    Path(__file__).parents[1]
    / "docs/figures/topology-visual-corroboration/mast-topology-operands.npz"
)


def _bowl_case(wall_r=None, wall_z=None):
    configure_dtypes()
    radial = jnp.linspace(0.0, 4.0, 13)
    vertical = jnp.linspace(-1.0, 1.0, 11)
    radius, height = jnp.meshgrid(radial, vertical)
    flux = (radius - 2.25) ** 2 + height**2 + 1.0
    if wall_r is None:
        wall_r = jnp.asarray([0.0, 4.0, 4.0, 0.0])
        wall_z = jnp.asarray([-1.0, -1.0, 1.0, 1.0])
    inside = jnp.ones_like(flux, dtype=bool)
    surface = cb.fit_tensor_spline(radial, vertical, flux)
    axis_r = jnp.asarray(2.25)
    axis_z = jnp.asarray(0.0)
    axis_flux = surface(axis_r, axis_z)
    exact_wall_flux = surface(wall_r, wall_z)
    return {
        "flux": flux,
        "radial": radial,
        "vertical": vertical,
        "inside": inside,
        "wall_r": wall_r,
        "wall_z": wall_z,
        "surface": surface,
        "axis_r": axis_r,
        "axis_z": axis_z,
        "axis_flux": axis_flux,
        "exact_wall_flux": exact_wall_flux,
    }


def _select(case, exact_wall_flux=None, region=None):
    return cb._select_reachable_wall_limiter(
        case["flux"],
        case["radial"],
        case["vertical"],
        case["inside"],
        case["wall_r"],
        case["wall_z"],
        case["exact_wall_flux"] if exact_wall_flux is None else exact_wall_flux,
        case["inside"] if region is None else region,
        case["axis_flux"],
        case["surface"],
        case["axis_r"],
        case["axis_z"],
    )


def test_wall_spline_minimum_matches_dense_segment_reference():
    """An interior derivative root and a corner minimum match dense sampling."""
    case = _bowl_case()
    result = _select(case)
    dense_r = np.linspace(0.0, 4.0, 20_001)
    dense_z = np.full_like(dense_r, -1.0)
    dense_value = np.asarray(case["surface"](dense_r, dense_z))
    dense_index = int(np.argmin(dense_value))
    np.testing.assert_allclose(
        [result["r"], result["z"], result["psi"]],
        [dense_r[dense_index], dense_z[dense_index], dense_value[dense_index]],
        rtol=0.0,
        atol=2.1e-5,
    )
    assert int(result["minimum_bracket_count"]) >= 2

    radial = case["radial"]
    vertical = case["vertical"]
    radius, height = jnp.meshgrid(radial, vertical)
    corner_flux = radius + 2.0 * height
    corner_case = _bowl_case()
    corner_case["flux"] = corner_flux
    corner_case["surface"] = cb.fit_tensor_spline(radial, vertical, corner_flux)
    corner_case["axis_flux"] = corner_case["surface"](2.25, 0.0)
    corner_case["exact_wall_flux"] = corner_case["surface"](
        corner_case["wall_r"], corner_case["wall_z"]
    )
    corner = _select(corner_case)
    np.testing.assert_allclose([corner["r"], corner["z"]], [4.0, 1.0], atol=0.0)


def test_wall_spline_minimum_is_invariant_to_inserted_wall_nodes():
    """Collinear wall vertices do not alter the restricted spline minimum."""
    original = _select(_bowl_case())
    inserted = _select(
        _bowl_case(
            jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 0.0]),
            jnp.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0]),
        )
    )
    np.testing.assert_allclose(
        [inserted["r"], inserted["z"], inserted["psi"]],
        [original["r"], original["z"], original["psi"]],
        rtol=0.0,
        atol=2.0e-12,
    )


def test_wall_spline_minimum_retains_reachable_subsegment_without_vertices():
    """Equal-arc eligibility retains a reachable run between wall vertices."""
    case = _bowl_case()
    region = jnp.zeros_like(case["inside"])
    midpoint_r = 2.25
    midpoint_z = -1.0
    column = int(jnp.argmin(jnp.abs(case["radial"] - midpoint_r)))
    row = int(jnp.argmin(jnp.abs(case["vertical"] - midpoint_z)))
    region = region.at[row, column].set(True)
    result = _select(case, region=region)
    assert not bool(jnp.any(result["reachable"]))
    assert bool(jnp.any(result["reachable_samples"]))
    assert bool(result["valid"])
    assert float(result["r"]) == pytest.approx(2.25, abs=0.02)


def test_wall_spline_minimum_masks_shadowed_brackets():
    """A spline extremum behind a re-entrant fin cannot form a bracket."""
    radial = jnp.linspace(0.2, 1.8, 17)
    vertical = jnp.linspace(-1.0, 1.0, 21)
    wall_r = jnp.asarray([0.2, 1.8, 1.8, 1.1, 1.1, 0.9, 0.9, 0.2])
    wall_z = jnp.asarray([-1.0, -1.0, 1.0, 1.0, 0.0, 0.0, -0.5, -0.5])
    radius, height = jnp.meshgrid(radial, vertical)
    axis_r = jnp.asarray(1.4)
    axis_z = jnp.asarray(0.05)
    flux = -((radius - axis_r) ** 2 + (height - axis_z) ** 2)
    surface = cb.fit_tensor_spline(radial, vertical, flux)
    inside = jnp.ones_like(flux, dtype=bool)
    selected = cb._select_reachable_wall_limiter(
        flux,
        radial,
        vertical,
        inside,
        wall_r,
        wall_z,
        surface(wall_r, wall_z),
        inside,
        surface(axis_r, axis_z),
        surface,
        axis_r,
        axis_z,
    )
    shadowed_node = 5
    assert not bool(selected["reachable"][shadowed_node])
    assert int(selected["node_index"]) != shadowed_node


def test_wall_spline_minimum_has_eager_jit_and_vmap_parity():
    """Inactive knot slots and active roots preserve transformed execution."""
    case = _bowl_case()

    def select(field, exact_wall_flux):
        surface = cb.fit_tensor_spline(case["radial"], case["vertical"], field)
        axis_flux = surface(case["axis_r"], case["axis_z"])
        return cb._select_reachable_wall_limiter(
            field,
            case["radial"],
            case["vertical"],
            case["inside"],
            case["wall_r"],
            case["wall_z"],
            exact_wall_flux,
            case["inside"],
            axis_flux,
            surface,
            case["axis_r"],
            case["axis_z"],
        )

    eager = select(case["flux"], case["exact_wall_flux"])
    compiled = jax.jit(select)(case["flux"], case["exact_wall_flux"])
    shifted_flux = case["flux"] + 3.0
    shifted_wall = case["exact_wall_flux"] + 3.0
    batched = jax.vmap(select)(
        jnp.stack((case["flux"], shifted_flux)),
        jnp.stack((case["exact_wall_flux"], shifted_wall)),
    )
    np.testing.assert_allclose(
        [compiled["r"], compiled["z"], compiled["psi"]],
        [eager["r"], eager["z"], eager["psi"]],
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(batched["r"], [eager["r"], eager["r"]], atol=2e-12)
    np.testing.assert_allclose(
        batched["psi"], [eager["psi"], eager["psi"] + 3.0], atol=2e-12
    )


def test_exact_wall_flux_is_an_independent_residual():
    """The exact wall vector changes its residual but not the spline minimum."""
    case = _bowl_case()
    baseline = _select(case)
    perturbed = _select(case, exact_wall_flux=case["exact_wall_flux"] + 7.0)
    np.testing.assert_allclose(
        [perturbed["r"], perturbed["z"], perturbed["psi"]],
        [baseline["r"], baseline["z"], baseline["psi"]],
        rtol=0.0,
        atol=0.0,
    )
    assert float(baseline["wall_flux_residual_max"]) == pytest.approx(0.0, abs=1e-12)
    assert float(perturbed["wall_flux_residual_max"]) == pytest.approx(7.0)


def test_limiter_class_margin_and_axis_use_one_spline_map():
    """Axis, hard wall binding, and class wall operand share one spline."""
    case = _bowl_case()
    ingredients = cb._read_ingredients(
        case["flux"],
        case["radial"],
        case["vertical"],
        case["inside"],
        case["axis_r"],
        case["axis_z"],
        16,
        8,
        case["wall_r"],
        case["wall_z"],
        case["exact_wall_flux"],
        jnp.asarray(jnp.nan),
        True,
    )
    direct_axis = case["surface"](case["axis_r"], case["axis_z"])
    bilinear_axis = cb._bilerp(
        case["flux"],
        case["radial"],
        case["vertical"],
        case["axis_r"],
        case["axis_z"],
    )
    assert float(ingredients["psi_axis"]) == pytest.approx(float(direct_axis))
    assert float(ingredients["psi_axis"]) != pytest.approx(
        float(bilinear_axis), abs=1e-6
    )
    assert float(ingredients["u_wall_c"]) == pytest.approx(
        float(ingredients["class_u_wall"]), abs=0.0
    )
    expected_wall = (
        ingredients["class_wall"]["psi"] - ingredients["psi_axis"]
    ) / ingredients["span_safe"]
    assert float(ingredients["class_u_wall"]) == pytest.approx(float(expected_wall))


def test_diverted_mast_row_polishes_only_its_census_wall_candidate():
    """A shadowed census wall point cannot discover a different limiter contact."""
    configure_dtypes()
    with np.load(_MAST_OPERANDS, allow_pickle=False) as stored:
        coordinate = stored["row_10_cell_rz"]
        field = stored["row_10_per_cell_flux_values"]
        axis = stored["row_10_selected_o"][0]
        saddle = stored["row_10_selected_x"][0]
        wall_point = stored["row_10_wall_point"][0]
        wall = stored["row_10_wall"]

    radial = np.unique(coordinate[:, 0])
    vertical = np.unique(coordinate[:, 1])
    field = field.reshape((radial.size, vertical.size)).T
    radial_grid, vertical_grid = np.meshgrid(radial, vertical)
    inside = inside_polygon(
        radial_grid.ravel(), vertical_grid.ravel(), wall[:, 0], wall[:, 1]
    ).reshape(field.shape)
    surface = cb.fit_tensor_spline(
        jnp.asarray(radial), jnp.asarray(vertical), jnp.asarray(field)
    )
    classification_x = jnp.full((30, 4), jnp.nan, dtype=jnp.float64)
    classification_x = classification_x.at[0].set(
        jnp.r_[saddle, surface(saddle[0], saddle[1]), 0.0]
    )
    classification_wall = jnp.r_[wall_point, surface(wall_point[0], wall_point[1])]
    result = cb._read_ingredients(
        jnp.asarray(field),
        jnp.asarray(radial),
        jnp.asarray(vertical),
        jnp.asarray(inside),
        jnp.asarray(axis[0]),
        jnp.asarray(axis[1]),
        96,
        18,
        jnp.asarray(wall[:, 0]),
        jnp.asarray(wall[:, 1]),
        surface(wall[:, 0], wall[:, 1]),
        jnp.asarray(jnp.nan),
        True,
        classification_x,
        classification_wall,
    )

    assert bool(result["class_wall_shadowed"])
    assert not bool(result["class_wall"]["valid"])
    assert np.isposinf(float(result["class_u_wall"]))
    assert np.isfinite(float(result["class_u_x"]))
    assert np.isposinf(float(result["class_u_wall"] - result["class_u_x"]))
