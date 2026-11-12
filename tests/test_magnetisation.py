"""Material-law, geometry and analytic controls for production iron solves."""

from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import PPoly
import shapely

from nova.biot import magnetisation as iron
from nova.biot.greens import greens_bz_br
from nova.biot.tiledassembly import cell_edges
from nova.jax.config import configure_dtypes
from tests.test_imas_iron_read import west as west

configure_dtypes()
assert jax.config.jax_enable_x64


@pytest.fixture(scope="module", autouse=True)
def reuse_identical_interactions():
    """Build each geometry/target operator once while varying the material law."""
    original = iron.cell_interaction
    cache = {}

    def response(polygons, radial, height, *, plan):
        key = (
            tuple(p.tobytes() for p in polygons),
            np.asarray(radial).tobytes(),
            np.asarray(height).tobytes(),
            plan,
        )
        if key not in cache:
            cache[key] = original(polygons, radial, height, plan=plan)
        return cache[key]

    original_average = iron.galerkin_cell_interaction
    averages = {}

    def averaged_response(polygons, *, plan):
        key = (tuple(p.tobytes() for p in polygons), plan)
        if key not in averages:
            averages[key] = original_average(polygons, plan=plan)
        return averages[key]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(iron, "galerkin_cell_interaction", averaged_response)
        patch.setattr(iron, "cell_interaction", response)
        yield


@lru_cache(maxsize=None)
def sphere():
    return (
        shapely.Point(0, 0)
        .buffer(0.5, 256)
        .intersection(shapely.box(0, -0.5, 0.5, 0.5))
    )


@lru_cache(maxsize=None)
def shell():
    return (
        shapely.Point(0, 0)
        .buffer(1, 512)
        .difference(shapely.Point(0, 0).buffer(0.9, 512))
        .intersection(shapely.box(0, -1, 1, 1))
    )


def uniform(points, strength=0.01):
    return np.tile([0.0, strength], (len(points), 1))


def loop(points, current=1.0):
    bz, br = greens_bz_br(points[:, 0], points[:, 1], 0.05, 0.0)
    return current * np.column_stack([br, bz])


@pytest.mark.parametrize("mu", [1.0, 10.0, 100.0, 1000.0])
@pytest.mark.slow
@pytest.mark.timeout(900)
def test_uniform_sphere_matches_prototype(mu, capsys):
    result = iron.solve_magnetisation(
        sphere(), uniform, [0, 10], [mu, mu], cell_size=0.1
    )
    assert result.converged
    assert result.cell_count == 44
    assert result.iterations <= 1
    assert result.magnetisation.dtype == np.float64
    points = result._mesh.centroids
    total = uniform(points) + result.field(points)
    inside = np.linalg.norm(points, axis=1) < 0.3
    exact = 3 * mu / (mu + 2) * 0.01
    error = np.max(np.abs(total[inside, 1] / exact - 1))
    # The polygonised sphere's interior is effectively uniform; the dipole
    # retains the finite boundary-resolution error measured by the ring control.
    probes = np.array([[1.0, 0.0], [0.0, 1.0]])
    measured = (uniform(probes) + result.field(probes))[:, 1]
    fraction = (mu - 1) / (mu + 2)
    expected = 0.01 * np.array([1 - fraction / 8, 1 + fraction / 4])
    recorded = {
        1.0: [0, 0],
        10.0: [9.736922259584446e-7, 1.4861685984079998e-6],
        100.0: [1.299614500682722e-6, 1.8379254441702741e-6],
        1000.0: [1.3400238558780586e-6, 1.8779696518977929e-6],
    }
    exterior_error = np.abs(measured / expected - 1)
    if mu == 1:
        np.testing.assert_array_equal(result.magnetisation, 0)
        np.testing.assert_array_equal(total, uniform(points))
    with capsys.disabled():
        print(
            f"\nsphere mu={mu:g}: cells={result.cell_count}, "
            f"iterations={result.iterations}, interior_exact={exact:.12g} T, "
            f"interior_max={error:.12g}, exterior_exact={expected.tolist()} T, "
            f"exterior_error={exterior_error.tolist()}",
            flush=True,
        )
    assert error < 1e-9
    np.testing.assert_allclose(exterior_error, recorded[mu], rtol=0, atol=2e-10)


def shell_observation(mu, cell_size, capsys):
    started = perf_counter()
    result = iron.solve_magnetisation(
        shell(), loop, [0, 10], [mu, mu], cell_size=cell_size
    )
    assert result.converged
    assert result.iterations <= 1
    probes = np.array([[2.0, 0], [0, 2.0], [3.0, 1.0]])
    air = loop(probes)
    total = air + result.field(probes)
    shielding = np.linalg.norm(air, axis=1) / np.linalg.norm(total, axis=1)
    exact = ((2 * mu + 1) * (mu + 2) - 2 * (mu - 1) ** 2 * 0.9**3) / (9 * mu)
    error = np.abs(shielding / exact - 1)
    recorded = {
        (10, 0.04): [1.486, 1.486, 1.487],
        (10, 0.02): [1.487, 1.487, 1.487],
        (100, 0.04): [6.846, 6.821, 6.861],
        (100, 0.02): [6.872, 6.868, 6.879],
        (1000, 0.04): [59.327, 56.949, 60.219],
        (1000, 0.02): [59.463270578450675, 59.38148554001727, 59.91152091657584],
    }[mu, cell_size]
    wall = perf_counter() - started
    centres = result._mesh.centroids
    maximum_field = np.linalg.norm(loop(centres) + result.field(centres), axis=1).max()
    with capsys.disabled():
        print(
            f"\nshell mu={mu:g}: cell_size={cell_size:g}, "
            f"cells={result.cell_count}, iterations={result.iterations}, "
            f"shielding_exact={exact:.12g}, production={shielding.tolist()}, "
            f"prototype={recorded}, analytic_relative_error={error.tolist()}, "
            f"maximum_field={maximum_field:.12g} T, "
            f"wall_seconds={wall:.6f}",
            flush=True,
        )
    assert np.all(np.isfinite(shielding))
    assert np.all(shielding > 0)
    assert maximum_field < 10, "linear shell must stay inside its two-knot table"
    return error, result.cell_count, wall


@pytest.mark.parametrize("mu", [10, 100, 1000])
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_closed_shell_refines_toward_exact_shielding(mu, capsys):
    started = perf_counter()
    coarse_error, coarse_count, _ = shell_observation(mu, 0.04, capsys)
    fine_error, fine_count, _ = shell_observation(mu, 0.02, capsys)
    order = np.log(coarse_error / fine_error) / np.log(2)
    with capsys.disabled():
        print(
            f"\nshell refinement mu={mu}: 0.04->0.02, "
            f"cells={coarse_count}->{fine_count}, order={order.tolist()}, "
            f"total_wall_seconds={perf_counter() - started:.6f}",
            flush=True,
        )
    assert fine_count > coarse_count
    assert np.all(fine_error < coarse_error), (
        "exact-shell error must fall at every probe"
    )


def test_mesh_merges_slivers_without_losing_material():
    region = shapely.Polygon([(1, 0), (1.93, 0), (1.43, 0.73), (1, 0.73)])
    cells = iron._mesh(region, 0.1)
    merged = shapely.union_all([shapely.Polygon(p) for p in cells.polygons])
    assert region.symmetric_difference(merged).area < 1e-14
    assert cells.areas.min() >= 0.15 * 0.1**2 * (1 - 1e-12)
    assert all(
        shapely.Polygon(p).contains(shapely.Point(c))
        for p, c in zip(cells.polygons, cells.centroids)
    )


def test_clipped_sphere_has_no_collapsed_sheet_edges():
    cells = iron._mesh(sphere(), 0.1)
    edges = cell_edges(cells.polygons)
    lengths = np.hypot(edges.end_r - edges.start_r, edges.end_z - edges.start_z)
    assert np.all(lengths > 0)


@pytest.fixture(scope="module")
def west_iron_region():
    """Read every iron segment at the source manifest's dictionary pin."""
    from nova.imas.machine import StaticMachineDescription

    manifest_path = Path(
        os.environ.get(
            "NOVA_WEST_SOURCE_MANIFEST",
            "/home/ITER/mcintos/Code/imas-efit/tests/data-manifest.json",
        )
    )
    if not manifest_path.is_file():
        pytest.skip("WEST source manifest is not mounted")
    manifest = json.loads(manifest_path.read_text())
    row = next(
        row for row in manifest["machine_descriptions"] if row["key"] == "west-dd4-1-0"
    )
    source = manifest_path.parent.parent / row["target"]
    machine = StaticMachineDescription.read_iron_core(
        f"imas:hdf5?path={source}", dd_version=row["dd_version"]
    )
    assert len(machine.iron_segments) == 6
    return shapely.union_all(
        [
            shapely.Polygon(np.column_stack((segment.outline.r, segment.outline.z)))
            for segment in machine.iron_segments
        ]
    )


@pytest.mark.parametrize("size", [0.08, 0.057, 0.04])
def test_west_mesh_preserves_iron_and_collocation_clearance(
    west_iron_region, size, capsys
):
    cells = iron._mesh(west_iron_region, size)
    polygons = [shapely.Polygon(p) for p in cells.polygons]
    union_error = west_iron_region.symmetric_difference(
        shapely.union_all(polygons)
    ).area
    clearances = np.array(
        [
            shapely.Point(centre).distance(polygon.boundary)
            for polygon, centre in zip(polygons, cells.centroids, strict=True)
        ]
    )
    assert all(
        polygon.contains(shapely.Point(centre))
        for polygon, centre in zip(polygons, cells.centroids, strict=True)
    )
    assert union_error == 0
    assert clearances.min() >= 0.01 * size
    with capsys.disabled():
        print(
            f"\nWEST mesh: cell_size={size:g}, cells={len(polygons)}, "
            f"union_error={union_error:g}, min_clearance={clearances.min():.12g} m",
            flush=True,
        )


def test_mesh_merges_thin_plate_without_losing_material(capsys):
    size = 0.1
    thickness = 0.03 * size
    region = shapely.box(1, 0, 2, thickness)
    cells = iron._mesh(region, size)
    polygons = [shapely.Polygon(p) for p in cells.polygons]
    assert region.symmetric_difference(shapely.union_all(polygons)).area == 0
    clearance = min(
        shapely.Point(centre).distance(polygon.boundary)
        for polygon, centre in zip(polygons, cells.centroids, strict=True)
    )
    assert all(
        polygon.contains(shapely.Point(centre))
        for polygon, centre in zip(polygons, cells.centroids, strict=True)
    )
    assert clearance >= 0.01 * size
    with capsys.disabled():
        print(
            f"\nthin plate: thickness={thickness:g} m, cells={len(polygons)}, "
            f"min_clearance={clearance:.12g} m",
            flush=True,
        )


def test_mesh_preserves_holes_and_refuses_isolated_slivers():
    region = shapely.box(1, 0, 1.2, 0.2).difference(shapely.box(1.04, 0.04, 1.06, 0.06))
    cells = iron._mesh(region, 0.1)
    merged = shapely.union_all([shapely.Polygon(p) for p in cells.polygons])
    assert region.symmetric_difference(merged).area < 1e-14
    with pytest.raises(ValueError, match="sliver"):
        iron._mesh(shapely.box(1, 0, 1.001, 0.001), 0.1)


def nonlinear_law():
    b = np.array([0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0, 6.0])
    return b, 215 + 1400 * b


def test_material_is_monotone_and_preserves_origin():
    b, mu = nonlinear_law()
    law = tuple(jnp.asarray(x) for x in iron._fit_law(b, mu))
    sample = jnp.linspace(1e-8, 8, 2001)

    def h(value):
        return value * (1 / iron._MU0 - iron._coefficient(value, *law))

    derivative = jax.vmap(jax.grad(h))(sample)
    assert np.all(np.isfinite(derivative))
    assert float(derivative.min()) > 0
    np.testing.assert_allclose(
        iron._coefficient(jnp.asarray(b), *law),
        (mu - 1) / (iron._MU0 * mu),
        rtol=2e-15,
    )
    at_zero = iron._coefficient(jnp.asarray(0.0), *law)
    np.testing.assert_allclose(at_zero, (215 - 1) / (iron._MU0 * 215))
    # Positive samples alone do not imply a positive differential material law.
    with pytest.raises(ValueError, match="differential permeability"):
        iron._fit_law([0, 1, 2], [2, 2, 10])


def assert_material_interpolation(b, mu):
    """Check polynomial joins and the differential law used by Newton."""
    knots, coefficients, ends = iron._fit_law(b, mu)
    polynomial = PPoly(coefficients, knots)
    expected = b / (iron._MU0 * mu)
    np.testing.assert_allclose(polynomial(b), expected, rtol=8e-15, atol=0)
    # Independent origin tangent from the tabulated permeability.
    assert coefficients[2, 0] == 1 / (iron._MU0 * mu[0])
    # Evaluate both sides at each exact join, including inserted breakpoints.
    widths = np.diff(knots)
    a, q, slope, value = coefficients
    right_value = ((a * widths + q) * widths + slope) * widths + value
    right_slope = (3 * a * widths + 2 * q) * widths + slope
    np.testing.assert_allclose(right_value[:-1], value[1:], rtol=8e-15, atol=0)
    np.testing.assert_allclose(right_slope[:-1], slope[1:], rtol=8e-14, atol=0)
    np.testing.assert_allclose(right_slope[-1], 1 / iron._MU0, rtol=8e-14)
    np.testing.assert_allclose(polynomial(b[-1]), expected[-1], rtol=8e-15)
    law = tuple(jnp.asarray(x) for x in (knots, coefficients, ends))

    def h(field):
        return field * (1 / iron._MU0 - iron._coefficient(field, *law))

    # Sample every polynomial, even a very short ramp, and the saturated tail.
    sample = np.concatenate(
        [
            np.linspace(start, stop, 201)
            for start, stop in zip(knots[:-1], knots[1:], strict=True)
        ]
        + [np.linspace(b[-1], 2 * b[-1], 201)]
    )
    differential = np.asarray(jax.vmap(jax.grad(h))(jnp.asarray(sample)))
    assert np.isfinite(differential).all()
    assert differential.min() > 0
    # Zero-magnitude branch check: the production branch uses its held origin.
    assert float(jax.grad(h)(0.0)) == pytest.approx(ends[0], rel=2e-14)
    print(
        f"material knots={len(b)}, fitted_knots={len(knots)}, "
        f"origin_slope={coefficients[2, 0]:.12g}, "
        f"min_dH_dB={differential.min():.12g}"
    )
    return knots


@pytest.fixture(scope="module")
def west_material(west):
    """Read the unchanged CEA table from its DD-pinned machine description."""
    from nova.imas.machine import StaticMachineDescription

    source, dd_version, _ = west
    machine = StaticMachineDescription.read_iron_core(source, dd_version=dd_version)
    assert len(machine.iron_segments) == 6
    return machine.iron_segments[0].material


def test_west_cea_material_interpolation(west_material):
    material = west_material
    b, mu = map(np.asarray, (material.b_field, material.permeability_relative))
    assert mu[0] == pytest.approx(215.11019, rel=1e-14)
    assert mu[1] > mu[0]
    knots = assert_material_interpolation(b, mu)
    assert len(knots) > len(b)


def test_west_cea_last_table_interval_matches_reference(west_material):
    """Pin the final table cubic independently of the saturation continuation."""
    knots, coefficients, _ = iron._fit_law(
        west_material.b_field, west_material.permeability_relative
    )
    np.testing.assert_array_equal(knots[(knots >= 5) & (knots <= 6)], [5, 6])
    np.testing.assert_allclose(
        PPoly(coefficients, knots)(np.linspace(5, 6, 5)),
        [
            2342561.5433828533,
            2554288.785523162,
            2774538.441967889,
            2977743.1965653542,
            3138335.7331638793,
        ],
        rtol=1e-14,
    )


def assert_saturated_material(b, mu):
    """Pin the tail-only blend, its bound and its joins to the exact table."""
    b, mu = np.asarray(b, dtype=float), np.asarray(mu, dtype=float)
    knots, coefficients, ends = iron._fit_law(b, mu)
    law = tuple(jnp.asarray(x) for x in (knots, coefficients, ends))
    vacuum = 1 / iron._MU0
    end_slope = vacuum / mu[-1]
    h = b / (iron._MU0 * mu)
    last_magnetisation = (vacuum - end_slope) * b[-1]
    expected_width = 0.02 * (
        iron._MU0 * last_magnetisation if last_magnetisation > 0 else b[-1]
    )
    width = knots[-1] - b[-1]
    expected = (vacuum - end_slope) * (b[-1] + width / 2)
    assert expected <= 1.01 * last_magnetisation
    assert ends[1] <= 1.01 * last_magnetisation
    assert width <= expected_width
    assert width == pytest.approx(expected_width, rel=2e-12)
    sample = jnp.asarray(
        np.r_[np.linspace(b[-1], knots[-1], 257), np.linspace(knots[-1], 100, 257)]
    )
    magnetisation = sample * iron._coefficient(sample, *law)
    offset = np.minimum(np.asarray(sample) - b[-1], width)
    reference = (vacuum - end_slope) * (b[-1] + offset - offset**2 / (2 * width))
    np.testing.assert_allclose(magnetisation, reference, rtol=3e-15, atol=1e-9)
    assert np.min(np.diff(magnetisation)) >= -16 * np.finfo(float).eps * max(
        1, expected
    )
    assert float(magnetisation.max()) <= expected * (1 + 4e-15) + 1e-9
    assert float(magnetisation.max()) <= (
        1.01 * last_magnetisation * (1 + 4e-15) + 1e-9
    )
    np.testing.assert_allclose(magnetisation[257:], expected, rtol=3e-15, atol=1e-9)

    def field_strength(field):
        return field * (1 / iron._MU0 - iron._coefficient(field, *law))

    derivative = jax.vmap(jax.grad(field_strength))
    slopes = np.asarray(derivative(sample))
    assert slopes.min() > 0
    np.testing.assert_allclose(
        slopes, end_slope + (vacuum - end_slope) * offset / width, rtol=2e-12
    )
    for join, slope in [(b[-1], end_slope), (knots[-1], vacuum)]:
        sides = jnp.asarray([np.nextafter(join, 0), join, np.nextafter(join, np.inf)])
        np.testing.assert_allclose(derivative(sides), slope, rtol=2e-10)
    np.testing.assert_allclose(
        jax.vmap(field_strength)(jnp.asarray(b)), h, rtol=2e-12, atol=1e-9
    )
    polynomial = PPoly(coefficients, knots)
    assert polynomial.derivative()(b[-1]) == pytest.approx(end_slope, rel=1e-13)
    assert polynomial.derivative()(knots[-1]) == pytest.approx(vacuum, rel=1e-13)
    rise_fraction = expected / last_magnetisation - 1 if last_magnetisation else 0
    print(
        f"saturation B_table={b[-1]:g} T, blend_width={width:.12g} T, "
        f"M_asymptote={expected:.12g} A/m, "
        f"rise_fraction={rise_fraction:.12g}, "
        f"min_dH_dB={slopes.min():.12g}"
    )


def test_west_cea_material_saturates(west_material):
    assert_saturated_material(
        west_material.b_field, west_material.permeability_relative
    )


@pytest.mark.parametrize(
    "b, mu",
    [([0, 1, 2], [10, 20, 2]), ([0, 1], [2, 1000]), ([0, 10], [1, 1])],
)
def test_synthetic_material_saturates(b, mu):
    assert_saturated_material(b, mu)


@pytest.mark.parametrize("mu", [1.0, 10.0, 1000.0, 3000.0])
def test_two_knot_material_is_linear_through_table(mu):
    knots, coefficients, ends = iron._fit_law([0, 10], [mu, mu])
    polynomial = PPoly(coefficients, knots)
    sample = np.linspace(0, 10, 1001)
    np.testing.assert_allclose(
        polynomial(sample), sample / (iron._MU0 * mu), rtol=1e-14
    )
    np.testing.assert_allclose(
        polynomial.derivative()(sample), 1 / (iron._MU0 * mu), rtol=1e-14
    )
    assert knots[1] == 10, "saturation must add no breakpoint inside the table"


def test_material_refuses_unrepresentable_split_width():
    b = np.array([0.0, 1.0, 1.0 + 4 * np.finfo(float).eps])
    h = 1 / (iron._MU0 * 2)
    mu = [2.0, 2.0, b[-1] / (iron._MU0 * np.nextafter(h, np.inf))]
    with pytest.raises(ValueError, match="split width is not representable"):
        iron._fit_law(b, mu)


def test_material_refuses_unrepresentable_tail_width():
    mu = np.nextafter(1.0, np.inf)
    with pytest.raises(ValueError, match="tail blend width is not representable"):
        iron._fit_law([0.0, 1.0], [mu, mu])


@pytest.mark.parametrize(
    "b, mu",
    [([0, 1], [2, 1000]), ([0, 1, 2], [10, 2, 3.9])],
)
def test_material_accepts_steep_endpoint_tangents(b, mu):
    b, mu = np.asarray(b, dtype=float), np.asarray(mu, dtype=float)
    h = b / (iron._MU0 * mu)
    ratios = (1 / (iron._MU0 * mu[[0, -1]])) / (np.diff(h) / np.diff(b))[[0, -1]]
    assert ratios.max() > 10
    assert_material_interpolation(b, mu)


@pytest.mark.parametrize(
    "b, mu",
    [([1, 2], [2, 2]), ([0, 0], [2, 2]), ([0, 1], [1, np.nan]), ([0, 1], [0.5, 2])],
)
def test_material_refuses_invalid_tables(b, mu):
    with pytest.raises(ValueError, match="B-H table"):
        iron._fit_law(b, mu)


def test_differential_jacobian_is_finite_at_zero():
    law = iron._fit_law([0, 1], [10, 10])
    operands = tuple(
        jnp.asarray(x) for x in (np.eye(4) * iron._MU0 * 0.5, np.zeros(4), *law)
    )
    jacobian = jax.jacfwd(iron._residual)(jnp.zeros(4), operands)
    np.testing.assert_allclose(jacobian, np.eye(4) * 0.55)


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_nonlinear_shell_uses_differential_newton(capsys):
    b, mu = nonlinear_law()
    result = iron.solve_magnetisation(
        shell(),
        lambda p: loop(p, current=2e6),
        b,
        mu,
        cell_size=0.04,
        max_iterations=10,
    )
    history = np.asarray(result.residual_history)
    assert history.size >= 3
    # For quadratic convergence this ratio equals the preceding contraction
    # and tends to zero; a fixed linear contraction instead gives one.
    quadratic_ratio = history[-1] * history[-3] / history[-2] ** 2
    observed_order = np.log(history[-1] / history[-2]) / np.log(
        history[-2] / history[-3]
    )
    centres = result._mesh.centroids
    total = loop(centres, current=2e6) + result.field(centres)
    maximum_field = np.linalg.norm(total, axis=1).max()
    with capsys.disabled():
        print(
            f"\nnonlinear shell: converged={result.converged}, "
            f"iterations={result.iterations}, residual={result.residual_history}, "
            f"quadratic_ratio={quadratic_ratio:.12g}, "
            f"observed_order={observed_order:.12g}, "
            f"maximum_field={maximum_field:.12g} T, "
            f"mu_range={result.permeability_relative.min():.6g}-"
            f"{result.permeability_relative.max():.6g}",
            flush=True,
        )
    assert result.converged
    assert maximum_field < b[-1], "shell must stay inside the unchanged table fit"
    assert result.iterations == 3
    assert result.regularisation_history == (0.0, 0.0, 0.0)
    assert quadratic_ratio < 0.1, "late residual contractions must accelerate"
    assert result.residual_history[-1] < 1e-8
    assert np.all(np.diff(result.residual_history) < 0)


def test_update_operands_and_factor_are_runtime_arguments():
    law = iron._fit_law([0, 1], [10, 10])
    operands = tuple(
        jnp.asarray(x) for x in (np.eye(6) * iron._MU0 * 0.5, np.ones(6) * 0.01, *law)
    )
    state = jnp.zeros(6)
    factor = iron._factor(state, operands)
    lowered = iron._update.lower(state, operands, factor, jnp.asarray(1.0))
    hlo = lowered.compiler_ir(dialect="hlo").as_hlo_text()
    assert "f64[6,6]{1,0} parameter(" in hlo
    assert "f64[6]{0} parameter(" in hlo
    assert "f64[6,6]{1,0} constant(" not in hlo
    result, residual = iron._update(state, operands, factor, jnp.asarray(1.0))
    assert float(jnp.linalg.norm(residual)) < 1e-10
    moved = (operands[0], operands[1] * 2, *operands[2:])
    shifted, _ = iron._update(state, moved, factor, jnp.asarray(1.0))
    np.testing.assert_allclose(shifted, 2 * result)
    damped, _ = iron._update(state, moved, factor, jnp.asarray(0.5))
    np.testing.assert_allclose(damped, result)


def test_regularised_newton_crosses_ill_conditioned_material_stall(monkeypatch, capsys):
    # A cell-volume similarity scaling of a symmetric response makes the
    # differential system ill-conditioned without changing its eigenvalues.
    response = (
        np.array(
            [
                [
                    0.27109626211628385,
                    6.215590257990723,
                    0.605505275409036,
                    -28.29345763919621,
                ],
                [
                    0.16973579554603527,
                    -0.5154273387446999,
                    3.2520303434762816,
                    10.607063515488663,
                ],
                [
                    6.038927604842041e-7,
                    0.00011876963403776738,
                    0.9925358710780197,
                    -0.007598651820290013,
                ],
                [
                    -0.0034226449016746085,
                    0.046987238944372414,
                    -0.9216594329649704,
                    -0.19293680537499014,
                ],
            ]
        )
        * iron._MU0
    )
    external = (
        np.array(
            [
                0.016291980616101504,
                -0.01907815806338487,
                -0.025975777223539,
                0.047051923861125396,
            ]
        )
        .reshape(2, 2)
        .T
    )
    monkeypatch.setattr(iron, "_interaction", lambda *args: response)
    monkeypatch.setattr(iron, "_average_external", lambda cells, callback: external)
    b, mu = nonlinear_law()
    operands = tuple(
        jnp.asarray(x)
        for x in (response, external.T.reshape(-1), *iron._fit_law(b, mu))
    )
    condition = np.linalg.cond(jax.jacfwd(iron._residual)(jnp.zeros(4), operands))
    assert condition > 1e4
    result = iron.solve_magnetisation(
        shapely.box(1, 0, 2, 2),
        lambda points: external,
        b,
        mu,
        cell_size=1,
        max_iterations=40,
    )
    with capsys.disabled():
        print(
            f"\nill-conditioned material: condition={condition:.12g}, "
            f"converged={result.converged}, iterations={result.iterations}, "
            f"residual={result.residual_history}, "
            f"lambda={result.regularisation_history}, ratios={result.reduction_ratios}",
            flush=True,
        )
    assert result.converged
    assert result.residual_history[-1] < 1e-8
    assert any(value > 0 for value in result.regularisation_history)
    assert result.regularisation_history[-1] == 0
    assert all(value > 0.25 for value in result.reduction_ratios)
    assert (
        len(result.regularisation_history)
        == len(result.reduction_ratios)
        == len(result.step_types)
        == result.iterations
    )
    assert all(
        kind == ("newton" if damping == 0 else "regularised")
        for damping, kind in zip(
            result.regularisation_history, result.step_types, strict=True
        )
    )


def test_material_restart_preserves_component_order_and_checks_shape(monkeypatch):
    response = np.eye(4) * iron._MU0 * 0.5
    monkeypatch.setattr(iron, "_interaction", lambda *args: response)
    monkeypatch.setattr(iron, "_average_external", lambda cells, callback: external)
    region = shapely.box(1, 0, 2, 2)
    initial = np.array([[100.0, -200.0], [300.0, -400.0]])
    external = np.array([[0.01, 0.02], [-0.03, 0.04]])
    b, mu = [0, 1, 10], [10, 10, 10]
    result = iron.solve_magnetisation(
        region,
        lambda points: external,
        b,
        mu,
        cell_size=1,
        initial_magnetisation=initial,
        max_iterations=0,
    )
    np.testing.assert_array_equal(result.magnetisation, initial)
    operands = tuple(
        jnp.asarray(x)
        for x in (response, external.T.reshape(-1), *iron._fit_law(b, mu))
    )
    expected = np.linalg.norm(
        iron._residual(jnp.asarray(initial.T.reshape(-1)), operands)
    ) / (np.linalg.norm(external) / iron._MU0)
    assert result.residual_history == pytest.approx((expected,))
    assert not result.converged
    for invalid in (np.zeros((4, 1)), np.full((2, 2), np.nan)):
        with pytest.raises(ValueError, match="initial_magnetisation"):
            iron.solve_magnetisation(
                region,
                lambda points: external,
                b,
                mu,
                cell_size=1,
                initial_magnetisation=invalid,
            )


@pytest.mark.parametrize(
    "region,size",
    [
        (shapely.box(1, 0, 1.7, 0.2), 0.3),
        (
            shapely.Point(0, 0)
            .buffer(0.5, 8)
            .intersection(shapely.box(0, -0.5, 0.5, 0.5)),
            0.2,
        ),
    ],
)
def test_cell_average_interaction_is_reciprocal_and_energy_bounded(region, size):
    cells = iron._mesh(region, size)
    gram = iron._interaction(cells, None, iron._PLAN)
    volume = np.tile(2 * np.pi * cells.areas * cells.centroids[:, 0], 2)
    energy = volume[:, None] * gram
    symmetry = np.linalg.norm(energy - energy.T) / np.linalg.norm(energy)
    assert symmetry < 5e-15, f"weighted reciprocity defect {symmetry}"
    scale = np.sqrt(volume)
    weighted = gram * scale[:, None] / scale[None, :] / iron._MU0
    spectrum = np.linalg.eigvalsh((weighted + weighted.T) / 2)
    assert spectrum[0] >= -2e-7, spectrum
    assert spectrum[-1] <= 1 + 2e-7, spectrum


def test_cell_average_external_uses_axisymmetric_volume():
    cells = iron._mesh(shapely.box(1, 0, 2, 1), 1)
    observed = []

    def field(points):
        observed.append(points.copy())
        return points.copy()

    average = iron._average_external(cells, field)
    np.testing.assert_allclose(average, [[14 / 9, 0.5]], rtol=2e-15)
    assert len(observed[0]) > len(cells.centroids)


def test_mutual_self_block_converges_with_logarithmic_quadrature():
    from nova.biot.tiledassembly import galerkin_cell_interaction

    polygon = np.array([[1.0, 0.0], [1.3, 0.0], [1.3, 0.2], [1.0, 0.2]])
    matrices = [
        galerkin_cell_interaction((polygon,), plan=iron._PLAN, order=order) / iron._MU0
        for order in (8, 16, 24, 32)
    ]
    errors = [np.linalg.norm(matrix - matrices[-1]) for matrix in matrices[:-1]]
    print(
        "self-block quadrature orders 8,16,24,32:",
        [matrix.tolist() for matrix in matrices],
        "errors:",
        errors,
    )
    assert errors[1] < errors[0]
    assert errors[2] < 2e-8
