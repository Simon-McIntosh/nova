"""Material-law, geometry and analytic controls for production iron solves."""

from __future__ import annotations

from functools import lru_cache
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import shapely

from nova.biot import magnetisation as iron
from nova.biot.greens import greens_bz_br
from nova.biot.tiledassembly import cell_edges
from nova.jax.config import configure_dtypes

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

    with pytest.MonkeyPatch.context() as patch:
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
        (100, 0.04): [6.846, 6.821, 6.861],
        (100, 0.02): [6.872, 6.868, 6.879],
        (100, 0.01): [6.893, 6.887, 6.894],
        (1000, 0.04): [59.327, 56.949, 60.219],
        (1000, 0.02): [59.463270578450675, 59.38148554001727, 59.91152091657584],
        (1000, 0.01): [60.42517459703494, 60.255699021490265, 60.530212046430016],
    }[mu, cell_size]
    wall = perf_counter() - started
    with capsys.disabled():
        print(
            f"\nshell mu={mu:g}: cell_size={cell_size:g}, "
            f"cells={result.cell_count}, iterations={result.iterations}, "
            f"shielding_exact={exact:.12g}, production={shielding.tolist()}, "
            f"prototype={recorded}, analytic_relative_error={error.tolist()}, "
            f"wall_seconds={wall:.6f}",
            flush=True,
        )
    assert np.all(np.isfinite(shielding))
    assert np.all(shielding > 0)
    return error, result.cell_count, wall


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_closed_shell_low_permeability(capsys):
    _, count, _ = shell_observation(10, 0.04, capsys)
    assert count == 228


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_closed_shell_refines_toward_exact_shielding(capsys):
    started = perf_counter()
    refine_further = False
    for mu in (100, 1000):
        previous_error, previous_count, _ = shell_observation(mu, 0.04, capsys)
        spacings = [0.02]
        previous_size = 0.04
        for cell_size in spacings:
            error, count, wall = shell_observation(mu, cell_size, capsys)
            order = np.log(previous_error / error) / np.log(previous_size / cell_size)
            with capsys.disabled():
                print(
                    f"\nshell refinement mu={mu}: {previous_size:g}->{cell_size:g}, "
                    f"cells={previous_count}->{count}, order={order.tolist()}",
                    flush=True,
                )
            assert count > previous_count
            assert np.all(error < previous_error), (
                "exact-shell error must fall at every probe"
            )
            if mu == 100 and cell_size == 0.02:
                # Halving the spacing roughly quadruples cells and edges, so
                # allow the optional operator only within a sixteenfold budget.
                refine_further = wall <= 120
                with capsys.disabled():
                    print(
                        f"\nshell optional 0.01 m: admitted={refine_further}, "
                        f"measured_0.02_wall_seconds={wall:.6f}, "
                        f"projected_0.01_wall_seconds={16 * wall:.6f}",
                        flush=True,
                    )
            if cell_size == 0.02 and refine_further:
                spacings.append(0.01)
            previous_error, previous_count, previous_size = error, count, cell_size
    with capsys.disabled():
        print(
            f"\nshell refinement total_wall_seconds={perf_counter() - started:.6f}",
            flush=True,
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


def test_material_is_monotone_and_holds_end_values():
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
    top = iron._coefficient(jnp.asarray([6.0, 8.0, 80.0]), *law)
    np.testing.assert_array_equal(top, np.full(3, float(top[0])))
    at_zero = iron._coefficient(jnp.asarray(0.0), *law)
    np.testing.assert_allclose(at_zero, (215 - 1) / (iron._MU0 * 215))
    # Positive samples alone do not imply a positive differential material law.
    with pytest.raises(ValueError, match="differential permeability"):
        iron._fit_law([0, 1, 2], [2, 2, 10])


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
    with capsys.disabled():
        print(
            f"\nnonlinear shell: converged={result.converged}, "
            f"iterations={result.iterations}, residual={result.residual_history}, "
            f"quadratic_ratio={quadratic_ratio:.12g}, "
            f"observed_order={observed_order:.12g}, "
            f"mu_range={result.permeability_relative.min():.6g}-"
            f"{result.permeability_relative.max():.6g}",
            flush=True,
        )
    assert result.converged
    assert result.iterations <= 10
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
