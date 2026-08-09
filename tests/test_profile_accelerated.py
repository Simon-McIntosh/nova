"""Accelerated-reconstruction contracts on a physically consistent machine.

One small machine built through the canonical Green kernels carries every
contract: the bootstrap-then-remeasure construction yields exactly
self-consistent magnetics, so the least-squares reconstruction has a known
generating profile.  Pinned here: cold-seed recovery of that profile, the
profile-relaxation damping guard sharing the undamped fixed point, the
reported axis being the solved sub-grid O-point (not the static seed), and
the promoted accelerators (Anderson, exact-tangent Newton-Krylov) reaching
the Picard fixed point far inside a shared cold-seed evaluation budget.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

    from nova.equilibrium import ProfileDegrees, ReconstructProfile
    from nova.equilibrium.measurement import Magnetics
    from nova.equilibrium.fixed_point import anderson, newton_krylov, picard
    from nova.equilibrium.stencil_nulls import magnetic_axis_subgrid


NR = NZ = 17


def test_equilibrium_package_import_does_not_require_jax():
    """The package boundary remains importable when JAX is unavailable."""
    probe = textwrap.dedent(
        """
        import builtins
        import sys

        original_import = builtins.__import__

        def block_jax(name, *args, **kwargs):
            if name == "jax" or name.startswith("jax."):
                raise ModuleNotFoundError("JAX blocked by package-boundary probe")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = block_jax
        import nova.equilibrium
        assert not any(name == "jax" or name.startswith("jax.") for name in sys.modules)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_profile_axis_is_independent_of_dtype_bootstrap_import_order():
    """Physical metadata and the published axis stay fp64 after setup."""
    probe = textwrap.dedent(
        """
        import importlib
        import json
        import sys

        import numpy as np

        for name in sys.argv[1:]:
            importlib.import_module(name)

        from nova.jax.config import configure_dtypes
        configure_dtypes()

        import jax.numpy as jnp
        from nova.equilibrium.profile import ProfileDegrees, ReconstructProfile

        size = 11
        radial = np.linspace(6.14, 6.26, size, dtype=np.float64)
        vertical = np.linspace(-3.75, -3.65, size, dtype=np.float64)
        rr, zz = np.meshgrid(radial, vertical)
        count = rr.size
        solver = ReconstructProfile(
            grid_r=radial,
            grid_z=vertical,
            inside_limiter=np.ones(rr.shape, dtype=bool),
            cell_area=np.full(count, 1.0e-4, dtype=np.float64),
            source_to_grid=np.zeros((count, 1), dtype=np.float64),
            plasma_to_grid=np.zeros((count, count), dtype=np.float64),
            source_to_sensor=np.zeros((1, 1), dtype=np.float64),
            plasma_to_sensor=np.zeros((1, count), dtype=np.float64),
            source_names=('source',),
            degrees=ProfileDegrees(1, 1),
            axis_seed=(6.2, -3.7),
            wall_r=np.array([6.14, 6.26, 6.26, 6.14], dtype=np.float64),
            wall_z=np.array([-3.75, -3.75, -3.65, -3.65], dtype=np.float64),
        )
        truth = np.array([6.2031, -3.6982], dtype=np.float64)
        flux = -((rr - truth[0]) ** 2 + 1.3 * (zz - truth[1]) ** 2)
        flux = jnp.asarray(flux.ravel(), dtype=jnp.float64)
        basis = jnp.zeros((count, 2), dtype=jnp.float64)
        topology = {
            'psi_bnd': jnp.asarray(-1.0, dtype=jnp.float64),
            'core_weight': jnp.ones(rr.shape, dtype=jnp.float64),
        }
        result = solver._result(
            flux,
            flux,
            basis,
            jnp.zeros(2, dtype=jnp.float64),
            topology,
        )
        print(json.dumps({
            'axis': np.asarray(result.axis).tolist(),
            'axis_dtype': str(result.axis.dtype),
            'grid_dtype': str(solver.grid_r.dtype),
        }))
        """
    )
    modules = (
        "nova.equilibrium.stencil_nulls",
        "nova.equilibrium.profile",
    )
    rows = []
    for order in (modules, modules[::-1]):
        completed = subprocess.run(
            [sys.executable, "-c", probe, *order],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        rows.append(json.loads(completed.stdout.strip().splitlines()[-1]))

    assert rows[0]["axis_dtype"] == rows[1]["axis_dtype"] == "float64"
    assert rows[0]["grid_dtype"] == rows[1]["grid_dtype"] == "float64"
    np.testing.assert_array_equal(rows[0]["axis"], rows[1]["axis"])


def _machine() -> ReconstructProfile:
    grid_r = np.linspace(0.65, 1.35, NR)
    grid_z = np.linspace(-0.5, 0.5, NZ)
    radius, height = np.meshgrid(grid_r, grid_z)
    inside = ((radius - 1.0) / 0.3) ** 2 + (height / 0.42) ** 2 <= 1.0
    angle = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    return ReconstructProfile.from_geometry(
        grid_r=grid_r,
        grid_z=grid_z,
        inside_limiter=inside,
        cell_width=np.array(grid_r[1] - grid_r[0]),
        cell_height=np.array(grid_z[1] - grid_z[0]),
        source_r=np.array([1.0, 1.0]),
        source_z=np.array([0.8, -0.8]),
        source_width=np.array([0.12, 0.12]),
        source_height=np.array([0.12, 0.12]),
        source_names=("shaping_upper", "shaping_lower"),
        magnetics=Magnetics(
            r=1.0 + 0.5 * np.cos(theta),
            z=0.6 * np.sin(theta),
            angle=np.zeros(16),
            flux_loop=np.ones(16, dtype=bool),
        ),
        degrees=ProfileDegrees(n_pressure=1, n_diamagnetic=1),
        axis_seed=(1.0, 0.0),
        wall_r=1.0 + 0.31 * np.cos(angle),
        wall_z=0.43 * np.sin(angle),
        iterations=8,
        relaxation=0.6,
        topology_levels=48,
        topology_bisections=12,
        topology_rays=32,
    )


@pytest.fixture(scope="module")
def problem():
    """Machine plus exactly self-consistent measurements and their profile.

    A bootstrap reconstruction converges from the compact current seed's own
    sensor image; re-measuring at that converged state closes the loop, so
    the returned coefficients generate the returned data exactly (up to the
    tiny ridge bias).
    """
    solver = _machine()
    source_current = solver.pack_source_currents(
        {"shaping_upper": -1.0e4, "shaping_lower": -1.0e4}
    )
    plasma_current = jnp.asarray(5.0e4)
    initial = solver.initial_flux(source_current, plasma_current)
    seed_cell = jnp.linalg.lstsq(
        solver.plasma_to_grid, initial - solver.source_to_grid @ source_current
    )[0]
    bootstrap = solver.source_to_sensor @ source_current + (
        solver.plasma_to_sensor @ seed_cell
    )
    scale = jnp.full(bootstrap.size, 1.0e-3)
    mask = jnp.ones(bootstrap.size, dtype=bool)
    boot_map = solver.least_squares_map(
        source_current, plasma_current, bootstrap, scale, mask
    )
    boot = picard(boot_map, initial, evaluations=200, relaxation=0.6)
    assert float(boot.residual) < 1e-10
    basis, topology = solver._profile_basis(boot.state)
    coefficients = solver._least_squares_coefficients(
        basis, source_current, plasma_current, bootstrap, scale, mask
    )
    assert np.isfinite(float(topology["psi_bnd"]))  # the read binds at truth
    measured = solver.source_to_sensor @ source_current + solver.plasma_to_sensor @ (
        basis @ coefficients
    )
    return {
        "solver": solver,
        "initial": initial,
        "coefficients": coefficients,
        "args": (source_current, plasma_current, measured, scale, mask),
    }


def test_cold_seed_recovery_of_the_generating_profile(problem):
    solver = problem["solver"]
    result = solver.least_squares(*problem["args"], problem["initial"])
    np.testing.assert_allclose(
        np.asarray(result.coefficients),
        np.asarray(problem["coefficients"]),
        rtol=2e-2,
    )
    assert float(result.residual) < 1e-2
    np.testing.assert_allclose(float(jnp.sum(result.cell_current)), 5.0e4, rtol=1e-9)


def test_profile_relaxation_shares_the_undamped_fixed_point(problem):
    """Damping slows the coefficient morph without moving the fixed point."""
    solver = problem["solver"]
    damped = dataclasses.replace(solver, profile_relaxation=0.4, iterations=24)
    result = damped.least_squares(*problem["args"], problem["initial"])
    reference = solver.least_squares(*problem["args"], problem["initial"])
    assert float(result.residual) < float(reference.residual)
    np.testing.assert_allclose(
        np.asarray(result.coefficients),
        np.asarray(reference.coefficients),
        rtol=1e-2,
    )


def test_profile_relaxation_validates():
    with pytest.raises(ValueError, match="profile_relaxation"):
        dataclasses.replace(_machine(), profile_relaxation=0.0)


def test_result_axis_is_the_solved_subgrid_null(problem):
    solver = problem["solver"]
    result = solver.least_squares(*problem["args"], problem["initial"])
    axis = np.asarray(result.axis)
    assert not np.allclose(axis, np.asarray(solver.axis_seed))  # Shafranov shift
    null = magnetic_axis_subgrid(
        np.asarray(result.flux).reshape(NZ, NR),
        solver.grid_r,
        solver.grid_z,
        solver.inside_limiter,
    )
    assert bool(null["found"])
    np.testing.assert_allclose(axis, [float(null["r"]), float(null["z"])], atol=1e-12)
    assert int(result.axis_state) == 2
    assert float(result.axis_confidence) >= 1.0
    assert int(result.axis_candidate_count) >= 1
    assert bool(result.axis_overflow) == (int(result.axis_candidate_count) > 1)


def test_accelerators_reach_the_fixed_point_inside_the_budget(problem):
    """Anderson and exact-tangent NK beat relaxed Picard at a shared budget.

    Budget 20 map evaluations from the cold seed; NK spends its 8-sweep
    warmup plus two Newton steps of (1 linearisation + 4 tangents + 1
    promotion) — the same evaluation count Picard and Anderson consume.
    """
    solver = problem["solver"]
    initial = problem["initial"]
    lsq_map = solver.least_squares_map(*problem["args"])
    plain = picard(lsq_map, initial, evaluations=20, relaxation=0.6)
    mixed = anderson(lsq_map, initial, evaluations=20, relaxation=0.6)
    newton = newton_krylov(
        lsq_map,
        initial,
        newton_steps=2,
        gmres_iterations=4,
        warmup=8,
        relaxation=0.6,
    )
    reference = picard(lsq_map, initial, evaluations=200, relaxation=0.6)
    assert float(mixed.residual) < 1e-3 * float(plain.residual)
    assert float(newton.residual) < 1e-3 * float(plain.residual)
    scale = float(jnp.max(jnp.abs(reference.state)))
    for accelerated in (mixed, newton):
        agreement = float(jnp.max(jnp.abs(accelerated.state - reference.state))) / scale
        assert agreement < 1e-8


if __name__ == "__main__":
    pytest.main([__file__])
