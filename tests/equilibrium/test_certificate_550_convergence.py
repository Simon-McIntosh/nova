"""The published limited carrier is solved by the certificate route."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_stationary_root_tangent_at_exact_seed() -> None:
    from dataclasses import replace

    import jax
    import jax.numpy as jnp
    import numpy as np

    from nova.equilibrium import fixed_points
    from nova.jax.config import configure_dtypes
    from tests.test_fixed_points import polynomial_cache

    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    print(f"ROOT_MODULE={fixed_points.__file__}")
    cache = polynomial_cache()
    rows = cache.rows.at[:, 0, 0, 0].set(0.2 * cache.nodes[:, 1])
    rows = rows.at[:, 0, 0, 2].set(0.2)
    cache = replace(cache, rows=rows)
    initial = jnp.asarray(((0.13, -0.21),))
    zero = jnp.zeros((1, 3))
    seed = fixed_points.solve(
        cache, zero, initial, initial, jnp.asarray(1.0)
    ).position
    at_seed = fixed_points.solve(cache, zero, seed, seed, jnp.asarray(1.0))
    assert int(at_seed.iterations[0]) == 0

    def root(amplitude):
        moments = jnp.asarray(((amplitude, 0.0, 0.0),))
        result = fixed_points.solve(cache, moments, seed, seed, jnp.asarray(1.0))
        return result.position[0]

    amplitude = jnp.asarray(0.0)
    _, tangent = jax.jvp(root, (amplitude,), (jnp.asarray(1.0),))
    step = 1.0 / 4096.0
    finite = (root(amplitude + step) - root(amplitude - step)) / (2 * step)
    print(f"ROOT_TANGENT jvp={np.asarray(tangent)} central={np.asarray(finite)}")
    np.testing.assert_allclose(tangent, finite, rtol=2e-6, atol=1e-10)
    assert abs(float(tangent[1])) > 0.05


@pytest.mark.slow
def test_certified_limited_carrier_converges(tmp_path: Path) -> None:
    receipt_path = tmp_path / "limited-solve.json"
    command = [
        sys.executable,
        "-m",
        "scripts.prototypes.fast_map_production.solver",
        "--kind",
        "limited",
        "--cells",
        "550",
        "--method",
        "gmres",
        "--line-search",
        "--line-search-depth",
        "20",
        "--output",
        str(receipt_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
    )
    assert receipt_path.exists(), completed.stdout + completed.stderr
    receipt = json.loads(receipt_path.read_text())
    assert receipt["cells"] == 553
    assert receipt["carrier_cache"]["hit"]
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert receipt["converged"]
    assert receipt["history"][-1]["residual"] <= 1e-12
