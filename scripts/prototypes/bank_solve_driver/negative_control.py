"""Show that a per-lane refactor branch rebuilds on a reuse step."""

import importlib.util
import os
from pathlib import Path
import sys

import pytest

import nova.equilibrium as equilibrium
from nova.equilibrium import support_read


SOURCE = """        correction = jax.lax.cond(
            rebuild,
            lambda _: jax.lax.map(one_fresh, (state, step_support)),
            lambda _: jax.lax.map(one_with_factor, (state, step_support, factor)),
            None,
        )"""
MUTATION = """        correction = jax.vmap(
            lambda one, one_read, prior, lane_ratio: newton_step(
                operands, one, one_read, profile, density_gradient, quadrature,
                fresh_geometry=True, condition_diagnostics=False,
                factor_cache=prior, contraction_ratio=lane_ratio,
            )
        )(
            state, step_support, factor,
            jnp.where(have_factor & ~changed, ratio, jnp.inf),
        )"""


def main():
    source = Path(support_read.__file__).read_text()
    assert source.count(SOURCE) == 1
    scratch = Path(os.environ["TMPDIR"])
    path = scratch / "support_read.py"
    path.write_text(source.replace(SOURCE, MUTATION))
    name = "nova.equilibrium.support_read"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    equilibrium.support_read = module
    print(f"MUTANT_MODULE={module.__file__} CWD={Path.cwd().resolve()}", flush=True)
    return pytest.main(
        [
            "-p",
            "no:cacheprovider",
            "tests/test_bank_solve_driver.py::test_batch_reuses_factor_without_rebuilding_jacobians",
            "-q",
            f"--basetemp={scratch / 'pytest'}",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
