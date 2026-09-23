"""Bitwise comparison of the exit-loop and capacity-scan steps on toy operators."""

import sys

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "docs/figures/forward-solver-route-integrity/single-site-krylov")
import scan_stream

from nova.equilibrium import fixed_point


def run(step, seed, size, scale):
    rng = np.random.default_rng(seed)
    matrix = jnp.asarray(np.eye(size) + scale * rng.standard_normal((size, size)))
    rhs = jnp.asarray(rng.standard_normal(size))
    result = jax.jit(
        lambda b: step(
            lambda v: matrix @ v,
            b,
            jnp.asarray(0.1),
            gmres_iterations=8,
            condition_ratio_limit=10.0,
            preceding_condition_baseline=jnp.asarray(2.0),
        )
    )(rhs)
    return [np.asarray(x) for x in result]


cases = [(s, n, a) for s in range(4) for n, a in ((12, 0.2), (40, 0.1))]
exit_rows = [run(fixed_point._qualified_krylov_step, *c) for c in cases]
scan_stream.install()
scan_rows = [run(fixed_point._qualified_krylov_step, *c) for c in cases]
equal = 0
for case, a, b in zip(cases, exit_rows, scan_rows, strict=True):
    same = all(np.array_equal(x, y, equal_nan=True) for x, y in zip(a, b))
    equal += same
    print(
        case,
        "bitwise"
        if same
        else f"max diff {max(float(np.max(np.abs(x - y))) for x, y in zip(a, b)):.3e}",
    )
print(f"EXIT_TOY_IDENTITY {equal}/{len(cases)}")
