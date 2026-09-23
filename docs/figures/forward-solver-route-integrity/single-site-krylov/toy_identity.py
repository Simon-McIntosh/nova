"""Bitwise comparison of the stream and per-site Krylov steps on dense operators."""

import sys

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax
import jax.numpy as jnp
import numpy as np

assert jax.config.jax_enable_x64 is True
sys.path.insert(0, "docs/figures/forward-solver-route-integrity/single-site-krylov")
import per_site

from nova.equilibrium import fixed_point

ELEMENTWISE = len(sys.argv) > 1
base = per_site.base_step()
rng = np.random.default_rng(7)
cases = 0
for size, scale in ((5, 0.3), (12, 1.0), (40, 3.0), (3, 0.0)):
    for iterations in (2, 8):
        matrix = jnp.asarray(np.eye(size) + scale * rng.standard_normal((size, size)))
        rhs = jnp.asarray(rng.standard_normal(size))
        kwargs = dict(
            gmres_iterations=iterations,
            condition_ratio_limit=10.0,
            preceding_condition_baseline=jnp.asarray(2.0),
        )
        diagonal = jnp.diag(matrix)
        args = (
            (lambda v: matrix @ v)
            if ELEMENTWISE is False
            else (lambda v: diagonal * v + 0.3 * scale * jnp.roll(v, 1)),
            rhs,
            jnp.asarray(0.1),
        )
        new = jax.jit(lambda: fixed_point._qualified_krylov_step(*args, **kwargs))()
        old = jax.jit(lambda: base(*args, **kwargs))()
        for field, a, b in zip(new._fields, new, old):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=field)
        cases += 1
        print(size, iterations, float(new.achieved_reduction), int(new.qualification))
print(f"TOY_BIT_IDENTITY_PASS cases={cases}")
