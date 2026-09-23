"""Exercise all Newton control flow behind one outlined map."""

import time
import numpy as np
import jax
import jax.numpy as jnp
from machine import operator_evaluation, shared_operator_call
from nova.equilibrium.fixed_point import newton_krylov
from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64
op = operator_evaluation(lambda x: 0.2 * x**2 + jnp.ones_like(x))


def solve(x):
    return newton_krylov(op, x, newton_steps=3, gmres_iterations=2, warmup=2)


x = jnp.zeros(2, dtype=jnp.float64)
start = time.perf_counter()
expected = jax.jit(solve)(x)
print("direct wall", time.perf_counter() - start, flush=True)
start = time.perf_counter()
compiled = jax.jit(lambda x: shared_operator_call(solve, x)).lower(x).compile()
print("machine compile wall", time.perf_counter() - start, flush=True)
actual = compiled(x)
for index, (a, b) in enumerate(zip(jax.tree.leaves(actual), jax.tree.leaves(expected))):
    np.testing.assert_array_equal(a, b, err_msg=f"leaf {index}")
print("solver bit-identical", flush=True)
