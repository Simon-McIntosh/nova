"""Compare shared operator control flow with nested JIT calls."""

import re
import numpy as np
import jax
import jax.numpy as jnp
from machine import operator_evaluation, shared_operator_call
from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64
op = operator_evaluation(lambda x: jnp.sin(x))


def direct(x):
    return op(op(op(x)))


def shared(x):
    return shared_operator_call(direct, x)


x = jnp.arange(4, dtype=jnp.float64)
for fn in (direct, shared):
    compiled = jax.jit(fn).lower(x).compile()
    hlo = compiled.as_text()
    count = len(re.findall(r" = f64\[4\].* sine\(", hlo))
    print(fn.__name__, "sine_count", count, "result", compiled(x), flush=True)
    if fn is shared:
        np.testing.assert_array_equal(compiled(x), jax.jit(direct)(x))


def controls(x):
    def body(i, state):
        return jax.lax.cond(
            i % 2 == 0, lambda y: op(y) + 0.2, lambda y: op(y + 0.3), state
        )

    x = jax.lax.fori_loop(0, 4, body, x)
    x, history = jax.lax.scan(
        lambda carry, y: (op(carry + y), carry),
        x,
        jnp.arange(3, dtype=x.dtype),
        reverse=True,
    )
    x = jax.lax.while_loop(
        lambda s: s[0] < 3, lambda s: (s[0] + 1, op(s[1])), (jnp.int32(0), x)
    )[1]
    primal, tangent = jax.linearize(op, x)
    return primal + tangent(jnp.ones_like(x)), history


for fn in (controls,):
    direct_result = jax.jit(fn)(x)
    shared_result = jax.jit(lambda y: shared_operator_call(fn, y))(x)
    for actual, expected in zip(
        jax.tree.leaves(shared_result), jax.tree.leaves(direct_result)
    ):
        np.testing.assert_array_equal(actual, expected)
    print(fn.__name__, "bit-identical", flush=True)
