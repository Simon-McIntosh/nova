"""Bitwise comparison of the stream and per-site Krylov steps on toy operators.

Every case is compared against two references: the base per-site body as
committed, and the same body with an optimisation barrier on both sides of
each operator application (the like-for-like fusion boundary). One JSON row
per case records which fields differ and by how much.
"""

import json
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

ELEMENTWISE = len(sys.argv) > 1 and sys.argv[1] == "elementwise"
barrier = jax.lax.optimization_barrier
base = per_site.base_step()
rng = np.random.default_rng(7)
rows = []
for size, scale in ((5, 0.3), (12, 1.0), (40, 3.0), (3, 0.0)):
    for iterations in (2, 8):
        matrix = jnp.asarray(np.eye(size) + scale * rng.standard_normal((size, size)))
        diagonal = jnp.diag(matrix)
        rhs = jnp.asarray(rng.standard_normal(size))
        if ELEMENTWISE:

            def action(v, diagonal=diagonal, scale=scale):
                return diagonal * v + 0.3 * scale * jnp.roll(v, 1)
        else:

            def action(v, matrix=matrix):
                return matrix @ v

        def fenced(v, action=action):
            return barrier(action(barrier(v)))

        kwargs = dict(
            gmres_iterations=iterations,
            condition_ratio_limit=10.0,
            preceding_condition_baseline=jnp.asarray(2.0),
        )
        rest = (rhs, jnp.asarray(0.1))
        new = jax.jit(
            lambda: fixed_point._qualified_krylov_step(action, *rest, **kwargs)
        )()
        row = dict(size=size, iterations=iterations)
        for name, reference in (
            ("base", jax.jit(lambda: base(action, *rest, **kwargs))()),
            ("base-fenced", jax.jit(lambda: base(fenced, *rest, **kwargs))()),
        ):
            differing = {}
            for field, a, b in zip(new._fields, new, reference):
                a, b = np.asarray(a), np.asarray(b)
                if not np.array_equal(a, b, equal_nan=True):
                    differing[field] = float(np.nanmax(np.abs(a - b)))
            row[name] = differing
        rows.append(row)
        print(json.dumps(row), flush=True)
for name in ("base", "base-fenced"):
    equal = sum(not row[name] for row in rows)
    print(f"TOY_{name.upper().replace('-', '_')} bit_identical={equal}/{len(rows)}")
