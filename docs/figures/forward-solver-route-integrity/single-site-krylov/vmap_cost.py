"""Wall time of one width-16 vmapped qualified step: stream against base body.

Under vmap the stream's cond becomes a select, so every slot applies the
operator. The operator is a dense 1529-square matrix, the certificate state
size at 300 cells, with eight GMRES iterations.
"""

import json
import statistics
import sys
import time

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax
import jax.numpy as jnp
import numpy as np

assert jax.config.jax_enable_x64 is True
sys.path.insert(0, "docs/figures/forward-solver-route-integrity/single-site-krylov")
import per_site

from nova.equilibrium import fixed_point

size, width, iterations = 1529, 16, 8
rng = np.random.default_rng(11)
matrix = jnp.asarray(np.eye(size) + 0.05 * rng.standard_normal((size, size)))
rhs = jnp.asarray(rng.standard_normal((width, size)))
base = per_site.base_step()


def batched(step):
    def one(b):
        return step(
            lambda v: matrix @ v,
            b,
            jnp.asarray(0.1),
            gmres_iterations=iterations,
            condition_ratio_limit=10.0,
            preceding_condition_baseline=jnp.asarray(2.0),
        ).step

    return jax.jit(jax.vmap(one))


receipt = dict(size=size, width=width, gmres_iterations=iterations, walls={})
for name, step in (("stream", fixed_point._qualified_krylov_step), ("base", base)):
    function = batched(step)
    function(rhs).block_until_ready()
    walls = []
    for _ in range(7):
        start = time.perf_counter()
        function(rhs).block_until_ready()
        walls.append(time.perf_counter() - start)
    receipt["walls"][name] = dict(median_seconds=statistics.median(walls), all=walls)
receipt["ratio_stream_over_base"] = (
    receipt["walls"]["stream"]["median_seconds"]
    / receipt["walls"]["base"]["median_seconds"]
)
print(json.dumps(receipt, indent=2))
print(f"VMAP_COST ratio={receipt['ratio_stream_over_base']:.3f}")
