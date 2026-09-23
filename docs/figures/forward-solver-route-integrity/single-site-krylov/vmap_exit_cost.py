"""Wall time of one width-16 vmapped qualified step: exit loop, base body, scan.

The operator is a dense 1529-square matrix, the certificate state size at 300
cells, with eight GMRES iterations, as in ``vmap_cost.py``. The first argument
is the output receipt; a second argument ``scan`` rebinds the fixed-capacity
cond-gated scan stream first, the negative control. Each member's unbatched
application count is recorded, so a batch whose members all need the full
83 slots is distinguishable from one that exits early.
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
import scan_stream

from nova.equilibrium import fixed_point

arm = sys.argv[2] if len(sys.argv) > 2 else "exit"
if arm == "scan":
    scan_stream.install()
size, width, iterations = 1529, 16, 8
rng = np.random.default_rng(11)
matrix = jnp.asarray(np.eye(size) + 0.05 * rng.standard_normal((size, size)))
rhs = jnp.asarray(rng.standard_normal((width, size)))
base = per_site.base_step()
applications = []


def call(step, operator, b):
    return step(
        operator,
        b,
        jnp.asarray(0.1),
        gmres_iterations=iterations,
        condition_ratio_limit=10.0,
        preceding_condition_baseline=jnp.asarray(2.0),
    ).step


def batched(step):
    return jax.jit(jax.vmap(lambda b: call(step, lambda v: matrix @ v, b)))


def counted(v):
    jax.debug.callback(lambda _: applications.append(1), v)
    return matrix @ v


member_applications = []
for b in rhs[:4]:
    applications.clear()
    jax.jit(lambda b: call(fixed_point._qualified_krylov_step, counted, b))(
        b
    ).block_until_ready()
    member_applications.append(len(applications))
receipt = dict(
    arm=arm,
    backend=jax.default_backend(),
    devices=[str(d) for d in jax.devices()],
    size=size,
    width=width,
    gmres_iterations=iterations,
    member_applications_first_four=member_applications,
    walls={},
)
for name, step in ((arm, fixed_point._qualified_krylov_step), ("base", base)):
    function = batched(step)
    start = time.perf_counter()
    function(rhs).block_until_ready()
    first = time.perf_counter() - start
    walls = []
    for _ in range(7):
        start = time.perf_counter()
        function(rhs).block_until_ready()
        walls.append(time.perf_counter() - start)
    receipt["walls"][name] = dict(
        median_seconds=statistics.median(walls), first_call_seconds=first, all=walls
    )
receipt["ratio_over_base"] = (
    receipt["walls"][arm]["median_seconds"] / receipt["walls"]["base"]["median_seconds"]
)
with open(sys.argv[1], "w") as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps(receipt, indent=2))
print(f"VMAP_COST arm={arm} ratio={receipt['ratio_over_base']:.3f}")
