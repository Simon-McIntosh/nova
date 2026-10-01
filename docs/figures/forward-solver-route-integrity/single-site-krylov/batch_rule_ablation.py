"""Where the width-16 vmapped Krylov stream spends its time, by ablation.

Each arm rebinds one piece of the stream's batching rule in the live module
and times the same width-16 dense step as ``vmap_exit_cost.py``, beside the
per-site base in the same process. Two floors time a bare 83-slot loop of
batched operator applications: one exiting on a device-side any-pending
predicate, one on an unbatched counter.
"""

import inspect
import json
import statistics
import sys
import time

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "docs/figures/forward-solver-route-integrity/single-site-krylov")
import per_site

from nova.equilibrium import fixed_point

size, width, iterations, slots = 1529, 16, 8, 83
rng = np.random.default_rng(11)
matrix = jnp.asarray(np.eye(size) + 0.05 * rng.standard_normal((size, size)))
rhs = jnp.asarray(rng.standard_normal((width, size)))
source = inspect.getsource(fixed_point._serve_krylov_batch)
original_batch = fixed_point._serve_krylov_batch
original_fence = fixed_point._fenced


def variant(old, new):
    assert old in source, old
    namespace = fixed_point.__dict__
    local = {}
    exec(
        compile(source.replace(old, new), fixed_point.__file__, "exec"),
        namespace,
        local,
    )
    return local["_serve_krylov_batch"]


arms = {
    "rule": {},
    "no_barrier": {"_fenced": lambda action, vector: action(vector)},
    "no_shared_dispatch": {
        "_serve_krylov_batch": variant(
            "branch = jnp.where(shared, stream.phase[0], mixed)", "branch = mixed"
        )
    },
    "projection_always": {
        "_serve_krylov_batch": variant(
            "candidate = jax.lax.cond(\n            jnp.any(~proceed), project, lambda _: stream.candidate, advanced\n        )",
            "candidate = project(advanced)",
        )
    },
}


def timed(function, *args):
    function(*args).block_until_ready()
    walls = []
    for _ in range(7):
        start = time.perf_counter()
        function(*args).block_until_ready()
        walls.append(time.perf_counter() - start)
    return statistics.median(walls)


def step_of(step):
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


receipt = dict(backend=jax.default_backend(), devices=[str(d) for d in jax.devices()])
receipt["base"] = timed(step_of(per_site.base_step()), rhs)
for name, patch in arms.items():
    fixed_point._serve_krylov_batch = original_batch
    fixed_point._fenced = original_fence
    for key, value in patch.items():
        setattr(fixed_point, key, value)
    receipt[name] = timed(step_of(fixed_point._qualified_krylov_step), rhs)
    print(name, receipt[name], flush=True)
fixed_point._serve_krylov_batch = original_batch
fixed_point._fenced = original_fence


@jax.jit
def floor_any_pending(vectors):
    def body(carry):
        count, v = carry
        return count + 1, (v @ matrix.T) / jnp.linalg.norm(v, axis=1, keepdims=True)

    counts = jnp.zeros(width, dtype=jnp.int32)
    return jax.lax.while_loop(lambda c: jnp.any(c[0] < slots), body, (counts, vectors))[
        1
    ]


@jax.jit
def floor_counter(vectors):
    def body(index, v):
        return (v @ matrix.T) / jnp.linalg.norm(v, axis=1, keepdims=True)

    return jax.lax.fori_loop(0, slots, body, vectors)


receipt["floor_any_pending"] = timed(floor_any_pending, rhs)
receipt["floor_counter"] = timed(floor_counter, rhs)
print(json.dumps(receipt, indent=2))
with open(sys.argv[1], "w") as stream:
    json.dump(receipt, stream, indent=2)
