import time, statistics
from nova.jax.config import configure_dtypes

configure_dtypes()
import jax, jax.numpy as jnp, numpy as np

n, k, width, slots = 1529, 9, 16, 83
rng = np.random.default_rng(1)
m = jnp.asarray(np.eye(n) + 0.05 * rng.standard_normal((n, n)))
v = jnp.asarray(rng.standard_normal((width, n)))


def loop(batched_predicate):
    def f(x):
        def body(c):
            i, stop, big, vec = c
            vec = m @ vec
            return (
                i + 1,
                stop,
                big.at[:, i % k].add(1e-3 * vec),
                vec / jnp.linalg.norm(vec),
            )

        pred = (
            (lambda c: c[0] < c[1]) if batched_predicate else (lambda c: c[0] < slots)
        )
        stop = jnp.int32(slots) + 0 * x[0].astype(jnp.int32)
        return jax.lax.while_loop(
            pred, body, (jnp.int32(0), stop, jnp.zeros((n, 2 * k + 6)), x)
        )[2]

    return f


for name, b in (("unbatched-predicate", False), ("batched-predicate", True)):
    g = jax.jit(jax.vmap(loop(b)))
    g(v).block_until_ready()
    w = []
    for _ in range(7):
        t = time.perf_counter()
        g(v).block_until_ready()
        w.append(time.perf_counter() - t)
    print(name, jax.default_backend(), statistics.median(w))
