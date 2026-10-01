"""Record, or compare against, the stream's own outputs on the toy operators.

``record`` writes every field of every dense and elementwise toy case to an
npz beside this script; ``compare`` recomputes them and requires each field
to be bitwise equal to the recording.
"""

import subprocess
import sys

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import fixed_point

assert jax.config.jax_enable_x64 is True
mode = sys.argv[1]
path = "docs/figures/forward-solver-route-integrity/single-site-krylov/toy-stream-outputs.npz"
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print(f"revision={revision} mode={mode}", flush=True)
outputs = {}
for arm in ("dense", "elementwise"):
    rng = np.random.default_rng(7)
    for size, scale in ((5, 0.3), (12, 1.0), (40, 3.0), (3, 0.0)):
        for iterations in (2, 8):
            matrix = jnp.asarray(np.eye(size) + scale * rng.standard_normal((size, size)))
            diagonal = jnp.diag(matrix)
            rhs = jnp.asarray(rng.standard_normal(size))
            if arm == "elementwise":

                def action(v, diagonal=diagonal, scale=scale):
                    return diagonal * v + 0.3 * scale * jnp.roll(v, 1)
            else:

                def action(v, matrix=matrix):
                    return matrix @ v

            result = jax.jit(
                lambda: fixed_point._qualified_krylov_step(
                    action,
                    rhs,
                    jnp.asarray(0.1),
                    gmres_iterations=iterations,
                    condition_ratio_limit=10.0,
                    preceding_condition_baseline=jnp.asarray(2.0),
                )
            )()
            for field, value in zip(result._fields, result):
                outputs[f"{arm}-{size}-{iterations}-{field}"] = np.asarray(value)
if mode == "record":
    np.savez(path, **outputs)
    print(f"TOY_STREAM_RECORDED fields={len(outputs)}")
else:
    recorded = np.load(path)
    differing = [
        key
        for key, value in outputs.items()
        if not np.array_equal(value, recorded[key], equal_nan=True)
    ]
    for key in differing:
        print("DIFFERS", key, float(np.nanmax(np.abs(outputs[key] - recorded[key]))))
    print(f"TOY_STREAM_SELF_IDENTITY equal={len(outputs) - len(differing)}/{len(outputs)}")
    raise SystemExit(1 if differing else 0)
