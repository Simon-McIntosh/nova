"""Dump the compiled width-16 vmapped step of the rule and of the base.

Arguments: output directory. Writes ``rule.hlo.txt`` and ``base.hlo.txt``, the
optimised programs the backend runs, so their control flow per slot (while
loops, conditionals, library calls) can be counted.
"""

import sys
from pathlib import Path

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "docs/figures/forward-solver-route-integrity/single-site-krylov")
import per_site

from nova.equilibrium import fixed_point

output = Path(sys.argv[1])
size, width, iterations = 1529, 16, 8
rng = np.random.default_rng(11)
matrix = jnp.asarray(np.eye(size) + 0.05 * rng.standard_normal((size, size)))
rhs = jnp.asarray(rng.standard_normal((width, size)))


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


for name, step in (
    ("rule", fixed_point._qualified_krylov_step),
    ("base", per_site.base_step()),
):
    text = step_of(step).lower(rhs).compile().as_text()
    (output / f"{name}-{jax.default_backend()}.hlo.txt").write_text(text)
    print(name, len(text.splitlines()), flush=True)
