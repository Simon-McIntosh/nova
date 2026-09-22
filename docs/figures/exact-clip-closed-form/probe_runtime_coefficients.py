"""Check explicitly compiled coefficient constants without outer constant folding."""

from functools import cache
from pathlib import Path
import json
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import nova.linalg.interpolant as interpolant
import reference_clip


@cache
def coefficients(order, extended_precision):
    with jax.ensure_compile_time_eval():
        dtype = jnp.int64 if extended_precision else jnp.int32

        @jax.jit
        def ratio(term):
            return jsp.special.gamma(order + 1) / (
                jsp.special.gamma(term + 1) * jsp.special.gamma(order - term + 1)
            )

        compiled = ratio.lower(jax.ShapeDtypeStruct((order + 1,), dtype)).compile()
        return tuple(np.asarray(compiled(jnp.arange(order + 1, dtype=dtype))).tolist())


output = (
    Path(__file__).resolve().parent
    / "bernstein-reference"
    / "runtime-coefficients-probe"
)
output.mkdir(exist_ok=True)
reference_arrays = np.load(output.parent / "production-patch-reference-arrays.npz")


def frozen():
    a = reference_arrays
    return (
        a["cell_ids"],
        a["atomic_vertices"],
        a["counts"],
        a["centres"],
        a["coefficients"][0],
        a["origin"][0, 0],
        a["scale"][0, 0],
    )


reference_clip.frozen_fixture = frozen
interpolant._binomial_coefficients = coefficients
reference_clip.measure(output)
path = output / "reference-summary.json"
result = json.loads(path.read_text())
result["runtime_mutation"] = (
    "Compile coefficient evaluation with an abstract term argument "
    "before freezing its output on the host."
)
path.write_text(json.dumps(result, indent=2) + "\n")
