"""Check continuous equality over in-cell stationary-point displacements."""

from nova.jax.config import configure_dtypes

configure_dtypes()

import json  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from nova.equilibrium import support_read  # noqa: E402
from tests.test_tracked_read import _case, _shifted  # noqa: E402


def main():
    operands, cache, roots, convention, policy, moments, analytic = _case("diverted")
    points = roots[None]
    states = jax.tree.map(lambda value: value[None], moments)
    run = jax.jit(
        lambda bank, rows, tracking: support_read.read_batch(
            bank, rows, states, points, points, convention, policy, tracking=tracking
        )
    )
    first = jax.block_until_ready(run(operands, cache, None))
    tracked = support_read.tracking_of(first)
    hessian = np.asarray(analytic.evaluate(roots[1]).hessian)
    pitch = float(jnp.median(operands.geometry.pitch))
    for distance in (1e-6, 1e-3, 0.01, 0.2):
        shifted, rows = _shifted(
            operands,
            cache,
            jnp.asarray(hessian @ (distance * pitch * np.array([1.0, 0.0]))),
        )
        carried = jax.block_until_ready(run(shifted, rows, tracked))
        fresh = jax.block_until_ready(run(shifted, rows, None))
        errors = {}
        for name in ("membership", "edge_interval", "fragment_area"):
            a, b = (
                np.asarray(getattr(carried.reading, name)),
                np.asarray(getattr(fresh.reading, name)),
            )
            errors[name] = float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))
        print(
            "SWEEP",
            json.dumps(
                dict(
                    displacement_pitch=distance,
                    rebuilt=np.asarray(carried.rebuilt).tolist(),
                    valid=np.asarray(carried.valid).tolist(),
                    fresh_valid=np.asarray(fresh.valid).tolist(),
                    changed_normal_cells=int(
                        jnp.sum(
                            carried.reading.normal_form_cells
                            != fresh.reading.normal_form_cells
                        )
                    ),
                    changed_fragments=int(
                        jnp.sum(
                            carried.reading.fragment_selected
                            != fresh.reading.fragment_selected
                        )
                    ),
                    errors=errors,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
