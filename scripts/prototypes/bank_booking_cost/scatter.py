"""Isolate production cell-bucket scatters from runtime derivative arrays."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mask", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root))
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from nova.equilibrium import current

    assert jax.config.jax_enable_x64
    if args.mask is None:
        from scripts.prototypes.bank_step_booking_batch.measure import inputs
        from scripts.prototypes.bank_solve_cycle import measure as cycle

        machine = cycle.reference.machine("diverted")
        _, support, _ = inputs(16, machine)
        reading = support.reading
        cut = (reading.membership > 0) & (reading.membership < 1 - 1e-14)
        normal = cut & reading.normal_form_cells
        assert bool(jnp.any(cut & ~normal)) and bool(jnp.any(normal))
        np.savez(
            args.output, conic=np.asarray(cut & ~normal), normal=np.asarray(normal)
        )
        return
    with np.load(args.mask) as data:
        conic, normal = jnp.asarray(data["conic"]), jnp.asarray(data["normal"])
    cells = conic.shape[1]
    values = jnp.asarray(np.random.default_rng(71).normal(size=(16, cells, 14, 3)))

    def one(value, conic_mask, normal_mask):
        zero = jnp.zeros_like(value)
        return current._integrate_bucket(
            conic_mask, 32, lambda i: value[i], zero
        ) + current._integrate_bucket(
            normal_mask,
            min(current._NORMAL_CUT_BATCH_WIDTH, cells),
            lambda i: value[i],
            zero,
        )

    program = jax.jit(jax.vmap(one))
    actual = jax.block_until_ready(program(values, conic, normal))
    np.testing.assert_allclose(
        actual, jnp.where((conic | normal)[..., None, None], values, 0)
    )
    times = []
    for _ in range(40):
        start = time.perf_counter()
        jax.block_until_ready(program(values, conic, normal))
        times.append(1000 * (time.perf_counter() - start))
    row = {
        "module": current.__file__,
        "median_ms": statistics.median(times),
        "samples_ms": times,
        "conic_counts": np.asarray(conic.sum(axis=1)).tolist(),
        "normal_counts": np.asarray(normal.sum(axis=1)).tolist(),
    }
    args.output.write_text(json.dumps(row, indent=2))
    print("SCATTER", json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
