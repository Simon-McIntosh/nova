"""Compare local tangent tile widths with fixed runtime states and matrix parity."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root))
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from nova.equilibrium import support_read as sr
    from scripts.prototypes.bank_step_booking_batch.measure import inputs
    from scripts.prototypes.bank_solve_cycle import measure as cycle

    assert jax.config.jax_enable_x64
    machine = cycle.reference.machine("diverted")
    bank, _, _, _, _, profile, quadrature, *_ = machine
    states, support, booked = inputs(16, machine)
    rows = []
    expected = None
    for width in (32, 64, 128):
        rule = replace(quadrature, cut_batch_width=width)

        def local(b, m, s, p):
            return jax.vmap(
                lambda x: sr._local_booking_sensitivities(
                    b, x[0], x[1], profile, lambda r, f: jnp.zeros_like(r), rule, x[2]
                )
            )((m, s, p))

        program = jax.jit(local).lower(bank, states, support, booked).compile()
        value = jax.block_until_ready(program(bank, states, support, booked))
        array = np.asarray(value)
        if expected is None:
            expected = array.copy()
        error = float(np.linalg.norm(array - expected) / np.linalg.norm(expected))
        assert error <= 1e-12, error
        samples = []
        for _ in range(3):
            start = time.perf_counter()
            jax.block_until_ready(program(bank, states, support, booked))
            samples.append(1000 * (time.perf_counter() - start))
        row = {
            "width": width,
            "median_ms": statistics.median(samples),
            "samples_ms": samples,
            "relative_error": error,
        }
        rows.append(row)
        args.output.write_text(json.dumps(rows, indent=2))
        print("TILE", json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
