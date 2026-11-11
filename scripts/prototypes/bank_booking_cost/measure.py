"""Paired runtime-argument booking attribution; accepts any repository root."""

import argparse
import json
from pathlib import Path
import re
import statistics
import sys
import time
from unittest.mock import patch


def compare_matrices(actual, expected, lanes):
    """Apply the base/head matrix parity guard used by the timing run."""
    import numpy as np

    relative = float(
        np.linalg.norm(np.asarray(actual) - expected) / np.linalg.norm(expected)
    )
    print("PARITY", lanes, relative, flush=True)
    assert relative <= 1e-12, (lanes, relative)
    return relative


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--focus", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from nova.equilibrium import current, support_read as sr
    from scripts.prototypes.bank_step_booking_batch.measure import inputs, jacobians
    from scripts.prototypes.bank_solve_cycle import measure as cycle

    assert jax.config.jax_enable_x64
    print("MODULE", sr.__file__, "CWD", Path.cwd().resolve(), flush=True)
    machine = cycle.reference.machine("diverted")
    bank, _, _, _, _, profile, quadrature, *_ = machine
    output = {
        "module": sr.__file__,
        "cwd": str(Path.cwd().resolve()),
        "backend": jax.default_backend(),
        "rows": [],
    }
    matrices = {}

    def measure(name, fn, *values):
        print("COMPILE", name, flush=True)
        lowered = jax.jit(fn).lower(*values)
        program = lowered.compile()
        result = jax.block_until_ready(program(*values))
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = jax.block_until_ready(program(*values))
            samples.append(1000 * (time.perf_counter() - start))
        hlo = program.as_text()
        operations = len(re.findall(r"^\s*%[^\n]* = ", hlo, re.M))
        stable = str(lowered.compiler_ir())
        row = {
            "name": name,
            "median_ms": statistics.median(samples),
            "samples_ms": samples,
            "optimized_hlo_operations": operations,
            "stablehlo_operations": stable.count("stablehlo."),
            "fusion_sites": len(re.findall(r"\bfusion\(", hlo)),
            "while_sites": len(re.findall(r"\bwhile\(", hlo)),
        }
        assert operations > 0 and row["stablehlo_operations"] > 0
        print("MEASURE", json.dumps(row), flush=True)
        output["rows"].append(row)
        args.output.write_text(json.dumps(output, indent=2))
        return result

    for lanes in (1, 16):
        states, support, booked = inputs(lanes, machine)

        def mapped(fn):
            return lambda b, m, s, p: jax.vmap(lambda x: fn(b, *x))((m, s, p))

        def local(b, m, s, p):
            return sr._local_booking_sensitivities(
                b, m, s, profile, lambda r, f: jnp.zeros_like(r), quadrature, p
            )

        matrix = measure(
            f"{lanes}:assembly",
            lambda b, m, s, p: jacobians(b, m, s, p, profile, quadrature),
            bank,
            states,
            support,
            booked,
        )
        matrices[f"matrix_{lanes}"] = np.asarray(matrix)
        if args.reference:
            with np.load(args.reference) as archive:
                expected = archive[f"matrix_{lanes}"]
            relative = compare_matrices(matrix, expected, lanes)
            output[f"parity_{lanes}"] = relative
        measure(f"{lanes}:local", mapped(local), bank, states, support, booked)
        if lanes == 16 and not args.focus:
            # Ablations retain the channel dimensions and all other production work.
            original = current._conic_ring_tangents

            def no_rates(*a):
                return jnp.zeros_like(original(*a))

            with patch.object(current, "_conic_ring_tangents", no_rates):
                measure(
                    f"{lanes}:local_without_conic_rates",
                    mapped(local),
                    bank,
                    states,
                    support,
                    booked,
                )
            original_integral = current.moving_polygon_current_moment_tangents

            def no_integral(*a):
                return jax.tree.map(jnp.zeros_like, original_integral(*a))

            with patch.object(
                current, "moving_polygon_current_moment_tangents", no_integral
            ):
                measure(
                    f"{lanes}:local_without_boundary_quadrature",
                    mapped(local),
                    bank,
                    states,
                    support,
                    booked,
                )

            def geometry(b, m, s, p):
                direction = jnp.ones_like(b.exterior)
                return sr._normal_geometry_booking_tangent(
                    b, m, direction, s, profile, quadrature, p
                )

            measure(
                f"{lanes}:normal_geometry",
                mapped(geometry),
                bank,
                states,
                support,
                booked,
            )
            ring_rates = sr._normal_ring_rates

            def no_normal_rates(*a):
                return jnp.zeros_like(ring_rates(*a))

            with patch.object(sr, "_normal_ring_rates", no_normal_rates):
                measure(
                    f"{lanes}:geometry_without_normal_ring_rates",
                    mapped(geometry),
                    bank,
                    states,
                    support,
                    booked,
                )
            # Isolate the final bank contraction and reshape from measured local arrays.
            values = jax.jit(mapped(local))(bank, states, support, booked)
            channels = jnp.ones(
                (lanes, 3 * states[0].shape[1], states[0].shape[1], values.shape[-1])
            )
            measure(
                f"{lanes}:matrix_contraction",
                lambda local, channels: jnp.einsum("bcok,bick->bcoi", local, channels),
                values,
                channels,
            )
    np.savez(args.output.with_suffix(".npz"), **matrices)
    args.output.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
