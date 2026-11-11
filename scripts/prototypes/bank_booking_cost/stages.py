"""Time explicit conic velocity and quadrature stages on actual cut cells."""

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
    parser.add_argument("--channels", type=int, choices=(6, 7), required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root))
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from nova.equilibrium import current, support_read as sr
    from scripts.prototypes.bank_step_booking_batch.measure import inputs
    from scripts.prototypes.bank_solve_cycle import measure as cycle

    assert jax.config.jax_enable_x64
    machine = cycle.reference.machine("diverted")
    bank, _, _, _, _, profile, quadrature, *_ = machine
    states, support, _ = inputs(16, machine)
    reading = support.reading
    conic = np.asarray(
        (reading.membership > 0)
        & (reading.membership < 1 - 1e-14)
        & ~reading.normal_form_cells
    )
    counts = conic.sum(axis=1)
    assert np.all(counts == counts[0]) and counts[0] > 0
    indices = jnp.asarray(np.stack([np.flatnonzero(mask) for mask in conic]))
    vertices = bank.geometry.vertices[indices]
    vertex_count = bank.geometry.vertex_count[indices]
    centre = bank.geometry.centre[indices]
    pitch = bank.geometry.pitch[indices]
    selected = jax.vmap(lambda values, i: values[i])(reading.fragment_selected, indices)
    signed = quadrature.sigma * jax.vmap(lambda values, i: values[i])(
        reading.field_coefficients, indices
    )
    signed = signed.at[:, :, 0].add(-quadrature.sigma * reading.boundary_flux[:, None])
    psi = jax.vmap(lambda m: sr.nodal_jets(bank, m)[bank.sample_index, 0])(states)
    coefficients = jnp.einsum("cij,bcj->bci", bank.geometry.fit_inverse, psi)
    normalized = (
        coefficients.at[:, :, 0].add(-reading.axis_flux[:, None])
        / (reading.boundary_flux - reading.axis_flux)[:, None, None]
    )
    normalized = jax.vmap(lambda values, i: values[i])(normalized, indices)
    directions = quadrature.sigma * jnp.eye(6)
    if args.channels == 7:
        directions = jnp.concatenate((directions, -directions[:1]), axis=0)
    result = {"channels": args.channels, "cut_counts": counts.tolist(), "rows": []}

    def timed(name, fn, *values):
        program = jax.jit(fn).lower(*values).compile()
        answer = jax.block_until_ready(program(*values))
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            answer = jax.block_until_ready(program(*values))
            samples.append(1000 * (time.perf_counter() - start))
        norm = float(
            jnp.linalg.norm(
                jnp.concatenate([x.reshape(-1) for x in jax.tree.leaves(answer)])
            )
        )
        assert np.isfinite(norm) and norm > 0, (name, norm)
        row = {
            "name": name,
            "median_ms": statistics.median(samples),
            "samples_ms": samples,
            "norm": norm,
        }
        print("STAGE", json.dumps(row), flush=True)
        result["rows"].append(row)
        args.output.write_text(json.dumps(result, indent=2))
        return answer

    def rings(v, n, c, p, s, f):
        return current._conic_rings(v, n, c, p, s, f, quadrature.segments)[:2]

    def velocity(v, n, c, p, s, f, d):
        return jax.vmap(
            lambda tangent: current._conic_ring_tangents(
                v, n, c, p, s, tangent, f, quadrature.segments
            )
        )(d)

    rings_now, ring_counts = timed(
        "conic_rings",
        jax.vmap(jax.vmap(rings)),
        vertices,
        vertex_count,
        centre,
        pitch,
        signed,
        selected,
    )
    speeds = timed(
        "conic_velocities",
        jax.vmap(
            jax.vmap(velocity, in_axes=(0, 0, 0, 0, 0, 0, None)),
            in_axes=(0, 0, 0, 0, 0, 0, None),
        ),
        vertices,
        vertex_count,
        centre,
        pitch,
        signed,
        selected,
        directions,
    )

    def integral(rings, speed, n, c, p, coefficient):
        def evaluate(chosen_rings, chosen_counts, chosen):
            moved = current.moving_polygon_current_moment_tangents(
                chosen_rings,
                speed[:, chosen],
                chosen_counts,
                c,
                p,
                coefficient,
                profile,
            )
            return jnp.stack(tuple(jnp.sum(value, axis=1) for value in moved), axis=-1)

        return current._integrate_live_rings(
            rings, n, evaluate, jnp.zeros((args.channels, 3)), width=2
        )

    timed(
        "clipped_boundary_quadrature",
        jax.vmap(jax.vmap(integral)),
        rings_now,
        speeds,
        ring_counts,
        centre,
        pitch,
        normalized,
    )
    print("COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
