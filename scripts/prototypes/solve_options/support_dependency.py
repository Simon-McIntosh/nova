"""Separate moving field, root-jet and root-position booking tangents."""

# ruff: noqa: E402
import argparse
import json
from pathlib import Path
import resource
import subprocess
from time import perf_counter

from live_iterations import _setup

import jax
import jax.numpy as jnp
import numpy as np
import nova
from nova.equilibrium import current, fixed_points, gs, support_read, topology


def _time(program, *args):
    started = perf_counter()
    executable = program.lower(*args).compile()
    compilation = perf_counter() - started
    jax.block_until_ready(executable(*args))
    walls = []
    for _ in range(3):
        started = perf_counter()
        jax.block_until_ready(executable(*args))
        walls.append((perf_counter() - started) * 1000)
    return executable, compilation, float(np.median(walls))


def run(kind, cells, cache, bias_row, newton_row, out):
    if jax.default_backend() != "gpu" or jax.config.jax_enable_x64 is not True:
        raise RuntimeError(
            "support dependency measurement requires H200 double precision"
        )
    (
        initial,
        operands,
        window,
        roots,
        convention,
        policy,
        profile,
        quadrature,
        provenance,
    ) = _setup(kind, cells, cache, bias_row, newton_row)
    fixed = support_read.read(
        operands,
        window,
        tuple(initial[:, component] for component in range(3)),
        roots,
        roots,
        convention,
        policy,
        axis_sign=-convention.sigma,
    )
    if not bool(fixed.valid):
        raise ValueError("analytic support invalid")
    geometry = operands.geometry
    index = operands.sample_index
    scale = jnp.maximum(jnp.max(jnp.abs(initial[:, 0])), 1e-30)
    direction = jnp.asarray(np.random.default_rng(731).normal(size=initial.shape))
    direction *= scale * jnp.stack(
        (jnp.ones_like(geometry.pitch),) + (geometry.pitch,) * 2, axis=-1
    )

    def map_at(moments, ops, cache_window, reference, mode):
        if mode == "full":
            residual = gs.moment_residual(
                moments,
                ops,
                cache_window,
                roots,
                convention,
                policy,
                profile,
                quadrature,
            )[0]
            return moments - residual
        jets = support_read.nodal_jets(
            ops, tuple(moments[:, component] for component in range(3))
        )
        psi = jets[index, 0]
        if mode == "fixed_read":
            coefficients = jnp.einsum("nij,nj->ni", geometry.fit_inverse, psi)
            reading = reference.reading._replace(field_coefficients=coefficients)
        else:
            bell = fixed_points.field_from_jets(jets, index, ops.mesh)
            stationary = reference.stationary
            if mode == "fixed_roots_refreshed_jets":
                stationary = stationary._replace(
                    jet=jax.vmap(bell.evaluate)(stationary.position)
                )
            reading = topology.read_at_stationary_points(
                bell, geometry, convention, policy, stationary
            )
        booked = current.book(psi, reading, geometry, profile, quadrature)
        return jnp.stack(booked[:3], axis=-1)

    modes = (
        "fixed_read",
        "fixed_roots_fixed_jets",
        "fixed_roots_refreshed_jets",
        "full",
    )
    receipts = {}
    tangents = {}
    for mode in modes:
        tangent_program = jax.jit(
            lambda moments, vector, ops, cache_window, reference: jax.jvp(
                lambda state: map_at(state, ops, cache_window, reference, mode),
                (moments,),
                (vector,),
            )[1]
        )
        executable, compilation, wall = _time(
            tangent_program, initial, direction, operands, window, fixed
        )
        response = executable(initial, direction, operands, window, fixed)
        zero = executable(initial, jnp.zeros_like(direction), operands, window, fixed)
        amplitude = float(jnp.max(jnp.abs(response)))
        zero_amplitude = float(jnp.max(jnp.abs(zero)))
        if not np.isfinite(amplitude) or amplitude <= 1e-12:
            raise ValueError(f"{mode} derivative response is empty or nonfinite")
        if zero_amplitude > 1e-12:
            raise ValueError(f"{mode} zero direction is nonzero")
        tangents[mode] = response
        receipts[mode] = dict(
            compile_seconds=compilation,
            wall_ms=wall,
            response_max=amplitude,
            zero_direction_response_max=zero_amplitude,
        )
        print(f"MODE {kind} {cells} {mode} {json.dumps(receipts[mode])}", flush=True)

    full = tangents["full"]
    full_amplitude = float(jnp.max(jnp.abs(full)))
    for mode in modes[:-1]:
        receipts[mode]["relative_to_full"] = (
            float(jnp.max(jnp.abs(tangents[mode] - full))) / full_amplitude
        )
    result = dict(
        kind=kind,
        requested_cells=cells,
        realised_cells=len(initial),
        provenance=provenance,
        reference_status=np.asarray(fixed.stationary.status).tolist(),
        reference_valid=bool(fixed.valid),
        modes=receipts,
        gpu_memory=jax.devices()[0].memory_stats(),
        host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{kind}-{cells}.json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--bias-row", type=Path, required=True)
    parser.add_argument("--newton-row", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"revision={revision} tree={Path.cwd().resolve()}", flush=True)
    print(f"NOVA_MODULE={nova.__file__} GS_MODULE={gs.__file__}", flush=True)
    print(f"CWD={Path.cwd().resolve()} DEVICE={jax.devices()}", flush=True)
    run(args.kind, args.cells, args.cache, args.bias_row, args.newton_row, args.out)


if __name__ == "__main__":
    main()
