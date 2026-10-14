"""Measure which Bell jet entries carry moving-support booking response."""

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
from nova.equilibrium import current, fixed_points, support_read, topology


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
        raise RuntimeError("jet component measurement requires H200 double precision")
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
    support = support_read.read(
        operands,
        window,
        tuple(initial[:, component] for component in range(3)),
        roots,
        roots,
        convention,
        policy,
        axis_sign=-convention.sigma,
    )
    if not bool(support.valid):
        raise ValueError("analytic support invalid")
    base_jets = support_read.nodal_jets(
        operands, tuple(initial[:, component] for component in range(3))
    )
    geometry = operands.geometry
    index = operands.sample_index
    scale = jnp.maximum(jnp.max(jnp.abs(initial[:, 0])), 1e-30)
    direction = jnp.asarray(np.random.default_rng(731).normal(size=initial.shape))
    direction *= scale * jnp.stack(
        (jnp.ones_like(geometry.pitch),) + (geometry.pitch,) * 2, axis=-1
    )

    def map_with_mask(moments, ops, reference, baseline, mask):
        varied = support_read.nodal_jets(
            ops, tuple(moments[:, component] for component in range(3))
        )
        jets = baseline + (varied - baseline) * mask
        bell = fixed_points.field_from_jets(jets, index, ops.mesh)
        reading = topology.read_at_stationary_points(
            bell, geometry, convention, policy, reference.stationary
        )
        booked = current.book(jets[index, 0], reading, geometry, profile, quadrature)
        return jnp.stack(booked[:3], axis=-1)

    masks = {
        "value": (1, 0, 0, 0, 0, 0),
        "gradient": (0, 1, 1, 0, 0, 0),
        "hessian": (0, 0, 0, 1, 1, 1),
        "all": (1, 1, 1, 1, 1, 1),
    }
    receipts = {}
    responses = {}
    for mode, entries in masks.items():
        mask = jnp.asarray(entries, dtype=base_jets.dtype)
        tangent_program = jax.jit(
            lambda moments, vector, ops, reference, baseline, mask: jax.jvp(
                lambda state: map_with_mask(state, ops, reference, baseline, mask),
                (moments,),
                (vector,),
            )[1]
        )
        executable, compilation, wall = _time(
            tangent_program, initial, direction, operands, support, base_jets, mask
        )
        response = executable(initial, direction, operands, support, base_jets, mask)
        zero = executable(
            initial, jnp.zeros_like(direction), operands, support, base_jets, mask
        )
        amplitude = float(jnp.max(jnp.abs(response)))
        zero_amplitude = float(jnp.max(jnp.abs(zero)))
        if not np.isfinite(amplitude) or zero_amplitude > 1e-12:
            raise ValueError(f"{mode} tangent controls failed")
        receipts[mode] = dict(
            compile_seconds=compilation,
            wall_ms=wall,
            response_max=amplitude,
            zero_direction_response_max=zero_amplitude,
        )
        responses[mode] = response
        print(f"MODE {kind} {cells} {mode} {json.dumps(receipts[mode])}", flush=True)

    total = responses["all"]
    total_amplitude = float(jnp.max(jnp.abs(total)))
    if total_amplitude <= 1e-12:
        raise ValueError("combined jet tangent is empty")
    for mode in ("value", "gradient", "hessian"):
        receipts[mode]["relative_to_all"] = (
            float(jnp.max(jnp.abs(responses[mode] - total))) / total_amplitude
        )
        receipts[mode]["response_share"] = (
            receipts[mode]["response_max"] / total_amplitude
        )
    summed = responses["value"] + responses["gradient"] + responses["hessian"]
    sum_error = float(jnp.max(jnp.abs(summed - total))) / total_amplitude
    result = dict(
        kind=kind,
        requested_cells=cells,
        realised_cells=len(initial),
        provenance=provenance,
        reference_status=np.asarray(support.stationary.status).tolist(),
        reference_valid=bool(support.valid),
        modes=receipts,
        decomposition_relative_error=sum_error,
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
    print(f"NOVA_MODULE={nova.__file__}", flush=True)
    print(f"CWD={Path.cwd().resolve()} DEVICE={jax.devices()}", flush=True)
    run(args.kind, args.cells, args.cache, args.bias_row, args.newton_row, args.out)


if __name__ == "__main__":
    main()
