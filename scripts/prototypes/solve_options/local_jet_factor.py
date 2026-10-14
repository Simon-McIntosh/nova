"""Contract a validated cell-local Bell-jet derivative into moment space."""

# ruff: noqa: E402
import argparse
import json
from pathlib import Path
import resource
import subprocess
from time import perf_counter

from colored_jet_locality import _CellVaryingField
from live_iterations import _setup

import jax
import jax.numpy as jnp
import jax.scipy.linalg as dense
import numpy as np
import nova
from nova.equilibrium import current, fixed_points, support_read, topology


def _time(program, *args):
    started = perf_counter()
    executable = jax.jit(program).lower(*args).compile()
    compile_seconds = perf_counter() - started
    jax.block_until_ready(executable(*args))
    walls = []
    for _ in range(3):
        started = perf_counter()
        jax.block_until_ready(executable(*args))
        walls.append((perf_counter() - started) * 1000)
    return executable, compile_seconds, float(np.median(walls))


def run(kind, cells, cache, bias_row, newton_row, block_row, out):
    if jax.default_backend() != "gpu" or jax.config.jax_enable_x64 is not True:
        raise RuntimeError("local jet factor requires H200 double precision")
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
    receipt = json.loads(block_row.read_text())
    if (
        receipt["mode"] != "fixed-point-evaluation"
        or not receipt["local_reconstruction_agrees"]
        or receipt["provenance"]["machine_key"] != provenance["machine_key"]
        or receipt["realised_cells"] != len(initial)
    ):
        raise ValueError("local blocks do not match the qualified carrier")
    blocks = jnp.asarray(np.load(receipt["local_blocks_path"], allow_pickle=False))
    index = operands.sample_index
    if blocks.shape != (*index.shape, 6, 3):
        raise ValueError("local block shape differs from the cell stencil")
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
    fixed_field = fixed_points.field_from_jets(base_jets, index, operands.mesh)
    geometry = operands.geometry

    def booked(moments, target):
        jets = support_read.nodal_jets(
            operands, tuple(moments[:, component] for component in range(3))
        )
        bell = fixed_points.field_from_jets(jets, index, operands.mesh)
        reading = topology.read_at_stationary_points(
            _CellVaryingField(bell, fixed_field),
            geometry,
            convention,
            policy,
            support.stationary,
        )
        quad = current.Quadrature(
            sigma=quadrature.sigma,
            target_current=target,
            segments=quadrature.segments,
        )
        result = current.book(jets[index, 0], reading, geometry, profile, quad)
        return jnp.stack(result[:3], axis=-1)

    raw = booked(initial, None)
    raw_current = jnp.sum(raw[:, 0])
    target = quadrature.target_current
    if not bool(jnp.all(jnp.isfinite(raw))) or not bool(jnp.isfinite(raw_current)):
        raise ValueError("unnormalised base booking is nonfinite")

    def assemble(blocks, rows, index, raw, raw_current, target):
        local = jnp.einsum("ikqp,ikjmq->ipjm", blocks, rows[index])
        total_current_row = jnp.sum(local[:, 0], axis=0).ravel()
        scale = target / raw_current
        derivative = scale * local.reshape(raw.size, raw.size)
        derivative -= (scale / raw_current) * jnp.outer(raw.ravel(), total_current_row)
        return jnp.eye(raw.size, dtype=derivative.dtype) - derivative

    matrix_executable, matrix_compile, matrix_wall = _time(
        assemble,
        blocks,
        operands.rows,
        index,
        raw,
        raw_current,
        target,
    )
    matrix = matrix_executable(blocks, operands.rows, index, raw, raw_current, target)
    direction = jnp.asarray(np.random.default_rng(397).normal(size=initial.shape))
    direction *= jnp.maximum(jnp.max(jnp.abs(initial[:, 0])), 1e-30) * jnp.stack(
        (jnp.ones_like(geometry.pitch),) + (geometry.pitch,) * 2, axis=-1
    )

    def tangent(moments, vector):
        return jax.jvp(lambda state: booked(state, target), (moments,), (vector,))[1]

    tangent_executable, tangent_compile, tangent_wall = _time(
        tangent, initial, direction
    )
    actual = tangent_executable(initial, direction).ravel()
    zero = tangent_executable(initial, jnp.zeros_like(direction)).ravel()
    response = float(jnp.max(jnp.abs(actual)))
    zero_max = float(jnp.max(jnp.abs(zero)))
    predicted = direction.ravel() - matrix @ direction.ravel()
    relative_error = float(jnp.max(jnp.abs(actual - predicted))) / response
    valid = bool(
        np.isfinite(response)
        and response > 1e-12
        and zero_max == 0
        and np.isfinite(relative_error)
        and relative_error <= 1e-8
    )
    factor_wall = None
    factor_compile = None
    if valid:
        factor_executable, factor_compile, factor_wall = _time(dense.lu_factor, matrix)
        jax.block_until_ready(factor_executable(matrix))
    result = dict(
        kind=kind,
        requested_cells=cells,
        realised_cells=len(initial),
        provenance=provenance,
        block_source=str(block_row),
        block_reconstruction_relative_error=receipt[
            "local_reconstruction_relative_error"
        ],
        local_color_assembly_ms=receipt["assembly_wall_ms"],
        matrix_compile_seconds=matrix_compile,
        matrix_wall_ms=matrix_wall,
        matrix_bytes=int(matrix.size * matrix.dtype.itemsize),
        tangent_compile_seconds=tangent_compile,
        tangent_wall_ms=tangent_wall,
        independent_response_max=response,
        zero_direction_response_max=zero_max,
        moment_matrix_relative_error=relative_error,
        moment_matrix_agrees=valid,
        factor_compile_seconds=factor_compile,
        factor_wall_ms=factor_wall,
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
    parser.add_argument("--block-row", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"revision={revision} tree={Path.cwd().resolve()}", flush=True)
    print(f"NOVA_MODULE={nova.__file__}", flush=True)
    print(f"CWD={Path.cwd().resolve()} DEVICE={jax.devices()}", flush=True)
    run(
        args.kind,
        args.cells,
        args.cache,
        args.bias_row,
        args.newton_row,
        args.block_row,
        args.out,
    )


if __name__ == "__main__":
    main()
