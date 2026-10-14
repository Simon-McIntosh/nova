"""Factor local booking blocks with live axis and boundary flux tangents."""

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
import jax.scipy.linalg as dense
import numpy as np
import nova
from nova.equilibrium import current, gs, support_read, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


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


def run(kind, cells, cache, bias_row, newton_row, local_row, out):
    if jax.default_backend() != "gpu" or jax.config.jax_enable_x64 is not True:
        raise RuntimeError("global structured factor requires H200 double precision")
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
    local_receipt = json.loads(local_row.read_text())
    assert local_receipt["machine_key"] == provenance["machine_key"]
    assert local_receipt["sampled_nodes"] == len(operands.nodes)
    blocks = jnp.asarray(
        np.load(local_receipt["local_blocks_path"], allow_pickle=False)
    )
    index = operands.sample_index
    assert blocks.shape == (*index.shape, 3)
    _, field, _, _, _ = _analytic_inputs(kind)
    field = certificate_field(field, kind)
    geometry = operands.geometry
    local_convention = topology.TopologyConvention.from_cocos(17, 1.0)
    reading = jax.jit(
        lambda: topology.read(field, geometry, local_convention, TopologyPolicy())
    )()
    assert bool(reading.valid) and bool(reading.qualified)
    nodal = jax.vmap(field.value)(operands.nodes)
    source = support_read.nodal_jets(
        operands, tuple(initial[:, component] for component in range(3))
    )[:, 0]
    span = json.loads(bias_row.read_text())["flux_span"]
    source_error = float(jnp.max(jnp.abs(source - nodal))) / span
    if source_error > 1e-10:
        raise ValueError(f"analytic nodal source mismatch: {source_error}")
    sample = nodal[index]
    coefficients = jnp.einsum("nij,nj->ni", geometry.fit_inverse, sample)
    frozen = reading._replace(field_coefficients=coefficients)
    unnormalised = current.book(
        sample,
        frozen,
        geometry,
        profile,
        current.Quadrature(sigma=local_convention.sigma),
    )
    if not bool(unnormalised.valid):
        raise ValueError("analytic booking invalid")
    raw = jnp.stack(unnormalised[:3], axis=-1)
    raw_current = unnormalised.raw_current
    target = quadrature.target_current

    def local_matrix(blocks, rows, index, raw, raw_current, target):
        sampled = rows[index, :, :, 0]
        local = jnp.einsum("ikp,ikjm->ipjm", blocks, sampled)
        sum_current = jnp.sum(local[:, 0], axis=0).ravel()
        scale = target / raw_current
        derivative = scale * local.reshape(raw.size, raw.size)
        derivative -= (scale / raw_current) * jnp.outer(raw.ravel(), sum_current)
        return jnp.eye(raw.size, dtype=derivative.dtype) - derivative

    local_program = jax.jit(local_matrix)
    local_executable, local_compile, local_wall = _time(
        local_program, blocks, operands.rows, index, raw, raw_current, target
    )
    partial = local_executable(blocks, operands.rows, index, raw, raw_current, target)

    def global_levels(moments, ops, cache_window):
        support = support_read.read(
            ops,
            cache_window,
            tuple(moments[:, component] for component in range(3)),
            roots,
            roots,
            convention,
            policy,
            axis_sign=-convention.sigma,
        )
        return jnp.stack((support.reading.axis_flux, support.reading.boundary_flux))

    global_program = jax.jit(jax.jacrev(global_levels, argnums=0))
    global_executable, global_compile, global_wall = _time(
        global_program, initial, operands, window
    )
    global_rows = global_executable(initial, operands, window).reshape(2, -1)
    if not bool(jnp.all(jnp.isfinite(global_rows))):
        raise ValueError("global level rows are nonfinite")

    def booking_at_levels(levels):
        varied = frozen._replace(axis_flux=levels[0], boundary_flux=levels[1])
        booked = current.book(sample, varied, geometry, profile, quadrature)
        return jnp.stack(booked[:3], axis=-1).ravel()

    levels = jnp.stack((reading.axis_flux, reading.boundary_flux))
    sensitivity_program = jax.jit(jax.jacfwd(booking_at_levels))
    sensitivity_executable, sensitivity_compile, sensitivity_wall = _time(
        sensitivity_program, levels
    )
    sensitivity = sensitivity_executable(levels)
    if not bool(jnp.all(jnp.isfinite(sensitivity))):
        raise ValueError("booking level sensitivity is nonfinite")

    matrix_program = jax.jit(lambda local, response, rows: local - response @ rows)
    matrix_executable, matrix_compile, matrix_wall = _time(
        matrix_program, partial, sensitivity, global_rows
    )
    matrix = matrix_executable(partial, sensitivity, global_rows)
    if not bool(jnp.all(jnp.isfinite(matrix))):
        raise ValueError("structured matrix is nonfinite")
    factor_program = jax.jit(dense.lu_factor)
    factor_executable, factor_compile, factor_wall = _time(factor_program, matrix)
    jax.block_until_ready(factor_executable(matrix))

    direction = jnp.asarray(np.random.default_rng(731).normal(size=initial.shape))
    direction *= jnp.maximum(jnp.max(jnp.abs(initial[:, 0])), 1e-30) * jnp.stack(
        (jnp.ones_like(geometry.pitch),) + (geometry.pitch,) * 2, axis=-1
    )

    def full_residual(moments, ops, cache_window):
        return gs.moment_residual(
            moments,
            ops,
            cache_window,
            roots,
            convention,
            policy,
            profile,
            quadrature,
        )[0].ravel()

    tangent_program = jax.jit(
        lambda moments, direction, ops, cache_window: jax.jvp(
            lambda state: full_residual(state, ops, cache_window),
            (moments,),
            (direction,),
        )[1]
    )
    tangent_executable, tangent_compile, tangent_wall = _time(
        tangent_program, initial, direction, operands, window
    )
    actual = tangent_executable(initial, direction, operands, window)
    predicted = matrix @ direction.ravel()
    partial_predicted = partial @ direction.ravel()
    booked_actual = direction.ravel() - actual
    response = float(jnp.max(jnp.abs(booked_actual)))
    if not np.isfinite(response) or response <= 1e-12:
        raise ValueError("full booking tangent is empty")
    error = float(jnp.max(jnp.abs(actual - predicted))) / response
    partial_error = float(jnp.max(jnp.abs(actual - partial_predicted))) / response
    zero = tangent_executable(initial, jnp.zeros_like(direction), operands, window)
    zero_response = float(jnp.max(jnp.abs(zero)))
    if zero_response > 1e-12:
        raise ValueError("zero-direction tangent is nonzero")
    result = dict(
        kind=kind,
        requested_cells=cells,
        realised_cells=len(initial),
        provenance=provenance,
        source_value_relative_error=source_error,
        local_color_assembly_ms=local_receipt["assembly_wall_ms"],
        local_matrix_compile_seconds=local_compile,
        local_matrix_wall_ms=local_wall,
        global_rows_compile_seconds=global_compile,
        global_rows_wall_ms=global_wall,
        booking_sensitivity_compile_seconds=sensitivity_compile,
        booking_sensitivity_wall_ms=sensitivity_wall,
        rank_two_update_compile_seconds=matrix_compile,
        rank_two_update_wall_ms=matrix_wall,
        matrix_bytes=int(matrix.size * matrix.dtype.itemsize),
        factor_compile_seconds=factor_compile,
        factor_wall_ms=factor_wall,
        full_jvp_compile_seconds=tangent_compile,
        full_jvp_wall_ms=tangent_wall,
        full_booking_jvp_response_max=response,
        zero_direction_response_max=zero_response,
        partial_booking_relative_error=partial_error,
        structured_booking_relative_error=error,
        full_jvp_agrees=bool(error <= 1e-8),
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
    parser.add_argument("--local-row", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"revision={revision} tree={Path.cwd().resolve()}", flush=True)
    print(f"NOVA_MODULE={nova.__file__} GS_MODULE={gs.__file__}", flush=True)
    print(f"CWD={Path.cwd().resolve()} DEVICE={jax.devices()}", flush=True)
    run(
        args.kind,
        args.cells,
        args.cache,
        args.bias_row,
        args.newton_row,
        args.local_row,
        args.out,
    )


if __name__ == "__main__":
    main()
