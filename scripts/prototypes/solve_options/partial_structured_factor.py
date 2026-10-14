"""Factor a fixed-support Jacobian from local cells and current normalization."""

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
    compile_seconds = perf_counter() - started
    jax.block_until_ready(executable(*args))
    walls = []
    for _ in range(3):
        started = perf_counter()
        jax.block_until_ready(executable(*args))
        walls.append((perf_counter() - started) * 1000)
    return executable, compile_seconds, float(np.median(walls))


def run(kind, cells, cache, bias_row, newton_row, local_row, out):
    if jax.default_backend() != "gpu" or jax.config.jax_enable_x64 is not True:
        raise RuntimeError(
            "structured factor measurement requires H200 double precision"
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
    check = support_read.nodal_jets(
        operands, tuple(initial[:, component] for component in range(3))
    )[:, 0]
    span = json.loads(bias_row.read_text())["flux_span"]
    source_value_error = float(jnp.max(jnp.abs(check - nodal))) / span
    if source_value_error > 1e-10:
        raise ValueError(
            f"nodal source and analytic field disagree: {source_value_error}"
        )
    sample = nodal[index]
    coefficients = jnp.einsum("nij,nj->ni", geometry.fit_inverse, sample)
    unnormalised = current.book(
        sample,
        reading._replace(field_coefficients=coefficients),
        geometry,
        profile,
        current.Quadrature(sigma=local_convention.sigma),
    )
    if not bool(unnormalised.valid):
        raise ValueError("unnormalised analytic booking invalid")
    raw = jnp.stack(unnormalised[:3], axis=-1)
    raw_current = unnormalised.raw_current
    target = quadrature.target_current
    normalized = raw * (target / raw_current)
    conservation_error = float(
        jnp.abs(jnp.sum(normalized[:, 0]) - target) / jnp.abs(target)
    )
    if conservation_error > 1e-12:
        raise ValueError("current normalization control failed")

    def assemble(blocks, rows, index, raw, raw_current, target):
        sampled = rows[index, :, :, 0]
        local = jnp.einsum("ikp,ikjm->ipjm", blocks, sampled)
        sum_current = jnp.sum(local[:, 0], axis=0).ravel()
        scale = target / raw_current
        derivative = scale * local.reshape(raw.size, raw.size)
        derivative -= (scale / raw_current) * jnp.outer(raw.ravel(), sum_current)
        return jnp.eye(raw.size, dtype=derivative.dtype) - derivative

    matrix_program = jax.jit(assemble)
    matrix_executable, matrix_compile, matrix_wall = _time(
        matrix_program, blocks, operands.rows, index, raw, raw_current, target
    )
    matrix = matrix_executable(blocks, operands.rows, index, raw, raw_current, target)
    if not bool(jnp.all(jnp.isfinite(matrix))):
        raise ValueError("partial structured Jacobian is nonfinite")
    factor_program = jax.jit(dense.lu_factor)
    factor_executable, factor_compile, factor_wall = _time(factor_program, matrix)
    factors = factor_executable(matrix)
    jax.block_until_ready(factors)

    def fixed_residual(moments, operands, index):
        jets = support_read.nodal_jets(
            operands, tuple(moments[:, component] for component in range(3))
        )
        psi = jets[index, 0]
        coefficients = jnp.einsum("nij,nj->ni", geometry.fit_inverse, psi)
        frozen = reading._replace(field_coefficients=coefficients)
        booked = current.book(psi, frozen, geometry, profile, quadrature)
        return (moments - jnp.stack(booked[:3], axis=-1)).ravel()

    random = jnp.asarray(np.random.default_rng(731).normal(size=initial.shape))
    random *= jnp.maximum(jnp.max(jnp.abs(initial[:, 0])), 1e-30) * jnp.stack(
        (jnp.ones_like(geometry.pitch),) + (geometry.pitch,) * 2,
        axis=-1,
    )
    fixed_tangent = jax.jit(
        lambda moments, direction, operands, index: jax.jvp(
            lambda state: fixed_residual(state, operands, index),
            (moments,),
            (direction,),
        )[1]
    )
    fixed_executable, fixed_compile, fixed_wall = _time(
        fixed_tangent, initial, random, operands, index
    )
    predicted = matrix @ random.ravel()
    actual_fixed = fixed_executable(initial, random, operands, index)
    amplitude = float(jnp.max(jnp.abs(actual_fixed)))
    if not np.isfinite(amplitude) or amplitude < 1e-12:
        raise ValueError("fixed-support residual tangent is empty")
    fixed_error = float(jnp.max(jnp.abs(actual_fixed - predicted))) / amplitude
    booked_predicted = random.ravel() - predicted
    booked_fixed = random.ravel() - actual_fixed
    booked_amplitude = float(jnp.max(jnp.abs(booked_fixed)))
    if not np.isfinite(booked_amplitude) or booked_amplitude < 1e-12:
        raise ValueError("fixed-support booking tangent is empty")
    booked_error = (
        float(jnp.max(jnp.abs(booked_fixed - booked_predicted))) / booked_amplitude
    )
    zero = fixed_executable(initial, jnp.zeros_like(random), operands, index)
    zero_response = float(jnp.max(jnp.abs(zero)))
    if zero_response > 1e-12:
        raise ValueError("zero-direction residual tangent is nonzero")

    full_tangent = jax.jit(
        lambda moments, direction, operands, window: jax.jvp(
            lambda state: gs.moment_residual(
                state,
                operands,
                window,
                roots,
                convention,
                policy,
                profile,
                quadrature,
            )[0],
            (moments,),
            (direction,),
        )[1]
    )
    full_executable, full_compile, full_wall = _time(
        full_tangent, initial, random, operands, window
    )
    actual_full = full_executable(initial, random, operands, window).ravel()
    full_amplitude = float(jnp.max(jnp.abs(actual_full)))
    missing_global_relative = float(jnp.max(jnp.abs(actual_full - predicted))) / max(
        full_amplitude, 1e-30
    )
    booked_full = random.ravel() - actual_full
    missing_global_booking = float(
        jnp.max(jnp.abs(booked_full - booked_predicted))
    ) / max(float(jnp.max(jnp.abs(booked_full))), 1e-30)
    result = dict(
        kind=kind,
        requested_cells=cells,
        realised_cells=len(initial),
        provenance=provenance,
        local_source=str(local_row),
        local_color_assembly_ms=local_receipt["assembly_wall_ms"],
        source_value_relative_error=source_value_error,
        conservation_relative_error=conservation_error,
        matrix_compile_seconds=matrix_compile,
        matrix_assembly_wall_ms=matrix_wall,
        matrix_bytes=int(matrix.size * matrix.dtype.itemsize),
        factor_compile_seconds=factor_compile,
        actual_partial_factor_wall_ms=factor_wall,
        fixed_jvp_compile_seconds=fixed_compile,
        fixed_jvp_wall_ms=fixed_wall,
        fixed_jvp_response_max=amplitude,
        fixed_jvp_zero_response_max=zero_response,
        fixed_jvp_relative_error=fixed_error,
        fixed_booking_jvp_response_max=booked_amplitude,
        fixed_booking_jvp_relative_error=booked_error,
        fixed_booking_jvp_agrees=bool(booked_error <= 1e-8),
        full_jvp_compile_seconds=full_compile,
        full_jvp_wall_ms=full_wall,
        full_jvp_response_max=full_amplitude,
        omitted_global_relative_response=missing_global_relative,
        omitted_global_booking_relative_response=missing_global_booking,
        actual_full_structured_jacobian=False,
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
