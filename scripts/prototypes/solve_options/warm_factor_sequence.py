"""Track neighbouring vacuum-field slices with one cached exact Newton factor."""

# ruff: noqa: E402
import argparse
from dataclasses import replace
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
from nova.equilibrium import gs, support_read


def _compile(program, *args):
    started = perf_counter()
    executable = program.lower(*args).compile()
    return executable, perf_counter() - started


def run(kind, cells, cache, bias_row, newton_row, out):
    if jax.default_backend() != "gpu" or jax.config.jax_enable_x64 is not True:
        raise RuntimeError("warm tracking requires H200 double precision")
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
    span = json.loads(bias_row.read_text())["flux_span"]
    pitch = operands.geometry.pitch
    current_scale = jnp.maximum(jnp.max(jnp.abs(initial[:, 0])), 1e-30)
    scale = current_scale * jnp.stack((jnp.ones_like(pitch), pitch, pitch), axis=-1)

    def evaluate(state, ops, cache_window, seeds):
        moments = state.reshape(initial.shape) * scale
        residual, (support, booking) = gs.moment_residual(
            moments,
            ops,
            cache_window,
            seeds,
            convention,
            policy,
            profile,
            quadrature,
        )
        scaled = (residual / scale).ravel()
        return (
            scaled,
            support.stationary.position,
            support.stationary.status,
            support.valid & booking.valid,
        )

    def equation(state, ops, cache_window, seeds):
        return evaluate(state, ops, cache_window, seeds)[0]

    cold = jax.jit(
        lambda seed, ops, cache_window: gs._newton_moments(
            seed,
            ops,
            cache_window,
            roots,
            convention,
            policy,
            profile,
            quadrature,
            method="dense",
            steps=4,
            tolerance=1e-12,
        )
    )
    cold_executable, cold_compile = _compile(cold, initial, operands, window)
    started = perf_counter()
    cold_result = cold_executable(initial, operands, window)
    jax.block_until_ready(cold_result)
    cold_wall = (perf_counter() - started) * 1000
    if not bool(cold_result.converged):
        raise ValueError(
            f"analytic base solve did not converge: {float(cold_result.residual)}"
        )
    base_moments = cold_result.moments
    base_state = (base_moments / scale).ravel()

    evaluator = jax.jit(evaluate)
    evaluator_executable, evaluator_compile = _compile(
        evaluator, base_state, operands, window, roots
    )
    base_residual, base_positions, base_status, base_valid = evaluator_executable(
        base_state, operands, window, roots
    )
    jax.block_until_ready((base_residual, base_positions, base_status, base_valid))
    if not bool(base_valid) or float(jnp.max(jnp.abs(base_residual))) > 1e-12:
        raise ValueError("base solve failed a fresh live residual and root read")
    base_root_shift = float(
        jnp.max(jnp.linalg.norm(base_positions - roots, axis=-1)) / jnp.median(pitch)
    )
    analytic_jets = support_read.nodal_jets(
        operands, tuple(initial[:, index] for index in range(3))
    )
    solved_jets = support_read.nodal_jets(
        operands, tuple(base_moments[:, index] for index in range(3))
    )
    base_flux_error = (
        float(jnp.max(jnp.abs(solved_jets[:, 0] - analytic_jets[:, 0]))) / span
    )
    if (
        base_root_shift > 1
        or not np.isfinite(base_flux_error)
        or base_flux_error > 1e-2
    ):
        raise ValueError("base solve misses analytic root or mapped-flux qualification")

    matrix_program = jax.jit(
        lambda state, ops, cache_window, seeds: gs.dense_jacobian(
            lambda candidate: equation(candidate, ops, cache_window, seeds), state
        )
    )
    matrix_executable, matrix_compile = _compile(
        matrix_program, base_state, operands, window, base_positions
    )
    started = perf_counter()
    matrix = matrix_executable(base_state, operands, window, base_positions)
    jax.block_until_ready(matrix)
    matrix_wall = (perf_counter() - started) * 1000
    if not bool(jnp.all(jnp.isfinite(matrix))):
        raise ValueError("base matrix has nonfinite entries")

    factor_program = jax.jit(dense.lu_factor)
    factor_executable, factor_compile = _compile(factor_program, matrix)
    started = perf_counter()
    factors = factor_executable(matrix)
    jax.block_until_ready(factors)
    factor_wall = (perf_counter() - started) * 1000
    linear_program = jax.jit(dense.lu_solve)
    linear_executable, linear_compile = _compile(
        linear_program, factors, -base_residual
    )
    jax.block_until_ready(linear_executable(factors, -base_residual))

    zero_residual, _, zero_status, zero_valid = evaluator_executable(
        jnp.zeros_like(base_state), operands, window, roots
    )
    jax.block_until_ready((zero_residual, zero_status, zero_valid))
    zero_control = dict(
        valid=bool(zero_valid),
        status=np.asarray(zero_status).tolist(),
        residual=float(jnp.max(jnp.abs(zero_residual))),
    )

    radius = operands.nodes[:, 0]
    reference_radius = float(jnp.median(operands.geometry.centre[:, 0]))
    perturbation = span * jnp.stack(
        (
            (radius / reference_radius) ** 2 - 1,
            2 * radius / reference_radius**2,
            jnp.zeros_like(radius),
            jnp.full_like(radius, 2 / reference_radius**2),
            jnp.zeros_like(radius),
            jnp.zeros_like(radius),
        ),
        axis=-1,
    )
    lookup = np.asarray(window.lookup)
    selected = np.flatnonzero(lookup >= 0)
    window_perturbation = (
        jnp.zeros_like(window.exterior).at[lookup[selected]].set(perturbation[selected])
    )
    jax.block_until_ready(window_perturbation)

    state = base_state
    seeds = base_positions
    slices = []
    for amplitude in (1e-8, 1e-7, 1e-6):
        started = perf_counter()
        moved = replace(operands, exterior=operands.exterior + amplitude * perturbation)
        moved_window = replace(
            window, exterior=window.exterior + amplitude * window_perturbation
        )
        jax.block_until_ready((moved, moved_window))
        refresh_wall = (perf_counter() - started) * 1000
        histories = []
        corrections = 0
        started = perf_counter()
        for index in range(3):
            before = perf_counter()
            residual, positions, status, valid = evaluator_executable(
                state, moved, moved_window, seeds
            )
            jax.block_until_ready((residual, positions, status, valid))
            residual_wall = (perf_counter() - before) * 1000
            norm = float(jnp.max(jnp.abs(residual)))
            entry = dict(
                residual=norm,
                valid=bool(valid),
                status=np.asarray(status).tolist(),
                residual_wall_ms=residual_wall,
            )
            histories.append(entry)
            if not bool(valid) or not np.isfinite(norm) or norm <= 1e-12 or index == 2:
                break
            before = perf_counter()
            correction = linear_executable(factors, -residual)
            jax.block_until_ready(correction)
            entry["linear_solve_wall_ms"] = (perf_counter() - before) * 1000
            state = state + correction
            jax.block_until_ready(state)
            corrections += 1
        warm_wall = (perf_counter() - started) * 1000
        converged = bool(histories[-1]["valid"] and histories[-1]["residual"] <= 1e-12)
        slices.append(
            dict(
                vacuum_amplitude=amplitude,
                refresh_wall_ms=refresh_wall,
                warm_wall_ms=warm_wall,
                corrections=corrections,
                history=histories,
                converged=converged,
                root_shift_in_pitches=float(
                    jnp.max(jnp.linalg.norm(positions - base_positions, axis=-1))
                    / jnp.median(pitch)
                ),
            )
        )
        print(f"SLICE {kind} {json.dumps(slices[-1])}", flush=True)
        if not converged:
            break
        seeds = positions

    result = dict(
        kind=kind,
        requested_cells=cells,
        realised_cells=len(initial),
        provenance=provenance,
        exterior_change="vacuum vertical field proportional to R squared",
        reference_radius_m=reference_radius,
        flux_span=span,
        base_compile_seconds=cold_compile,
        base_cold_wall_ms=cold_wall,
        base_iterations=int(cold_result.iterations),
        base_residual=float(cold_result.residual),
        base_history=np.asarray(cold_result.residual_history).tolist(),
        base_status=np.asarray(base_status).tolist(),
        base_root_shift_in_pitches=base_root_shift,
        base_mapped_flux_error=base_flux_error,
        production_like_zero_control=zero_control,
        residual_compile_seconds=evaluator_compile,
        matrix_compile_seconds=matrix_compile,
        matrix_wall_ms=matrix_wall,
        matrix_bytes=int(matrix.size * matrix.dtype.itemsize),
        factor_compile_seconds=factor_compile,
        factor_wall_ms=factor_wall,
        triangular_solve_compile_seconds=linear_compile,
        slices=slices,
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
