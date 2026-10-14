"""Assemble the implicit stationary-point tangent in moment coordinates."""

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
from nova.equilibrium import fixed_points, support_read


def _root_values(moments, window, roots, pitch, axis_sign):
    solved = fixed_points.solve(
        window, moments, roots, roots, pitch, axis_sign=axis_sign
    )
    return jnp.concatenate((solved.position.ravel(), solved.jet.value))


def _local_jet(moments, window, positions):
    field = fixed_points._CachedField(
        window.mesh, window.lookup, window.rows, moments, window.exterior
    )
    return jax.vmap(field.evaluate)(positions)


def _read_values(moments, operands, window, roots, convention, policy):
    reading = support_read.read(
        operands,
        window,
        tuple(moments[:, index] for index in range(3)),
        roots,
        roots,
        convention,
        policy,
        axis_sign=-convention.sigma,
    )
    return jnp.stack((reading.reading.axis_flux, reading.reading.boundary_flux))


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


def run(kind, cells, cache, bias_row, newton_row, out):
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "gpu"
    (
        initial,
        operands,
        window,
        roots,
        convention,
        policy,
        _profile,
        _quadrature,
        provenance,
    ) = _setup(kind, cells, cache, bias_row, newton_row)
    pitch = jnp.median(operands.geometry.pitch)
    axis_sign = -convention.sigma
    solved = fixed_points.solve(
        window, initial, roots, roots, pitch, axis_sign=axis_sign
    )
    if not bool(jnp.all(solved.valid)):
        raise ValueError(f"reference roots invalid: {np.asarray(solved.status)}")
    count = len(roots)
    positions = solved.position
    hessian = solved.jet.hessian

    def fixed_position_rows(moments, window, positions):
        jet = _local_jet(moments, window, positions)
        return jnp.concatenate((jet.gradient.ravel(), jet.value))

    rows_program = jax.jit(jax.jacrev(fixed_position_rows, argnums=0))
    rows_executable, rows_compile, rows_wall = _time(
        rows_program, initial, window, positions
    )
    rows = rows_executable(initial, window, positions)
    gradient_rows = rows[: 2 * count].reshape(count, 2, -1)
    root_rows = -jnp.linalg.solve(hessian, gradient_rows).reshape(
        count, 2, *initial.shape
    )
    flux_rows = rows[2 * count :].reshape(count, *initial.shape)
    if not bool(jnp.all(jnp.isfinite(root_rows))):
        raise ValueError("implicit root rows are nonfinite")

    scale = jnp.maximum(jnp.max(jnp.abs(initial[:, 0])), 1e-30)
    direction = jnp.asarray(np.random.default_rng(731).normal(size=initial.shape))
    direction *= scale * jnp.stack(
        (jnp.ones_like(operands.geometry.pitch),) + (operands.geometry.pitch,) * 2,
        axis=-1,
    )
    product = jnp.concatenate(
        (
            jnp.einsum("pcnm,nm->pc", root_rows, direction).ravel(),
            jnp.einsum("pnm,nm->p", flux_rows, direction),
        )
    )
    root_tangent = jax.jit(
        lambda moments, direction, window: jax.jvp(
            lambda state: _root_values(state, window, roots, pitch, axis_sign),
            (moments,),
            (direction,),
        )[1]
    )
    root_executable, root_compile, root_wall = _time(
        root_tangent, initial, direction, window
    )
    actual = root_executable(initial, direction, window)
    amplitude = float(jnp.max(jnp.abs(actual)))
    zero = root_executable(initial, jnp.zeros_like(direction), window)
    zero_response = float(jnp.max(jnp.abs(zero)))
    if not np.isfinite(amplitude) or amplitude < 1e-12 or zero_response > 1e-12:
        raise ValueError("implicit root tangent controls failed")
    identity_error = float(jnp.max(jnp.abs(actual - product))) / amplitude
    position_size = 2 * count
    position_amplitude = float(jnp.max(jnp.abs(actual[:position_size])))
    flux_amplitude = float(jnp.max(jnp.abs(actual[position_size:])))
    position_identity_error = float(
        jnp.max(jnp.abs(actual[:position_size] - product[:position_size]))
    ) / max(position_amplitude, 1e-30)
    flux_identity_error = float(
        jnp.max(jnp.abs(actual[position_size:] - product[position_size:]))
    ) / max(flux_amplitude, 1e-30)
    identity_valid = bool(
        np.isfinite(position_identity_error)
        and np.isfinite(flux_identity_error)
        and position_identity_error <= 1e-8
        and flux_identity_error <= 1e-8
    )

    differences = {}
    position_differences = {}
    flux_differences = {}
    statuses = {}
    for step in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        plus = fixed_points.solve(
            window, initial + step * direction, roots, roots, pitch, axis_sign=axis_sign
        )
        minus = fixed_points.solve(
            window, initial - step * direction, roots, roots, pitch, axis_sign=axis_sign
        )
        jax.block_until_ready((plus, minus))
        statuses[str(step)] = [
            np.asarray(plus.status).tolist(),
            np.asarray(minus.status).tolist(),
        ]
        central = (
            jnp.concatenate((plus.position.ravel(), plus.jet.value))
            - jnp.concatenate((minus.position.ravel(), minus.jet.value))
        ) / (2 * step)
        differences[str(step)] = float(jnp.max(jnp.abs(central - actual))) / amplitude
        position_differences[str(step)] = float(
            jnp.max(jnp.abs(central[:position_size] - actual[:position_size]))
        ) / max(position_amplitude, 1e-30)
        flux_differences[str(step)] = float(
            jnp.max(jnp.abs(central[position_size:] - actual[position_size:]))
        ) / max(flux_amplitude, 1e-30)

    read_tangent = jax.jit(
        lambda moments, direction, operands, window: jax.jvp(
            lambda state: _read_values(
                state, operands, window, roots, convention, policy
            ),
            (moments,),
            (direction,),
        )[1]
    )
    read_executable, read_compile, read_wall = _time(
        read_tangent, initial, direction, operands, window
    )
    read_response = read_executable(initial, direction, operands, window)
    read_zero = read_executable(initial, jnp.zeros_like(direction), operands, window)
    result = dict(
        kind=kind,
        requested_cells=cells,
        realised_cells=len(initial),
        provenance=provenance,
        reference_status=np.asarray(solved.status).tolist(),
        root_row_shape=list(root_rows.shape),
        flux_row_shape=list(flux_rows.shape),
        row_compile_seconds=rows_compile,
        row_assembly_wall_ms=rows_wall,
        root_jvp_compile_seconds=root_compile,
        root_jvp_wall_ms=root_wall,
        implicit_row_jvp_relative_error=identity_error,
        position_row_jvp_relative_error=position_identity_error,
        flux_row_jvp_relative_error=flux_identity_error,
        implicit_row_jvp_agrees=identity_valid,
        root_jvp_response_max=amplitude,
        position_response_max=position_amplitude,
        flux_response_max=flux_amplitude,
        zero_direction_response_max=zero_response,
        finite_difference_relative_errors=differences,
        position_finite_difference_relative_errors=position_differences,
        flux_finite_difference_relative_errors=flux_differences,
        finite_difference_statuses=statuses,
        read_global_jvp_compile_seconds=read_compile,
        read_global_jvp_wall_ms=read_wall,
        read_global_response=np.asarray(read_response).tolist(),
        read_global_zero_response=np.asarray(read_zero).tolist(),
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
    print(
        f"NOVA_MODULE={nova.__file__} FIXED_POINTS_MODULE={fixed_points.__file__}",
        flush=True,
    )
    print(f"CWD={Path.cwd().resolve()} DEVICE={jax.devices()}", flush=True)
    run(args.kind, args.cells, args.cache, args.bias_row, args.newton_row, args.out)


if __name__ == "__main__":
    main()
