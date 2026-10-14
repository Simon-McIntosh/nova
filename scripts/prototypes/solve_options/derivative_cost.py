"""Time full moving-support moment JVPs and verify their response."""

# ruff: noqa: E402
import argparse
import json
from pathlib import Path
import subprocess
from time import perf_counter

from live_iterations import _setup

import jax
import jax.numpy as jnp
import numpy as np
import nova
from nova.equilibrium import gs


def _stencil_colors(index):
    cells = np.asarray(index, dtype=np.int64)
    count = int(np.max(cells)) + 1
    adjacent = [set() for _ in range(count)]
    for cell in cells:
        members = set(map(int, cell))
        for node in members:
            adjacent[node].update(members - {node})
    colors = [-1] * count
    for node in sorted(
        range(count), key=lambda item: len(adjacent[item]), reverse=True
    ):
        occupied = {colors[other] for other in adjacent[node] if colors[other] >= 0}
        color = 0
        while color in occupied:
            color += 1
        colors[node] = color
    for cell in cells:
        assert len({colors[int(node)] for node in cell}) == len(set(map(int, cell)))
    return max(colors) + 1


def _time(program, *args):
    start = perf_counter()
    executable = program.lower(*args).compile()
    compile_seconds = perf_counter() - start
    jax.block_until_ready(executable(*args))
    walls = []
    for _ in range(3):
        before = perf_counter()
        jax.block_until_ready(executable(*args))
        walls.append((perf_counter() - before) * 1000)
    return executable, compile_seconds, float(np.median(walls))


def run(kind, cells, cache, bias_row, newton_row, out):
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "gpu"
    (
        initial,
        operands,
        cache_window,
        roots,
        convention,
        policy,
        profile,
        quadrature,
        provenance,
    ) = _setup(kind, cells, cache, bias_row, newton_row)
    pitch = operands.geometry.pitch
    current_scale = jnp.maximum(jnp.max(jnp.abs(initial[:, 0])), 1e-30)
    scale = current_scale * jnp.stack((jnp.ones_like(pitch), pitch, pitch), axis=-1)
    random = jnp.asarray(np.random.default_rng(731).normal(size=initial.shape))
    direction = random * scale

    def residual(state, operands, cache_window):
        value, _ = gs.moment_residual(
            state,
            operands,
            cache_window,
            roots,
            convention,
            policy,
            profile,
            quadrature,
        )
        return value / scale

    primal = jax.jit(residual)
    tangent = jax.jit(
        lambda state, direction, operands, cache_window: jax.jvp(
            lambda candidate: residual(candidate, operands, cache_window),
            (state,),
            (direction,),
        )[1]
    )
    primal_executable, primal_compile, primal_wall = _time(
        primal, initial, operands, cache_window
    )
    tangent_executable, tangent_compile, tangent_wall = _time(
        tangent, initial, direction, operands, cache_window
    )
    derivative = tangent_executable(initial, direction, operands, cache_window)
    amplitude = float(jnp.max(jnp.abs(derivative)))
    if not np.isfinite(amplitude) or amplitude < 1e-12:
        raise ValueError("the nonzero moment direction produced no finite response")
    zero = tangent_executable(
        initial, jnp.zeros_like(direction), operands, cache_window
    )
    zero_response = float(jnp.max(jnp.abs(zero)))
    assert zero_response < 1e-12
    differences = {}
    for step in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        central = (
            primal_executable(initial + step * direction, operands, cache_window)
            - primal_executable(initial - step * direction, operands, cache_window)
        ) / (2 * step)
        differences[str(step)] = (
            float(jnp.max(jnp.abs(central - derivative))) / amplitude
        )
    result = dict(
        kind=kind,
        requested_cells=cells,
        realised_cells=len(initial),
        provenance=provenance,
        primal_compile_seconds=primal_compile,
        primal_wall_ms=primal_wall,
        jvp_compile_seconds=tangent_compile,
        jvp_wall_ms=tangent_wall,
        jvp_response_max=amplitude,
        zero_direction_response_max=zero_response,
        finite_difference_relative_errors=differences,
        local_stencil_colors=_stencil_colors(operands.sample_index),
        dense_jacobian_bytes=8 * initial.size**2,
        gpu_memory=jax.devices()[0].memory_stats(),
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
