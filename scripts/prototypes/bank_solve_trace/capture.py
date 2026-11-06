"""Capture warmed production solve loops and preserve their compiler provenance."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
import subprocess
import time

import jax
import numpy as np

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_solve_profile import fixture


def emit(label, **values):
    print(label, json.dumps(values, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    emit(
        "HEADER",
        revision=revision,
        tree=str(root),
        cwd=str(Path.cwd().resolve()),
        nova=nova.__file__,
        module=support_read.__file__,
        devices=[str(device) for device in jax.devices()],
        x64=bool(jax.config.jax_enable_x64),
    )
    assert jax.config.jax_enable_x64 and jax.default_backend() == "gpu"
    batch = fixture.build(16)
    jax.block_until_ready(batch.states)
    programs, operands, rows = {}, {}, {}
    for lanes in (1, 16):

        @jax.jit
        def solve(bank, cache, state, seeds, previous):
            return support_read.solve_batch(
                bank,
                cache,
                state,
                seeds,
                previous,
                batch.convention,
                batch.policy,
                batch.profile,
                fixture.gradient,
                batch.quadrature,
                max_steps=60,
                reuse_factor=False,
                contraction_limit=0.5,
            )

        inputs = (
            batch.operands,
            batch.cache,
            tuple(value[:lanes] for value in batch.states),
            batch.seeds_batch[:lanes],
            batch.previous_batch[:lanes],
        )
        emit("COMPILING", lanes=lanes)
        started = time.perf_counter()
        compiled = solve.lower(*inputs).compile()
        (args.run / f"solve-{lanes}.hlo").write_text(compiled.as_text())
        modules = compiled.runtime_executable().hlo_modules()
        for index, module in enumerate(modules):
            (args.run / f"solve-{lanes}-{index}.hlo.pb").write_bytes(
                module.as_serialized_hlo_module_proto()
            )
        jax.block_until_ready(compiled(*inputs))
        emit("WARM", lanes=lanes, setup_seconds=time.perf_counter() - started)
        walls = []
        for _ in range(5):
            started = time.perf_counter()
            result = compiled(*inputs)
            jax.block_until_ready(result)
            walls.append(1000 * (time.perf_counter() - started))
        row = {
            "lanes": lanes,
            "unprofiled_ms": walls,
            **{
                name: np.asarray(getattr(result, name)).tolist()
                for name in ("steps", "reads", "refactors", "converged", "residual")
            },
        }
        assert np.all(np.asarray(result.converged))
        rows[lanes] = row
        programs[lanes], operands[lanes] = compiled, inputs
        emit("SOLVE", **row)
        (args.run / "capture.json").write_text(json.dumps(rows, indent=2))

    host = np.arange(262144, dtype=np.float64).reshape(512, 512)
    control_input = jax.device_put(host)
    control = jax.jit(lambda value: value @ value.T)
    jax.block_until_ready(control(control_input))
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 2
    options.enable_hlo_proto = True
    options.raise_error_on_start_failure = True
    with jax.profiler.trace(args.run / "trace", profiler_options=options):
        with jax.profiler.TraceAnnotation("instrument_control"):
            copied = jax.device_put(host.copy())
            jax.block_until_ready(control(copied))
        for lanes in (1, 16):
            rows[lanes]["profiled_ms"] = []
            for repeat in range(2):
                with jax.profiler.TraceAnnotation(f"solve_{lanes}_{repeat}"):
                    started = time.perf_counter()
                    result = programs[lanes](*operands[lanes])
                    jax.block_until_ready(result)
                    rows[lanes]["profiled_ms"].append(
                        1000 * (time.perf_counter() - started)
                    )
            emit("PROFILED", **rows[lanes])
    (args.run / "capture.json").write_text(json.dumps(rows, indent=2))
    emit("CAPTURE_COMPLETE", run=str(args.run), control_copy_bytes=host.nbytes)


if __name__ == "__main__":
    main()
