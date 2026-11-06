"""Capture paired solve walls, linear-device attribution and physical precision."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import jax
import numpy as np

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_solve_profile import fixture
from scripts.prototypes.bank_solve_trace import analyse
from scripts.prototypes.bank_newton_converge.fresh_solve import analytic_oracle
from tests.equilibrium import test_topology_read as analytic_fixture


def emit(label, **values):
    print(label, json.dumps(values), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    emit(
        "HEADER",
        revision=args.revision,
        tree=str(Path(nova.__file__).parents[1]),
        cwd=str(Path.cwd().resolve()),
        nova=nova.__file__,
        module=support_read.__file__,
        devices=str(jax.devices()),
        x64=jax.config.jax_enable_x64,
    )
    assert jax.config.jax_enable_x64 and jax.default_backend() == "gpu"
    batch = fixture.build(16)
    _, analytic, wall, _, _ = analytic_fixture._analytic_inputs("diverted")
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 1
    options.enable_hlo_proto = False
    options.raise_error_on_start_failure = True
    host = np.arange(262144, dtype=np.float64).reshape(512, 512)
    control = jax.jit(lambda value: value @ value.T)
    jax.block_until_ready(control(jax.device_put(host)))
    with jax.profiler.trace(args.output / "control", profiler_options=options):
        with jax.profiler.TraceAnnotation("instrument_control"):
            jax.block_until_ready(control(jax.device_put(host.copy())))
    device, _, _ = analyse.events(args.output / "control")
    copied = sum(analyse.copy_bytes(e) for e in device if e["name"] == "MemcpyH2D")
    assert copied == host.nbytes and any("kernel_details" in e["stats"] for e in device)
    emit("POSITIVE_CONTROL", bytes=copied)
    rows, capture = [], {}
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
            tuple(v[:lanes] for v in batch.states),
            batch.seeds_batch[:lanes],
            batch.previous_batch[:lanes],
        )
        start = perf_counter()
        program = solve.lower(*inputs).compile()
        emit("COMPILE", lanes=lanes, seconds=perf_counter() - start)
        for index, module in enumerate(program.runtime_executable().hlo_modules()):
            (args.output / f"solve-{lanes}-{index}.hlo.pb").write_bytes(
                module.as_serialized_hlo_module_proto()
            )
        jax.block_until_ready(program(*inputs))
        walls = []
        for _ in range(5):
            start = perf_counter()
            result = jax.block_until_ready(program(*inputs))
            walls.append(1000 * (perf_counter() - start))
        row = dict(
            lanes=lanes,
            call=0,
            wall_ms=median(walls),
            per_slice_ms=median(walls) / lanes,
            samples_ms=walls,
        )
        row.update(
            {
                name: np.asarray(getattr(result, name)).tolist()
                for name in ("steps", "reads", "residual", "converged")
            }
        )
        reading = result.support.reading
        oracle, roots = analytic_oracle(
            analytic, batch.roots, jax.tree.map(lambda v: v[0], reading), wall
        )
        flux = jax.vmap(
            lambda state: support_read.nodal_jets(batch.operands, state)[:, 0]
        )(result.moments)
        row.update(
            axis=np.asarray(reading.axis).tolist(),
            saddle=np.asarray(reading.boundary).tolist(),
            boundary_flux=np.asarray(reading.boundary_flux).tolist(),
            span=oracle["flux_span"],
            minor_radius=oracle["minor_radius"],
            flux=np.asarray(flux).tolist(),
            oracle=oracle,
            oracle_roots=np.asarray(roots).tolist(),
        )
        emit("SOLVE", **{k: v for k, v in row.items() if k != "flux"})
        assert all(row["converged"]) and max(row["residual"]) <= 1e-10
        with jax.profiler.trace(
            args.output / f"trace-{lanes}", profiler_options=options
        ):
            with jax.profiler.TraceAnnotation(f"solve_{lanes}_0"):
                start = perf_counter()
                jax.block_until_ready(program(*inputs))
                profiled = 1000 * (perf_counter() - start)
        capture[str(lanes)] = dict(profiled_ms=[profiled])
        (args.output / "capture.json").write_text(json.dumps(capture))
        attribution = analyse.analyse(args.output, lanes)
        # The source classifier assigns setup and update exclusively, including
        # nested Krylov frames under newton_step without changing the classifier.
        groups = attribution["groups"]
        linear = sum(
            groups[k]["device_ms"]
            for k in ("equilibrate + lu_factor", "lu_solve + update")
        )
        assert linear > 0, "linear attribution must see known-present solve work"
        row.update(linear_device_ms=linear, attribution=attribution)
        rows.append(row)
        (args.output / "receipt.json").write_text(
            json.dumps(
                dict(revision=args.revision, module=support_read.__file__, rows=rows)
            )
        )
        emit("DEVICE", lanes=lanes, linear_ms=linear, total_ms=attribution["device_ms"])
        if hasattr(support_read, "_block_gmres"):
            for state_name, state in [
                ("trial", inputs[2]),
                ("converged", result.moments),
            ]:

                @jax.jit
                def linear_steps(bank, states):
                    reads = support_read.read_batch(
                        bank,
                        batch.cache,
                        states,
                        inputs[3],
                        inputs[4],
                        batch.convention,
                        batch.policy,
                    )
                    return jax.vmap(
                        lambda values, read: support_read.newton_step(
                            bank,
                            values,
                            read,
                            batch.profile,
                            fixture.gradient,
                            batch.quadrature,
                            fresh_geometry=True,
                        )
                    )(states, reads)

                step = jax.block_until_ready(linear_steps(batch.operands, state))
                diagnostics = {
                    name: np.asarray(getattr(step, name)).tolist()
                    for name in (
                        "krylov_condition",
                        "krylov_iterations",
                        "krylov_residual",
                    )
                }
                row[state_name] = diagnostics
                emit("LINEAR", lanes=lanes, state=state_name, **diagnostics)
                (args.output / "receipt.json").write_text(
                    json.dumps(
                        dict(
                            revision=args.revision,
                            module=support_read.__file__,
                            rows=rows,
                        )
                    )
                )
    emit("COMPLETE", rows=len(rows))


if __name__ == "__main__":
    main()
