"""Paired bank solves from common cold seeds and common warm-stream inputs."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
import hashlib
import os
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_newton_converge.fresh_solve import analytic_oracle
from scripts.prototypes.bank_newton_step.measure import machine
from scripts.prototypes.bank_solve_profile.fixture import gradient
from scripts.prototypes.bank_warm_start import streams
from tests.equilibrium.test_topology_read import _analytic_inputs


def emit(label, **values):
    print(
        label, json.dumps(values, default=lambda a: np.asarray(a).tolist()), flush=True
    )


def solve_program(parts):
    bank, cache, roots, convention, policy, profile, quadrature, _, _ = parts

    def solve(state, seeds, previous, amplitude, bank, cache):
        return support_read.solve_batch(
            bank,
            cache,
            state,
            seeds,
            previous,
            convention,
            policy,
            streams.scaled_profile(profile, amplitude),
            gradient,
            quadrature,
            max_steps=60,
            tolerance=1e-10,
            reuse_factor=False,
        )

    return jax.jit(solve)


def build_inputs(parts, path, lanes):
    bank, cache, roots, _, _, _, _, moments, _ = parts
    seed = roots + 0.03 * jnp.median(bank.geometry.pitch)
    scales = jnp.linspace(2e-5, 6e-5, 16)[:lanes]
    cold = tuple(value[None] * (1 + scales[:, None]) for value in moments)
    seeds = jnp.broadcast_to(seed, (lanes, *seed.shape))
    prior = jnp.broadcast_to(roots, seeds.shape)
    payload = {f"cold_m{i}": np.asarray(value) for i, value in enumerate(cold)}
    payload.update(cold_seeds=np.asarray(seeds), cold_previous=np.asarray(prior))
    program = solve_program(parts)
    root = jax.block_until_ready(program(cold, seeds, prior, 1.0, bank, cache))
    assert bool(jnp.all(root.converged)), np.asarray(root.residual)
    for step in (1e-5, 1e-3):
        state, points = root.moments, root.support.stationary.position
        for index in range(1, streams.SLICES):
            key = f"warm_{step}_{index}"
            payload.update(
                {f"{key}_m{i}": np.asarray(value) for i, value in enumerate(state)}
            )
            payload[f"{key}_seeds"] = np.asarray(points)
            payload[f"{key}_previous"] = np.asarray(points)
            result = jax.block_until_ready(
                program(state, points, points, 1 + index * step, bank, cache)
            )
            emit("INPUT", key=key, residual=result.residual, reads=result.reads)
            assert bool(jnp.all(result.converged)), (key, np.asarray(result.residual))
            state, points = result.moments, result.support.stationary.position
    np.savez(path, **payload)
    emit(
        "COMMON_INPUTS",
        path=str(path),
        keys=sorted(payload),
        lanes=lanes,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def input_rows(archive, lanes):
    labels = [("cold", 0.0, 0)] + [
        (f"warm_{step}_{index}", step, index)
        for step in (1e-5, 1e-3)
        for index in range(1, streams.SLICES)
    ]
    for key, step, index in labels:
        assert lanes == 16
        state = tuple(jnp.asarray(archive[f"{key}_m{i}"][:lanes]) for i in range(3))
        seeds = jnp.asarray(archive[f"{key}_seeds"][:lanes])
        previous = jnp.asarray(archive[f"{key}_previous"][:lanes])
        yield key, step, index, state, seeds, previous


def read_cost(parts, state, seeds, previous, output):
    """Profile standalone full and carried reads on the timed row's input."""
    from scripts.prototypes.bank_solve_trace.analyse import events

    bank, cache, _, convention, policy, *_ = parts

    def read(bank, cache, state, seeds, previous, tracking):
        return support_read.read_batch(
            bank, cache, state, seeds, previous, convention, policy, tracking=tracking
        )

    program = jax.jit(read)
    base = jax.block_until_ready(program(bank, cache, state, seeds, previous, None))
    tracking = support_read.tracking_of(base)
    costs = {}
    for label, tracked in (("full", None), ("carried", tracking)):
        if label == "carried" and not hasattr(
            __import__(
                "nova.equilibrium.topology", fromlist=["refresh_at_stationary_points"]
            ),
            "refresh_at_stationary_points",
        ):
            continue
        jax.block_until_ready(program(bank, cache, state, seeds, previous, tracked))
        options = jax.profiler.ProfileOptions()
        options.python_tracer_level = 0
        options.host_tracer_level = 1
        options.enable_hlo_proto = False
        options.raise_error_on_start_failure = True
        directory = output / label
        with jax.profiler.trace(directory, profiler_options=options):
            result = jax.block_until_ready(
                program(bank, cache, state, seeds, previous, tracked)
            )
        gpu, _, _ = events(directory)
        duration = sum(event["duration"] for event in gpu) / 1e6
        assert duration > 0, (label, "profiler saw no device work")
        costs[label] = dict(
            device_ms=duration, rebuilt=np.asarray(result.rebuilt).tolist()
        )
    return costs


def certificate_program(parts):
    _, _, _, convention, policy, profile, quadrature, *_ = parts

    def certify(values, seeds, previous, amplitude, bank, cache):
        terminal = support_read.read_batch(
            bank, cache, values, seeds, previous, convention, policy
        )
        residual = jax.vmap(
            lambda state, reading: support_read.step_residual(
                bank,
                state,
                state,
                reading,
                streams.scaled_profile(profile, amplitude),
                quadrature,
            )
        )(values, terminal)
        norm = jnp.max(jnp.abs(jnp.concatenate(residual.residual, axis=-1)), axis=-1)
        return terminal, norm

    return jax.jit(certify)


def counted_program(parts):
    """Count full and continuous branch executions outside timed calls.

    The diagnostic executable has callbacks, so its wall is never compared.
    Its lane update and trial counts must agree with the uninstrumented solve
    before its read census can accompany that solve's timing.
    """
    observed = []
    original = support_read._rebuilt_reading

    def count(*args, **kwargs):
        decision = args[6] if len(args) > 6 else kwargs.get("decision")
        jax.debug.callback(
            lambda full, point: observed.append(bool(full)),
            jnp.asarray(decision is None),
            args[1].solved.position[0, 0],
        )
        return original(*args, **kwargs)

    support_read._rebuilt_reading = count
    program = solve_program(parts)
    return program, observed, original, count


def memory_receipt(parts, archive, lanes, census):
    """Compile each runtime program before admitting a wider device batch."""
    bank, cache, *_ = parts
    _, step, index, state, seeds, previous = next(input_rows(archive, lanes))
    call = (state, seeds, previous, 1 + index * step, bank, cache)
    stats = jax.devices()[0].memory_stats() or {}
    limit = int(stats.get("bytes_limit", 16 * 1024**3))
    programs = {
        "solve": solve_program(parts),
        "certificate": certificate_program(parts),
    }
    original = None
    if census:
        diagnostic, _, original, counter = counted_program(parts)
        support_read._rebuilt_reading = original
        programs["census"] = diagnostic
    rows = {}
    for name, program in programs.items():
        if name == "census":
            support_read._rebuilt_reading = counter
        try:
            compiled = program.lower(*call).compile()
            memory = compiled.memory_analysis()
            entry = {
                key: int(getattr(memory, key, 0))
                for key in (
                    "temp_size_in_bytes",
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "alias_size_in_bytes",
                    "generated_code_size_in_bytes",
                )
            }
            working = (
                entry["temp_size_in_bytes"]
                + entry["argument_size_in_bytes"]
                + entry["output_size_in_bytes"]
                - entry["alias_size_in_bytes"]
            )
            entry["estimated_working_bytes"] = working
            entry["fits"] = working < int(0.9 * limit)
            rows[name] = entry
            emit(
                "COMPILED_MEMORY",
                lanes=lanes,
                program=name,
                device_limit_bytes=limit,
                wall_device="H200, width 16",
                **entry,
            )
        finally:
            if original is not None:
                support_read._rebuilt_reading = original
    return dict(
        lanes=lanes,
        wall_device="H200, width 16",
        programs=rows,
        device_limit_bytes=limit,
        fits=all(row["fits"] for row in rows.values()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--census", action="store_true")
    parser.add_argument("--widths", type=int, nargs="+", choices=[16], default=[16])
    parser.add_argument("--probe-memory", action="store_true")
    args = parser.parse_args()
    emit(
        "HEADER",
        revision=os.environ["MEASURE_REVISION"],
        nova=nova.__file__,
        module=support_read.__file__,
        cwd=str(Path.cwd().resolve()),
        devices=[str(device) for device in jax.devices()],
        device_kinds=[device.device_kind for device in jax.devices()],
        x64=jax.config.jax_enable_x64,
        cache=os.environ.get("JAX_COMPILATION_CACHE_DIR"),
        wall_device="H200, width 16",
        memory_fitting_level=jax.config.jax_memory_fitting_level,
        memory_fitting_effort=jax.config.jax_memory_fitting_effort,
        memory_stats=jax.devices()[0].memory_stats(),
    )
    assert jax.config.jax_enable_x64 and jax.default_backend() == "gpu"
    assert "H200" in jax.devices()[0].device_kind, jax.devices()[0].device_kind
    parts = machine("diverted")
    bank, cache, roots, convention, policy, profile, quadrature, _, analytic = parts
    if args.build:
        build_inputs(parts, args.inputs, args.widths[0])
        return
    inputs = np.load(args.inputs, allow_pickle=False)
    if args.probe_memory:
        capacity = [
            memory_receipt(parts, inputs, width, args.census) for width in args.widths
        ]
        args.output.write_text(
            json.dumps(dict(wall_device="H200, width 16", widths=capacity))
        )
        return
    program = solve_program(parts)
    certify = certificate_program(parts)
    rows = []
    payload = {
        "revision": os.environ["MEASURE_REVISION"],
        "wall_device": "H200, width 16",
        "device_kinds": [device.device_kind for device in jax.devices()],
        "rows": rows,
        "inputs_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
    }
    _, _, wall, _, _ = _analytic_inputs("diverted")
    diagnostic, observed, original, counter = (None, [], None, None)
    if args.census:
        diagnostic, observed, original, counter = counted_program(parts)
        support_read._rebuilt_reading = original
    for lanes in args.widths:
        for key, step, index, state, seeds, previous in input_rows(inputs, lanes):
            amplitude = 1 + index * step
            call = (state, seeds, previous, amplitude, bank, cache)
            result = jax.block_until_ready(program(*call))
            walls = []
            for _ in range(3):
                start = perf_counter()
                result = jax.block_until_ready(program(*call))
                walls.append((perf_counter() - start) * 1e3)
            # An independent full read at the returned state certifies the receipt.
            terminal, fresh_norm = jax.block_until_ready(
                certify(result.moments, seeds, previous, amplitude, bank, cache)
            )
            reading = result.support.reading
            oracle, oracle_roots = analytic_oracle(
                analytic, roots, jax.tree.map(lambda value: value[0], reading), wall
            )
            row = dict(
                lanes=lanes,
                wall_device="H200, width 16",
                key=key,
                step=step,
                slice=index,
                reads=np.asarray(result.reads).tolist(),
                updates=np.asarray(result.steps).tolist(),
                accepted_full_reads=np.asarray(result.support.rebuilds).tolist(),
                residual=np.asarray(fresh_norm).tolist(),
                converged=np.asarray(result.converged).tolist(),
                walls_ms=walls,
                wall_per_slice_ms=float(np.median(walls)) / lanes,
                flux=np.asarray(
                    jax.vmap(
                        lambda values: support_read.nodal_jets(bank, values)[:, 0]
                    )(result.moments)
                ).tolist(),
                axis=np.asarray(reading.axis).tolist(),
                saddle=np.asarray(reading.boundary).tolist(),
                boundary_flux=np.asarray(reading.boundary_flux).tolist(),
                span=oracle["flux_span"],
                minor_radius=oracle["minor_radius"],
                oracle=oracle,
                oracle_roots=oracle_roots.tolist(),
                peak_memory=jax.devices()[0].memory_stats(),
            )
            if diagnostic is not None:
                observed.clear()
                support_read._rebuilt_reading = counter
                counted = jax.block_until_ready(diagnostic(*call))
                jax.effects_barrier()
                support_read._rebuilt_reading = original
                row["census_matches"] = bool(
                    np.array_equal(counted.steps, result.steps)
                    and np.array_equal(counted.reads, result.reads)
                )
                full = sum(observed)
                carried = len(observed) - full
                # Every solve takes a known full read before any carry branch.
                assert full >= lanes, "read census missed the initial full read"
                assert len(observed) >= lanes * int(jnp.max(counted.reads))
                assert full % lanes == 0 and carried % lanes == 0
                row["full_reads"] = full // lanes
                row["carried_reads"] = carried // lanes
                row["diagnostic_reads"] = np.asarray(counted.reads).tolist()
                row["diagnostic_updates"] = np.asarray(counted.steps).tolist()
                emit(
                    "READ_CENSUS",
                    key=key,
                    lanes=lanes,
                    full=row["full_reads"],
                    carried=row["carried_reads"],
                    matches=row["census_matches"],
                )
            emit(
                "ROW",
                **{k: v for k, v in row.items() if k not in ("flux", "peak_memory")},
            )
            rows.append(row)
            args.output.write_text(json.dumps(payload))
            assert bool(jnp.all(result.converged)) and bool(jnp.all(terminal.valid))
            assert float(jnp.max(fresh_norm)) <= 1e-10, (key, np.asarray(fresh_norm))
        if args.profile:
            # Cold row at each width; costs are per standalone read, not a
            # synthetic attribution of whole-solve wall.
            key, step, index, state, seeds, previous = next(input_rows(inputs, lanes))
            payload.setdefault("read_device_cost", {})[lanes] = read_cost(
                parts,
                state,
                seeds,
                previous,
                args.output.parent / (args.output.stem + f"-read-{lanes}"),
            )
            args.output.write_text(json.dumps(payload))


if __name__ == "__main__":
    main()
