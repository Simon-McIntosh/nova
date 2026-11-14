"""Time runtime-operand bank programs and persist every completed stage."""

import argparse
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import re
import statistics
import sys
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--kind",
        choices=[
            "inputs",
            "read",
            "wall",
            "booking",
            "scalar",
            "stream",
            "lane",
            "fixture",
        ],
        required=True,
    )
    parser.add_argument("--width", type=int, choices=[1, 16], required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--driver", type=Path)
    parser.add_argument("--fixture-seed", type=Path)
    parser.add_argument("--step", type=float, default=1e-5)
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--population", choices=["diverted", "mixed"], default="diverted"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from nova.equilibrium import support_read as sr, drive_continuation as drive
    from scripts.prototypes.bank_newton_step.measure import machine
    from scripts.prototypes.bank_solve_profile.fixture import gradient
    from scripts.prototypes.bank_warm_start.streams import scaled_profile
    from scripts.prototypes.bank_warm_predictor.receipts import array_hash

    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == args.backend
    if args.backend == "gpu":
        assert "H200" in jax.devices()[0].device_kind
    assert Path(sr.__file__).resolve() == root / "nova/equilibrium/support_read.py"
    receipt = dict(
        revision=args.revision,
        module=sr.__file__,
        cwd=str(Path.cwd().resolve()),
        device=jax.devices()[0].device_kind,
        x64=True,
        kind=args.kind,
        arm=args.arm,
        width=args.width,
        complete=False,
        rows=[],
        population=args.population,
        smoke=args.smoke,
    )
    print("HEADER", json.dumps(receipt), flush=True)
    args.output.write_text(json.dumps(receipt, indent=2))
    parts = None if args.kind == "wall" else machine("diverted")
    if parts is not None:
        bank, cache, roots, convention, policy, profile, rule, moments, _ = parts
    if args.fixture_seed:
        receipt["fixture_seed_sha256"] = hashlib.sha256(
            args.fixture_seed.read_bytes()
        ).hexdigest()
        with np.load(args.fixture_seed, allow_pickle=False) as frozen:
            moments = tuple(jnp.asarray(frozen[f"m{i}"]) for i in range(3))
            roots = jnp.asarray(frozen["roots"])
        parts = (*parts[:2], roots, *parts[3:7], moments, parts[8])
    if args.kind == "scalar":
        rule = replace(rule, scalar_cut_batch_width=int(args.arm))
    from tests.equilibrium.test_topology_read import _analytic_inputs

    _, _, wall, _, _ = _analytic_inputs("diverted")
    receipt["minor_radius"] = float(np.ptp(np.asarray(wall)[:, 0]) / 2)
    width = args.width
    if args.kind == "wall":
        from tests.test_read_wall_skip import wall_case, batch_inputs

        classes = (
            [0] * width
            if args.population == "diverted"
            else [i % 2 for i in range(width)]
        )
        if args.population == "mixed" and width == 1:
            classes = [1]
        bank, cache, state, seeds, previous, convention, policy = batch_inputs(
            wall_case(550), classes
        )
        receipt["classes"] = classes
    else:
        scales = jnp.linspace(2e-5, 6e-5, 16)[:width]
        state = tuple(x[None] * (1 + scales[:, None]) for x in moments)
        seeds = jnp.broadcast_to(
            roots + 0.03 * jnp.median(bank.geometry.pitch), (width, *roots.shape)
        )
        previous = jnp.broadcast_to(roots, seeds.shape)
    receipt["bank_sha256"] = array_hash(jax.tree.leaves((bank, cache)))
    receipt["input_sha256"] = array_hash((*state, seeds, previous))
    initial_moments_sha256 = array_hash(state)
    receipt["instrument_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    arrays = {}
    if args.kind == "fixture":
        np.savez(
            args.output.with_suffix(".npz"),
            **{f"m{i}": np.asarray(x) for i, x in enumerate(moments)},
            roots=np.asarray(roots),
        )
        receipt["complete"] = True
        args.output.write_text(json.dumps(receipt, indent=2))
        print("SUMMARY canonical fixture exported", flush=True)
        return

    def persist():
        args.output.write_text(json.dumps(receipt, indent=2))
        if arrays:
            np.savez(args.output.with_suffix(".npz"), **arrays)

    def compiled(name, fn, values):
        started = perf_counter()
        lowered = jax.jit(fn).lower(*values)
        stable = str(lowered.compiler_ir())
        program = lowered.compile()
        seconds = perf_counter() - started
        hlo = program.as_text()
        operations = len(re.findall(r"^\s*%[^\n]* = ", hlo, re.M))
        assert operations > 0 and stable.count("stablehlo.") > 0
        row = dict(
            name=name,
            compile_seconds=seconds,
            lowered_sha256=hashlib.sha256(stable.encode()).hexdigest(),
            optimized_operations=operations,
            stable_operations=stable.count("stablehlo."),
            fusion_sites=len(re.findall(r"\bfusion\(", hlo)),
            while_sites=len(re.findall(r"\bwhile\(", hlo)),
        )
        receipt["rows"].append(row)
        persist()
        print("COMPILED", json.dumps(row), flush=True)
        return program, row

    def timed(name, fn, values):
        program, row = compiled(name, fn, values)
        result = jax.block_until_ready(program(*values))
        samples = []
        for _ in range(1 if args.smoke else 9):
            start = perf_counter()
            result = jax.block_until_ready(program(*values))
            samples.append(1000 * (perf_counter() - start))
        row.update(samples_ms=samples, median_ms=statistics.median(samples))
        persist()
        print("MEASURE", json.dumps(row), flush=True)
        return result

    def read(b, c, m, s, p, tracking):
        return sr.read_batch(b, c, m, s, p, convention, policy, tracking=tracking)

    initial = jax.block_until_ready(
        jax.jit(read)(bank, cache, state, seeds, previous, None)
    )
    assert bool(jnp.all(initial.valid))
    tracking = sr.tracking_of(initial)

    def geometry(label, support):
        reading = support.reading
        arrays[label + "-axis"] = np.asarray(reading.axis)
        arrays[label + "-saddle"] = np.asarray(reading.boundary)
        arrays[label + "-boundary"] = np.asarray(reading.boundary_flux)
        arrays[label + "-span"] = np.asarray(reading.boundary_flux - reading.axis_flux)
        assert np.all(np.asarray(support.valid))

    if args.kind in ("read", "wall", "scalar"):
        for kind, tracked in [("full", None), ("carried", tracking)]:
            if args.kind in ("read", "wall"):
                fn, values = read, (bank, cache, state, seeds, previous, tracked)
            else:

                def whole(b, c, m, s, p, tr, q):
                    support = read(b, c, m, s, p, tr)
                    booked = jax.vmap(
                        lambda x, r: sr.book(
                            b, r, sr.nodal_jets(b, x)[b.sample_index, 0], profile, q
                        )
                    )(m, support)
                    return support, booked

                fn, values = whole, (bank, cache, state, seeds, previous, tracked, rule)
            result = timed(kind, fn, values)
            support = result if args.kind in ("read", "wall") else result[0]
            geometry(kind, support)
            if args.kind == "scalar":
                assert bool(jnp.all(result[1].valid))
            assert bool(jnp.all(support.rebuilt == (kind == "full")))
    elif args.kind == "booking":
        from scripts.prototypes.bank_step_booking_batch.measure import inputs, jacobians

        state, support, booked = inputs(width, parts)
        receipt["timed_moments_sha256"] = array_hash(state)
        assert receipt["timed_moments_sha256"] == initial_moments_sha256
        matrix = timed(
            "assembly",
            lambda b, m, s, p: jacobians(
                b,
                m,
                s,
                p,
                profile,
                rule,
                mapping=lambda function, values: jax.vmap(function)(values),
            ),
            (bank, state, support, booked),
        )
        arrays["matrix"] = np.asarray(matrix)
        update = timed(
            "update",
            lambda b, m, s: jax.vmap(
                lambda x, r: sr.newton_step(
                    b, x, r, profile, gradient, rule, fresh_geometry=True
                )
            )(m, s),
            (bank, state, support),
        )
        assert bool(jnp.all(update.converged))
        np.testing.assert_allclose(update.factor.matrix, matrix, rtol=1e-12, atol=1e-12)
        receipt["assembly_matches_update_factor"] = True
        geometry("input", support)
    else:
        reuse = args.kind == "stream" and args.arm == "on"
        inputs = (bank, cache, convention, policy, rule)
        options = dict(max_steps=60, tolerance=1e-10, reuse_factor=reuse)
        options = {
            k: v
            for k, v in options.items()
            if k in inspect.signature(sr.solve_batch).parameters
        }

        def solve(b, c, cv, pol, q, m, s, p, amplitude, factor=None, valid=True):
            opt = dict(options)
            if reuse:
                opt.update(factor_cache=factor, factor_valid=valid)
            return sr.solve_batch(
                b,
                c,
                m,
                s,
                p,
                cv,
                pol,
                scaled_profile(profile, amplitude),
                gradient,
                q,
                **opt,
            )

        if args.kind == "inputs":
            result = timed(
                "cold", solve, (*inputs, state, seeds, previous, jnp.asarray(1.0))
            )
            assert (
                bool(jnp.all(result.converged))
                and float(jnp.max(result.residual)) <= 1e-10
            )
            np.savez(
                args.inputs,
                **{f"m{i}": np.asarray(x) for i, x in enumerate(result.moments)},
                stationary=np.asarray(result.support.stationary.position),
            )
        else:
            archive = np.load(args.inputs, allow_pickle=False)
            state = tuple(jnp.asarray(archive[f"m{i}"][:width]) for i in range(3))
            seeds = jnp.asarray(archive["stationary"][:width])
            receipt["input_sha256"] = array_hash((*state, seeds))
            if args.kind == "lane":
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    "measured_drive", args.driver
                )
                driver = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = driver
                spec.loader.exec_module(driver)
                steps = jnp.asarray(([1e-5, 1e-3] * 8)[:width])
                target = 1.0 + 7 * steps

                def lane_solve(b, c, cv, pol, q, m, s, p, amplitude, tolerance):
                    return sr.solve_batch(
                        b,
                        c,
                        m,
                        s,
                        p,
                        cv,
                        pol,
                        lambda lane: scaled_profile(profile, amplitude[lane]),
                        gradient,
                        q,
                        max_steps=driver.UPDATE_BUDGET,
                        tolerance=tolerance,
                        reuse_factor=False,
                    )

                call = (
                    *inputs,
                    state,
                    seeds,
                    seeds,
                    jnp.ones(width),
                    jnp.full((width,), 1e-10),
                )
                program, row = compiled("mixed-drive", lane_solve, call)
                jax.block_until_ready(program(*call))
                calls = []
                recording_seconds = [0.0]

                def adapter(
                    _b,
                    _c,
                    m,
                    s,
                    p,
                    _cv,
                    _pol,
                    amplitude,
                    _g,
                    _q,
                    _sign,
                    *,
                    tolerance,
                    **unused,
                ):
                    amplitudes = (
                        jnp.ones(width) + (amplitude - 1000.0) * steps
                        if args.arm == "off"
                        else amplitude
                    )
                    start = perf_counter()
                    result = jax.block_until_ready(
                        program(
                            *inputs,
                            m,
                            s,
                            p,
                            amplitudes,
                            jnp.broadcast_to(jnp.asarray(tolerance), (width,)),
                        )
                    )
                    elapsed = 1000 * (perf_counter() - start)
                    recording_start = perf_counter()
                    calls.append(
                        dict(
                            milliseconds=elapsed,
                            residual=np.asarray(result.residual).tolist(),
                            reads=np.asarray(result.reads).tolist(),
                            updates=np.asarray(result.steps).tolist(),
                        )
                    )
                    receipt["subsolves"] = calls
                    persist()
                    recording_seconds[0] += perf_counter() - recording_start
                    return result

                samples = []
                for sample in range(1 if args.smoke else 5):
                    receipt["sample"] = sample
                    calls.clear()
                    recording_seconds[0] = 0.0
                    started = perf_counter()
                    result = driver.advance_drive(
                        bank,
                        cache,
                        state,
                        seeds,
                        seeds,
                        convention,
                        policy,
                        lambda x: x,
                        gradient,
                        rule,
                        solve=adapter,
                        start_drive=1000.0 if args.arm == "off" else jnp.ones(width),
                        target_drive=1007.0 if args.arm == "off" else target,
                        initial_relative_step=1e-3 if args.arm == "off" else steps,
                        max_substeps=64,
                    )
                    elapsed = 1000 * (perf_counter() - started - recording_seconds[0])
                    assert result.certified
                    samples.append(elapsed)
                    row.update(
                        median_ms=statistics.median(samples),
                        samples_ms=samples,
                        execution_counts=dict(
                            reads=result.reads,
                            updates=result.updates,
                            subsolves=len(result.substeps),
                        ),
                    )
                    persist()
                terminal = result.result
                geometry("terminal", terminal.support)
                receipt["residual"] = np.asarray(terminal.residual).tolist()
                assert float(jnp.max(terminal.residual)) <= 1e-10
            else:
                previous_state, previous_drive = None, None
                amplitude = 1.0
                factor, valid = None, True
                program = None
                predictor = None
                for index in range(1, 3 if args.smoke else 8):
                    target = 1.0 + index * args.step
                    if index == 2:
                        predictor, _ = compiled(
                            "secant",
                            lambda m, p, d, pd, t: drive.predict_seed(
                                m, d, t, previous_moments=p, previous_drive=pd
                            ),
                            (state, previous_state, amplitude, previous_drive, target),
                        )
                    predicted = (
                        state
                        if previous_state is None
                        else predictor(
                            state,
                            previous_state,
                            amplitude,
                            previous_drive,
                            target,
                        )
                    )
                    call = (
                        *inputs,
                        predicted,
                        seeds,
                        seeds,
                        jnp.asarray(target),
                        factor,
                        valid,
                    )
                    signature = factor is not None
                    if program is None or (reuse and index == 2):
                        program, _ = compiled(
                            "solve-factor" if signature else "solve-fresh", solve, call
                        )
                        jax.block_until_ready(program(*call))
                    samples = []
                    for _ in range(1 if args.smoke else 9):
                        started = perf_counter()
                        predicted = (
                            state
                            if previous_state is None
                            else predictor(
                                state,
                                previous_state,
                                amplitude,
                                previous_drive,
                                target,
                            )
                        )
                        result = jax.block_until_ready(
                            program(
                                *inputs,
                                predicted,
                                seeds,
                                seeds,
                                jnp.asarray(target),
                                factor,
                                valid,
                            )
                        )
                        samples.append(1000 * (perf_counter() - started))
                    elapsed = statistics.median(samples)
                    assert (
                        bool(jnp.all(result.converged))
                        and float(jnp.max(result.residual)) <= 1e-10
                    )
                    geometry(str(index), result.support)
                    row = dict(
                        name=f"slice-{index}",
                        median_ms=elapsed,
                        samples_ms=samples,
                        per_slice_ms=elapsed / width,
                        residual=np.asarray(result.residual).tolist(),
                        reads=np.asarray(result.reads).tolist(),
                        updates=np.asarray(result.steps).tolist(),
                    )
                    receipt["rows"].append(row)
                    print("SLICE", json.dumps(row), flush=True)
                    persist()
                    previous_state, previous_drive = state, amplitude
                    state, amplitude, seeds = (
                        result.moments,
                        target,
                        result.support.stationary.position,
                    )
                    if reuse:
                        factor, valid = result.factor, result.factor_valid
    receipt["complete"] = True
    persist()
    print(
        "SUMMARY",
        json.dumps(dict(complete=True, rows=len(receipt["rows"]))),
        flush=True,
    )


if __name__ == "__main__":
    main()
