"""Profile a neighbouring bank slice and the continuous carried read."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter


def closure(stages, wall):
    gap = (sum(stages.values()) - wall) / wall
    assert abs(gap) <= 0.10, f"stage closure gap {gap:.6%} exceeds 10%"
    return gap


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from nova.equilibrium import support_read as sr, drive_continuation as drive
    from scripts.prototypes.bank_newton_step.measure import machine
    from scripts.prototypes.bank_solve_profile.fixture import gradient
    from scripts.prototypes.bank_warm_start.streams import scaled_profile

    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == ("cpu" if args.smoke else "gpu")
    if not args.smoke:
        assert "H200" in jax.devices()[0].device_kind
    args.output.mkdir(exist_ok=True)
    record = dict(
        revision=args.revision,
        instrument_revision=args.instrument,
        instrument_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        module=sr.__file__,
        module_sha256=hashlib.sha256(Path(sr.__file__).read_bytes()).hexdigest(),
        inputs_path=str(args.inputs.resolve()),
        cwd=str(Path.cwd().resolve()),
        width=args.width,
        device=jax.devices()[0].device_kind,
        x64=True,
        complete=False,
        programs={},
    )

    def save():
        (args.output / "receipt.json").write_text(json.dumps(record, indent=2))

    print("HEADER", json.dumps(record), flush=True)
    save()
    bank, cache, roots, cv, policy, profile, rule, _, analytic = machine("diverted")
    source = np.load(args.inputs, allow_pickle=False)
    indices = np.arange(args.width) % len(source["m0"])
    state = tuple(jnp.asarray(source[f"m{i}"][indices]) for i in range(3))
    seeds = jnp.asarray(source["stationary"][indices])
    record["inputs_sha256"] = hashlib.sha256(args.inputs.read_bytes()).hexdigest()
    record["lane_source_indices"] = indices.tolist()

    # Name scopes add attribution metadata without changing numerical expressions.
    original = sr._rebuilt_reading

    def named_reading(*values, **kwargs):
        decision = values[6] if len(values) > 6 else kwargs.get("decision")
        with jax.named_scope(
            "full_geometry" if decision is None else "carried_geometry"
        ):
            return original(*values, **kwargs)

    sr._rebuilt_reading = named_reading

    def compile_program(label, fn, inputs):
        started = perf_counter()
        lowered = jax.jit(fn).lower(*inputs)
        program = lowered.compile()
        memory = program.memory_analysis()
        sizes = {
            name: int(getattr(memory, name, 0))
            for name in (
                "temp_size_in_bytes",
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "alias_size_in_bytes",
                "generated_code_size_in_bytes",
            )
        }
        working = (
            sizes["temp_size_in_bytes"]
            + sizes["argument_size_in_bytes"]
            + sizes["output_size_in_bytes"]
            - sizes["alias_size_in_bytes"]
        )
        stats = jax.devices()[0].memory_stats() or {}
        limit = stats.get("bytes_limit", 96 * 1024**3)
        record["programs"][label] = dict(
            compile_seconds=perf_counter() - started,
            memory=sizes,
            working_bytes=working,
            device_limit_bytes=limit,
            hlo_sha256=hashlib.sha256(program.as_text().encode()).hexdigest(),
        )
        save()
        print("COMPILED", label, record["programs"][label], flush=True)
        if not args.smoke and working > 0.90 * limit:
            record.update(memory_limited=True, limiting_program=label, complete=True)
            save()
            raise SystemExit(77)
        return program

    def timed(label, program, inputs):
        result = jax.block_until_ready(program(*inputs))
        samples = []
        for _ in range(1 if args.smoke else 11):
            started = perf_counter()
            result = jax.block_until_ready(program(*inputs))
            samples.append(1000 * (perf_counter() - started))
        record["programs"][label].update(samples_ms=samples, median_ms=median(samples))
        save()
        return result

    def capture(label, program, inputs):
        if args.smoke:
            return
        from scripts.prototypes.bank_solve_trace import analyse
        from scripts.prototypes.bank_solve_trace import groups
        from nova.equilibrium import topology

        groups.source_lines = lambda: Path(topology.__file__).read_text().splitlines()
        groups.normal_set_lines.cache_clear()
        group = groups.group

        if "profiler_control" not in record:
            host = np.arange(262144, dtype=np.float64).reshape(512, 512)
            matrix = jax.jit(lambda value: value @ value.T)
            jax.block_until_ready(matrix(jax.device_put(host)))
            control_options = jax.profiler.ProfileOptions()
            control_options.python_tracer_level = 0
            control_options.host_tracer_level = 1
            control_options.raise_error_on_start_failure = True
            with jax.profiler.trace(
                args.output / "control", profiler_options=control_options
            ):
                jax.block_until_ready(matrix(jax.device_put(host.copy())))
            events, _, _ = analyse.events(args.output / "control")
            transferred = sum(
                analyse.copy_bytes(e) for e in events if e["name"] == "MemcpyH2D"
            )
            kernels = sum("kernel_details" in e["stats"] for e in events)
            assert transferred == host.nbytes and kernels > 0, (transferred, kernels)
            record["profiler_control"] = dict(h2d_bytes=transferred, kernels=kernels)
            save()
        target = args.output / label
        target.mkdir(exist_ok=True)
        width = args.width
        for index, module in enumerate(program.runtime_executable().hlo_modules()):
            (target / f"solve-{width}-{index}.hlo.pb").write_bytes(
                module.as_serialized_hlo_module_proto()
            )
        (target / f"solve-{width}.hlo").write_text(program.as_text())
        options = jax.profiler.ProfileOptions()
        options.python_tracer_level = 0
        options.host_tracer_level = 1
        options.enable_hlo_proto = False
        options.raise_error_on_start_failure = True
        with jax.profiler.trace(target / f"trace-{width}", profiler_options=options):
            with jax.profiler.TraceAnnotation(f"solve_{width}_0"):
                started = perf_counter()
                jax.block_until_ready(program(*inputs))
                elapsed = 1000 * (perf_counter() - started)
        (target / "capture.json").write_text(
            json.dumps({width: dict(profiled_ms=[elapsed])})
        )

        def owner(item):
            names = {f["function"].split(".<locals>")[0] for f in item["stack"]}
            if "predict_seed" in names:
                return "predictor"
            if "carried_decision_changed" in names:
                return "carried read: incidence guard"
            if "step_jacobian" in names:
                return "assembly"
            if "step_residual" in names:
                return "update residual" if "newton_step" in names else "read residual"
            if "newton_step" in names:
                return "block GMRES and scaling"
            base = group(item)
            if "carried_geometry" in item["op_name"]:
                return "carried read: " + base
            if "full_geometry" in item["op_name"]:
                return "full read: " + base
            return base

        analyse.group = owner
        row = analyse.analyse(target, width)
        stages = {name: value["device_ms"] for name, value in row["groups"].items()}
        stages["transfer and device fills"] = sum(
            v["device_ms"] for v in row["copies"].values()
        )
        stages["host and device idle gaps"] = row["gap_ms"]
        stages["overlap correction"] = -row["overlap_ms"]
        row["stage_closure"] = closure(stages, elapsed)
        omitted = max(stages, key=stages.get)
        try:
            closure({k: v for k, v in stages.items() if k != omitted}, elapsed)
        except AssertionError as error:
            row["omission_refusal"] = str(error)
            print("OMISSION_REFUSED", label, omitted, str(error), flush=True)
        else:
            raise AssertionError("dominant stage omission escaped closure guard")
        record["programs"][label]["profile"] = row
        save()

    def read(b, c, m, s, tracking):
        return sr.read_batch(b, c, m, s, s, cv, policy, tracking=tracking)

    if args.read_only:
        full = compile_program("full", read, (bank, cache, state, seeds, None))
        support = timed("full", full, (bank, cache, state, seeds, None))
        tracking = sr.tracking_of(support)
        carried = compile_program(
            "carried", read, (bank, cache, state, seeds, tracking)
        )
        continued = timed("carried", carried, (bank, cache, state, seeds, tracking))
        assert np.all(np.asarray(continued.valid)) and not np.any(
            np.asarray(continued.rebuilt)
        )
        capture("full", full, (bank, cache, state, seeds, None))
        capture("carried", carried, (bank, cache, state, seeds, tracking))
    else:

        def solve(b, c, m, s, amplitude):
            return sr.solve_batch(
                b,
                c,
                m,
                s,
                s,
                cv,
                policy,
                scaled_profile(profile, amplitude),
                gradient,
                rule,
                max_steps=60,
                tolerance=1e-10,
                reuse_factor=False,
            )

        program = compile_program(
            "solve", solve, (bank, cache, state, seeds, jnp.asarray(1.0))
        )
        previous_state, amplitude = None, 1.0
        predictor = None
        for index in range(1, 4):
            target = 1.0 + index * 1e-5
            if index == 2:
                predictor = compile_program(
                    "predictor",
                    lambda m, p, d, pd, t: drive.predict_seed(
                        m, d, t, previous_moments=p, previous_drive=pd
                    ),
                    (state, previous_state, amplitude, amplitude - 1e-5, target),
                )
            predicted = (
                state
                if predictor is None
                else predictor(
                    state, previous_state, amplitude, amplitude - 1e-5, target
                )
            )
            result = jax.block_until_ready(
                program(bank, cache, predicted, seeds, jnp.asarray(target))
            )
            assert (
                np.all(np.asarray(result.converged))
                and np.max(np.asarray(result.residual)) <= 1e-10
            )
            if index == 3:
                predictor_inputs = (
                    state,
                    previous_state,
                    amplitude,
                    amplitude - 1e-5,
                    target,
                )
                solve_inputs = (bank, cache, predicted, seeds, jnp.asarray(target))
                timed("predictor", predictor, predictor_inputs)
                result = timed("solve", program, solve_inputs)
                walls = []
                for _ in range(1 if args.smoke else 11):
                    start = perf_counter()
                    predicted = predictor(*predictor_inputs)
                    result = jax.block_until_ready(
                        program(bank, cache, predicted, seeds, jnp.asarray(target))
                    )
                    walls.append(1000 * (perf_counter() - start))
                record["unprofiled_stage_closure"] = closure(
                    {
                        name: record["programs"][name]["median_ms"]
                        for name in ("predictor", "solve")
                    },
                    median(walls),
                )
                record.update(
                    end_to_end_ms=median(walls),
                    samples_ms=walls,
                    per_slice_ms=median(walls) / args.width,
                    reads=np.asarray(result.reads).tolist(),
                    updates=np.asarray(result.steps).tolist(),
                    residual=np.asarray(result.residual).tolist(),
                )
                save()
                capture("solve", program, solve_inputs)
                capture("predictor", predictor, predictor_inputs)
            previous_state, state, amplitude, seeds = (
                state,
                result.moments,
                target,
                result.support.stationary.position,
            )

        # An independent full read and booked residual must qualify every lane.
        def certificate(b, c, m, s):
            r = read(b, c, m, s, None)
            residual = jax.vmap(
                lambda x, y: sr.step_residual(
                    b, x, x, y, scaled_profile(profile, amplitude), rule
                )
            )(m, r)
            return r, jnp.max(
                jnp.abs(jnp.concatenate(residual.residual, axis=-1)), axis=-1
            )

        certify = compile_program(
            "certificate", certificate, (bank, cache, state, seeds)
        )
        support, residual = jax.block_until_ready(certify(bank, cache, state, seeds))
        assert (
            np.all(np.asarray(support.valid)) and np.max(np.asarray(residual)) <= 1e-10
        )
        from scripts.prototypes.tracked_read.measure import newton_reference
        from tests.equilibrium.test_topology_read import _analytic_inputs

        oracle, *_ = _analytic_inputs("diverted")
        axis, af = newton_reference(analytic, roots[0])
        saddle, bf = newton_reference(analytic, roots[1])
        reading = support.reading
        errors = dict(
            axis=np.linalg.norm(np.asarray(reading.axis) - axis, axis=-1)
            / float(oracle.minor_radius),
            saddle=np.linalg.norm(np.asarray(reading.boundary) - saddle, axis=-1)
            / float(oracle.minor_radius),
            boundary=np.abs(np.asarray(reading.boundary_flux) - bf) / abs(bf - af),
        )
        record["precision"] = {k: v.tolist() for k, v in errors.items()}
        record["fresh_residual"] = np.asarray(residual).tolist()
        save()
        assert np.max(errors["axis"]) <= 5e-4 and np.max(errors["saddle"]) <= 5e-4
        assert np.max(errors["boundary"]) <= 1e-4
    record["complete"] = True
    save()
    print(
        "SUMMARY",
        json.dumps(
            dict(
                complete=True, width=args.width, per_slice_ms=record.get("per_slice_ms")
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
