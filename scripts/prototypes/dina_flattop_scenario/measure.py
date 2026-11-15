"""Measure sequential bank solves with the slice profiler's blocking timing loop."""

# ruff: noqa: E402
import argparse
from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path
from statistics import median
from time import perf_counter

from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import (
    current,
    drive_continuation as drive,
    support_read as sr,
    topology,
)
from nova.equilibrium.constraint import VerticalPositionPin, uniform_radial_field_jet
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.prototypes.dina_flattop_scenario.inputs import load_bank, write


class CompilationLog(logging.Handler):
    """Keep real dispatcher cache misses and completed backend compilations."""

    def __init__(self, path):
        super().__init__()
        self.events = []
        self.stream = path.open("w")

    def emit(self, record):
        message = record.getMessage()
        if "Compiling jit(" in message:
            self.events.append(message)
        self.stream.write(message + "\n")
        self.stream.flush()


def classify(result, reference, seconds, compiles):
    reading = result.support.reading
    stationary = result.support.stationary
    residual = np.asarray(result.residual)
    valid = np.asarray(result.support.valid)
    converged = (
        np.asarray(result.converged)
        & valid
        & np.isfinite(residual)
        & (residual <= 1e-10)
    )
    hessian = np.asarray(stationary.jet.hessian)
    determinants = np.linalg.det(hessian)
    classifications = np.where(
        determinants < 0,
        "saddle",
        np.where(hessian[..., 0, 0] > 0, "minimum", "maximum"),
    )
    expected = np.array(["minimum"] + ["saddle"] * (hessian.shape[1] - 1))
    stage = [
        "converged"
        if ok
        else "stationary_read"
        if not rvalid
        else "pin"
        if abs(error) > 1e-8
        else "fresh_residual"
        for ok, rvalid, error in zip(
            converged, valid, np.asarray(result.pin_error), strict=True
        )
    ]
    return dict(
        index=reference["index"],
        time_s=reference["time_s"],
        wall_ms=seconds * 1000,
        per_slice_ms=seconds * 1000 / len(residual),
        slices_per_second=len(residual) / seconds,
        residual=residual,
        converged=converged,
        failure_stage=stage,
        new_compilations=compiles,
        updates=np.asarray(result.steps),
        reads=np.asarray(result.reads),
        axis_m=np.asarray(reading.axis),
        xpoint_m=np.asarray(stationary.position)[:, 1],
        boundary_position_m=np.asarray(reading.boundary),
        axis_difference_m=np.asarray(reading.axis) - reference["axis"],
        xpoint_difference_m=np.asarray(stationary.position)[:, 1]
        - reference["xpoints"][0],
        boundary_flux_wb=np.asarray(reading.boundary_flux),
        boundary_flux_difference_wb=np.asarray(reading.boundary_flux)
        - reference["boundary_flux_wb"],
        ip_a=np.asarray(result.moments[0]).sum(axis=-1),
        ip_difference_a=np.asarray(result.moments[0]).sum(axis=-1) - reference["ip_a"],
        null_iterations=np.asarray(stationary.iterations),
        null_status=np.asarray(stationary.status),
        null_valid=np.asarray(stationary.valid),
        null_classification=classifications,
        dina_null_classification=expected,
        classification_agrees=classifications == expected,
        pin_amplitude=np.asarray(result.pin_amplitude),
        pin_error_m=np.asarray(result.pin_error),
        centroid_target_m=reference["centroid_m"],
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--controls", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert jax.config.jax_enable_x64 is True
    assert "H200" in jax.devices()[0].device_kind
    config = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    selection = json.loads((args.inputs / "selection.json").read_text())
    bank, window = load_bank(args.inputs / "bank")
    cv = topology.TopologyConvention.from_cocos(17, selection["slices"][0]["ip_a"])
    policy = TopologyPolicy()
    quadrature = current.Quadrature(sigma=cv.sigma, segments=64)
    source = [
        dict(np.load(args.inputs / f"source-{r['index']}.npz", allow_pickle=False))
        for r in selection["slices"]
    ]
    width = args.width

    def batch(value):
        return jnp.asarray(
            np.broadcast_to(value, (width, *np.shape(value))).copy(), dtype=jnp.float64
        )

    inputs = [
        (
            batch(s["currents"]),
            batch(s["pressure"]),
            batch(s["ff"]),
            batch(s["centroid"]),
        )
        for s in source
    ]
    seeds = batch(np.concatenate((source[0]["axis"][None], source[0]["xpoints"])))
    moments = tuple(batch(source[0]["moments"][:, i]) for i in range(3))
    amplitude = jnp.zeros(width, dtype=jnp.float64)
    tolerance = jnp.asarray(1e-10, dtype=jnp.float64)
    pin = VerticalPositionPin(0.0, source_scale=1e-4, row_scale=1.0, tolerance=1e-8)

    def solve(
        op,
        cache,
        state,
        roots,
        warm,
        amps,
        pressure,
        ff,
        target,
        pin_amplitude,
        tol,
        *,
        budget=8,
    ):
        return sr.solve_batch(
            op,
            cache,
            state,
            roots,
            warm,
            cv,
            policy,
            lambda lane: current.TabulatedProfile(pressure[lane], ff[lane]),
            None,
            quadrature,
            axis_sign=1.0,
            max_steps=budget,
            tolerance=tol,
            coil_currents=amps,
            vertical_pin=replace(
                pin, target_height=target, initial_unknown=pin_amplitude
            ),
        )

    program = jax.jit(solve)

    # The full read and residual are rebuilt independently of the solve receipt.
    @jax.jit
    def certify(op, cache, state, roots, amps, pressure, ff, target, pin_amplitude):
        def lane(i):
            driven = sr.driven_operands(op, amps[i])
            driven = replace(
                driven,
                exterior=driven.exterior
                + pin_amplitude[i]
                * pin.source_scale
                * uniform_radial_field_jet(driven.nodes),
            )
            cached = sr._driven_cache(driven, cache)
            local = tuple(m[i] for m in state)
            support = sr.read(
                driven, cached, local, roots[i], roots[i], cv, policy, axis_sign=1.0
            )
            residual = sr.step_residual(
                driven,
                local,
                local,
                support,
                current.TabulatedProfile(pressure[i], ff[i]),
                quadrature,
            )
            return (
                support.valid,
                jnp.max(jnp.abs(jnp.concatenate(residual.residual))),
                sr.nodal_jets(driven, local)[:, 0],
            )

        return jax.vmap(lane)(jnp.arange(width))

    logger = logging.getLogger("jax")
    capture = CompilationLog(args.output / "compilations.log")
    logger.addHandler(capture)
    logger.setLevel(logging.WARNING)
    backend = []

    def event(name, seconds, **metadata):
        if "backend_compile" in name:
            backend.append(dict(event=name, seconds=seconds, metadata=metadata))
            write(args.output / "backend-compilations.json", backend)

    jax.monitoring.register_event_duration_secs_listener(event)
    record = dict(
        revision=args.revision,
        module=sr.__file__,
        cwd=str(Path.cwd().resolve()),
        instrument_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        bank_digest=selection["bank_digest"],
        device=jax.devices()[0].device_kind,
        width=width,
        cache=str(config.directory),
        x64=True,
        complete=False,
        slices=[],
    )
    write(args.output / "receipt.json", record)
    print("HEADER", json.dumps(record), flush=True)
    previous_converged = False
    with jax.log_compiles(True):
        for ordinal, (reference, data) in enumerate(
            zip(selection["slices"], inputs, strict=True)
        ):
            before, backend_before = len(capture.events), len(backend)
            state_in, warm_in, amp_in = moments, seeds, amplitude
            started = perf_counter()
            attempts = []
            if ordinal and previous_converged:
                old_data = inputs[ordinal - 1]

                def at(fraction):
                    return tuple(
                        left + fraction * (right - left)
                        for left, right in zip(old_data, data, strict=True)
                    )

                def step(
                    op,
                    cache,
                    state,
                    roots,
                    warm,
                    convention,
                    rules,
                    packed,
                    gradient,
                    rule,
                    axis_sign,
                    **options,
                ):
                    result = jax.block_until_ready(
                        program(
                            op,
                            cache,
                            state,
                            roots,
                            warm,
                            *packed,
                            amp_in,
                            jnp.asarray(options["tolerance"], dtype=jnp.float64),
                        )
                    )
                    attempts.append(result)
                    return result

                continuation = drive.advance_drive(
                    bank,
                    window,
                    moments,
                    seeds,
                    seeds,
                    cv,
                    policy,
                    at,
                    None,
                    quadrature,
                    start_drive=0.0,
                    target_drive=1.0,
                    initial_relative_step=1.0,
                    max_substeps=8,
                    update_budget=8,
                    axis_sign=1.0,
                    solve=step,
                )
                result = continuation.result if continuation.certified else attempts[-1]
                continuation_note = dict(
                    certified=continuation.certified,
                    reached=continuation.drive,
                    reason=continuation.reason,
                    substeps=[list(s) for s in continuation.substeps],
                )
            else:
                result = jax.block_until_ready(
                    program(
                        bank, window, moments, seeds, seeds, *data, amplitude, tolerance
                    )
                )
                continuation_note = None
            first_seconds = perf_counter() - started
            solve_compiles = len(capture.events) - before
            compile_seconds = sum(e["seconds"] for e in backend[backend_before:])
            # Reuse the existing slice profiler's synchronized eleven-sample loop.
            # Each sample repeats the same input state; only the selected terminal
            # result enters the next physical slice's warm start.
            samples = []
            repeated = result
            for _ in range(11 if bool(np.all(np.asarray(result.converged))) else 0):
                started = perf_counter()
                repeated = jax.block_until_ready(
                    program(
                        bank,
                        window,
                        state_in,
                        warm_in,
                        warm_in,
                        *data,
                        amp_in,
                        tolerance,
                    )
                )
                samples.append(1000 * (perf_counter() - started))
            row = classify(result, reference, first_seconds, solve_compiles)
            row.update(
                first_call_s=first_seconds,
                compile_s=compile_seconds,
                warm_samples_ms=samples,
                replay_median_ms=median(samples) if samples else None,
                warm_start_from=None
                if ordinal == 0
                else selection["slices"][ordinal - 1]["index"],
                continuation=continuation_note,
                replay_residual=np.asarray(repeated.residual),
            )
            fresh_valid, fresh, flux = jax.block_until_ready(
                certify(
                    bank,
                    window,
                    result.moments,
                    result.support.stationary.position,
                    *data,
                    result.pin_amplitude,
                )
            )
            row["fresh_residual"] = np.asarray(fresh)
            row["fresh_valid"] = np.asarray(fresh_valid)
            row["converged"] &= (
                np.asarray(fresh_valid)
                & np.isfinite(np.asarray(fresh))
                & (np.asarray(fresh) <= 1e-10)
            )
            if continuation_note is not None and not continuation_note["certified"]:
                row["converged"][:] = False
                row["failure_stage"] = ["continuation" for _ in range(width)]
            for lane, ok in enumerate(row["converged"]):
                if not ok and row["failure_stage"][lane] == "converged":
                    row["failure_stage"][lane] = "fresh_certificate"
            row["all_compilations"] = len(capture.events) - before
            row["backend_compilations"] = len(backend) - backend_before
            write(args.output / f"slice-{reference['index']}.json", row)
            if width == 1:
                np.savez_compressed(
                    args.output / f"state-{reference['index']}.npz",
                    flux=np.asarray(flux[0])[: int(bank.active_nodes)],
                    nodes=np.asarray(bank.nodes)[: int(bank.active_nodes)],
                    moments=np.stack(
                        [np.asarray(m[0]) for m in result.moments], axis=-1
                    ),
                    axis=np.asarray(result.support.reading.axis[0]),
                    boundary=np.asarray(result.support.reading.boundary[0]),
                    stationary=np.asarray(result.support.stationary.position[0]),
                    null_valid=np.asarray(result.support.stationary.valid[0]),
                )
            record["slices"].append(row)
            write(args.output / "receipt.json", record)
            print(
                "SLICE",
                reference["index"],
                json.dumps(
                    dict(
                        residual=row["fresh_residual"],
                        converged=row["converged"],
                        stage=row["failure_stage"],
                        wall_ms=row["wall_ms"],
                        compiles=row["all_compilations"],
                    ),
                    default=lambda x: np.asarray(x).tolist(),
                ),
                flush=True,
            )
            moments, seeds, amplitude = (
                result.moments,
                result.support.stationary.position,
                result.pin_amplitude,
            )
            previous_converged = bool(np.all(row["converged"]))
            if not np.all(np.isfinite(np.asarray(moments))):
                record["stopped"] = (
                    "nonfinite terminal moments cannot seed the next slice"
                )
                break
        if args.controls:
            counts = []
            for data in inputs[:2]:
                captured = data[0]

                @jax.jit
                def closure_program(
                    op, cache, state, roots, pressure, ff, target, pin_amplitude, tol
                ):
                    return solve(
                        op,
                        cache,
                        state,
                        roots,
                        roots,
                        captured,
                        pressure,
                        ff,
                        target,
                        pin_amplitude,
                        tol,
                        budget=0,
                    )

                before = len(capture.events)
                jax.block_until_ready(
                    closure_program(
                        bank, window, moments, seeds, *data[1:], amplitude, tolerance
                    )
                )
                counts.append(len(capture.events) - before)
            data = inputs[-1]
            displaced = tuple(m * 1.01 for m in moments)
            zero = jax.jit(
                lambda op, cache, state, roots, amps, p, f, target, a, t: solve(
                    op, cache, state, roots, roots, amps, p, f, target, a, t, budget=0
                )
            )
            failed = jax.block_until_ready(
                zero(bank, window, displaced, seeds, *data, amplitude, tolerance)
            )
            control = classify(failed, selection["slices"][-1], 1.0, 0)
            converged = sum(bool(np.all(r["converged"])) for r in record["slices"])
            control.update(
                closure_compilations=counts,
                normal_fraction=converged / len(record["slices"]),
                with_failed_fraction=converged / (len(record["slices"]) + 1),
                refused=not bool(np.any(control["converged"]))
                and all(c > 0 for c in counts),
            )
            write(args.output / "negative-controls.json", control)
            record["negative_controls"] = control
            print(
                "NEGATIVE_CONTROLS",
                json.dumps(control, default=lambda x: np.asarray(x).tolist()),
                flush=True,
            )
    record.update(
        complete=True,
        converged_slices=sum(bool(np.all(r["converged"])) for r in record["slices"]),
        median_warm_ms=median(r["wall_ms"] for r in record["slices"][1:]),
        post_first_compilations=sum(
            r["all_compilations"] for r in record["slices"][1:]
        ),
    )
    write(args.output / "receipt.json", record)
    print(
        "SUMMARY",
        json.dumps(
            {
                k: record[k]
                for k in (
                    "complete",
                    "width",
                    "converged_slices",
                    "median_warm_ms",
                    "post_first_compilations",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
