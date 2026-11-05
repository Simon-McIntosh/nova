"""Time exact source regions of the Jacobian on captured production step inputs."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import inspect
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import textwrap
from time import perf_counter
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from scripts.prototypes.bank_solve_cycle import measure as cycle

sr = cycle.support_read
emit = cycle.reference.emit
ROOT = Path(__file__).resolve().parents[3]


def parts(profile, quadrature, destination):
    """Extract contiguous production expressions; fail if their anchors change."""
    source = inspect.getsource(sr.step_jacobian)

    def region(begin, end):
        assert source.count(begin) == source.count(end) == 1
        return textwrap.dedent(source[source.index(begin) : source.index(end)])

    common = "    cells = moments[0].size\n"
    local = region(
        "    if quadrature.target_current is None:", "    cells = moments[0].size"
    )
    bank = region(
        "    def bank_channels(index):", "    if quadrature.target_current is not None:"
    )
    matrix = region(
        "    if quadrature.target_current is not None:", "    if fresh_geometry:"
    )
    stencil = region(
        "        point = support.reading.saddle_form.position",
        "        def local_channel(index):",
    )
    channels = region(
        "        def local_channel(index):",
        '        response = jnp.einsum("kco,ki->coi", local, channels)',
    )
    closing = region(
        '        response = jnp.einsum("kco,ki->coi", local, channels)',
        "        matrix = matrix - response",
    )
    single = channels[: channels.index("channel_index =")]
    specs = {
        "local": ("", local, "local"),
        "bank": (", local", bank, "response"),
        "stencil": ("", stencil, "(nodes, count, channels)"),
        "channels": (
            ", nodes, count",
            channels.replace("stencil.size", "nodes.size"),
            "local",
        ),
        "closing": (", local, channels", closing, "response"),
        "geometry": ("", stencil + channels + closing, "response"),
        "matrix": (
            ", response, geometry_response",
            matrix,
            "matrix - geometry_response",
        ),
        "single": (", nodes, index", single, "local_channel(index)"),
    }
    namespace = dict(
        vars(sr),
        profile=profile,
        quadrature=quadrature,
        density_gradient=lambda radius, flux: jnp.zeros_like(radius),
    )
    generated = []
    for name, (extra, body, result) in specs.items():
        definition = (
            f"def {name}(operands, moments, support, booked{extra}):\n"
            + common
            + textwrap.indent(body, "    ")
            + f"    return {result}\n"
        )
        generated.append(definition)
        exec(compile(definition, f"<jacobian-{name}>", "exec"), namespace)
    destination.write_text("\n".join(generated))
    return {name: namespace[name] for name in specs}


def closure(total, whole):
    gap = total / whole - 1
    return {
        "sum_ms": total,
        "whole_ms": whole,
        "gap_fraction": gap,
        "within_15_percent": abs(gap) <= 0.15,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    assert args.repeats >= 10 and jax.config.jax_enable_x64
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    emit(
        "HEADER",
        revision=revision,
        cwd=str(Path.cwd().resolve()),
        nova=cycle.reference.nova.__file__,
        module=sr.__file__,
        backend=jax.default_backend(),
        devices=[str(d) for d in jax.devices()],
        x64=bool(jax.config.jax_enable_x64),
        job=os.environ.get("SLURM_JOB_ID"),
    )
    assert jax.default_backend() == "gpu"
    snapshots, machine = {}, []
    build = cycle.reference.machine

    def retain_machine(kind):
        result = build(kind)
        machine.append(result)
        return result

    def capture(state, support, steps, active):
        lanes = len(steps)
        index = int(np.max(steps))
        if index <= 1 and (lanes, index) not in snapshots:
            snapshots[(lanes, index)] = (state, support)
            emit("CAPTURE", lanes=lanes, step=index, active=active, steps=steps)

    source = inspect.getsource(sr.solve_batch)
    anchor = "        correction = jax.lax.cond("
    assert source.count(anchor) == 1
    source = source.replace(
        anchor,
        "        jax.debug.callback(capture, state, step_support, steps, active, "
        "ordered=True)\n"
        + anchor,
    )
    namespace = dict(vars(sr), capture=capture)
    exec(compile(source, "<captured-solve>", "exec"), namespace)
    with (
        patch.object(sr, "solve_batch", namespace["solve_batch"]),
        patch.object(cycle.reference, "machine", retain_machine),
        patch.object(
            sys,
            "argv",
            [
                "measure",
                "--output",
                str(args.output.with_name("solve.json")),
                "--revision",
                revision,
            ],
        ),
    ):
        cycle.main()
    jax.effects_barrier()
    bank, _, _, _, _, profile, quadrature, *_ = machine[0]
    functions = parts(profile, quadrature, args.output.with_name("extracted.py"))
    functions["whole"] = lambda operands, moments, support, booked: sr.step_jacobian(
        operands,
        moments,
        support,
        profile,
        lambda radius, flux: jnp.zeros_like(radius),
        quadrature,
        booked,
        fresh_geometry=True,
    )
    functions["booking"] = lambda operands, moments, support: (
        sr.step_residual(
            operands, moments, moments, support, profile, quadrature
        ).booked
    )
    receipt = {"revision": revision, "rows": [], "repeats": args.repeats}
    programs = {}
    for (lanes, step), (state, support) in sorted(snapshots.items()):
        state, support = jax.tree.map(jnp.asarray, (state, support))
        times = {}

        def call(name, *inputs, mapping="map", measure=True):
            key = (name, mapping, lanes)
            if key not in programs:
                function = functions[name]

                def mapped(operands, *values):
                    def one(items):
                        return function(operands, *items)
                    return (
                        jax.lax.map(one, values)
                        if mapping == "map"
                        else jax.vmap(one)(values)
                    )

                programs[key] = jax.jit(mapped)
            program = programs[key]
            start = perf_counter()
            value = jax.block_until_ready(program(bank, *inputs))
            emit(
                "WARM",
                lanes=lanes,
                step=step,
                part=name,
                mapping=mapping,
                seconds=perf_counter() - start,
            )
            if measure:
                samples = []
                for _ in range(args.repeats):
                    start = perf_counter()
                    value = jax.block_until_ready(program(bank, *inputs))
                    samples.append(1000 * (perf_counter() - start))
                times[name if mapping == "map" else name + "_vmap"] = median(samples)
                emit(
                    "TIMING",
                    lanes=lanes,
                    step=step,
                    part=name,
                    mapping=mapping,
                    median_ms=median(samples),
                    samples_ms=samples,
                )
            return value

        booked = call("booking", state, support, measure=False)
        inputs = (state, support, booked)
        whole = call("whole", *inputs)
        local = call("local", *inputs)
        response = call("bank", *inputs, local)
        nodes, count, channels = call("stencil", *inputs)
        tangent = call("channels", *inputs, nodes, count)
        geometry = call("closing", *inputs, tangent, channels)
        fresh = call("geometry", *inputs)
        assembled = call("matrix", *inputs, response, geometry)
        call("single", *inputs, nodes, jnp.zeros(lanes, dtype=jnp.int32))
        call("whole", *inputs, mapping="vmap")
        error = float(jnp.linalg.norm(assembled - whole) / jnp.linalg.norm(whole))
        geometry_error = float(
            jnp.linalg.norm(geometry - fresh) / jnp.linalg.norm(fresh)
        )
        assert np.isfinite(error) and error < 1e-10, error
        assert np.isfinite(geometry_error) and geometry_error < 1e-10, geometry_error
        active_cells = np.sum(
            np.asarray(
                (support.reading.membership > 0)
                & (support.reading.membership < 1 - 1e-14)
                & support.reading.normal_form_cells
            ),
            axis=-1,
        )
        assert np.all(active_cells > 0) and np.all(np.asarray(count) > 0)
        top = sum(times[name] for name in ("local", "bank", "geometry", "matrix"))
        detailed = sum(
            times[name]
            for name in ("local", "bank", "stencil", "channels", "closing", "matrix")
        )
        control = closure(top - times["geometry"], times["whole"])
        assert not control["within_15_percent"], control
        row = dict(
            lanes=lanes,
            step=step,
            timings_ms=times,
            active_stencil_channels=(6 * np.asarray(count)).tolist(),
            executed_stencil_channels=6 * nodes.shape[-1],
            active_normal_cells=active_cells.tolist(),
            matrix_relative_error=error,
            geometry_relative_error=geometry_error,
            closure=closure(top, times["whole"]),
            detailed_closure=closure(detailed, times["whole"]),
            geometry_closure=closure(
                sum(times[n] for n in ("stencil", "channels", "closing")),
                times["geometry"],
            ),
            omitted_geometry_control=control,
        )
        receipt["rows"].append(row)
        args.output.write_text(json.dumps(receipt, indent=2))
        emit("SUMMARY", **row)


if __name__ == "__main__":
    main()
