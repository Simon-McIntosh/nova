"""Close the Newton step wall over residual, assembly, factor and back-solve."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
from unittest.mock import patch

import jax
import jax.numpy as jnp
from jax.scipy import linalg
import numpy as np

from scripts.prototypes.bank_solve_profile.receipts import NAMES, closure

spec = importlib.util.spec_from_file_location(
    "booking_measure", Path(__file__).with_name("measure.py")
)
measure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measure)
cycle, inputs, jacobians, production_mapping, sr, timer = (
    getattr(measure, name)
    for name in ("cycle", "inputs", "jacobians", "production_mapping", "sr", "timer")
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    if args.arm == "head":
        args.revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    cycle.reference.emit(
        "HEADER",
        revision=args.revision,
        nova=cycle.reference.nova.__file__,
        module=sr.__file__,
        cwd=str(Path.cwd().resolve()),
    )
    machine = cycle.reference.machine("diverted")
    bank, _, _, _, _, profile, quadrature, *_ = machine
    rows = []
    for lanes in (1, 16):
        states, support, booked = inputs(lanes, machine)
        times = {}

        def emit(kind, **data):
            cycle.reference.emit(kind, lanes=lanes, **data)
            if kind == "STAGE":
                times[data["name"]] = data["median_ms"]

        with patch.object(timer, "emit", emit):
            residual = timer.measure(
                "step residual",
                lambda b, m, s: production_mapping(
                    lambda x: sr.step_residual(
                        b, x[0], x[0], x[1], profile, quadrature
                    ),
                    (m, s),
                ),
                bank,
                states,
                support,
            )
            matrix = timer.measure(
                "step assembly",
                lambda b, m, s, p: jacobians(b, m, s, p, profile, quadrature),
                bank,
                states,
                support,
                booked,
            )

            def factor(a, m, p):
                def one(x):
                    balanced, row, column = sr._equilibrate(*x)
                    lu, pivots = linalg.lu_factor(balanced)
                    return row, column, lu, pivots

                return production_mapping(one, (a, m, p))

            factored = timer.measure("factor", factor, matrix, states, booked)

            def backsolve(a, factors, residuals):
                def one(x):
                    matrix, (row, column, lu, pivots), residual = x
                    right = -jnp.concatenate(residual)
                    solution = column * linalg.lu_solve((lu, pivots), right / row)
                    return solution, jnp.linalg.norm(
                        matrix @ solution - right
                    ) / jnp.linalg.norm(right)

                return production_mapping(one, (a, factors, residuals))

            timer.measure("back-solve", backsolve, matrix, factored, residual.residual)
            timer.measure(
                "whole step",
                lambda b, m, s: production_mapping(
                    lambda x: sr.newton_step(
                        b,
                        *x,
                        profile,
                        lambda radius, flux: jnp.zeros_like(radius),
                        quadrature,
                        fresh_geometry=True,
                        condition_diagnostics=False,
                    ),
                    (m, s),
                ),
                bank,
                states,
                support,
            )
        # The standalone Newton step consumes a read; it performs no read stage.
        values = {name: times.get(name, 0.0) for name in NAMES}
        row = dict(
            lanes=lanes, timings_ms=times, closure=closure(values, times["whole step"])
        )
        source = inspect.getsource(sr.step_jacobian)
        start = source.index("        point = support.reading.saddle_form.position")
        end = source.index("        def local_channel(index):")
        body = textwrap.dedent(source[start:end])
        namespace = dict(vars(sr))
        exec(
            "def stencil(operands, moments, support):\n    cells = moments[0].size\n"
            + textwrap.indent(body, "    ")
            + "    return count\n",
            namespace,
        )
        counts = jax.jit(
            lambda b, m, s: jax.vmap(lambda x: namespace["stencil"](b, *x))((m, s))
        )(bank, states, support)
        row["stencil_nodes"] = np.asarray(counts).tolist()
        assert np.all(np.asarray(counts) == 3)
        rows.append(row)
        (args.run / f"{args.arm}-step-profile.json").write_text(
            json.dumps(rows, indent=2)
        )
        cycle.reference.emit("CLOSURE", **row)
    if args.arm == "head":
        for name in ("timings", "solve"):
            shutil.copy2(
                args.run / f"head-{name}.json", args.run / f"bucket-only-{name}.json"
            )
        from scripts.prototypes.bank_step_booking_batch.control import run

        command = [
            sys.executable,
            str(Path(__file__).with_name("measure.py")),
            "--run",
            str(args.run),
            "--arm",
            "head",
            "--revision",
            args.revision,
            "--solve",
        ]
        assert (
            run(
                command,
                root,
                args.run / "device-head-rings.log",
                args.revision,
                BOOKING_SKIP_CONTROL="1",
            )
            == 0
        )


if __name__ == "__main__":
    main()
