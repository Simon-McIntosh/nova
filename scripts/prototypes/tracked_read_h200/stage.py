"""Attribute a tracked read's stationary-point error to the stage that carries it.

Three references are compared at each certificate state:

``read``
    what ``support_read.read`` returns.
``field``
    a damped Newton solve on the read's own derivative source (the Bell
    interpolant of the nodal jets), started from the analytic extremum. A
    distance from ``read`` is the solve's own error: seed, Newton tolerance and
    the line search.
``analytic
    a direct Newton solve on the analytic flux, sharing no code with any read.
    A distance from ``field`` is the derivative source's error: the interpolant
    on the nodal jets and the moments booked at that state.

A sweep of Newton iteration counts separates a tolerance that stops early from
a source that has converged to the wrong place.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import fixed_points, support_read

ROOT = Path(__file__).resolve().parents[3]
CELLS = 550
POLICY_RADIUS = 0.054806712567833996
SWEEP = (4, 8, 12, 24, 48)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=CELLS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from scripts.prototypes.tracked_read.measure import (
        build_kind,
        cache_for,
        newton_reference,
    )

    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    print(
        "# revision "
        + revision
        + " module "
        + support_read.__file__
        + " cwd "
        + str(Path.cwd().resolve())
        + " device "
        + str([str(d) for d in jax.local_devices()]),
        flush=True,
    )
    rows = []
    for kind in ("limited", "diverted"):
        case = build_kind(kind, args.cells, POLICY_RADIUS)
        operands = support_read.SupportOperands.from_rows(
            case["geometry"],
            case["matrix"],
            case["exterior"],
            case["nodes"],
            case["sample_index"],
            bucket=len(case["nodes"]),
        )
        cache = cache_for(operands)
        analytic_points = [
            newton_reference(case["analytic"], seed)[0] for seed in case["roots"]
        ]
        minor_radius = float(
            np.max(
                np.linalg.norm(
                    np.asarray(case["geometry"].sample_points).reshape(-1, 2)
                    - analytic_points[0],
                    axis=-1,
                )
            )
        )
        for state_name, moments in case["states"].items():
            field = fixed_points._CachedField(
                operands.mesh,
                jnp.arange(len(np.asarray(operands.nodes)), dtype=jnp.int32),
                operands.rows,
                jnp.stack(tuple(moments), axis=-1),
                operands.exterior,
            )
            field_points = []
            for seed in analytic_points:
                position = np.asarray(seed, dtype=float)
                for _ in range(80):
                    jet = field.evaluate(jnp.asarray(position))
                    step = np.linalg.solve(
                        np.asarray(jet.hessian), np.asarray(jet.gradient)
                    )
                    position = position - step
                    if np.linalg.norm(step) < 1e-15:
                        break
                field_points.append(position)
            for iterations in SWEEP:
                reading = support_read.read(
                    operands,
                    cache,
                    moments,
                    jnp.asarray(case["roots"]),
                    jnp.asarray(case["roots"]),
                    case["convention"],
                    case["policy"],
                    iterations=iterations,
                )
                jax.block_until_ready(reading)
                position = np.asarray(reading.stationary.position)
                rows.append(
                    dict(
                        kind=kind,
                        state=state_name,
                        iterations=iterations,
                        valid=bool(reading.valid),
                        solve_error_m=float(
                            np.linalg.norm(position[0] - field_points[0])
                        ),
                        source_error_m=float(
                            np.linalg.norm(field_points[0] - analytic_points[0])
                        ),
                        axis_error_m=float(
                            np.linalg.norm(position[0] - analytic_points[0])
                        ),
                        saddle_solve_error_m=(
                            None
                            if len(analytic_points) < 2
                            else float(
                                np.linalg.norm(position[1] - field_points[1])
                            )
                        ),
                        saddle_source_error_m=(
                            None
                            if len(analytic_points) < 2
                            else float(
                                np.linalg.norm(field_points[1] - analytic_points[1])
                            )
                        ),
                        budget_m=5e-4 * minor_radius,
                    )
                )
                print("STAGE " + json.dumps(rows[-1], sort_keys=True), flush=True)
    if args.output is not None:
        args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print("RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()