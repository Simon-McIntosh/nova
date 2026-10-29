"""Render the tracked read at the two trial states as poloidal contour panels.

Each panel draws the analytic flux and the read's own cell field on one shared
level set, with both null sets and the wall, using the repository's poloidal
painter. The read is the one the fast route consumes at that booking.
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

from nova.equilibrium import support_read

ROOT = Path(__file__).resolve().parents[3]
CELLS = 550
POLICY_RADIUS = 0.054806712567833996


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--cells", type=int, default=CELLS)
    args = parser.parse_args()

    from scripts.prototypes.tracked_read.measure import build_kind, cache_for
    from tests.equilibrium.test_topology_read import (
        _analytic_inputs,
        _realised_hex_geometry,
        _render_read_panel,
    )

    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    print(
        "# revision "
        + revision
        + " tree "
        + str(ROOT)
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
        reading = support_read.read(
            operands,
            cache,
            case["states"]["trial"],
            jnp.asarray(case["roots"]),
            jnp.asarray(case["roots"]),
            case["convention"],
            case["policy"],
            iterations=24,
        )
        jax.block_until_ready(reading)
        _, analytic, wall, axis, saddle = _analytic_inputs(kind)
        geometry = _realised_hex_geometry(wall, args.cells)
        _render_read_panel(
            args.directory,
            f"{kind}-trial",
            analytic,
            geometry,
            reading.reading,
            wall,
            np.asarray(axis),
            None if saddle is None else np.asarray(saddle),
            suffix="-read",
        )
        rows.append(
            dict(
                kind=kind,
                state="trial",
                valid=bool(reading.valid),
                qualified=bool(reading.reading.qualified),
                axis=[float(v) for v in np.asarray(reading.stationary.position)[0]],
                x_points=[
                    [float(v) for v in point]
                    for point in np.asarray(reading.reading.x_points)
                ],
                boundary_flux=float(reading.reading.boundary_flux),
                axis_flux=float(reading.reading.axis_flux),
            )
        )
        print("FIGURE_ROW " + json.dumps(rows[-1], sort_keys=True), flush=True)
    args.directory.mkdir(parents=True, exist_ok=True)
    print("RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()