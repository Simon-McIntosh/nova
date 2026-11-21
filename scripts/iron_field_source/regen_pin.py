"""Rebuild a mesh's pinned linear operator and the drive receipt beside it.

The pinned operator under ``linear-magnetisation/west-h200`` is a pure function
of the mesh: the linear Galerkin cell interaction over the WEST iron region.
Where that receipt has been lost, the operator is rebuilt here and the drive
receipt records the solve that pins it, so nothing is asserted that was not
measured.

    python scripts/iron_field_source/regen_pin.py uniform-0.08
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import numpy as np  # noqa: E402

from nova.biot import magnetisation as iron  # noqa: E402
from scripts.iron_graded_refinement.measure import west_inputs  # noqa: E402
from scripts.iron_field_source import measure  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh")
    parser.add_argument("--operator-only", action="store_true")
    args = parser.parse_args()
    size = float(args.mesh.split("-")[1])
    assert jax.config.jax_enable_x64
    region, table_b, table_mu, _ = measure.west_region()
    started = time.perf_counter()
    cells = iron._mesh(region, size)
    matrix = iron.linear_galerkin_cell_interaction(
        cells.polygons, plan=iron._PLAN
    )[0]
    operator_seconds = time.perf_counter() - started
    measure.LINEAR.mkdir(parents=True, exist_ok=True)
    path = measure.LINEAR / f"{args.mesh}-operator.npy"
    iron.save_magnetisation_interaction(
        path, np.asarray(matrix), region=region, cell_size=size, basis="linear"
    )
    if args.operator_only:
        pin = dict(
            stage="pin",
            mesh=args.mesh,
            cells=len(cells.polygons),
            unknowns=6 * len(cells.polygons),
            operator_seconds=operator_seconds,
            device=jax.devices()[0].device_kind,
            backend=str(jax.default_backend()),
            x64=bool(jax.config.jax_enable_x64),
            operator_path=str(path),
            operator_bytes=int(path.stat().st_size),
            completed=False,
            full_drive=None,
            regenerated_from=(
                "rebuilt after the evidence-root prune; the drive waits on the "
                "WEST inputs pin owned by node icm-west-inputs-repin"
            ),
        )
        (measure.LINEAR / f"{args.mesh}.json").write_text(
            json.dumps(pin, indent=2) + "\n")
        print(json.dumps(pin), flush=True)
        return
    interaction = iron.load_magnetisation_interaction(
        path, region=region, cell_size=size, basis="linear"
    )
    inputs = west_inputs()
    started = time.perf_counter()
    result = iron.solve_magnetisation(
        region,
        inputs["external"],
        table_b,
        table_mu,
        cell_size=size,
        basis="linear",
        interaction=interaction,
        tolerance=1e-8,
        max_iterations=60,
    )
    solve_seconds = time.perf_counter() - started
    pin = dict(
        stage="pin",
        mesh=args.mesh,
        cells=len(cells.polygons),
        unknowns=6 * len(cells.polygons),
        operator_seconds=operator_seconds,
        solve_seconds=solve_seconds,
        device=jax.devices()[0].device_kind,
        backend=str(jax.default_backend()),
        x64=bool(jax.config.jax_enable_x64),
        operator_path=str(path),
        operator_bytes=int(path.stat().st_size),
        completed=True,
        full_drive=dict(
            converged=bool(result.converged),
            termination=result.termination,
            iterations=result.iterations,
            residual_history=[float(r) for r in result.residual_history],
        ),
        regenerated_from=(
            "rebuilt after the evidence-root prune of 2026-11-19T06:16Z"
        ),
    )
    (measure.LINEAR / f"{args.mesh}.json").write_text(
        json.dumps(pin, indent=2) + "\n"
    )
    print(json.dumps(pin), flush=True)


if __name__ == "__main__":
    main()