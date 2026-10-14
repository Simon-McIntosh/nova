"""Verify completed fixed-support receipts against exterior target coordinates."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
import subprocess

import jax
import numpy as np
import nova
from scipy.spatial import cKDTree

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import topology
from scripts.prototypes.solve_options.profile import _machine


def validate(source, cache):
    row = json.loads(source.read_text())
    kind = row["kind"]
    cells = row["requested_cells"]
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, _, _ = certificate._case(name, clip_mode="exact")
    machine, key, _ = _machine(cache, carrier, cells)
    assert key == row["machine_key"]
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    tree = cKDTree(coordinates)
    samples = np.asarray(geometry.sample_points).reshape(-1, 2)
    distance, _ = tree.query(samples)
    maximum = float(np.max(distance))
    pitch = float(np.median(np.asarray(geometry.pitch)))
    control = np.min(coordinates, axis=0) - 10 * pitch
    control_distance = float(tree.query(control)[0])
    assert maximum <= 2e-11
    assert control_distance > pitch
    methods = {}
    for label, result in row["rows"].items():
        history = result["residual_history"]
        assert history and len(history) == result["iterations"]
        assert len(result["iteration_wall_ms"]) == len(history)
        methods[label] = dict(
            iterations=len(history),
            residual_history=history,
            iteration_wall_ms=result["iteration_wall_ms"],
            initial_residual=history[0],
            terminal_residual=history[-1],
            converged=result["converged"],
            flux_error=result["flux_error"],
        )
    return dict(
        source=str(source),
        kind=kind,
        realised_cells=row["realised_cells"],
        matched_sample_points=len(samples),
        maximum_exterior_target_distance_m=maximum,
        mismatch_control_distance_m=control_distance,
        median_pitch_m=pitch,
        methods=methods,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"revision={revision} tree={Path.cwd().resolve()}", flush=True)
    print(f"NOVA_MODULE={nova.__file__} CWD={Path.cwd().resolve()}", flush=True)
    assert jax.config.jax_enable_x64 is True
    common = Path(
        "/home/ITER/mcintos/.config/reckon/crew/runs/"
        "r-20261013T185617649894-moment-space-newton/"
        "device-forward-20261013T221023Z/support/cache"
    )
    diverted = Path(
        "/home/ITER/mcintos/.config/reckon/crew/runs/"
        "r-20261013T160532295843-carrier-requalify/rows/cache"
    )
    validation = []
    for cells in (2000, 5000):
        for kind in ("limited", "diverted"):
            source = args.rows / f"{kind}-{cells}.json"
            cache = diverted if kind == "diverted" and cells == 5000 else common
            validation.append(validate(source, cache))
            print("VALIDATED " + json.dumps(validation[-1]), flush=True)
    args.out.write_text(json.dumps(validation, indent=2) + "\n")


if __name__ == "__main__":
    main()
