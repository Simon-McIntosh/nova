"""Persist and attribute whole-mesh sheet-tile discrepancies."""

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import sys

import numpy as np

import nova.biot.tiledassembly as assembly

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
import test_sheet_tile as gate


def build(bounds):
    """Capture the production edge fields and the host reference for one shard."""
    held = {}
    original = assembly.sheet_edge_field
    original_reference = gate.unbatched_edge_field

    def capture(*args, **kwargs):
        values = original(*args, **kwargs)
        held["fields"] = values
        return values

    def capture_reference(*args, **kwargs):
        values = original_reference(*args, **kwargs)
        held["reference"] = values
        return values

    assembly.sheet_edge_field = capture
    gate.unbatched_edge_field = capture_reference
    try:
        lower, upper, tiled, reference, scale = gate.build_slice(bounds)
    finally:
        assembly.sheet_edge_field = original
        gate.unbatched_edge_field = original_reference
    polygons, centres = gate.west_mesh()
    edges = assembly.cell_edges(polygons)
    radial, height = held["reference"]
    differences = (held["fields"][0] - radial, held["fields"][1] - height)
    error = np.abs(tiled - reference) / scale
    rows, columns = np.where(error > gate.GATE)
    targets = lower + rows % (upper - lower)
    components = rows // (upper - lower)
    cells = columns % len(polygons)
    inputs = columns // len(polygons)
    records, attributed = [], []
    for row, column, target, component, cell, source_component in zip(
        rows, columns, targets, components, cells, inputs, strict=True
    ):
        coefficient = (edges.coefficient_r, edges.coefficient_z)[source_component][cell]
        active = np.flatnonzero(coefficient)
        contributions = (
            differences[component][target - lower, active] * coefficient[active]
        )
        dominant = int(active[np.argmax(np.abs(contributions))])
        records.append(
            (
                row,
                column,
                target,
                component,
                cell,
                source_component,
                error[row, column],
                scale[row, column],
                tiled[row, column],
                reference[row, column],
                dominant,
            )
        )
        for edge, contribution in zip(active, contributions, strict=True):
            dr = edges.end_r[edge] - edges.start_r[edge]
            dz = edges.end_z[edge] - edges.start_z[edge]
            length = np.hypot(dr, dz)
            displacement = centres[target] - [edges.start_r[edge], edges.start_z[edge]]
            along = np.dot(displacement, [dr, dz]) / length**2
            line_distance = abs(displacement[0] * dz - displacement[1] * dr) / length
            distance = np.hypot(
                *(displacement - np.clip(along, 0, 1) * np.array([dr, dz]))
            )
            route = "pancake" if dz == 0 else "cylinder" if dr == 0 else "sloped"
            attributed.append(
                {
                    "target": int(target),
                    "cell": int(cell),
                    "component": int(component),
                    "input": int(source_component),
                    "edge": int(edge),
                    "route": route,
                    "contribution": float(contribution),
                    "dominant": bool(edge == dominant),
                    "distance_over_length": float(distance / length),
                    "line_distance_over_length": float(line_distance / length),
                    "plane_distance_over_length": float(abs(displacement[1]) / length),
                    "slope_dr_dz": float(dr / dz) if dz != 0 else None,
                    "target_radius_over_length": float(centres[target, 0] / length),
                    "start": [float(edges.start_r[edge]), float(edges.start_z[edge])],
                    "end": [float(edges.end_r[edge]), float(edges.end_z[edge])],
                    "target_position": centres[target].tolist(),
                }
            )
    output = Path(os.environ["SHEET_RECEIPTS"])
    np.savez_compressed(
        output / f"entries-{lower:04d}.npz",
        records=np.asarray(records).reshape(-1, 11),
        columns=np.array(
            [
                "local_row",
                "column",
                "target",
                "component",
                "cell",
                "input",
                "error",
                "per_end_scale",
                "tiled",
                "reference",
                "dominant_edge",
            ]
        ),
    )
    (output / f"edges-{lower:04d}.json").write_text(json.dumps(attributed, indent=2))
    finite = bool(
        np.all(np.isfinite(tiled))
        and np.all(np.isfinite(reference))
        and np.all(np.isfinite(error))
    )
    print(
        f"shard {lower}:{upper}: {len(rows)} failures, "
        f"worst {np.max(error):.9e}, finite {finite}",
        flush=True,
    )
    return error.ravel(), finite


def main():
    """Use one allocation to measure every target and retain every failure."""
    workers = gate.WORKERS
    _, centres = gate.west_mesh()
    count = len(centres)
    step = (count + workers - 1) // workers
    bounds = [(lower, min(lower + step, count)) for lower in range(0, count, step)]
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        built = list(pool.map(build, bounds))
    error = np.concatenate([item[0] for item in built])
    summary = {
        "entries": int(error.size),
        "above_floor": int(np.sum(error > gate.GATE)),
        "worst": float(np.max(error)),
        "finite": all(item[1] for item in built),
        "percentiles": {
            str(p): float(np.percentile(error, p)) for p in (50, 90, 99, 99.9, 100)
        },
    }
    Path(os.environ["SHEET_RECEIPTS"], "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
