"""Census production WEST mesh acceptance without assembling interactions."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from time import perf_counter

import numpy as np
import shapely

from nova.biot import magnetisation as material
from nova.imas.machine import StaticMachineDescription
from nova.jax.config import configure_dtypes
from scripts.west_iron_solve.measure import MACHINE, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_dtypes()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"# revision: {revision} tree: {Path.cwd().resolve()} command: "
        + shlex.join([sys.executable, *sys.argv]),
        flush=True,
    )
    record = dict(
        revision=revision,
        module=material.__file__,
        cwd=str(Path.cwd().resolve()),
        rows=[],
    )
    control = material._mesh(shapely.box(1.0, 0.0, 1.25, 0.25), 0.125)
    assert len(control.polygons) == 4
    record["positive_control_cells"] = len(control.polygons)
    machine = StaticMachineDescription.read_iron_core(
        f"imas:hdf5?path={MACHINE}", dd_version="4.1.0"
    )
    assert len(machine.iron_segments) == 6
    region = shapely.union_all(
        [
            shapely.Polygon(np.column_stack((s.outline.r, s.outline.z)))
            for s in machine.iron_segments
        ]
    )
    assert region.is_valid
    record.update(segments=6, region_area_m2=region.area)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, record)
    for size in (0.08, 0.0566, 0.04, 0.0283, 0.02):
        started = perf_counter()
        row = dict(cell_size_m=size, accepted=False, cells=None)
        try:
            cells = material._mesh(region, size)
            count = len(cells.polygons)
            row.update(
                accepted=True,
                cells=count,
                dense_system_bytes=(2 * count) ** 2 * 8,
                area_m2=float(cells.areas.sum()),
                minimum_area_fraction=float(cells.areas.min() / size**2),
            )
        except ValueError as error:
            row["refusal"] = str(error)
        row["wall_seconds"] = perf_counter() - started
        record["rows"].append(row)
        save(args.output, record)
        print("MESH " + json.dumps(row), flush=True)
    print("SUMMARY " + json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
