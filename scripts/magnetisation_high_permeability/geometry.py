"""Cross-check shell radii from the union of the actual clipped cells."""

import argparse
import json
from pathlib import Path

import numpy as np
import shapely

from measure import PERMEABILITIES, SIZES, dump, shell_mesh_volumes, shielding
from nova.biot import magnetisation as iron


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    half = shapely.box(0, -1, 1, 1)
    outer = shapely.Point(0, 0).buffer(1, 512).intersection(half)
    inner = shapely.Point(0, 0).buffer(0.9, 512).intersection(half)
    region = outer.difference(inner)
    record = {"geometry": []}
    print(f"GEOMETRY_MODULE {iron.__file__} cwd={Path.cwd()}", flush=True)
    for size in SIZES:
        cells = iron._mesh(region, size)
        vo, vi, merged = shell_mesh_volumes(cells)
        difference = merged.symmetric_difference(region).area
        assert difference < 1e-14
        row = dict(
            h=size,
            cells=len(cells.polygons),
            volume=vo - vi,
            outer_volume=vo,
            inner_volume=vi,
            outer_radius=(3 * vo / (4 * np.pi)) ** (1 / 3),
            inner_radius=(3 * vi / (4 * np.pi)) ** (1 / 3),
            fraction=vi / vo,
            symmetric_difference_area=difference,
            volume_implied={str(mu): shielding(mu, vi / vo) for mu in PERMEABILITIES},
        )
        record["geometry"].append(row)
        dump(args.output, record)
        print("MESH_UNION", json.dumps(row), flush=True)
    print("GEOMETRY_SUMMARY", json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
