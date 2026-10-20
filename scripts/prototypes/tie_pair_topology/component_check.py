"""Check the axis component on both sides of its first saddle level."""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.path import Path as PolygonPath


def check(receipt, field_data):
    rows = json.loads(Path(receipt).read_text())
    with np.load(field_data) as data:
        r, z = data["r"], data["z"]
        rr, zz = np.meshgrid(r, z, indexing="xy")
        points = np.stack((rr.ravel(), zz.ravel()), axis=1)
        for row in rows:
            key = str(row["offset"]).replace(".", "_")
            inside = data[key + "_inside"]
            passage = data[key + "_passage"]
            flux = data[key + "_flux"]
            axis = next(
                point for point in row["independent"] if point["kind"] == "maximum"
            )
            upper = max(
                (point for point in row["independent"] if point["kind"] == "saddle"),
                key=lambda point: point["position"][1],
            )
            edge = inside & (
                ~np.roll(inside, 1, axis=0)
                | ~np.roll(inside, -1, axis=0)
                | ~np.roll(inside, 1, axis=1)
                | ~np.roll(inside, -1, axis=1)
            )
            support = np.zeros(inside.shape, dtype=bool)
            for cell in row["read"]["support"]:
                polygon = PolygonPath(np.asarray(cell["vertices"]), closed=True)
                support |= polygon.contains_points(points).reshape(inside.shape)
            persistence = axis["level"] - upper["level"]
            levels = {
                "above": upper["level"] + persistence / 2,
                "below": upper["level"] - persistence / 4,
            }
            counts = {}
            for side, level in levels.items():
                component = inside & (passage >= level) & (flux >= level)
                counts[side] = {
                    "level": level,
                    "axis_component_pixels": int(component.sum()),
                    "wall_pixels": int((component & edge).sum()),
                    "support_pixels": int((component & support).sum()),
                }
            assert counts["above"]["axis_component_pixels"] > 0
            assert counts["above"]["wall_pixels"] == 0
            assert counts["above"]["support_pixels"] == 0
            assert counts["below"]["wall_pixels"] > 0
            assert counts["below"]["support_pixels"] > 0
            print(json.dumps({"offset": row["offset"], "counts": counts}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--field", type=Path, required=True)
    args = parser.parse_args()
    check(args.receipt, args.field)


if __name__ == "__main__":
    main()
