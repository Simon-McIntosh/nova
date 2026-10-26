"""Compare one cached mixed derivative with differences of its field values."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np

from nova.biot import polygonanalytic
from nova.equilibrium.topology import BiotMomentCoupling


def run(directory):
    point = np.asarray((6.176290619919, 0.0))
    results = []
    for arm in ("split", "integration"):
        data = np.load(directory / f"{arm}-stages.npz")
        raw = np.load(directory / f"{arm}-raw-geometry.npz")
        setup = json.loads((directory / f"{arm}-setup.json").read_text())
        matrix = np.load(setup["jets"]["cache_path"], mmap_mode="r")
        delta = data["native.input"] - data["cold"]
        for cell in (212, 234, 157, 533):
            vertices = raw["cell_polygon_vertices"][
                raw["cell_polygon_offsets"][cell] : raw["cell_polygon_offsets"][
                    cell + 1
                ]
            ]
            coupling = BiotMomentCoupling.from_polygons((vertices,))
            coefficient = np.concatenate(
                (
                    delta[cell, :1],
                    np.asarray(coupling.inverse_second_moment)[0] @ delta[cell, 1:],
                )
            )

            def gradient(target):
                rows = polygonanalytic.packed_analytic_moments(
                    np,
                    np.asarray([target[0]]),
                    np.asarray([target[1]]),
                    np.asarray(coupling.edge),
                    np.asarray(coupling.weight),
                    np.asarray(coupling.norm),
                    np.asarray(coupling.centre).T,
                    np.asarray(coupling.centre).T,
                    np.asarray(coupling.reflection_axis),
                    np.asarray(coupling.reflection_partner),
                )
                radial = np.stack(rows[6:9], axis=-1)[0] @ coefficient
                vertical = -np.stack(rows[3:6], axis=-1)[0] @ coefficient
                return 2 * np.pi * target[0] * np.asarray((radial, vertical))

            estimates = []
            for step in (1e-4, 1e-5, 1e-6):
                plus, minus = point + [0, step], point - [0, step]
                estimates.append(
                    float((gradient(plus)[0] - gradient(minus)[0]) / (2 * step))
                )
            row = {
                "arm": arm,
                "cell": cell,
                "node": point.tolist(),
                "cached_mixed": float(matrix[1213, cell, :, 4] @ delta[cell]),
                "finite_difference": estimates,
                "nearest_horizontal_vertex_gap": float(np.min(np.abs(vertices[:, 1]))),
            }
            print("SOURCE_DERIVATIVE", json.dumps(row), flush=True)
            results.append(row)
    (directory / "source-derivatives.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


if __name__ == "__main__":
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"revision={revision} tree={Path.cwd()} "
        f"command={shlex.join([sys.executable, *sys.argv])}",
        flush=True,
    )
    print("measurement_module=" + polygonanalytic.__file__, flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    run(args.directory)
