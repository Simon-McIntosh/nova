"""Compare cell-aligned carrier observations and report their first separation."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def relative(left, right):
    """Return sup error relative to the larger operand's sup norm."""
    left, right = np.asarray(left, float), np.asarray(right, float)
    delta = np.abs(left - right)
    finite = np.isfinite(delta)
    if not np.all(finite | (np.isnan(left) & np.isnan(right))):
        return float("inf"), float("inf")
    maximum = float(np.max(delta[finite], initial=0))
    scale = max(
        float(np.max(np.abs(left[finite]), initial=0)),
        float(np.max(np.abs(right[finite]), initial=0)),
        1e-300,
    )
    return maximum, maximum / scale


def compare(run, arm):
    left = np.load(run / "split-stages.npz")
    right = np.load(run / f"{arm}-stages.npz")
    cold_distance = np.linalg.norm(
        (left["cold"] / left["scale"])[:, None]
        - (right["cold"] / right["scale"])[None],
        axis=-1,
    )
    spatial = np.linalg.norm(
        left["geometry.centre"][:, None] - right["geometry.centre"][None], axis=-1
    )
    li, ri = linear_sum_assignment(spatial + np.where(cold_distance < 1e-10, 0, 1e6))
    assert np.max(cold_distance[li, ri]) < 1e-10
    assert np.max(spatial[li, ri]) < 1e-12
    print(
        "MATCH_CONTROL",
        "cold_distance",
        np.max(cold_distance[li, ri]),
        "spatial_distance",
        np.max(spatial[li, ri]),
        "changed_indices",
        np.sum(li != ri),
    )
    count = len(li)
    result = {}
    for state in ("cold", "native", "common"):
        for field in (
            "input",
            "roots.position",
            "read.axis",
            "read.axis_flux",
            "read.boundary",
            "read.boundary_flux",
            "read.boundary_class",
            "read.wall_candidate",
            "read.saddle_candidate",
            "read.field_coefficients",
            "read.geometric_membership",
            "read.path_weight",
            "read.membership",
            "read.booking_cells",
            "map.moments",
            "map.moment_residual",
        ):
            key = f"{state}.{field}"
            a, b = left[key], right[key]
            if a.ndim and a.shape[0] == count:
                a, b = a[li], b[ri]
            absolute, rel = relative(a, b)
            result[key] = {"absolute": absolute, "relative": rel}
            print("DIFF", arm, key, "absolute", absolute, "relative", rel)
            if a.ndim and a.shape[0] == count and rel > 1e-12:
                delta = np.abs(a.astype(float) - b.astype(float)).reshape(count, -1)
                selected = np.argsort(np.max(delta, axis=1))[-8:][::-1]
                for cell in selected:
                    if np.max(delta[cell]) == 0:
                        continue
                    print(
                        "CELL",
                        key,
                        "split",
                        int(li[cell]),
                        "other",
                        int(ri[cell]),
                        "a",
                        a[cell].tolist(),
                        "b",
                        b[cell].tolist(),
                    )
    a, b = (
        left["geometry.contour_sample_points"],
        right["geometry.contour_sample_points"],
    )
    print("TREE_SHAPES", a.shape, b.shape)
    rows, columns = linear_sum_assignment(np.linalg.norm(a[:, None] - b[None], axis=-1))
    print("TREE_COORDINATES", relative(a[rows], b[columns]))
    for state in ("cold", "common"):
        print(
            "TREE_VALUES",
            state,
            relative(
                left[f"{state}.tree_values"][rows],
                right[f"{state}.tree_values"][columns],
            ),
        )
    if "common.nodal_jets" in left and "common.nodal_jets" in right:
        assert np.array_equal(left["jet.nodes"], right["jet.nodes"])
        for name, columns in (
            ("value", [0]),
            ("gradient", [1, 2]),
            ("hessian", [3, 4, 5]),
        ):
            print(
                "NODAL",
                name,
                relative(
                    left["common.nodal_jets"][:, columns],
                    right["common.nodal_jets"][:, columns],
                ),
            )
    for label, data in (("split", left), (arm, right)):
        key = "common.contour.node_valid"
        if key in data:
            live = data[key]
            ids = data["common.contour.node_vertex"][live]
            print(
                "CONTOUR_NODES",
                label,
                "count",
                int(np.sum(live)),
                "coordinates",
                data["tree.vertex_rz"][ids].tolist(),
                "flux",
                data["common.contour.node_psi"][live].tolist(),
                "types",
                data["common.contour.critical_type"][live].tolist(),
            )
    (run / f"{arm}-comparison.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"revision={revision} tree={Path.cwd()} "
        f"command={shlex.join([sys.executable, *sys.argv])}",
        flush=True,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("arm")
    args = parser.parse_args()
    compare(args.run, args.arm)
