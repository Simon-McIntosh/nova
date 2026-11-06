"""Locate the interior-flood promoted nodes against the flux they sit in.

The promoted nodes split by selection source, and the interior-flood source
itself splits in two: an upper group whose held flux agrees with the field of
the currents to a nanoweber, and a lower group whose held flux disagrees by
tens of milliwèbers.  Both come from the same mask, so the difference is not
the selection rule; it is where the selected cells sit in the flux map.

This measures that: every promoted row's normalised flux, the polished lower
saddle the solve did not admit, the pocket's flux against that saddle's flux,
and each lower node's distance to the separatrix and to the wall.  Everything
is read from the archived fields of the one measurement, so no solve repeats.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from nova.equilibrium.boundary_comparison import (  # noqa: E402
    _point_to_segment_distances,
)
from nova.equilibrium.flux_surface_connectivity import (  # noqa: E402
    polish_stationary_points,
)
from nova.jax.config import configure_dtypes  # noqa: E402
from nova.linalg.tensor_spline import fit_tensor_spline  # noqa: E402

LOWER_SEED_RZ = (0.60, -1.20)
FLOOD_SOURCE = "interior flood (private flux)"
WALL_SOURCE = "private-wall band"
SADDLE_TYPE = {1: "extremum", -1: "saddle", 0: "degenerate"}


def _surface(fields, name):
    return fit_tensor_spline(
        jnp.asarray(np.asarray(fields["radius"], float)),
        jnp.asarray(np.asarray(fields["height"], float)),
        jnp.asarray(np.asarray(fields[name], float)),
    )


def sample(surface, points):
    """Return the spline's flux at an (n, 2) array of radius-height points."""
    points = np.asarray(points, float)
    return np.asarray(
        surface(jnp.asarray(points[:, 0]), jnp.asarray(points[:, 1])), float
    )


def flux_levels(state_surface, image_surface, receipt):
    """Return the axis, boundary and lower-saddle flux of both fields."""
    axis_rz = np.asarray(receipt["nulls"]["axis_rz"], float)
    seed = jnp.asarray(np.asarray(LOWER_SEED_RZ, float).reshape(1, 2))
    levels = {
        "axis_wb": float(sample(state_surface, axis_rz[None, :])[0]),
        "axis_image_wb": float(sample(image_surface, axis_rz[None, :])[0]),
        "boundary_wb": float(receipt["separatrix"]["state_boundary_flux_wb"]),
        "boundary_image_wb": float(receipt["separatrix"]["image_level_wb"]),
    }
    for name, surface in (("state", state_surface), ("image", image_surface)):
        polished = polish_stationary_points(surface, seed, jnp.ones((1,), dtype=bool))
        position = np.asarray(polished["position_rz"], float).reshape(2)
        levels["lower_%s" % name] = {
            "position_rz": [float(axis) for axis in position],
            "flux_wb": float(np.asarray(polished["value"], float).reshape(())),
            "hessian_type": SADDLE_TYPE.get(
                int(np.asarray(polished["hessian_type"]).reshape(())), "unknown"
            ),
            "converged": bool(np.asarray(polished["converged"]).reshape(())),
        }
    return levels


def wall_distance(points, units):
    """Return each point's distance in metres to the nearest wall segment.

    The wall archive is a polyline, so the distance to it is the distance to
    its segments: a vertex-only reading passes over the span between two
    widely spaced vertices and overstates the clearance of any node that sits
    alongside a straight run.
    """
    distances = np.full(points.shape[0], np.inf)
    for unit in units:
        wall = np.column_stack(
            [np.asarray(unit["radius"], float), np.asarray(unit["height"], float)]
        )
        if bool(unit.get("closed")) and not np.allclose(wall[0], wall[-1]):
            wall = np.vstack([wall, wall[:1]])
        distances = np.minimum(distances, _point_to_segment_distances(points, wall))
    return distances


def pocket(receipt, fields):
    """Return the flux map of the promoted nodes and the lower pocket's rows."""
    state_surface = _surface(fields, "state_field")
    image_surface = _surface(fields, "image_field")
    levels = flux_levels(state_surface, image_surface, receipt)
    span = levels["boundary_wb"] - levels["axis_wb"]
    saddle = levels["lower_state"]
    saddle_rz = np.asarray(saddle["position_rz"], float)
    x_point_z = float(receipt["x_point"]["state_rz"][1])

    coordinates = np.asarray(
        [[row["radius_m"], row["height_m"]] for row in receipt["promoted"]], float
    )
    spline = sample(state_surface, coordinates)
    distances = wall_distance(coordinates, receipt["wall_units"])
    rows = []
    for row, state, wall in zip(receipt["promoted"], spline, distances, strict=True):
        point = np.asarray([row["radius_m"], row["height_m"]], float)
        rows.append(
            {
                **row,
                "half": "upper" if row["height_m"] > x_point_z else "lower",
                "state_spline_wb": float(state),
                "psi_norm": float((state - levels["axis_wb"]) / span),
                "flux_above_lower_saddle_wb": float(state - saddle["flux_wb"]),
                "distance_to_wall_mm": float(1.0e3 * wall),
                "below_lower_saddle_mm": float(1.0e3 * (saddle_rz[1] - point[1])),
                "distance_to_lower_saddle_mm": float(
                    1.0e3 * np.linalg.norm(point - saddle_rz)
                ),
            }
        )

    groups = {}
    for half in ("upper", "lower"):
        for source in (FLOOD_SOURCE, WALL_SOURCE):
            selected = [
                row
                for row in rows
                if row["half"] == half and row["promoted_source"] == source
            ]
            if not selected:
                continue
            magnitudes = np.abs([row["image_minus_state_wb"] for row in selected])
            groups["%s / %s" % (half, source)] = {
                "count": len(selected),
                "max_abs_wb": float(magnitudes.max()),
                "median_abs_wb": float(np.median(magnitudes)),
                "flux_min_wb": float(min(row["state_wb"] for row in selected)),
                "flux_max_wb": float(max(row["state_wb"] for row in selected)),
                "psi_norm_min": float(min(row["psi_norm"] for row in selected)),
                "psi_norm_max": float(max(row["psi_norm"] for row in selected)),
                "distance_to_separatrix_mm_min": float(
                    min(row["distance_to_separatrix_mm"] for row in selected)
                ),
                "distance_to_separatrix_mm_max": float(
                    max(row["distance_to_separatrix_mm"] for row in selected)
                ),
                "distance_to_wall_mm_range": [
                    float(min(row["distance_to_wall_mm"] for row in selected)),
                    float(max(row["distance_to_wall_mm"] for row in selected)),
                ],
            }

    lower_flood = [
        row
        for row in rows
        if row["half"] == "lower" and row["promoted_source"] == FLOOD_SOURCE
    ]
    lower_flood.sort(key=lambda row: -abs(row["image_minus_state_wb"]))
    keys = (
        "node",
        "radius_m",
        "height_m",
        "state_wb",
        "image_wb",
        "image_minus_state_wb",
        "operator_image_minus_state_wb",
        "psi_norm",
        "flux_above_lower_saddle_wb",
        "distance_to_separatrix_mm",
        "distance_to_lower_saddle_mm",
        "below_lower_saddle_mm",
        "distance_to_wall_mm",
        "domain_label",
    )
    return {
        "schema": "nova.promoted-nodes-xpoint-angles.private-flux-pocket",
        "lower_seed_rz": list(LOWER_SEED_RZ),
        "x_point_rz": list(np.asarray(receipt["x_point"]["state_rz"], float)),
        "levels": levels,
        "groups": groups,
        "pocket": {
            "count": len(lower_flood),
            "nodes": [{key: row[key] for key in keys} for row in lower_flood],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    receipt = json.loads(arguments.receipt.read_text(encoding="utf-8"))
    fields = dict(np.load(arguments.fields))
    found = pocket(receipt, fields)
    found["source_receipt"] = str(arguments.receipt)
    found["source_fields"] = str(arguments.fields)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(found, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name, group in sorted(found["groups"].items()):
        print(
            "GROUP %-36s n=%2d max %.4e median %.4e "
            "psi_norm %.4f..%.4f sep %.1f..%.1f mm"
            % (
                name,
                group["count"],
                group["max_abs_wb"],
                group["median_abs_wb"],
                group["psi_norm_min"],
                group["psi_norm_max"],
                group["distance_to_separatrix_mm_min"],
                group["distance_to_separatrix_mm_max"],
            )
        )
    print("POCKET %s" % arguments.output)


if __name__ == "__main__":
    main()
