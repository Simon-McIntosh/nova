"""Measure the local separatrix directions at each field's polished X-point.

Two independent readings, both taken at the null rather than at the far end of
a branch:

* the asymptotic directions: the eigenvectors of the poloidal-flux Hessian at
  the polished saddle, which are the four straight lines the separatrix
  approaches there, reported as four half-line bearings;
* the direction each assembled branch takes over its first one to two display
  cells, read as the chord of that branch between 10 and 20 mm of arc length
  from the saddle.

Everything is read from the archived fields of one measurement, so no solve is
repeated.  The closed-branch vertex counts the two fields carry are diagnosed
here too: the same curve subdivided differently and a different curve are told
apart by arc length against segment length.
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

from nova.equilibrium.flux_surface_connectivity import (  # noqa: E402
    polish_stationary_points,
)
from nova.jax.config import configure_dtypes  # noqa: E402
from nova.linalg.tensor_spline import fit_tensor_spline  # noqa: E402
from nova.media.poloidal import sample_cubic_controls  # noqa: E402

MU0 = 4.0e-7 * np.pi
NEAR_M = 0.010
FAR_M = 0.020
DEG = "\N{DEGREE SIGN}"


def bearing(direction):
    """Return a direction's bearing in degrees from +R toward +Z."""
    return float(np.degrees(np.arctan2(direction[1], direction[0])))


def angle_between(left_deg, right_deg):
    """Return the angle between two bearings, as degrees from 0 to 180."""
    difference = (float(left_deg) - float(right_deg) + 180.0) % 360.0 - 180.0
    return abs(difference)


def hessian_name(value):
    """Name a Hessian's type from the sign of its determinant."""
    return {1: "extremum", -1: "saddle", 0: "degenerate"}.get(int(value), "unknown")


def asymptotic(values, radius, height, seed):
    """Return the polished saddle, its principal axes and its asymptotes.

    The Hessian's eigenvectors are the principal axes, which are orthogonal by
    construction and are not the separatrix asymptotes.  The asymptotes are the
    directions d with d'Hd = 0: in the eigenbasis, l1 cos^2(t) + l2 sin^2(t) = 0,
    so the half angle from the major axis is atan(sqrt(-l1/l2)) and is 45
    degrees only when the two eigenvalues are equal and opposite.
    """
    surface = fit_tensor_spline(
        jnp.asarray(radius), jnp.asarray(height), jnp.asarray(values)
    )
    polished = polish_stationary_points(
        surface,
        jnp.asarray(np.asarray(seed, dtype=float).reshape(1, 2)),
        jnp.ones((1,), dtype=bool),
    )
    position = np.asarray(polished["position_rz"], dtype=float).reshape(2)
    hessian = np.asarray(polished["hessian"], dtype=float).reshape(2, 2)
    kind = int(np.asarray(polished["hessian_type"]).reshape(()))
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    major = eigenvectors[:, 1]
    minor = eigenvectors[:, 0]
    trace = float(eigenvalues[0] + eigenvalues[1])
    ratio = (
        float(-eigenvalues[1] / eigenvalues[0]) if eigenvalues[1] > 0 else float("nan")
    )
    half_angle = (
        float(np.degrees(np.arctan(np.sqrt(ratio)))) if ratio >= 0 else float("nan")
    )
    directions = []
    if np.isfinite(half_angle):
        cosine = np.cos(np.radians(half_angle))
        sine = np.sin(np.radians(half_angle))
        for first in (cosine * major + sine * minor, cosine * major - sine * minor):
            for direction in (first, -first):
                directions.append(
                    {
                        "bearing_deg": bearing(direction),
                        "unit": [float(value) for value in direction],
                    }
                )
    directions.sort(key=lambda entry: entry["bearing_deg"])
    axes = []
    for column in (1, 0):
        axes.append(
            {
                "bearing_deg": bearing(eigenvectors[:, column]),
                "unit": [float(value) for value in eigenvectors[:, column]],
                "eigenvalue_wb_per_m2": float(eigenvalues[column]),
            }
        )
    return {
        "position_rz": [float(value) for value in position],
        "hessian": [[float(value) for value in row] for row in hessian],
        "eigenvalues": [float(value) for value in eigenvalues],
        "eigenvalue_ratio": ratio,
        "asymptote_half_angle_deg": half_angle,
        "asymptote_separation_deg": 2.0 * half_angle,
        "hessian_trace_wb_per_m2": trace,
        "toroidal_current_density_a_per_m2": -trace / MU0,
        "hessian_type": hessian_name(kind),
        "admitted": bool(kind < 0 and np.all(np.isfinite(position))),
        "directions": directions,
        "axes": axes,
    }


def arm(points, saddle):
    """Return the 10-20 mm chord of one arm, or None when it ends short."""
    points = np.asarray(points, dtype=float)
    if points.shape[0] < 3:
        return None
    travelled = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    reach = float(travelled[-1])
    if reach < FAR_M:
        return None
    near = points[int(np.searchsorted(travelled, NEAR_M))]
    far = points[int(np.searchsorted(travelled, FAR_M))]
    offset = far - near
    span = float(np.linalg.norm(offset))
    if span < 1.0e-12:
        return None
    return {
        "bearing_deg": bearing(offset),
        "start_rz": [float(value) for value in near],
        "end_rz": [float(value) for value in far],
        "span_m": span,
        "reach_m": reach,
        "start_from_saddle_mm": float(
            1.0e3 * np.linalg.norm(near - np.asarray(saddle, float))
        ),
    }


def closed_arms(polyline, saddle):
    """Return the two arms the closed branch leaves the null along.

    The traced loop begins and ends at the null, so its two arms are the two
    halves of that loop rather than its two traversal directions: leaving from
    the null one way reaches the far side, and from the null the other way
    reaches the same far side backwards.  The loop is opened at the null and
    split where it is furthest from it.
    """
    polyline = np.asarray(polyline, dtype=float)
    distance = np.linalg.norm(polyline - np.asarray(saddle, float), axis=1)
    start = int(np.argmin(distance))
    rolled = np.roll(polyline, -start, axis=0)
    span = np.linalg.norm(rolled - np.asarray(saddle, float), axis=1)
    far = int(np.argmax(span))
    arms = []
    for name, points in (
        ("closed out", rolled[: far + 1]),
        ("closed back", rolled[far + 1 :][::-1]),
    ):
        found = arm(points, saddle)
        if found is not None:
            found["arm"] = name
            arms.append(found)
    return arms, float(distance[start])


def open_arms(controls, valid, saddle):
    """Return every open branch's 10-20 mm chord, oriented from the null."""
    arms = []
    controls = np.asarray(controls, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    for slot in range(controls.shape[0]):
        points = np.asarray(sample_cubic_controls(controls[slot], valid[slot]), float)
        if points.shape[0] < 3:
            continue
        if np.linalg.norm(points[-1] - saddle) < np.linalg.norm(points[0] - saddle):
            points = points[::-1]
        found = arm(points, saddle)
        if found is not None:
            found["arm"] = "open %d" % slot
            found["slot"] = int(slot)
            arms.append(found)
    return arms


def closed_geometry(polyline):
    """Describe a closed branch's own subdivision and length."""
    polyline = np.asarray(polyline, dtype=float)
    segments = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    return {
        "vertices": int(polyline.shape[0]),
        "arc_length_m": float(segments.sum()),
        "median_segment_mm": float(1.0e3 * np.median(segments)),
        "max_segment_mm": float(1.0e3 * np.max(segments)),
        "segments_under_1mm": int(np.count_nonzero(segments < 1.0e-3)),
        "cells_per_segment": float(
            float(segments.mean()) / max(np.median(segments), 1.0e-12)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fields-output", type=Path, required=True)
    arguments = parser.parse_args()
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    receipt = json.loads(arguments.receipt.read_text(encoding="utf-8"))
    archive = dict(np.load(arguments.fields))
    radius = np.asarray(archive["radius"], float)
    height = np.asarray(archive["height"], float)

    # The receipt's displacement compares the state's null, polished on the
    # lattice spline, with the image's, taken from the display map, so the two
    # come from different interpolants.  Polishing both fields on one grid is
    # the like-for-like measurement, and it is reported for both grids.
    grids = {"receipt": {"displacement_mm": receipt["x_point"]["displacement_mm"]}}
    for grid, grid_radius, grid_height, state_values, image_values in (
        (
            "lattice",
            radius,
            height,
            archive["state_field"],
            archive["image_field"],
        ),
        (
            "display",
            np.asarray(archive["display_radius"], float),
            np.asarray(archive["display_height"], float),
            np.asarray(archive["display_state_flux"], float).T,
            np.asarray(archive["display_flux"], float).T,
        ),
    ):
        left = asymptotic(
            state_values, grid_radius, grid_height, receipt["x_point"]["state_rz"]
        )
        right = asymptotic(
            image_values, grid_radius, grid_height, receipt["x_point"]["image_rz"]
        )
        left_rz = np.asarray(left["position_rz"], float)
        right_rz = np.asarray(right["position_rz"], float)
        grids[grid] = {
            "state_rz": left["position_rz"],
            "image_rz": right["position_rz"],
            "displacement_mm": float(1.0e3 * np.linalg.norm(right_rz - left_rz)),
            "hessian_relative_difference": float(
                np.max(
                    np.abs(np.asarray(right["hessian"]) - np.asarray(left["hessian"]))
                )
                / max(np.max(np.abs(np.asarray(left["hessian"]))), 1.0e-30)
            ),
            "state_bearings_deg": [
                entry["bearing_deg"] for entry in left["directions"]
            ],
            "image_bearings_deg": [
                entry["bearing_deg"] for entry in right["directions"]
            ],
            "bearing_shifts_deg": [
                float(
                    abs(
                        np.degrees(
                            np.arctan2(
                                np.sin(np.radians(a - b)),
                                np.cos(np.radians(a - b)),
                            )
                        )
                    )
                )
                for a, b in zip(
                    [entry["bearing_deg"] for entry in left["directions"]],
                    [entry["bearing_deg"] for entry in right["directions"]],
                    strict=False,
                )
            ],
        }

    # Each field's branches stay on the grid its own separatrix was assembled
    # on: the state's from the solve's lattice, the image's from the display
    # map.  Both traces are reported with the grid and sampling they carry.
    blocks = {}
    sampled = {}
    for name, values, seed, closed, controls, valid in (
        (
            "state",
            archive["state_field"],
            receipt["x_point"]["state_rz"],
            archive["state_closed_polyline"],
            archive["state_open_controls"],
            archive["state_open_valid"],
        ),
        (
            "image",
            archive["image_field"],
            receipt["x_point"]["image_rz"],
            archive["image_closed_polyline"],
            archive["image_open_controls"],
            archive["image_open_valid"],
        ),
    ):
        null_field = asymptotic(values, radius, height, seed)
        null = np.asarray(null_field["position_rz"], dtype=float)
        closed_found, gap = closed_arms(closed, null)
        opens = open_arms(controls, valid, null)
        blocks[name] = {
            "asymptotic": null_field,
            "assembled_null_gap_mm": float(1.0e3 * gap),
            "closed": closed_geometry(closed),
            "branches": closed_found + opens,
        }
        sampled[name] = {
            "closed": np.asarray(closed, float),
            "open": {
                found["slot"]: np.asarray(
                    sample_cubic_controls(
                        controls[found["slot"]], valid[found["slot"]]
                    ),
                    float,
                )
                for found in opens
            },
        }

    output = {
        "schema": "nova.promoted-nodes-xpoint-angles.null-directions",
        "window_m": [NEAR_M, FAR_M],
        "lattice_spacing_m": float(receipt["lattice_spacing_m"]),
        "state": blocks["state"],
        "image": blocks["image"],
        "pairs": pair_blocks(blocks),
        "grids": grids,
        "polylines": {
            "grid": "display map, %.4f m"
            % float(np.median(np.diff(np.asarray(archive["display_radius"], float)))),
            "archived_state_grid": "solve lattice, %.4f m"
            % float(receipt["lattice_spacing_m"]),
            "archived_state_vertices": int(
                np.asarray(archive["state_closed_polyline"]).shape[0]
            ),
            "archived_image_vertices": int(
                np.asarray(archive["image_closed_polyline"]).shape[0]
            ),
        },
        "source_receipt": str(arguments.receipt),
        "source_fields": str(arguments.fields),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    arrays = {
        "display_radius": np.asarray(archive["display_radius"], float),
        "display_height": np.asarray(archive["display_height"], float),
        "display_flux": np.asarray(archive["display_flux"], float),
        "display_state_flux": np.asarray(archive["display_state_flux"], float),
        "promoted_coordinate": np.asarray(archive["promoted_coordinate"], float),
        "state_closed_polyline": sampled["state"]["closed"],
        "image_closed_polyline": sampled["image"]["closed"],
    }
    for name in ("state", "image"):
        for slot, points in sampled[name]["open"].items():
            arrays["%s_open_%d" % (name, slot)] = points
    arguments.fields_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.fields_output, **arrays)

    print("ANGLES %s" % arguments.output)
    print("ANGLES_FIELDS %s" % arguments.fields_output)
    for name in ("state", "image"):
        block = blocks[name]
        print(
            "%-5s asymptotic %s | branches %s | closed vertices %d arc %.4f m"
            % (
                name,
                ", ".join(
                    "%.1f" % entry["bearing_deg"]
                    for entry in block["asymptotic"]["directions"]
                ),
                ", ".join("%.1f" % entry["bearing_deg"] for entry in block["branches"]),
                block["closed"]["vertices"],
                block["closed"]["arc_length_m"],
            )
        )
    for pair in output["pairs"]:
        print(
            "PAIR %-10s %7.1f -> %7.1f  shift %.3f"
            % (
                pair["kind"],
                pair["state_bearing_deg"],
                pair["image_bearing_deg"],
                pair["shift_deg"],
            )
        )


def pair_blocks(blocks):
    """Pair each state direction with the nearest image direction, per kind."""
    pairs = []
    for kind in ("asymptotic", "branch"):
        if kind == "asymptotic":
            state = blocks["state"]["asymptotic"]["directions"]
            image = blocks["image"]["asymptotic"]["directions"]
        else:
            state = blocks["state"]["branches"]
            image = blocks["image"]["branches"]
        for left in state:
            if kind == "branch":
                # Reach identifies a branch far more strongly than a bearing
                # that is still curving 20 mm from the null.
                metric = [abs(left["reach_m"] - right["reach_m"]) for right in image]
            else:
                metric = [
                    angle_between(left["bearing_deg"], right["bearing_deg"])
                    for right in image
                ]
            partner = int(np.argmin(metric)) if metric else None
            pairs.append(
                {
                    "kind": kind,
                    "state_bearing_deg": left["bearing_deg"],
                    "image_bearing_deg": (
                        image[partner]["bearing_deg"] if partner is not None else None
                    ),
                    # The shift belongs to the branch this one was paired with.
                    # Reporting the closest of all image branches instead would
                    # read as agreement whenever any other branch happened to
                    # lie nearby, which is the opposite of a pairing.
                    "shift_deg": (
                        float(
                            angle_between(
                                left["bearing_deg"], image[partner]["bearing_deg"]
                            )
                        )
                        if partner is not None
                        else None
                    ),
                    "unpaired_image_deg": sorted(
                        entry["bearing_deg"] for entry in image
                    ),
                }
            )
    return pairs


if __name__ == "__main__":
    main()
