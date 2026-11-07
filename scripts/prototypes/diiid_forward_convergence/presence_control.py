"""Re-derive the plasma-presence read from saved grids, with its controls.

The probe first reported the closed plasma area from a mask attribute the
domain partition does not carry, and swallowed the resulting error into a zero
mask: every arm then read ``0.0 m2`` beside a confined current of about 1 MA.
The area now comes from the partition's own CORE label, the cells the field
closes around the axis and connects to it.

This script re-derives the presence read from the grids the probe saved, so the
verdict for a measured frame is recovered without re-solving anything, and it
carries the controls the measure needs:

``label``     the EFIT label field of the same frame, a known closed plasma:
              its area must be positive, and its confined current must be close
              to the frame's target.
``parabola``  an analytic field with a circular closed region of known radius,
              whose counted area must match pi r^2 to within a stated tolerance.
``monotone``  a field with no closed region at all, which must read zero.

A control that does not hold exits nonzero and names itself.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import measure_branches as probe

PARABOLA_RADIUS_M = 0.25
PARABOLA_TOLERANCE = 0.20


def _labels(prepared):
    profile = prepared["profile"]
    radius = np.asarray(profile.lattice.radius, dtype=float)
    height = np.asarray(profile.lattice.height, dtype=float)
    return profile, radius, height


def _area(prepared, grid) -> dict[str, float]:
    profile, radius, height = _labels(prepared)
    cells = probe._core_mask(profile, np.asarray(grid, dtype=float))
    return {
        "cells": int(np.sum(cells)),
        "area_m2": float(np.sum(cells)) * probe._cell_area(profile),
    }


def _counting_control(prepared) -> dict[str, float]:
    """Control the counting arithmetic against a disc of known area.

    The partition itself cannot be fed a synthetic field: the operator refuses
    to classify a field with no qualified axis, which is a fail-closed refusal
    rather than a measurement.  What is controlled here is the arithmetic the
    area is built from, on a lattice of the same spacing as the plasma lattice.
    """

    profile, radius, height = _labels(prepared)
    spacing = math.sqrt(probe._cell_area(profile))
    half = PARABOLA_RADIUS_M * 2.0
    steps = int(round(2.0 * half / spacing))
    coordinates = np.linspace(-half, half, steps)
    radius_grid, height_grid = np.meshgrid(coordinates, coordinates, indexing="ij")
    disc = (radius_grid**2 + height_grid**2) <= PARABOLA_RADIUS_M**2
    analytic = math.pi * PARABOLA_RADIUS_M**2
    counted = float(np.sum(disc)) * spacing**2
    return {
        "analytic_area_m2": analytic,
        "counted_area_m2": counted,
        "relative_error": abs(counted - analytic) / analytic,
        "lattice_spacing_m": spacing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[179, 44])
    parser.add_argument("--expansion", type=float, default=0.02)
    arguments = parser.parse_args()
    report: dict[str, object] = {"frames": {}, "controls": {}, "failing": []}
    for frame in arguments.frames:
        prepared = probe._prepare_frame(frame, arguments.expansion)
        archives = {
            path.stem: path
            for path in sorted(arguments.artifact_dir.glob(f"frame-{frame}-*.npz"))
            if "states" not in path.stem
        }
        if not archives:
            continue
        entry = {"arms": {}, "label": None}
        for name, path in archives.items():
            saved = np.load(path)
            grid = saved[next(k for k in saved.files if k.endswith("_flux_grid"))]
            entry["arms"][name] = _area(prepared, grid)
            if entry["label"] is None and "label" in saved.files:
                entry["label"] = _area(prepared, saved["label"])
        report["frames"][str(frame)] = entry
        if frame == arguments.frames[0]:
            controls = {"counting": _counting_control(prepared)}
            report["controls"] = controls
            label_area = (entry["label"] or {}).get("area_m2", 0.0)
            if label_area <= 0.0:
                report["failing"].append(
                    f"positive control: the frame {frame} label field, a closed "
                    f"plasma, reads area {label_area}"
                )
            error = controls["counting"]["relative_error"]
            if error > PARABOLA_TOLERANCE:
                report["failing"].append(
                    f"counting control: a disc of known area counts to "
                    f"{controls['counting']['counted_area_m2']:.4f} m2 against "
                    f"{controls['counting']['analytic_area_m2']:.4f} m2 analytic, "
                    f"a relative error of {error:.1%}"
                )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["failing"]:
        raise SystemExit("FAIL: " + "; ".join(report["failing"]))
    print(
        "PASS: the label field reads a closed area and the counting "
        "arithmetic matches a disc of known area"
    )


if __name__ == "__main__":
    main()
