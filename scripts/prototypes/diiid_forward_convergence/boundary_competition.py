"""Decide which surface bounds each measured terminal, from the stored map.

Two candidate boundaries bound a tokamak plasma: the separatrix through the
admitted X-point, and the largest closed surface that clears the limiter. The
bounding one is the inner of the two, so the comparison is between two flux
values read off the same solved map:

    psi_limiter = the largest flux any point of the governed limiter ring carries
    psi_saddle  = the flux at the admitted saddle

The plasma is bounded by whichever is larger: a limiter point carrying more flux
than the saddle means a limiter surface closes inside the separatrix (limited),
and the converse means the separatrix closes first (diverted).

Everything here is read from one arm's stored archive and the receipt's recorded
nulls. No solve is run, and no quantity is inferred from a padded or fixed-shape
record.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from nova.media.poloidal import inside_wall_units

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_panels import _oriented  # noqa: E402  (same orientation the panels draw)


def _grid(archive: Path) -> dict[str, Any]:
    with np.load(archive, allow_pickle=True) as data:
        keys = [key for key in data.files if key.endswith("_flux_grid")]
        if len(keys) != 1:
            raise ValueError(f"{archive} carries {len(keys)} flux grids")
        radius = np.asarray(data["radius"], dtype=float)
        height = np.asarray(data["height"], dtype=float)
        return {
            "flux": _oriented(np.asarray(data[keys[0]], dtype=float), radius, height),
            "label": _oriented(np.asarray(data["label"], dtype=float), radius, height),
            "radius": radius,
            "height": height,
            "limiter": np.asarray(data["limiter"], dtype=float),
            "wall": np.asarray(data["wall"], dtype=float),
            "label_boundary": np.asarray(data["label_boundary"], dtype=float),
        }


def _interpolator(radius: np.ndarray, height: np.ndarray, field: np.ndarray):
    return RegularGridInterpolator(
        (radius, height), field, method="cubic", bounds_error=False, fill_value=np.nan
    )


def _sample(interpolator, points: np.ndarray) -> np.ndarray:
    values = interpolator(np.asarray(points, dtype=float))
    return np.asarray(values, dtype=float).ravel()


def analyse(archive: Path, axis: np.ndarray, saddle: np.ndarray) -> dict[str, Any]:
    grid = _grid(archive)
    radius, height = grid["radius"], grid["height"]
    flux = _interpolator(radius, height, grid["flux"])
    limiter = grid["limiter"]

    inside_box = (
        (limiter[:, 0] >= radius[0])
        & (limiter[:, 0] <= radius[-1])
        & (limiter[:, 1] >= height[0])
        & (limiter[:, 1] <= height[-1])
    )
    ring = limiter[inside_box]
    ring_flux = _sample(flux, ring)
    usable = np.isfinite(ring_flux)
    ring, ring_flux = ring[usable], ring_flux[usable]
    contact = ring[int(np.argmax(ring_flux))]
    psi_limiter = float(np.max(ring_flux))

    psi_axis = float(_sample(flux, axis[None, :])[0])
    psi_saddle = (
        float(_sample(flux, saddle[None, :])[0]) if saddle.size == 2 else float("nan")
    )

    saddle_inside = (
        bool(inside_wall_units(saddle[None, :], limiter)[0])
        if saddle.size == 2
        else False
    )
    bounds_limiter = psi_limiter >= psi_saddle
    boundary_flux = psi_limiter if bounds_limiter else psi_saddle
    return {
        "archive": archive.name,
        "psi_axis": psi_axis,
        "psi_saddle": psi_saddle,
        "psi_limiter": psi_limiter,
        "bounding_flux": boundary_flux,
        "bounds": "limiter" if bounds_limiter else "separatrix",
        "achieved_class": "limited" if bounds_limiter else "diverted",
        "margin": float(abs(psi_limiter - psi_saddle)),
        "contact_rz_m": contact.tolist(),
        "saddle_rz_m": saddle.tolist() if saddle.size == 2 else None,
        "saddle_inside_limiter": saddle_inside,
        "limiter_points_sampled": int(ring.size // 2),
        "fraction_of_axis_flux": float(
            (boundary_flux - psi_axis) / (psi_limiter - psi_axis)
            if psi_limiter != psi_axis
            else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    probe = json.loads(arguments.probe.read_text())
    result: dict[str, Any] = {"frames": {}}
    for frame, record in probe["frames"].items():
        for branch_name, branch in (record.get("branches") or {}).items():
            for arm_name, arm in (branch.get("arms") or {}).items():
                if arm.get("status") != "solved":
                    continue
                presence = arm.get("presence") or {}
                axis = np.asarray(presence.get("axis_rz_m") or [], dtype=float)
                saddle = np.asarray(arm.get("x_point_rz_m") or [], dtype=float).reshape(
                    -1
                )
                if axis.size != 2:
                    continue
                archive = arguments.artifact_dir / pathlib_name(
                    frame, branch_name, arm_name
                )
                if not archive.exists():
                    continue
                entry = analyse(archive, axis, saddle)
                entry["requested_class"] = branch.get("requested_class")
                entry["recorded_achieved_class"] = arm.get("achieved_class")
                result["frames"].setdefault(frame, {}).setdefault(branch_name, {})[
                    arm_name
                ] = entry
                print(
                    f"frame {frame:>3s} {branch_name:8s} {arm_name:8s} "
                    f"psi_axis={entry['psi_axis']:+.4f} "
                    f"psi_saddle={entry['psi_saddle']:+.4f} "
                    f"psi_lim={entry['psi_limiter']:+.4f} "
                    f"bounds={entry['bounds']:10s} "
                    f"class={entry['achieved_class']:8s} "
                    f"contact=({entry['contact_rz_m'][0]:.3f},{entry['contact_rz_m'][1]:+.3f})"
                )
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


def pathlib_name(frame: str, branch: str, arm: str) -> str:
    return f"frame-{frame}-{branch}-{arm}.npz"


if __name__ == "__main__":
    raise SystemExit(main())
