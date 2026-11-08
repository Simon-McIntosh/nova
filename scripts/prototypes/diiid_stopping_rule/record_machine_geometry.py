"""Record the machine geometry each terminal panel needs.

The probe's artifact carries the pseudo wall the solve was seeded with, and that
array is the flux-grid bounding box, which reads as an axes frame when drawn.
This script records the governed DIII-D limiter ring and the bounding surface the
host boundary read traces from the terminal flux, so a panel can show where the
plasma sat relative to the machine.  The grid the boundary read needs is carried
by the artifact's own lattice axes plus the governed ring, so no solve is
repeated to draw a state that has already been measured.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from nova.equilibrium.connectivity_boundary import LCFS_ANGLES, host_boundary_read

PROBE = Path(__file__).resolve().parent.parent / "diiid_forward_convergence"
sys.path.insert(0, str(PROBE))

from boundary_competition import Grid  # noqa: E402
from render_panels import _oriented  # noqa: E402



def _governed_ring() -> np.ndarray:
    """Read the governed limiter ring through the machine description."""

    import measure_branches  # noqa: PLC0415

    bench = measure_branches.bench
    ring, _receipt = bench._physical_wall_ring(
        bench.DEFAULT_MACHINE_ARTIFACT_CACHE,
        bench.DEFAULT_MACHINE_ARTIFACT_DIGEST,
    )
    return np.asarray(ring, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    limiter = _governed_ring()
    data = np.load(arguments.artifact)
    key = (
        f"frames.{arguments.frame}.branches.{arguments.branch}"
        ".arms.default._flux_grid"
    )
    flux = np.asarray(data[key], dtype=float)
    radius = np.asarray(data["radius"], dtype=float)
    height = np.asarray(data["height"], dtype=float)
    flux = _oriented(flux, radius, height)
    grid = Grid(radius, height, limiter)

    record = json.loads(arguments.record.read_text())
    arm = record["frames"][str(arguments.frame)]["branches"][arguments.branch]["arms"][
        "default"
    ]
    axis = tuple(float(value) for value in arm["presence"]["axis_rz_m"])
    angles = np.asarray(LCFS_ANGLES, dtype=float)
    try:
        boundary = host_boundary_read(flux, grid, axis)
    except ValueError as error:
        surface = np.zeros((0, 2))
        refusal = str(error)
        margin = float("nan")
        found = False
    else:
        surface = np.column_stack(
            (
                axis[0] + np.asarray(boundary.radii, dtype=float) * np.cos(angles),
                axis[1] + np.asarray(boundary.radii, dtype=float) * np.sin(angles),
            )
        )
        refusal = ""
        margin = float(boundary.class_margin)
        found = bool(boundary.found)

    np.savez_compressed(
        arguments.output,
        limiter=limiter,
        surface=surface,
        surface_found=found,
        surface_class_margin=margin,
        surface_refusal=refusal,
        axis=np.asarray(axis, dtype=float),
        grid_r=np.asarray([float(radius[0]), float(radius[-1])]),
        grid_z=np.asarray([float(height[0]), float(height[-1])]),
    )
    print(
        f"WROTE {arguments.output} limiter={limiter.shape} "
        f"surface={surface.shape} found={found} margin={margin:.3f} "
        f"refusal={refusal[:80]}"
    )


if __name__ == "__main__":
    main()