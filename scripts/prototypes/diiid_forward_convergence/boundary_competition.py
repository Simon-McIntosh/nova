"""Read each stored terminal's bounding surface with the repository's own reader.

The binding surface of a plasma is the outermost closed surface that encloses
the axis and stays inside the wall.  It is not the largest flux on the wall
ring: on a diverted equilibrium the wall beneath the X-point carries more flux
than the separatrix, inside the private-flux region, so an argmax over the ring
returns a point under the X-point and calls every arm limited.

``nova.equilibrium.connectivity_boundary.host_boundary_read`` computes the
surface by a saddle-aware, axis-connected flood fill, which never enters a
private lobe or a divertor leg, and reports the binding obstruction and a
signed class margin.  ``nova.equilibrium.boundary_comparison.classify_boundary_mode``
turns that margin into the class and raises on an indeterminate one.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_panels import _oriented  # noqa: E402

from nova.equilibrium.boundary_comparison import BoundaryMode, classify_boundary_mode  # noqa: E402
from nova.equilibrium.connectivity_boundary import host_boundary_read  # noqa: E402
from nova.media.poloidal import inside_wall_units  # noqa: E402

# The read's own default is eight angles, which draws the surface as an octagon
# and leaves a ray unresolved when one lands on a wall node.
SURFACE_ANGLES = np.linspace(0.0, 2.0 * np.pi, 73)[:-1]


class Grid:
    """The attributes the boundary read resolves through an equilibrium grid."""

    def __init__(self, radius, height, limiter):
        self.rg = np.asarray(radius, dtype=float)
        self.zg = np.asarray(height, dtype=float)
        self.limiter_r = np.asarray(limiter[:, 0], dtype=float)
        self.limiter_z = np.asarray(limiter[:, 1], dtype=float)
        grid_r, grid_z = np.meshgrid(self.rg, self.zg)
        points = np.column_stack([grid_r.ravel(), grid_z.ravel()])
        self.inside_limiter = np.asarray(
            inside_wall_units(points, limiter), dtype=bool
        ).reshape(grid_r.shape)


def _archive(archive: Path) -> dict[str, Any]:
    with np.load(archive, allow_pickle=True) as data:
        keys = [key for key in data.files if key.endswith("_flux_grid")]
        if len(keys) != 1:
            raise ValueError(f"{archive} carries {len(keys)} flux grids")
        radius = np.asarray(data["radius"], dtype=float)
        height = np.asarray(data["height"], dtype=float)
        return {
            "flux": _oriented(np.asarray(data[keys[0]], dtype=float), radius, height),
            "radius": radius,
            "height": height,
            "limiter": np.asarray(data["limiter"], dtype=float),
        }


def _wall_distance(
    points: np.ndarray, ring: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Distance from each point to the wall polyline, and the point on it.

    Measured to the wall's *segments*, not its vertices: the governed limiter is
    84 points spanning up to a metre of wall between them, so a vertex distance
    reports a contact gap that is only the outline's own discretisation.
    """
    start = ring
    end = np.roll(ring, -1, axis=0)
    edge = end - start
    # A closed outline repeats its first vertex, so the wrapping edge is
    # zero-length; keep a safe denominator and let that edge collapse onto it.
    length2 = np.einsum("ij,ij->i", edge, edge)
    safe = np.where(length2 > 0.0, length2, 1.0)
    offset = points[:, None, :] - start[None, :, :]
    fraction = np.einsum("ijk,jk->ij", offset, edge) / safe[None, :]
    fraction = np.clip(fraction, 0.0, 1.0)
    onto = start[None, :, :] + fraction[..., None] * edge[None, :, :]
    return np.linalg.norm(points[:, None, :] - onto, axis=2), onto


def analyse(archive: Path, axis: np.ndarray) -> dict[str, Any]:
    grid_data = _archive(archive)
    grid = Grid(grid_data["radius"], grid_data["height"], grid_data["limiter"])
    try:
        boundary = host_boundary_read(
            grid_data["flux"],
            grid,
            (float(axis[0]), float(axis[1])),
            angles=SURFACE_ANGLES,
        )
    except ValueError as error:
        # The read's own precondition: an axis that is not inside the wall has
        # no axis-connected plasma to grow, which is the finding, not a defect.
        return {
            "archive": archive.name,
            "status": "refused",
            "refusal_message": str(error),
            "axis_rz_m": [float(axis[0]), float(axis[1])],
        }
    mode = classify_boundary_mode(boundary.class_margin)
    angles = np.asarray(SURFACE_ANGLES, dtype=float)
    # The ray-cast radii are measured about the seed the read was grown from, so
    # the surface is reconstructed about that same point.  The read's own
    # ``axis`` is a sub-grid O-point diagnostic and is NaN when no in-wall
    # O-point polishes, which says nothing about where the radii were taken.
    axis_r, axis_z = float(axis[0]), float(axis[1])
    surface = np.column_stack(
        [
            axis_r + boundary.radii * np.cos(angles),
            axis_z + boundary.radii * np.sin(angles),
        ]
    )
    # The binding obstruction is the single point of the surface that stands on
    # the wall: for a limited state it is the contact point, and for a diverted
    # state the same point is where the open surface meets the wall.
    ring = grid_data["limiter"]
    finite = np.isfinite(surface).all(axis=1)
    if finite.any():
        gap, _ = _wall_distance(surface[finite], ring)
        point = int(np.argmin(gap.min(axis=1)))
        distance = float(gap[point].min())
        # The contact is the boundary point itself, never a projection onto the
        # wall: a recorded contact the drawn surface does not pass through is
        # the defect this field exists to make visible.
        contact = surface[finite][point]
    else:
        distance = float("nan")
        contact = np.full(2, np.nan)
    return {
        "archive": archive.name,
        "status": "solved",
        "found": bool(boundary.found),
        "psi_bnd": float(boundary.psi_bnd),
        "psi_lcfs": float(boundary.psi_lcfs),
        "psi_axis": float(boundary.psi_axis),
        "class_margin": float(boundary.class_margin),
        "achieved_class": "diverted" if mode is BoundaryMode.DIVERTED else "limited",
        "is_diverted": bool(boundary.is_diverted),
        "n_core_cells": int(boundary.n_core_cells),
        "axis_rz_m": [float(boundary.axis[0]), float(boundary.axis[1])],
        "seed_rz_m": [float(axis_r), float(axis_z)],
        "axis_state": int(boundary.axis_state),
        "x_point_rz_m": [
            [float(value) for value in point]
            for point in np.asarray(boundary.xset, dtype=float)
            if np.all(np.isfinite(point))
        ],
        "boundary_rz_m": surface[finite].tolist(),
        "surface_points": int(finite.sum()),
        "grid_step_m": [
            float(np.abs(np.diff(grid_data["radius"])).max()),
            float(np.abs(np.diff(grid_data["height"])).max()),
        ],
        "contact_rz_m": [float(contact[0]), float(contact[1])],
        "contact_gap_m": distance,
        "x_overflow": bool(boundary.x_overflow),
        "x_unresolved_count": int(boundary.x_unresolved_count),
        "boundary_resolved": bool(boundary.boundary_resolved),
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
                if axis.size != 2 or not np.all(np.isfinite(axis)):
                    continue
                archive = arguments.artifact_dir / (
                    f"frame-{frame}-{branch_name}-{arm_name}.npz"
                )
                if not archive.exists():
                    continue
                entry = analyse(archive, axis)
                if entry.get("status") == "refused":
                    print(
                        f"frame {frame:>3s} {branch_name:8s} {arm_name:8s} "
                        f"class=refused {entry['refusal_message'][:90]}",
                        flush=True,
                    )
                    result["frames"].setdefault(frame, {}).setdefault(branch_name, {})[
                        arm_name
                    ] = entry
                    continue
                entry["requested_class"] = branch.get("requested_class")
                entry["recorded_achieved_class"] = arm.get("achieved_class")
                result["frames"].setdefault(frame, {}).setdefault(branch_name, {})[
                    arm_name
                ] = entry
                print(
                    f"frame {frame:>3s} {branch_name:8s} {arm_name:8s} "
                    f"class={entry['achieved_class']:8s} "
                    f"margin={entry['class_margin']:+.4f} "
                    f"psi_bnd={entry['psi_bnd']:+.4f} "
                    f"contact=({entry['contact_rz_m'][0]:.3f},"
                    f"{entry['contact_rz_m'][1]:+.3f}) "
                    f"gap={entry['contact_gap_m']:.3f} "
                    f"cells={entry['n_core_cells']}",
                    flush=True,
                )
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
