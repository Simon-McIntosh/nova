"""Record the machine geometry each terminal panel needs.

The probe's artifact carries the pseudo wall the solve was seeded with, and that
array is the flux-grid bounding box, which reads as an axes frame when drawn.
This script records the governed DIII-D limiter ring, the EFIT axis of the
frame's own label field, and the bounding surface the host boundary read traces
from the terminal flux, so a panel can show where the plasma sat relative to the
machine.  The grid the boundary read needs is carried by the artifact's own
lattice axes plus the governed ring, so no solve is repeated to draw a state
that has already been measured.

Every field a panel draws is written here, including the reason a surface is
absent: a figure that drops one silently reads as though the read had failed to
matter, when the point of the read is that it can fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from nova.equilibrium.connectivity_boundary import LCFS_ANGLES, host_boundary_read
from nova.media.poloidal import inside_wall_units

PROBE = Path(__file__).resolve().parent.parent / "diiid_forward_convergence"
sys.path.insert(0, str(PROBE))

from boundary_competition import Grid  # noqa: E402
from render_panels import _oriented  # noqa: E402


def _governed_ring() -> tuple[np.ndarray, str]:
    """Read the governed limiter ring through the machine description.

    The ring is read from the machine artifact the solver itself consumes, so a
    panel and a solve cannot disagree about which wall they mean.
    """

    try:
        import measure_branches  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - environment fault
        raise SystemExit(
            f"cannot import measure_branches from {PROBE}: {error}"
        ) from error
    bench = getattr(measure_branches, "bench", None)
    if bench is None:  # pragma: no cover - environment fault
        raise SystemExit(
            f"{PROBE / 'measure_branches.py'} exposes no bench module; the "
            "governed limiter is read through measure_branches.bench"
        )
    ring, receipt = bench._physical_wall_ring(
        bench.DEFAULT_MACHINE_ARTIFACT_CACHE,
        bench.DEFAULT_MACHINE_ARTIFACT_DIGEST,
    )
    return np.asarray(ring, dtype=float), json.dumps(receipt, default=str)[:400]


def _efit_axis(
    label: np.ndarray, radius: np.ndarray, height: np.ndarray, boundary: np.ndarray
) -> np.ndarray:
    """Locate the label field's extremum inside its own contour.

    The artifact carries the frame's EFIT flux on the same lattice as the solve,
    and its contour is drawn from that field, so the extremum inside the
    contour is the reference magnetic axis.  The lattice is coarse (33 nodes
    over 1.7 m), which is why the panels report it to the centimetre.
    """

    grid_radius, grid_height = np.meshgrid(radius, height, indexing="ij")
    inside = np.asarray(
        inside_wall_units(
            np.column_stack((grid_radius.ravel(), grid_height.ravel())), boundary
        ),
        dtype=bool,
    ).reshape(label.shape)
    if not inside.any():
        raise SystemExit("the EFIT contour encloses no lattice node")
    masked = np.where(inside, label, np.nan)
    index = np.unravel_index(np.nanargmax(masked), label.shape)
    return np.asarray([grid_radius[index], grid_height[index]], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    limiter, limiter_source = _governed_ring()
    data = np.load(arguments.artifact)
    key = (
        f"frames.{arguments.frame}.branches.{arguments.branch}.arms.default._flux_grid"
    )
    if key not in data.files:
        raise SystemExit(f"{arguments.artifact} carries no {key}")
    flux = np.asarray(data[key], dtype=float)
    label = np.asarray(data["label"], dtype=float)
    label_boundary = np.asarray(data["label_boundary"], dtype=float)
    radius = np.asarray(data["radius"], dtype=float)
    height = np.asarray(data["height"], dtype=float)
    flux = _oriented(flux, radius, height)
    grid = Grid(radius, height, limiter)

    record = json.loads(arguments.record.read_text())
    arm = record["frames"][str(arguments.frame)]["branches"][arguments.branch]["arms"][
        "default"
    ]
    axis = tuple(float(value) for value in arm["presence"]["axis_rz_m"])
    # The read returns one radius per angle of its own LCFS_ANGLES set, which is
    # eight; the probe's finer SURFACE_ANGLES is for its own diagrid read and
    # does not describe this one.
    angles = np.asarray(LCFS_ANGLES, dtype=float)
    try:
        boundary = host_boundary_read(flux, grid, axis)
    except (ValueError, RuntimeError) as error:
        surface = np.zeros((0, 2))
        refusal = f"the read raised {type(error).__name__}: {error}"
        margin = float("nan")
        found = False
    else:
        radii = np.asarray(boundary.radii, dtype=float)
        finite = np.isfinite(radii)
        surface = np.column_stack(
            (
                axis[0] + radii[finite] * np.cos(angles[finite]),
                axis[1] + radii[finite] * np.sin(angles[finite]),
            )
        )
        margin = float(boundary.class_margin)
        found = bool(boundary.found)
        if found:
            refusal = ""
        elif np.isfinite(margin):
            refusal = (
                f"the read traced {int(finite.sum())} of {radii.size} ray radii "
                f"with class margin {margin:.3f}, too few to close a surface"
            )
        else:
            refusal = (
                f"the read traced {int(finite.sum())} of {radii.size} ray radii "
                "with an infinite class margin, so it returns a fragment"
            )

    np.savez_compressed(
        arguments.output,
        limiter=limiter,
        limiter_source=limiter_source,
        surface=surface,
        surface_found=found,
        surface_class_margin=margin,
        surface_refusal=refusal,
        efit_axis=_efit_axis(label, radius, height, label_boundary),
        efit_x_point=label_boundary[int(np.argmin(label_boundary[:, 1]))],
        efit_boundary=label_boundary,
        axis=np.asarray(axis, dtype=float),
        grid_r=np.asarray([float(radius[0]), float(radius[-1])]),
        grid_z=np.asarray([float(height[0]), float(height[-1])]),
    )
    print(
        f"WROTE {arguments.output} limiter={limiter.shape} "
        f"surface={surface.shape} found={found} margin={margin:.3f} "
        f"refusal={refusal[:90]}"
    )


if __name__ == "__main__":
    main()
