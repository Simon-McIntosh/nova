"""Show the read-back check refusing the superseded receipt's own panels.

The superseded receipt took its contact as the largest flux anywhere on the
limiter ring.  On a diverted equilibrium the ring under the X-point carries more
flux than the separatrix, inside the private-flux region, so the recorded
contact sits on the divertor floor while the surface the same receipt drew is
the level set of that flux, which is a small closed ring around the plasma.  The
two never meet, and this measures the distance between them from the stored
terminal grid and the superseded receipt.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boundary_competition import _archive  # noqa: E402
from boundary_readback import _polyline_distance  # noqa: E402


def _level(flux, radius, height, level):
    """Every contour segment of one level, as (N, 2) polylines."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    figure = pyplot.figure()
    axes = figure.add_subplot(111)
    lines = axes.contour(radius, height, flux, levels=[level]).allsegs[0]
    pyplot.close(figure)
    return [np.asarray(segment, dtype=float) for segment in lines]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--arm", default="179:diverted:default")
    arguments = parser.parse_args()

    frame, branch, arm = arguments.arm.split(":")
    receipt = json.loads(arguments.receipt.read_text())
    entry = receipt["frames"][frame][branch][arm]
    contact = np.asarray(entry["contact_rz_m"], dtype=float)
    level = float(entry["bounding_flux"])

    grid = _archive(arguments.artifact_dir / f"frame-{frame}-{branch}-{arm}.npz")
    segments = _level(grid["flux"], grid["radius"], grid["height"], level)
    print(f"superseded contact=({contact[0]:.3f},{contact[1]:+.3f}) level={level:+.4f}")
    print(f"the level set is drawn as {len(segments)} polyline(s)")
    best = None
    for segment in segments:
        distance = _polyline_distance(contact, segment)
        if best is None or distance < best[0]:
            best = (distance, segment)
    distance, segment = best
    cell = float(np.abs(np.diff(grid["radius"])).max())
    print(
        f"nearest drawn surface passes {distance:.3f} m from the recorded contact "
        f"(one grid cell is {cell:.3f} m)"
    )
    refused = distance > cell
    print("READBACK REFUSES" if refused else "READBACK ACCEPTS")
    return 0 if refused else 1


if __name__ == "__main__":
    raise SystemExit(main())
