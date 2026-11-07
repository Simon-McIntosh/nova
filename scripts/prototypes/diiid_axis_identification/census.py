"""Locate every polished critical point of the frame 144 limited seed.

The refusals in the convergence record are raised at axis identification, so
the question this answers is whether the seed really carries no admissible
axis or whether the identification rejected one that is there.  The census is
independent of the product topology read: it forms the gradient of the seed
flux on the lattice, locates every zero crossing of both gradient components
to sub-cell precision, classifies each by the eigenvalues of the local Hessian
(Morse index), and closes the account with the winding number of the gradient
direction around the domain boundary.

Coverage is the check that makes the census trustworthy.  For a vector field
on a domain the boundary winding number equals the sum of the indices of the
enclosed zeros, extrema counting +1 and saddles -1.  A mismatch means at least
one critical point inside the domain was missed, and no absence claim may be
rested on an incomplete census.
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
PROTOTYPES = REPOSITORY / "scripts" / "prototypes" / "diiid_forward_convergence"
if str(PROTOTYPES) not in sys.path:
    sys.path.insert(0, str(PROTOTYPES))

from measure_branches import _prepare_frame  # noqa: E402
from nova.equilibrium.topology import TopologyClass  # noqa: E402
from nova.jax.config import configure_dtypes  # noqa: E402


def _bilinear_zero_crossings(component, axis):
    """Return refined zero crossings of one gradient component.

    A zero crossing on an edge is located by linear interpolation between the
    two samples that bracket it, which is second-order accurate on a smooth
    field and exact for a bilinear cell.
    """
    crossings = []
    first = np.moveaxis(component, axis, 0)
    shape = first.shape
    for i in range(shape[0] - 1):
        low, high = first[i], first[i + 1]
        straddle = (low * high) < 0.0
        if not np.any(straddle):
            continue
        weight = np.abs(low[straddle]) / (
            np.abs(low[straddle]) + np.abs(high[straddle])
        )
        for index, w in zip(np.argwhere(straddle), weight):
            index = tuple(index)
            crossings.append((i + float(w), index))
    return crossings


def _critical_points(psi, radius, height):
    """Return every critical point with its refined position and index."""
    dr = np.gradient(psi, radius, axis=0)
    dz = np.gradient(psi, height, axis=1)
    drr = np.gradient(dr, radius, axis=0)
    dzz = np.gradient(dz, height, axis=1)
    drz = np.gradient(dr, height, axis=1)

    # A critical point is a cell corner where both components change sign
    # between this corner and the corners below/right of it.
    sign_r = np.sign(dr[:-1, :-1]) + np.sign(dr[1:, :-1]) + np.sign(dr[:-1, 1:]) + np.sign(dr[1:, 1:])
    sign_z = np.sign(dz[:-1, :-1]) + np.sign(dz[1:, :-1]) + np.sign(dz[:-1, 1:]) + np.sign(dz[1:, 1:])
    candidate = (np.abs(sign_r) < 4) & (np.abs(sign_z) < 4)

    points = []
    for i, j in np.argwhere(candidate):
        # Refine inside the cell by a least-squares linear solve on the four
        # corner gradients (bilinear interpolation of a linear model).
        corners = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]
        rows, rhs_r, rhs_z = [], [], []
        for ci, cj in corners:
            rows.append([radius[ci] - radius[i], height[cj] - height[j], 1.0])
            rhs_r.append(dr[ci, cj])
            rhs_z.append(dz[ci, cj])
        matrix = np.asarray(rows)
        solution_r, *_ = np.linalg.lstsq(matrix, np.asarray(rhs_r), rcond=None)
        solution_z, *_ = np.linalg.lstsq(matrix, np.asarray(rhs_z), rcond=None)
        a_r, c_r, b_r = solution_r
        a_z, c_z, b_z = solution_z
        determinant = a_r * c_z - a_z * c_r
        if abs(determinant) < 1e-30:
            continue
        delta_r = (-b_r * c_z + b_z * c_r) / determinant
        delta_z = (-a_r * b_z + a_z * b_r) / determinant
        if not (-1.0 <= delta_r <= 1.0 and -1.0 <= delta_z <= 1.0):
            continue
        r0 = radius[i] + delta_r
        z0 = height[j] + delta_z

        # Hessian at the refined point, by bilinear weight over the corners.
        weights = [
            (1 - delta_r) * (1 - delta_z),
            delta_r * (1 - delta_z),
            (1 - delta_r) * delta_z,
            delta_r * delta_z,
        ]
        h_rr = sum(w * drr[ci, cj] for w, (ci, cj) in zip(weights, corners))
        h_zz = sum(w * dzz[ci, cj] for w, (ci, cj) in zip(weights, corners))
        h_rz = sum(w * drz[ci, cj] for w, (ci, cj) in zip(weights, corners))
        eigenvalues = np.linalg.eigvalsh(np.array([[h_rr, h_rz], [h_rz, h_zz]]))
        product = float(eigenvalues[0] * eigenvalues[1])
        if product > 0:
            index = 1
            kind = "extremum"
        elif product < 0:
            index = -1
            kind = "saddle"
        else:
            index = 0
            kind = "degenerate"
        points.append(
            {
                "r": float(r0),
                "z": float(z0),
                "psi": float(np.interp(r0, radius, psi[:, j])),
                "kind": kind,
                "morse_index": index,
                "eigenvalues": [float(eigenvalues[0]), float(eigenvalues[1])],
                "minimum": bool(product > 0 and eigenvalues[0] > 0),
                "maximum": bool(product > 0 and eigenvalues[1] < 0),
            }
        )

    # Merge duplicates that both bracket the same zero (adjacent cells).
    merged = []
    for point in sorted(points, key=lambda item: (item["r"], item["z"])):
        for kept in merged:
            if (
                abs(kept["r"] - point["r"]) <= 2 * abs(np.median(np.diff(radius)))
                and abs(kept["z"] - point["z"]) <= 2 * abs(np.median(np.diff(height)))
            ):
                break
        else:
            merged.append(point)
    return merged


def _boundary_winding(psi, radius, height):
    """Return the winding number of the gradient around the domain boundary."""
    dr = np.gradient(psi, radius, axis=0)
    dz = np.gradient(psi, height, axis=1)

    def trace(r0, z0, r1, z1, samples):
        rs = np.linspace(r0, r1, samples, endpoint=False)
        zs = np.linspace(z0, z1, samples, endpoint=False)
        rr = np.interp(rs, radius, np.arange(radius.size))
        zz = np.interp(zs, height, np.arange(height.size))
        from scipy.ndimage import map_coordinates

        return (
            map_coordinates(dr, [rr, zz], order=1),
            map_coordinates(dz, [rr, zz], order=1),
        )

    r_lo, r_hi = radius[0], radius[-1]
    z_lo, z_hi = height[0], height[-1]
    samples = 400
    pieces = [
        trace(r_lo, z_lo, r_hi, z_lo, samples),
        trace(r_hi, z_lo, r_hi, z_hi, samples),
        trace(r_hi, z_hi, r_lo, z_hi, samples),
        trace(r_lo, z_hi, r_lo, z_lo, samples),
    ]
    gr = np.concatenate([piece[0] for piece in pieces])
    gz = np.concatenate([piece[1] for piece in pieces])
    angle = np.unwrap(np.arctan2(gz, gr))
    return float((angle[-1] - angle[0]) / (2 * np.pi))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=int, default=144)
    parser.add_argument("--expansion", type=float, default=float(__import__("benchmarks.diiid_forward_gs_match", fromlist=["x"]).REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION))
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()

    configure_dtypes()
    prepared = _prepare_frame(arguments.frame, arguments.expansion)
    cold = prepared["cold"]
    seed = np.asarray(cold.branches.flux[int(TopologyClass.LIMITED)], dtype=float)
    lattice = prepared["profile"].lattice
    radius = np.asarray(lattice.radius, dtype=float)
    height = np.asarray(lattice.height, dtype=float)
    node_count = int(lattice.node_count)
    if seed.size < node_count:
        raise SystemExit(f"seed carries {seed.size} nodes, shorter than the {node_count} plasma nodes")
    # The state vector carries the wall nodes after the plasma lattice, so the
    # plasma flux is the leading slice rather than the whole array.
    seed = np.reshape(seed[:node_count], (radius.size, height.size), order="C")

    points = _critical_points(seed, radius, height)
    winding = _boundary_winding(seed, radius, height)
    index_sum = int(sum(point["morse_index"] for point in points))
    residual = int(round(winding)) - index_sum

    record = {
        "frame": arguments.frame,
        "branch": "limited",
        "grid": {"radius": [float(radius[0]), float(radius[-1]), radius.size],
                 "height": [float(height[0]), float(height[-1]), height.size]},
        "seed_min": float(seed.min()),
        "seed_max": float(seed.max()),
        "critical_points": points,
        "extremum_count": sum(point["kind"] == "extremum" for point in points),
        "saddle_count": sum(point["kind"] == "saddle" for point in points),
        "census_index_sum": index_sum,
        "boundary_winding": winding,
        "winding_minus_index_sum": residual,
        "coverage_ok": residual == 0,
        "efit_axis": [float(v) for v in prepared["axis"]],
    }
    text = json.dumps(record, indent=2, sort_keys=True)
    if arguments.output:
        arguments.output.write_text(text)
    print(text)
    return 0 if residual == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())