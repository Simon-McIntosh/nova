"""The generating state's own flux on the study lattice, with its nulls.

The contour panel sets both codes' reconstructions against the state the rows
were generated from, so that state's field has to be drawn on the same lattice
and in the same gauge.  The persisted seed carries the flux at every cell but
no coordinates, so the machine's geometry is loaded once here to place it; the
resampling is the study's own, so the reference panel and the reconstructed
panels are drawn from one rule.

The generating X-point is found on the lattice: the persisted reference carries
the axis but no saddle, and the saddle of a flux grid is the interior
stationary point whose Hessian determinant is negative.  The axis comes from
the persisted reference, and both nulls are reported with the flux the
interpolated field takes there.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import study

STUDY = study.STUDY


class _Topology:
    def __init__(self, axis, flux_axis, flux_boundary):
        self.axis = np.asarray(axis, dtype=float)
        self.axis_flux = float(flux_axis)
        self.boundary_flux = float(flux_boundary)
        self.flux_span = self.boundary_flux - self.axis_flux


class _State:
    """The minimum a flux map reads: the cell flux and its topology."""

    def __init__(self, flux, axis, flux_axis, flux_boundary):
        self.flux = np.asarray(flux, dtype=float)
        self.topology = _Topology(axis, flux_axis, flux_boundary)


def _hessian(values, radius, height, i, j):
    """Second derivatives of psi at one lattice point, from its neighbours."""
    step_r = radius[i + 1] - radius[i - 1]
    step_z = height[j + 1] - height[j - 1]
    drr = (values[i + 1, j] - 2.0 * values[i, j] + values[i - 1, j]) / (
        0.5 * step_r
    ) ** 2
    dzz = (values[i, j + 1] - 2.0 * values[i, j] + values[i, j - 1]) / (
        0.5 * step_z
    ) ** 2
    mixed = (
        values[i + 1, j + 1]
        - values[i + 1, j - 1]
        - values[i - 1, j + 1]
        + values[i - 1, j - 1]
    ) / (step_r * step_z)
    return drr, dzz, mixed


def inside_vessel(radius, height, wall_r, wall_z):
    """Lattice points the limiter ring encloses."""
    import matplotlib.path

    ring = matplotlib.path.Path(np.column_stack([wall_r, wall_z]))
    grid_r, grid_z = np.meshgrid(radius, height, indexing="ij")
    return ring.contains_points(
        np.column_stack([grid_r.ravel(), grid_z.ravel()])
    ).reshape(grid_r.shape)


def saddles(psi, radius, height, axis, inside, exclude=0.6, count=4):
    """Interior stationary points of a flux grid whose Hessian is indefinite.

    A stationary point is a local minimum of the gradient magnitude, and a
    saddle is a stationary point with a negative Hessian determinant.  Points
    within ``exclude`` metres of the axis are skipped: the axis is a stationary
    point of the same field and is not the saddle being looked for.
    """
    dphi_dr, dphi_dz = np.gradient(psi, radius, height)
    magnitude = np.hypot(dphi_dr, dphi_dz)
    interior = np.zeros(psi.shape, bool)
    interior[1:-1, 1:-1] = True
    interior = interior & np.asarray(inside, bool)
    seed_r = radius[:, None] - float(axis[0])
    seed_z = height[None, :] - float(axis[1])
    away = np.hypot(seed_r, seed_z) > exclude
    candidates = np.where(interior & away, magnitude, np.inf)
    found = []
    for _ in range(count):
        if not np.isfinite(candidates).any():
            break
        i, j = np.unravel_index(int(np.argmin(candidates)), candidates.shape)
        candidates[i, j] = np.inf
        drr, dzz, mixed = _hessian(psi, radius, height, i, j)
        if drr * dzz - mixed * mixed >= 0.0:
            continue
        # Neighbouring lattice points resolve one stationary point, so an
        # accepted saddle excludes its own neighbourhood.
        point = np.asarray([float(radius[i]), float(height[j])])
        if any(np.hypot(*(point - other)) < 0.5 for other in found):
            continue
        found.append(point.tolist())
    return np.asarray(found, dtype=float).reshape(-1, 2)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=500)
    args = parser.parse_args(argv)
    case = study.geometry_case(args.cells)
    reference = json.loads((STUDY / "generating.json").read_text())
    stored = np.load(STUDY / "generated.npz")
    state = _State(
        stored["flux"],
        reference["axis"],
        reference["flux_axis"],
        reference["flux_boundary"],
    )
    data = dict(case)
    data["reference"] = reference
    flux_map = study.nova_flux_map(data, state)
    psi = np.asarray(flux_map.flux, dtype=float)
    axis = np.asarray(reference["axis"], dtype=float)
    ring = np.asarray(
        np.load(STUDY / "conf" / "machine-wall.npz")["wall"], dtype=float
    )
    wall_r, wall_z = ring[:, 0], ring[:, 1]
    inside = inside_vessel(study.GRID_RADIUS, study.GRID_HEIGHT, wall_r, wall_z)
    found = saddles(psi, study.GRID_RADIUS, study.GRID_HEIGHT, axis, inside)
    # Only a saddle at or below the boundary flux bounds the plasma, so only
    # those are the generating state's admitted nulls; a stationary point
    # outside the boundary is a private-flux feature of the same field.
    x_points = np.asarray(
        [
            point
            for point in found
            if float(
                psi[
                    int(np.argmin(np.abs(study.GRID_RADIUS - point[0]))),
                    int(np.argmin(np.abs(study.GRID_HEIGHT - point[1]))),
                ]
            )
            < float(reference["flux_boundary"])
        ],
        dtype=float,
    ).reshape(-1, 2)
    # One additive constant settles the lattice interpolation's residual at the
    # axis, so the reference contours sit on the same gauge as the persisted
    # flux_axis the panels' level array is built from.
    distance0 = np.hypot(
        study.GRID_RADIUS[:, None] - axis[0], study.GRID_HEIGHT[None, :] - axis[1]
    )
    i0, j0 = np.unravel_index(int(np.argmin(distance0)), psi.shape)
    shift = float(reference["flux_axis"]) - float(psi[i0, j0])
    psi = psi + shift
    np.savez(
        STUDY / "reference-grid.npz",
        radius=study.GRID_RADIUS,
        height=study.GRID_HEIGHT,
        psi=psi,
        axis=axis,
        x_points=x_points,
        flux_axis=np.asarray(float(reference["flux_axis"])),
        flux_boundary=np.asarray(float(reference["flux_boundary"])),
    )
    distance = np.hypot(
        study.GRID_RADIUS[:, None] - axis[0],
        study.GRID_HEIGHT[None, :] - axis[1],
    )
    i, j = np.unravel_index(int(np.argmin(distance)), psi.shape)
    print(
        "REFERENCE_GRID at_axis=%.4f persisted=%.4f saddles=%s"
        % (psi[i, j], float(reference["flux_axis"]), x_points.tolist()),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())