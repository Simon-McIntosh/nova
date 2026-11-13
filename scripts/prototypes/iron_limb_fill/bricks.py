"""Solve linear iron with six discrete limbs as uniformly magnetised bricks.

The circuit is WEST-like: a round central column, a hexagonal hub, six radial
arms of constant rectangular section and six vertical return limbs, mirrored
about the midplane. One sixty-degree sector above the midplane carries the
unknowns; rotation and mirror images supply the rest. Each brick holds one
magnetisation vector, collocated at its centroid, and its field is the closed
form of six uniformly charged rectangles. The drive is axisymmetric and
up-down symmetric, so the toroidal average of this solution is what an
axisymmetric iron model has to reproduce.

Usage: bricks.py --output <dir> --size <brick edge m> [--mu 50 200 1000]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

MU0 = 4e-7 * np.pi
COLUMN_RADIUS = 0.664
ARM_HALF_WIDTH = 0.4
HUB_APOTHEM = ARM_HALF_WIDTH / np.tan(np.pi / 6)
ARM_Z = (2.4, 3.6)
LIMB_X = (4.8, 6.0)
SECTORS = 6


def sector_bricks(size: float) -> tuple[np.ndarray, np.ndarray]:
    """Return brick centres and the common half-edge for the upper sector."""
    x = np.arange(size / 2, LIMB_X[1], size)
    y = np.arange(-2.0 + size / 2, 2.0, size)
    z = np.arange(size / 2, ARM_Z[1], size)
    xx, yy, zz = (a.ravel() for a in np.meshgrid(x, y, z, indexing="ij"))
    radius, angle = np.hypot(xx, yy), np.arctan2(yy, xx)
    wedge = np.abs(angle) < np.pi / SECTORS
    column = wedge & (radius < COLUMN_RADIUS) & (zz < ARM_Z[0])
    hub = wedge & (xx < HUB_APOTHEM) & (zz > ARM_Z[0])
    arm = (xx > HUB_APOTHEM) & (np.abs(yy) < ARM_HALF_WIDTH) & (zz > ARM_Z[0])
    limb = (xx > LIMB_X[0]) & (np.abs(yy) < ARM_HALF_WIDTH) & (zz < ARM_Z[0])
    keep = column | hub | arm | limb
    return np.column_stack((xx, yy, zz))[keep], np.full(3, size / 2)


def _rectangle(u, v, w, half_u, half_v):
    """Return (H_u, H_v, H_w) per unit charge density of a rectangle at w = 0."""
    hu = np.zeros_like(u)
    hv = np.zeros_like(u)
    hw = np.zeros_like(u)
    for su in (-1.0, 1.0):
        for sv in (-1.0, 1.0):
            du = u - su * half_u
            dv = v - sv * half_v
            r = np.sqrt(du * du + dv * dv + w * w)
            sign = su * sv
            # log(d + r) loses digits for d < 0; use the conjugate form there.
            hu -= sign * np.where(
                dv > 0, np.log(dv + r), -np.log(r - dv) + np.log(du * du + w * w)
            )
            hv -= sign * np.where(
                du > 0, np.log(du + r), -np.log(r - du) + np.log(dv * dv + w * w)
            )
            hw += sign * np.arctan(du * dv / (np.where(w == 0, 1e-300, w) * r))
    return hu / (4 * np.pi), hv / (4 * np.pi), hw / (4 * np.pi)


def brick_tensor(offset: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Return H = T M at offsets (..., 3) from the centre of a uniform brick."""
    tensor = np.zeros(offset.shape[:-1] + (3, 3))
    for axis in range(3):
        first, second = (axis + 1) % 3, (axis + 2) % 3
        for side in (-1.0, 1.0):
            hu, hv, hw = _rectangle(
                offset[..., first],
                offset[..., second],
                offset[..., axis] - side * half[axis],
                half[first],
                half[second],
            )
            # The face with outward normal side*axis carries charge side*M_axis.
            tensor[..., first, axis] += side * hu
            tensor[..., second, axis] += side * hv
            tensor[..., axis, axis] += side * hw
    return tensor


def images():
    """Yield (frame, moment map) for the twelve rotation and mirror images."""
    for k in range(SECTORS):
        angle = 2 * np.pi * k / SECTORS
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        for mirror in (1.0, -1.0):
            frame = rotation @ np.diag([1, 1, mirror])
            # B_R is odd and B_Z even about the midplane, and M follows B.
            moment = rotation @ np.diag([mirror, mirror, 1.0])
            yield frame, moment


def _rows(task):
    targets, centres, half, nodes = task
    rows = np.zeros((len(targets), 3, len(centres), 3))
    for node in nodes:
        for frame, moment in images():
            offset = ((targets + node) @ frame)[:, None, :] - centres[None, :, :]
            rows += np.einsum(
                "ab,tnbc,cd->tand",
                frame,
                brick_tensor(offset, half),
                frame.T @ moment,
                optimize=True,
            )
    return rows / len(nodes)


def gauss_nodes(half, order):
    """Return equal-weight offsets of a tensor midpoint rule inside a brick."""
    if order == 1:
        return np.zeros((1, 3))
    axis = ((np.arange(order) + 0.5) / order - 0.5) * 2
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    return grid * half


def response(targets, centres, half, pool=None, nodes=None) -> np.ndarray:
    """Return H at targets per sector magnetisation, shape (3T, 3N).

    With ``nodes`` the field is averaged over those offsets about each target,
    which tests the equation over the brick volume instead of at its centre.
    """
    nodes = np.zeros((1, 3)) if nodes is None else nodes
    tasks = [
        (targets[start : start + 32], centres, half, nodes)
        for start in range(0, len(targets), 32)
    ]
    rows = pool.map(_rows, tasks) if pool else map(_rows, tasks)
    return np.concatenate(list(rows)).reshape(3 * len(targets), 3 * len(centres))


def drive(sources):
    """Return a Cartesian B(points) for axisymmetric filaments (r, z, amps)."""
    from nova.biot.greens import greens_bz_br

    radius, height, amps = (np.asarray(sources[k]) for k in ("r", "z", "amps"))

    def evaluate(points):
        r = np.hypot(points[:, 0], points[:, 1])
        result = np.empty_like(points)
        for start in range(0, len(points), 256):
            stop = start + 256
            bz, br = greens_bz_br(
                r[start:stop, None], points[start:stop, 2, None], radius, height
            )
            radial = br @ amps
            safe = np.where(r[start:stop] > 0, r[start:stop], 1.0)
            result[start:stop, 0] = radial * points[start:stop, 0] / safe
            result[start:stop, 1] = radial * points[start:stop, 1] / safe
            result[start:stop, 2] = bz @ amps
        return result

    return evaluate


def footprint(centres, half, radius, z, samples=7200):
    """Return the toroidal fraction of iron at each (radius, z), with overlaps."""
    angle = (np.arange(samples) + 0.5) * (2 * np.pi / SECTORS) / samples
    angle -= np.pi / SECTORS
    fraction = np.zeros(len(radius))
    layer = np.abs(centres[:, 2][None] - np.abs(z)[:, None]) < half[2]
    for k in (-1, 0, 1):
        turn = angle + 2 * np.pi * k / SECTORS
        for i, r in enumerate(radius):
            bricks = centres[layer[i]]
            x, y = r * np.cos(turn), r * np.sin(turn)
            inside = (np.abs(x[:, None] - bricks[None, :, 0]) < half[0]) & (
                np.abs(y[:, None] - bricks[None, :, 1]) < half[1]
            )
            fraction[i] += inside.sum() / samples
    return fraction


def toroidal_average(points_rz, field, samples=24):
    """Return the mean (B_R, B_Z) over one sector at (R, Z) points."""
    angle = (np.arange(samples) + 0.5) * (2 * np.pi / SECTORS) / samples
    angle -= np.pi / SECTORS
    total = np.zeros((len(points_rz), 2))
    resolved = []
    for phi in angle:
        c, s = np.cos(phi), np.sin(phi)
        xyz = np.column_stack(
            (points_rz[:, 0] * c, points_rz[:, 0] * s, points_rz[:, 1])
        )
        b = field(xyz)
        cylindrical = np.column_stack((b[:, 0] * c + b[:, 1] * s, b[:, 2]))
        resolved.append(cylindrical)
        total += cylindrical / samples
    return total, np.asarray(resolved), angle


def self_test():
    """Check the brick field against a dipole and its interior demagnetisation."""
    half = np.array([0.1, 0.15, 0.2])
    far = np.array([[3.0, 2.0, 1.5]])
    tensor = brick_tensor(far, half)[0]
    volume = 8 * half.prod()
    r = np.linalg.norm(far)
    unit = far[0] / r
    dipole = volume / (4 * np.pi * r**3) * (3 * np.outer(unit, unit) - np.eye(3))
    assert np.allclose(tensor, dipole, rtol=5e-3, atol=1e-9), (tensor, dipole)
    inside = brick_tensor(np.array([[0.01, -0.02, 0.03]]), half)[0]
    assert abs(np.trace(inside) + 1) < 1e-12, np.trace(inside)
    cube = brick_tensor(np.zeros((1, 3)), np.full(3, 0.1))[0]
    assert np.allclose(cube, -np.eye(3) / 3, atol=1e-12), cube
    print("SELFTEST dipole, interior trace and cube demagnetisation pass", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=float, required=True)
    parser.add_argument("--mu", type=float, nargs="+", default=[50.0, 200.0, 1000.0])
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--order", type=int, default=1)
    args = parser.parse_args()
    self_test()
    case = json.loads(args.case.read_text())
    probes = np.asarray(case["probes"])
    external = drive(case["sources"])
    centres, half = sector_bricks(args.size)
    print(f"BRICKS size={args.size} count={len(centres)}", flush=True)
    started = time.perf_counter()
    pool = multiprocessing.Pool(args.workers) if args.workers > 1 else None
    nodes = gauss_nodes(half, args.order)
    system = response(centres, centres, half, pool, nodes)
    print(f"ASSEMBLED seconds={time.perf_counter() - started:.1f}", flush=True)
    drive_field = (
        sum(external(centres + node) for node in nodes).reshape(-1) / len(nodes) / MU0
    )
    rings = np.asarray(case["footprint_rz"])
    tag = f"{args.size:.4g}-order{args.order}"
    record = dict(
        size=args.size,
        order=args.order,
        bricks=len(centres),
        volume_m3=float(12 * len(centres) * args.size**3),
        footprint_rz=rings.tolist(),
        footprint=footprint(centres, half, rings[:, 0], rings[:, 1]).tolist(),
        probes=probes.tolist(),
        rows=[],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / f"centres-{tag}.npy", centres)
    for mu in args.mu:
        chi = mu - 1
        moment = np.linalg.solve(
            np.eye(system.shape[0]) - chi * system, chi * drive_field
        )
        check = np.linalg.norm(moment - chi * (drive_field + system @ moment))
        np.save(args.output / f"moment-{tag}-mu{mu:g}.npy", moment)

        def iron(points, moment=moment):
            return MU0 * (response(points, centres, half, pool) @ moment).reshape(-1, 3)

        average, resolved, angle = toroidal_average(probes, iron)
        air = toroidal_average(probes, external, samples=1)[0]
        midplane = np.abs(centres[:, 2] - args.size / 2) < 1e-9
        core = midplane & (np.hypot(centres[:, 0], centres[:, 1]) < COLUMN_RADIUS)
        h_z = drive_field.reshape(-1, 3)[:, 2] + (system @ moment).reshape(-1, 3)[:, 2]
        b_z = MU0 * (h_z + moment.reshape(-1, 3)[:, 2])
        record["rows"].append(
            dict(
                mu=mu,
                residual=float(check / np.linalg.norm(moment)),
                iron_average_T=average.tolist(),
                air_T=air.tolist(),
                iron_resolved_T=resolved.tolist(),
                angle=angle.tolist(),
                column_flux_Wb=float(SECTORS * b_z[core].sum() * args.size**2),
                column_mean_bz_T=float(b_z[core].mean()),
            )
        )
        print(
            f"SOLVED mu={mu:g} residual={record['rows'][-1]['residual']:.2e} "
            f"flux={record['rows'][-1]['column_flux_Wb']:.5f}",
            flush=True,
        )
        (args.output / f"bricks-{tag}.json").write_text(
            json.dumps(record, indent=1) + "\n"
        )


if __name__ == "__main__":
    main()
