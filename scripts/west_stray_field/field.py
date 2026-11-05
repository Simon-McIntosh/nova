"""Direct Biot-Savart evaluation of psi, B_R and B_Z for the stray-field figure.

Coils use :func:`nova.biot.greens.cylinder_greens`, the exact rectangular
section ring that nova's coil route evaluates. Each current-carrying hex plasma
cell, clipped to the last closed flux surface, uses
:func:`nova.biot.greens.moment_filament` (centroid filament corrected by the
section's second and third moments) beyond three cell widths of its centroid,
and an equal-area square section through ``cylinder_greens`` within them, so the
map stays finite and smooth through the plasma. Every field component is
evaluated from its own Green's function; nothing is differentiated from psi.

Usage: field.py <hex solve pickle> <out.npz> [grid spacing m]
"""

from __future__ import annotations

import pickle
import sys
from time import perf_counter

import numpy as np
import shapely

from nova.biot.greens import cylinder_greens, moment_filament


def sources(data):
    """Coil rectangles with ampere-turns and LCFS-clipped plasma cells with amps."""
    coils = [
        (c["r"], c["z"], c["dr"], c["dz"], data["currents"][c["name"]] * c["nturn"])
        for c in data["coils"]
    ]
    region = shapely.Polygon(data["boundary"])
    cells = []
    for cell, current in zip(data["plasma_cells"], data["plasma_weight"]):
        part = shapely.Polygon(cell).intersection(region)
        pieces = [p for p in getattr(part, "geoms", [part]) if p.area > 0]
        area = sum(p.area for p in pieces)
        for piece in pieces:
            cells.append(
                (np.asarray(piece.exterior.coords)[:-1], current * piece.area / area)
            )
    return coils, cells


def evaluate(points: np.ndarray, coils, cells) -> np.ndarray:
    """(3, n) array of psi [Wb], B_R [T], B_Z [T] at ``points``."""
    r, z = points[:, 0], points[:, 1]
    total = np.zeros((3, r.size))
    for a, z0, da, dz, current in coils:
        total += current * np.asarray(cylinder_greens(r, z, a, z0, da, dz))
    for vertices, current in cells:
        polygon = shapely.Polygon(vertices)
        centre = np.asarray(polygon.centroid.coords[0])
        side = np.sqrt(polygon.area)
        width = max(np.ptp(vertices[:, 0]), np.ptp(vertices[:, 1]))
        near = np.hypot(r - centre[0], z - centre[1]) < 3 * width
        far = ~near
        if far.any():
            total[:, far] += current * np.asarray(
                moment_filament(r[far], z[far], vertices)
            )
        if near.any():
            total[:, near] += current * np.asarray(
                cylinder_greens(r[near], z[near], centre[0], centre[1], side, side)
            )
    return total


if __name__ == "__main__":
    data = pickle.load(open(sys.argv[1], "rb"))
    spacing = float(sys.argv[3]) if len(sys.argv) > 3 else 0.025
    coils, cells = sources(data)
    start = perf_counter()
    lines = np.vstack(
        [
            np.column_stack((data["radius"], np.full_like(data["radius"], z)))
            for z in data["lines"].values()
        ]
    )
    on_lines = evaluate(lines, coils, cells)
    bmag = np.hypot(on_lines[1], on_lines[2]).reshape(len(data["lines"]), -1)
    far = data["radius"] > data["edge"] + 0.05
    control = np.max(np.abs(bmag[:, far] / data["bmag"][:, far] - 1))
    print(
        f"lines {perf_counter() - start:.1f} s; against the CoilSet solve "
        f"beyond 5 cm of the coil edge: max rel {control:.2e}",
        flush=True,
    )
    x = np.arange(0.05, 12.0 + spacing / 2, spacing)
    zz = np.arange(-3.2, 3.2 + spacing / 2, spacing)
    mesh = np.column_stack([a.ravel() for a in np.meshgrid(x, zz, indexing="ij")])
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    chunks = np.array_split(mesh, 48)
    with ProcessPoolExecutor(12) as pool:
        parts = list(pool.map(partial(evaluate, coils=coils, cells=cells), chunks))
    field = np.concatenate(parts, axis=1).reshape(3, x.size, zz.size)
    print(f"grid {x.size}x{zz.size} at {spacing} m in {perf_counter() - start:.1f} s")
    np.savez(
        sys.argv[2],
        x=x,
        z=zz,
        psi=field[0],
        br=field[1],
        bz=field[2],
        line_bmag=bmag,
        line_control=control,
        spacing=spacing,
    )
