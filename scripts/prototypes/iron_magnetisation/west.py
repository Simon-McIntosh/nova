"""WEST 62751 iron response with the edge-sheet magnetisation prototype.

Sources are the machine-description coils (each rectangle split into 6 x 6
filaments) at their pulse currents and the nova hex plasma cells as filaments at
their prescribed currents, read from the stray-field solve. The iron is the six
WEST iron_core segments meshed into square cells, with the CEA B-H table from the
same IDS. Outputs |B| along the three outboard lines with and without iron.

Usage: west.py <hex solve pickle> <out.pkl> [cell size m]
"""

from __future__ import annotations

import logging
import pickle
import sys
from time import perf_counter

import numpy as np
import shapely

from magnetisation import Material, interaction, mesh, ring_field, solve, solve_newton
from nova.imas.machine import StaticMachineDescription

logging.disable(logging.WARNING)
solve_path, out_path = sys.argv[1], sys.argv[2]
size = float(sys.argv[3]) if len(sys.argv) > 3 else 0.08
data = pickle.load(open(solve_path, "rb"))
MACHINE = "/home/ITER/mcintos/Code/imas-efit/tests/data/imas/WEST/machine/dd4-1-0"
machine = StaticMachineDescription.read_iron_core(
    f"imas:hdf5?path={MACHINE}", dd_version="4.1.0"
)
segments = [
    shapely.Polygon(np.column_stack((segment.outline.r, segment.outline.z)))
    for segment in machine.iron_segments
]
table = machine.iron_segments[0].material
material = Material(
    np.asarray(table.b_field, float), np.asarray(table.permeability_relative, float)
)
print(
    "B-H table:",
    np.round(material.b_field, 2)[:: max(1, len(material.b_field) // 8)],
    "mu_r:",
    np.round(material.mu_r, 0)[:: max(1, len(material.mu_r) // 8)],
)

start = perf_counter()
polygons = []
for segment in segments:
    polygons.extend(mesh(segment.buffer(0), size).polygons)
from magnetisation import Cells  # noqa: E402

cells = Cells(tuple(polygons))
centroids = cells.centroids
print(f"iron cells {len(polygons)} at {size} m, area {cells.areas.sum():.2f} m^2")

# filament sources
filament_r, filament_z, filament_i = [], [], []
for coil in data["coils"]:
    current = data["currents"][coil["name"]] * coil["nturn"]
    offsets = (np.arange(6) + 0.5) / 6 - 0.5
    rr, zz = np.meshgrid(
        coil["r"] + offsets * coil["dr"], coil["z"] + offsets * coil["dz"]
    )
    filament_r.append(rr.ravel())
    filament_z.append(zz.ravel())
    filament_i.append(np.full(rr.size, current / rr.size))
for cell, current in zip(data["plasma_cells"], data["plasma_weight"]):
    centre = shapely.Polygon(cell).centroid
    filament_r.append([centre.x])
    filament_z.append([centre.y])
    filament_i.append([current])
filament_r = np.concatenate(filament_r)
filament_z = np.concatenate(filament_z)
filament_i = np.concatenate(filament_i)


def external(points):
    br, bz = ring_field(points[:, 0], points[:, 1], filament_r, filament_z)
    return np.concatenate((br @ filament_i, bz @ filament_i))


radius = data["radius"]
lines = np.vstack(
    [np.column_stack((radius, np.full_like(radius, z))) for z in data["lines"].values()]
)
b_ext = external(centroids)
b_line_air = external(lines)
count = lines.shape[0]
air = np.hypot(b_line_air[:count], b_line_air[count:]).reshape(len(data["lines"]), -1)
far = radius > data["edge"] + 0.3
control = np.max(np.abs(air[:, far] / data["bmag"][:, far] - 1))
print(f"filament sources against the nova solve beyond 0.3 m: max rel {control:.2e}")

cache = f"{out_path}.gram.npz"
try:
    stored = np.load(cache)
    gram, g_lines = stored["gram"], stored["lines"]
except FileNotFoundError:
    gram, g_lines = interaction(cells, centroids), interaction(cells, lines)
    np.savez(cache, gram=gram, lines=g_lines)
print(f"interaction {gram.shape} ready {perf_counter() - start:.0f} s", flush=True)
law = sys.argv[4] if len(sys.argv) > 4 else "table"
if law == "table":
    magnetisation, mu_r, history = solve_newton(gram, b_ext, material)
else:
    constant = float(law)
    magnetisation, mu_r, history = solve(
        gram, b_ext, lambda b: np.full_like(b, constant), iterations=1
    )
print(
    f"material iterations {len(history)}, last change {history[-1]:.1e}, "
    f"mu_r range {mu_r.min():.0f}-{mu_r.max():.0f}"
)
b_line = b_line_air + g_lines @ magnetisation
with_iron = np.hypot(b_line[:count], b_line[count:]).reshape(len(data["lines"]), -1)
b_cell = b_ext + gram @ magnetisation
n = len(polygons)
for k, name in enumerate(data["lines"]):
    ratio = with_iron[k] / air[k]
    picks = [
        np.argmin(np.abs(radius - data["edge"] - s)) for s in (0.1, 0.5, 1, 2, 4, 6)
    ]
    print(
        f"{name:9s} iron/air at +0.1/0.5/1/2/4/6 m: "
        + " ".join(f"{ratio[i]:.3f}" for i in picks)
    )
pickle.dump(
    dict(
        radius=radius,
        lines=data["lines"],
        edge=data["edge"],
        air=air,
        with_iron=with_iron,
        polygons=polygons,
        mu_r=mu_r,
        bmod=np.hypot(b_cell[:n], b_cell[n:]),
        magnetisation=magnetisation,
        history=history,
        size=size,
        control=control,
    ),
    open(out_path, "wb"),
)
print(f"done {perf_counter() - start:.0f} s")
