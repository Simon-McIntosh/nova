"""WEST 62751 stray poloidal field outboard of the equatorial PF coils.

Coils come from the WEST machine description (pf_active, DD 4.1.0) and their
currents from the 62751 pf_active IDS. The plasma is nova's hexagonal plasma
grid inside the limiter: each cell carries the NICE current density
(j = R p' + FF'/(mu0 R), through PlasmaSource.prescribe) at its centroid times
its area inside the last closed flux surface, normalised to the NICE plasma
current.

Biot interactions depend on geometry only, so the solved CoilSet is stored
under ``cache`` and reloaded; currents and plasma weights are applied after
the load, which costs nothing. Usage: solve.py <out.pkl> <cache-dir>.
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()

import imas  # noqa: E402
import jax  # noqa: E402
import shapely  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

from nova.frame.coilset import CoilSet  # noqa: E402
from nova.imas.plasma_source import PlasmaSource  # noqa: E402
from nova.imas.machine import StaticMachineDescription  # noqa: E402

logging.disable(logging.WARNING)
ROOT = "/home/ITER/mcintos/Code/imas-efit/tests/data/imas/WEST"
MACHINE = f"{ROOT}/machine/dd4-1-0"
PULSE = f"{ROOT}/62751"
EQUILIBRIUM = f"{ROOT}/62751/validation/nice"
INDEX = 14
PLASMA_CELLS = 2000
OUT, CACHE = sys.argv[1], Path(sys.argv[2])
LINES = {"midplane": 0.0, "Fb plane": -0.637, "Fh plane": 0.637}
GRID = dict(number=6000, limit=[0.2, 12.0, -4.0, 4.0])
CORE = dict(limit=[1.75, 3.15, -0.95, 0.95], shape=(57, 77))


def entry(path, dd):
    return imas.DBEntry(f"imas:hdf5?path={path}", "r", dd_version=dd)


with entry(MACHINE, "4.1.0") as md:
    pf_md = md.get("pf_active", autoconvert=False)
    wall = md.get("wall", autoconvert=False)
coils = [
    dict(
        name=str(coil.name).strip(),
        r=float(element.geometry.rectangle.r),
        z=float(element.geometry.rectangle.z),
        dr=float(element.geometry.rectangle.width),
        dz=float(element.geometry.rectangle.height),
        nturn=float(element.turns_with_sign),
    )
    for coil in pf_md.coil
    for element in coil.element
]
outline = wall.description_2d[0].limiter.unit[0].outline
limiter = np.column_stack((np.asarray(outline.r), np.asarray(outline.z)))
machine = StaticMachineDescription.read_iron_core(
    f"imas:hdf5?path={MACHINE}", dd_version="4.1.0"
)
iron_outlines = [
    np.column_stack((segment.outline.r, segment.outline.z))
    for segment in machine.iron_segments
    if len(segment.outline.r)
]

with entry(EQUILIBRIUM, "3.42.2") as eq_entry, entry(PULSE, "4.1.0") as pulse:
    equilibrium = eq_entry.get("equilibrium", autoconvert=False)
    active = pulse.get("pf_active", autoconvert=False)
    source = PlasmaSource.from_imas(
        equilibrium, active, provenance="WEST 62751 NICE", indices=[INDEX]
    )
slice_ = source.slices[0]
time = slice_.time
currents = dict(zip(source.coils.names, source.coils.at(time)))
prescribed = source.prescribe(0)
print(
    f"t={time:.4f}s  Ip={prescribed.plasma_current / 1e3:.1f} kA  "
    f"backend={jax.default_backend()} devices={jax.devices()}",
    flush=True,
)

edge = max(c["r"] + c["dr"] / 2 for c in coils if c["name"] in ("Fh", "Fb"))
radius = np.linspace(edge, 16.0, 400)
line_targets = np.vstack(
    [np.column_stack((radius, np.full_like(radius, z))) for z in LINES.values()]
)
core_x = np.linspace(*CORE["limit"][:2], CORE["shape"][0])
core_z = np.linspace(*CORE["limit"][2:], CORE["shape"][1])
core_targets = np.column_stack(
    [a.ravel() for a in np.meshgrid(core_x, core_z, indexing="ij")]
)
targets = np.vstack((line_targets, core_targets))

CACHE.mkdir(parents=True, exist_ok=True)
stored = CACHE / "west-62751-coilset.nc"
start = perf_counter()
if stored.is_file():
    coilset = CoilSet(filename=stored.stem, dirname=CACHE).load()
    print(f"loaded solved coilset {stored} in {perf_counter() - start:.1f} s")
else:
    coilset = CoilSet(
        dcoil=-1,
        dplasma=-PLASMA_CELLS,
        tplasma="hex",
        filename=stored.stem,
        dirname=CACHE,
    )
    for c in coils:
        coilset.coil.insert(
            c["r"],
            c["z"],
            c["dr"],
            c["dz"],
            nturn=c["nturn"],
            name=c["name"],
            part="pf",
        )
    coilset.firstwall.insert(
        [limiter[:, 0].tolist(), limiter[:, 1].tolist()], turn="hex"
    )
    coilset.grid.solve(GRID["number"], limit=GRID["limit"])
    print(f"grid solved {perf_counter() - start:.1f} s", flush=True)
    coilset.point.solve(targets)
    print(f"points solved {perf_counter() - start:.1f} s", flush=True)
    coilset.store()

# currents: coils from pf_active, plasma cells weighted by j * area inside LCFS
for c in coils:
    coilset.sloc[c["name"], "Ic"] = currents[c["name"]]
plasma_rows = np.flatnonzero(np.asarray(coilset.subframe["part"]) == "plasma")
polys = np.asarray(coilset.subframe["poly"], dtype=object)[plasma_rows]
cells = [np.asarray(p.boundary, float).reshape(-1, 2) for p in polys]
lcfs = shapely.Polygon(prescribed.boundary)
density = RegularGridInterpolator(
    (prescribed.radius, prescribed.height),
    prescribed.current_density,
    bounds_error=False,
    fill_value=0.0,
)
weight = np.zeros(len(cells))
for k, cell in enumerate(cells):
    part = shapely.Polygon(cell).intersection(lcfs)
    if part.area > 0:
        centroid = part.centroid
        weight[k] = density([[centroid.x, centroid.y]])[0] * part.area
weight *= prescribed.plasma_current / weight.sum()
ionize = weight != 0
coilset.plasma.ionize = ionize
coilset.plasma.nturn = weight[ionize] / weight[ionize].sum()
plasma_name = [
    n for n, p in zip(coilset.frame.index, coilset.frame["part"]) if p == "plasma"
][0]
coilset.sloc[plasma_name, "Ic"] = prescribed.plasma_current
print(
    f"hex cells {len(cells)}, carrying current {int(ionize.sum())}, "
    f"sum {weight.sum() / 1e3:.1f} kA",
    flush=True,
)

br, bz = np.asarray(coilset.point.br), np.asarray(coilset.point.bz)
psi_points = np.asarray(coilset.point.psi)
nline = line_targets.shape[0]
bmag = np.hypot(br[:nline], bz[:nline]).reshape(len(LINES), -1)
psi = np.asarray(coilset.grid.psi_)
gx, gz = np.asarray(coilset.grid.data.x), np.asarray(coilset.grid.data.z)
bgrid = np.hypot(
    np.asarray(coilset.grid.br).reshape(psi.shape),
    np.asarray(coilset.grid.bz).reshape(psi.shape),
)
for k, name in enumerate(LINES):
    b = bmag[k] * 1e3
    idx = np.flatnonzero((b[:-1] >= 5) & (b[1:] < 5))
    r5 = np.interp(
        5, [b[idx[-1] + 1], b[idx[-1]]], [radius[idx[-1] + 1], radius[idx[-1]]]
    )
    print(
        f"{name:9s} |B| at edge {b[0]:.1f} mT; 5 mT at R={r5:.3f} m "
        f"({r5 - edge:.3f} m from the edge)"
    )

with open(OUT, "wb") as f:
    pickle.dump(
        dict(
            time=time,
            coils=coils,
            currents=currents,
            limiter=limiter,
            iron=iron_outlines,
            gx=gx,
            gz=gz,
            psi=psi,
            bgrid=bgrid,
            psi_core=psi_points[nline:].reshape(CORE["shape"]),
            core_x=core_x,
            core_z=core_z,
            radius=radius,
            lines=LINES,
            bmag=bmag,
            edge=edge,
            boundary=prescribed.boundary,
            axis=slice_.axis,
            ip=float(weight.sum()),
            plasma_cells=[c for c, on in zip(cells, ionize) if on],
            plasma_weight=weight[ionize],
        ),
        f,
    )
print(f"done {perf_counter() - start:.1f} s")
