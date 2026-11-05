"""WEST 62751 stray poloidal field outboard of the equatorial PF coils.

Coils from the WEST machine description (pf_active, DD 4.1.0), currents from the
58072 pf_active IDS, plasma current from the NICE equilibrium's 2D flux map and
flux functions (j = R p' + FF'/(mu0 R)) via nova PlasmaSource.prescribe.
"""

from __future__ import annotations

import logging
import pickle
import sys
from time import perf_counter

import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()

import imas  # noqa: E402
import scipy.special  # noqa: E402

from nova.frame.coilset import CoilSet  # noqa: E402
from nova.imas.plasma_source import PlasmaSource  # noqa: E402

logging.disable(logging.WARNING)
ROOT = "/home/ITER/mcintos/Code/imas-efit/tests/data/imas/WEST"
MACHINE = f"{ROOT}/machine/dd4-1-0"
PULSE = f"{ROOT}/62751"
EQUILIBRIUM = f"{ROOT}/62751/validation/nice"
INDEX = 14
OUT = sys.argv[1]


def entry(path, dd):
    return imas.DBEntry(f"imas:hdf5?path={path}", "r", dd_version=dd)


with entry(MACHINE, "4.1.0") as md:
    pf_md = md.get("pf_active", autoconvert=False)
    wall = md.get("wall", autoconvert=False)
    passive = md.get("pf_passive", autoconvert=False)
    coils = []
    for coil in pf_md.coil:
        for element in coil.element:
            rect = element.geometry.rectangle
            coils.append(
                dict(
                    name=str(coil.name).strip(),
                    r=float(rect.r),
                    z=float(rect.z),
                    dr=float(rect.width),
                    dz=float(rect.height),
                    nturn=float(element.turns_with_sign),
                )
            )
    limiter = wall.description_2d[0].limiter.unit[0].outline
    limiter = np.column_stack((np.asarray(limiter.r), np.asarray(limiter.z)))
    vessel = []
    for loop in passive.loop:
        for element in loop.element:
            g = element.geometry
            gtype = int(g.geometry_type)
            if gtype == 1 and len(g.outline.r):
                vessel.append(np.column_stack((g.outline.r, g.outline.z)))
            elif gtype == 2:
                r, z, w, h = (
                    float(g.rectangle.r),
                    float(g.rectangle.z),
                    float(g.rectangle.width),
                    float(g.rectangle.height),
                )
                vessel.append(
                    np.array(
                        [
                            [r - w / 2, z - h / 2],
                            [r + w / 2, z - h / 2],
                            [r + w / 2, z + h / 2],
                            [r - w / 2, z + h / 2],
                        ]
                    )
                )
            elif gtype == 3:
                o = g.oblique
                r, z, a, b = (
                    float(o.r),
                    float(o.z),
                    float(o.length_alpha),
                    float(o.length_beta),
                )
                al, be = float(o.alpha), float(o.beta)
                p0 = np.array([r, z])
                ea = a * np.array([np.cos(al), np.sin(al)])
                eb = b * np.array([-np.sin(be), np.cos(be)])
                vessel.append(np.array([p0, p0 + ea, p0 + ea + eb, p0 + eb]))
    iron = md.get("iron_core", autoconvert=False)
    iron_outlines = [
        np.column_stack(
            (np.asarray(s.geometry.outline.r), np.asarray(s.geometry.outline.z))
        )
        for s in iron.segment
        if len(s.geometry.outline.r)
    ]

with entry(EQUILIBRIUM, "3.42.2") as eq_entry, entry(PULSE, "4.1.0") as pulse:
    equilibrium = eq_entry.get("equilibrium", autoconvert=False)
    active = pulse.get("pf_active", autoconvert=False)
    source = PlasmaSource.from_imas(
        equilibrium, active, provenance="WEST 62751 NICE", indices=[INDEX]
    )
slice_ = source.slices[0]
time = slice_.time
names = list(source.coils.names)
currents = dict(zip(names, source.coils.at(time)))
prescribed = source.prescribe(0)
print(
    f"t={time:.3f}s  Ip(map)={prescribed.cell_current.sum() / 1e3:.1f} kA  "
    f"Ip(ids)={slice_.plasma_current / 1e3:.1f} kA  "
    f"scale={prescribed.scale_factor:.4f}  "
    f"boundary={prescribed.boundary_source}"
)
for c in coils:
    print(
        f"  {c['name']:22s} I={currents[c['name']]:9.1f} A/turn  "
        f"It={currents[c['name']] * c['nturn'] / 1e3:8.2f} kAt"
    )

dr = float(np.mean(np.diff(prescribed.radius)))
dz = float(np.mean(np.diff(prescribed.height)))
grid_r, grid_z = np.meshgrid(prescribed.radius, prescribed.height, indexing="ij")
live = np.flatnonzero(prescribed.cell_current.ravel())
cell_r, cell_z = grid_r.ravel()[live], grid_z.ravel()[live]
cell_I = prescribed.cell_current.ravel()[live]
print(f"plasma cells {live.size}  dr={dr * 1e3:.1f} mm dz={dz * 1e3:.1f} mm")

coilset = CoilSet(dcoil=-1)
for c in coils:
    coilset.coil.insert(
        c["r"],
        c["z"],
        c["dr"],
        c["dz"],
        nturn=c["nturn"],
        Ic=currents[c["name"]],
        name=c["name"],
        part="pf",
    )
coilset.coil.insert(
    cell_r,
    cell_z,
    dr,
    dz,
    nturn=1.0,
    Ic=cell_I,
    part="plasma",
    name=[f"pl{i}" for i in range(live.size)],
)
print(
    "coilset Ic*nturn check: total plasma It",
    float(
        (coilset.subframe.Ic * coilset.subframe.nturn)[
            coilset.subframe.part == "plasma"
        ].sum()
    )
    / 1e3,
    "kA",
)


edge = max(c["r"] + c["dr"] / 2 for c in coils if c["name"] in ("Fh", "Fb"))
radius = np.linspace(edge, 16.0, 400)
lines = {"midplane": 0.0, "Fb plane": -0.637, "Fh plane": 0.637}
targets = np.vstack(
    [np.column_stack((radius, np.full_like(radius, z))) for z in lines.values()]
)
coilset.point.solve(targets)
print("point solve done", flush=True)
br = np.asarray(coilset.point.br).reshape(len(lines), -1)
bz = np.asarray(coilset.point.bz).reshape(len(lines), -1)
bmag = np.hypot(br, bz)


# independent positive control: filament ring sum with the elliptic integrals
def ring_field(a, zc, r, z, current):
    mu0 = 4e-7 * np.pi
    dz_ = z - zc
    m = 4 * a * r / ((a + r) ** 2 + dz_**2)
    K, E = scipy.special.ellipk(m), scipy.special.ellipe(m)
    den = np.sqrt((a + r) ** 2 + dz_**2)
    q = (a - r) ** 2 + dz_**2
    bz_ = mu0 * current / (2 * np.pi * den) * (K + (a**2 - r**2 - dz_**2) / q * E)
    br_ = (
        mu0
        * current
        * dz_
        / (2 * np.pi * r * den)
        * (-K + (a**2 + r**2 + dz_**2) / q * E)
    )
    return br_, bz_


rr, zz = targets[:, 0], targets[:, 1]
cbr, cbz = np.zeros_like(rr), np.zeros_like(rr)
for c in coils:
    b = ring_field(c["r"], c["z"], rr, zz, currents[c["name"]] * c["nturn"])
    cbr += b[0]
    cbz += b[1]
for a, zc, i in zip(cell_r, cell_z, cell_I):
    b = ring_field(a, zc, rr, zz, i)
    cbr += b[0]
    cbz += b[1]
check = np.hypot(cbr, cbz).reshape(len(lines), -1)
rel = np.abs(check - bmag) / bmag
far = radius > edge + 0.3
print(
    f"ring-sum control: max rel diff {rel.max():.2e} (all), "
    f"{rel[:, far].max():.2e} (>0.3 m from coil edge)"
)
for k, name in enumerate(lines):
    b = bmag[k] * 1e3

    def cross(level):
        idx = np.flatnonzero((b[:-1] >= level) & (b[1:] < level))
        return radius[idx[-1]] if idx.size else np.nan

    print(
        f"{name:9s} |B| at edge {b[0]:.1f} mT; 20 mT at R={cross(20):.2f} m, "
        f"10 mT {cross(10):.2f}, 5 mT {cross(5):.2f}, "
        f"2 mT {cross(2):.2f}, 1 mT {cross(1):.2f} m"
    )

pickle.dump(
    dict(radius=radius, bmag=bmag, br=br, bz=bz, check=check),
    open(OUT + ".points", "wb"),
)
_t0 = perf_counter()
coilset.grid.solve(6000, limit=[0.2, 12.0, -4.0, 4.0])
print(f"grid solve {perf_counter() - _t0:.0f} s", flush=True)
psi = np.asarray(coilset.grid.psi_)
gx = np.asarray(coilset.grid.data.x)
gz = np.asarray(coilset.grid.data.z)
bgrid = np.hypot(
    np.asarray(coilset.grid.br).reshape(psi.shape),
    np.asarray(coilset.grid.bz).reshape(psi.shape),
)

# finer map over the plasma so closed flux surfaces resolve
coilset.grid.solve(2500, limit=[1.75, 3.15, -0.95, 0.95])
psi_core = np.asarray(coilset.grid.psi_)
core_x = np.asarray(coilset.grid.data.x)
core_z = np.asarray(coilset.grid.data.z)
print(f"core grid {core_x.size}x{core_z.size} solved", flush=True)

with open(OUT, "wb") as f:
    pickle.dump(
        dict(
            time=time,
            coils=coils,
            currents=currents,
            limiter=limiter,
            vessel=vessel,
            iron=iron_outlines,
            gx=gx,
            gz=gz,
            psi=psi,
            bgrid=bgrid,
            psi_core=psi_core,
            core_x=core_x,
            core_z=core_z,
            radius=radius,
            lines=lines,
            bmag=bmag,
            br=br,
            bz=bz,
            edge=edge,
            boundary=prescribed.boundary,
            axis=slice_.axis,
            x_points=slice_.x_points,
            psi_boundary=slice_.psi_boundary,
            psi_axis=slice_.psi_axis,
            ip=float(cell_I.sum()),
            check=check,
        ),
        f,
    )
