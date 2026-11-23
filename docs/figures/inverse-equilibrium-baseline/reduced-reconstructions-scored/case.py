"""Build the ITER row-exchange case inputs for the reduced reconstructions."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nova.biot.greens import hybrid_greens
from nova.equilibrium.measurement import Magnetics, SliceMeasurement
from nova.equilibrium.wall_mask import WallUnit, build_wall_mask
from nova.imas.dataset import open_db_entry

CASE = Path(
    "/home/ITER/mcintos/.cache/nova-inverse-equilibrium-baseline/efit-on-nova-rows-case"
)


def conductor_sources(pulse: Path):
    """Return (r, z, dr, dz, name) for every pf_active / pf_passive element."""

    sources = []
    with open_db_entry(str(pulse), backend="hdf5") as entry:
        active = entry.get("pf_active", 0, lazy=False, autoconvert=False)
        passive = entry.get("pf_passive", 0, lazy=False, autoconvert=False)
        for ids, prefix, kind in ((active, "pf_active", "active"), (passive, "pf_passive", "passive")):
            if kind == "active":
                coils = [
                    (
                        coil,
                        float(np.asarray(coil.current.data).ravel()[0]),
                    )
                    for coil in ids.coil
                ]
            else:
                coils = [
                    (loop, float(np.asarray(loop.current).ravel()[0]))
                    for loop in ids.loop
                ]
            for number, (coil, current) in enumerate(coils):
                for index, element in enumerate(coil.element):
                    turns = element.turns_with_sign
                    turns = (
                        float(np.asarray(turns.data).ravel()[0])
                        if hasattr(turns, "data")
                        else float(turns)
                    )
                    geometry = element.geometry
                    kind_of = int(geometry.geometry_type)
                    if kind_of == 2:
                        rectangle = geometry.rectangle
                        r = float(rectangle.r)
                        z = float(rectangle.z)
                        dr = float(rectangle.width)
                        dz = float(rectangle.height)
                    elif kind_of == 3:
                        oblique = geometry.oblique
                        r = float(oblique.r)
                        z = float(oblique.z)
                        dr = float(oblique.length_beta)
                        dz = float(oblique.length_alpha)
                    else:
                        raise ValueError(f"unsupported geometry type {kind_of}")
                    sources.append(
                        (r, z, dr, dz, current * turns, f"{prefix}:{number}:{index}")
                    )
    return sources


def build():
    record = json.loads((CASE / "record-10.json").read_text())
    rows = record["rows"]
    generating = np.load(CASE / "generating-map.npz")
    wall = np.asarray(generating["wall"], dtype=float)

    magnetics = Magnetics(
        r=np.asarray(rows["sensor_r"], dtype=float),
        z=np.asarray(rows["sensor_z"], dtype=float),
        angle=np.asarray(rows["sensor_angle"], dtype=float),
        flux_loop=np.asarray(rows["sensor_flux_loop"], dtype=float).astype(bool),
    )

    unit = WallUnit(r=wall[:, 0], z=wall[:, 1], kind="vessel", name="vessel")
    margin = 0.02
    rg = np.linspace(wall[:, 0].min() - margin, wall[:, 0].max() + margin, 33)
    zg = np.linspace(wall[:, 1].min() - margin, wall[:, 1].max() + margin, 49)
    inside, _diagnostics = build_wall_mask(rg, zg, [unit])
    r0 = float(0.5 * (wall[:, 0].min() + wall[:, 0].max()))
    grid = SimpleNamespace(
        rg=rg,
        zg=zg,
        inside_limiter=inside,
        limiter_r=wall[:, 0],
        limiter_z=wall[:, 1],
        wall_r=wall[:, 0],
        wall_z=wall[:, 1],
        wall_units=(unit,),
        r0=r0,
    )

    sources = conductor_sources(CASE / "pulse-10")
    mesh_r, mesh_z = np.meshgrid(rg, zg)
    target_r = mesh_r.ravel()
    target_z = mesh_z.ravel()
    vacuum_flux = np.zeros(target_r.size)
    for r, z, dr, dz, current, _name in sources:
        psi, _br, _bz = hybrid_greens(target_r, target_z, r, z, dr, dz)
        vacuum_flux += psi * current
    vacuum_flux = vacuum_flux.reshape(len(zg), len(rg))

    measured = np.asarray(rows["total"], dtype=float)
    scale = np.asarray(rows["scale"], dtype=float)
    mask = np.isfinite(measured) & (scale > 0.0)
    measurement = SliceMeasurement(
        measured=measured,
        vacuum=np.asarray(rows["vacuum"], dtype=float),
        mask=mask,
        scale=scale,
        plasma_current=float(rows["plasma_current"]),
        vacuum_flux=vacuum_flux,
    )
    return dict(
        magnetics=magnetics,
        grid=grid,
        wall=wall,
        sources=sources,
        vacuum_flux=vacuum_flux,
        measurement=measurement,
        record=record,
        rg=rg,
        zg=zg,
        inside=inside,
        r0=r0,
    )
