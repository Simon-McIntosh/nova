"""Measure the seam on the sampled WEST target set the CPU turn allows.

A CoilSet point solve assembles the plasma closed coupling for its whole target
set on every call, and on CPU that assembly costs about 3.3 s per target per
attribute. So this stage runs two solves only -- the air-core field and the
zero-magnetisation control, which is the falsifying pair -- at three sampling
planes of six radii, and lifts the solved iron onto those points and onto every
in-vessel diagnostic with ``BoundCurrents``, the operand the passive frame
carries. The frame-versus-direct identity that licenses that composition is
pinned by ``tests/test_magnetisation.py::
test_frame_carries_the_bound_currents_exactly``.

Usage: sampled.py [mesh]
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from nova.biot.tiledassembly import linear_cell_interaction

measure = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location(
        "measure", Path(__file__).with_name("measure.py")
    )
)
measure.__spec__.loader.exec_module(measure)

MESH = "uniform-0.08"
measure.MESH = MESH
measure.CELL_SIZE = float(MESH.split("-")[1])


def affine_field(bound, points):
    """Return the iron's (B_R, B_Z) from the package's affine evaluator.

    ``nova.biot.tiledassembly.linear_cell_interaction`` is the owner's
    evaluation of an affine magnetisation, and it is a different call path from
    ``BoundCurrents.field``: a comparison of the two is a measurement, where a
    comparison of ``BoundCurrents.field`` with itself cannot fail.
    """
    points = np.asarray(points, dtype=np.float64)
    response = linear_cell_interaction(bound.polygons, points, plan=bound.plan)
    return (response @ bound.currents).reshape(2, -1).T


def affine_flux(bound, points, samples=257):
    """Return the iron's poloidal flux [Wb] by radial integration of that field.

    ``psi(R, Z) = int_0^R 2 pi r' B_z(r', Z) dr'`` is the definition of the
    poloidal flux, so this evaluates ``BoundCurrents.psi`` against a different
    formulation over a different evaluator rather than against itself.
    """
    flux = np.empty(len(points))
    for index, (radius, height) in enumerate(points):
        radii = np.linspace(0.0, radius, samples)[1:]
        grid = np.column_stack((radii, np.full_like(radii, height)))
        moment = 2 * np.pi * radii * affine_field(bound, grid)[:, 1]
        flux[index] = np.sum(0.5 * (moment[1:] + moment[:-1]) * np.diff(radii))
    return flux


def main():
    from scripts.iron_graded_refinement.measure import west_inputs

    bound = measure.bound_from_state()
    coilset, coils, probes, loops, time = measure.frame_coilset()
    edge = max(
        coil["r"] + coil["dr"] / 2 for coil in coils if coil["name"] in ("Fh", "Fb")
    )
    radius = np.linspace(edge, 16.0, 6)
    line_targets = np.vstack(
        [
            np.column_stack((radius, np.full_like(radius, z)))
            for z in measure.LINES.values()
        ]
    )
    started = perf_counter()
    coilset.point.solve(line_targets)
    air_lines = np.column_stack(
        (np.asarray(coilset.point.br), np.asarray(coilset.point.bz))
    )
    air_psi = np.asarray(coilset.point.psi).copy()
    air_seconds = perf_counter() - started
    measure.insert_iron(coilset, bound.zero())
    started = perf_counter()
    coilset.point.solve(line_targets)
    zero_lines = np.column_stack(
        (np.asarray(coilset.point.br), np.asarray(coilset.point.bz))
    )
    zero_psi = np.asarray(coilset.point.psi).copy()
    zero_seconds = perf_counter() - started
    control = float(np.max(np.abs(zero_lines - air_lines)))
    control_psi = float(np.max(np.abs(zero_psi - air_psi)))

    inputs = west_inputs()
    diagnostic_points = np.vstack((probes, loops))
    air_diagnostic = np.asarray(inputs["external"](diagnostic_points))
    contribution_probe = bound.field(probes)
    contribution_loop = bound.psi(loops)
    affine_probe = affine_field(bound, probes)
    affine_loop = affine_flux(bound, loops)
    nline = len(line_targets)
    contribution_lines = bound.field(line_targets)
    with_iron_lines = air_lines + contribution_lines
    bmag_air = np.hypot(*air_lines.T).reshape(len(measure.LINES), -1)
    bmag_iron = np.hypot(*with_iron_lines.T).reshape(len(measure.LINES), -1)
    stem = f"west-62751-{MESH}-sampled"
    np.savez(
        measure.EVIDENCE / f"{stem}.npz",
        radius=radius,
        line_names=np.asarray(list(measure.LINES)),
        lines=np.asarray(list(measure.LINES.values())),
        edge=edge,
        bmag_air=bmag_air,
        bmag_iron=bmag_iron,
        probes=probes,
        loops=loops,
        frame_probe=contribution_probe,
        direct_probe=air_diagnostic[: len(probes)],
        frame_loop=contribution_loop,
        direct_loop=air_diagnostic[len(probes) :],
        affine_probe=affine_probe,
        affine_loop=affine_loop,
        contribution=contribution_lines,
        air=air_lines,
        with_iron=with_iron_lines,
    )
    receipt = dict(
        stage="sampled",
        mesh=MESH,
        targets=int(nline),
        plane_radii=radius.tolist(),
        probe_count=int(len(probes)),
        loop_count=int(len(loops)),
        zero_control_field=control,
        zero_control_psi=control_psi,
        air_seconds=air_seconds,
        zero_control_seconds=zero_seconds,
        unknowns=int(bound.unknowns),
        frame_versus_direct_probe=float(
            np.max(np.abs(contribution_probe - affine_probe))
        ),
        frame_versus_direct_loop=float(np.max(np.abs(contribution_loop - affine_loop))),
        frame_versus_direct_from=(
            "tiledassembly.linear_cell_interaction, and its radial flux integral"
        ),
        diagnostic_from="west_inputs external (prototype coils+plasma quadrature)",
        iron_from="BoundCurrents, the operand the passive frame carries",
        completed=True,
    )
    (measure.EVIDENCE / f"{stem}-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
