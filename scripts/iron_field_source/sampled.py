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

measure = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location(
        "measure", Path(__file__).with_name("measure.py")
    )
)
measure.__spec__.loader.exec_module(measure)

MESH = "uniform-0.08"
measure.MESH = MESH
measure.CELL_SIZE = float(MESH.split("-")[1])


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
    nline = len(line_targets)
    with_iron_lines = air_lines + bound.field(line_targets)
    bmag_air = np.hypot(*air_lines.T).reshape(len(measure.LINES), -1)
    bmag_iron = np.hypot(*with_iron_lines.T).reshape(len(measure.LINES), -1)
    np.savez(
        measure.field_path(),
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
        contribution=np.zeros((nline, 2)),
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
            np.max(np.abs(contribution_probe - bound.field(probes)))
        ),
        frame_versus_direct_loop=float(
            np.max(np.abs(contribution_loop - bound.psi(loops)))
        ),
        diagnostic_from="west_inputs external (prototype coils+plasma quadrature)",
        iron_from="BoundCurrents, the operand the passive frame carries",
        completed=True,
    )
    measure.receipt_path().write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
