"""Stray field and diagnostics from the iron's own operand.

The CoilSet point solve assembles the plasma closed coupling for its whole
target set on every call, which is the wall this allocation could not pay, so
the sampled fields here come from the same external term the material solve
consumes (the prototype's coil and plasma quadrature, pinned by
``west_inputs``) plus ``BoundCurrents``, the operand the passive frame carries.
``tests/test_magnetisation.py::test_frame_carries_the_bound_currents_exactly``
pins the frame's contribution to that operand in field and flux.

Usage: diagnostics.py
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
measure.MESH = "uniform-0.08"
measure.CELL_SIZE = 0.08


def main():
    from scripts.iron_graded_refinement.measure import west_inputs

    started = perf_counter()
    inputs = west_inputs()
    bound = measure.bound_from_state()
    coilset, coils, probes, loops, time = measure.frame_coilset()
    edge = max(
        coil["r"] + coil["dr"] / 2 for coil in coils if coil["name"] in ("Fh", "Fb")
    )
    radius = np.linspace(edge, 16.0, 40)
    lines = np.vstack(
        [
            np.column_stack((radius, np.full_like(radius, z)))
            for z in measure.LINES.values()
        ]
    )
    external = np.asarray(inputs["external"](lines))
    iron = bound.field(lines)
    contribution_probe = bound.field(probes)
    contribution_loop = bound.psi(loops)
    air_diagnostic = np.asarray(inputs["external"](np.vstack((probes, loops))))
    bmag_air = np.hypot(*external.T).reshape(len(measure.LINES), -1)
    bmag_iron = np.hypot(*(external + iron).T).reshape(len(measure.LINES), -1)
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
        contribution=iron,
        air=external,
        with_iron=external + iron,
    )
    receipt = dict(
        stage="diagnostics",
        mesh="uniform-0.08",
        unknowns=int(bound.unknowns),
        plane_points=int(len(radius)),
        probe_count=int(len(probes)),
        loop_count=int(len(loops)),
        external="west_inputs prototype coil+plasma quadrature (air_drift 0.0)",
        iron="BoundCurrents.field and .psi",
        coilset_targets=None,
        seconds=perf_counter() - started,
        completed=True,
    )
    measure.receipt_path().write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
