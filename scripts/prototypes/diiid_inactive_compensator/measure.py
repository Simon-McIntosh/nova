"""Observe per-trip support and axis on the established booked-source cohort."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from scripts.prototypes.diiid_finite_edge_booking import measure as booked

branch = booked.extent.held_measure.core.branch_measure
original_load = branch.original_load
captures = []


def observed_load(name, path):
    """Retain the cohort's executed operand capture without changing the solve."""
    module = original_load(name, path)
    if name == "operand_capture":
        original = module.Capture

        class Capture(original):
            def __init__(self, output):
                super().__init__(output)
                captures.append(self)

        module.Capture = Capture
    return module


def trip_read(capture):
    """Read the live production topology at incoming and returned inner states."""
    operator = capture.profile.operator
    system = capture.system
    output = capture.output
    terminal = dict(np.load(output / "terminal.npz"))
    flux_size = terminal["state"].size
    scale = system.flux_scale

    @jax.jit
    def read(flux):
        _, topology, connected, admitted = operator._fixed_design_read(flux)
        return connected.sum(), topology.axis, admitted

    def reading(state):
        flux = jnp.asarray(state[:flux_size]) * scale
        cells, axis, admitted = jax.device_get(read(flux))
        return dict(
            connected_cells=int(cells),
            axis_rz_m=np.asarray(axis).tolist(),
            admitted=bool(admitted),
            inactive_unknown=np.asarray(state[flux_size:]).tolist(),
        )

    receipt = json.loads((output / "receipt.json").read_text())
    cells, axis, admitted = jax.device_get(read(jnp.asarray(terminal["state"])))
    assert int(cells) == receipt["support"], "terminal support positive control"
    result = dict(
        terminal=dict(
            connected_cells=int(cells),
            axis_rz_m=np.asarray(axis).tolist(),
            admitted=bool(admitted),
        ),
        trips=[],
    )
    assert capture.counts["inner"] == len(capture.inner) > 0
    for index, metrics in enumerate(capture.inner):
        arrays = dict(np.load(output / f"inner-{index}.npz"))
        result["trips"].append(
            dict(
                index=index,
                metrics=metrics,
                initial=reading(arrays["initial"]),
                returned=reading(arrays["state"]),
            )
        )
        (output / "trips.json").write_text(json.dumps(result, indent=2) + "\n")
    print("TRIP_READ " + json.dumps(result), flush=True)


def main():
    branch.original_load = observed_load
    booked.main()
    assert len(captures) == 1, "the production operand capture must execute"
    trip_read(captures[0])


if __name__ == "__main__":
    main()
