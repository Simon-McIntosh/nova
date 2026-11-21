"""Solve one cohort row with the inactive compensator masked, or as its control.

The solve, the seeding and the receipts are the booked finite-edge cohort's.
This driver adds two things and changes nothing else: it can restore the
inactive image as the booked unmasked control, and it reads the vertical pin's
field at every inner endpoint so a terminal state's field is attributed rather
than inferred.
"""

from __future__ import annotations

import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from scripts.prototypes.diiid_finite_edge_booking import measure as booked
from nova.equilibrium.constraint import ConstraintPair, CurrentCentroidConstraint
from nova.equilibrium.solve_request import ForwardSolveReceipt

MASK_OFF = os.environ.get("NOVA_INACTIVE_MASK_OFF") == "1"

branch = booked.extent.held_measure.core.branch_measure
original_load = branch.original_load
captures = []


def restore_unmasked_image():
    """Reproduce the withdrawn arm: a bound source applies whatever its mask.

    The control exists only in this measurement process.  No product source is
    edited and the compiled program is built after the substitution, so both
    arms are one tree with the rule on and off and every other input identical.
    """

    def unmasked_image(self, profile, context, normalized):
        return self.unknown.flux_delta(
            profile, context, self.functional, self.binding.payload, normalized
        )

    ConstraintPair.flux_delta = unmasked_image


def capture_solve(original):
    """Retain the equilibrium carried by the public solve receipt."""

    def solve(profile, *args, **kwargs):
        result = original(profile, *args, **kwargs)
        booked.SOLVED["value"] = (
            result.equilibrium if isinstance(result, ForwardSolveReceipt) else result
        )
        return result

    return solve


def observed_load(name, path):
    """Retain the cohort's executed operand capture without changing the solve."""
    module = original_load(name, path)
    if name == "operand_capture":
        original = module.Capture

        class Capture(original):
            def __init__(self, output):
                super().__init__(output)
                self.pairs = ()
                captures.append(self)

            def install(self):
                super().install()
                from nova.equilibrium import forward

                assembled = forward.assemble_augmented_system

                def assembly(profile, initial_flux, pairs, **kwargs):
                    self.pairs = pairs
                    return assembled(profile, initial_flux, pairs, **kwargs)

                forward.assemble_augmented_system = assembly

        module.Capture = Capture
    return module


def pin_basis(pairs):
    """Return the vertical pin's flux basis and its tesla per unit unknown."""
    for pair in pairs:
        if isinstance(pair.functional, CurrentCentroidConstraint):
            return (
                np.asarray(pair.unknown.flux_basis),
                float(np.asarray(pair.unknown.field_scale).reshape(-1)[0]),
            )
    raise AssertionError("the row must carry one current-centroid pin")


def trip_read(capture):
    """Read the live topology and the pin field at every inner endpoint."""
    operator = capture.profile.operator
    system = capture.system
    output = capture.output
    terminal = dict(np.load(output / "terminal.npz"))
    flux_size = system.flux_size
    scale = system.flux_scale
    basis, field_scale = pin_basis(capture.pairs)

    @jax.jit
    def read(flux):
        _, topology, connected, admitted = operator._fixed_design_read(flux)
        return connected.sum(), topology.axis, admitted

    def reading(state):
        flux = jnp.asarray(state[:flux_size]) * scale
        cells, axis, admitted = jax.device_get(read(flux))
        unknown = np.asarray(state[flux_size:])
        return dict(
            connected_cells=int(cells),
            axis_rz_m=np.asarray(axis).tolist(),
            admitted=bool(admitted),
            pin_unknown=unknown.tolist(),
            pin_field_mt=float(unknown[0] * field_scale * 1e3),
            pin_image_sup_wb=float(np.max(np.abs(unknown[0] * basis), initial=0.0)),
        )

    receipt = json.loads((output / "receipt.json").read_text())
    cells, axis, admitted = jax.device_get(read(jnp.asarray(terminal["state"])))
    assert int(cells) == receipt["support"], "terminal support positive control"
    result = dict(
        mask_off=MASK_OFF,
        pin_field_scale_t=field_scale,
        terminal=dict(
            connected_cells=int(cells),
            axis_rz_m=np.asarray(axis).tolist(),
            admitted=bool(admitted),
        ),
        trips=[],
        cohort_topology_reads=json.loads((output / "trips.json").read_text()),
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
    last = result["trips"][-1]["returned"]
    result["terminal"].update(
        pin_unknown=last["pin_unknown"],
        pin_field_mt=last["pin_field_mt"],
        pin_image_sup_wb=last["pin_image_sup_wb"],
    )
    (output / "trips.json").write_text(json.dumps(result, indent=2) + "\n")
    print("TRIP_READ " + json.dumps(result), flush=True)


def main():
    if MASK_OFF:
        restore_unmasked_image()
    branch.original_load = observed_load
    booked.capture_solve = capture_solve
    booked.main()
    assert len(captures) == 1, "the production operand capture must execute"
    trip_read(captures[0])


if __name__ == "__main__":
    main()
