"""Read nulls from the same requested-input fields saved for presentation."""

# ruff: noqa: E402
import argparse
from dataclasses import replace
import json
from pathlib import Path

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import RectBivariateSpline

from nova.equilibrium import support_read as sr, topology
from nova.equilibrium.constraint import uniform_radial_field_jet
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.prototypes.dina_flattop_scenario.inputs import load_bank, write


def classes(hessian):
    determinant = np.linalg.det(hessian)
    return np.where(
        determinant < 0,
        "saddle",
        np.where(hessian[..., 0, 0] > 0, "minimum", "maximum"),
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 and jax.default_backend() == "cpu"
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((args.results / "receipt.json").read_text())
    selection = json.loads((args.inputs / "selection.json").read_text())
    references = {r["index"]: r for r in selection["slices"]}
    bank, cache = load_bank(args.inputs / "bank")
    convention = topology.TopologyConvention.from_cocos(17, references[129]["ip_a"])
    policy = TopologyPolicy()

    @jax.jit
    def read(op, window, moments, roots, amps, amplitude):
        driven = sr.driven_operands(op, amps)
        driven = replace(
            driven,
            exterior=driven.exterior
            + amplitude * 1e-4 * uniform_radial_field_jet(driven.nodes),
        )
        window = sr._driven_cache(driven, window)
        support = sr.read(
            driven, window, moments, roots, roots, convention, policy, axis_sign=1.0
        )
        return support, sr.nodal_jets(driven, moments)[:, 0]

    record = dict(
        module=sr.__file__,
        cwd=str(Path.cwd()),
        backend="cpu",
        complete=False,
        source_revision=receipt["revision"],
        rows=[],
    )
    for row in receipt["slices"]:
        index = row["index"]
        state = np.load(args.results / f"state-{index}.npz", allow_pickle=False)
        source = np.load(args.inputs / f"source-{index}.npz", allow_pickle=False)
        moments = tuple(jnp.asarray(state["moments"][:, i]) for i in range(3))
        support, flux = jax.block_until_ready(
            read(
                bank,
                cache,
                moments,
                jnp.asarray(state["stationary"]),
                jnp.asarray(source["currents"]),
                jnp.asarray(row["pin_amplitude"][0]),
            )
        )
        field_difference = float(
            np.max(np.abs(np.asarray(flux)[: int(bank.active_nodes)] - state["flux"]))
        )
        assert field_difference <= 1e-10, field_difference
        stationary = support.stationary
        reference = references[index]
        spline = RectBivariateSpline(source["radius"], source["height"], source["flux"])
        roots = np.concatenate((source["axis"][None], source["xpoints"]))
        rr, zz = roots.T
        radial = spline.ev(rr, zz, dx=2)
        mixed = spline.ev(rr, zz, dx=1, dy=1)
        vertical = spline.ev(rr, zz, dy=2)
        hessian = np.stack((radial, mixed, mixed, vertical), axis=-1).reshape(-1, 2, 2)
        native_classes = classes(hessian)
        actual_classes = classes(np.asarray(stationary.jet.hessian))
        observed = dict(
            index=index,
            stationary=np.asarray(stationary.position),
            null_valid=np.asarray(stationary.valid),
            null_iterations=np.asarray(stationary.iterations),
            null_status=np.asarray(stationary.status),
            null_classification=actual_classes,
            dina_null_classification=native_classes,
            classification_agrees=actual_classes == native_classes,
            axis_m=np.asarray(support.reading.axis),
            xpoint_m=np.asarray(support.reading.boundary),
            axis_difference_m=np.asarray(support.reading.axis) - reference["axis"],
            xpoint_difference_m=np.asarray(support.reading.boundary)
            - reference["xpoints"][0],
            boundary_flux_difference_wb=float(support.reading.boundary_flux)
            - reference["boundary_flux_wb"],
            maximum_flux_difference_wb=field_difference,
            trial_marker_displacement_m=np.linalg.norm(
                np.asarray(stationary.position) - state["stationary"], axis=-1
            ),
        )
        record["rows"].append(observed)
        write(args.output / "observables.json", record)
        print(
            "FIELD_READ",
            index,
            "flux_difference",
            field_difference,
            "marker_displacement",
            observed["trial_marker_displacement_m"],
            flush=True,
        )
    record["complete"] = True
    write(args.output / "observables.json", record)
    print("FIELDS_VERIFIED", len(record["rows"]), flush=True)


if __name__ == "__main__":
    main()
