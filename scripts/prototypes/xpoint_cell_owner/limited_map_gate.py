"""Measure the limited carrier through both owner selections on one map."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

import nova
from benchmarks import solovev_certificate as certificate
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.fine_grid_convergence import context
from scripts.prototypes.lean_map.probe_bias import _image
from scripts.prototypes.solve_options.point_bias import _exterior
from scripts.prototypes.xpoint_cell_owner.owner_map import map_moments
from scripts.prototypes.xpoint_cell_owner.runtime_solve import without_profile


def _limited_context(cells, cache):
    """Use the shared builder with the limited certificate inputs."""
    analytic_inputs = context._analytic_inputs
    certificate_field = context.certificate_field
    limited_case = SimpleNamespace(
        _case=lambda _name, clip_mode: certificate._case(
            "weak-rotation-reactor-static", clip_mode=clip_mode
        )
    )
    with (
        patch.object(context, "certificate", limited_case),
        patch.object(
            context, "_analytic_inputs", lambda _name: analytic_inputs("limited")
        ),
        patch.object(
            context,
            "certificate_field",
            lambda field, _name: certificate_field(field, "limited"),
        ),
    ):
        return context.build(cells, cache)


def run_row(cells, cache):
    built = _limited_context(cells, cache)
    print(f"BUILT requested={cells} realised={len(built.reference)}", flush=True)
    reading = built.reading(built.reference)
    owner_cells = np.flatnonzero(
        np.asarray(reading.booking_cells & (reading.membership > 0))
    ).tolist()
    operands = without_profile(built.operands)

    def mapped(component):
        return jax.jit(
            lambda moments: map_moments(
                moments,
                *operands[:7],
                built.profile,
                *operands[7:],
                component=component,
            )
        )(built.reference)

    lobe = mapped("lobe")
    halfplane = mapped("halfplane")
    assert bool(lobe.booking_valid) and bool(halfplane.booking_valid)
    assert bool(lobe.topology_valid) and bool(halfplane.topology_valid)
    exterior, _ = _exterior(cache, built.carrier_key)
    if exterior is None:
        raise FileNotFoundError("the limited carrier has no read-only exterior")
    operator = fixture.forward_operator(built.source, built.machine)
    coordinates = np.vstack(
        (built.machine.node, built.machine.wall_node, built.machine.sample_coordinates)
    )
    analytic = jnp.asarray(
        certificate._exact_state(
            "weak-rotation-reactor-static", built.exact, coordinates
        )
    )
    span = float(jnp.abs(reading.boundary_flux - reading.axis_flux))
    assert span > 0 and float(jnp.ptp(analytic)) > 0

    def error(moments):
        return (
            float(jnp.max(jnp.abs(exterior + _image(operator, moments) - analytic)))
            / span
        )

    result = {
        "requested": cells,
        "realised": len(built.reference),
        "owner_cells": owner_cells,
        "lobe_error": error(lobe.moments),
        "halfplane_error": error(halfplane.moments),
        "reference_error": error(built.reference),
        "max_lobe_halfplane_moment_delta": float(
            jnp.max(jnp.abs(lobe.moments - halfplane.moments))
        ),
    }
    print("SUMMARY " + json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    receipt = {"rows": [], "errors": []}

    def persist():
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")

    persist()
    for cells in args.sizes:
        try:
            receipt["rows"].append(run_row(cells, args.cache))
        except Exception as error:  # noqa: BLE001
            receipt["errors"].append({"requested": cells, "error": repr(error)})
            print(f"FAILED requested={cells} error={error!r}", flush=True)
        persist()
    if receipt["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
