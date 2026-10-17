"""Attribute a diverted map's flux error to the cells that carry it."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import nova
from benchmarks import solovev_certificate as certificate
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.fine_grid_convergence.context import build
from scripts.prototypes.lean_map.probe_bias import _image
from scripts.prototypes.solve_options.point_bias import _exterior
from scripts.prototypes.xpoint_cell_owner.owner_map import map_moments
from scripts.prototypes.xpoint_cell_owner.runtime_solve import without_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=6)
    parser.add_argument(
        "--component",
        choices=(
            "halfplane",
            "bell",
            "saddle",
            "pinned",
            "anchored",
            "shifted",
            "ordinary",
            "oriented",
        ),
        default="halfplane",
    )
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() in ("gpu", "cpu")
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()} "
        f"component={args.component}",
        flush=True,
    )
    context = build(args.cells, args.cache)
    count = len(context.reference)
    operands = without_profile(context.operands)
    profile = context.profile
    mapped = jax.jit(
        lambda moments, carrier: map_moments(
            moments, *carrier[:7], profile, *carrier[7:], component=args.component
        )
    )
    state = mapped(context.reference, operands)
    booked = state.moments
    residual = np.asarray(booked - context.reference)
    coordinates = np.vstack(
        (
            context.machine.node,
            context.machine.wall_node,
            context.machine.sample_coordinates,
        )
    )
    analytic = jnp.asarray(
        certificate._exact_state("diverted-single-null", context.exact, coordinates)
    )
    exterior, _ = _exterior(context.cache, context.carrier_key)
    operator = fixture.forward_operator(context.source, context.machine)
    reading = context.reading(context.reference)
    span = float(jnp.abs(reading.boundary_flux - reading.axis_flux))

    def flux_error(moments):
        flux = exterior + _image(operator, moments)
        return float(jnp.max(jnp.abs(flux - analytic))) / span

    delivered = flux_error(booked)
    reference_only = flux_error(context.reference)
    magnitude = np.linalg.norm(residual, axis=1)
    order = np.argsort(-magnitude)
    scale = float(np.max(np.abs(np.asarray(context.reference))))
    result = {
        "kind": "attribution",
        "component": args.component,
        "cells": count,
        "span": span,
        "delivered_flux_error": delivered,
        "reference_flux_error": reference_only,
        "support": int(state.graph_cells),
        "rows": [],
        "cumulative": [],
    }
    path = args.output
    path.write_text(json.dumps(result, indent=2) + "\n")
    patched = booked
    for rank, cell in enumerate(order[: args.top]):
        row = {
            "rank": rank,
            "cell": int(cell),
            "moment_residual": residual[cell].tolist(),
            "relative_moment_residual": float(magnitude[cell] / scale),
            "booked_membership": float(reading.geometric_membership[cell]),
            "booking_cell": bool(reading.booking_cells[cell]),
            "path_weight": float(reading.path_weight[cell]),
            "error_without": flux_error(booked.at[cell].set(context.reference[cell])),
        }
        result["rows"].append(row)
        patched = patched.at[cell].set(context.reference[cell])
        result["cumulative"].append(
            {
                "through_rank": rank,
                "cells": [int(item) for item in order[: rank + 1]],
                "error_with_removed": flux_error(patched),
            }
        )
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"ATTRIB {json.dumps(row)}", flush=True)
    print(
        f"SUMMARY component={args.component} cells={count} delivered={delivered:.12g} "
        f"reference={reference_only:.12g} "
        f"removed={result['cumulative'][-1]['error_with_removed']:.12g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
