"""Compare the owner partition and its booked integrals against exact references.

For every owner cell at a carrier size this reports four integrals: the booked
lobe (degree-five rule on the normal-form fan), the same lobe in closed form
(the independent reference), the remainder booked by the owner machinery (the
cell fan less the lobe fan, one integral, one rule), and the whole cell in
closed form on a dense fan. Agreement of the booked lobe with the closed form
measures the quadrature; agreement of booked lobe plus remainder with the dense
whole-cell reference measures the partition.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from scripts.prototypes.fast_map_production import program
from scripts.prototypes.fine_grid_convergence.context import build
from scripts.prototypes.xpoint_cell_owner import lobe_reference as reference
from scripts.prototypes.xpoint_cell_owner.lobe_booking import lobe_moments
from scripts.prototypes.xpoint_cell_owner.lobe_booking import remaining_moments

DENSE = 64


def relative(candidate, expected):
    scale = np.maximum(np.abs(np.asarray(expected)), 1e-300)
    return float(np.max(np.abs(np.asarray(candidate) - np.asarray(expected)) / scale))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segments", type=int, default=4)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True

    context = build(args.cells, args.cache)
    geometry = context.geometry
    reading = context.reading(context.reference)
    profile = context.profile
    density = program._density_coefficients(geometry, reading, profile)
    span = reading.boundary_flux - reading.axis_flux
    sigma = context.convention.sigma

    selected = np.asarray(reading.booking_cells) & (np.asarray(reading.membership) > 0)
    indices = np.nonzero(selected)[0]
    booked, valid = lobe_moments(
        geometry,
        reading,
        profile,
        jnp.asarray(indices),
        density,
        span,
        sigma,
        density_mode="fitted",
        segments=args.segments,
    )
    booked = np.asarray(booked, dtype=float)
    remainder, remainder_valid = remaining_moments(
        geometry,
        reading,
        profile,
        jnp.asarray(indices),
        density,
        span,
        sigma,
        density_mode="fitted",
        segments=args.segments,
    )
    remainder = np.asarray(remainder, dtype=float)

    rows = []
    for slot, index in enumerate(indices):
        lobe, rings, sizes = reference.lobe_reference(
            geometry, reading, int(index), density, segments=args.segments
        )
        dense = reference.polygon_reference(geometry, int(index), density, DENSE)
        whole = np.asarray(lobe) + remainder[slot]
        rows.append(
            {
                "cell": int(index),
                "booked_lobe": booked[slot].tolist(),
                "closed_lobe": np.asarray(lobe).tolist(),
                "booked_remainder": remainder[slot].tolist(),
                "whole_booked": whole.tolist(),
                "dense_whole": np.asarray(dense).tolist(),
                "quadrature_relative": relative(booked[slot], lobe),
                "partition_relative": relative(whole, dense),
            }
        )

    result = {
        "cells": int(len(context.reference)),
        "segments": args.segments,
        "dense_segments": DENSE,
        "owner_cells": int(indices.size),
        "booking_valid": bool(valid),
        "remainder_valid": bool(remainder_valid),
        "rows": rows,
        "max_quadrature_relative": max(
            (row["quadrature_relative"] for row in rows), default=None
        ),
        "max_partition_relative": max(
            (row["partition_relative"] for row in rows), default=None
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"PROBE cells={result['cells']} owners={result['owner_cells']} "
        f"quadrature={result['max_quadrature_relative']:.3e} "
        f"partition={result['max_partition_relative']:.3e}",
        flush=True,
    )
    for row in rows:
        print(
            f"  cell={row['cell']} quad={row['quadrature_relative']:.3e} "
            f"part={row['partition_relative']:.3e} "
            f"lobe_current={row['booked_lobe'][1]:.12g} "
            f"remainder_current={row['booked_remainder'][1]:.12g} "
            f"dense_current={row['dense_whole'][1]:.12g}",
            flush=True,
        )


if __name__ == "__main__":
    main()