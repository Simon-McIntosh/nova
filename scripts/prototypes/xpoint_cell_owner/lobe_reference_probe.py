"""Compare the booked lobe integral against an independent exact-rule reference.

For every owner cell at a carrier size this reports the lobe the booking cuts,
the same fan integrated in closed form, the cell's whole polygon in closed form,
and the remainder the partition would book (polygon less lobe). The exact route
shares no quadrature with the booking, so agreement to roundoff is what the two
sides of the partition must show.
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
    capacity = max(int(indices.size), 1)
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

    rows = []
    for slot, index in enumerate(indices):
        lobe, rings, sizes = reference.lobe_reference(
            geometry, reading, int(index), density, segments=args.segments
        )
        polygon = reference.polygon_reference(geometry, int(index), density)
        remainder = polygon - lobe
        rows.append(
            {
                "cell": int(index),
                "booked": booked[slot].tolist(),
                "exact_lobe": np.asarray(lobe).tolist(),
                "exact_polygon": np.asarray(polygon).tolist(),
                "remainder": np.asarray(remainder).tolist(),
                "quadrature_relative": relative(booked[slot], lobe),
                "partition_relative": relative(lobe + remainder, polygon),
            }
        )

    result = {
        "cells": int(len(context.reference)),
        "segments": args.segments,
        "owner_cells": int(indices.size),
        "booking_valid": bool(valid),
        "rows": rows,
        "max_quadrature_relative": max(
            (row["quadrature_relative"] for row in rows), default=None
        ),
        "max_partition_relative": max(
            (row["partition_relative"] for row in rows), default=None
        ),
        "capacity": capacity,
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
            f"  cell={row['cell']} lobe_area={row['exact_lobe'][0]:.12g} "
            f"lobe_current={row['exact_lobe'][1]:.12g} "
            f"poly_current={row['exact_polygon'][1]:.12g} "
            f"remainder_current={row['remainder'][1]:.12g}",
            flush=True,
        )


if __name__ == "__main__":
    main()