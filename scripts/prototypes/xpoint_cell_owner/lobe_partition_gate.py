"""Check the booked lobe-and-remainder partition against dense references.

For every owner cell of a carrier the lobe booking is compared with the closed
form over the same ring fan, and the booked lobe plus the booked remainder is
compared with the closed form over the whole cell polygon. The references share
the ring builder with the booking but no quadrature, so a disagreement larger
than roundoff is a disagreement about the region each side integrates.
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
from scripts.prototypes.xpoint_cell_owner.lobe_booking import (
    lobe_moments,
    remaining_moments,
)
from scripts.prototypes.xpoint_cell_owner.lobe_reference import (
    lobe_reference,
    polygon_reference,
)

COLUMNS = ("area", "current", "radial_moment", "vertical_moment")


def deviation(booked, reference, scale):
    booked = np.asarray(booked)
    reference = np.asarray(reference)
    absolute = np.abs(booked - reference)
    relative = absolute / np.where(np.abs(reference) > 0, np.abs(reference), 1.0)
    return {
        "max_absolute": float(np.max(absolute)),
        "max_relative": float(np.max(relative)),
        "per_column_absolute": absolute.tolist(),
        "per_column_relative": relative.tolist(),
        "scale": float(scale),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--segments", type=int, default=4)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True

    context = build(args.cells, args.cache)
    reading = context.reading(context.reference)
    geometry = context.geometry
    profile = context.profile
    density = program._density_coefficients(geometry, reading, profile)
    span = reading.boundary_flux - reading.axis_flux
    sigma = context.convention.sigma

    active = np.asarray(reading.booking_cells) & (np.asarray(reading.membership) > 0)
    indices = np.flatnonzero(active)
    result = {
        "cells": args.cells,
        "segments": args.segments,
        "owner_cells": indices.tolist(),
        "rows": [],
    }
    if indices.size == 0 or indices.size > args.capacity:
        result["skipped"] = "owner cell count outside capacity"
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        return

    packed = jnp.asarray(indices)
    lobe, lobe_valid = lobe_moments(
        geometry, reading, profile, packed, density, span, sigma, segments=args.segments
    )
    remainder, remainder_valid = remaining_moments(
        geometry, reading, profile, packed, density, span, sigma, segments=args.segments
    )
    lobe = np.asarray(lobe)
    remainder = np.asarray(remainder)
    scale = float(span)

    lobe_rows = []
    cell_rows = []
    lobe_reference_rows = []
    cell_reference_rows = []
    for position, index in enumerate(indices.tolist()):
        dense_lobe, _, _ = lobe_reference(
            geometry, reading, index, density, segments=args.segments
        )
        dense_cell = polygon_reference(geometry, index, density, segments=args.segments)
        lobe_reference_rows.append(np.asarray(dense_lobe))
        cell_reference_rows.append(np.asarray(dense_cell))
        lobe_rows.append(lobe[position])
        cell_rows.append(lobe[position] + remainder[position])

    lobe_rows = np.asarray(lobe_rows)
    cell_rows = np.asarray(cell_rows)
    lobe_reference_rows = np.asarray(lobe_reference_rows)
    cell_reference_rows = np.asarray(cell_reference_rows)

    result["lobe_valid"] = bool(np.all(np.asarray(lobe_valid)))
    result["remainder_valid"] = bool(np.all(np.asarray(remainder_valid)))
    result["lobe_vs_reference"] = deviation(lobe_rows, lobe_reference_rows, scale)
    result["partition_vs_whole_cell"] = deviation(
        cell_rows, cell_reference_rows, scale
    )
    for position, index in enumerate(indices.tolist()):
        result["rows"].append(
            {
                "cell": index,
                "lobe": lobe_rows[position].tolist(),
                "remainder": remainder[position].tolist(),
                "booked_whole_cell": cell_rows[position].tolist(),
                "dense_lobe": lobe_reference_rows[position].tolist(),
                "dense_whole_cell": cell_reference_rows[position].tolist(),
            }
        )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"cells={args.cells} owners={indices.tolist()} "
        f"lobe_max_abs={result['lobe_vs_reference']['max_absolute']:.3e} "
        f"partition_max_abs={result['partition_vs_whole_cell']['max_absolute']:.3e}",
        flush=True,
    )


if __name__ == "__main__":
    main()