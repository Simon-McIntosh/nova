"""Check the booked lobe-and-remainder partition against dense references.

For every owner cell of a carrier the lobe booking is compared with the closed
form over the same ring fan, and the booked lobe plus the booked remainder is
compared with the closed form over the whole cell polygon. The references share
the ring builder with the booking but no quadrature, so a disagreement larger
than roundoff is a disagreement about the region each side integrates.

One job carries every requested carrier size as a row and persists each row as
it lands, so an allocation that ends early loses one row rather than the run.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
from time import perf_counter

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


class Stage:
    """Wall time between successive checkpoints, printed as each one passes."""

    def __init__(self):
        self.started = perf_counter()
        self.previous = self.started

    def mark(self, name):
        now = perf_counter()
        print(
            f"STAGE {name} seconds={now - self.previous:.3f} "
            f"total={now - self.started:.3f}",
            flush=True,
        )
        self.previous = now


def deviation(booked, reference):
    booked = np.asarray(booked)
    reference = np.asarray(reference)
    absolute = np.abs(booked - reference)
    relative = absolute / np.where(np.abs(reference) > 0, np.abs(reference), 1.0)
    return {
        "max_absolute": float(np.max(absolute)),
        "max_relative": float(np.max(relative)),
        "per_column_absolute": absolute.tolist(),
        "per_column_relative": relative.tolist(),
    }


def run_row(cells, cache, capacity, segments):
    stage = Stage()
    context = build(cells, cache)
    stage.mark("build")
    reading = context.reading(context.reference)
    stage.mark("read")
    geometry = context.geometry
    profile = context.profile
    density = program._density_coefficients(geometry, reading, profile)
    stage.mark("density")
    span = reading.boundary_flux - reading.axis_flux
    sigma = context.convention.sigma

    active = np.asarray(reading.booking_cells) & (np.asarray(reading.membership) > 0)
    indices = np.flatnonzero(active)
    row = {"cells": cells, "segments": segments, "owner_cells": indices.tolist()}
    if indices.size == 0 or indices.size > capacity:
        row["skipped"] = "owner cell count outside capacity"
        return row

    packed = jnp.asarray(indices)
    lobe, lobe_valid = lobe_moments(
        geometry, reading, profile, packed, density, span, sigma, segments=segments
    )
    stage.mark("lobe_booking")
    remainder, remainder_valid = remaining_moments(
        geometry, reading, profile, packed, density, span, sigma, segments=segments
    )
    stage.mark("remainder_booking")
    lobe = np.asarray(lobe)
    remainder = np.asarray(remainder)

    lobe_booked = []
    cell_booked = []
    lobe_dense = []
    cell_dense = []
    for position, index in enumerate(indices.tolist()):
        dense_lobe, _, _ = lobe_reference(
            geometry, reading, index, density, segments=segments
        )
        dense_cell = polygon_reference(geometry, index, density, segments=segments)
        lobe_dense.append(np.asarray(dense_lobe))
        cell_dense.append(np.asarray(dense_cell))
        lobe_booked.append(lobe[position])
        cell_booked.append(lobe[position] + remainder[position])
    stage.mark("references")

    lobe_booked = np.asarray(lobe_booked)
    cell_booked = np.asarray(cell_booked)
    lobe_dense = np.asarray(lobe_dense)
    cell_dense = np.asarray(cell_dense)

    row["lobe_valid"] = bool(np.all(np.asarray(lobe_valid)))
    row["remainder_valid"] = bool(np.all(np.asarray(remainder_valid)))
    row["columns"] = list(COLUMNS)
    row["lobe_vs_dense_lobe"] = deviation(lobe_booked, lobe_dense)
    row["partition_vs_dense_whole_cell"] = deviation(cell_booked, cell_dense)
    row["rows"] = [
        {
            "cell": index,
            "lobe": lobe_booked[position].tolist(),
            "remainder": remainder[position].tolist(),
            "booked_whole_cell": cell_booked[position].tolist(),
            "dense_lobe": lobe_dense[position].tolist(),
            "dense_whole_cell": cell_dense[position].tolist(),
        }
        for position, index in enumerate(indices.tolist())
    ]
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--segments", type=int, default=4)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True

    result = {
        "jax_default_backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "cache": str(args.cache),
        "rows": [],
        "errors": [],
    }

    def persist():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    persist()
    for cells in args.sizes:
        print(f"ROW cells={cells}", flush=True)
        try:
            row = run_row(cells, args.cache, args.capacity, args.segments)
        except Exception as error:  # noqa: BLE001
            result["errors"].append({"cells": cells, "error": repr(error)})
            persist()
            print(f"ROW cells={cells} FAILED {error!r}", flush=True)
            continue
        result["rows"].append(row)
        persist()
        print(f"ROW cells={cells} {json.dumps(row)[:400]}", flush=True)

    persist()
    for row in result["rows"]:
        if "skipped" in row:
            print(f"SUMMARY cells={row['cells']} skipped={row['skipped']}", flush=True)
            continue
        print(
            f"SUMMARY cells={row['cells']} "
            f"lobe_max_abs={row['lobe_vs_dense_lobe']['max_absolute']:.3e} "
            f"partition_max_abs="
            f"{row['partition_vs_dense_whole_cell']['max_absolute']:.3e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
