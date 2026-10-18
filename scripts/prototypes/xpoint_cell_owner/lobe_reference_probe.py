"""Check every booked owner cell against independent closed-form references."""

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

EXPECTED_CARRIERS = {250: 282, 500: 550, 2000: 2098}
TOLERANCE = 1e-12
COLUMNS = ("area", "current", "radial_moment", "vertical_moment")


def deviation(booked, reference):
    difference = np.abs(np.asarray(booked) - np.asarray(reference))
    return {
        "max_absolute": float(np.max(difference)),
        "per_column_absolute": np.max(difference, axis=0).tolist(),
    }


def run_row(cells, cache, segments):
    built = build(cells, cache)
    reading = built.reading(built.reference)
    geometry = built.geometry
    density = program._density_coefficients(geometry, reading, built.profile)
    indices = np.flatnonzero(
        np.asarray(reading.booking_cells) & (np.asarray(reading.membership) > 0)
    )
    if not indices.size:
        raise AssertionError(f"carrier {cells} has no owner cells")
    if len(built.reference) != EXPECTED_CARRIERS[cells]:
        raise AssertionError(
            f"carrier {cells} realised {len(built.reference)}, "
            f"expected {EXPECTED_CARRIERS[cells]}"
        )

    arguments = (
        geometry,
        reading,
        built.profile,
        jnp.asarray(indices),
        density,
        reading.boundary_flux - reading.axis_flux,
        built.convention.sigma,
    )
    lobe, lobe_valid = lobe_moments(*arguments, segments=segments)
    remainder, remainder_valid = remaining_moments(*arguments, segments=segments)
    lobe = np.asarray(lobe)
    remainder = np.asarray(remainder)
    expected_lobe = []
    expected_whole = []
    for index in indices.tolist():
        dense_lobe, _, _ = lobe_reference(
            geometry, reading, index, density, segments=segments
        )
        expected_lobe.append(np.asarray(dense_lobe))
        expected_whole.append(
            np.asarray(polygon_reference(geometry, index, density, segments=segments))
        )
    lobe_error = deviation(lobe, np.asarray(expected_lobe))
    partition_error = deviation(lobe + remainder, np.asarray(expected_whole))
    row = {
        "requested": cells,
        "realised": len(built.reference),
        "owner_cells": indices.tolist(),
        "lobe_valid": bool(np.all(np.asarray(lobe_valid))),
        "remainder_valid": bool(np.all(np.asarray(remainder_valid))),
        "columns": COLUMNS,
        "lobe_vs_dense_lobe": lobe_error,
        "partition_vs_dense_whole_cell": partition_error,
    }
    for label, values in (("lobe", lobe), ("remainder", remainder)):
        if values.shape != (indices.size, len(COLUMNS)) or not np.all(
            np.isfinite(values)
        ):
            raise AssertionError(f"carrier {cells} has invalid {label} values")
    if not row["lobe_valid"] or not row["remainder_valid"]:
        raise AssertionError(f"carrier {cells} has invalid bookings")
    if (
        not np.isfinite(lobe_error["max_absolute"])
        or lobe_error["max_absolute"] > TOLERANCE
    ):
        raise AssertionError(f"carrier {cells} lobe error {lobe_error['max_absolute']}")
    if (
        not np.isfinite(partition_error["max_absolute"])
        or partition_error["max_absolute"] > TOLERANCE
    ):
        raise AssertionError(
            f"carrier {cells} partition error {partition_error['max_absolute']}"
        )
    print(
        f"SUMMARY realised={row['realised']} owners={indices.size} "
        f"lobe_max_abs={lobe_error['max_absolute']:.12g} "
        f"partition_max_abs={partition_error['max_absolute']:.12g}",
        flush=True,
    )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segments", type=int, default=4)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    if sorted(args.sizes) != sorted(EXPECTED_CARRIERS):
        raise SystemExit(
            f"all requested carriers are required: {sorted(EXPECTED_CARRIERS)}"
        )
    receipt = {"rows": [], "errors": []}

    def persist():
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")

    persist()
    for cells in args.sizes:
        try:
            receipt["rows"].append(run_row(cells, args.cache, args.segments))
        except Exception as error:  # noqa: BLE001
            receipt["errors"].append({"requested": cells, "error": repr(error)})
            print(f"FAILED requested={cells} error={error!r}", flush=True)
        persist()
    if receipt["errors"] or len(receipt["rows"]) != len(EXPECTED_CARRIERS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
