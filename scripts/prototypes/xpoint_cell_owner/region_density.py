"""Split the owner-slice current error between quadrature order and representation.

For every owner cell at one carrier size the booked current and first moments
are compared against a dense reference integrated over the same half-plane
region under two density representations. The difference between the traced
integral and the dense reference taken under the *same* representation is the
quadrature-order term; the difference between the two dense references is the
representation term. Which one carries the error decides the repair.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path

import jax
import numpy as np

import nova
from scripts.prototypes.fast_map_production import program
from scripts.prototypes.fine_grid_convergence.context import build
from scripts.prototypes.xpoint_cell_owner.owner_map import owned_cut
from scripts.prototypes.xpoint_cell_owner.region_reference import region_moments

MODES = ("fitted", "profile")


def relative(observed, reference, pitch):
    """Scale a current and first-moment residual by the moment that carries it.

    The owner integral emits current, radial moment and vertical moment; the
    area is membership and is reported separately.
    """
    current = max(abs(reference[0]), 1e-14)
    scale = np.maximum(
        np.abs(reference), np.array((current, current * pitch, current * pitch))
    )
    return np.abs(observed - reference) / np.maximum(scale, 1e-14)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() in ("gpu", "cpu")
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()} "
        f"backend={jax.default_backend()}",
        flush=True,
    )
    context = build(args.cells, args.cache)
    count = len(context.reference)
    reading = context.reading(context.reference)
    geometry = context.geometry
    profile = context.profile
    sigma = context.convention.sigma
    span = float(reading.boundary_flux - reading.axis_flux)
    density = program._density_coefficients(geometry, reading, profile)
    active = np.asarray(reading.booking_cells)
    owner = np.flatnonzero(active)
    result = {
        "kind": "region-density",
        "cells": count,
        "span": span,
        "owner_cells": [int(cell) for cell in owner],
        "rows": [],
    }
    path = args.output
    path.write_text(json.dumps(result, indent=2) + "\n")
    traced = {}
    for mode in MODES:
        _, moments, valid = owned_cut(
            reading,
            geometry,
            profile,
            sigma,
            component="halfplane",
            density_mode=mode,
        )
        traced[mode] = (np.asarray(moments), bool(valid))
    for cell in owner:
        vertices = np.asarray(
            geometry.vertices[cell, : int(geometry.vertex_count[cell])]
        )
        centre = np.asarray(geometry.centre[cell])
        pitch = float(geometry.pitch[cell])
        coefficients = np.asarray(reading.field_coefficients[cell])
        row = {
            "cell": int(cell),
            "pitch": pitch,
            "full_area": float(geometry.full_area[cell]),
            "traced": {},
            "dense": {},
        }
        for mode in MODES:
            moments = traced[mode][0][cell]
            row["traced"][mode] = moments.tolist()
            dense, dense_valid = region_moments(
                vertices,
                centre,
                pitch,
                coefficients,
                float(reading.boundary_flux),
                float(sigma),
                np.asarray(reading.saddle_form.position),
                np.asarray(reading.axis),
                np.asarray(density[cell]),
                span=span,
                profile=profile,
                density_mode=mode,
            )
            row["dense"][mode] = dense.tolist()
            row[f"dense_valid_{mode}"] = dense_valid
        area = row["dense"]["fitted"][0]
        row["area_fraction"] = area / row["full_area"]
        row["dense_area_fraction_profile"] = (
            row["dense"]["profile"][0] / row["full_area"]
        )
        row["quadrature_order"] = relative(
            np.asarray(row["traced"]["fitted"]),
            np.asarray(row["dense"]["fitted"])[1:],
            pitch,
        ).tolist()
        row["representation"] = relative(
            np.asarray(row["dense"]["fitted"])[1:],
            np.asarray(row["dense"]["profile"])[1:],
            pitch,
        ).tolist()
        row["traced_gap"] = relative(
            np.asarray(row["traced"]["fitted"]),
            np.asarray(row["traced"]["profile"]),
            pitch,
        ).tolist()
        result["rows"].append(row)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"CELL {json.dumps(row)}", flush=True)
    quadrature = max(max(row["quadrature_order"]) for row in result["rows"])
    representation = max(max(row["representation"]) for row in result["rows"])
    result["max_quadrature_order"] = quadrature
    result["max_representation"] = representation
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"SUMMARY cells={count} owners={len(owner)} "
        f"quadrature={quadrature:.12g} representation={representation:.12g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
