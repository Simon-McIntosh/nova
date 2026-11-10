"""Compare every material qualification target-edge pair with precise rings."""

import argparse
import concurrent.futures
import json
import math
import multiprocessing
from pathlib import Path
import subprocess
import sys
import types
import warnings

import numpy as np
import shapely

from nova.jax.config import configure_dtypes

from nova.biot import sheet
from nova.biot.tiledassembly import cell_edges
from scripts.edge_sheet.reference import graded_precise_ring_reference

configure_dtypes()


def meshes():
    owner = types.ModuleType("committed_mesh_owner")
    sys.modules[owner.__name__] = owner
    source = subprocess.check_output(
        ["git", "show", "25eabd5de:nova/biot/magnetisation.py"], text=True
    )
    exec(compile(source, "<committed material mesh owner>", "exec"), owner.__dict__)
    sphere = (
        shapely.Point(0, 0)
        .buffer(0.5, 256)
        .intersection(shapely.box(0, -0.5, 0.5, 0.5))
    )
    shell = (
        shapely.Point(0, 0)
        .buffer(1, 512)
        .difference(shapely.Point(0, 0).buffer(0.9, 512))
        .intersection(shapely.box(0, -1, 1, 1))
    )
    return [
        ("sphere", owner._mesh(sphere, 0.1), np.array([[1.0, 0.0], [0.0, 1.0]])),
        (
            "shell",
            owner._mesh(shell, 0.04),
            np.array([[2.0, 0.0], [0.0, 2.0], [3.0, 1.0]]),
        ),
    ]


def scalar_assembly_roundoff(owner, radius, target_z):
    """Bound the six scalar final-assembly operations before cancellation.

    The existing endpoint allowance retains the kernel and common-scale work.
    This term uses the four uncancelled S and r*S_r contributions instead of
    their nearly cancelling endpoint sums. It affects only the vertical field.
    """
    if (
        radius == 0
        or owner._flat
        or owner._slope == 0
        or owner.start[0] == owner.end[0] == 0
    ):
        return None
    if sheet._near_axis(np, radius, target_z, *owner.start, *owner.end):
        return None
    if sheet._source_series_ends(np, radius, target_z, *owner.start, *owner.end)[0]:
        return None
    factor = sheet.MU0 / (2 * np.pi * radius * (1 + owner._slope**2))
    terms, endpoints = [], []
    for source, height, sign in (
        (owner.end[0], owner.end[1], 1.0),
        (owner.start[0], owner.start[1], -1.0),
    ):
        value, _, radial = map(
            float, owner._closed_gradient(height - target_z, radius, source)
        )
        contributions = [sign * factor * value, sign * factor * radius * radial]
        terms.extend(contributions)
        endpoints.append(
            {"S": value, "r_Sr": radius * radial, "physical_terms": contributions}
        )
    unit_roundoff = 2.0**-53
    operations = 6  # two products, two endpoint sums, one difference, one scale
    magnitude_sum = math.fsum(abs(value) for value in terms)
    # Outward rounding covers two products per physical contribution and the
    # correctly rounded positive fsum used to measure its conditioning scale.
    magnitude_upper = np.nextafter(magnitude_sum / (1 - 3 * unit_roundoff), np.inf)
    gamma = operations * unit_roundoff / (1 - operations * unit_roundoff)
    bound = float(np.nextafter(gamma * magnitude_upper, np.inf))
    return {
        "operations": operations,
        "unit_roundoff": unit_roundoff,
        "field_scale": factor,
        "endpoints": endpoints,
        "physical_terms": terms,
        "sum_abs_terms": magnitude_sum,
        "sum_abs_terms_upper": float(magnitude_upper),
        "scalar_assembly_roundoff": bound,
    }


def admit_cached_census(source, output):
    """Admit recorded scalar misses using only their assembly-rounding receipt.

    This reads the completed census; it performs no new mesh census, ring
    integration or packed evaluation. Already-passing rows stay certified by
    their original allowance because the additional term is nonnegative.
    """
    output.mkdir(parents=True, exist_ok=True)
    final = []
    for name in ("sphere", "shell"):
        operands = np.load(source / f"{name}-operands.npz")
        values = np.load(source / f"{name}-reference.npz")
        reference = values["reference"]
        epsilon = np.finfo(float).eps
        original = (
            1e-10 * abs(reference)
            + 3504 * epsilon / (1 - 3504 * epsilon) * operands["end_scales"]
        )
        summary = {
            "mesh": name,
            "pairs": len(reference),
            "source_receipt": str(source),
            "new_census_executed": False,
            "additional_term_nonnegative": True,
        }
        for route, field in [
            ("scalar", values["scalar"]),
            ("packed", operands["packed"]),
            ("dilated", operands["dilated"]),
        ]:
            bad = (abs(field - reference) > original).any(axis=1) | ~np.isfinite(
                field
            ).all(axis=1)
            receipts = []
            remaining = 0
            for index in np.flatnonzero(bad):
                if route != "scalar":
                    remaining += 1
                    continue
                row = operands["rows"][index]
                owner = sheet.SheetFlux(row[2:4], row[4:6])
                receipt = scalar_assembly_roundoff(owner, *row[:2])
                assert receipt is not None
                # The named pair is a positive control for the instrument.
                assert owner.field(*row[:2])[1] == field[index, 1]
                allowance = np.array([0.0, receipt["scalar_assembly_roundoff"]])
                admitted = np.all(
                    abs(field[index] - reference[index]) <= original[index] + allowance
                )
                remaining += int(not admitted)
                receipt.update(
                    {
                        "pair_index": int(index),
                        "pair": row.tolist(),
                        "recorded_field": field[index].tolist(),
                        "reference": reference[index].tolist(),
                        "absolute_error": abs(field[index] - reference[index]).tolist(),
                        "original_budget": original[index].tolist(),
                        "adjusted_budget": (original[index] + allowance).tolist(),
                        "admitted": bool(admitted),
                    }
                )
                receipts.append(receipt)
            summary[route] = {
                "previous_failed_pairs": int(bad.sum()),
                "remaining_failed_pairs": remaining,
                "rounding_receipts": receipts,
            }
        print(json.dumps(summary), flush=True)
        (output / f"{name}-admission.json").write_text(json.dumps(summary, indent=2))
        final.append(summary)
    assert all(
        value[route]["remaining_failed_pairs"] == 0
        for value in final
        for route in ("scalar", "packed", "dilated")
    ), "derived term does not admit the receipt"


def reference_chunk(rows):
    result = []
    for row in rows:
        r, z, a, h, b, k = row
        value = graded_precise_ring_reference((a, h), (b, k), (r, z))
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            owner = sheet.SheetFlux((a, h), (b, k))
            scalar = owner.field(r, z)
            rounding = scalar_assembly_roundoff(owner, r, z)
        result.append(
            (
                *scalar,
                *value[1:],
                0.0 if rounding is None else rounding["scalar_assembly_roundoff"],
            )
        )
    return np.asarray(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--admit-cached", type=Path)
    args = parser.parse_args()
    if args.admit_cached is not None:
        admit_cached_census(args.admit_cached, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        "module:",
        sheet.__file__,
        "cwd:",
        Path.cwd(),
        "revision:",
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        flush=True,
    )
    import jax
    import jax.numpy as jnp

    assert jax.config.jax_enable_x64
    evaluate = jax.jit(lambda *p: jnp.stack(sheet.packed_sheet_field(jnp, *p), axis=-1))
    scales = jax.jit(
        lambda *p: jnp.stack(sheet.packed_sheet_end_scale(jnp, *p), axis=-1)
    )
    failures = 0
    for name, mesh, probes in meshes():
        # The material owner uses this exact dilation before edge identity rounding.
        edges = cell_edges(tuple(p * 1024 for p in mesh.polygons))
        sources = (
            np.column_stack((edges.start_r, edges.start_z, edges.end_r, edges.end_z))
            / 1024
        )
        authored = np.concatenate(mesh.polygons)
        identity_movement = np.max(
            np.abs(np.round(authored * 1024, 9) / 1024 - authored)
        )
        assert identity_movement > 0, (
            "the edge-rounding control must see a moved coordinate"
        )
        print(
            name,
            "identity grid step:",
            1e-9 / 1024,
            "known-present grid movement:",
            identity_movement,
            flush=True,
        )
        targets = np.concatenate((mesh.centroids, probes))
        rows = np.column_stack(
            (
                np.repeat(targets, len(sources), axis=0),
                np.tile(sources, (len(targets), 1)),
            )
        )
        print(
            name,
            "cells:",
            len(mesh.centroids),
            "edges:",
            len(sources),
            "targets:",
            len(targets),
            "pairs:",
            len(rows),
            flush=True,
        )
        canonical = sheet._binary_field_geometry(np, *rows.T)
        dilated_canonical = sheet._binary_field_geometry(np, *(rows * 1024).T)
        assert all(np.array_equal(a, b) for a, b in zip(canonical, dilated_canonical))
        print(
            name,
            "canonical operands bit-identical under 1024 dilation: True",
            flush=True,
        )
        packed = []
        end_scales = []
        dilated = []
        for start in range(0, len(rows), 256):
            batch = rows[np.minimum(np.arange(start, start + 256), len(rows) - 1)]
            n = min(256, len(rows) - start)
            packed.append(np.asarray(evaluate(*map(jnp.asarray, batch.T)))[:n])
            dilated.append(
                np.asarray(evaluate(*map(jnp.asarray, (batch * 1024).T)))[:n]
            )
            end_scales.append(np.asarray(scales(*map(jnp.asarray, batch.T)))[:n])
        packed = np.concatenate(packed)
        dilated = np.concatenate(dilated)
        end_scales = np.concatenate(end_scales)
        np.savez(
            args.output / f"{name}-operands.npz",
            rows=rows,
            packed=packed,
            dilated=dilated,
            end_scales=end_scales,
        )
        chunks = [rows[i : i + 32] for i in range(0, len(rows), 32)]
        result = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for index, values in enumerate(pool.map(reference_chunk, chunks)):
                result.append(values)
                if index % 100 == 0:
                    print(
                        name,
                        "reference pairs:",
                        min((index + 1) * 32, len(rows)),
                        flush=True,
                    )
        result = np.concatenate(result)
        scalar = result[:, :2]
        reference = result[:, 2:4]
        scalar_assembly = np.column_stack((np.zeros(len(result)), result[:, 4]))
        np.savez(
            args.output / f"{name}-reference.npz",
            rows=rows,
            scalar=scalar,
            reference=reference,
            scalar_assembly_roundoff=scalar_assembly,
        )
        # Count the actual remainder sum, multipole recurrence and elementary
        # endpoint work before applying the standard gamma_n rounding allowance.
        operations = sheet._SHEET_FIELD_PANELS * sheet._SHEET_FIELD_NODES + 80 + 64
        epsilon = np.finfo(float).eps
        gamma = operations * epsilon / (1 - operations * epsilon)
        floor = gamma * end_scales
        # Both routes and the reference use the same once-rounded edges.
        # Binary dilation and canonicalisation are exact: additional geometry
        # rounding is zero, so no allowance can be added for it.
        dilation_geometry_rounding = np.zeros_like(floor)
        base_tolerance = 1e-10 * np.abs(reference) + floor + dilation_geometry_rounding
        assert np.array_equal(packed, dilated), (
            "canonical field must preserve every bit"
        )
        summary = {
            "mesh": name,
            "pairs": len(rows),
            "absolute_floor": f"gamma_{operations} times absolute endpoint sum",
            "max_floor": float(floor.max()),
            "dilation_additional_geometry_floor": 0.0,
            "identity_grid_step": 1e-9 / 1024,
            "max_identity_movement": float(identity_movement),
            "packed_dilated_bit_identical": True,
            "scalar_assembly_operations": 6,
            "scalar_unit_roundoff": 2.0**-53,
            "max_scalar_assembly_roundoff": float(scalar_assembly.max()),
        }
        for route, values in [
            ("scalar", scalar),
            ("packed", packed),
            ("dilated", dilated),
        ]:
            tolerance = base_tolerance + (scalar_assembly if route == "scalar" else 0.0)
            bad = (np.abs(values - reference) > tolerance) | ~np.isfinite(values)
            count = int(bad.any(axis=1).sum())
            failures += count
            ratio = np.abs(values - reference) / np.maximum(
                tolerance, np.finfo(float).tiny
            )
            worst = np.unravel_index(np.argmax(ratio), ratio.shape)
            summary[route] = {
                "nonfinite": int((~np.isfinite(values).all(axis=1)).sum()),
                "failed_pairs": count,
                "worst_budget_ratio": float(ratio[worst]),
                "worst_pair": rows[worst[0]].tolist(),
                "got": values[worst[0]].tolist(),
                "reference": reference[worst[0]].tolist(),
            }
        print(json.dumps(summary), flush=True)
        (args.output / f"{name}-summary.json").write_text(json.dumps(summary, indent=2))
    assert failures == 0, f"{failures} inaccurate pairs"


if __name__ == "__main__":
    main()
