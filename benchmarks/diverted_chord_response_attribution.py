"""Attribute archived chord-map errors with fixed-current counterfactuals.

Class projection shares sum to one; squared-error reductions need not be positive.
The Green comparison integrates the same affine full-cell current representation
with two source-area quadratures. It does not change the support or fit amplitudes.
"""

from __future__ import annotations

# Precision must be configured before importing fixture modules that build arrays.
# ruff: noqa: E402

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import jax
from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64
import jax.numpy as jnp
import numpy as np
from shapely.geometry import Point, Polygon

from benchmarks import solovev_certificate as certificate
from benchmarks.plasma_cell_map_fidelity import norms
from nova.biot.greens import traced_filament_greens
from scripts.analytic_oracle_fixtures import measure as fixture

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "docs/figures/plasma-cell-read-fidelity/map-fidelity"
OUTPUT = ROOT / "docs/figures/cut-cell-current-attribution/diverted-chord-response"
CLASSES = ("interior", "separatrix-cut", "X-point cell", "wall-cut", "exterior")
CONTROL = "run the same decomposition on the passing weak-rotation chord rows"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def reduction(error, correction):
    return float(1 - np.sum((error - correction) ** 2) / np.sum(error**2))


def coefficients(physical, second):
    result = physical.copy()
    matrix = np.stack((second[:, [0, 2]], second[:, [2, 1]]), axis=1)
    result[1:] = np.linalg.solve(matrix, physical[1:].T[..., None])[..., 0].T
    return result


def physical_moments(value, second):
    result = value.copy()
    result[1] = second[:, 0] * value[1] + second[:, 2] * value[2]
    result[2] = second[:, 2] * value[1] + second[:, 1] * value[2]
    return result


def rule(order):
    node, weight = np.polynomial.legendre.leggauss(order)
    node, weight = (node + 1) / 2, weight / 2
    u, v = np.meshgrid(node, node, indexing="ij")
    return (
        jnp.asarray(u.ravel()),
        jnp.asarray(v.ravel()),
        jnp.asarray((weight[:, None] * weight[None, :]).ravel()),
    )


@jax.jit
def far_rows(target, polygon, centre, area, u, v, weight):
    a, b = polygon - centre, jnp.roll(polygon, -1, axis=0) - centre
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    points = centre + u[None, :, None] * (
        (1 - v[None, :, None]) * a[:, None] + v[None, :, None] * b[:, None]
    )
    live = jnp.abs(cross) > 1e-25
    points = jnp.where(live[:, None, None], points, centre + 10)
    weights = (cross[:, None] * u[None, :] * weight[None, :]).ravel() / area
    points = points.reshape(-1, 2)
    # Near-field values are replaced by the target-centred rule below.
    gap = jnp.sum((target[:, None] - points[None, :]) ** 2, axis=-1)
    source_r = jnp.where(gap > 0, points[None, :, 0], points[None, :, 0] + 1)
    kernel = traced_filament_greens(
        jnp, target[:, None, 0], target[:, None, 1], source_r, points[None, :, 1]
    )[0]
    basis = jnp.stack((jnp.ones(len(points)), *(points - centre).T))
    return (kernel * weights) @ basis.T


@jax.jit
def near_rows(target, polygon, centre, area, u, v, weight):
    a = polygon[None, :, :] - target[:, None, :]
    b = jnp.roll(a, -1, axis=1)
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    points = target[:, None, None, :] + u[None, None, :, None] ** 2 * (
        (1 - v[None, None, :, None]) * a[:, :, None]
        + v[None, None, :, None] * b[:, :, None]
    )
    live = jnp.abs(cross) > 1e-25
    points = jnp.where(live[..., None, None], points, target[:, None, None] + 10)
    weights = cross[..., None] * (2 * u**3 * weight)[None, None, :] / area
    kernel = traced_filament_greens(
        jnp,
        target[:, None, None, 0],
        target[:, None, None, 1],
        points[..., 0],
        points[..., 1],
    )[0]
    integrand = kernel * weights
    return jnp.stack(
        (
            jnp.sum(integrand, axis=(1, 2)),
            jnp.sum(integrand * (points[..., 0] - centre[0]), axis=(1, 2)),
            jnp.sum(integrand * (points[..., 1] - centre[1]), axis=(1, 2)),
        ),
        axis=-1,
    )


def quadrature(machine, target, booked, analytic, order, near_order):
    """Image both currents and their difference without cancellation of totals."""
    result = np.zeros((3, len(target)))
    centres = machine.moment_geometry.atomic_mesh.centroids
    far_rule, near_rule = rule(order), rule(near_order)
    target_device = jnp.asarray(target)
    for cell, vertices in enumerate(machine.cell_polygons):
        if not np.any(booked[:, cell]) and not np.any(analytic[:, cell]):
            continue
        polygon = np.asarray(vertices)
        signed = np.sum(
            polygon[:, 0] * np.roll(polygon[:, 1], -1)
            - polygon[:, 1] * np.roll(polygon[:, 0], -1)
        )
        if signed < 0:
            polygon = polygon[::-1]
        polygon = np.pad(polygon, ((0, 16 - len(polygon)), (0, 0)), mode="edge")
        centre = np.asarray(centres[cell])
        radius = np.max(np.linalg.norm(vertices - centre, axis=1))
        rows = np.asarray(
            far_rows(
                target_device,
                jnp.asarray(polygon),
                jnp.asarray(centre),
                machine.area[cell],
                *far_rule,
            )
        ).copy()
        near = np.flatnonzero(np.linalg.norm(target - centre, axis=1) < 3 * radius)
        for start in range(0, len(near), 128):
            indices = near[start : start + 128]
            points = np.pad(
                target[indices], ((0, 128 - len(indices)), (0, 0)), mode="edge"
            )
            rows[indices] = np.asarray(
                near_rows(
                    jnp.asarray(points),
                    jnp.asarray(polygon),
                    jnp.asarray(centre),
                    machine.area[cell],
                    *near_rule,
                )
            )[: len(indices)]
        vectors = np.stack(
            (booked[:, cell], analytic[:, cell], booked[:, cell] - analytic[:, cell])
        )
        result += vectors @ rows.T
        if cell % 200 == 0:
            print(f"QUADRATURE order={order}/{near_order} cell={cell}", flush=True)
    assert np.all(np.isfinite(result)), "nonfinite refined Green response"
    return result


def render(row, data, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from benchmarks.plasma_cell_terminal_state import _draw_nulls
    from nova.media import poloidal
    from nova.media.ink import DEFAULT_INK, poloidal_axes
    from nova.media.sources.frame import WallUnit

    units = tuple(
        WallUnit(data["wall"][a:b, 0], data["wall"][a:b, 1], closed=closed, kind=kind)
        for a, b, closed, kind in row["wall_units"]
    )
    field = data["mapped"] - data["analytic"]
    levels = np.linspace(-float(np.max(abs(field))), float(np.max(abs(field))), 17)
    radial, height, raster = certificate._raster_field(
        data["coordinates"], field, data["wall"]
    )
    fig, axis = plt.subplots(figsize=(6, 7), constrained_layout=True)
    contour = poloidal.draw_flux_contours(
        axis, radial, height, raster, levels, color="#444444"
    )
    assert any(len(part) > 1 for group in contour.allsegs for part in group)
    poloidal.draw_wall(axis, units=units)
    _draw_nulls(
        axis,
        row["reference_nulls"],
        units,
        DEFAULT_INK.variant(
            axis_color="#3366cc",
            xpoint_color="#3366cc",
            axis_markersize=10,
            xpoint_markersize=12,
        ),
    )
    _draw_nulls(
        axis,
        row["mapped_nulls"],
        units,
        DEFAULT_INK.variant(axis_markersize=5, xpoint_markersize=6),
    )
    poloidal_axes(axis)
    axis.set_title("Diverted chord: mapped minus analytic / 132 cells")
    fig.supxlabel(
        "17 levels from −0.000997524 to +0.000997524 Wb\n"
        "Blue: analytic nulls; red: booked-map nulls\n"
        "Map sup mismatch 0.0950848; no solve; convergence not applicable",
        fontsize=9,
    )
    for suffix in ("png", "svg"):
        fig.savefig(output / f"mismatch-field.{suffix}", dpi=160)
    plt.close(fig)
    return levels.tolist()


def measure(path, output):
    row = json.loads(path.read_text())
    with np.load(INPUT / row["state_archive"]) as archive:
        data = dict(archive)
    error = data["plasma"] - data["analytic_plasma"]
    measured = norms(error, data["analytic_plasma"])
    for key in ("sup_relative", "rms_relative"):
        np.testing.assert_allclose(
            measured[key], row["plasma_response_mismatch"][key], rtol=1e-9, atol=0
        )
    map_score = norms(data["mapped"] - data["analytic"], data["analytic"])
    np.testing.assert_allclose(
        map_score["sup_relative"], row["mismatch"]["sup_relative"], rtol=1e-9, atol=0
    )
    print(
        f"REPRODUCED {path.stem} map={map_score['sup_relative']:.12g} "
        f"plasma={measured['sup_relative']:.12g}",
        flush=True,
    )
    carrier, _source, exact = certificate._case(row["case"])
    machine = certificate._case_machine(
        row["case"], carrier, exact, row["requested_cells"]
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    np.testing.assert_array_equal(coordinates, data["coordinates"])
    second = np.asarray(machine.moment_geometry.second_moment)
    analytic = coefficients(data["analytic_physical_moments"], second)
    booked = row["lambda"] * data["moments"]
    blocks = np.stack(
        [
            np.vstack(
                [
                    getattr(machine, f"plasma_to_{name}{suffix}")
                    for name in ("grid", "wall", "sample")
                ]
            )
            for suffix in ("", "_r", "_z")
        ]
    )

    def image(values):
        return np.einsum("ktc,kc->t", blocks, values)

    reconstructed = image(booked - analytic)
    reconstruction_relative_sup = float(
        np.max(abs(reconstructed - error)) / np.max(abs(error))
    )
    assert reconstruction_relative_sup <= 1e-9
    for key in ("sup_relative", "rms_relative"):
        np.testing.assert_allclose(
            norms(reconstructed, data["analytic_plasma"])[key],
            measured[key],
            rtol=1e-9,
            atol=0,
        )
    boundary = Polygon(fixture._analytic_separatrix(exact))
    saddle = getattr(exact, "x_point", None)
    classes, cells = [], []
    physical = physical_moments(booked, second)
    centres = np.asarray(machine.moment_geometry.atomic_mesh.centroids)
    for cell, polygon in enumerate(machine.cell_polygons):
        shape = Polygon(polygon)
        overlap = shape.intersection(boundary).area / shape.area
        wall_cut = shape.area < Polygon(machine.sampling_vertices[cell]).area * (
            1 - 1e-8
        )
        x_cell = saddle is not None and shape.buffer(1e-10).covers(
            Point(np.asarray(saddle))
        )
        label = (
            "X-point cell"
            if x_cell
            else "wall-cut"
            if wall_cut
            else "exterior"
            if overlap < 1e-10
            else "interior"
            if overlap > 1 - 1e-8
            else "separatrix-cut"
        )
        classes.append(label)
        cells.append(
            {
                "cell": cell,
                "class": label,
                "analytic_area_fraction": overlap,
                "wall_cut": bool(wall_cut),
                "x_point_cell": bool(x_cell),
                "booked_current_a": float(booked[0, cell]),
                "analytic_current_a": float(analytic[0, cell]),
                "booked_centroid_m": (
                    centres[cell] + physical[1:, cell] / booked[0, cell]
                ).tolist()
                if booked[0, cell]
                else None,
                "analytic_centroid_m": (
                    centres[cell]
                    + data["analytic_physical_moments"][1:, cell] / analytic[0, cell]
                ).tolist()
                if analytic[0, cell]
                else None,
            }
        )
    classes = np.asarray(classes)
    energy = float(error @ error)
    class_fields, class_summary = [], {}
    for label in CLASSES:
        mask = classes == label
        field = image((booked - analytic) * mask)
        class_fields.append(field)
        class_summary[label] = {
            "cells": int(mask.sum()),
            "projection_share": float(error @ field / energy),
            "squared_error_reduction": reduction(error, field),
            "map_sup_relative": float(
                np.max(abs(field)) / np.max(abs(data["analytic"]))
            ),
        }
    np.testing.assert_allclose(
        np.sum(class_fields, axis=0),
        reconstructed,
        rtol=1e-9,
        atol=2e-12 * np.max(abs(error)),
    )
    corrected_physical = physical.copy()
    xmask = classes == "X-point cell"
    for cell in np.flatnonzero(xmask):
        assert analytic[0, cell] != 0, (
            "X-point analytic current cannot define a centroid"
        )
        corrected_physical[1:, cell] = (
            booked[0, cell]
            * data["analytic_physical_moments"][1:, cell]
            / analytic[0, cell]
        )
    placement = image(booked - coefficients(corrected_physical, second))
    result = {
        "case": row["case"],
        "cells": row["realised_cells"],
        "input_archive": str(INPUT / row["state_archive"]),
        "input_sha256": hashlib.sha256(
            (INPUT / row["state_archive"]).read_bytes()
        ).hexdigest(),
        "map_mismatch": map_score,
        "plasma_response_mismatch": measured,
        "positive_control_relative_tolerance": 1e-9,
        "reconstruction_relative_sup": reconstruction_relative_sup,
        "class_precedence": list(
            ("X-point cell", "wall-cut", "exterior", "interior", "separatrix-cut")
        ),
        "class_summary": class_summary,
        "per_cell": cells,
        "x_point_placement_squared_error_reduction": reduction(error, placement),
        "centroid_control_current_max_change_a": float(
            np.max(abs(corrected_physical[0] - physical[0]))
        ),
        "completed": False,
    }
    write(output / (path.stem + ".json"), result)
    low = quadrature(machine, coordinates, booked, analytic, 4, 16)
    high = quadrature(machine, coordinates, booked, analytic, 6, 32)
    common_correction = error - high[2]
    fixed_exterior_correction = data["plasma"] - high[0]
    result.update(
        {
            "green_common_operator_squared_error_reduction": reduction(
                error, common_correction
            ),
            "green_fixed_exterior_squared_error_reduction": reduction(
                error, fixed_exterior_correction
            ),
            "quadrature_orders": [[4, 16], [6, 32]],
            "quadrature_difference_error_relative_norm": float(
                np.linalg.norm(high[2] - low[2]) / np.linalg.norm(error)
            ),
            "quadrature_booked_difference_error_relative_norm": float(
                np.linalg.norm(high[0] - low[0]) / np.linalg.norm(error)
            ),
            "green_change_error_relative_norm": float(
                np.linalg.norm(common_correction) / np.linalg.norm(error)
            ),
            "weak_rotation_control_pass": bool(map_score["sup_relative"] <= 1e-4)
            if saddle is None
            else None,
            "completed": True,
        }
    )
    np.savez_compressed(
        output / (path.stem + ".npz"),
        class_fields=class_fields,
        error=error,
        reconstructed=reconstructed,
        placement_correction=placement,
        quadrature_low=low,
        quadrature_high=high,
    )
    if row["case"] == "diverted-single-null" and row["realised_cells"] == 132:
        result["mismatch_levels_wb"] = render(row, data, output)
    write(output / (path.stem + ".json"), result)
    print(
        f"ATTRIBUTED {path.stem} "
        f"placement={result['x_point_placement_squared_error_reduction']:.6g} "
        f"green={result['green_common_operator_squared_error_reduction']:.6g}",
        flush=True,
    )
    return result


def summarize(report, output):
    """Write the interpretation from the persisted numerical evidence."""
    assert report["completed"] and len(report["rows"]) == 10
    diverted = [r for r in report["rows"] if r["case"].startswith("diverted")]
    placement = [r["x_point_placement_squared_error_reduction"] for r in diverted]
    green = [r["green_common_operator_squared_error_reduction"] for r in diverted]
    resolved = (
        max(placement) < 0.1
        and max(abs(value) for value in green) < 0.01
        and max(r["quadrature_difference_error_relative_norm"] for r in diverted)
        < 0.001
    )
    report["conclusion"] = (
        "Neither X-point current-centroid placement nor Green evaluation explains "
        "the non-monotone sequence: the archived booked-versus-analytic current "
        "moments reproduce it, with mesh-dependent signed cancellation between "
        "interior, separatrix-cut, X-point, wall-cut and exterior contributions."
        if resolved
        else "The candidate mechanisms remain unresolved; inspect the per-row "
        "squared-error reductions and quadrature convergence."
    )
    report["limitation"] = (
        "Class projection shares are signed and additive, not independent causal "
        "percentages. Quadrature refinement bounds its numerical sensitivity, "
        "not a rigorous integration error. The exact upstream stage that changes "
        "the current moments is not identified by this fixed-current experiment."
    )
    write(output / "report.json", report)
    lines = [
        "# Diverted chord response attribution",
        "",
        report["conclusion"],
        "",
        report["variance_definition"],
        "",
        report["limitation"],
        "",
        "| Case | Cells | Map sup | X centroid reduction | Green reduction "
        "| Quadrature delta/error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in report["rows"]:
        lines.append(
            f"| {r['case']} | {r['cells']} | {r['map_mismatch']['sup_relative']:.8g} "
            f"| {r['x_point_placement_squared_error_reduction']:.8g} "
            f"| {r['green_common_operator_squared_error_reduction']:.8g} "
            f"| {r['quadrature_difference_error_relative_norm']:.8g} |"
        )
    lines += [
        "",
        "Signed class projection shares (sum to one):",
        "",
        "| Cells | Interior | Separatrix-cut | X-point | Wall-cut | Exterior |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in diverted:
        shares = " | ".join(
            f"{row['class_summary'][label]['projection_share']:.8g}"
            for label in CLASSES
        )
        lines.append(f"| {row['cells']} | {shares} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(json.loads((args.output / "report.json").read_text()), args.output)
        return
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    print(f"revision={revision} tree={ROOT} command={sys.argv!r}", flush=True)
    assert jax.default_backend() == "gpu"
    print(f"device={jax.devices()} x64={jax.config.jax_enable_x64}", flush=True)
    report = {
        "revision": revision,
        "worktree": str(ROOT),
        "job": os.environ.get("SLURM_JOB_ID"),
        "completed": False,
        "rows": [],
        "variance_definition": (
            "1 - sum((error-correction)^2)/sum(error^2), "
            "uncentered squared field error; no fitted coefficients"
        ),
        "green_interpretation": (
            "common operator changes both booked and analytic images; "
            "fixed exterior changes booked image only"
        ),
        "class_interpretation": (
            "exclusive classes; X-point then wall-cut precedence; "
            "signed projection shares sum to one"
        ),
    }
    write(args.output / "report.json", report)
    control = args.output / "negative-control.log"
    control.write_text(CONTROL + "\n")
    started = time.monotonic()
    for requested in (110, 300, 500, 1000, 2500):
        for case in ("diverted-single-null", "weak-rotation-reactor-static"):
            path = INPUT / f"{case}-cells-{requested}-chord.json"
            result = measure(path, args.output)
            report["rows"].append(result)
            write(args.output / "report.json", report)
            if case.startswith("weak"):
                with control.open("a") as stream:
                    stream.write(
                        f"cells={result['cells']} "
                        f"mismatch={result['map_mismatch']['sup_relative']:.17g} "
                        f"pass={result['weak_rotation_control_pass']}\n"
                    )
                assert result["weak_rotation_control_pass"], (
                    "weak-rotation control exceeds 1e-4"
                )
    report["completed"] = True
    report["wall_seconds"] = time.monotonic() - started
    summarize(report, args.output)
    print("ATTRIBUTION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
