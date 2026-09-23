"""Separate current integration from support geometry on analytic input states.

Physical moments are evaluated before their conversion to coupling coefficients.
The signed decomposition is true minus booked = geometry minus integration,
where integration is booked minus the analytic integral on the same polygon.
The archived fixture target is retained separately from the true-region integral.
No equilibrium solve is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "docs/figures/plasma-cell-read-fidelity/map-fidelity"
OUTPUT = ROOT / "docs/figures/cut-cell-current-attribution/exact-support-floor"
CASES = ("diverted-single-null", "weak-rotation-reactor-static")
RUNGS = (110, 300, 500, 1000, 2500)
CLASSES = ("interior", "separatrix-cut", "X-point cell", "wall-cut", "exterior")


def write_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(value) for value in item]
        if isinstance(item, np.ndarray):
            return clean(item.tolist())
        if isinstance(item, np.generic):
            return clean(item.item())
        return item

    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def ring_integral(vertices, density, centre, order):
    """Integrate a smooth density with signed fan triangles, including concavity."""
    vertices = np.asarray(vertices)
    if len(vertices) < 3:
        return np.zeros(4)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes, weights = (nodes + 1) / 2, weights / 2
    first = vertices[0]
    a, b = vertices[1:-1] - first, vertices[2:] - first
    determinant = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    points = first + nodes[None, :, None, None] * (
        (1 - nodes[None, None, :, None]) * a[:, None, None, :]
        + nodes[None, None, :, None] * b[:, None, None, :]
    )
    area_weights = (
        determinant[:, None, None]
        * nodes[None, :, None]
        * weights[None, :, None]
        * weights[None, None, :]
    )
    values = density(points[..., 0], points[..., 1]) * area_weights
    offset = points - centre
    return np.array(
        [
            values.sum(),
            (values * offset[..., 0]).sum(),
            (values * offset[..., 1]).sum(),
            area_weights.sum(),
        ]
    )


def polygon_integral(geometry, density, centre, order=12):
    """Integrate polygon components and subtract holes, without convexifying."""
    if geometry.is_empty:
        return np.zeros(4)
    if geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        return sum(
            (polygon_integral(g, density, centre, order) for g in geometry.geoms),
            np.zeros(4),
        )
    if geometry.geom_type != "Polygon":
        return np.zeros(4)
    from shapely.geometry.polygon import orient

    geometry = orient(geometry, sign=1)
    result = ring_integral(geometry.exterior.coords, density, centre, order)
    for ring in geometry.interiors:
        result += ring_integral(ring.coords, density, centre, order)
    np.testing.assert_allclose(result[3], geometry.area, rtol=2e-10, atol=1e-13)
    return result


def instrument_controls():
    from shapely.geometry import Polygon

    concave = Polygon([(1, 0), (3, 0), (3, 1), (2, 1), (2, 2), (1, 2)])
    value = polygon_integral(concave, lambda r, z: np.ones_like(r), np.zeros(2))
    np.testing.assert_allclose(value, [3, 5.5, 2.5, 3], rtol=1e-13)
    holed = Polygon(
        [(1, 0), (4, 0), (4, 3), (1, 3)], holes=[[(2, 1), (3, 1), (3, 2), (2, 2)]]
    )
    np.testing.assert_allclose(
        polygon_integral(holed, lambda r, z: 2 * np.ones_like(r), np.zeros(2)),
        [16, 40, 24, 8],
        rtol=1e-13,
    )
    true, polygon, booked = 10.0, 9.0, 8.0
    assert true - booked == (true - polygon) - (booked - polygon)
    assert (true - 0.0) != 0
    return {
        "concave_constant_density": value.tolist(),
        "hole_subtraction": True,
        "missing_current_positive_control_a": 10.0,
        "decomposition_sign_control": True,
    }


def centroid(moments, centres):
    return (moments[0, :, None] * centres + moments[1:3].T).sum(axis=0) / moments[
        0
    ].sum()


def physical_from_coefficients(coefficients, second):
    result = coefficients.copy()
    result[1] = second[:, 0] * coefficients[1] + second[:, 2] * coefficients[2]
    result[2] = second[:, 2] * coefficients[1] + second[:, 1] * coefficients[2]
    return result


def measure(case_name, rung, output):
    import jax
    import jax.numpy as jnp
    from shapely.geometry import Point, Polygon
    from benchmarks import solovev_certificate as certificate
    from nova.equilibrium.forward_operator import set_support_clip_mode
    from scripts.analytic_oracle_fixtures import measure as fixture

    start = time.monotonic()
    label = f"{case_name}-cells-{rung}"
    print(f"BUILD {label}", flush=True)
    carrier, source, exact = certificate._case(case_name)
    machine = certificate._case_machine(case_name, carrier, exact, -rung)
    operator = fixture.forward_operator(source, machine)
    centres = np.asarray(operator.moment_geometry.atomic_mesh.centroids)
    second = np.asarray(operator.moment_geometry.second_moment)
    mesh = operator.moment_geometry.atomic_mesh
    atomic = [
        Polygon(np.asarray(mesh.node_coordinates)[idx[:n]])
        for idx, n in zip(
            np.asarray(mesh.cell_nodes), np.asarray(mesh.cell_vertex_count), strict=True
        )
    ]
    boundary = Polygon(fixture._analytic_separatrix(exact, points=11521))
    fine_boundary = Polygon(fixture._analytic_separatrix(exact, points=23041))
    assert boundary.is_valid and fine_boundary.is_valid
    wall = Polygon(machine.wall_node)
    density = source.toroidal_current_density
    true = np.stack(
        [
            polygon_integral(cell.intersection(fine_boundary), density, centre)
            for cell, centre in zip(atomic, centres, strict=True)
        ]
    )
    coarse = np.stack(
        [
            polygon_integral(cell.intersection(boundary), density, centre)
            for cell, centre in zip(atomic, centres, strict=True)
        ]
    )
    reference_refinement = float(
        np.sum(np.abs(true[:, 0] - coarse[:, 0])) / true[:, 0].sum()
    )
    assert reference_refinement < 1e-5, reference_refinement
    xpoint = getattr(exact, "x_point", None)
    categories = []
    for cell, integral in zip(atomic, true, strict=True):
        if xpoint is not None and cell.covers(Point(xpoint)):
            category = "X-point cell"
        elif cell.boundary.intersection(wall.boundary.buffer(1e-9)).length > 1e-8:
            category = "wall-cut"
        elif integral[3] > 1e-10 * cell.area and integral[3] < (1 - 1e-8) * cell.area:
            category = "separatrix-cut"
        elif integral[3] >= (1 - 1e-8) * cell.area:
            category = "interior"
        else:
            category = "exterior"
        categories.append(category)
    archived = {}
    arrays = {}
    for mode in ("exact", "chord"):
        path = INPUT / f"{label}-{mode}.json"
        archived[mode] = json.loads(path.read_text())
        with np.load(INPUT / f"{label}-{mode}.npz") as bank:
            arrays[mode] = {key: bank[key] for key in bank.files}
    state = arrays["exact"]["analytic"]
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    np.testing.assert_array_equal(coordinates, arrays["exact"]["coordinates"])
    np.testing.assert_allclose(
        certificate._exact_state(case_name, exact, coordinates),
        state,
        rtol=0,
        atol=1e-12,
    )
    target = archived["exact"]["analytic_plasma_current_a"]
    results = {}
    for mode in ("exact", "chord"):
        print(f"EVALUATE {label} {mode}", flush=True)
        set_support_clip_mode(mode)
        jax.clear_caches()

        @jax.jit
        def evaluate(state, active):
            partition = active._support_partition(state, None)
            masks, topology, samples, support = partition
            moments = active.source.current_moments(
                active._moment_support_masks(masks, support),
                active.support_current_moments,
                support,
                sample_flux=samples,
            )
            return support, moments, active.coupling_current_moments(moments), topology

        support, physical, coefficients, topology = jax.device_get(
            evaluate(jnp.asarray(state), operator)
        )
        physical = np.asarray(physical)
        assert np.isfinite(physical).all()
        np.testing.assert_allclose(
            coefficients, arrays[mode]["moments"], rtol=1e-8, atol=1e-7
        )
        fraction = float(physical[0].sum() / target)
        np.testing.assert_allclose(
            fraction, archived[mode]["unscaled_over_analytic_current"], rtol=1e-9
        )
        expected_physical = physical_from_coefficients(arrays[mode]["moments"], second)
        np.testing.assert_allclose(physical, expected_physical, rtol=1e-8, atol=1e-7)
        polygons = [
            Polygon(v[:n]) if n >= 3 else Polygon()
            for v, n in zip(support.support_vertices, support.vertex_count, strict=True)
        ]
        invalid = [i for i, polygon in enumerate(polygons) if not polygon.is_valid]
        assert not invalid, f"invalid exact polygons: {invalid}"
        same = np.stack(
            [
                polygon_integral(polygon, density, centre)
                for polygon, centre in zip(polygons, centres, strict=True)
            ]
        )
        repeated = np.stack(
            [
                polygon_integral(polygon, density, centre, order=24)
                for polygon, centre in zip(polygons, centres, strict=True)
            ]
        )
        quadrature_error = float(np.sum(np.abs(same[:, 0] - repeated[:, 0])) / target)
        assert quadrature_error < 1e-9, quadrature_error
        integration = physical.T - same[:, :3]
        geometry = true[:, :3] - same[:, :3]
        missing = true[:, :3] - physical.T
        np.testing.assert_allclose(
            missing, geometry - integration, rtol=1e-10, atol=1e-8
        )
        categories_array = np.asarray(categories)
        classes = {}
        for category in CLASSES:
            selected = categories_array == category
            classes[category] = {
                "count": int(selected.sum()),
                "true_current_a": float(true[selected, 0].sum()),
                "booked_current_a": float(physical[0, selected].sum()),
                "same_polygon_analytic_current_a": float(same[selected, 0].sum()),
                "missing_current_a": float(missing[selected, 0].sum()),
                "moment_integration_error_a": float(integration[selected, 0].sum()),
                "support_geometry_error_a": float(geometry[selected, 0].sum()),
                "moment_integration_absolute_error_a": float(
                    np.abs(integration[selected, 0]).sum()
                ),
                "support_geometry_absolute_error_a": float(
                    np.abs(geometry[selected, 0]).sum()
                ),
            }
        fixture_centroid = centroid(arrays[mode]["analytic_physical_moments"], centres)
        measured_centroid = centroid(physical, centres)
        displacement_mm = 1000 * (measured_centroid - fixture_centroid)
        cells = []
        for i, category in enumerate(categories):
            cells.append(
                {
                    "cell": i,
                    "class": category,
                    "centre_rz_m": centres[i],
                    "atomic_vertices_rz_m": np.asarray(atomic[i].exterior.coords),
                    "support_vertices_rz_m": np.asarray(polygons[i].exterior.coords),
                    "support_vertex_count": int(support.vertex_count[i]),
                    "support_boundary": bool(support.boundary[i]),
                    "atomic_area_m2": atomic[i].area,
                    "true_plasma_area_m2": true[i, 3],
                    "support_area_m2": same[i, 3],
                    "booked_moments_a_am_am": physical[:, i],
                    "analytic_same_polygon_moments_a_am_am": same[i, :3],
                    "analytic_true_region_moments_a_am_am": true[i, :3],
                    "moment_integration_error_a_am_am": integration[i],
                    "support_geometry_error_a_am_am": geometry[i],
                    "missing_current_a": missing[i, 0],
                }
            )
        integration_deficit = -float(integration[:, 0].sum())
        geometry_deficit = float(geometry[:, 0].sum())
        dominant = (
            "moment integration"
            if abs(integration_deficit) > abs(geometry_deficit)
            else "support geometry"
        )
        key = (
            "moment_integration_absolute_error_a"
            if dominant == "moment integration"
            else "support_geometry_absolute_error_a"
        )
        where = max(classes, key=lambda k: classes[k][key])
        result = {
            "clip_mode": mode,
            "support_fraction_of_archived_target": fraction,
            "archived_support_fraction": archived[mode][
                "unscaled_over_analytic_current"
            ],
            "booked_current_a": float(physical[0].sum()),
            "true_region_current_a": float(true[:, 0].sum()),
            "archived_target_current_a": target,
            "archived_fixture_current_a": float(
                arrays[mode]["analytic_physical_moments"][0].sum()
            ),
            "archived_target_minus_true_region_a": float(target - true[:, 0].sum()),
            "missing_current_against_archived_target_a": float(
                target - physical[0].sum()
            ),
            "missing_current_against_true_region_a": float(missing[:, 0].sum()),
            "moment_integration_error_a": -integration_deficit,
            "support_geometry_error_a": geometry_deficit,
            "moment_integration_deficit_a": integration_deficit,
            "true_region_centroid_rz_m": centroid(true[:, :3].T, centres),
            "archived_fixture_centroid_rz_m": fixture_centroid,
            "booked_centroid_rz_m": measured_centroid,
            "centroid_displacement_from_fixture_mm": displacement_mm,
            "quadrature_order_doubling_l1_relative": quadrature_error,
            "classes": classes,
            "cells": cells,
            "attribution": (
                f"{dominant} carries the larger signed deficit, "
                f"concentrated in {where} cells."
            ),
            "nulls": archived[mode]["mapped_nulls"],
            "read_axis_rz_m": np.asarray(topology.axis),
        }
        if mode == "exact" and case_name == "diverted-single-null" and rung == 110:
            result["centroid_positive_control_within_1mm"] = bool(
                abs(displacement_mm[1] + 35.4) < 1
            )
            assert result["centroid_positive_control_within_1mm"], displacement_mm
        if mode == "chord":
            deficit = 1 - fraction
            result["negative_control"] = {
                "declaration": "evaluate the same cells in chord clip mode",
                "signed_missing_fraction": deficit,
                "missing_fraction_below_one_per_mille": bool(deficit < 1e-3),
                "absolute_error_fraction": abs(deficit),
                "absolute_error_below_one_per_mille": bool(abs(deficit) < 1e-3),
            }
        results[mode] = result
        print(
            f"REPRODUCED {label} {mode} fraction={fraction:.12g} "
            f"centroid_dz_mm={displacement_mm[1]:.9g} "
            f"integration_deficit_A={integration_deficit:.9g} "
            f"geometry_deficit_A={geometry_deficit:.9g}",
            flush=True,
        )
    row = {
        "case": case_name,
        "requested_cells": rung,
        "realised_cells": len(centres),
        "true_boundary_sampling_points": 23041,
        "boundary_refinement_l1_current_relative": reference_refinement,
        "class_precedence": list(CLASSES[i] for i in (2, 3, 1, 0, 4)),
        "modes": results,
        "seconds": time.monotonic() - start,
        "analytic_axis_rz_m": np.asarray(exact.magnetic_axis),
        "analytic_xpoint_rz_m": None if xpoint is None else np.asarray(xpoint),
        "input_sha256": {
            mode: hashlib.sha256(
                (INPUT / f"{label}-{mode}.npz").read_bytes()
            ).hexdigest()
            for mode in ("exact", "chord")
        },
    }
    write_json(output / f"{label}.json", row)
    return row


def render(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from benchmarks import solovev_certificate as certificate
    from benchmarks.plasma_cell_terminal_state import _draw_nulls
    from nova.media import poloidal
    from nova.media.ink import DEFAULT_INK, poloidal_axes
    from nova.media.sources.frame import WallUnit

    selected = [r for r in rows if r["requested_cells"] == 300]
    maximum = max(
        abs(cell[key][0]) / row["modes"]["exact"]["archived_target_current_a"] * 100
        for row in selected
        for cell in row["modes"]["exact"]["cells"]
        for key in (
            "moment_integration_error_a_am_am",
            "support_geometry_error_a_am_am",
        )
    )
    levels = np.linspace(-maximum, maximum, 13)
    panels = []
    for row in selected:
        case_name = row["case"]
        label = f"{case_name}-cells-300"
        meta = json.loads((INPUT / f"{label}-exact.json").read_text())
        with np.load(INPUT / f"{label}-exact.npz") as data:
            coordinates, wall = data["coordinates"], data["wall"]
            reference, mapped = data["analytic"], data["mapped"]
        units = tuple(
            WallUnit(wall[a:b, 0], wall[a:b, 1], closed=closed, kind=kind)
            for a, b, closed, kind in meta["wall_units"]
        )
        flux_levels = poloidal.contour_levels(
            reference,
            count=12,
            axis=meta["reference_nulls"]["axis_flux_wb"],
            boundary=meta["reference_nulls"]["boundary_flux_wb"],
        )
        analytic_nulls = {
            "axis_rz_m": row["analytic_axis_rz_m"],
            "x_point_rz_m": row["analytic_xpoint_rz_m"],
            "qualified_saddles_rz_m": [],
        }
        blue = DEFAULT_INK.variant(
            axis_color="#3366cc",
            xpoint_color="#3366cc",
            axis_markersize=9,
            xpoint_markersize=11,
        )
        figure, axes = plt.subplots(1, 4, figsize=(16, 6), constrained_layout=True)
        for values, colour in ((reference, "#3366cc"), (mapped, "#a34828")):
            radius, height, raster = certificate._raster_field(
                coordinates, values, wall
            )
            poloidal.draw_flux_contours(
                axes[0],
                radius,
                height,
                raster,
                flux_levels,
                color=colour,
                linewidth=0.7,
            )
        axes[0].set_title("Analytic / exact map\nshared flux levels")
        cells = row["modes"]["exact"]["cells"]
        centres = np.array([c["centre_rz_m"] for c in cells])
        target = row["modes"]["exact"]["archived_target_current_a"]
        quantities = (
            -np.array([c["moment_integration_error_a_am_am"][0] for c in cells]),
            np.array([c["support_geometry_error_a_am_am"][0] for c in cells]),
            np.array([c["missing_current_a"] for c in cells]),
        )
        titles = (
            "Integration deficit",
            "Support geometry deficit",
            "Total true-region deficit",
        )
        radial_grid, vertical_grid = np.meshgrid(radius, height)
        for axis, quantity, title in zip(axes[1:], quantities, titles, strict=True):
            field = griddata(
                centres,
                quantity / target * 100,
                (radial_grid, vertical_grid),
                method="linear",
            )
            poloidal.draw_flux_contours(
                axis, radius, height, field, levels, color="#a34828", linewidth=0.8
            )
            axis.set_title(title + "\n% target current per cell")
        for axis in axes:
            poloidal.draw_wall(axis, units=units)
            _draw_nulls(axis, analytic_nulls, blue, hollow=True)
            _draw_nulls(axis, meta["mapped_nulls"], DEFAULT_INK, hollow=False)
            poloidal_axes(axis)
        figure.suptitle(
            f"{case_name}: 300 requested cells; evaluation only, no solve\n"
            "Blue hollow nulls: analytic; solid nulls: archived exact map; "
            "current contours interpolate cell totals\n"
            "Shared error levels (% target/cell): "
            + ", ".join(f"{x:.3g}" for x in levels),
            fontsize=10,
        )
        path = output / f"{case_name}-cells-300.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        panels.append(
            {
                "path": path.name,
                "current_levels_percent_of_target_per_cell": levels.tolist(),
                "flux_levels_wb": np.asarray(flux_levels).tolist(),
            }
        )
    return panels


def summarize(rows, output, controls, panels):
    report = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "worktree": str(ROOT),
        "command": sys.argv,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "cases": CASES,
        "rungs": RUNGS,
        "completed": len(rows) == len(CASES) * len(RUNGS),
        "instrument_controls": controls,
        "panels": panels,
        "sign_convention": (
            "true minus booked = support_geometry_error minus moment_integration_error"
        ),
        "target_caveat": (
            "The archived diverted target integrates density on traced "
            "fixture polygons. "
            "It is not an independent true-plasma-region integral."
        ),
        "rows": rows,
    }
    write_json(output / "report.json", report)
    lines = [
        "# Exact support current attribution",
        "",
        report["sign_convention"],
        "",
        report["target_caveat"],
        "",
        "| Case | Cells | Exact fraction | Vertical shift mm | Missing A (true region) "
        "| Integration deficit A | Geometry deficit A | Chord fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        e, c = row["modes"]["exact"], row["modes"]["chord"]
        lines.append(
            f"| {row['case']} | {row['requested_cells']} "
            f"| {e['support_fraction_of_archived_target']:.9f} "
            f"| {e['centroid_displacement_from_fixture_mm'][1]:.5f} "
            f"| {e['missing_current_against_true_region_a']:.8g} "
            f"| {e['moment_integration_deficit_a']:.8g} "
            f"| {e['support_geometry_error_a']:.8g} "
            f"| {c['support_fraction_of_archived_target']:.9f} |"
        )
    for row in rows:
        e = row["modes"]["exact"]
        lines += [
            "",
            f"## {row['case']}, {row['requested_cells']} cells",
            "",
            e["attribution"],
            "",
            "| Cell class | Count | Missing A | Moment integration error A "
            "| Geometry error A |",
            "|---|---:|---:|---:|---:|",
        ]
        for key, value in e["classes"].items():
            lines.append(
                f"| {key} | {value['count']} | {value['missing_current_a']:.9g} "
                f"| {value['moment_integration_error_a']:.9g} "
                f"| {value['support_geometry_error_a']:.9g} |"
            )
        lines += [
            "",
            "Boundary refinement L1 / true current: "
            f"{row['boundary_refinement_l1_current_relative']:.3g}; "
            "quadrature order doubling L1 / target: "
            f"{e['quadrature_order_doubling_l1_relative']:.3g}.",
        ]
    lines += [
        "",
        "Chord control uses the signed missing fraction and also reports the absolute "
        "current error. A current excess can pass the missing-current criterion while "
        "failing absolute agreement; both outcomes remain explicit.",
        "",
        "Every cell and all three physical moments are retained in report.json and "
        "the individual row JSON files. No solver step or production repair is made.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    import jax
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    print(
        f"revision={revision} tree={ROOT} command={sys.argv!r}",
        flush=True,
    )
    controls = instrument_controls()
    rows = []
    if args.render_only:
        rows = [
            json.loads((args.output / f"{case}-cells-{rung}.json").read_text())
            for case in CASES
            for rung in RUNGS
        ]
    else:
        assert jax.default_backend() == "gpu", jax.devices()
        print(f"devices={jax.devices()} x64={jax.config.jax_enable_x64}", flush=True)
        for case in CASES:
            for rung in RUNGS:
                rows.append(measure(case, rung, args.output))
                summarize(rows, args.output, controls, [])
    panels = render(rows, args.output)
    report = summarize(rows, args.output, controls, panels)
    assert report["completed"]
    assert all(
        row["modes"]["chord"]["negative_control"][
            "missing_fraction_below_one_per_mille"
        ]
        for row in rows
    )
    print(
        "ATTRIBUTION_COMPLETE: all rows, reproductions, numerical controls "
        "and panels persisted",
        flush=True,
    )


if __name__ == "__main__":
    main()
