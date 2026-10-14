"""Measure DINA prescribed current and plasma flux against stored diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imas
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline

from nova.biot.greens import hybrid_greens
from nova.biot.polygon import polygon_greens
from nova.imas.machine import CrossSection
from nova.imas.prescribed_plasma import plasma_flux_at_points
from nova.imas.profiles import prescribe_current_density


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "docs/figures/iter-prescribed-plasma"
OUTPUT = Path(
    "/home/ITER/mcintos/Code/nova/.cache/iter-passive-plasma-circuit/"
    "gate-logs/iter-prescribed-plasma"
)
URI = "imas:hdf5?user=mcintos;pulse=105014;run=1;database=iter;version=3"
DD_VERSION = "3.38.1"


def geometry_flux(target_r, target_z, element):
    """Flux [Wb/A] from one finite IMAS conductor section."""
    shape = CrossSection(element.geometry).data.poly
    vertices = np.asarray(shape.exterior.coords)[:-1]
    if int(element.geometry.geometry_type) == 2:
        centroid = shape.centroid
        extent = np.ptp(vertices, axis=0)
        return hybrid_greens(
            target_r,
            target_z,
            centroid.x,
            centroid.y,
            extent[0],
            extent[1],
        )[0]
    return polygon_greens(target_r, target_z, vertices)[0]


def conductor_sources(active, passive, target_r, target_z):
    """Precompute finite-section flux signatures of scenario conductors."""
    columns = []
    owners = []
    for category, groups in (("coil", active.coil), ("passive", passive.loop)):
        for group_index, group in enumerate(groups):
            for element in group.element:
                turns = float(element.turns_with_sign)
                columns.append(turns * geometry_flux(target_r, target_z, element))
                owners.append((category, group_index))
    return np.stack(columns, axis=1), owners


def current_vector(active, passive, owners, index):
    """Scenario current of each finite source section at an equilibrium index."""
    return np.array(
        [
            float(
                np.asarray(
                    (active.coil if category == "coil" else passive.loop)[
                        group
                    ].current.data
                )[index]
            )
            for category, group in owners
        ]
    )


def prescribe_slice(slice_):
    """Prescribe one native-DD equilibrium slice from its stored profiles."""
    global_ = slice_.global_quantities
    profile = slice_.profiles_1d
    map_ = slice_.profiles_2d[0]
    outline = slice_.boundary_separatrix.outline
    return prescribe_current_density(
        np.asarray(map_.grid.dim1),
        np.asarray(map_.grid.dim2),
        np.asarray(map_.psi),
        np.asarray(profile.psi),
        np.asarray(profile.dpressure_dpsi),
        np.asarray(profile.f_df_dpsi),
        float(global_.psi_axis),
        float(global_.psi_boundary),
        float(global_.ip),
        np.c_[np.asarray(outline.r), np.asarray(outline.z)],
        (float(global_.magnetic_axis.r), float(global_.magnetic_axis.z)),
    )


def loop_locations(magnetics):
    """One radius and height per scenario flux-loop diagnostic."""
    return np.array(
        [
            (float(loop.position[0].r), float(loop.position[0].z))
            for loop in magnetics.flux_loop
        ]
    )


def coil_locations(active):
    """Coil section centroids, with one target per stored element."""
    locations = []
    names = []
    for coil in active.coil:
        for element in coil.element:
            centroid = CrossSection(element.geometry).data.poly.centroid
            locations.append((centroid.x, centroid.y))
            names.append(str(coil.name).strip())
    return np.asarray(locations), names


def map_flux(slice_, locations):
    """Stored 2-D total flux at locations inside the map's supported domain."""
    map_ = slice_.profiles_2d[0]
    radius = np.asarray(map_.grid.dim1)
    height = np.asarray(map_.grid.dim2)
    values = np.full(len(locations), np.nan)
    inside = (
        (locations[:, 0] >= radius[0])
        & (locations[:, 0] <= radius[-1])
        & (locations[:, 1] >= height[0])
        & (locations[:, 1] <= height[-1])
    )
    spline = RectBivariateSpline(radius, height, np.asarray(map_.psi))
    values[inside] = spline.ev(locations[inside, 0], locations[inside, 1])
    return values


def metric(values):
    """Finite distribution summary with explicit count and quantiles."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return {
        "count": int(values.size),
        "minimum": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
    }


def style_axes(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(False)
    axis.tick_params(labelsize=16)
    axis.xaxis.label.set_size(18)
    axis.yaxis.label.set_size(18)


def save_svg(figure, name):
    """Write stable SVG text without renderer-added trailing spaces."""
    path = FIGURES / name
    figure.savefig(path, metadata={"Date": None})
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
    )


def draw_figures(rows, representatives, linkage, names):
    """Save spatial, time-history and residual figures as stand-alone SVGs."""
    matplotlib.rcParams["svg.hashsalt"] = "plasma-current"
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(representatives), figsize=(14, 5))
    for axis, (index, current, slice_) in zip(axes, representatives):
        rr, zz = np.meshgrid(current.radius, current.height, indexing="ij")
        levels = -np.array([0.2, 0.4, 0.6, 0.8]) * abs(current.current_density.min())
        axis.contour(
            rr,
            zz,
            current.current_density,
            levels=np.sort(levels),
            colors="#315f78",
            linewidths=1.5,
        )
        axis.plot(
            current.boundary[:, 0], current.boundary[:, 1], color="#a06635", lw=2.2
        )
        outline = slice_.boundary_separatrix.outline
        if len(outline.r):
            axis.plot(
                np.asarray(outline.r),
                np.asarray(outline.z),
                color="#777777",
                lw=1.1,
                ls="--",
            )
        axis.plot(*current.centroid, marker="o", color="#315f78", ms=6)
        axis.plot(
            float(slice_.global_quantities.current_centre.r),
            float(slice_.global_quantities.current_centre.z),
            marker="x",
            color="#222222",
            ms=7,
        )
        axis.set_aspect("equal")
        axis.set_axis_off()
        axis.text(
            0.04,
            0.96,
            f"t = {float(slice_.time):.1f} s",
            transform=axis.transAxes,
            va="top",
            fontsize=16,
        )
    fig.tight_layout()
    save_svg(fig, "current-boundary.svg")
    plt.close(fig)

    time = np.array([row["time"] for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(time, [row["scale_factor"] for row in rows], color="#315f78", lw=3)
    axes[0].set_ylabel("current scale")
    axes[1].plot(
        time,
        [row["centroid_error_r"] for row in rows],
        color="#315f78",
        lw=2.6,
        label="radial",
    )
    axes[1].plot(
        time,
        [row["centroid_error_z"] for row in rows],
        color="#a06635",
        lw=2.6,
        label="vertical",
    )
    axes[1].axhline(0.09375, color="#777777", lw=1.2, ls=":")
    axes[1].axhline(-0.09375, color="#777777", lw=1.2, ls=":")
    axes[1].set_ylabel("centroid error [m]")
    axes[1].set_xlabel("scenario time [s]")
    axes[1].legend(frameon=False, fontsize=15)
    for axis in axes:
        style_axes(axis)
    fig.tight_layout()
    save_svg(fig, "scale-centroid.svg")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(14, 5))
    for category, colour in (("coil", "#315f78"), ("loop", "#a06635")):
        selected = [row for row in linkage if row["category"] == category]
        if selected:
            axis.scatter(
                [row["time"] for row in selected],
                [row["relative_residual"] for row in selected],
                s=18,
                color=colour,
                label=category,
            )
    axis.axhline(0, color="#777777", lw=1.2)
    axis.set_ylabel("plasma flux residual / |reference|")
    axis.set_xlabel("scenario time [s]")
    axis.legend(frameon=False, fontsize=15)
    style_axes(axis)
    fig.tight_layout()
    save_svg(fig, "linked-flux-residual.svg")
    plt.close(fig)


def main():
    """Run the current census and independent linked-flux comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coil-flux-tolerance", type=float, default=0.025)
    parser.add_argument("--loop-flux-tolerance", type=float, default=0.005)
    parser.add_argument("--active-grid-tolerance", type=float, default=0.005)
    limits = parser.parse_args()
    if (
        min(
            limits.coil_flux_tolerance,
            limits.loop_flux_tolerance,
            limits.active_grid_tolerance,
        )
        <= 0
    ):
        parser.error("flux tolerances must be positive")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with imas.DBEntry(URI, "r", dd_version=DD_VERSION) as entry:
        equilibrium = entry.get("equilibrium", lazy=False, autoconvert=False)
        active = entry.get("pf_active", lazy=False, autoconvert=False)
        passive = entry.get("pf_passive", lazy=False, autoconvert=False)
        magnetics = entry.get("magnetics", lazy=False, autoconvert=False)
        coupling = entry.get("em_coupling", lazy=False, autoconvert=False)
    loop_points = loop_locations(magnetics)
    coil_points, coil_names = coil_locations(active)
    locations = np.vstack((loop_points, coil_points))
    kernel, owners = conductor_sources(
        active, passive, locations[:, 0], locations[:, 1]
    )
    grid_to_loops = np.asarray(coupling.mutual_loops_grid)
    grid_to_coils = np.asarray(coupling.mutual_grid_active)
    if grid_to_loops.shape != (len(loop_points), 8385):
        raise ValueError("DINA loop-grid mutual shape does not match this grid")
    if grid_to_coils.shape != (8385, 15):
        raise ValueError("DINA active-grid mutual shape does not match this grid")
    print(
        f"REFERENCE_CONTROL loops={len(loop_points)} coils={len(coil_points)} "
        f"sources={len(owners)} kernel_nonzero={np.count_nonzero(kernel)}"
    )
    eligible = [
        index
        for index, slice_ in enumerate(equilibrium.time_slice)
        if abs(float(slice_.global_quantities.ip)) >= 1e5
    ]
    comparison = set(np.linspace(eligible[0], eligible[-1], 24, dtype=int))
    representatives = set(np.linspace(eligible[0], eligible[-1], 3, dtype=int))
    rows = []
    linkage = []
    grid_columns = []
    subtraction = []
    selected = []
    failures = []
    map_control = []
    vertical_control = None
    for index in eligible:
        slice_ = equilibrium.time_slice[index]
        try:
            current = prescribe_slice(slice_)
        except ValueError as error:
            failures.append((index, str(error)))
            continue
        global_ = slice_.global_quantities
        centre = (float(global_.current_centre.r), float(global_.current_centre.z))
        rows.append(
            {
                "index": index,
                "time": float(slice_.time),
                "ip": current.plasma_current,
                "scale_factor": current.scale_factor,
                "boundary_source": current.boundary_source,
                "endpoint_gap": current.endpoint_gap,
                "grid_spacing_m": float(
                    min(
                        np.min(np.diff(current.radius)), np.min(np.diff(current.height))
                    )
                ),
                "centroid_error_r": current.centroid[0] - centre[0],
                "centroid_error_z": current.centroid[1] - centre[1],
            }
        )
        if index in representatives:
            selected.append((index, current, slice_))
        if index == 200:
            target_r = np.array([loop_points[9, 0]])
            target_z = np.array([loop_points[9, 1]])
            displacement = 0.01
            probe = 0.001
            base = plasma_flux_at_points(current, target_r, target_z)[0]
            moved = plasma_flux_at_points(
                current, target_r, target_z, vertical_offset=displacement
            )[0]
            gradient = (
                plasma_flux_at_points(current, target_r, target_z + probe)[0]
                - plasma_flux_at_points(current, target_r, target_z - probe)[0]
            ) / (2 * probe)
            predicted_change = -displacement * gradient
            observed_change = moved - base
            relative_error = abs(observed_change - predicted_change) / abs(
                predicted_change
            )
            if abs(predicted_change) < 1e-4 or relative_error > 0.03:
                raise AssertionError(
                    "vertical-shift flux response differs from gradient"
                )
            vertical_control = {
                "index": index,
                "target": [float(target_r[0]), float(target_z[0])],
                "displacement_m": displacement,
                "predicted_change_wb": float(predicted_change),
                "observed_change_wb": float(observed_change),
                "relative_error": float(relative_error),
            }
            print("NEGATIVE_CONTROL " + json.dumps(vertical_control, sort_keys=True))
        if index not in comparison:
            continue
        stored = map_flux(slice_, locations)
        source = kernel @ current_vector(active, passive, owners, index)
        inferred = stored - source
        predicted = plasma_flux_at_points(current, locations[:, 0], locations[:, 1])
        density = np.asarray(slice_.profiles_2d[0].j_tor)
        cell_area = float(
            np.mean(np.diff(current.radius)) * np.mean(np.diff(current.height))
        )
        grid_current = (density * cell_area).ravel()
        loop_reference = grid_to_loops @ grid_current
        coil_reference = grid_to_coils.T @ grid_current
        prescribed_grid_flux = grid_to_coils.T @ current.cell_current.ravel()
        for column, (value, reference) in enumerate(
            zip(prescribed_grid_flux, coil_reference)
        ):
            grid_columns.append(
                {
                    "index": index,
                    "column": column,
                    "reference": float(reference),
                    "predicted": float(value),
                    "relative_residual": float(
                        (value - reference) / max(abs(reference), 1.0)
                    ),
                }
            )
        # DINA's central-solenoid circuit joins two stored sections in one column.
        predicted_coils = np.r_[
            predicted[len(loop_points) : len(loop_points) + 2],
            predicted[len(loop_points) + 2 : len(loop_points) + 4].sum(),
            predicted[len(loop_points) + 4 : len(loop_points) + 12],
        ]
        for category, names, model, reference_values in (
            (
                "loop",
                [str(loop.name).strip() for loop in magnetics.flux_loop],
                predicted[: len(loop_points)],
                loop_reference,
            ),
            (
                "coil",
                [str(coil.name).strip() for coil in list(active.coil)[:11]],
                predicted_coils,
                coil_reference[:11],
            ),
        ):
            for name, value, reference in zip(names, model, reference_values):
                residual = float(value - reference)
                linkage.append(
                    {
                        "index": index,
                        "time": float(slice_.time),
                        "category": category,
                        "name": name,
                        "reference": float(reference),
                        "predicted": float(value),
                        "residual": residual,
                        "relative_residual": residual / max(abs(reference), 1.0),
                    }
                )
        for point_index in range(len(locations)):
            if not np.isfinite(inferred[point_index]):
                continue
            reference = float(inferred[point_index])
            subtraction.append(
                {
                    "index": index,
                    "category": "loop" if point_index < len(loop_points) else "coil",
                    "relative_residual": float(
                        (predicted[point_index] - reference) / max(abs(reference), 1.0)
                    ),
                }
            )
            if point_index < len(loop_points):
                loop = magnetics.flux_loop[point_index]
                measured = float(np.asarray(loop.flux.data)[index])
                map_control.append(stored[point_index] - measured)
    scale = metric([row["scale_factor"] for row in rows])
    centroid = metric(
        [
            max(abs(row["centroid_error_r"]), abs(row["centroid_error_z"]))
            for row in rows
        ]
    )
    residuals = {
        category: metric(
            [
                abs(row["relative_residual"])
                for row in linkage
                if row["category"] == category
            ]
        )
        for category in ("coil", "loop")
    }
    summary = {
        "case": "ITER 105014/1",
        "dd_version": DD_VERSION,
        "eligible_slices": len(eligible),
        "prescribed_slices": len(rows),
        "failures": failures,
        "scale_factor": scale,
        "centroid_max_coordinate_error_m": centroid,
        "fallback_slices": sum(row["boundary_source"] != "separatrix" for row in rows),
        "grid_spacing_m": min(row["grid_spacing_m"] for row in rows),
        "linked_flux_relative_error": residuals,
        "all_active_grid_mutual_relative_error": metric(
            [abs(row["relative_residual"]) for row in grid_columns]
        ),
        "remaining_active_grid_columns_relative_error": metric(
            [
                abs(row["relative_residual"])
                for row in grid_columns
                if row["column"] >= 11
            ]
        ),
        "geometric_map_subtraction_relative_error": {
            category: metric(
                [
                    abs(row["relative_residual"])
                    for row in subtraction
                    if row["category"] == category
                ]
            )
            for category in ("coil", "loop")
        },
        "map_minus_magnetics_flux_loop_wb": metric(map_control),
        "independent_coil_families": [
            str(coil.name).strip() for coil in list(active.coil)[:11]
        ],
        "geometry_unmapped_coil_family": "VS3: 4 coupling columns, 2 stored elements",
        "vertical_shift_control": vertical_control,
        "tolerances": {
            "centroid_grid_cells": 1.0,
            "coil_flux_relative": limits.coil_flux_tolerance,
            "loop_flux_relative": limits.loop_flux_tolerance,
            "active_grid_relative": limits.active_grid_tolerance,
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "slices": rows,
                "linkage": linkage,
                "active_grid_columns": grid_columns,
                "geometric_map_subtraction": subtraction,
            },
            indent=2,
        )
    )
    print("SUMMARY " + json.dumps(summary, sort_keys=True))
    if failures or len(rows) != len(eligible):
        raise AssertionError(f"prescription failed on {len(failures)} eligible slices")
    if vertical_control is None:
        raise AssertionError("vertical-shift control was not sampled")
    if centroid["count"] != len(rows):
        raise AssertionError("centroid measurement did not cover every slice")
    if centroid["maximum"] > summary["grid_spacing_m"]:
        raise AssertionError(
            f"centroid error {centroid['maximum']:.6g} m exceeds one grid cell"
        )
    for label, observed, tolerance in (
        ("coil flux", residuals["coil"], limits.coil_flux_tolerance),
        ("flux-loop flux", residuals["loop"], limits.loop_flux_tolerance),
        (
            "active-grid flux",
            summary["all_active_grid_mutual_relative_error"],
            limits.active_grid_tolerance,
        ),
    ):
        if observed["maximum"] > tolerance:
            raise AssertionError(
                f"{label} residual {observed['maximum']:.6g} exceeds {tolerance:.6g}"
            )
    print("GATE_PASS centroid, coil flux, flux-loop flux, active-grid flux")
    draw_figures(rows, selected, linkage, coil_names)


if __name__ == "__main__":
    main()
