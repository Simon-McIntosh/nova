"""Render the prescribed-input DIII-D diverted solve beside its labelled map.

The named frame is rebuilt through the landed current-eliminated arm in
``benchmarks.diiid_current_pinned_forward``.  Nova's equilibrium package is a
read-only dependency here: this module selects the one published input, calls
the existing solve, validates its recorded convergence evidence, and draws the
result.  The labelled equilibrium is never used as a fit target.

Both panels use one physical contour-level array computed once from the full
65 by 65 given map in Wb per radian.  The raw panel retains the additive gauge;
the second adds the single mean core offset to Nova's map.  Both maps remain
unfilled line contours, so the comparison cannot be hidden by independent
colour scaling.

The machine ink reproduces values read from imas-ink's ``InkStyle`` through
``benchmarks.diiid_poloidal_figures``: wall ``#000000`` at 1.0 pt, shipped
coil edges ``#888888`` with no fill at 0.4 pt, and the five netCDF-only groups
in the separatrix ``#cc0000`` at 1.5 pt.  Nova flux is ``#3366cc`` and the
given labelled flux is ``#cc0000``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.patches import Rectangle

from benchmarks import diiid_current_pinned_forward as pinned
from benchmarks.diiid_corpus_conventions import nova_total_flux_to_corpus
from benchmarks.diiid_diverted_root_full_currents import (
    POLARITY_AFFECTED_SHOT_COUNT,
    _omitted_vertices,
    append_recovered_conductors,
    current_arms,
)
from benchmarks.diiid_forward_gs_match import (
    DEFAULT_DATA,
    REGISTERED_GRID_STRIDE,
    _CURRENT_COLUMNS,
    _GEOMETRY_COLUMNS,
    _LABEL_COLUMNS,
    _plasma_mask,
    _read,
    _separatrix,
    build_profile,
    canonical_axes,
    gauge_metrics,
)
from benchmarks.diiid_poloidal_figures import (
    STYLE,
    StaticGeometry,
    read_static_geometry,
)
from benchmarks.diiid_state_of_play_figures import (
    _boundary_separation,
    boundary_gradient_minimum,
)
from nova.equilibrium.topology import TopologyClass
from nova.imas.diiid_description import POLOIDAL_CONDUCTORS
from nova.jax.config import configure_dtypes


DEFAULT_OUTPUT = Path("docs/figures/diiid-forward-onboarding/diverted-solve")
FIGURE_NAME = "diverted_solve_overlay.png"
RECEIPT_NAME = "diverted_solve_overlay_receipt.json"
FRAME_SHOT = "d3d_shot_00000c4a7b.parquet"
FRAME_INDEX = 102
EXPECTED_TIME_MS = 2200.0
EXPECTED_RELATIVE_RESIDUAL = 1.5899788903681545e-9
EXPECTED_ITERATIONS = 4
RESIDUAL_REPRODUCTION_RELATIVE_TOLERANCE = 5.0e-4
RESIDUAL_REPRODUCTION_ABSOLUTE_TOLERANCE = 1.0e-12
CURRENT_RELATIVE_ERROR_TOLERANCE = 1.0e-12
LABEL_REPRESENTABILITY_CEILING = 0.0429
CONTOUR_COUNT = 15


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _closed(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=float)
    if np.array_equal(vertices[0], vertices[-1]):
        return vertices
    return np.vstack((vertices, vertices[0]))


def contour_levels_from_given(
    given_wb_per_radian: np.ndarray, count: int = CONTOUR_COUNT
) -> np.ndarray:
    """Return the sole physical contour-level array used in both panels."""

    finite = np.asarray(given_wb_per_radian, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("the given labelled map has no finite contour source")
    lower, upper = np.quantile(finite, [0.01, 0.99])
    if np.isclose(lower, upper):
        lower, upper = float(np.min(finite)), float(np.max(finite))
    if np.isclose(lower, upper):
        raise ValueError("the given labelled map has no finite contour span")
    return np.linspace(float(lower), float(upper), count)


def gauge_match(
    given_total_wb: np.ndarray,
    solved_total_wb: np.ndarray,
    interior: np.ndarray,
) -> dict[str, Any]:
    """Align only the additive gauge and retain the label diagnostic."""

    r_squared, fractional_rms, correction, aligned = gauge_metrics(
        given_total_wb, solved_total_wb, interior
    )
    correction_per_radian = float(nova_total_flux_to_corpus(correction))
    return {
        "aligned_total_wb": aligned,
        "additive_correction_wb_per_radian": correction_per_radian,
        "removed_solve_minus_label_offset_wb_per_radian": -correction_per_radian,
        "fractional_rms": fractional_rms,
        "r_squared": r_squared,
    }


def classify_conductors(geometry: StaticGeometry) -> dict[str, Any]:
    """Partition netCDF coil groups by their presence in released inputs."""

    shipped_names = set(POLOIDAL_CONDUCTORS)
    shipped = tuple(coil for coil in geometry.coils if coil.name in shipped_names)
    netcdf_only = tuple(
        coil for coil in geometry.coils if coil.name not in shipped_names
    )
    return {
        "shipped": shipped,
        "netcdf_only": netcdf_only,
        "shipped_elements": sum(len(coil.elements) for coil in shipped),
        "netcdf_only_elements": sum(len(coil.elements) for coil in netcdf_only),
    }


def _draw_machine(
    axis: Axes,
    geometry: StaticGeometry,
    full_radius: np.ndarray,
    full_height: np.ndarray,
) -> None:
    groups = classify_conductors(geometry)
    for coil in groups["shipped"]:
        for element in coil.elements:
            axis.add_patch(
                PolygonPatch(
                    element,
                    closed=True,
                    fill=False,
                    facecolor="none",
                    edgecolor=STYLE.coil_edgecolor,
                    linewidth=STYLE.coil_linewidth,
                    zorder=4,
                )
            )
    for coil in groups["netcdf_only"]:
        for element in coil.elements:
            axis.add_patch(
                PolygonPatch(
                    element,
                    closed=True,
                    fill=False,
                    facecolor="none",
                    edgecolor=STYLE.separatrix_color,
                    linewidth=STYLE.separatrix_linewidth,
                    linestyle="--",
                    zorder=4,
                )
            )
    limiter = _closed(geometry.limiter)
    axis.plot(
        limiter[:, 0],
        limiter[:, 1],
        color=STYLE.wall_color,
        linewidth=STYLE.wall_linewidth,
        zorder=7,
    )
    axis.add_patch(
        Rectangle(
            (float(full_radius[0]), float(full_height[0])),
            float(np.ptp(full_radius)),
            float(np.ptp(full_height)),
            fill=False,
            edgecolor=STYLE.contour_color,
            linewidth=STYLE.contour_linewidth,
            linestyle=":",
            zorder=3,
        )
    )


def _draw_flux_pair(
    axis: Axes,
    full_radius: np.ndarray,
    full_height: np.ndarray,
    given_wb_per_radian_zr: np.ndarray,
    solve_radius: np.ndarray,
    solve_height: np.ndarray,
    solved_wb_per_radian_rz: np.ndarray,
    levels: np.ndarray,
) -> None:
    """Draw two unfilled line-contour maps on the identical level object."""

    axis.contour(
        full_radius,
        full_height,
        given_wb_per_radian_zr,
        levels=levels,
        colors=STYLE.separatrix_color,
        linewidths=0.55,
        linestyles="dashed",
        zorder=1,
    )
    axis.contour(
        solve_radius,
        solve_height,
        solved_wb_per_radian_rz.T,
        levels=levels,
        colors=STYLE.flux_color,
        linewidths=STYLE.flux_linewidth,
        linestyles="solid",
        zorder=2,
    )


def _draw_topology(
    axis: Axes,
    given_boundary: np.ndarray,
    solved_boundary: np.ndarray,
    given_x_point: np.ndarray,
    solved_x_point: np.ndarray,
) -> None:
    given = _closed(given_boundary)
    solved = _closed(solved_boundary)
    axis.plot(
        given[:, 0],
        given[:, 1],
        color=STYLE.separatrix_color,
        linewidth=STYLE.separatrix_linewidth,
        zorder=8,
    )
    axis.plot(
        solved[:, 0],
        solved[:, 1],
        color=STYLE.flux_color,
        linewidth=STYLE.separatrix_linewidth,
        zorder=8,
    )
    axis.plot(
        *given_x_point,
        marker="x",
        color=STYLE.separatrix_color,
        markersize=8,
        markeredgewidth=1.5,
        linestyle="none",
        zorder=10,
    )
    axis.plot(
        *solved_x_point,
        marker="o",
        markerfacecolor="none",
        color=STYLE.flux_color,
        markersize=7,
        markeredgewidth=1.5,
        linestyle="none",
        zorder=10,
    )


def _format_axis(axis: Axes, title: str) -> None:
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("R [m]")
    axis.set_ylabel("Z [m]")
    axis.set_title(title)


def render_figure(fields: dict[str, Any], geometry: StaticGeometry, path: Path) -> None:
    """Write the raw and gauge-separated physical R-Z comparison."""

    figure, axes = plt.subplots(
        1, 2, figsize=(15.5, 7.6), sharex=True, sharey=True, constrained_layout=True
    )
    comparison = (
        (
            fields["solved_raw_wb_per_radian_rz"],
            "Raw same-level pair",
            "additive gauge retained",
        ),
        (
            fields["solved_aligned_wb_per_radian_rz"],
            "Gauge-matched same-level pair",
            (
                "removed solve−label offset\n"
                f"{fields['removed_offset_wb_per_radian']:+.6e} Wb/rad"
            ),
        ),
    )
    for axis, (solved, title, gauge_text) in zip(axes, comparison, strict=True):
        _draw_machine(axis, geometry, fields["full_radius"], fields["full_height"])
        _draw_flux_pair(
            axis,
            fields["full_radius"],
            fields["full_height"],
            fields["given_full_wb_per_radian_zr"],
            fields["solve_radius"],
            fields["solve_height"],
            solved,
            fields["levels_wb_per_radian"],
        )
        _draw_topology(
            axis,
            fields["given_boundary_rz_m"],
            fields["solved_boundary_rz_m"],
            fields["given_x_point_rz_m"],
            fields["solved_x_point_rz_m"],
        )
        axis.text(
            0.02,
            0.02,
            gauge_text,
            transform=axis.transAxes,
            fontsize=8,
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            zorder=12,
        )
        axis.text(
            0.02,
            0.98,
            (
                f"LCFS Δsym = {fields['boundary_separation_m']:.4f} m\n"
                f"X-point Δ = {fields['x_point_separation_m']:.4f} m"
            ),
            transform=axis.transAxes,
            fontsize=8,
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            zorder=12,
        )
        _format_axis(axis, title)

    handles = [
        Line2D([], [], color=STYLE.flux_color, lw=STYLE.flux_linewidth),
        Line2D([], [], color=STYLE.separatrix_color, lw=0.55, ls="--"),
        Line2D([], [], color=STYLE.flux_color, lw=STYLE.separatrix_linewidth),
        Line2D([], [], color=STYLE.separatrix_color, lw=STYLE.separatrix_linewidth),
        Line2D([], [], color=STYLE.wall_color, lw=STYLE.wall_linewidth),
        Line2D([], [], color=STYLE.coil_edgecolor, lw=STYLE.coil_linewidth),
        Line2D(
            [], [], color=STYLE.separatrix_color, lw=STYLE.separatrix_linewidth, ls="--"
        ),
    ]
    labels = [
        "Nova converged solve contours",
        "given labelled-map contours",
        "Nova boundary / ○ X point",
        "given boundary / × derived X point",
        "82-vertex limiter",
        "19 shipped conductor outlines",
        "5 netCDF-only conductor groups",
    ]
    axes[1].legend(handles, labels, loc="upper right", fontsize=7.5)
    figure.suptitle(
        f"DIII-D prescribed-input diverted solve — {FRAME_SHOT}, frame {FRAME_INDEX}",
        fontsize=12,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=STYLE.figure_dpi)
    plt.close(figure)


def _named_input() -> tuple[Any, dict[str, Any]]:
    polarity = json.loads(pinned.POLARITY_RECEIPT.read_text())["full_corpus_census"]
    affected = set(polarity["affected_shots"])
    if len(affected) != POLARITY_AFFECTED_SHOT_COUNT:
        raise RuntimeError("polarity authority is not the landed 603-shot population")
    selected, _low = pinned._recovery_inputs(affected)
    matches = [
        item
        for item in selected
        if item.shot == FRAME_SHOT and item.frame == FRAME_INDEX
    ]
    if len(matches) != 1:
        raise RuntimeError("the named current-pinned input is not unique")
    declarations = [
        item
        for item in pinned.REPRESENTATIVE_COHORT
        if item["shot"] == FRAME_SHOT and int(item["frame"]) == FRAME_INDEX
    ]
    if len(declarations) != 1:
        raise RuntimeError("the named current-pinned declaration is not unique")
    return matches[0], declarations[0]


def _validate_reproduction(result: dict[str, Any]) -> None:
    residual = float(result["relative_residual"])
    if not np.isclose(
        residual,
        EXPECTED_RELATIVE_RESIDUAL,
        rtol=RESIDUAL_REPRODUCTION_RELATIVE_TOLERANCE,
        atol=RESIDUAL_REPRODUCTION_ABSOLUTE_TOLERANCE,
    ):
        raise RuntimeError(
            f"relative residual {residual:.17g} did not reproduce "
            f"{EXPECTED_RELATIVE_RESIDUAL:.17g}"
        )
    if int(result["iterations"]) != EXPECTED_ITERATIONS:
        raise RuntimeError(
            f"eliminated solve used {result['iterations']} iterations, not 4"
        )
    if result["topology"] != "diverted":
        raise RuntimeError("terminal current-pinned topology is not diverted")
    if float(result["current_relative_error"]) > CURRENT_RELATIVE_ERROR_TOLERANCE:
        raise RuntimeError("terminal current constraint is outside tolerance")


def run(data: Path, output: Path) -> dict[str, Any]:
    """Rerun the landed eliminated arm and publish its comparison artifacts."""

    configure_dtypes()
    frame_input, declared = _named_input()
    columns = tuple(
        dict.fromkeys(
            (
                *_LABEL_COLUMNS,
                *_CURRENT_COLUMNS,
                *_GEOMETRY_COLUMNS,
                *pinned.PLASMA_CURRENT_COLUMNS,
            )
        )
    )
    source_path = data / frame_input.shot
    row = _read(source_path, columns)
    row["_source_path"] = str(source_path)
    profile, seed, given_registered_total, _wall, reliable, statement = build_profile(
        row, frame_input.frame, pinned.PSEUDO_WALL_EXPANSION
    )
    response_geometry = _omitted_vertices()
    profile = append_recovered_conductors(profile, response_geometry)
    _shipped, complete_current = current_arms(profile, frame_input.recovered_currents_a)
    time_ms = float(row["efit_times"][frame_input.frame])
    target_current = pinned._target_current(row, time_ms)
    seed_unscaled_current = float(
        np.sum(np.asarray(profile.operator.cell_current(seed, TopologyClass.DIVERTED)))
    )
    measured_preflight = np.asarray(
        [
            time_ms,
            target_current,
            seed_unscaled_current,
            pinned._lambda_value(target_current, seed_unscaled_current),
        ]
    )
    declared_preflight = np.asarray(
        [
            declared["time_ms"],
            declared["recorded_ip_a"],
            declared["unscaled_source_ip_a"],
            declared["seed_lambda"],
        ],
        dtype=float,
    )
    if not np.allclose(
        measured_preflight,
        declared_preflight,
        rtol=pinned.COHORT_PREFLIGHT_RELATIVE_TOLERANCE,
        atol=1.0e-9,
    ):
        raise RuntimeError("the named current-pinned preflight inputs drifted")

    result = pinned.solve_eliminated(profile, seed, complete_current, target_current)
    _validate_reproduction(result)
    solve_radius = np.asarray(profile.lattice.radius, dtype=float)
    solve_height = np.asarray(profile.lattice.height, dtype=float)
    solved_total = np.asarray(result["state"], dtype=float)[
        : profile.lattice.node_count
    ].reshape(profile.lattice.shape)
    interior = _plasma_mask(row, frame_input.frame, solve_radius, solve_height)
    gauge = gauge_match(given_registered_total, solved_total, interior)

    _masks, solved_topology = profile.operator.read(np.asarray(result["state"]))
    solved_boundary = _separatrix(
        solve_radius,
        solve_height,
        solved_total,
        float(solved_topology.axis_flux),
        float(solved_topology.boundary_flux),
    )
    if len(solved_boundary) < 4:
        raise RuntimeError("the converged solve did not yield a drawable boundary")
    solved_x_point = np.asarray(solved_topology.x_point, dtype=float)
    count = int(row["efit_lcfs_n"][frame_input.frame])
    given_boundary = np.column_stack(
        (
            np.asarray(row["efit_lcfs_r"][frame_input.frame][:count], dtype=float),
            np.asarray(row["efit_lcfs_z"][frame_input.frame][:count], dtype=float),
        )
    )
    full_radius, full_height = canonical_axes(row)
    given_full_wb_per_radian_zr = np.asarray(
        row["efit_psirz"][frame_input.frame], dtype=float
    )
    given_x_point = boundary_gradient_minimum(
        full_radius,
        full_height,
        given_full_wb_per_radian_zr,
        given_boundary,
    )
    levels = contour_levels_from_given(given_full_wb_per_radian_zr)
    solved_raw_wb_per_radian = nova_total_flux_to_corpus(solved_total)
    solved_aligned_wb_per_radian = nova_total_flux_to_corpus(gauge["aligned_total_wb"])
    boundary_separation = _boundary_separation(solved_boundary, given_boundary)
    x_point_separation = float(np.linalg.norm(solved_x_point - given_x_point))

    geometry = read_static_geometry()
    conductor_groups = classify_conductors(geometry)
    if len(geometry.limiter) != 82:
        raise RuntimeError("the netCDF limiter no longer has 82 vertices")
    if len(conductor_groups["shipped"]) != 19:
        raise RuntimeError("the released input no longer has 19 conductor groups")
    if len(conductor_groups["netcdf_only"]) != 5:
        raise RuntimeError("the netCDF-only conductor count is no longer five")
    if given_full_wb_per_radian_zr.shape != (65, 65):
        raise RuntimeError("the released labelled map is no longer 65 by 65")

    fields = {
        "full_radius": full_radius,
        "full_height": full_height,
        "given_full_wb_per_radian_zr": given_full_wb_per_radian_zr,
        "solve_radius": solve_radius,
        "solve_height": solve_height,
        "solved_raw_wb_per_radian_rz": solved_raw_wb_per_radian,
        "solved_aligned_wb_per_radian_rz": solved_aligned_wb_per_radian,
        "levels_wb_per_radian": levels,
        "given_boundary_rz_m": given_boundary,
        "solved_boundary_rz_m": solved_boundary,
        "given_x_point_rz_m": given_x_point,
        "solved_x_point_rz_m": solved_x_point,
        "removed_offset_wb_per_radian": gauge[
            "removed_solve_minus_label_offset_wb_per_radian"
        ],
        "boundary_separation_m": boundary_separation,
        "x_point_separation_m": x_point_separation,
    }
    output.mkdir(parents=True, exist_ok=True)
    figure_path = output / FIGURE_NAME
    render_figure(fields, geometry, figure_path)

    receipt = {
        "frame": {
            "shot": frame_input.shot,
            "frame": frame_input.frame,
            "time_ms": time_ms,
            "source_parquet": str(source_path),
            "source_parquet_sha256": _sha256(source_path),
        },
        "solve_reproduction": {
            "arm": "current-pinned closed-form eliminated amplitude",
            "implementation": (
                "benchmarks.diiid_current_pinned_forward.solve_eliminated"
            ),
            "relative_residual": float(result["relative_residual"]),
            "recorded_relative_residual": EXPECTED_RELATIVE_RESIDUAL,
            "residual_reproduction_relative_tolerance": (
                RESIDUAL_REPRODUCTION_RELATIVE_TOLERANCE
            ),
            "residual_reproduction_absolute_tolerance": (
                RESIDUAL_REPRODUCTION_ABSOLUTE_TOLERANCE
            ),
            "iterations": int(result["iterations"]),
            "recorded_iterations": EXPECTED_ITERATIONS,
            "terminal_topology": result["topology"],
            "current_relative_error": float(result["current_relative_error"]),
            "current_relative_error_tolerance": CURRENT_RELATIVE_ERROR_TOLERANCE,
            "target_plasma_current_a": target_current,
            "profile_amplitude": float(result["amplitude"]),
            "termination": result["termination"],
            "residual_history": [float(value) for value in result["residual_history"]],
        },
        "prescribed_inputs": {
            "complete_poloidal_conductor_count": int(len(complete_current)),
            "recovered_currents_a": {
                name: float(value)
                for name, value in zip(
                    pinned.OMITTED_COILS,
                    frame_input.recovered_currents_a,
                    strict=True,
                )
            },
            "target_current_role": (
                "declared competition input or partner transport input"
            ),
            "reliable_extracted_flux_function_surfaces": reliable,
            "control_surface": statement,
            "coefficients_fitted_to_label": 0,
            "currents_adjusted_to_label": 0,
        },
        "comparison": {
            "interpretation": (
                "Nova's result is a self-consistent equilibrium from prescribed "
                "inputs, not a fit to the labelled map"
            ),
            "fractional_rms_against_label_after_additive_gauge": float(
                gauge["fractional_rms"]
            ),
            "interior_r_squared_after_additive_gauge": float(gauge["r_squared"]),
            "label_representability_ceiling_fractional_rms": (
                LABEL_REPRESENTABILITY_CEILING
            ),
            "fractional_rms_minus_representability_ceiling": float(
                gauge["fractional_rms"] - LABEL_REPRESENTABILITY_CEILING
            ),
            "additive_correction_applied_to_solve_wb_per_radian": float(
                gauge["additive_correction_wb_per_radian"]
            ),
            "removed_solve_minus_label_offset_wb_per_radian": float(
                gauge["removed_solve_minus_label_offset_wb_per_radian"]
            ),
            "raw_panel": "same physical levels with the additive gauge retained",
            "gauge_matched_panel": (
                "same physical levels after one spatially constant core-mean offset"
            ),
        },
        "contours": {
            "kind": "unfilled line contours",
            "level_source": "full 65 by 65 given labelled map in Wb per radian",
            "computed_once": True,
            "applied_verbatim_to_both_maps_in_both_panels": True,
            "level_values_wb_per_radian": [float(value) for value in levels],
            "given_colour": STYLE.separatrix_color,
            "nova_colour": STYLE.flux_color,
        },
        "topology": {
            "given_boundary_source": "released LCFS coordinate string",
            "nova_boundary_source": (
                "terminal diverted topology axis and boundary flux"
            ),
            "boundary_symmetric_mean_separation_m": boundary_separation,
            "given_x_point_rz_m": [float(value) for value in given_x_point],
            "given_x_point_source": (
                "derived minimum flux-gradient norm on released LCFS; not shipped"
            ),
            "nova_x_point_rz_m": [float(value) for value in solved_x_point],
            "x_point_separation_m": x_point_separation,
        },
        "machine_geometry": {
            "source_netcdf": geometry.source_path,
            "source_netcdf_sha256": _sha256(Path(geometry.source_path)),
            "limiter_vertices": int(len(geometry.limiter)),
            "netcdf_conductor_groups": int(len(geometry.coils)),
            "shipped_conductor_groups": int(len(conductor_groups["shipped"])),
            "shipped_conductor_elements": int(conductor_groups["shipped_elements"]),
            "netcdf_only_conductor_groups": int(len(conductor_groups["netcdf_only"])),
            "netcdf_only_conductor_names": sorted(
                coil.name for coil in conductor_groups["netcdf_only"]
            ),
            "netcdf_only_conductor_elements": int(
                conductor_groups["netcdf_only_elements"]
            ),
            "released_grid_shape": [65, 65],
            "registered_solve_grid_shape": [
                int(len(solve_radius)),
                int(len(solve_height)),
            ],
            "registered_grid_stride": REGISTERED_GRID_STRIDE,
            "grid_extent_m": {
                "r": [float(full_radius[0]), float(full_radius[-1])],
                "z": [float(full_height[0]), float(full_height[-1])],
            },
            "physical_axes": "R and Z in metres with equal aspect",
        },
        "artifacts": {
            "figure": str(figure_path),
            "receipt": str(output / RECEIPT_NAME),
        },
        "nova_equilibrium_modified": False,
    }
    receipt_path = output / RECEIPT_NAME
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    receipt = run(arguments.data, arguments.output)
    print(
        json.dumps(
            {
                "figure": receipt["artifacts"]["figure"],
                "receipt": receipt["artifacts"]["receipt"],
                "relative_residual": receipt["solve_reproduction"]["relative_residual"],
                "iterations": receipt["solve_reproduction"]["iterations"],
                "terminal_topology": receipt["solve_reproduction"]["terminal_topology"],
                "current_relative_error": receipt["solve_reproduction"][
                    "current_relative_error"
                ],
                "fractional_rms": receipt["comparison"][
                    "fractional_rms_against_label_after_additive_gauge"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
