"""Draw pre-saddle wall reachability for banked real equilibria.

Every classified cell receives Nova's canonical four-connected component
label.  The public region is the positive component containing the magnetic
axis immediately before the selected in-vessel saddle level; every other
positive label is private.  No height band or vertical extent is consulted.
"""

from __future__ import annotations

import json
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
import numpy as np
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import root
from scipy.spatial import cKDTree

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.diiid_forward_gs_match import (
    DEFAULT_DATA as DIIID_DATA,
    DEFAULT_MACHINE_ARTIFACT_CACHE,
    DEFAULT_MACHINE_ARTIFACT_DIGEST,
    POLOIDAL_CONDUCTORS,
    TOPOLOGY_SURFACE_GMRES_ITERATIONS,
    TOPOLOGY_SURFACE_NEWTON_STEPS,
    _build_profile,
    _margin_graded_newton_krylov,
    _target_current,
    _terminal_xpoint_diagnostics,
    _wall_topology_row,
    complete_profile_current_adapter,
    dataset_machine_description,
    shipped_current_at,
)
from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    FIXED_POINT_CRITERION,
    GMRES_ITERATIONS,
    NEWTON_STEPS,
    RELAXATION,
    STEP_CAP,
    WARMUP_SWEEPS,
    _mast_case_from_selection,
    _passive_inclusive_case,
    select_slices_by_shot,
)
from benchmarks.label_seed_residual_field import _persisted_response_cache
from nova.equilibrium.flux_surface_connectivity import (
    label_connected_components,
    private_flux_mask,
)
from nova.equilibrium.forward import SaddleSeedGeometry
from nova.equilibrium.topology import TopologyClass
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.io import geqdsk
from nova.jax.config import configure_dtypes


HERE = Path(__file__).resolve().parent
OUTPUT_PNG = HERE / "real-equilibria-reachability-grid.png"
OUTPUT_JSON = HERE / "real-equilibria-reachability-grid.json"
MAST_BANK = Path("docs/figures/dual-branch-selection/pinned-branch-contrast.json")
CONTAINMENT_BANK = HERE / "typed-saddle-containment.json"
DIIID_BANK = Path("docs/figures/plateau-input-attribution/margin-frame-remeasure.json")
TORAX_SPEC = find_spec("torax")
if TORAX_SPEC is None or TORAX_SPEC.origin is None:
    raise RuntimeError("TORAX installation is required for the ITER equilibrium bank")
ITER_EQDSK = (
    Path(TORAX_SPEC.origin).resolve().parent
    / "data/third_party/geo/iterhybrid_cocos17.eqdsk"
)
WALL_SEGMENTS = 420
CLASSIFICATION_OFFSET_FRACTION = 2.0e-4
MANDATORY_REFERENCES = ((21983, 35), (21989, 55))


def _closed_wall(wall: np.ndarray) -> np.ndarray:
    finite = np.asarray(wall, dtype=float)[np.isfinite(wall).all(axis=1)]
    if not np.array_equal(finite[0], finite[-1]):
        finite = np.vstack((finite, finite[0]))
    return finite


def _densify_wall(wall: np.ndarray) -> np.ndarray:
    segment = np.hypot(np.diff(wall[:, 0]), np.diff(wall[:, 1]))
    distance = np.concatenate(([0.0], np.cumsum(segment)))
    query = np.linspace(0.0, distance[-1], WALL_SEGMENTS + 1)
    return np.column_stack(
        (np.interp(query, distance, wall[:, 0]), np.interp(query, distance, wall[:, 1]))
    )


def _mast_states(
    profile, seed: jax.Array, target_current: float
) -> dict[str, jax.Array]:
    initial = jnp.stack((seed, seed))
    portfolio = profile.solve_portfolio(
        initial,
        route="newton_krylov",
        target_current=target_current,
        tolerance=FIXED_POINT_CRITERION,
        newton_steps=NEWTON_STEPS,
        gmres_iterations=GMRES_ITERATIONS,
        warmup=WARMUP_SWEEPS,
        relaxation=RELAXATION,
        step_cap=STEP_CAP,
    )
    portfolio.branches.equilibrium.flux.block_until_ready()
    pure = portfolio.branches.equilibrium.flux[int(TopologyClass.DIVERTED)]
    mapped = profile.flux_map(
        requested_class=TopologyClass.DIVERTED, target_current=target_current
    )
    mixed = _margin_graded_newton_krylov(
        mapped,
        profile.operator.topology_margin,
        seed,
        newton_steps=NEWTON_STEPS,
        gmres_iterations=GMRES_ITERATIONS,
    ).state
    mixed.block_until_ready()
    return {"pure": pure, "mixed": mixed}


def _diiid_state(shot_name: str, frame: int) -> tuple[Any, jax.Array]:
    row = _wall_topology_row(DIIID_DATA / shot_name)
    built = _build_profile(
        row,
        frame,
        None,
        machine_artifact_cache=DEFAULT_MACHINE_ARTIFACT_CACHE,
        machine_artifact_digest=DEFAULT_MACHINE_ARTIFACT_DIGEST,
    )
    time_ms = float(row["efit_times"][frame])
    machine = dataset_machine_description(row, source_row=str(row["_source_path"]))
    shipped_current = shipped_current_at(
        row, machine.physical, POLOIDAL_CONDUCTORS, time_ms
    )
    adapter = complete_profile_current_adapter(
        built.profile,
        shipped_names=POLOIDAL_CONDUCTORS,
        shipped_current_a=shipped_current,
        use_circuit=True,
    )
    profile = adapter.profile
    current = np.asarray(adapter.resolution.current(()), dtype=float)
    target_current = _target_current(row, time_ms)
    count = int(row["efit_lcfs_n"][frame])
    contour = np.c_[
        np.asarray(row["efit_lcfs_r"][frame][:count], dtype=float),
        np.asarray(row["efit_lcfs_z"][frame][:count], dtype=float),
    ]
    axis = np.asarray((row["efit_r_axis"][frame], row["efit_z_axis"][frame]))
    saddle = contour[int(np.argmin(contour[:, 1]))]
    cold = profile.cold_seed_portfolio(
        target_current,
        axis,
        current=jnp.asarray(current),
        diverted_geometry=SaddleSeedGeometry(tuple(axis), tuple(saddle)),
    )
    seed = cold.branches.flux[int(TopologyClass.DIVERTED)]
    mapped = profile.flux_map(
        jnp.asarray(current), TopologyClass.DIVERTED, target_current
    )
    state = _margin_graded_newton_krylov(
        mapped,
        profile.operator.topology_margin,
        seed,
        newton_steps=TOPOLOGY_SURFACE_NEWTON_STEPS,
        gmres_iterations=TOPOLOGY_SURFACE_GMRES_ITERATIONS,
    ).state
    state.block_until_ready()
    return profile, state


def _grid_geometry(profile, state: jax.Array) -> dict[str, Any]:
    operator = profile.operator
    physical = jnp.asarray(state)[: operator.physical_node_number]
    grid_flux, _wall_flux = operator.topology.split_flux_map(physical)
    radius, height, shape = operator.connectivity_grid_axes()
    radial_count, vertical_count = shape
    flux = np.asarray(grid_flux).reshape((radial_count, vertical_count)).T
    inside = (
        np.asarray(operator.inside_material)
        .reshape((radial_count, vertical_count))
        .T.astype(bool)
    )
    _masks, topology = operator.read(state)
    diagnostics = _terminal_xpoint_diagnostics(profile, state, topology)
    wall = _closed_wall(np.asarray(operator.wall.coordinate, dtype=float))
    typed_saddles = np.asarray(
        [row["coordinate_m"] for row in diagnostics["typed_saddle_candidates"]],
        dtype=float,
    )
    typed_saddle_flux = np.asarray(
        [row["flux_wb"] for row in diagnostics["typed_saddle_candidates"]],
        dtype=float,
    )
    typed_inside = (
        MplPath(wall).contains_points(typed_saddles, radius=1.0e-12)
        if typed_saddles.size
        else np.empty(0, dtype=bool)
    )
    return {
        "radius": np.asarray(radius),
        "height": np.asarray(height),
        "flux": flux,
        "inside": inside,
        "axis": np.asarray(topology.axis, dtype=float),
        "wall": wall,
        "selected_x": np.asarray(diagnostics["selected_x_coordinate_m"], dtype=float),
        "selected_x_flux": float(diagnostics["selected_x_flux_wb"]),
        "selected_wall": np.asarray(
            diagnostics["wall_operand"]["coordinate_m"], dtype=float
        ),
        "class_margin": float(topology.class_margin),
        "typed_saddle_coordinates_m": typed_saddles.tolist(),
        "typed_saddle_flux_wb": typed_saddle_flux.tolist(),
        "typed_saddles_inside_wall": typed_inside.tolist(),
    }


def _iter_geometry() -> dict[str, Any]:
    data = geqdsk.read(str(ITER_EQDSK))
    radius = np.asarray(data["x"], dtype=float)
    height = np.asarray(data["z"], dtype=float)
    flux = np.asarray(data["psi"], dtype=float).T
    wall = _closed_wall(np.c_[data["xlim"], data["zlim"]])
    axis = np.asarray((data["xmagx"], data["zmagx"]), dtype=float)
    spline = RectBivariateSpline(height, radius, flux)

    def gradient(point: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                spline.ev(point[1], point[0], dx=0, dy=1),
                spline.ev(point[1], point[0], dx=1, dy=0),
            ]
        )

    vertical_gradient, radial_gradient = np.gradient(flux, height, radius)
    gradient_norm = np.hypot(radial_gradient, vertical_gradient)
    local_minimum = gradient_norm == ndimage.minimum_filter(gradient_norm, size=5)
    local_minimum[:2] = False
    local_minimum[-2:] = False
    local_minimum[:, :2] = False
    local_minimum[:, -2:] = False
    candidate_indices = np.argwhere(local_minimum)
    candidate_indices = candidate_indices[
        np.argsort(gradient_norm[local_minimum])[:100]
    ]
    saddles: list[np.ndarray] = []
    for vertical_index, radial_index in candidate_indices:
        solved = root(
            gradient,
            np.asarray((radius[radial_index], height[vertical_index]), dtype=float),
        )
        candidate = np.asarray(solved.x, dtype=float)
        if not solved.success or np.linalg.norm(gradient(candidate)) > 1.0e-6:
            continue
        if not (
            radius[0] <= candidate[0] <= radius[-1]
            and height[0] <= candidate[1] <= height[-1]
            and MplPath(wall).contains_point(candidate, radius=1.0e-12)
        ):
            continue
        mixed_derivative = spline.ev(candidate[1], candidate[0], dx=1, dy=1)
        hessian = np.asarray(
            [
                [
                    spline.ev(candidate[1], candidate[0], dx=0, dy=2),
                    mixed_derivative,
                ],
                [
                    mixed_derivative,
                    spline.ev(candidate[1], candidate[0], dx=2, dy=0),
                ],
            ]
        )
        if np.linalg.det(hessian) >= 0.0:
            continue
        if any(
            np.linalg.norm(candidate - retained) < 0.5 * np.mean(np.diff(radius))
            for retained in saddles
        ):
            continue
        saddles.append(candidate)
    saddle_array = np.stack(saddles) if saddles else np.empty((0, 2), dtype=float)
    if saddles:
        saddle_flux = spline.ev(saddle_array[:, 1], saddle_array[:, 0])
        selected_index = int(np.argmin(np.abs(saddle_flux - float(data["sibdry"]))))
        selected_x = saddle_array[selected_index]
        selected_x_flux = float(spline.ev(selected_x[1], selected_x[0]))
        selection_status = "selected_hessian_typed_in_limiter_saddle"
    else:
        selected_x = None
        selected_x_flux = None
        selection_status = "no_resolved_hessian_typed_in_limiter_saddle"
    wall_flux = spline.ev(wall[:, 1], wall[:, 0])
    axis_flux = float(spline.ev(axis[1], axis[0]))
    outward_sign = np.sign(float(data["sibdry"]) - axis_flux) or 1.0
    wall_index = int(np.argmin(outward_sign * (wall_flux[:-1] - axis_flux)))
    rr, zz = np.meshgrid(radius, height)
    inside = (
        (rr >= wall[:, 0].min())
        & (rr <= wall[:, 0].max())
        & (zz >= wall[:, 1].min())
        & (zz <= wall[:, 1].max())
    )
    return {
        "radius": radius,
        "height": height,
        "flux": flux,
        "inside": inside,
        "axis": axis,
        "wall": wall,
        "selected_x": selected_x,
        "selected_x_flux": selected_x_flux,
        "fallback_boundary_flux": float(data["sibdry"]),
        "selected_wall": wall[wall_index],
        "class_margin": None,
        "typed_saddle_coordinates_m": saddle_array.tolist(),
        "typed_saddle_flux_wb": (
            spline.ev(saddle_array[:, 1], saddle_array[:, 0]).tolist()
            if saddles
            else []
        ),
        "typed_saddles_inside_wall": [True] * len(saddle_array),
        "selection_status": selection_status,
        "topology_hint": "limited",
    }


def _classify(geometry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    radius = geometry["radius"]
    height = geometry["height"]
    flux = geometry["flux"]
    inside = geometry["inside"]
    axis = geometry["axis"]
    selected_x_flux = geometry["selected_x_flux"]
    reachability_available = selected_x_flux is not None
    axis_row = int(np.argmin(np.abs(height - axis[1])))
    axis_column = int(np.argmin(np.abs(radius - axis[0])))
    axis_flux = float(flux[axis_row, axis_column])
    binding_flux = (
        selected_x_flux
        if reachability_available
        else geometry["fallback_boundary_flux"]
    )
    outward_sign = np.sign(binding_flux - axis_flux) or 1.0
    outward = outward_sign * (flux - axis_flux)
    saddle_level = outward_sign * (binding_flux - axis_flux)
    flux_range = max(float(np.ptp(outward[inside])), np.finfo(float).eps)
    offset = CLASSIFICATION_OFFSET_FRACTION * flux_range
    classification_level = saddle_level - offset
    confined = inside & (outward <= classification_level)
    labels = np.asarray(
        label_connected_components(jnp.asarray(confined), sum(confined.shape))
    )
    rr, zz = np.meshgrid(radius, height)
    axis_distance = np.hypot(rr - axis[0], zz - axis[1])
    nominal_axis_distance = float(axis_distance[axis_row, axis_column])
    nominal_axis_confined = bool(confined[axis_row, axis_column])
    confined_distance = np.where(confined, axis_distance, np.inf)
    confined_flat = int(np.argmin(confined_distance))
    if not np.isfinite(confined_distance.ravel()[confined_flat]):
        raise RuntimeError("pre-saddle region contains no confined grid cell")
    axis_row, axis_column = np.unravel_index(confined_flat, confined.shape)
    classified_axis_distance = float(axis_distance[axis_row, axis_column])
    axis_label = int(labels[axis_row, axis_column])
    if axis_label <= 0:
        raise RuntimeError("nearest confined axis seed has no positive component label")
    nominal_axis_inside_material = bool(inside[axis_row, axis_column])
    if nominal_axis_confined:
        exclusion_reason = None
    elif not nominal_axis_inside_material:
        exclusion_reason = "nearest_grid_cell_outside_material_mask"
    elif classification_level < 0.0:
        exclusion_reason = "inward_offset_exceeds_selected_saddle_level"
    else:
        exclusion_reason = "nearest_grid_cell_outside_pre_saddle_flux_threshold"
    seed = np.zeros_like(inside)
    seed[axis_row, axis_column] = True
    private = np.asarray(private_flux_mask(jnp.asarray(labels), jnp.asarray(seed)))
    public = labels == axis_label
    dense_wall = _densify_wall(geometry["wall"])
    coordinates = np.column_stack((rr[inside], zz[inside]))
    nearest = cKDTree(coordinates).query(dense_wall[:-1], k=1)[1]
    wall_labels = labels[inside][nearest]
    reachable = wall_labels == axis_label
    private_labels = np.unique(labels[private])
    contained_saddle_count = int(
        np.count_nonzero(np.asarray(geometry["typed_saddles_inside_wall"], dtype=bool))
    )
    typed_inside = np.asarray(geometry["typed_saddles_inside_wall"], dtype=bool)
    typed_flux = np.asarray(geometry["typed_saddle_flux_wb"], dtype=float)
    typed_levels = outward_sign * (typed_flux - axis_flux)
    first_crossing_saddle_count = int(
        np.count_nonzero(typed_inside & (np.abs(typed_levels - saddle_level) <= offset))
    )
    topology = (
        geometry["topology_hint"]
        if not reachability_available
        else "limited"
        if np.any(reachable)
        else "double-null"
        if first_crossing_saddle_count >= 2
        else "diverted"
    )
    cell_area = float(np.mean(np.diff(radius)) * np.mean(np.diff(height)))
    record = {
        "topology": topology,
        "reachability_available": reachability_available,
        "selection_status": geometry.get("selection_status", "selected_typed_saddle"),
        "grid_shape_height_by_radius": list(flux.shape),
        "connectivity": 4,
        "axis_component_label": axis_label,
        "classification_level_wb_from_axis_outward": classification_level,
        "classification_inward_offset_wb": offset,
        "selected_saddle_level_wb_from_axis_outward": saddle_level,
        "nominal_nearest_axis_cell_coordinate_m": [
            float(radius[int(np.argmin(np.abs(radius - axis[0])))]),
            float(height[int(np.argmin(np.abs(height - axis[1])))]),
        ],
        "nominal_nearest_axis_cell_confined": nominal_axis_confined,
        "nominal_nearest_axis_cell_outward_flux_wb": float(
            outward[
                int(np.argmin(np.abs(height - axis[1]))),
                int(np.argmin(np.abs(radius - axis[0]))),
            ]
        ),
        "axis_to_nominal_nearest_cell_distance_m": nominal_axis_distance,
        "classified_axis_seed_coordinate_m": [
            float(radius[axis_column]),
            float(height[axis_row]),
        ],
        "axis_to_classified_cell_distance_m": classified_axis_distance,
        "nominal_axis_cell_exclusion_reason": exclusion_reason,
        "reachable_wall_segment_count": (
            int(np.count_nonzero(reachable)) if reachability_available else None
        ),
        "shadowed_wall_segment_count": (
            int(np.count_nonzero(~reachable)) if reachability_available else None
        ),
        "wall_segment_count": int(reachable.size),
        "public_region_area_m2": float(np.count_nonzero(public) * cell_area),
        "private_region_count": int(private_labels.size),
        "private_region_area_m2": float(np.count_nonzero(private) * cell_area),
        "contained_typed_saddle_count": contained_saddle_count,
        "first_crossing_typed_saddle_count": first_crossing_saddle_count,
        "typed_saddle_levels_wb_from_axis_outward": typed_levels.tolist(),
        "typed_saddle_coordinates_m": geometry["typed_saddle_coordinates_m"],
        "axis_coordinate_m": axis.tolist(),
        "selected_x_coordinate_m": (
            geometry["selected_x"].tolist() if reachability_available else None
        ),
        "selected_wall_coordinate_m": geometry["selected_wall"].tolist(),
        "class_margin": geometry["class_margin"],
    }
    plot = geometry | {
        "outward": outward,
        "binding_level": saddle_level,
        "public": public,
        "private": private,
        "dense_wall": dense_wall,
        "reachable": reachable,
    }
    return record, plot


def _draw(axis_plot, plot: dict[str, Any]) -> None:
    code = np.zeros(plot["inside"].shape, dtype=int)
    code[plot["public"]] = 1
    code[plot["private"]] = 2
    axis_plot.pcolormesh(
        plot["radius"],
        plot["height"],
        np.ma.masked_where(code == 0, code),
        cmap=ListedColormap(["#85c6d4", "#d89b55"]),
        vmin=1,
        vmax=2,
        shading="nearest",
        alpha=0.62,
        rasterized=True,
    )
    axis_plot.contour(
        plot["radius"],
        plot["height"],
        np.where(plot["inside"], plot["outward"], np.nan),
        levels=[plot["binding_level"]],
        colors=["#232323"],
        linewidths=1.2,
    )
    wall = plot["dense_wall"]
    for index, reachable in enumerate(plot["reachable"]):
        axis_plot.plot(
            wall[index : index + 2, 0],
            wall[index : index + 2, 1],
            color=(
                "#777777"
                if not plot["record"]["reachability_available"]
                else "#198754"
                if reachable
                else "#b43b3b"
            ),
            linewidth=2.4,
            solid_capstyle="butt",
        )
    axis_plot.scatter(*plot["axis"], marker="o", s=28, color="#135d6a", zorder=6)
    if plot["record"]["reachability_available"]:
        axis_plot.scatter(
            *plot["selected_x"], marker="X", s=48, color="#512b81", zorder=7
        )
    axis_plot.scatter(
        *plot["selected_wall"], marker="D", s=32, color="#f2c14e", zorder=7
    )
    row = plot["record"]
    if row.get("selection_changed_by_containment"):
        before_x = np.asarray(row["pre_repair_selected_x_coordinate_m"], dtype=float)
        after_x = np.asarray(row["post_repair_selected_x_coordinate_m"], dtype=float)
        axis_plot.plot(
            [before_x[0], after_x[0]],
            [before_x[1], after_x[1]],
            color="#7b7b7b",
            linewidth=1.0,
            linestyle="--",
            zorder=6,
        )
        axis_plot.scatter(
            *before_x,
            marker="x",
            s=44,
            color="#7b7b7b",
            linewidths=1.5,
            zorder=7,
        )
    axis_plot.set_title(
        f"{row['panel']}  {row['machine']} {row['shot_slice']} · {row['topology']}",
        loc="left",
        fontsize=9.5,
        fontweight="semibold",
    )
    wall_summary = (
        f"wall {row['reachable_wall_segment_count']} / "
        f"{row['shadowed_wall_segment_count']}\n"
        if row["reachability_available"]
        else "wall reachability unavailable: no contained saddle\n"
    )
    axis_plot.text(
        0.02,
        0.02,
        wall_summary + f"public {row['public_region_area_m2']:.3f} m² · private "
        f"{row['private_region_count']} / {row['private_region_area_m2']:.3f} m²",
        transform=axis_plot.transAxes,
        fontsize=7.5,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
    )
    axis_plot.set_aspect("equal", adjustable="box")
    axis_plot.set_xlabel("R [m]")
    axis_plot.set_ylabel("Z [m]")
    axis_plot.spines[["top", "right"]].set_visible(False)


def run() -> dict[str, Any]:
    configure_dtypes()
    before_rows = json.loads(CONTAINMENT_BANK.read_text())["rows"]
    before = {(row["reference"], row["arm"]): row for row in before_rows}
    selected = {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in select_slices_by_shot(DECOMPOSITION_BANK)
    }
    response_cache, carrier_evidence = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for key, (selected_row, qualification) in selected.items():
        case, context = _mast_case_from_selection(
            SHOT_STORE, selected_row, qualification
        )
        passive_case, profile, policy = _passive_inclusive_case(
            case, context, response_cache
        )
        if policy["section_kernel_evaluations_this_shot"] != 0:
            raise RuntimeError("MAST reconstruction entered a direct response builder")
        seed = jnp.asarray(passive_case["state"])
        target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
        for arm, state in _mast_states(profile, seed, target_current).items():
            record, plot = _classify(_grid_geometry(profile, state))
            reference = f"{key[0]}/{key[1]}"
            delta = before[(reference, arm)]
            record.update(
                {
                    "machine": "MAST",
                    "shot": key[0],
                    "slice_index": key[1],
                    "shot_slice": reference,
                    "arm": arm,
                    "field_source": (
                        "reconstructed terminal state from persisted response carrier"
                    ),
                    "selection_source_commit": "5acfe07b",
                    "pre_repair_selected_x_coordinate_m": delta["before"][
                        "selected_x_coordinate_m"
                    ],
                    "post_repair_selected_x_coordinate_m": delta["after"][
                        "selected_x_coordinate_m"
                    ],
                    "selection_changed_by_containment": delta["changed"],
                    "pre_repair_class_margin": delta["before"]["class_margin"],
                    "post_repair_class_margin": delta["after"]["class_margin"],
                    "post_repair_class_margin_nonfinite": delta["after"][
                        "class_margin_nonfinite"
                    ],
                }
            )
            plot["record"] = record
            candidates.append((record, plot))

    diiid_receipt = json.loads(DIIID_BANK.read_text())
    diiid_row = diiid_receipt["arms"]["physical_ring"]["frame_records"][0]
    shot_name = diiid_row["shot"]
    frame = int(diiid_row["frame"])
    profile, state = _diiid_state(shot_name, frame)
    record, plot = _classify(_grid_geometry(profile, state))
    shot_label = shot_name.removeprefix("d3d_shot_").removesuffix(".parquet")
    record.update(
        {
            "machine": "DIII-D",
            "shot": shot_label,
            "slice_index": frame,
            "shot_slice": f"{shot_label}/{frame}",
            "arm": "physical ring",
            "field_source": "regenerated terminal behind margin-frame-remeasure.json",
            "selection_source_commit": "5acfe07b",
        }
    )
    plot["record"] = record
    candidates.append((record, plot))

    record, plot = _classify(_iter_geometry())
    record.update(
        {
            "machine": "ITER",
            "shot": None,
            "slice_index": 0,
            "shot_slice": "shot unavailable / slice 0",
            "arm": "banked EQDSK",
            "field_source": str(ITER_EQDSK),
            "wall_source": "EQDSK limiter contour",
            "selection_source_commit": "5acfe07b",
            "reachability_unavailable_reason": (
                "the banked field resolves its magnetic-axis minimum but no "
                "Hessian-typed stationary saddle inside the EQDSK limiter; "
                "containment therefore leaves no eligible selected X"
            ),
        }
    )
    plot["record"] = record
    candidates.append((record, plot))

    chosen: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for key in MANDATORY_REFERENCES:
        chosen.append(
            next(
                item
                for item in candidates
                if item[0].get("machine") == "MAST"
                and (item[0]["shot"], item[0]["slice_index"]) == key
                and item[0]["arm"] == "pure"
            )
        )
    for topology in ("limited", "diverted", "double-null"):
        if not any(item[0]["topology"] == topology for item in chosen):
            match = next(
                (item for item in candidates if item[0]["topology"] == topology), None
            )
            if match is not None:
                chosen.append(match)
    for machine in ("DIII-D", "ITER"):
        if not any(item[0]["machine"] == machine for item in chosen):
            chosen.append(
                next(item for item in candidates if item[0]["machine"] == machine)
            )
    unique: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in chosen:
        identity = (
            item[0]["machine"],
            item[0]["shot"],
            item[0]["slice_index"],
            item[0]["arm"],
        )
        if identity not in seen:
            unique.append(item)
            seen.add(identity)
    chosen = unique
    for index, (row, _plot) in enumerate(chosen):
        row["panel"] = chr(ord("A") + index)

    columns = 3
    rows = (len(chosen) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows, columns, figsize=(12.8, 4.5 * rows), constrained_layout=True
    )
    axes_array = np.atleast_1d(axes).ravel()
    for axis_plot, (_row, panel) in zip(axes_array, chosen, strict=False):
        _draw(axis_plot, panel)
    for axis_plot in axes_array[len(chosen) :]:
        axis_plot.remove()
    fig.legend(
        handles=[
            Line2D([0], [0], color="#198754", lw=3, label="reachable wall"),
            Line2D([0], [0], color="#b43b3b", lw=3, label="shadowed wall"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#85c6d4",
                label="public axis component",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#d89b55",
                label="private component",
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color="none",
                markerfacecolor="#512b81",
                label="selected in-vessel X",
            ),
            Line2D(
                [0],
                [0],
                marker="x",
                color="#7b7b7b",
                label="pre-containment selected X",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor="#f2c14e",
                label="selected wall point",
            ),
        ],
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    fig.savefig(OUTPUT_PNG, dpi=190, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "artifact": "real-equilibrium pre-saddle wall reachability grid",
        "project_absolute_src": (
            "/nova/figures/primary-xpoint-evidence/"
            "real-equilibria-reachability-grid.png"
        ),
        "method": {
            "definition": (
                "reachable wall segments touch the magnetic-axis-connected "
                "component immediately before selected saddle crossing"
            ),
            "private_predicate": (
                "positive component label unequal to the magnetic-axis component label"
            ),
            "height_band_used": False,
            "wall_segment_sampling": WALL_SEGMENTS,
            "selection_repair_source_commit": "5acfe07b",
            "fixture_grid_source_commit": "04a4a6ca",
            "topology_assignment": (
                "limited when the axis component reaches the wall before the "
                "selected saddle; otherwise double-null when two or more contained "
                "typed saddles enter within the measured inward offset at the first "
                "crossing, and diverted when only one does"
            ),
        },
        "source_banks": {
            "mast": str(MAST_BANK),
            "mast_arm_count_reconstructed": 12,
            "diiid": str(DIIID_BANK),
            "diiid_terminal_count_available": 10,
            "iter": str(ITER_EQDSK),
            "mast_carrier": carrier_evidence,
        },
        "mast_axis_seed_audit": [
            {
                key: row[key]
                for key in (
                    "shot_slice",
                    "arm",
                    "selected_saddle_level_wb_from_axis_outward",
                    "classification_inward_offset_wb",
                    "classification_level_wb_from_axis_outward",
                    "axis_to_nominal_nearest_cell_distance_m",
                    "nominal_nearest_axis_cell_outward_flux_wb",
                    "nominal_nearest_axis_cell_confined",
                    "axis_to_classified_cell_distance_m",
                    "nominal_axis_cell_exclusion_reason",
                )
            }
            for row, _plot in candidates
            if row["machine"] == "MAST"
        ],
        "coverage": {
            "panel_count": len(chosen),
            "machines": sorted({row["machine"] for row, _plot in chosen}),
            "topologies": sorted({row["topology"] for row, _plot in chosen}),
            "double_null_first_crossing_available_in_reconstructed_bank": any(
                row["reachability_available"] and row["topology"] == "double-null"
                for row, _plot in candidates
            ),
            "mandatory_references_present": all(
                any(
                    row.get("machine") == "MAST"
                    and (row["shot"], row["slice_index"]) == key
                    for row, _plot in chosen
                )
                for key in MANDATORY_REFERENCES
            ),
        },
        "reconstructed_bank_topology_census": [
            {
                key: row[key]
                for key in (
                    "machine",
                    "shot_slice",
                    "arm",
                    "topology",
                    "reachability_available",
                    "reachable_wall_segment_count",
                    "contained_typed_saddle_count",
                    "first_crossing_typed_saddle_count",
                )
            }
            for row, _plot in candidates
        ],
        "panels": [row for row, _plot in chosen],
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


if __name__ == "__main__":
    report = run()
    for row in report["panels"]:
        print(
            row["panel"],
            row["machine"],
            row["shot_slice"],
            row["topology"],
            row["reachable_wall_segment_count"],
            row["shadowed_wall_segment_count"],
            row["private_region_count"],
        )
