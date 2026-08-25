"""Compose production reachability with the independent EFIT overlay.

The solved fields and EFIT geometry come through the existing corroboration and
real-equilibrium drivers.  Cell ownership is recomputed with the production
saddle-aware six-neighbour partition, immediately inside the binding level.
"""

from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.diiid_forward_gs_match import _wall_topology_row
from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    _mast_case_from_selection,
    _passive_inclusive_case,
    select_slices_by_shot,
)
from benchmarks.label_seed_residual_field import _persisted_response_cache
from nova.equilibrium.connectivity_boundary import (
    _PRE_SADDLE_OFFSET_FRACTION,
    _raster_hex_partition_geometry,
    _wall_nodes_touching_region,
)
from nova.equilibrium.flux_surface_connectivity import (
    hex_edge_admissibility,
    label_saddle_aware_hex_connected_components,
)
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import configure_dtypes


HERE = Path(__file__).resolve().parent
OUTPUT_PNG = HERE / "reachability-grid-final.png"
OUTPUT_JSON = HERE / "reachability-grid-final.json"
CORROBORATION_SCRIPT = HERE / "efit_topology_corroboration.py"
CORROBORATION_JSON = HERE / "efit-topology-corroboration.json"
PINNED_BANK = HERE.parent / "dual-branch-selection/pinned-branch-contrast.json"
DIIID_BANK = HERE.parent / "plateau-input-attribution/margin-frame-remeasure.json"
PROJECT_SRC = "/nova/figures/primary-xpoint-evidence/reachability-grid-final.png"


def _load_module(path: Path, name: str):
    """Load a neighbouring evidence driver as the implementation authority."""

    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load evidence driver {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _achieved_classes() -> dict[tuple[str, str], str]:
    """Read post-rebaseline class-margin outcomes from the frozen MAST bank."""

    payload = json.loads(PINNED_BANK.read_text())
    result: dict[tuple[str, str], str] = {}
    for reference in payload["references"]:
        identity = (
            f"{reference['reference']['shot']}/{reference['reference']['slice_index']}"
        )
        for arm in ("pure", "mixed"):
            arm_record = reference[f"{arm}_arm"]
            margin = arm_record["class_margin"]
            nonfinite = arm_record["class_margin_nonfinite"]
            if margin is not None:
                result[(identity, arm)] = (
                    "diverted" if float(margin) >= 0.0 else "limited"
                )
            elif nonfinite == "positive_infinity":
                result[(identity, arm)] = "diverted"
            elif nonfinite == "negative_infinity":
                result[(identity, arm)] = "limited"
            else:
                raise RuntimeError(f"the class outcome is absent for {identity} {arm}")
    if len(result) != 12:
        raise RuntimeError("the pinned bank does not carry twelve achieved classes")
    return result


def _efit_rows() -> dict[tuple[str, str], dict[str, Any]]:
    """Read the independent MAST labels and geometry from its banked receipt."""

    payload = json.loads(CORROBORATION_JSON.read_text())
    rows = {(row["identity"], row["arm"]): row for row in payload["rows"]}
    if len(rows) != 12:
        raise RuntimeError("the corroboration receipt does not carry twelve arms")
    return rows


def _partition(geometry: dict[str, Any], binding_flux: float) -> dict[str, Any]:
    """Return the exact production saddle-aware partition before binding."""

    radius = jnp.asarray(geometry["radius"], dtype=jnp.float64)
    height = jnp.asarray(geometry["height"], dtype=jnp.float64)
    flux = jnp.asarray(geometry["flux"], dtype=jnp.float64)
    inside = jnp.asarray(geometry["inside"], dtype=bool)
    axis = np.asarray(geometry["axis"], dtype=float)
    axis_row = int(np.argmin(np.abs(np.asarray(height) - axis[1])))
    axis_column = int(np.argmin(np.abs(np.asarray(radius) - axis[0])))
    axis_flux = float(flux[axis_row, axis_column])
    outward_sign = float(np.sign(binding_flux - axis_flux) or 1.0)
    outward = outward_sign * (flux - axis_flux)
    binding_level = outward_sign * (binding_flux - axis_flux)
    inside_values = jnp.where(inside, outward, jnp.nan)
    flux_range = jnp.nanmax(inside_values) - jnp.nanmin(inside_values)
    inward_offset = _PRE_SADDLE_OFFSET_FRACTION * flux_range
    classified_level = binding_level - inward_offset
    confined = inside & (outward <= classified_level)

    rr, zz = jnp.meshgrid(radius, height)
    distance2 = (rr - axis[0]) ** 2 + (zz - axis[1]) ** 2
    seed_index = int(
        jnp.argmin(jnp.where(confined.reshape(-1), distance2.reshape(-1), jnp.inf))
    )
    if not bool(jnp.any(confined)):
        raise RuntimeError("the pre-saddle partition has no confined cell")
    seed = jnp.zeros_like(confined).reshape(-1).at[seed_index].set(True)
    seed = seed.reshape(confined.shape)
    rings, shared_edges = _raster_hex_partition_geometry(radius, height)
    link_admissible = hex_edge_admissibility(
        outward,
        radius,
        height,
        classified_level,
        jnp.asarray(0.0, dtype=outward.dtype),
        shared_edges,
    )
    labels = label_saddle_aware_hex_connected_components(
        confined, rings, link_admissible, sum(confined.shape)
    )
    labels_host = np.asarray(jax.device_get(labels))
    seed_host = np.asarray(seed)
    axis_label = int(labels_host[seed_host][0])
    public = labels_host == axis_label
    private = (labels_host > 0) & ~public

    wall = np.asarray(geometry["wall"], dtype=float)
    if np.allclose(wall[0], wall[-1], rtol=0.0, atol=1.0e-12):
        wall = wall[:-1]
    reachable_nodes = np.asarray(
        jax.device_get(
            _wall_nodes_touching_region(
                jnp.asarray(public), inside, radius, height, wall[:, 0], wall[:, 1]
            )
        ),
        dtype=bool,
    )
    reachable_segments = reachable_nodes & np.roll(reachable_nodes, -1)
    radial_pitch = float(np.mean(np.diff(np.asarray(radius))))
    vertical_pitch = float(np.mean(np.diff(np.asarray(height))))
    cell_area = radial_pitch * vertical_pitch
    private_labels = np.unique(labels_host[private])
    classified_seed = np.unravel_index(seed_index, confined.shape)
    return {
        "outward": np.asarray(outward),
        "binding_level": float(binding_level),
        "binding_flux": float(binding_flux),
        "classified_level": float(classified_level),
        "inward_offset": float(inward_offset),
        "public": public,
        "private": private,
        "wall": wall,
        "reachable_segments": reachable_segments,
        "public_area": float(np.count_nonzero(public) * cell_area),
        "private_area": float(np.count_nonzero(private) * cell_area),
        "private_count": int(private_labels.size),
        "axis_to_seed_distance": float(np.sqrt(np.asarray(distance2)[classified_seed])),
    }


def _mast_panels(corroboration, reachability) -> list[dict[str, Any]]:
    """Reconstruct all frozen MAST arms and join their independent overlays."""

    achieved = _achieved_classes()
    referee = _efit_rows()
    selected = select_slices_by_shot(DECOMPOSITION_BANK)
    response_cache, _carrier = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    panels: list[dict[str, Any]] = []
    for selected_row, qualification in selected:
        shot = int(selected_row["shot"])
        slice_index = int(selected_row["slice_index"])
        identity = f"{shot}/{slice_index}"
        print(f"reconstructing MAST {identity}", flush=True)
        case, context = _mast_case_from_selection(
            SHOT_STORE, selected_row, qualification
        )
        passive_case, profile, policy = _passive_inclusive_case(
            case, context, response_cache
        )
        if policy["section_kernel_evaluations_this_shot"] != 0:
            raise RuntimeError("MAST reconstruction entered a direct response builder")
        target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
        states = reachability._mast_states(
            profile, jnp.asarray(passive_case["state"]), target_current
        )
        for arm, state in states.items():
            key = (identity, arm)
            overlay = referee[key]
            geometry = reachability._grid_geometry(profile, state)
            _masks, topology = profile.operator.read(state)
            boundary = corroboration._post_cutover_geometry(profile, state, topology)
            partition = _partition(geometry, boundary["binding_flux"])
            nova_class = achieved[key]
            if boundary["achieved_class"] != nova_class:
                raise RuntimeError(
                    f"live class for {identity} {arm} disagrees with the pinned bank"
                )
            panels.append(
                {
                    "machine": "MAST",
                    "identity": identity,
                    "arm": arm,
                    "radius": geometry["radius"],
                    "height": geometry["height"],
                    "inside": geometry["inside"],
                    "axis": geometry["axis"],
                    "selected_saddle": boundary["selected_saddle"],
                    "limiter": boundary["limiter_coordinate"],
                    "binding_contour": corroboration._binding_contour(
                        {
                            "radius": geometry["radius"],
                            "height": geometry["height"],
                            "flux": geometry["flux"],
                            "axis": geometry["axis"],
                            "boundary_flux": boundary["binding_flux"],
                        },
                        boundary["selected_saddle"]
                        if nova_class == "diverted"
                        else boundary["limiter_coordinate"],
                    ),
                    "efit_lcfs": np.asarray(overlay["efit_lcfs_m"], dtype=float),
                    "efit_x_points": np.asarray(
                        overlay["efit_x_points_m"], dtype=float
                    ),
                    "efit_label": overlay["efit_label"],
                    "efit_label_authority": str(CORROBORATION_JSON),
                    "nova_class": nova_class,
                    "partition": partition,
                }
            )
    return panels


def _diiid_panel(corroboration, reachability) -> dict[str, Any]:
    """Reconstruct the banked DIII-D physical-ring terminal."""

    receipt = json.loads(DIIID_BANK.read_text())
    banked = receipt["arms"]["physical_ring"]["frame_records"][0]
    shot_name = banked["shot"]
    frame = int(banked["frame"])
    print(f"reconstructing DIII-D {shot_name}/{frame}", flush=True)
    profile, state = reachability._diiid_state(shot_name, frame)
    geometry = reachability._grid_geometry(profile, state)
    _masks, topology = profile.operator.read(state)
    boundary = corroboration._post_cutover_geometry(profile, state, topology)
    partition = _partition(geometry, boundary["binding_flux"])
    source = _wall_topology_row(reachability.DIIID_DATA / shot_name)
    count = int(source["efit_lcfs_n"][frame])
    efit_lcfs = np.c_[
        np.asarray(source["efit_lcfs_r"][frame][:count], dtype=float),
        np.asarray(source["efit_lcfs_z"][frame][:count], dtype=float),
    ]
    efit_x = efit_lcfs[int(np.argmin(efit_lcfs[:, 1]))][None, :]
    achieved = banked["margin_graded"]["terminal_topology_class"]
    if boundary["achieved_class"] != achieved:
        raise RuntimeError("live DIII-D class disagrees with the margin receipt")
    shot = shot_name.removeprefix("d3d_shot_").removesuffix(".parquet")
    return {
        "machine": "DIII-D",
        "identity": f"{shot}/{frame}",
        "arm": "physical ring",
        "radius": geometry["radius"],
        "height": geometry["height"],
        "inside": geometry["inside"],
        "axis": geometry["axis"],
        "selected_saddle": boundary["selected_saddle"],
        "limiter": boundary["limiter_coordinate"],
        "binding_contour": corroboration._binding_contour(
            {
                "radius": geometry["radius"],
                "height": geometry["height"],
                "flux": geometry["flux"],
                "axis": geometry["axis"],
                "boundary_flux": boundary["binding_flux"],
            },
            boundary["selected_saddle"]
            if achieved == "diverted"
            else boundary["limiter_coordinate"],
        ),
        "efit_lcfs": efit_lcfs,
        "efit_x_points": efit_x,
        "efit_label": "diverted",
        "efit_label_authority": (
            "stored EFIT LCFS lower X-point in the banked DIII-D source frame"
        ),
        "nova_class": achieved,
        "partition": partition,
    }


def _draw_panel(axis, panel: dict[str, Any]) -> None:
    """Draw one reachability partition and independent reconstruction overlay."""

    partition = panel["partition"]
    code = np.zeros_like(partition["public"], dtype=int)
    code[partition["public"]] = 1
    code[partition["private"]] = 2
    axis.pcolormesh(
        panel["radius"],
        panel["height"],
        np.ma.masked_where(code == 0, code),
        cmap=ListedColormap(["#bfe3ec", "#e8b97e"]),
        vmin=1,
        vmax=2,
        shading="nearest",
        alpha=0.55,
        rasterized=True,
    )
    contour = np.asarray(panel["binding_contour"])
    axis.plot(contour[:, 0], contour[:, 1], color="#202020", linewidth=1.35)
    efit = np.asarray(panel["efit_lcfs"])
    axis.plot(efit[:, 0], efit[:, 1], color="#087e8b", linewidth=1.45)
    wall = partition["wall"]
    for index, reachable in enumerate(partition["reachable_segments"]):
        following = (index + 1) % len(wall)
        axis.plot(
            wall[[index, following], 0],
            wall[[index, following], 1],
            color="#198754" if reachable else "#b43b3b",
            linewidth=2.15,
            solid_capstyle="butt",
        )
    axis.scatter(*panel["selected_saddle"], marker="X", s=38, color="#512b81", zorder=7)
    for point in panel["efit_x_points"]:
        axis.scatter(
            *point, marker="+", s=50, color="#087e8b", linewidths=1.5, zorder=7
        )
    if np.isfinite(panel["limiter"]).all():
        axis.scatter(*panel["limiter"], marker="D", s=27, color="#f2c14e", zorder=7)
    agreement = panel["nova_class"] == panel["efit_label"]
    axis.set_title(
        f"{panel['panel']}  {panel['machine']} {panel['identity']} {panel['arm']}\n"
        f"achieved {panel['nova_class']} · EFIT {panel['efit_label']} · "
        f"{'AGREE' if agreement else 'DISAGREE'}",
        loc="left",
        fontsize=8.0,
        fontweight="semibold",
    )
    axis.text(
        0.02,
        0.02,
        f"wall {np.count_nonzero(partition['reachable_segments'])} reachable / "
        f"{np.count_nonzero(~partition['reachable_segments'])} shadowed\n"
        f"public {partition['public_area']:.3f} m² · private "
        f"{partition['private_count']} / {partition['private_area']:.3f} m²",
        transform=axis.transAxes,
        fontsize=6.6,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("R [m]")
    axis.set_ylabel("Z [m]")
    axis.spines[["top", "right"]].set_visible(False)


def _record(panel: dict[str, Any]) -> dict[str, Any]:
    """Reduce a plotted panel to its strict-JSON evidence record."""

    partition = panel["partition"]
    reachable = int(np.count_nonzero(partition["reachable_segments"]))
    return {
        "panel": panel["panel"],
        "machine": panel["machine"],
        "shot_slice": panel["identity"],
        "arm": panel["arm"],
        "reachable_wall_segment_count": reachable,
        "shadowed_wall_segment_count": int(
            len(partition["reachable_segments"]) - reachable
        ),
        "wall_segment_count": int(len(partition["reachable_segments"])),
        "public_area_m2": partition["public_area"],
        "private_region_count": partition["private_count"],
        "private_area_m2": partition["private_area"],
        "binding_flux_wb": partition["binding_flux"],
        "binding_level_wb_from_axis_outward": partition["binding_level"],
        "classification_inward_offset_wb": partition["inward_offset"],
        "axis_to_classified_cell_distance_m": partition["axis_to_seed_distance"],
        "nova_achieved_class": panel["nova_class"],
        "efit_label": panel["efit_label"],
        "efit_label_authority": panel["efit_label_authority"],
        "label_agreement": panel["nova_class"] == panel["efit_label"],
        "selected_saddle_m": np.asarray(panel["selected_saddle"]).tolist(),
        "efit_x_points_m": np.asarray(panel["efit_x_points"]).tolist(),
        "reachable_wall_limiter_point_m": (
            np.asarray(panel["limiter"]).tolist()
            if np.isfinite(panel["limiter"]).all()
            else None
        ),
    }


def run() -> dict[str, Any]:
    """Generate the thirteen-panel current reachability evidence artifact."""

    configure_dtypes()
    corroboration = _load_module(CORROBORATION_SCRIPT, "efit_corroboration")
    reachability = corroboration._reachability_module()
    panels = _mast_panels(corroboration, reachability)
    panels.append(_diiid_panel(corroboration, reachability))
    for index, panel in enumerate(panels):
        panel["panel"] = chr(ord("A") + index)

    figure, axes = plt.subplots(7, 2, figsize=(10.6, 27.0), constrained_layout=True)
    axes_flat = axes.ravel()
    for axis, panel in zip(axes_flat, panels, strict=False):
        _draw_panel(axis, panel)
    for axis in axes_flat[len(panels) :]:
        axis.remove()
    figure.legend(
        handles=[
            Line2D([0], [0], color="#202020", lw=2, label="true binding contour"),
            Line2D([0], [0], color="#087e8b", lw=2, label="EFIT efm LCFS"),
            Line2D([0], [0], color="#198754", lw=3, label="reachable wall"),
            Line2D([0], [0], color="#b43b3b", lw=3, label="shadowed wall"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#bfe3ec",
                label="pre-saddle public region",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#e8b97e",
                label="private region",
            ),
            Line2D(
                [0], [0], marker="X", color="#512b81", lw=0, label="selected saddle"
            ),
            Line2D([0], [0], marker="+", color="#087e8b", lw=0, label="EFIT X-point"),
            Line2D(
                [0],
                [0],
                marker="D",
                color="#f2c14e",
                lw=0,
                label="reachable-wall limiter",
            ),
        ],
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    figure.savefig(
        OUTPUT_PNG,
        dpi=190,
        bbox_inches="tight",
        metadata={"Description": f"Project-absolute src: {PROJECT_SRC}"},
    )
    plt.close(figure)

    rows = [_record(panel) for panel in panels]
    agreement = sum(row["label_agreement"] for row in rows)
    payload = {
        "artifact": "post-cutover wall reachability with independent EFIT overlay",
        "project_absolute_src": PROJECT_SRC,
        "method": {
            "binding": "s_star = min(u_wall, u_x)",
            "public_region": (
                "production saddle-aware axis component immediately before binding"
            ),
            "private_region": (
                "positive production component label unequal to the axis label"
            ),
            "height_band_used": False,
            "wall_segment_rule": (
                "segment is reachable only when both endpoint wall nodes touch "
                "the public region"
            ),
            "mast_achieved_class_authority": str(PINNED_BANK),
            "mast_achieved_class_rule": (
                "the banked post-rebaseline class margin, never the legacy "
                "ForwardTopologyState.diverted-derived achieved_class string"
            ),
            "mast_efit_label_authority": str(CORROBORATION_JSON),
            "diiid_efit_label_authority": (
                "stored EFIT LCFS lower X-point in the banked DIII-D source frame"
            ),
        },
        "coverage": {
            "panel_count": len(rows),
            "mast_arm_count": sum(row["machine"] == "MAST" for row in rows),
            "diiid_terminal_count": sum(row["machine"] == "DIII-D" for row in rows),
            "iter_included": False,
            "iter_exclusion_reason": (
                "retired fixed-boundary field with no divertor saddle"
            ),
            "agreement_count": agreement,
            "disagreement_count": len(rows) - agreement,
        },
        "panels": rows,
    }
    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return payload


if __name__ == "__main__":
    print(json.dumps(run()["coverage"], indent=2))
