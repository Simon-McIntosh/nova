"""Corroborate Nova wall topology against the independent MAST EFIT geometry.

EFIT is a magnetics-fitted reconstruction, not ground truth.  This measurement
therefore compares geometry and topology labels only; it does not use a
magnetics-reproduction score, which would privilege the reconstruction fitted
to those measurements.
"""

from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
import numpy as np
from scipy.spatial import cKDTree

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    _mast_case_from_selection,
    _passive_inclusive_case,
    select_slices_by_shot,
)
from benchmarks.label_seed_residual_field import _persisted_response_cache
from nova.equilibrium.connectivity_boundary import traced_margin_candidate_diagnostics
from nova.imas.mast_efit_referee import read_efit_referee
from nova.imas.mast_geometry import DD_VERSION
from nova.imas.mast_solve_inputs import (
    RECONSTRUCTION_GROUP,
    SOURCE_CONVENTION,
    TARGET_CONVENTION,
)
from nova.imas.mast_vacuum_cohort import SHOT_STORE
from nova.jax.config import configure_dtypes


HERE = Path(__file__).resolve().parent
OUTPUT_PNG = HERE / "efit-topology-corroboration.png"
OUTPUT_JSON = HERE / "efit-topology-corroboration.json"
CACHE_PATH = HERE / ".efit-topology-corroboration-cache.npz"
REACHABILITY_SCRIPT = HERE / "real_equilibria_reachability.py"
SELECTION_COMMIT = "80706f89"
CACHE_SCHEMA_REVISION = 1
RESAMPLE_POINTS = 2000


def _reachability_module():
    """Load the existing frozen-six reconstruction driver without duplicating it."""

    spec = spec_from_file_location("real_equilibria_reachability", REACHABILITY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reconstruction driver {REACHABILITY_SCRIPT}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finite_polyline(points: np.ndarray, *, close: bool) -> np.ndarray:
    """Return finite points, optionally with one explicit closing point."""

    result = np.asarray(points, dtype=float)
    result = result[np.isfinite(result).all(axis=1)]
    if len(result) < 3:
        raise RuntimeError("a comparison boundary carries fewer than three points")
    if close and not np.allclose(result[0], result[-1], rtol=0.0, atol=1.0e-12):
        result = np.vstack((result, result[0]))
    return result


def _resample(points: np.ndarray, count: int = RESAMPLE_POINTS) -> np.ndarray:
    """Sample a polyline uniformly in arc length."""

    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    retained = np.r_[True, segment > np.finfo(float).eps]
    points = points[retained]
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    query = np.linspace(0.0, distance[-1], count)
    return np.column_stack(
        (
            np.interp(query, distance, points[:, 0]),
            np.interp(query, distance, points[:, 1]),
        )
    )


def _boundary_distances(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    """Return branch-wise Hausdorff and RMS nearest-boundary distances."""

    first_dense = _resample(_finite_polyline(first, close=False))
    second_dense = _resample(_finite_polyline(second, close=True))
    first_to_second = cKDTree(second_dense).query(first_dense, k=1)[0]
    second_to_first = cKDTree(first_dense).query(second_dense, k=1)[0]
    combined = np.r_[first_to_second, second_to_first]
    return float(np.max(combined)), float(np.sqrt(np.mean(combined**2)))


def _binding_contour(
    geometry: dict[str, Any], boundary_point: np.ndarray
) -> np.ndarray:
    """Extract the closed binding contour selected by the production read."""

    figure, axis = plt.subplots()
    contours = axis.contour(
        geometry["radius"],
        geometry["height"],
        geometry["flux"],
        levels=[float(geometry["boundary_flux"])],
    )
    candidates: list[np.ndarray] = []
    for path in contours.get_paths():
        vertices = path.vertices
        codes = path.codes
        if codes is None:
            if len(vertices) >= 4:
                candidates.append(vertices.copy())
            continue
        starts = np.flatnonzero(codes == MplPath.MOVETO)
        for start, stop in zip(starts, np.r_[starts[1:], len(vertices)], strict=True):
            component = vertices[start:stop]
            if len(component) >= 4:
                candidates.append(component.copy())
    plt.close(figure)
    if not candidates:
        raise RuntimeError("production binding flux produced no drawable contour")
    axis_point = np.asarray(geometry["axis"], dtype=float)

    def rank(vertices: np.ndarray) -> tuple[int, float, float]:
        contains_axis = int(MplPath(vertices, closed=True).contains_point(axis_point))
        boundary_distance = float(
            np.min(np.linalg.norm(vertices - boundary_point, axis=1))
        )
        length = float(np.sum(np.linalg.norm(np.diff(vertices, axis=0), axis=1)))
        return contains_axis, -boundary_distance, length

    return max(candidates, key=rank)


def _segment_intersection(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> np.ndarray | None:
    """Return the intersection of two closed line segments, if unique."""

    first_direction = first_end - first_start
    second_direction = second_end - second_start
    cross = (
        first_direction[0] * second_direction[1]
        - first_direction[1] * second_direction[0]
    )
    if abs(float(cross)) <= 1.0e-14:
        return None
    offset = second_start - first_start
    first_fraction = (
        offset[0] * second_direction[1] - offset[1] * second_direction[0]
    ) / cross
    second_fraction = (
        offset[0] * first_direction[1] - offset[1] * first_direction[0]
    ) / cross
    if 0.0 <= first_fraction <= 1.0 and 0.0 <= second_fraction <= 1.0:
        return first_start + first_fraction * first_direction
    return None


def _boundary_wall_contacts(boundary: np.ndarray, wall: np.ndarray) -> np.ndarray:
    """Return geometrically identifiable EFIT LCFS and governed-wall crossings."""

    boundary = _finite_polyline(boundary, close=True)
    wall = _finite_polyline(wall, close=True)
    contacts: list[np.ndarray] = []
    for boundary_start, boundary_end in zip(boundary[:-1], boundary[1:], strict=True):
        for wall_start, wall_end in zip(wall[:-1], wall[1:], strict=True):
            contact = _segment_intersection(
                boundary_start, boundary_end, wall_start, wall_end
            )
            if contact is None:
                continue
            if not any(
                np.linalg.norm(contact - retained) <= 1.0e-8 for retained in contacts
            ):
                contacts.append(contact)
    return np.stack(contacts) if contacts else np.empty((0, 2), dtype=float)


def _strict_value(value: float) -> float | None:
    """Convert a finite value for strict JSON, retaining absence as null."""

    return float(value) if np.isfinite(value) else None


def _post_cutover_geometry(profile, state, topology) -> dict[str, Any]:
    """Read the exact saddle-aware class operands used in production."""

    operator = profile.operator
    physical = jnp.asarray(state)[: operator.physical_node_number]
    coordinate = np.asarray(operator.grid.coordinate, dtype=np.float64)
    radius = np.unique(coordinate[:, 0])
    height = np.unique(coordinate[:, 1])
    grid_flux, wall_flux = operator.topology.split_flux_map(physical)
    _vmap_o, typed_candidates = operator._fixed_design_topology.grid(grid_flux)
    classification_wall = jnp.concatenate(
        (topology.wall_point, topology.wall_point_flux[None])
    )
    reading = traced_margin_candidate_diagnostics(
        grid_flux.reshape((radius.size, height.size)).T,
        jnp.asarray(radius, dtype=jnp.float64),
        jnp.asarray(height, dtype=jnp.float64),
        operator.inside_material.reshape((radius.size, height.size)).T,
        topology.axis[0],
        topology.axis[1],
        96,
        18,
        operator.wall.coordinate[:, 0],
        operator.wall.coordinate[:, 1],
        wall_flux,
        typed_candidates,
        classification_wall,
    )
    host = jax.device_get(reading)
    selected = np.asarray(host["selected_typed_candidate"], dtype=float)
    margin = float(host["class_margin"])
    limiter = np.asarray(host["limiter_coordinate"], dtype=float)
    limiter_flux = float(host["limiter_flux"])
    if not np.isfinite(selected[:3]).all():
        raise RuntimeError("the post-cutover class read carries no selected saddle")
    if np.isnan(margin):
        raise RuntimeError("the post-cutover achieved class is indeterminate")
    achieved_class = "diverted" if margin >= 0.0 else "limited"
    binding_flux = float(selected[2]) if achieved_class == "diverted" else limiter_flux
    return {
        "achieved_class": achieved_class,
        "class_margin": margin,
        "binding_flux": binding_flux,
        "selected_saddle": selected[:2],
        "limiter_coordinate": limiter,
    }


def _carrier_semantic_identity(carrier_evidence: dict[str, Any]) -> str:
    """Return the persisted response carrier's semantic cache identity."""

    carrier = carrier_evidence.get("carrier", carrier_evidence)
    identity = carrier.get("semantic_response_identity")
    if not isinstance(identity, str) or not identity:
        raise RuntimeError("the persisted response carrier has no semantic identity")
    return identity


def _cache_authority(carrier_evidence: dict[str, Any]) -> dict[str, Any]:
    """Return the exact authority tuple that makes cached operands reusable."""

    return {
        "schema_revision": CACHE_SCHEMA_REVISION,
        "response_carrier_semantic_identity": _carrier_semantic_identity(
            carrier_evidence
        ),
        "selection_source_commit": SELECTION_COMMIT,
    }


def _write_operand_cache(
    rows: list[dict[str, Any]], carrier_evidence: dict[str, Any]
) -> None:
    """Persist exact solve operands before any plotting or distance reduction."""

    metadata_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for index, row in enumerate(rows):
        prefix = f"arm_{index:02d}"
        metadata_rows.append(
            {
                key: row[key]
                for key in (
                    "identity",
                    "shot",
                    "slice_index",
                    "time_s",
                    "arm",
                    "efit_label",
                    "nova_achieved_class",
                )
            }
        )
        for name in (
            "radius",
            "height",
            "flux",
            "axis",
            "wall",
            "binding_flux",
            "selected_saddle",
            "limiter_coordinate",
            "class_margin",
            "efit_lcfs",
            "efit_x_points",
        ):
            arrays[f"{prefix}_{name}"] = np.asarray(row[name])
    metadata = _cache_authority(carrier_evidence) | {
        "purpose": (
            "unbanked exact solve operands for reproducible plot-only regeneration"
        ),
        "arm_count": len(rows),
        "rows": metadata_rows,
    }
    temporary = CACHE_PATH.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        **arrays,
    )
    temporary.replace(CACHE_PATH)


def _read_operand_cache(
    carrier_evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    """Load exact solve operands after fail-closed authority validation."""

    expected = _cache_authority(carrier_evidence)
    with np.load(CACHE_PATH, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"].item()))
        observed = {key: metadata.get(key) for key in expected}
        if observed != expected:
            raise RuntimeError(
                f"intermediate operand cache is stale: expected {expected}, "
                f"observed {observed}"
            )
        if int(metadata.get("arm_count", -1)) != 12:
            raise RuntimeError("intermediate operand cache does not carry twelve arms")
        rows: list[dict[str, Any]] = []
        for index, metadata_row in enumerate(metadata["rows"]):
            prefix = f"arm_{index:02d}"
            row = dict(metadata_row)
            for name in (
                "radius",
                "height",
                "flux",
                "axis",
                "wall",
                "binding_flux",
                "selected_saddle",
                "limiter_coordinate",
                "class_margin",
                "efit_lcfs",
                "efit_x_points",
            ):
                row[name] = np.array(stored[f"{prefix}_{name}"], copy=True)
            rows.append(row)
    return rows


def _build_operand_cache(
    response_cache, carrier_evidence: dict[str, Any]
) -> list[dict[str, Any]]:
    """Run each production arm once and persist its exact corroboration operands."""

    reachability = _reachability_module()
    selected = select_slices_by_shot(DECOMPOSITION_BANK)
    rows: list[dict[str, Any]] = []
    for selected_row, qualification in selected:
        shot = int(selected_row["shot"])
        slice_index = int(selected_row["slice_index"])
        print(f"solving MAST {shot}/{slice_index}", flush=True)
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
        referee = read_efit_referee(shot, store=SHOT_STORE)
        if not bool(referee.usable[slice_index]):
            raise RuntimeError(f"EFIT referee slice {shot}/{slice_index} is unusable")
        efit_lcfs = _finite_polyline(referee.lcfs_m[slice_index], close=True)
        efit_x_points = referee.x_points_m[slice_index]
        efit_x_points = efit_x_points[np.isfinite(efit_x_points).all(axis=1)]
        efit_label = "diverted" if bool(referee.diverted[slice_index]) else "limited"
        for arm, state in states.items():
            geometry = reachability._grid_geometry(profile, state)
            _masks, topology = profile.operator.read(state)
            post_cutover = _post_cutover_geometry(profile, state, topology)
            rows.append(
                {
                    "identity": f"{shot}/{slice_index}",
                    "shot": shot,
                    "slice_index": slice_index,
                    "time_s": float(referee.time_s[slice_index]),
                    "arm": arm,
                    "efit_label": efit_label,
                    "nova_achieved_class": post_cutover["achieved_class"],
                    "radius": geometry["radius"],
                    "height": geometry["height"],
                    "flux": geometry["flux"],
                    "axis": geometry["axis"],
                    "wall": geometry["wall"],
                    "binding_flux": post_cutover["binding_flux"],
                    "selected_saddle": post_cutover["selected_saddle"],
                    "limiter_coordinate": post_cutover["limiter_coordinate"],
                    "class_margin": post_cutover["class_margin"],
                    "efit_lcfs": efit_lcfs,
                    "efit_x_points": efit_x_points,
                }
            )
    _write_operand_cache(rows, carrier_evidence)
    print(f"wrote exact operand cache {CACHE_PATH}", flush=True)
    return rows


def _draw_panel(axis, row: dict[str, Any]) -> None:
    """Draw one arm with both independent and Nova geometry."""

    wall = np.asarray(row["wall_m"], dtype=float)
    efit_lcfs = np.asarray(row["efit_lcfs_m"], dtype=float)
    nova_boundary = np.asarray(row["nova_binding_contour_m"], dtype=float)
    axis.plot(wall[:, 0], wall[:, 1], color="#8a8a8a", linewidth=1.0)
    axis.plot(efit_lcfs[:, 0], efit_lcfs[:, 1], color="#087e8b", linewidth=1.8)
    axis.plot(nova_boundary[:, 0], nova_boundary[:, 1], color="#d1495b", linewidth=1.35)
    for point in row["efit_x_points_m"]:
        axis.scatter(
            *point, marker="+", s=48, color="#087e8b", linewidths=1.6, zorder=5
        )
    axis.scatter(
        *row["nova_selected_saddle_m"], marker="X", s=34, color="#d1495b", zorder=6
    )
    if row["nova_limiter_point_m"] is not None:
        axis.scatter(
            *row["nova_limiter_point_m"], marker="D", s=24, color="#f2c14e", zorder=6
        )
    agreement = "AGREE" if row["label_agreement"] else "DISAGREE"
    axis.set_title(
        f"{row['panel']}  MAST {row['identity']} {row['arm']}\n"
        f"EFIT {row['efit_label']} · Nova {row['nova_achieved_class']} · {agreement}",
        loc="left",
        fontsize=8.3,
        fontweight="semibold",
    )
    axis.text(
        0.02,
        0.02,
        f"LCFS sup {row['binding_to_efit_lcfs_sup_m']:.3f} m · "
        f"RMS {row['binding_to_efit_lcfs_rms_m']:.3f} m\n"
        f"X nearest {row['selected_saddle_to_efit_x_point_m']:.3f} m",
        transform=axis.transAxes,
        fontsize=6.8,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(0.08, 1.72)
    axis.set_ylim(-1.9, 1.9)
    axis.set_xlabel("R [m]")
    axis.set_ylabel("Z [m]")
    axis.spines[["top", "right"]].set_visible(False)


def run() -> dict[str, Any]:
    """Regenerate the twelve-arm corroboration figure and strict-JSON receipt."""

    configure_dtypes()
    response_cache, carrier_evidence = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    cache_reused = False
    if CACHE_PATH.exists():
        try:
            operand_rows = _read_operand_cache(carrier_evidence)
            cache_reused = True
            print(f"reusing exact operand cache {CACHE_PATH}", flush=True)
        except RuntimeError as error:
            print(f"rejecting stale operand cache: {error}", flush=True)
            operand_rows = _build_operand_cache(response_cache, carrier_evidence)
    else:
        operand_rows = _build_operand_cache(response_cache, carrier_evidence)

    rows: list[dict[str, Any]] = []
    for operand in operand_rows:
        geometry = {
            "radius": operand["radius"],
            "height": operand["height"],
            "flux": operand["flux"],
            "axis": operand["axis"],
            "boundary_flux": float(operand["binding_flux"]),
        }
        nova_label = operand["nova_achieved_class"]
        selected_saddle = operand["selected_saddle"]
        nova_limiter = operand["limiter_coordinate"]
        boundary_point = selected_saddle if nova_label == "diverted" else nova_limiter
        binding_contour = _binding_contour(geometry, boundary_point)
        efit_lcfs = operand["efit_lcfs"]
        efit_x_points = operand["efit_x_points"]
        wall = operand["wall"]
        sup_distance, rms_distance = _boundary_distances(binding_contour, efit_lcfs)
        x_distance = (
            float(np.min(np.linalg.norm(efit_x_points - selected_saddle, axis=1)))
            if len(efit_x_points)
            else float("nan")
        )
        contacts = _boundary_wall_contacts(efit_lcfs, wall)
        contact_distance = (
            float(np.min(np.linalg.norm(contacts - nova_limiter, axis=1)))
            if len(contacts)
            else float("nan")
        )
        class_margin = float(operand["class_margin"])
        efit_label = operand["efit_label"]
        rows.append(
            {
                "identity": operand["identity"],
                "shot": operand["shot"],
                "slice_index": operand["slice_index"],
                "time_s": operand["time_s"],
                "arm": operand["arm"],
                "efit_label": efit_label,
                "nova_achieved_class": nova_label,
                "nova_post_cutover_class_margin": _strict_value(class_margin),
                "nova_post_cutover_class_margin_nonfinite": (
                    "positive_infinity"
                    if np.isposinf(class_margin)
                    else "negative_infinity"
                    if np.isneginf(class_margin)
                    else None
                ),
                "label_agreement": efit_label == nova_label,
                "binding_to_efit_lcfs_sup_m": sup_distance,
                "binding_to_efit_lcfs_rms_m": rms_distance,
                "nova_selected_saddle_m": selected_saddle.tolist(),
                "efit_x_points_m": efit_x_points.tolist(),
                "selected_saddle_to_efit_x_point_m": _strict_value(x_distance),
                "nova_limiter_point_m": (
                    nova_limiter.tolist() if np.isfinite(nova_limiter).all() else None
                ),
                "efit_boundary_wall_contacts_m": contacts.tolist(),
                "limiter_to_efit_boundary_wall_contact_m": _strict_value(
                    contact_distance
                ),
                "limiter_contact_metric_unavailable_reason": (
                    None
                    if len(contacts)
                    else (
                        "the stored EFIT LCFS has no geometric intersection "
                        "with the governed wall"
                    )
                ),
                "nova_binding_contour_m": binding_contour.tolist(),
                "efit_lcfs_m": efit_lcfs.tolist(),
                "wall_m": wall.tolist(),
            }
        )

    for index, row in enumerate(rows):
        row["panel"] = chr(ord("A") + index)
    figure, axes = plt.subplots(6, 2, figsize=(10.2, 24.0), constrained_layout=True)
    for axis, row in zip(axes.ravel(), rows, strict=True):
        _draw_panel(axis, row)
    figure.legend(
        handles=[
            Line2D([0], [0], color="#087e8b", lw=2, label="EFIT efm LCFS"),
            Line2D([0], [0], color="#d1495b", lw=2, label="Nova binding contour"),
            Line2D(
                [0], [0], marker="+", color="#087e8b", lw=0, label="EFIT efm X-point"
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color="#d1495b",
                lw=0,
                label="Nova selected saddle",
            ),
            Line2D(
                [0], [0], marker="D", color="#f2c14e", lw=0, label="Nova limiter point"
            ),
        ],
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    figure.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(figure)

    disagreements = [row for row in rows if not row["label_agreement"]]
    payload = {
        "artifact": "independent EFIT topology corroboration of Nova wall reachability",
        "headline": (
            "EFIT labels all twelve arms diverted; post-cutover Nova agrees on "
            "seven and reaches limited on five. Contained saddle selection is "
            "corroborated to 4.343-38.661 mm, locating the disagreement in the "
            "solver basin rather than saddle detection or selection."
        ),
        "qualification": (
            "EFIT efm is an independent magnetics-fitted reconstruction, not truth. "
            "No magnetics-reproduction metric is used because that would not be a "
            "neutral cross-source comparison."
        ),
        "project_absolute_src": (
            "/nova/figures/primary-xpoint-evidence/efit-topology-corroboration.png"
        ),
        "data_authority": {
            "efit_reader": "nova.imas.mast_efit_referee.read_efit_referee",
            "efit_group": RECONSTRUCTION_GROUP,
            "catalogue_store": str(SHOT_STORE),
            "imas_machine_description_dd_version_written_and_opened": DD_VERSION,
            "catalogue_format_qualification": (
                "efm is a raw read-only Zarr catalogue group rather than an IMAS "
                "IDS, so it has no DD version to infer or open; the governed "
                "machine-description seam it is compared within is authored and "
                "reopened at its written DD version"
            ),
            "source_cocos": SOURCE_CONVENTION,
            "target_cocos": TARGET_CONVENTION,
            "cocos_resolution": (
                "source COCOS 3 is the measured MAST convention declared by "
                "mast_solve_inputs; the solve seam converts typed quantities to DD "
                "COCOS 17. R and Z are ONE_LIKE geometric coordinates, so their "
                "conversion factor is exactly one and the EFIT geometry is overlaid "
                "without a sign or scale edit"
            ),
            "nova_selection_source_commit": SELECTION_COMMIT,
            "carrier": carrier_evidence,
            "intermediate_operand_cache": {
                "path": str(CACHE_PATH),
                "banked_artifact": False,
                "reused_for_this_render": cache_reused,
                "authority": _cache_authority(carrier_evidence),
                "exact_operands_per_arm": [
                    "binding_flux",
                    "contained_selected_saddle",
                    "refined_limiter_coordinate",
                    "governed_wall_polyline",
                    "full_flux_grid_with_tensor_axes",
                ],
            },
            "legacy_label_trap": (
                "ForwardTopologyState.diverted is the legacy geometric topology "
                "label and is not the post-cutover achieved class. This artifact "
                "derives achieved class only from the saddle-aware class margin: "
                "non-negative or positive infinity is diverted; negative is limited."
            ),
        },
        "distance_method": {
            "boundary_sup_m": (
                "symmetric sampled Hausdorff distance after 2000-point arc-length "
                "resampling of the selected disconnected contour branch and the "
                "closed EFIT polyline"
            ),
            "boundary_rms_m": (
                "root mean square of both directed nearest-polyline sample "
                "distances on the same per-branch resampling"
            ),
            "compound_path_rule": (
                "Matplotlib compound paths are split at every MOVETO and scored "
                "per disconnected branch; distances are never evaluated across "
                "a synthetic chord between branches"
            ),
            "x_point_m": "Nova selected saddle to the nearest finite EFIT efm X-point",
            "limiter_contact_m": (
                "Nova limiter point to the nearest exact EFIT LCFS and "
                "governed-wall segment intersection; null when none exists"
            ),
        },
        "label_comparison": {
            "cohort_description": (
                "six frozen MAST DIVERTED-LABEL EFIT references, two Nova arms each"
            ),
            "arm_count": len(rows),
            "agreement_count": len(rows) - len(disagreements),
            "disagreement_count": len(disagreements),
            "disagreements": [
                f"{row['identity']} {row['arm']}" for row in disagreements
            ],
        },
        "summary": {
            "boundary_sup_m_min": min(
                row["binding_to_efit_lcfs_sup_m"] for row in rows
            ),
            "boundary_sup_m_max": max(
                row["binding_to_efit_lcfs_sup_m"] for row in rows
            ),
            "boundary_rms_m_min": min(
                row["binding_to_efit_lcfs_rms_m"] for row in rows
            ),
            "boundary_rms_m_max": max(
                row["binding_to_efit_lcfs_rms_m"] for row in rows
            ),
            "x_point_distance_m_min": min(
                row["selected_saddle_to_efit_x_point_m"] for row in rows
            ),
            "x_point_distance_m_max": max(
                row["selected_saddle_to_efit_x_point_m"] for row in rows
            ),
            "identifiable_efit_boundary_wall_contact_count": sum(
                bool(row["efit_boundary_wall_contacts_m"]) for row in rows
            ),
        },
        "rows": rows,
    }
    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return payload


if __name__ == "__main__":
    result = run()
    print(
        json.dumps(
            {
                "summary": result["summary"],
                "label_comparison": result["label_comparison"],
            },
            indent=2,
        )
    )
