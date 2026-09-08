"""Decompose the seed-centroid radial field by MAST current family.

The decomposition uses the stored EFIT current vector exactly once.  It groups
its shaped sections into the thirteen active circuits, the eight coil-case
groups with direct current transducers, and every remaining passive or vessel
section.  The plasma term is the native EFIT current density integrated through
the same shaped-section Green kernel at the stored current centroid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import shapely
import zarr

from benchmarks import settled_mask_stall as settled
from benchmarks.efit_flux_decomposition import _density_from_flux
from benchmarks.efit_forward_parity_slice import _circuit_drives
from benchmarks.efit_native_grid_decomposition import _uniform_axis
from benchmarks.efit_topology_boundary_score import _live_flux_map, _stored_lcfs
from nova.biot.polygon import polygon_greens
from nova.catalog.mast_geometry import MachineGeometryRegistry, shaped_section_vertices
from nova.equilibrium.constraint import (
    CircuitCurrentUnknown,
    ConstraintBinding,
    ConstraintPair,
    CurrentCentroidConstraint,
)
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
from nova.equilibrium.observation import MomentIntegralSupport
from nova.equilibrium.solve_request import ExplicitSolveSeed, ForwardSolveRequest
from nova.equilibrium.wall_mask import inside_polygon
from nova.imas.mast_channel_drive import case_plate_channels
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.imas.mast_vacuum_cohort import CASE_CURRENT_CHANNELS
from nova.imas.mast_passive_response import passive_sections
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)


matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / (
    "docs/figures/solver-convergence-regression/vertical-mode/force-decomposition"
)
ROWS = ((21986, 46), (21989, 55))
SCAN_HEIGHTS_M = (0.01889428493417936, 0.13889428493417936)
POSITION_OFFSET_M = {
    "p4_lower": 0.006638,
    "p4_upper": 0.002941,
    "p5_lower": 0.013070,
    "p5_upper": 0.000562,
}
COMPENSATION_REFERENCE_A = {"21989/55": 3022.326}
CASE_CURRENT_REPAIR_OUTPUT = ROOT / (
    "docs/figures/solver-convergence-regression/vertical-mode/case-current-repair"
)
ELIMINATION_HEIGHTS_M = (
    0.03889428493417936,
    0.05889428493417936,
    0.07889428493417936,
    0.09889428493417936,
    0.11889428493417936,
)


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _strict_float(value: Any) -> float | None:
    result = float(np.asarray(value))
    return result if np.isfinite(result) else None


def _efm_group(shot: int) -> zarr.Group:
    return zarr.open_group(str(Path(SHOT_STORE) / f"{shot}.zarr" / "efm"), mode="r")


def _centroid(group: zarr.Group, row: int) -> tuple[float, float]:
    radius = float(group["current_centrd_r"][row])
    height = float(group["current_centrd_z"][row])
    if not np.isfinite(radius) or not np.isfinite(height):
        raise ValueError("EFIT current centroid is not finite")
    return radius, height


def _seed_current_centroid(shot: int, row: int) -> tuple[float, float]:
    """Evaluate the seed current centroid through the public moment observation."""

    selected = {
        (int(item["shot"]), int(item["slice_index"])): (item, qualification)
        for item, qualification in settled.select_slices_by_shot(
            settled.DECOMPOSITION_BANK
        )
    }
    selected_row, qualification = selected[(shot, row)]
    case, context = settled._mast_case_from_selection(
        settled.SHOT_STORE, selected_row, qualification
    )
    target_current = abs(float(case["reference"]["plasma_current_a"]))
    observation = context["profile"].current_moment_observation(
        jnp.asarray(case["state"]),
        support=MomentIntegralSupport.ALL_DOMAIN,
        target_current=target_current,
    )
    return float(observation.centroid_r), float(observation.centroid_z)


def _element_data(group: zarr.Group) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(group[name], dtype=np.float64)
        for name in (
            "fcoil_circ",
            "fcoil_r",
            "fcoil_z",
            "fcoil_width",
            "fcoil_height",
            "fcoil_ang1",
            "fcoil_ang2",
            "fcoil_turns",
            "fcoil_xmult",
        )
    }


def _element_vertices(elements: dict[str, np.ndarray], index: int) -> np.ndarray:
    return shaped_section_vertices(
        elements["fcoil_r"][index],
        elements["fcoil_z"][index],
        elements["fcoil_width"][index],
        elements["fcoil_height"][index],
        elements["fcoil_ang1"][index],
        elements["fcoil_ang2"][index],
    )


def _element_br_per_ampere(
    elements: dict[str, np.ndarray], index: int, target: tuple[float, float]
) -> float:
    vertices = _element_vertices(elements, index)
    _psi, radial, _vertical = polygon_greens(target[0], target[1], vertices)
    return (
        float(radial)
        * float(elements["fcoil_turns"][index])
        * float(elements["fcoil_xmult"][index])
    )


def _offset_element_br_per_ampere(
    elements: dict[str, np.ndarray],
    index: int,
    target: tuple[float, float],
    radial_offset_m: float,
    vertical_offset_m: float,
) -> float:
    vertices = _element_vertices(elements, index) + np.asarray(
        [radial_offset_m, vertical_offset_m]
    )
    _psi, radial, _vertical = polygon_greens(target[0], target[1], vertices)
    return (
        float(radial)
        * float(elements["fcoil_turns"][index])
        * float(elements["fcoil_xmult"][index])
    )


def _case_element_groups(
    elements: dict[str, np.ndarray],
    geometry: dict[str, Any],
    active_circuits: set[int],
) -> dict[int, str]:
    """Map stored shaped sections to the eight independently measured case groups."""

    channels_to_plates = case_plate_channels(geometry)
    inverse = {channel: family for family, channel in CASE_CURRENT_CHANNELS.items()}
    plates = passive_sections(geometry).get("coil_cases")
    if plates is None:
        raise ValueError("the machine geometry has no coil-case plates")
    group_polygons = {
        inverse[channel]: [shapely.Polygon(plates[index]) for index in indices]
        for channel, indices in channels_to_plates.items()
    }
    groups: dict[int, str] = {}
    for index in range(len(elements["fcoil_circ"])):
        if int(elements["fcoil_circ"][index]) in active_circuits:
            continue
        element = shapely.Polygon(_element_vertices(elements, index))
        overlap = {
            family: max(
                (element.intersection(plate).area for plate in polygons), default=0.0
            )
            for family, polygons in group_polygons.items()
        }
        family = max(overlap, key=overlap.get)
        if overlap[family] > 0.0:
            groups[index] = family
    missing = sorted(set(CASE_CURRENT_CHANNELS) - set(groups.values()))
    if missing:
        raise ValueError(f"no stored case sections overlap groups {missing}")
    return groups


def _interpolated_case_currents(
    shot: int, time_s: float
) -> dict[str, dict[str, float]]:
    root = zarr.open_group(str(Path(SHOT_STORE) / f"{shot}.zarr"), mode="r")
    current = root["amc"]
    source_time = np.asarray(current["time"], dtype=np.float64)
    if np.any(np.diff(source_time) <= 0.0):
        raise ValueError("the current clock must be strictly increasing")
    values: dict[str, dict[str, float]] = {}
    for family, channel in CASE_CURRENT_CHANNELS.items():
        trace = np.asarray(current[channel], dtype=np.float64) * 1.0e3
        if time_s < source_time[0] or time_s > source_time[-1]:
            raise ValueError(f"{channel} does not cover the EFIT slice time")
        finite = np.isfinite(trace)
        if not np.any(finite):
            raise ValueError(f"{channel} has no finite current samples")
        candidates = np.flatnonzero(finite)
        index = int(candidates[np.argmin(np.abs(source_time[finite] - time_s))])
        values[family] = {
            "channel": channel,
            "time_s": float(time_s),
            "sample_time_s": float(source_time[index]),
            "sample_time_offset_s": float(source_time[index] - time_s),
            "selection": "nearest finite amc sample to the EFIT slice time",
            "current_a": float(trace[index]),
        }
    return values


def _plasma_br(
    group: zarr.Group, row: int, target: tuple[float, float]
) -> tuple[float, dict[str, Any]]:
    """Integrate the seed current density on its native 65-point lattice."""

    radius = _uniform_axis(np.asarray(group["gridr"], dtype=np.float64), "gridr")
    height = _uniform_axis(np.asarray(group["gridz"], dtype=np.float64), "gridz")
    lattice = FluxLattice(radius, height)
    total_flux = TOTAL_FLUX_FACTOR * _live_flux_map(group, row, len(radius))
    density, valid = _density_from_flux(lattice, total_flux)
    lcfs = _stored_lcfs(group, row)
    coordinate = lattice.coordinate
    supported = np.asarray(
        inside_polygon(coordinate[:, 0], coordinate[:, 1], lcfs[:, 0], lcfs[:, 1]),
        dtype=bool,
    ).reshape(lattice.shape)
    admitted = valid & supported & np.isfinite(density)
    current = density * lattice.radial_step * lattice.vertical_step
    radial = 0.0
    for (ir, iz), value in np.ndenumerate(current):
        if not admitted[ir, iz] or value == 0.0:
            continue
        vertices = np.asarray(
            [
                [
                    radius[ir] - lattice.radial_step / 2,
                    height[iz] - lattice.vertical_step / 2,
                ],
                [
                    radius[ir] + lattice.radial_step / 2,
                    height[iz] - lattice.vertical_step / 2,
                ],
                [
                    radius[ir] + lattice.radial_step / 2,
                    height[iz] + lattice.vertical_step / 2,
                ],
                [
                    radius[ir] - lattice.radial_step / 2,
                    height[iz] + lattice.vertical_step / 2,
                ],
            ],
            dtype=np.float64,
        )
        _psi, response, _vertical = polygon_greens(target[0], target[1], vertices)
        radial += float(value) * float(response)
    return radial, {
        "source": (
            "native efm current density through delta-star and shaped cell sections"
        ),
        "native_lattice_shape_rz": list(lattice.shape),
        "admitted_cell_count": int(np.count_nonzero(admitted)),
        "admitted_current_a": float(np.sum(np.where(admitted, current, 0.0))),
    }


def _p6_response(
    elements: dict[str, np.ndarray], active: dict[int, str], target: tuple[float, float]
) -> float:
    inverse = {family: circuit for circuit, family in active.items()}
    upper = inverse["p6_upper"]
    lower = inverse["p6_lower"]
    response = 0.0
    for index, circuit in enumerate(elements["fcoil_circ"]):
        if int(circuit) == upper:
            response += _element_br_per_ampere(elements, index, target)
        elif int(circuit) == lower:
            response -= _element_br_per_ampere(elements, index, target)
    if abs(response) < np.finfo(np.float64).tiny:
        raise ValueError("P6 radial response is zero")
    return response


def _circuit_br_per_ampere(
    elements: dict[str, np.ndarray], circuit: int, target: tuple[float, float]
) -> float:
    return sum(
        _element_br_per_ampere(elements, int(index), target)
        for index in np.flatnonzero(elements["fcoil_circ"] == circuit)
    )


def _displaced_circuit_field(
    elements: dict[str, np.ndarray],
    circuit: int,
    current_a: float,
    target: tuple[float, float],
    radial_offset_m: float,
    vertical_offset_m: float,
) -> float:
    return current_a * sum(
        _offset_element_br_per_ampere(
            elements,
            int(index),
            target,
            radial_offset_m,
            vertical_offset_m,
        )
        for index in np.flatnonzero(elements["fcoil_circ"] == circuit)
    )


def _position_sensitivity(
    elements: dict[str, np.ndarray],
    circuit: int,
    current_a: float,
    target: tuple[float, float],
    p6_response_t_per_a: float,
    radial: bool,
) -> float:
    offset_m = 0.001
    positive = _displaced_circuit_field(
        elements,
        circuit,
        current_a,
        target,
        offset_m if radial else 0.0,
        0.0 if radial else offset_m,
    )
    negative = _displaced_circuit_field(
        elements,
        circuit,
        current_a,
        target,
        -offset_m if radial else 0.0,
        0.0 if radial else -offset_m,
    )
    return (positive - negative) / (2.0 * p6_response_t_per_a)


def _sensitivity_row(shot: int, row: int) -> dict[str, Any]:
    group = _efm_group(shot)
    target = _seed_current_centroid(shot, row)
    geometry = MachineGeometryRegistry.default().select(shot).configuration.geometry
    _families, _drives, mapping = _circuit_drives(group, row, geometry, "fcoil_c")
    active = {int(item["stored_circuit"]): str(item["family"]) for item in mapping}
    elements = _element_data(group)
    fitted = np.asarray(group["fcoil_c"][row], dtype=np.float64)
    p6 = _p6_response(elements, active, target)
    active_rows = []
    for circuit, family in sorted(active.items()):
        response = _circuit_br_per_ampere(elements, circuit, target) / p6
        active_rows.append(
            {
                "family": family,
                "stored_circuit": circuit,
                "fitted_current_a": float(fitted[circuit - 1]),
                "efm_current_uncertainty_a": None,
                "p6_ampere_equivalent_per_ampere": response,
                "p6_ampere_equivalent_per_uncertainty": None,
                "efm_fit_chi_squared": _strict_float(
                    group["fcoil_chisq"][row, circuit - 1]
                ),
                "efm_fit_weight": _strict_float(group["fwtfc"][row, circuit - 1]),
            }
        )
    positions = []
    inverse = {family: circuit for circuit, family in active.items()}
    for family, vertical_offset_m in POSITION_OFFSET_M.items():
        circuit = inverse[family]
        current_a = float(fitted[circuit - 1])
        radial_per_mm = _position_sensitivity(
            elements, circuit, current_a, target, p6, radial=True
        )
        vertical_per_mm = _position_sensitivity(
            elements, circuit, current_a, target, p6, radial=False
        )
        positions.append(
            {
                "family": family,
                "stored_circuit": circuit,
                "fitted_current_a": current_a,
                "p6_ampere_equivalent_per_mm_radial": radial_per_mm,
                "p6_ampere_equivalent_per_mm_vertical": vertical_per_mm,
                "radial_position_tolerance_mm": None,
                "vertical_position_tolerance_mm": vertical_offset_m * 1.0e3,
                "vertical_p6_ampere_equivalent_at_tolerance": vertical_per_mm
                * vertical_offset_m
                * 1.0e3,
            }
        )
    identity = f"{shot}/{row}"
    compensation = COMPENSATION_REFERENCE_A.get(identity)
    ranked = []
    if compensation is not None:
        for item in positions:
            available = abs(item["vertical_p6_ampere_equivalent_at_tolerance"])
            ranked.append(
                {
                    "perturbation": f"{item['family']} vertical position",
                    "available_p6_adjustment_a": available,
                    "closure_fraction": min(available / compensation, 1.0),
                    "closes_reference": available >= compensation,
                }
            )
        ranked.sort(key=lambda item: item["available_p6_adjustment_a"], reverse=True)
    return {
        "identity": identity,
        "current_centroid": {"r_m": target[0], "z_m": target[1]},
        "p6_radial_response_t_per_a": p6,
        "p6_definition": "stored P6 upper current minus stored P6 lower current",
        "current_uncertainty": {
            "availability": "not carried by the EFM store",
            "reason": (
                "fcoil_chisq is a fit diagnostic and fwtfc is a fit weight; neither "
                "declares a current standard error"
            ),
        },
        "active_circuit_sensitivities": active_rows,
        "coil_position_sensitivities": positions,
        "compensation_reference_a": compensation,
        "ranked_available_perturbations": ranked,
    }


def _draw_sensitivity(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 7))
    for row in rows:
        values = row["active_circuit_sensitivities"]
        labels = [item["family"] for item in values]
        response = [item["p6_ampere_equivalent_per_ampere"] for item in values]
        axes[0].plot(response, labels, "o", label=row["identity"])
    axes[0].axvline(0.0, color="0.3", linewidth=0.8)
    axes[0].set_xlabel("P6 ampere-equivalent per circuit ampere [A/A]")
    axes[0].set_title("Active-current response")
    axes[0].grid(axis="x", alpha=0.2)
    axes[0].legend(frameon=False)
    measured = next(row for row in rows if row["identity"] == "21989/55")
    positions = measured["coil_position_sensitivities"]
    labels = [item["family"] for item in positions]
    available = [
        abs(item["vertical_p6_ampere_equivalent_at_tolerance"]) / 1.0e3
        for item in positions
    ]
    axes[1].bar(labels, available)
    axes[1].axhline(3.0, color="0.3", linewidth=0.8, label="3 kA reference")
    axes[1].set_ylabel("Available vertical-position adjustment [kA]")
    axes[1].set_title("Published copper-offset interval")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False)
    figure.suptitle("Seed-centroid radial-field sensitivity")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def measure_sensitivity(output: Path) -> dict[str, Any]:
    configure_dtypes()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    output.mkdir(parents=True, exist_ok=True)
    rows = [_sensitivity_row(shot, row) for shot, row in ROWS]
    measured = next(row for row in rows if row["identity"] == "21989/55")
    ranked = measured["ranked_available_perturbations"]
    receipt = {
        "receipt": "seed-centroid radial-field sensitivity",
        "source": {
            "revision": _source_revision(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
        },
        "configuration": {
            "field": "B_R at the public seed current centroid",
            "p6_equivalent": (
                "B_R divided by the P6 upper-minus-lower unit response at that centroid"
            ),
            "current_uncertainty": (
                "reported only when the store declares a current standard error"
            ),
            "position_derivative": "symmetric one-millimetre finite difference",
            "vertical_position_interval": (
                "published Hall-probe copper z-offset magnitude; no radial interval is "
                "available from that source"
            ),
            "persistent_compilation_cache": {
                "directory": str(cache.directory),
                "version": cache.version_key,
            },
        },
        "rows": rows,
    }
    receipt["verdict"] = {
        "reference_compensation_a": measured["compensation_reference_a"],
        "leading_available_perturbation": ranked[0] if ranked else None,
        "single_perturbation_closes_reference": any(
            item["closes_reference"] for item in ranked
        ),
    }
    (output / "sensitivity-21989.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    _draw_sensitivity(rows, output / "sensitivity-21989.png")
    return receipt


def _decompose_row(shot: int, row: int) -> dict[str, Any]:
    group = _efm_group(shot)
    archived_centroid = _centroid(group, row)
    target = _seed_current_centroid(shot, row)
    time_s = float(group["time"][row])
    geometry = MachineGeometryRegistry.default().select(shot).configuration.geometry
    _families, _drives, mapping = _circuit_drives(group, row, geometry, "fcoil_c")
    active = {int(item["stored_circuit"]): str(item["family"]) for item in mapping}
    elements = _element_data(group)
    fitted = np.asarray(group["fcoil_c"][row], dtype=np.float64)
    indices = np.asarray(group["fcoil_n"], dtype=int)
    if not np.array_equal(indices, np.arange(fitted.size)):
        raise ValueError("fcoil_n does not use zero-based stored-current order")
    case_groups = _case_element_groups(elements, geometry, set(active))
    transducers = _interpolated_case_currents(shot, time_s)
    p6 = _p6_response(elements, active, target)
    plasma_key = "plasma::self"
    component_fields: dict[str, float] = {plasma_key: 0.0}
    replacement_fields: dict[str, float] = {plasma_key: 0.0}
    component_meta: dict[str, dict[str, Any]] = {}
    plasma, plasma_meta = _plasma_br(group, row, target)
    component_fields[plasma_key] = plasma
    replacement_fields[plasma_key] = plasma
    component_meta[plasma_key] = {
        "category": "plasma",
        "family": "plasma_self",
        **plasma_meta,
    }
    direct_fitted = plasma
    direct_replacement = plasma
    for circuit in range(1, fitted.size + 1):
        active_family = active.get(circuit)
        for index in np.flatnonzero(elements["fcoil_circ"] == circuit):
            case_family = case_groups.get(int(index))
            if active_family is not None:
                key = f"active::{active_family}"
                family = active_family
                category = "active_circuit"
            elif case_family is not None:
                key = f"case::{case_family}"
                family = case_family
                category = "instrumented_coil_case"
            else:
                key = "passive::vessel_remaining"
                family = "passive_vessel_remaining"
                category = "passive_or_vessel"
            field = _element_br_per_ampere(elements, int(index), target)
            fitted_field = fitted[circuit - 1] * field
            component_fields[key] = component_fields.get(key, 0.0) + fitted_field
            if category == "instrumented_coil_case":
                replacement_current = transducers[family]["current_a"]
            else:
                replacement_current = fitted[circuit - 1]
            replacement_field = replacement_current * field
            replacement_fields[key] = (
                replacement_fields.get(key, 0.0) + replacement_field
            )
            direct_fitted += fitted_field
            direct_replacement += replacement_field
            metadata = component_meta.setdefault(
                key,
                {
                    "category": category,
                    "family": family,
                    "stored_circuits": set(),
                    "section_element_count": 0,
                },
            )
            metadata["stored_circuits"].add(circuit)
            metadata["section_element_count"] += 1
    components = []
    for key, field in component_fields.items():
        metadata = component_meta[key]
        if "stored_circuits" in metadata:
            metadata["stored_circuits"] = sorted(metadata["stored_circuits"])
        replacement = replacement_fields[key]
        item = {
            "radial_field_mT": field * 1.0e3,
            "p6_ampere_equivalent_a": field / p6,
            "radial_field_mT_with_case_transducers": replacement * 1.0e3,
            "p6_ampere_equivalent_a_with_case_transducers": replacement / p6,
            **metadata,
        }
        if metadata["category"] == "instrumented_coil_case":
            item["case_transducer"] = transducers[metadata["family"]]
            item["case_swap_delta_mT"] = (replacement - field) * 1.0e3
            item["case_swap_delta_p6_a"] = (replacement - field) / p6
        components.append(item)
    components.sort(key=lambda item: (item["category"], item["family"]))
    fitted_total = float(sum(item["radial_field_mT"] for item in components))
    transducer_total = float(
        sum(item["radial_field_mT_with_case_transducers"] for item in components)
    )
    direct_fitted_mT = direct_fitted * 1.0e3
    direct_replacement_mT = direct_replacement * 1.0e3
    fitted_error = abs(fitted_total - direct_fitted_mT) / max(
        abs(direct_fitted_mT), np.finfo(np.float64).tiny
    )
    replacement_error = abs(transducer_total - direct_replacement_mT) / max(
        abs(direct_replacement_mT), np.finfo(np.float64).tiny
    )
    active_count = sum(item["category"] == "active_circuit" for item in components)
    case_count = sum(
        item["category"] == "instrumented_coil_case" for item in components
    )
    if active_count != 13 or case_count != 8:
        raise RuntimeError(
            "current partition produced "
            f"{active_count} active and {case_count} case groups"
        )
    return {
        "identity": f"{shot}/{row}",
        "efit_time_s": time_s,
        "current_centroid": {"r_m": target[0], "z_m": target[1]},
        "current_centroid_source": (
            "public seed current-moment observation on all-domain support"
        ),
        "archived_scalar_centroid": {
            "r_m": archived_centroid[0],
            "z_m": archived_centroid[1],
        },
        "p6_radial_response_t_per_a": p6,
        "p6_definition": "stored P6 upper current minus stored P6 lower current",
        "components": components,
        "totals": {
            "fitted_radial_field_mT": fitted_total,
            "fitted_p6_ampere_equivalent_a": fitted_total * 1.0e-3 / p6,
            "case_transducer_swap_radial_field_mT": transducer_total,
            "case_transducer_swap_p6_ampere_equivalent_a": transducer_total
            * 1.0e-3
            / p6,
            "case_swap_delta_mT": transducer_total - fitted_total,
            "case_swap_delta_p6_a": (transducer_total - fitted_total) * 1.0e-3 / p6,
            "direct_fitted_radial_field_mT": direct_fitted_mT,
            "direct_case_transducer_radial_field_mT": direct_replacement_mT,
            "sum_relative_error": max(fitted_error, replacement_error),
        },
    }


def _p6_constraint(
    profile, policy: dict[str, Any], target: float, span: float
) -> ConstraintPair:
    active = {
        str(item["family"]): int(item["stored_circuit"])
        for item in policy["active_mapping"]
    }
    prescribed = profile.operator.prescribed_current_field
    if prescribed is None:
        raise RuntimeError("the passive-inclusive profile has no prescribed circuits")
    direction = np.zeros(prescribed.circuit_count, dtype=np.float64)
    direction[active["p6_upper"] - 1] = 1.0
    direction[active["p6_lower"] - 1] = -1.0
    response_span = float(np.ptp(np.asarray(prescribed.response) @ direction))
    return ConstraintPair(
        functional=CurrentCentroidConstraint(
            components=("centroid_z",), support=MomentIntegralSupport.ALL_DOMAIN
        ),
        unknown=CircuitCurrentUnknown(
            direction=direction, ampere_scale=np.asarray([span / response_span])
        ),
        binding=ConstraintBinding(
            target=jnp.asarray([target]),
            tolerance=jnp.asarray([1.0e-6]),
            scale=jnp.asarray([float(np.ptp(np.asarray(profile.lattice.height)))]),
            initial_unknown=jnp.asarray([0.0]),
            payload=None,
            policy="imposed",
        ),
    )


def _current_digest(current: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(current, dtype=np.float64).tobytes()).hexdigest()


def _case_replacement_current(
    shot: int, row: int, profile
) -> tuple[np.ndarray, dict[str, Any]]:
    group = _efm_group(shot)
    time_s = float(group["time"][row])
    geometry = MachineGeometryRegistry.default().select(shot).configuration.geometry
    _families, _drives, mapping = _circuit_drives(group, row, geometry, "fcoil_c")
    active = {int(item["stored_circuit"]): str(item["family"]) for item in mapping}
    elements = _element_data(group)
    case_groups = _case_element_groups(elements, geometry, set(active))
    transducers = _interpolated_case_currents(shot, time_s)
    fitted = np.asarray(group["fcoil_c"][row], dtype=np.float64)
    field = profile.operator.prescribed_current_field
    if field is None:
        raise RuntimeError("the replacement solve requires prescribed circuit currents")
    baseline = np.asarray(field.current, dtype=np.float64)
    if not np.array_equal(baseline, fitted):
        raise RuntimeError("the prescribed field is not the EFIT fitted current vector")
    circuits_by_family: dict[str, set[int]] = {
        family: set() for family in CASE_CURRENT_CHANNELS
    }
    for index, family in case_groups.items():
        circuits_by_family[family].add(int(elements["fcoil_circ"][index]))
    replacement = fitted.copy()
    groups = []
    selected = []
    for family in sorted(circuits_by_family):
        circuits = sorted(circuits_by_family[family])
        if len(circuits) != 1:
            raise RuntimeError(
                f"case group {family} maps to stored circuits {circuits}, not one"
            )
        circuit = circuits[0]
        selected.append(circuit - 1)
        reading = transducers[family]
        replacement[circuit - 1] = reading["current_a"]
        groups.append(
            {
                "family": family,
                "stored_circuit": circuit,
                "fitted_current_a": float(fitted[circuit - 1]),
                "transducer_current_a": float(reading["current_a"]),
                "delta_current_a": float(reading["current_a"] - fitted[circuit - 1]),
                "transducer": reading,
            }
        )
    selected_indices = np.asarray(sorted(selected), dtype=int)
    if selected_indices.size != 8 or np.unique(selected_indices).size != 8:
        raise RuntimeError(
            "the instrumented current replacement must select eight circuits"
        )
    other_indices = np.setdiff1d(np.arange(fitted.size), selected_indices)
    if other_indices.size != 93 or not np.array_equal(
        replacement[other_indices], fitted[other_indices]
    ):
        raise RuntimeError(
            "a current outside the eight instrumented case circuits changed"
        )
    return replacement, {
        "source": "nearest finite amc sample at the EFIT slice time",
        "fitted_current_digest_sha256": _current_digest(fitted),
        "replacement_current_digest_sha256": _current_digest(replacement),
        "stored_circuit_count": int(fitted.size),
        "replaced_case_circuit_count": int(selected_indices.size),
        "unchanged_circuit_count": int(other_indices.size),
        "groups": groups,
    }


def _terminal_centroid(profile, flux, target_current_a: float) -> dict[str, float]:
    observation = profile.current_moment_observation(
        jnp.asarray(flux),
        support=MomentIntegralSupport.ALL_DOMAIN,
        target_current=target_current_a,
    )
    return {
        "r_m": float(observation.centroid_r),
        "z_m": float(observation.centroid_z),
    }


def _solve_case_current_state(
    profile,
    seed,
    target_current_a: float,
    prescribed_current: np.ndarray,
    carrier_identity: str,
    seed_centroid: dict[str, float],
    constraint_pair: ConstraintPair | None = None,
) -> dict[str, Any]:
    request = ForwardSolveRequest.from_defaults(
        carrier_identity=carrier_identity,
        source_profile=profile.source,
        seed_policy=ExplicitSolveSeed(jnp.asarray(seed)),
        target_current=target_current_a,
        prescribed_current=jnp.asarray(prescribed_current),
        constraint_pairs=() if constraint_pair is None else (constraint_pair,),
    )
    solved = profile.solve(request)
    equilibrium = solved.equilibrium
    fixed = equilibrium.fixed_point
    centroid = _terminal_centroid(profile, equilibrium.flux, target_current_a)
    result = {
        "terminal_residual": _strict_float(fixed.residual),
        "converged": bool(np.asarray(fixed.converged)),
        "termination_code": int(np.asarray(fixed.termination_reason)),
        "active_set_trips": int(np.asarray(fixed.active_set_iterations)),
        "terminal_centroid": centroid,
        "terminal_centroid_minus_seed_m": {
            "r_m": centroid["r_m"] - seed_centroid["r_m"],
            "z_m": centroid["z_m"] - seed_centroid["z_m"],
        },
        "resolved_defaults": solved.resolved_defaults.to_dict(),
    }
    if constraint_pair is not None:
        record = equilibrium.constraints[0]
        result["compensating_p6_current_a"] = _strict_float(record.physical_unknown[0])
        result["constraint_qualified"] = bool(np.asarray(record.qualified[0]))
    return result


def _write_receipt_part(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _case_current_part_path(output: Path, shot: int, row: int, name: str) -> Path:
    return output / "parts" / f"{shot}-{row}-{name}.json"


def _checkpointed_case_current_solve(
    *,
    output: Path,
    shot: int,
    row: int,
    name: str,
    profile,
    seed,
    target_current_a: float,
    prescribed_current: np.ndarray,
    request_identity: str,
    seed_centroid: dict[str, float],
    constraint_pair: ConstraintPair | None = None,
) -> tuple[dict[str, Any], bool]:
    """Return a prior result or atomically bank this solve before continuing."""

    path = _case_current_part_path(output, shot, row, name)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("request_identity") != request_identity:
            raise RuntimeError(f"receipt part {path} has a different request identity")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"receipt part {path} has no solve result")
        return result, False
    started = perf_counter()
    result = _solve_case_current_state(
        profile,
        seed,
        target_current_a,
        prescribed_current,
        request_identity,
        seed_centroid,
        constraint_pair,
    )
    result["subsolve_wall_s"] = perf_counter() - started
    _write_receipt_part(
        path,
        {
            "identity": f"{shot}/{row}",
            "part": name,
            "request_identity": request_identity,
            "prescribed_current_digest_sha256": _current_digest(prescribed_current),
            "result": result,
        },
    )
    return result, True


def _case_current_repair_row(
    output: Path, shot: int, row: int, maximum_new_solves: int | None = None
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    cache, carrier_evidence = settled._persisted_response_cache(
        settled.response_carrier.DEFAULT_CARRIER,
        settled.response_carrier.DEFAULT_RECEIPT,
    )
    selected = {
        (int(item["shot"]), int(item["slice_index"])): (item, qualification)
        for item, qualification in settled.select_slices_by_shot(
            settled.DECOMPOSITION_BANK
        )
    }
    selected_row, qualification = selected[(shot, row)]
    case, context = settled._mast_case_from_selection(
        settled.SHOT_STORE, selected_row, qualification
    )
    passive_case, profile, policy = settled._passive_inclusive_case(
        case, context, cache
    )
    if int(policy["section_kernel_evaluations_this_shot"]) != 0:
        raise RuntimeError("case-current repair rebuilt a direct response matrix")
    seed = jnp.asarray(passive_case["state"])
    target_current_a = abs(float(passive_case["reference"]["plasma_current_a"]))
    seed_centroid = _terminal_centroid(profile, seed, target_current_a)
    fitted = np.asarray(
        profile.operator.prescribed_current_field.current, dtype=np.float64
    )
    replacement, replacement_evidence = _case_replacement_current(shot, row, profile)
    completed_parts: list[str] = []
    new_solves = 0

    def solve_part(
        name: str,
        prescribed_current: np.ndarray,
        request_identity: str,
        constraint_pair: ConstraintPair | None = None,
    ) -> dict[str, Any] | None:
        nonlocal new_solves
        path = _case_current_part_path(output, shot, row, name)
        if (
            maximum_new_solves is not None
            and new_solves >= maximum_new_solves
            and not path.exists()
        ):
            return None
        result, wrote = _checkpointed_case_current_solve(
            output=output,
            shot=shot,
            row=row,
            name=name,
            profile=profile,
            seed=seed,
            target_current_a=target_current_a,
            prescribed_current=prescribed_current,
            request_identity=request_identity,
            seed_centroid=seed_centroid,
            constraint_pair=constraint_pair,
        )
        completed_parts.append(str(path.relative_to(output)))
        new_solves += int(wrote)
        return result

    fitted_free_identity = f"mast:{shot}:{row}:case-current:fitted:free"
    fitted_free = solve_part("fitted-free", fitted, fitted_free_identity)
    if fitted_free is None:
        return None, {
            "identity": f"{shot}/{row}",
            "completed_parts": completed_parts,
            "new_solves": new_solves,
            "complete": False,
        }
    measured_free_identity = f"mast:{shot}:{row}:case-current:measured:free"
    measured_free = solve_part("measured-free", replacement, measured_free_identity)
    if measured_free is None:
        return None, {
            "identity": f"{shot}/{row}",
            "completed_parts": completed_parts,
            "new_solves": new_solves,
            "complete": False,
        }
    free = {
        "fitted_current_baseline": fitted_free,
        "measured_case_current": measured_free,
    }
    scan = []
    for target_z_m in ELIMINATION_HEIGHTS_M:
        pair = _p6_constraint(
            profile, policy, target_z_m, float(passive_case["span_wb"])
        )
        fitted_identity = f"mast:{shot}:{row}:case-current:fitted:z:{target_z_m:.9f}"
        fitted_result = solve_part(
            f"fitted-z-{target_z_m:.9f}", fitted, fitted_identity, pair
        )
        if fitted_result is None:
            return None, {
                "identity": f"{shot}/{row}",
                "completed_parts": completed_parts,
                "new_solves": new_solves,
                "complete": False,
            }
        measured_identity = (
            f"mast:{shot}:{row}:case-current:measured:z:{target_z_m:.9f}"
        )
        measured_result = solve_part(
            f"measured-z-{target_z_m:.9f}", replacement, measured_identity, pair
        )
        if measured_result is None:
            return None, {
                "identity": f"{shot}/{row}",
                "completed_parts": completed_parts,
                "new_solves": new_solves,
                "complete": False,
            }
        scan.append(
            {
                "target_centroid_z_m": target_z_m,
                "fitted_current_baseline": fitted_result,
                "measured_case_current": measured_result,
            }
        )
    fitted_values = [
        sample["fitted_current_baseline"]["compensating_p6_current_a"]
        for sample in scan
    ]
    measured_values = [
        sample["measured_case_current"]["compensating_p6_current_a"] for sample in scan
    ]
    return {
        "identity": f"{shot}/{row}",
        "seed_centroid": seed_centroid,
        "free_production_solve": free,
        "elimination_scan": scan,
        "case_current_replacement": replacement_evidence,
        "policy": policy,
        "carrier_evidence": carrier_evidence,
        "minimum_fitted_current_compensation_a": min(fitted_values, key=abs),
        "minimum_measured_case_compensation_a": min(measured_values, key=abs),
        "receipt_parts": completed_parts,
    }, {
        "identity": f"{shot}/{row}",
        "completed_parts": completed_parts,
        "new_solves": new_solves,
        "complete": True,
    }


def _draw_case_current_repair(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    for row in rows:
        free = row["free_production_solve"]
        labels = ("fitted", "case measured")
        residuals = [
            free["fitted_current_baseline"]["terminal_residual"],
            free["measured_case_current"]["terminal_residual"],
        ]
        axes[0].plot(labels, residuals, "o-", label=row["identity"])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Free-solve terminal residual")
    axes[0].set_title("Free production solve from bank seed")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    for row in rows:
        x = [item["target_centroid_z_m"] for item in row["elimination_scan"]]
        fitted = [
            item["fitted_current_baseline"]["compensating_p6_current_a"] / 1.0e3
            for item in row["elimination_scan"]
        ]
        measured = [
            item["measured_case_current"]["compensating_p6_current_a"] / 1.0e3
            for item in row["elimination_scan"]
        ]
        axes[1].plot(x, fitted, "o--", label=f"{row['identity']} fitted")
        axes[1].plot(x, measured, "o-", label=f"{row['identity']} case measured")
    axes[1].axhline(0.0, color="0.3", linewidth=0.8)
    axes[1].set_xlabel("Constrained current-centroid Z [m]")
    axes[1].set_ylabel("P6 compensation [kA]")
    axes[1].set_title("Five-height elimination scan")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)
    figure.suptitle("Effect of measured coil-case currents on vertical balance")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def measure_case_current_repair(
    output: Path,
    requested_rows: tuple[tuple[int, int], ...] = ROWS,
    maximum_new_solves: int | None = None,
) -> dict[str, Any]:
    configure_dtypes()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "case-current-repair.json"
    existing: dict[str, Any] = {}
    existing_rows = []
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        existing_rows = existing.get("rows", [])
    rows_by_identity = {row["identity"]: row for row in existing_rows}
    progress = []
    remaining_new_solves = maximum_new_solves
    for shot, row in requested_rows:
        measured, state = _case_current_repair_row(
            output, shot, row, remaining_new_solves
        )
        progress.append(state)
        if remaining_new_solves is not None:
            remaining_new_solves -= int(state["new_solves"])
        if measured is not None:
            rows_by_identity[measured["identity"]] = measured
        rows = [
            rows_by_identity[f"{item_shot}/{item_row}"]
            for item_shot, item_row in ROWS
            if f"{item_shot}/{item_row}" in rows_by_identity
        ]
        in_progress_receipt = dict(existing)
        in_progress_receipt["rows"] = rows
        in_progress_receipt["progress"] = progress
        receipt_path.write_text(
            json.dumps(in_progress_receipt, indent=2) + "\n",
            encoding="utf-8",
        )
    rows = [
        rows_by_identity[f"{shot}/{row}"]
        for shot, row in ROWS
        if f"{shot}/{row}" in rows_by_identity
    ]
    if len(rows) != len(ROWS):
        return {
            "rows": rows,
            "progress": progress,
            "verdict": {"complete": False},
        }
    receipt = {
        "receipt": "free solve and elimination scan with measured coil-case currents",
        "source": {
            "revision": _source_revision(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
        },
        "configuration": {
            "replacement": "eight instrumented case-circuit values replace fcoil_c",
            "other_circuits": (
                "the remaining 93 stored circuit currents are byte-identical"
            ),
            "scan_heights_m": list(ELIMINATION_HEIGHTS_M),
            "persistent_compilation_cache": {
                "directory": str(cache.directory),
                "version": cache.version_key,
            },
        },
        "rows": rows,
    }
    repaired = next(row for row in rows if row["identity"] == "21986/46")
    subject = next(row for row in rows if row["identity"] == "21989/55")
    fitted_minimum = min(
        (
            sample["fitted_current_baseline"]["compensating_p6_current_a"]
            for sample in subject["elimination_scan"]
        ),
        key=abs,
    )
    measured_minimum = subject["minimum_measured_case_compensation_a"]
    compensation_effect = (
        "removes"
        if measured_minimum == 0.0
        else "reduces"
        if abs(measured_minimum) < abs(fitted_minimum)
        else "increases"
    )
    receipt["verdict"] = {
        "minimum_measured_case_compensation_21986_46_a": repaired[
            "minimum_measured_case_compensation_a"
        ],
        "free_measured_case_converged_21986_46": repaired["free_production_solve"][
            "measured_case_current"
        ]["converged"],
        "free_measured_case_converged_21989_55": next(
            row for row in rows if row["identity"] == "21989/55"
        )["free_production_solve"]["measured_case_current"]["converged"],
        "minimum_fitted_current_compensation_21989_55_a": fitted_minimum,
        "minimum_measured_case_compensation_21989_55_a": measured_minimum,
        "measured_to_fitted_compensation_ratio_21989_55": abs(measured_minimum)
        / abs(fitted_minimum),
        "case_current_substitution_effect_21989_55": compensation_effect,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    _draw_case_current_repair(rows, output / "case-current-repair.png")
    return receipt


def checkpoint_smoke(output: Path) -> dict[str, Any]:
    """Exercise atomic part persistence without constructing a plasma solve."""

    path = output / "parts" / "synthetic-checkpoint.json"
    request_identity = "synthetic:case-current:checkpoint"
    resumed = path.exists()
    if resumed:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("request_identity") != request_identity:
            raise RuntimeError("synthetic checkpoint has a different request identity")
    else:
        payload = {
            "identity": "synthetic/0",
            "part": "synthetic-checkpoint",
            "request_identity": request_identity,
            "prescribed_current_digest_sha256": _current_digest(np.zeros(1)),
            "result": {"subsolve_wall_s": 0.0, "synthetic": True},
        }
        _write_receipt_part(path, payload)
    return {
        "verdict": {
            "part_written": path.exists(),
            "resumed_existing_part": resumed,
            "no_solver_constructed": True,
        },
        "part": str(path),
    }


def _scan_current(output: Path) -> dict[str, Any]:
    """Run distant continuation samples through the typed public solve seam."""

    cache, carrier_evidence = settled._persisted_response_cache(
        settled.response_carrier.DEFAULT_CARRIER,
        settled.response_carrier.DEFAULT_RECEIPT,
    )
    selected = {
        (int(item["shot"]), int(item["slice_index"])): (item, qualification)
        for item, qualification in settled.select_slices_by_shot(
            settled.DECOMPOSITION_BANK
        )
    }
    selected_row, qualification = selected[(21986, 46)]
    case, context = settled._mast_case_from_selection(
        settled.SHOT_STORE, selected_row, qualification
    )
    passive_case, profile, policy = settled._passive_inclusive_case(
        case, context, cache
    )
    if int(policy["section_kernel_evaluations_this_shot"]) != 0:
        raise RuntimeError("the scan rebuilt a direct response matrix")
    seed = jnp.asarray(passive_case["state"])
    target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
    samples = []
    for target in SCAN_HEIGHTS_M:
        pair = _p6_constraint(profile, policy, target, float(passive_case["span_wb"]))
        request = ForwardSolveRequest.from_defaults(
            carrier_identity=f"mast:21986:46:vertical-force-scan:{target:.9f}",
            source_profile=profile.source,
            seed_policy=ExplicitSolveSeed(seed),
            target_current=target_current,
            constraint_pairs=(pair,),
        )
        solve = profile.solve(request)
        equilibrium = solve.equilibrium
        record = equilibrium.constraints[0]
        samples.append(
            {
                "target_centroid_z_m": target,
                "compensating_current_a": _strict_float(record.physical_unknown[0]),
                "terminal_residual": _strict_float(equilibrium.fixed_point.residual),
                "converged": bool(np.asarray(equilibrium.fixed_point.converged)),
                "active_set_trips": int(
                    np.asarray(equilibrium.fixed_point.active_set_iterations)
                ),
                "termination_code": int(
                    np.asarray(equilibrium.fixed_point.termination_reason)
                ),
                "resolved_defaults": solve.resolved_defaults.to_dict(),
            }
        )
        output.write_text(
            json.dumps({"samples": samples}, indent=2) + "\n", encoding="utf-8"
        )
    values = [sample["compensating_current_a"] for sample in samples]
    zero_bracket = any(
        first * second <= 0.0 for first, second in zip(values, values[1:], strict=False)
    )
    scan = {
        "identity": "21986/46",
        "baseline_protocol_samples_a": {
            "0.038894285": -2730.0,
            "0.058894285": -2581.0,
            "0.078894285": -2885.0,
            "0.098894285": -4103.0,
            "0.118894285": -5764.0,
        },
        "extension_samples": samples,
        "zero_exists_in_extended_samples": zero_bracket,
        "verdict": (
            "a zero-current sign change is present in the two requested distant samples"
            if zero_bracket
            else "no zero-current sign change appears in the requested distant samples"
        ),
        "carrier_evidence": carrier_evidence,
    }
    output.write_text(json.dumps(scan, indent=2) + "\n", encoding="utf-8")
    return scan


def _draw(rows: list[dict[str, Any]], scan: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(
        1, 3, figsize=(21, 7), gridspec_kw={"width_ratios": (1.15, 1.15, 0.8)}
    )
    for axis, row in zip(axes[:2], rows, strict=True):
        components = sorted(
            row["components"],
            key=lambda item: abs(item["p6_ampere_equivalent_a"]),
            reverse=True,
        )
        labels = [
            (
                f"case / {item['family']}"
                if item["category"] == "instrumented_coil_case"
                else f"active / {item['family']}"
                if item["category"] == "active_circuit"
                else item["family"]
            )
            for item in components
        ]
        values = [item["p6_ampere_equivalent_a"] / 1.0e3 for item in components]
        transducer = [
            item["p6_ampere_equivalent_a_with_case_transducers"] / 1.0e3
            for item in components
        ]
        y = np.arange(len(labels))
        axis.barh(y - 0.18, values, 0.36, label="fitted")
        axis.barh(y + 0.18, transducer, 0.36, label="case readings")
        axis.axvline(0.0, color="0.3", linewidth=0.8)
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_xlabel("P6 ampere-equivalent [kA]")
        axis.set_title(row["identity"])
        axis.grid(axis="x", alpha=0.2)
    axes[0].legend(frameon=False)
    scan_axis = axes[2]
    protocol = scan["baseline_protocol_samples_a"]
    x = np.asarray([float(key) for key in protocol])
    y = np.asarray(list(protocol.values())) / 1.0e3
    extension = scan["extension_samples"]
    x = np.r_[x, [item["target_centroid_z_m"] for item in extension]]
    y = np.r_[y, [item["compensating_current_a"] / 1.0e3 for item in extension]]
    order = np.argsort(x)
    scan_axis.plot(x[order], y[order], "o-")
    scan_axis.axhline(0.0, color="0.3", linewidth=0.8)
    scan_axis.set_title("21986/46 scan")
    scan_axis.set_xlabel("target Z [m]")
    scan_axis.set_ylabel("P6 [kA]")
    figure.suptitle(
        "Seed-centroid radial-field decomposition by current family", y=0.98
    )
    figure.subplots_adjust(left=0.17, right=0.98, bottom=0.13, top=0.90, wspace=0.50)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def measure(output: Path) -> dict[str, Any]:
    configure_dtypes()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    output.mkdir(parents=True, exist_ok=True)
    rows = [_decompose_row(shot, row) for shot, row in ROWS]
    scan_path = output / "extended-compensating-current-scan.json"
    scan = _scan_current(scan_path)
    receipt = {
        "receipt": "seed-centroid radial-field decomposition by current family",
        "source": {
            "revision": _source_revision(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
        },
        "configuration": {
            "field": "B_R at the stored EFIT current centroid",
            "p6_equivalent": (
                "B_R divided by the P6 upper-minus-lower unit response at that centroid"
            ),
            "case_swap": (
                "replace each fitted coil-case group current by its nearest finite "
                "transducer sample at the EFIT slice time"
            ),
            "plasma": (
                "native EFIT delta-star current density imaged through "
                "shaped cell sections"
            ),
            "persistent_compilation_cache": {
                "directory": str(cache.directory),
                "version": cache.version_key,
            },
        },
        "rows": rows,
        "extended_scan": scan,
    }
    receipt["verdict"] = {
        "sum_relative_error_max": max(
            row["totals"]["sum_relative_error"] for row in rows
        ),
        "extended_zero_exists": scan["zero_exists_in_extended_samples"],
        "largest_fitted_component_per_row": {
            row["identity"]: max(
                row["components"], key=lambda item: abs(item["p6_ampere_equivalent_a"])
            )["family"]
            for row in rows
        },
    }
    (output / "force-decomposition.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    _draw(rows, scan, output / "force-decomposition.png")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sensitivity-only",
        action="store_true",
        help="write the active-current and coil-position sensitivity receipt only",
    )
    parser.add_argument(
        "--case-current-repair",
        action="store_true",
        help="solve fitted and measured-case current vectors from each bank seed",
    )
    parser.add_argument(
        "--case-current-identity",
        choices=[f"{shot}/{row}" for shot, row in ROWS],
        action="append",
        help="limit case-current repair to one missing shot and row",
    )
    parser.add_argument(
        "--case-current-part-limit",
        type=int,
        help="solve at most this many missing case-current receipt parts",
    )
    parser.add_argument(
        "--case-current-checkpoint-smoke",
        action="store_true",
        help="write or resume a synthetic receipt part without constructing a solve",
    )
    args = parser.parse_args()
    if (
        sum(
            (
                args.sensitivity_only,
                args.case_current_repair,
                args.case_current_checkpoint_smoke,
            )
        )
        > 1
    ):
        raise ValueError("choose one focused measurement")
    if args.case_current_part_limit is not None and args.case_current_part_limit < 1:
        raise ValueError("case-current part limit must be positive")
    if args.sensitivity_only:
        receipt = measure_sensitivity(args.output)
    elif args.case_current_checkpoint_smoke:
        receipt = checkpoint_smoke(args.output)
    elif args.case_current_repair:
        requested_rows = (
            tuple(
                (int(identity.split("/")[0]), int(identity.split("/")[1]))
                for identity in args.case_current_identity
            )
            if args.case_current_identity
            else ROWS
        )
        receipt = measure_case_current_repair(
            args.output, requested_rows, args.case_current_part_limit
        )
    else:
        receipt = measure(args.output)
        receipt["sensitivity"] = measure_sensitivity(args.output)
    print(json.dumps(receipt["verdict"], sort_keys=True))


if __name__ == "__main__":
    main()
