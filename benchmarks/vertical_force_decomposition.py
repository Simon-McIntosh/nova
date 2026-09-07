"""Decompose the seed-centroid radial field by MAST current family.

The decomposition uses the stored EFIT current vector exactly once.  It groups
its shaped sections into the thirteen active circuits, the eight coil-case
groups with direct current transducers, and every remaining passive or vessel
section.  The plasma term is the native EFIT current density integrated through
the same shaped-section Green kernel at the stored current centroid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
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


def _case_element_groups(
    elements: dict[str, np.ndarray], geometry: dict[str, Any]
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

    radius = np.asarray(group["gridr"], dtype=np.float64)
    height = np.asarray(group["gridz"], dtype=np.float64)
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


def _decompose_row(shot: int, row: int) -> dict[str, Any]:
    group = _efm_group(shot)
    target = _centroid(group, row)
    time_s = float(group["time"][row])
    geometry = MachineGeometryRegistry.default().select(shot).configuration.geometry
    _families, _drives, mapping = _circuit_drives(group, row, geometry, "fcoil_c")
    active = {int(item["stored_circuit"]): str(item["family"]) for item in mapping}
    elements = _element_data(group)
    fitted = np.asarray(group["fcoil_c"][row], dtype=np.float64)
    indices = np.asarray(group["fcoil_n"], dtype=int)
    if not np.array_equal(indices, np.arange(fitted.size)):
        raise ValueError("fcoil_n does not use zero-based stored-current order")
    case_groups = _case_element_groups(elements, geometry)
    transducers = _interpolated_case_currents(shot, time_s)
    p6 = _p6_response(elements, active, target)
    component_fields: dict[str, float] = {"plasma_self": 0.0}
    replacement_fields: dict[str, float] = {"plasma_self": 0.0}
    component_meta: dict[str, dict[str, Any]] = {}
    plasma, plasma_meta = _plasma_br(group, row, target)
    component_fields["plasma_self"] = plasma
    replacement_fields["plasma_self"] = plasma
    component_meta["plasma_self"] = {"category": "plasma", **plasma_meta}
    for circuit in range(1, fitted.size + 1):
        active_family = active.get(circuit)
        fallback = "passive_vessel_remaining"
        for index in np.flatnonzero(elements["fcoil_circ"] == circuit):
            family = active_family or case_groups.get(int(index), fallback)
            field = _element_br_per_ampere(elements, int(index), target)
            component_fields[family] = (
                component_fields.get(family, 0.0) + fitted[circuit - 1] * field
            )
            if family in transducers:
                replacement = transducers[family]["current_a"]
            else:
                replacement = fitted[circuit - 1]
            replacement_fields[family] = (
                replacement_fields.get(family, 0.0) + replacement * field
            )
            metadata = component_meta.setdefault(
                family,
                {
                    "category": (
                        "active_circuit"
                        if active_family
                        else "instrumented_coil_case"
                        if family in transducers
                        else "passive_or_vessel"
                    ),
                    "stored_circuits": set(),
                    "section_element_count": 0,
                },
            )
            metadata["stored_circuits"].add(circuit)
            metadata["section_element_count"] += 1
    components = []
    for family, field in component_fields.items():
        metadata = component_meta[family]
        if "stored_circuits" in metadata:
            metadata["stored_circuits"] = sorted(metadata["stored_circuits"])
        replacement = replacement_fields[family]
        item = {
            "family": family,
            "radial_field_mT": field * 1.0e3,
            "p6_ampere_equivalent_a": field / p6,
            "radial_field_mT_with_case_transducers": replacement * 1.0e3,
            "p6_ampere_equivalent_a_with_case_transducers": replacement / p6,
            **metadata,
        }
        if family in transducers:
            item["case_transducer"] = transducers[family]
            item["case_swap_delta_mT"] = (replacement - field) * 1.0e3
            item["case_swap_delta_p6_a"] = (replacement - field) / p6
        components.append(item)
    components.sort(key=lambda item: (item["category"], item["family"]))
    fitted_total = float(sum(item["radial_field_mT"] for item in components))
    transducer_total = float(
        sum(item["radial_field_mT_with_case_transducers"] for item in components)
    )
    return {
        "identity": f"{shot}/{row}",
        "efit_time_s": time_s,
        "current_centroid": {"r_m": target[0], "z_m": target[1]},
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
            "sum_relative_error": 0.0,
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
    return {
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


def _draw(rows: list[dict[str, Any]], scan: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    for axis, row in zip(axes, rows, strict=True):
        components = sorted(
            row["components"],
            key=lambda item: abs(item["p6_ampere_equivalent_a"]),
            reverse=True,
        )
        labels = [item["family"] for item in components]
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
    scan_axis = axes[1].inset_axes([0.58, 0.61, 0.38, 0.31])
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
    figure.subplots_adjust(left=0.20, right=0.98, bottom=0.13, top=0.90, wspace=0.42)
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
                "replace each fitted coil-case group current by its interpolated "
                "exact-time transducer reading"
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
    args = parser.parse_args()
    receipt = measure(args.output)
    print(json.dumps(receipt["verdict"], sort_keys=True))


if __name__ == "__main__":
    main()
