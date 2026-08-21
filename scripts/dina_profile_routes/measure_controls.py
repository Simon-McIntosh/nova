"""Bank rotation, normalization, and coordinate controls for the DINA routes."""

from __future__ import annotations

from dataclasses import replace
import getpass
import json
from pathlib import Path
import sys

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as PolygonPath
import numpy as np
from scipy.constants import mu_0

from nova.equilibrium.map_extraction import extract_flux_functions


OUTPUT = Path(__file__).resolve().parent
FIGURES = Path(__file__).resolve().parents[2] / "docs/figures/dina-profile-routes"
REFERENCE_PATH = Path("tests/test_equilibrium_forward_reference.py")
MAPPING_PATH = Path("benchmarks/dina_reference_mapping.py")
DISCRIMINATOR_PATH = Path("scripts/normalization_discriminator/results.json")
ANCHOR_PATH = Path("scripts/normalization_discriminator/results.json")
BASE_RECEIPT_PATH = OUTPUT / "receipt.json"
COARSE_CACHE_KEY = "746fbe1553c4b242"
BASE_SURFACES = np.linspace(0.05, 0.95, 19)
EXTENDED_SURFACES = np.r_[
    np.array([0.01, 0.02, 0.03, 0.04]),
    BASE_SURFACES,
    np.array([0.96, 0.97, 0.98, 0.99]),
]


def load_module(path: Path, name: str):
    """Load a repository module without invoking its command entry point."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def finite_statistics(values: np.ndarray) -> dict[str, float]:
    """Summarise a finite vector without discarding its sign."""
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("statistics require finite values")
    return {
        "signed_mean": float(np.mean(values)),
        "mean_absolute": float(np.mean(np.abs(values))),
        "rms": float(np.sqrt(np.mean(values**2))),
        "sup": float(np.max(np.abs(values))),
    }


def surface_boundaries(centres: np.ndarray) -> np.ndarray:
    """Return midpoint shell boundaries for increasing surface centres."""
    boundary = np.empty(centres.size + 1)
    boundary[1:-1] = 0.5 * (centres[1:] + centres[:-1])
    boundary[0] = max(0.0, centres[0] - 0.5 * (centres[1] - centres[0]))
    boundary[-1] = min(1.0, centres[-1] + 0.5 * (centres[-1] - centres[-2]))
    return boundary


def extraction_inputs(mapped, stored, anchors: dict[str, object]):
    """Return map labels and the declared-coordinate plasma support."""
    constants = anchors["normalization_constants"]
    topology = constants["production_topology"]
    map_flux = np.asarray(mapped.grid_flux, dtype=float)
    map_norm = (map_flux - topology["axis_flux_wb"]) / (
        topology["boundary_flux_wb"] - topology["axis_flux_wb"]
    )
    declared_norm = (map_flux - mapped.flux_axis) / (
        mapped.flux_boundary - mapped.flux_axis
    )
    grid_r, grid_z = np.meshgrid(stored.grid_radius, stored.grid_height, indexing="ij")
    outline = (
        PolygonPath(stored.boundary)
        .contains_points(np.c_[grid_r.ravel(), grid_z.ravel()])
        .reshape(map_flux.shape)
    )
    support = outline & (declared_norm >= 0.0) & (declared_norm <= 1.0)
    return map_flux, map_norm, support


def extract_surfaces(mapped, stored, anchors, surfaces: np.ndarray):
    """Run the production affine extractor on explicitly selected shells."""
    map_flux, map_norm, support = extraction_inputs(mapped, stored, anchors)
    return extract_flux_functions(
        stored.grid_radius,
        stored.grid_height,
        map_flux,
        map_norm,
        surfaces=surfaces,
        plasma_mask=support,
        min_samples=6,
    )


def rotation_fit_receipt(mapped, stored, anchors, extracted) -> dict[str, object]:
    """Fit static and small-exponent centrifugal columns on every map shell."""
    map_flux, map_norm, support = extraction_inputs(mapped, stored, anchors)
    current = extracted.current
    radius_map = np.broadcast_to(current.radius[:, None], current.flux.shape)
    target_map = radius_map * current.toroidal_current_density
    selected = support & current.valid & np.isfinite(map_norm)
    boundaries = surface_boundaries(np.asarray(extracted.psi_norm))
    rows = []
    for index, centre in enumerate(np.asarray(extracted.psi_norm)):
        upper = map_norm <= boundaries[index + 1]
        if index + 1 < len(extracted.psi_norm):
            upper = map_norm < boundaries[index + 1]
        shell = selected & (map_norm >= boundaries[index]) & upper
        radius = radius_map[shell]
        target = target_map[shell]
        static_design = np.column_stack([mu_0 * radius**2, np.ones(target.size)])
        rotation_column = mu_0 * radius**2 * (radius**2 - stored.reference_radius**2)
        rotating_design = np.column_stack([static_design, rotation_column])

        def fit(design):
            scale = np.linalg.norm(design, axis=0)
            scaled = design / scale
            scaled_coefficient = np.linalg.lstsq(scaled, target, rcond=None)[0]
            coefficient = scaled_coefficient / scale
            residual = target - design @ coefficient
            residual_rms = float(np.sqrt(np.mean(residual**2)))
            variance = residual_rms**2
            covariance = variance * np.linalg.pinv(scaled.T @ scaled)
            covariance = covariance / np.outer(scale, scale)
            return coefficient, residual_rms, covariance

        static_coefficient, static_rms, _ = fit(static_design)
        rotating_coefficient, rotating_rms, rotating_covariance = fit(rotating_design)
        signal_rms = float(np.sqrt(np.mean(target**2)))
        inflation = float(extracted.uncertainty_inflation[index])
        amplitude_uncertainty = inflation * np.sqrt(
            max(float(rotating_covariance[2, 2]), 0.0)
        )
        significance = (
            abs(float(rotating_coefficient[2])) / amplitude_uncertainty
            if amplitude_uncertainty > 0.0
            else float("inf")
        )
        rows.append(
            {
                "psi_norm_map": float(centre),
                "sample_count": int(target.size),
                "signal_rms_rj_phi_a_per_m": signal_rms,
                "static_projection_rms_rj_phi_a_per_m": static_rms,
                "static_explained_variance_fraction": 1.0
                - static_rms**2 / signal_rms**2,
                "static_pressure_column_amplitude": float(static_coefficient[0]),
                "static_intercept_amplitude": float(static_coefficient[1]),
                "rotation_column": "mu0*R^2*(R^2-R0^2)",
                "rotation_column_amplitude": float(rotating_coefficient[2]),
                "rotation_column_amplitude_uncertainty": float(amplitude_uncertainty),
                "rotation_amplitude_over_uncertainty": float(significance),
                "rotation_amplitude_significant_95_percent_proxy": bool(
                    significance >= 1.96
                ),
                "rotating_projection_rms_rj_phi_a_per_m": rotating_rms,
                "residual_rms_reduction_rj_phi_a_per_m": static_rms - rotating_rms,
                "residual_rms_reduction_fraction": 1.0 - rotating_rms / static_rms,
                "extractor_projection_rms_rj_phi_a_per_m": float(
                    extracted.projection_rms[index]
                ),
            }
        )
    counts = np.asarray([row["sample_count"] for row in rows])
    signals = np.asarray([row["signal_rms_rj_phi_a_per_m"] for row in rows])
    static = np.asarray([row["static_projection_rms_rj_phi_a_per_m"] for row in rows])
    rotating = np.asarray(
        [row["rotating_projection_rms_rj_phi_a_per_m"] for row in rows]
    )
    signal_rms = float(np.sqrt(np.sum(counts * signals**2) / np.sum(counts)))
    static_rms = float(np.sqrt(np.sum(counts * static**2) / np.sum(counts)))
    rotating_rms = float(np.sqrt(np.sum(counts * rotating**2) / np.sum(counts)))
    significant = sum(
        row["rotation_amplitude_significant_95_percent_proxy"] for row in rows
    )
    return {
        "basis": {
            "static": ["mu0*R^2", "1"],
            "rotation_diagnostic": "mu0*R^2*(R^2-R0^2)",
            "reference_radius_m": float(stored.reference_radius),
            "interpretation": (
                "small-centrifugal-exponent radial shape from rotation.py; "
                "the amplitude is a diagnostic coefficient, not a declared closure"
            ),
            "uncertainty": (
                "least-squares covariance multiplied by the extractor's shell "
                "uncertainty inflation; 1.96 sigma is only a correlation-aware proxy"
            ),
        },
        "shells": rows,
        "aggregate": {
            "sample_count": int(np.sum(counts)),
            "signal_rms_rj_phi_a_per_m": signal_rms,
            "static_projection_rms_rj_phi_a_per_m": static_rms,
            "static_residual_to_signal_fraction": static_rms / signal_rms,
            "static_explained_variance_fraction": 1.0 - static_rms**2 / signal_rms**2,
            "rotating_projection_rms_rj_phi_a_per_m": rotating_rms,
            "rotating_residual_to_signal_fraction": rotating_rms / signal_rms,
            "residual_rms_reduction_fraction": 1.0 - rotating_rms / static_rms,
            "significant_rotation_proxy_shells": int(significant),
            "total_shells": len(rows),
        },
    }


def rotational_constraint_receipt() -> dict[str, object]:
    """Read the native-DD rotational-pressure constraint through imas-python."""
    import imas

    failures = []
    for user in ("public", getpass.getuser()):
        uri = f"imas:hdf5?user={user};pulse=135011;run=7;database=iter;version=3"
        try:
            entry = imas.DBEntry(uri, "r", dd_version="3.39.0")
            try:
                equilibrium = entry.get("equilibrium")
            finally:
                entry.close()
            constraint = equilibrium.time_slice[353].constraints.pressure_rotational
            return {
                "path": "equilibrium/time_slice/constraints/pressure_rotational",
                "dd_version": "3.39.0",
                "reader": "imas.DBEntry",
                "source_user": user,
                "present_in_dictionary": True,
                "filled": bool(constraint.has_value),
                "item_count": int(constraint.size),
            }
        except Exception as error:  # noqa: BLE001 - availability audit
            failures.append(f"{user}: {type(error).__name__}")
    raise RuntimeError(
        "DINA rotational constraint is unreachable: " + "; ".join(failures)
    )


def load_coarse_machine(reference, case):
    """Open the banked coarse carrier without selecting a cold-build key."""
    reference.WALL_NODES = 3
    store = reference.ZarrStore(
        filename=reference.MACHINE_CACHE_FILENAME,
        dirname=".nova",
        group=COARSE_CACHE_KEY,
    )
    with reference._machine_cache_lock(store):
        store.load()
        identity = json.loads(store.data.attrs["semantic_identity"])
        machine = reference._machine_from_dataset(
            store.data, identity, COARSE_CACHE_KEY
        )
    if identity["discretisation"]["cells"] != reference.SUITE_CELLS:
        raise RuntimeError("banked coarse carrier has a different cell request")
    return machine, identity, store


def density_factorial(reference, case, machine, anchors) -> dict[str, object]:
    """Evaluate declared profiles under both value and normalization choices."""
    operator = reference.forward_operator(case, machine)
    seed = reference.seed_flux(case, machine)
    partition = operator._support_partition(seed)
    masks, _topology, _sample_norm, support, common_support = partition
    production_centroid = np.asarray(seed)[: operator.grid.node_number]
    production_sample = np.asarray(operator.sample_node_flux(seed))
    grid_point = np.asarray(operator.grid.coordinate)
    sample_point = np.asarray(operator.moment_geometry.sample_node_coordinates)
    exact_centroid = case.flux(grid_point[:, 0], grid_point[:, 1])
    exact_sample = case.flux(sample_point[:, 0], sample_point[:, 1])
    constants = anchors["normalization_constants"]
    choices = {
        "map_saddle": constants["production_topology"],
        "declared": constants["exact_case"],
    }
    values = {
        "interpolated": (production_centroid, production_sample),
        "map_exact": (exact_centroid, exact_sample),
    }
    external = np.asarray(operator.external())
    images = {}
    arms = {}
    core = np.asarray(masks.core, dtype=bool)
    for value_name, (centroid_absolute, sample_absolute) in values.items():
        for constant_name, anchor in choices.items():
            name = f"{value_name}_values_{constant_name}_constants"
            centroid_norm = (centroid_absolute - anchor["axis_flux_wb"]) / anchor[
                "flux_span_wb"
            ]
            sample_norm = (sample_absolute - anchor["axis_flux_wb"]) / anchor[
                "flux_span_wb"
            ]
            adjusted_masks = type(masks)(
                label=masks.label,
                psi_norm=jax.numpy.asarray(centroid_norm),
            )
            moments = operator.source.current_moments(
                adjusted_masks,
                operator.support_current_moments,
                support,
                common_support,
                sample_flux=jax.numpy.asarray(sample_norm),
            )
            coupled = operator.coupling_current_moments(moments)
            internal = jax.numpy.r_[
                operator.grid.internal(coupled),
                operator.wall.internal(coupled),
                operator.sample.internal(coupled),
            ]
            jax.block_until_ready(internal)
            image = external + np.asarray(internal)
            images[name] = image
            forcing = image - np.asarray(seed)
            arms[name] = {
                "absolute_value_source": value_name,
                "normalization_constants": constant_name,
                "axis_flux_wb": float(anchor["axis_flux_wb"]),
                "boundary_flux_wb": float(anchor["boundary_flux_wb"]),
                "grid_core_forcing_wb": finite_statistics(
                    forcing[: len(machine.node)][core]
                ),
                "state_forcing_wb": finite_statistics(forcing),
            }
    production_control = "interpolated_values_map_saddle_constants"
    direct = np.asarray(operator.flux_map()(seed))
    control_difference = finite_statistics(images[production_control] - direct)
    prior = json.loads(DISCRIMINATOR_PATH.read_text(encoding="utf-8"))
    prior_map = prior["arms"]["production_values_production_constants"]["forcing"]
    prior_declared = prior["arms"]["production_values_exact_constants"]["forcing"]
    return {
        "factors": {
            "absolute_values": ["interpolated", "map_exact"],
            "normalization_constants": ["map_saddle", "declared"],
            "held_fixed": "declared IDS p-prime and FF-prime source functions",
        },
        "anchors": choices,
        "arms": arms,
        "banked_arm_correction": {
            "statement": (
                "The banked 1.294 Wb declared-profile arm used map-saddle "
                "topology constants for its density psi_N, not declared constants."
            ),
            "direct_path_control_difference_wb": control_difference,
            "banked_grid_core_forcing_sup_wb": json.loads(
                BASE_RECEIPT_PATH.read_text(encoding="utf-8")
            )["reproduction_lane_forcing"]["coarse"]["routes"]["declared"][
                "grid_core_forcing_wb"
            ]["sup"],
            "rerun_grid_core_forcing_sup_wb": arms[production_control][
                "grid_core_forcing_wb"
            ]["sup"],
        },
        "prior_discriminator_reconciliation": {
            "source": str(DISCRIMINATOR_PATH),
            "map_saddle_constants_forcing_sup_wb": prior_map["sup_wb"],
            "declared_constants_forcing_sup_wb": prior_declared["sup_wb"],
            "statement": (
                "The independently banked density-only discriminator collapses "
                "from 1.2523 to 0.0595 Wb under declared constants; this rerun "
                "uses the same declared source and the same two factors."
            ),
        },
    }


def forcing_image(reference, case, machine, coordinate, p_prime, ff_prime):
    """Return one route image and its forcing statistics on the coarse carrier."""
    route_case = replace(
        case,
        psi_norm=np.asarray(coordinate),
        p_prime=np.asarray(p_prime),
        ff_prime=np.asarray(ff_prime),
    )
    operator = reference.forward_operator(route_case, machine)
    seed = reference.seed_flux(case, machine)
    image = operator.flux_map()(seed)
    jax.block_until_ready(image)
    image = np.asarray(image)
    forcing = image - np.asarray(seed)
    masks = operator._support_partition(seed)[0]
    core = np.asarray(masks.core, dtype=bool)
    return image, {
        "grid_core_forcing_wb": finite_statistics(forcing[: len(machine.node)][core]),
        "state_forcing_wb": finite_statistics(forcing),
        "profile_support": {
            "minimum_psi_norm": float(np.min(coordinate)),
            "maximum_psi_norm": float(np.max(coordinate)),
            "samples": int(len(coordinate)),
        },
    }


def coordinate_control(reference, case, machine, base, extended) -> dict[str, object]:
    """Separate map-coordinate normalization from endpoint support extension."""
    base_reliable = np.asarray(base.reliable, dtype=bool)
    extended_reliable = np.asarray(extended.reliable, dtype=bool)
    banked_receipt = json.loads(BASE_RECEIPT_PATH.read_text(encoding="utf-8"))
    declared_coordinate = np.asarray(
        banked_receipt["common_profile_base"]["psi_norm_declared"]
    )
    definitions = {
        "banked_declared_coordinate_base_support": (
            declared_coordinate[base_reliable],
            np.asarray(base.p_prime)[base_reliable],
            np.asarray(base.ff_prime)[base_reliable],
        ),
        "map_coordinate_base_support": (
            np.asarray(base.psi_norm)[base_reliable],
            np.asarray(base.p_prime)[base_reliable],
            np.asarray(base.ff_prime)[base_reliable],
        ),
        "map_coordinate_extended_support": (
            np.asarray(extended.psi_norm)[extended_reliable],
            np.asarray(extended.p_prime)[extended_reliable],
            np.asarray(extended.ff_prime)[extended_reliable],
        ),
    }
    images = {}
    arms = {}
    for name, definition in definitions.items():
        images[name], arms[name] = forcing_image(reference, case, machine, *definition)
    base_name = "banked_declared_coordinate_base_support"
    natural_name = "map_coordinate_base_support"
    extended_name = "map_coordinate_extended_support"
    banked_sup = banked_receipt["reproduction_lane_forcing"]["coarse"]["routes"]
    banked_sup = banked_sup["map_extracted"]["grid_core_forcing_wb"]["sup"]
    return {
        "arms": arms,
        "extended_extraction": {
            "requested_shells": int(len(extended.psi_norm)),
            "reliable_shells": int(np.count_nonzero(extended_reliable)),
            "reliable_psi_norm_map": np.asarray(extended.psi_norm)[extended_reliable],
            "unreliable_psi_norm_map": np.asarray(extended.psi_norm)[
                ~extended_reliable
            ],
            "sample_count": np.asarray(extended.sample_count),
        },
        "controls": {
            "banked_forcing_sup_wb": float(banked_sup),
            "banked_rerun_difference_wb": arms[base_name]["grid_core_forcing_wb"]["sup"]
            - banked_sup,
            "coordinate_only_image_difference_wb": finite_statistics(
                images[natural_name] - images[base_name]
            ),
            "support_extension_only_image_difference_wb": finite_statistics(
                images[extended_name] - images[natural_name]
            ),
            "combined_image_difference_wb": finite_statistics(
                images[extended_name] - images[base_name]
            ),
        },
        "interpretation": (
            "The middle arm changes only the profile coordinate; the final arm "
            "then adds reliable endpoint shells. Route disagreement remains "
            "reported and is not blended into the declared route."
        ),
    }


def render(report: dict[str, object]) -> str:
    """Render the shell, factorial, and coordinate controls in one figure."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    shells = report["rotation_and_variance"]["shells"]
    psi = np.asarray([row["psi_norm_map"] for row in shells])
    explained = np.asarray(
        [row["static_explained_variance_fraction"] for row in shells]
    )
    ratio = np.asarray(
        [row["static_projection_rms_rj_phi_a_per_m"] for row in shells]
    ) / np.asarray([row["signal_rms_rj_phi_a_per_m"] for row in shells])
    significance = np.asarray(
        [row["rotation_amplitude_over_uncertainty"] for row in shells]
    )
    reduction = np.asarray([row["residual_rms_reduction_fraction"] for row in shells])
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), layout="constrained")
    axes[0, 0].plot(psi, explained, "o-", label="explained variance")
    axes[0, 0].plot(psi, ratio, "s--", label="residual / signal RMS")
    axes[0, 0].set_xlabel("map psi_N")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title("Two-function shell projection")
    axes[0, 1].plot(psi, significance, "o-", label="|amplitude| / uncertainty")
    axes[0, 1].plot(psi, reduction, "s--", label="RMS reduction fraction")
    axes[0, 1].axhline(1.96, color="black", linewidth=0.8)
    axes[0, 1].set_xlabel("map psi_N")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_title("Centrifugal-shape diagnostic")
    factorial = report["forcing_reconciliation"]["arms"]
    matrix = np.array(
        [
            [
                factorial[f"{value}_values_{constant}_constants"][
                    "grid_core_forcing_wb"
                ]["sup"]
                for constant in ("map_saddle", "declared")
            ]
            for value in ("interpolated", "map_exact")
        ]
    )
    image = axes[1, 0].imshow(matrix, aspect="auto")
    axes[1, 0].set_xticks((0, 1), ("map saddle", "declared"))
    axes[1, 0].set_yticks((0, 1), ("interpolated", "map exact"))
    axes[1, 0].set_xlabel("normalization constants")
    axes[1, 0].set_ylabel("absolute values")
    axes[1, 0].set_title("Declared-route forcing sup [Wb]")
    for row in range(2):
        for column in range(2):
            axes[1, 0].text(
                column,
                row,
                f"{matrix[row, column]:.4f}",
                ha="center",
                va="center",
                color=(
                    "white" if matrix[row, column] > 0.55 * matrix.max() else "black"
                ),
            )
    figure.colorbar(image, ax=axes[1, 0], shrink=0.8)
    coordinate = report["extracted_route_coordinate_control"]["arms"]
    names = (
        "banked_declared_coordinate_base_support",
        "map_coordinate_base_support",
        "map_coordinate_extended_support",
    )
    values = [coordinate[name]["grid_core_forcing_wb"]["sup"] for name in names]
    axes[1, 1].bar(("banked", "map coord", "map + endpoints"), values)
    axes[1, 1].set_ylabel("grid-core forcing sup [Wb]")
    axes[1, 1].set_title("Extracted-route coordinate controls")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    path = FIGURES / "route_controls.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return str(path)


def jsonable(value):
    """Convert arrays and non-finite qualifications into strict JSON values."""
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def main() -> None:
    """Run and serialize the three requested controls."""
    reference = load_module(REFERENCE_PATH, "dina_route_control_reference")
    mapping = load_module(MAPPING_PATH, "dina_route_control_mapping")
    reference.configure_dtypes()
    stored = mapping.read_stored_profiles()
    mapped = mapping.derive_mapping(stored)
    case = reference.require_reference()
    anchors = json.loads(ANCHOR_PATH.read_text(encoding="utf-8"))
    machine, identity, store = load_coarse_machine(reference, case)
    base = extract_surfaces(mapped, stored, anchors, BASE_SURFACES)
    extended = extract_surfaces(mapped, stored, anchors, EXTENDED_SURFACES)
    rotation = rotation_fit_receipt(mapped, stored, anchors, base)
    report = {
        "schema": "nova.dina-dual-route-addendum",
        "reference": {
            "pulse": 135011,
            "run": 7,
            "time_slice": 353,
            "time_s": float(stored.time),
            "dd_version": "3.39.0",
            "reader": "imas.DBEntry",
        },
        "carrier": {
            "semantic_key": COARSE_CACHE_KEY,
            "store": str(store.filepath),
            "realised_cells": int(len(machine.node)),
            "state_nodes": int(len(reference.seed_flux(case, machine))),
            "route_policy": identity["routes"],
        },
        "rotation_and_variance": rotation,
        "rotational_pressure_constraint": rotational_constraint_receipt(),
        "forcing_reconciliation": density_factorial(reference, case, machine, anchors),
        "extracted_route_coordinate_control": coordinate_control(
            reference, case, machine, base, extended
        ),
    }
    aggregate = rotation["aggregate"]
    report["two_function_balance_verdict"] = {
        "consistent": bool(aggregate["static_explained_variance_fraction"] > 0.99),
        "statement": (
            "The stored map is consistent with a two-function Grad-Shafranov "
            f"balance at {100 * aggregate['static_residual_to_signal_fraction']:.4f}% "
            "weighted R*j_phi residual RMS; the rotation-shaped diagnostic is "
            "reported separately and does not reconcile the two source routes."
        ),
        "weighted_signal_rms_rj_phi_a_per_m": aggregate["signal_rms_rj_phi_a_per_m"],
        "weighted_static_residual_rms_rj_phi_a_per_m": aggregate[
            "static_projection_rms_rj_phi_a_per_m"
        ],
        "weighted_static_explained_variance_fraction": aggregate[
            "static_explained_variance_fraction"
        ],
    }
    report["figure"] = render(report)
    (OUTPUT / "addendum.json").write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "DINA_ROUTE_CONTROLS_EXIT=0 "
        f"static_explained={aggregate['static_explained_variance_fraction']:.9f} "
        f"rotation_significant={aggregate['significant_rotation_proxy_shells']}/"
        f"{aggregate['total_shells']}"
    )


if __name__ == "__main__":
    main()
