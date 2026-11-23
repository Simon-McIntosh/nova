"""Separate in-cell flux values from the constants that normalise them."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path(__file__).resolve().parent
REFERENCE_PATH = Path("tests/test_equilibrium_forward_reference.py")
FORCING_PATH = Path("scripts/map_forcing_attribution/measure.py")
DECOMPOSITION_PATH = Path("scripts/forcing_residual_decomposition/measure.py")
GATE_PATH = Path("scripts/root_gate_attribution/measure_root_attribution.py")
LADDER_PATH = Path("scripts/density_forcing_ladder/measure.py")
TERMINAL_PATH = Path("scripts/root_gate_attribution/coarse-terminal-root.npz")
RESIDUAL_RESULTS = Path("scripts/forcing_residual_decomposition/results.json")
EXPECTED_CACHE_KEY = "746fbe1553c4b242"
EXPECTED_DENSITY_SHARE = 0.991762605972982
EXPECTED_DENSITY_SUP_WB = 1.252292371968979


def load_module(path: Path, name: str):
    """Load a repository measurement module without collecting its tests."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_state(path: Path) -> np.ndarray:
    """Read the serialized coarse terminal state."""
    with np.load(path) as stored:
        return stored["state"]


def error_statistics(values: np.ndarray) -> dict[str, float]:
    """Summarise one residual without discarding its sign."""
    values = np.asarray(values)
    return {
        "signed_mean": float(np.mean(values)),
        "mean_absolute": float(np.mean(np.abs(values))),
        "rms": float(np.sqrt(np.mean(values**2))),
        "sup": float(np.max(np.abs(values))),
    }


def normalise(values: np.ndarray, axis_flux: float, boundary_flux: float):
    """Form normalised flux from explicitly supplied affine constants."""
    return (np.asarray(values) - axis_flux) / (boundary_flux - axis_flux)


def render(report: dict[str, object]) -> None:
    """Render the four forcing and projected-response factorials."""
    arms = report["arms"]
    rows = ("production_values", "exact_values")
    columns = ("production_constants", "exact_constants")
    keys = [[f"{row}_{column}" for column in columns] for row in rows]
    panels = (
        (
            "Density forcing share",
            "forcing_share_percent",
            "% of comparator",
            "coolwarm",
        ),
        ("Forcing sup", ("forcing", "sup_wb"), "Wb", "viridis"),
        (
            "Projected axis response",
            ("root_response", "axis_signed_projection_mm"),
            "mm",
            "coolwarm",
        ),
        (
            "Projected flux response",
            ("root_response", "flux_signed_peak_percent_of_span"),
            "% of span",
            "coolwarm",
        ),
    )

    def value(arm, key):
        if isinstance(key, tuple):
            return arm[key[0]][key[1]]
        return arm[key]

    figure, axes = plt.subplots(2, 2, figsize=(10.8, 8.3), layout="constrained")
    for axis, (title, key, unit, colour) in zip(axes.flat, panels, strict=True):
        matrix = np.asarray(
            [[value(arms[name], key) for name in row] for row in keys], dtype=float
        )
        image = axis.imshow(matrix, cmap=colour, aspect="auto")
        axis.set_xticks(range(2), ["production", "exact"])
        axis.set_yticks(range(2), ["production", "exact"])
        axis.set_xlabel("normalisation constants")
        axis.set_ylabel("in-cell absolute flux values")
        axis.set_title(title)
        scale = max(float(np.max(np.abs(matrix))), 1.0e-300)
        for row in range(2):
            for column in range(2):
                contrast = abs(matrix[row, column]) > 0.55 * scale
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.5g} {unit}",
                    ha="center",
                    va="center",
                    color="white" if contrast else "black",
                    fontsize=9,
                )
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.savefig(OUTPUT / "factorial.png", dpi=180)
    plt.close(figure)


def main() -> None:
    """Evaluate the coarse exact state under the complete two-factor swap."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference = load_module(REFERENCE_PATH, "normalization_reference")
    forcing_module = load_module(FORCING_PATH, "normalization_forcing")
    decomposition = load_module(DECOMPOSITION_PATH, "normalization_decomposition")
    gate_module = load_module(GATE_PATH, "normalization_gate")
    ladder = load_module(LADDER_PATH, "normalization_ladder")
    reference.configure_dtypes()
    case = reference.require_reference()
    terminal = load_state(TERMINAL_PATH)
    residual_bank = json.loads(RESIDUAL_RESULTS.read_text(encoding="utf-8"))

    reference.WALL_NODES = 3
    machine = reference.cached_machine(case, reference.SUITE_CELLS, passive=True)
    receipt = machine.cache_receipt
    if receipt is None or not receipt.hit or receipt.key != EXPECTED_CACHE_KEY:
        raise AssertionError(
            "coarse machine did not warm-load from the semantic carrier"
        )
    print(reference.machine_cache_summary("coarse", machine), flush=True)
    operator = reference.forward_operator(case, machine)
    exact = reference.seed_flux(case, machine)
    if len(exact) != len(terminal):
        raise AssertionError("coarse exact and terminal states differ in shape")

    print(f"LINEARIZE state_size={len(exact)}", flush=True)
    mapped, tangent = jax.linearize(operator.flux_map(), exact)
    jax.block_until_ready(mapped)
    forcing = np.asarray(mapped - exact)
    coupled = operator.cell_current_moments(exact)
    second_residual, _actual, _reconstructed = forcing_module.second_moment_residual(
        case, operator, exact, coupled
    )
    print("COMPARATOR contracting omitted second moments", flush=True)
    correction, active_cells, pair_count = forcing_module.second_order_correction(
        machine, second_residual, forcing_module.HESSIAN_STEPS[-1]
    )
    comparator = forcing + correction

    partition = operator._support_partition(exact)
    masks, topology, sample_norm, support, _common_support = partition
    selection = np.asarray(masks.core | masks.common_sol)
    if len(operator._support_moment_stencils) != 1:
        raise AssertionError("coarse carrier no longer has one six-vertex stencil")
    stencil = operator._support_moment_stencils[0]
    if stencil.ring_profile_point.shape[1] != 672:
        raise AssertionError("coarse density projection sample count changed")

    ring = np.asarray(stencil.ring_centre)
    profile_points = np.asarray(stencil.ring_profile_point)
    centroid_absolute = np.asarray(exact)[: operator.grid.node_number]
    sample_absolute = np.asarray(operator.sample_node_flux(exact))
    absolute_pool = np.concatenate([centroid_absolute, sample_absolute])
    gathered_absolute = absolute_pool[np.asarray(stencil.ring_gather_index)]
    coefficient = np.einsum(
        "nps,ns->np", np.asarray(stencil.ring_flux_weight), gathered_absolute
    )
    production_values = np.einsum(
        "nqp,np->nq", np.asarray(stencil.ring_profile_flux_design), coefficient
    )
    exact_values = case.flux(profile_points[..., 0], profile_points[..., 1])

    production_axis = float(topology.axis_flux)
    production_boundary = float(topology.boundary_flux)
    exact_axis = float(case.flux_axis)
    exact_boundary = float(case.flux_boundary)
    production_span = production_boundary - production_axis
    exact_span = exact_boundary - exact_axis
    existing_pool = np.concatenate(
        [np.asarray(masks.psi_norm), np.asarray(sample_norm)]
    )
    existing_gather = existing_pool[np.asarray(stencil.ring_gather_index)]
    existing_coefficient = np.einsum(
        "nps,ns->np", np.asarray(stencil.ring_flux_weight), existing_gather
    )
    existing_profile_norm = np.einsum(
        "nqp,np->nq",
        np.asarray(stencil.ring_profile_flux_design),
        existing_coefficient,
    )
    split_identity_sup = float(
        np.max(
            np.abs(
                normalise(production_values, production_axis, production_boundary)
                - existing_profile_norm
            )
        )
    )
    if split_identity_sup > 2.0e-13:
        raise AssertionError(
            "absolute-value interpolation does not reproduce production"
        )

    definitions = {
        "production_values_production_constants": (
            production_values,
            production_axis,
            production_boundary,
        ),
        "exact_values_production_constants": (
            exact_values,
            production_axis,
            production_boundary,
        ),
        "production_values_exact_constants": (
            production_values,
            exact_axis,
            exact_boundary,
        ),
        "exact_values_exact_constants": (
            exact_values,
            exact_axis,
            exact_boundary,
        ),
    }

    _reference_support, exact_current, exact_first, _reference_coupled = (
        gate_module.support_reference(case, operator, exact, masks, topology)
    )
    external = np.asarray(operator.external())
    exact_image = external + decomposition.internal_flux(
        operator, exact_current, exact_first
    )
    evaluator = ladder.build_image_evaluator(
        operator, stencil, support, selection, None, None
    )
    compiled = jax.jit(evaluator)
    exact_profile_norm = normalise(exact_values, exact_axis, exact_boundary)
    exact_density = np.asarray(
        operator.source.core.current_density(
            jnp.asarray(profile_points[..., 0]), jnp.asarray(exact_profile_norm)
        )
    )

    arms = {}
    images = {}
    for name, (values, axis_flux, boundary_flux) in definitions.items():
        print(f"ARM name={name}", flush=True)
        profile_norm = normalise(values, axis_flux, boundary_flux)
        internal = compiled(jnp.asarray(profile_norm))
        jax.block_until_ready(internal)
        image = external + np.asarray(internal)
        images[name] = image
        component = image + correction - exact_image
        forcing_score = decomposition.score(component, comparator)
        step, solve = forcing_module.solve_response(tangent, component)
        response = decomposition.response_observables(
            case, machine, operator, exact, terminal, step
        )
        density = np.asarray(
            operator.source.core.current_density(
                jnp.asarray(profile_points[..., 0]), jnp.asarray(profile_norm)
            )
        )
        arms[name] = {
            "absolute_values": (
                "production_quadratic_interpolant"
                if values is production_values
                else "exact_analytic_at_density_points"
            ),
            "normalization_constants": (
                "production_topology"
                if axis_flux == production_axis and boundary_flux == production_boundary
                else "exact_case"
            ),
            "forcing": forcing_score,
            "forcing_share_percent": 100.0 * forcing_score["projection_fraction"],
            "root_response": {**solve, **response},
            "psi_norm_error_against_exact_constants": error_statistics(
                profile_norm - exact_profile_norm
            ),
            "density_error_against_exact_case": error_statistics(
                density - exact_density
            ),
        }

    control_name = "production_values_production_constants"
    control = arms[control_name]
    control_image_difference = float(
        np.max(np.abs(images[control_name] - np.asarray(mapped)))
    )
    control_difference = {
        "production_image_sup_wb": control_image_difference,
        "density_projection_fraction": (
            control["forcing"]["projection_fraction"] - EXPECTED_DENSITY_SHARE
        ),
        "density_forcing_sup_wb": (
            control["forcing"]["sup_wb"] - EXPECTED_DENSITY_SUP_WB
        ),
        "normalised_interpolation_identity_sup": split_identity_sup,
    }
    if max(abs(value) for value in control_difference.values()) > 2.0e-11:
        raise AssertionError(
            f"production control did not reproduce: {control_difference}"
        )

    shares = {name: arm["forcing"]["projection_fraction"] for name, arm in arms.items()}
    baseline = shares[control_name]
    exact_values_production = shares["exact_values_production_constants"]
    production_values_exact = shares["production_values_exact_constants"]
    exact_values_exact = shares["exact_values_exact_constants"]
    removals = {
        "exact_values_with_production_constants_percentage_points": 100.0
        * (baseline - exact_values_production),
        "exact_constants_with_production_values_percentage_points": 100.0
        * (baseline - production_values_exact),
        "exact_constants_with_exact_values_percentage_points": 100.0
        * (exact_values_production - exact_values_exact),
        "fully_exact_percentage_points": 100.0 * (baseline - exact_values_exact),
    }
    exact_values_near_full = abs(exact_values_production - baseline) <= 0.25 * abs(
        baseline
    )
    constants_collapse_production_values = (
        baseline - production_values_exact >= 0.75 * abs(baseline)
    )
    constants_collapse_exact_values = (
        exact_values_production - exact_values_exact >= 0.75 * abs(baseline)
    )
    if (
        exact_values_near_full
        and constants_collapse_production_values
        and constants_collapse_exact_values
    ):
        carrier = "normalization_constants"
        statement = (
            "Both exact-constant arms collapse while exact values with production "
            "constants retain the forcing, so the affine psi-norm constants carry "
            "the density error."
        )
    elif baseline - exact_values_production >= 0.75 * abs(baseline):
        carrier = "in_cell_flux_values"
        statement = (
            "Exact values collapse the production-constant arm, so the in-cell "
            "absolute flux values carry the density error."
        )
    else:
        carrier = "interaction_or_other"
        statement = (
            "Neither factor alone satisfies the collapse discriminator; the "
            "measured interaction is the finding."
        )

    saddle = residual_bank["oracle"]["core"]
    saddle_psi_norm = float(saddle["saddle_psi_norm"])
    report = {
        "schema": "nova.normalization-constant-discriminator",
        "fixture": {
            "plasma_cells": len(machine.node),
            "state_size": len(exact),
            "density_evaluation_points_per_cell": int(
                stencil.ring_profile_point.shape[1]
            ),
            "density_support_rings": len(ring),
            "cache": {
                "store": receipt.store,
                "semantic_key": receipt.key,
                "warm_hit": receipt.hit,
                "load_seconds": receipt.load_seconds,
                "arrays_verified": receipt.arrays_verified,
                "bytes_verified": receipt.bytes_verified,
            },
            "second_moment_comparator": {
                "active_cells": active_cells,
                "near_source_target_pairs": pair_count,
                "correction_sup_wb": float(np.max(np.abs(correction))),
            },
        },
        "held_fixed": [
            "exact analytic state and production tangent inverse",
            "production straight clipped supports and topology qualification",
            "degree-nine weighted density projection at 672 fixed points per cell",
            "exact clipped zeroth and first moments",
            "first-moment coupling blocks and second-moment comparator correction",
        ],
        "factorial": {
            "absolute_value_factor": [
                "production quadratic interpolant in absolute Wb",
                "exact analytic absolute flux at each density evaluation point",
            ],
            "normalization_factor": [
                "production topology-derived axis and boundary flux",
                "exact case axis and boundary flux",
            ],
            "contraction": (
                "Each arm converts the selected absolute values with the selected "
                "affine constants, then applies the same degree-nine density "
                "projection and production coupling."
            ),
        },
        "normalization_constants": {
            "offset_definition": "production minus exact case",
            "production_topology": {
                "axis_flux_wb": production_axis,
                "boundary_flux_wb": production_boundary,
                "flux_span_wb": production_span,
            },
            "exact_case": {
                "axis_flux_wb": exact_axis,
                "boundary_flux_wb": exact_boundary,
                "flux_span_wb": exact_span,
            },
            "offsets": {
                "axis_flux_wb": production_axis - exact_axis,
                "axis_flux_in_exact_psi_norm": (production_axis - exact_axis)
                / exact_span,
                "boundary_flux_wb": production_boundary - exact_boundary,
                "boundary_flux_in_exact_psi_norm": (
                    production_boundary - exact_boundary
                )
                / exact_span,
                "flux_span_wb": production_span - exact_span,
                "flux_span_fraction_of_exact": (production_span - exact_span)
                / exact_span,
            },
            "production_constants_in_exact_coordinate": {
                "axis_psi_norm": (production_axis - exact_axis) / exact_span,
                "boundary_psi_norm": (production_boundary - exact_axis) / exact_span,
            },
            "banked_clip_to_newton_saddle": {
                "clip_constant_psi_norm": 1.0,
                "newton_saddle_psi_norm": saddle_psi_norm,
                "clip_beyond_saddle_psi_norm": 1.0 - saddle_psi_norm,
                "banked_saddle_level_offset": float(saddle["saddle_level_offset"]),
            },
        },
        "arms": arms,
        "production_control_difference": control_difference,
        "factor_effects": {
            "banked_density_projection_percent": 100.0 * EXPECTED_DENSITY_SHARE,
            "forcing_projection_removal": removals,
            "classification_policy": (
                "Normalization constants carry the collapse when exact values with "
                "production constants remain within 25 percent of the production "
                "density share and switching constants removes at least 75 percent "
                "of that share in both value rows. Exact values carry it when they "
                "remove at least 75 percent under production constants."
            ),
        },
        "verdict": {
            "collapse_carrier": carrier,
            "exact_values_with_production_constants_near_full": exact_values_near_full,
            "exact_constants_collapse_with_production_values": (
                constants_collapse_production_values
            ),
            "exact_constants_collapse_with_exact_values": (
                constants_collapse_exact_values
            ),
            "statement": statement,
            "measurement_only": True,
        },
        "source_artifacts": {
            "density_ladder": "scripts/density_forcing_ladder/results.json",
            "forcing_decomposition": str(RESIDUAL_RESULTS),
            "terminal_root": str(TERMINAL_PATH),
        },
    }
    render(report)
    report["artifacts"] = {
        "figure": str(OUTPUT / "factorial.png"),
        "figure_bytes": (OUTPUT / "factorial.png").stat().st_size,
    }
    (OUTPUT / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"RESULT carrier={carrier} baseline_share={100.0 * baseline:.9f}% "
        f"exact_values_production_constants="
        f"{100.0 * exact_values_production:.9f}% "
        f"production_values_exact_constants="
        f"{100.0 * production_values_exact:.9f}% "
        f"exact_values_exact_constants={100.0 * exact_values_exact:.9f}%",
        flush=True,
    )
    print("NORMALIZATION_DISCRIMINATOR_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
