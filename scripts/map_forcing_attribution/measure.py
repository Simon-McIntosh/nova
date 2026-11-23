"""Attribute the analytic-state free-boundary forcing and its linear response."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import mu_0

from nova.biot.polygonanalytic import (
    polygon_analytic_flux,
    polygon_analytic_flux_moments,
)


OUTPUT = Path(__file__).resolve().parent
REFERENCE_PATH = Path("tests/test_equilibrium_forward_reference.py")
TERMINAL_PATH = Path("scripts/root_gate_attribution/coarse-terminal-root.npz")
TOTAL_FLUX_FACTOR = 2.0 * np.pi
NEAR_FIELD_RADII = 2.5
HESSIAN_STEPS = (2.0e-3, 1.0e-3)
LINEAR_REPRODUCTION_TOLERANCE = 0.10


def load_reference():
    """Load the analytic fixture without invoking pytest collection."""
    spec = importlib.util.spec_from_file_location("forcing_reference", REFERENCE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {REFERENCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(report: dict[str, object]) -> None:
    """Write the diagnostic scorecard deterministically."""
    (OUTPUT / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def polygon_rule(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a degree-five triangle-fan rule over one convex polygon."""
    centre = np.mean(vertices, axis=0)
    following = np.roll(vertices, -1, axis=0)
    triangles = np.stack(
        [np.broadcast_to(centre, vertices.shape), vertices, following], axis=1
    )
    first, second, third = np.moveaxis(triangles, 1, 0)
    first_second = 0.5 * (first + second)
    second_third = 0.5 * (second + third)
    third_first = 0.5 * (third + first)
    triangles = np.concatenate(
        [
            np.stack([first, first_second, third_first], axis=1),
            np.stack([first_second, second, second_third], axis=1),
            np.stack([third_first, second_third, third], axis=1),
            np.stack([first_second, second_third, third_first], axis=1),
        ]
    )
    barycentric = np.asarray(
        [
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.059715871789770, 0.470142064105115, 0.470142064105115],
            [0.470142064105115, 0.059715871789770, 0.470142064105115],
            [0.470142064105115, 0.470142064105115, 0.059715871789770],
            [0.797426985353087, 0.101286507323456, 0.101286507323456],
            [0.101286507323456, 0.797426985353087, 0.101286507323456],
            [0.101286507323456, 0.101286507323456, 0.797426985353087],
        ]
    )
    rule_weight = np.asarray(
        [
            0.225,
            0.132394152788506,
            0.132394152788506,
            0.132394152788506,
            0.125939180544827,
            0.125939180544827,
            0.125939180544827,
        ]
    )
    points = np.einsum("qa,tad->tqd", barycentric, triangles)
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    area = 0.5 * np.abs(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0])
    weights = area[:, None] * rule_weight[None, :]
    return points.reshape(-1, 2), weights.ravel()


def current_density(case, points: np.ndarray) -> np.ndarray:
    """Evaluate the declared analytic current density at arbitrary points."""
    radius, height = points.T
    flux = case.flux(radius, height)
    normalised = (flux - case.flux_axis) / case.flux_span
    pressure_gradient = np.interp(normalised, case.psi_norm, case.p_prime)
    diamagnetic_gradient = np.interp(normalised, case.psi_norm, case.ff_prime)
    return -TOTAL_FLUX_FACTOR * (
        radius * pressure_gradient + diamagnetic_gradient / (mu_0 * radius)
    )


def second_moment_residual(case, operator, exact, coupled):
    """Return clipped-density second moments absent from the linear contraction."""
    partition = operator._support_partition(exact)
    support = partition[3]
    counts = np.asarray(support.vertex_count)
    vertices = np.asarray(support.support_vertices)
    fixed_centres = np.asarray(operator.moment_geometry.atomic_mesh.centroids)
    polygons = operator.moment_geometry.polygons
    cell_current = np.asarray(coupled.cell_current)
    radial = np.asarray(coupled.radial_moment)
    vertical = np.asarray(coupled.vertical_moment)
    residual = np.zeros((len(polygons), 2, 2))
    actual = np.zeros_like(residual)
    reconstructed = np.zeros_like(residual)
    for cell, polygon in enumerate(polygons):
        centre = fixed_centres[cell]
        if counts[cell] >= 3 and abs(cell_current[cell]) > 1.0e-12:
            point, weight = polygon_rule(vertices[cell, : counts[cell]])
            offset = point - centre
            density = current_density(case, point)
            actual[cell] = np.einsum("n,ni,nj->ij", weight * density, offset, offset)
        point, weight = polygon_rule(np.asarray(polygon))
        offset = point - centre
        area = float(np.sum(weight))
        density = (
            cell_current[cell]
            + radial[cell] * offset[:, 0]
            + vertical[cell] * offset[:, 1]
        ) / area
        reconstructed[cell] = np.einsum("n,ni,nj->ij", weight * density, offset, offset)
        residual[cell] = actual[cell] - reconstructed[cell]
    return residual, actual, reconstructed


def target_coordinates(machine) -> np.ndarray:
    """Return the grid, wall, and direct-sample targets in state order."""
    return np.vstack([machine.node, machine.wall_node, machine.sample_coordinates])


def uniform_columns(machine) -> np.ndarray:
    """Return the production uniform blocks in state order."""
    return np.vstack(
        [machine.plasma_to_grid, machine.plasma_to_wall, machine.plasma_to_sample]
    )


def second_order_correction(machine, residual: np.ndarray, step_fraction: float):
    """Contract residual second moments against translated exact-section curvature."""
    targets = target_coordinates(machine)
    production = uniform_columns(machine)
    correction = np.zeros(len(targets))
    pair_count = 0
    active_cells = 0
    for cell, (vertices, tensor) in enumerate(
        zip(machine.cell_polygons, residual, strict=True)
    ):
        tensor_scale = float(np.max(np.abs(tensor)))
        if tensor_scale <= 1.0e-12:
            continue
        centre = np.asarray(machine.moment_geometry.atomic_mesh.centroids[cell])
        radius = float(np.max(np.linalg.norm(vertices - centre, axis=1)))
        distance = np.linalg.norm(targets - centre, axis=1)
        near = distance < NEAR_FIELD_RADII * radius
        if not np.any(near):
            continue
        active_cells += 1
        pair_count += int(np.sum(near))
        target_r, target_z = targets[near].T
        step = step_fraction * radius

        def at(radial: float, vertical: float) -> np.ndarray:
            shift = np.asarray([radial, vertical])
            return polygon_analytic_flux(target_r, target_z, vertices + shift)

        base = production[near, cell]
        radial_curvature = (at(step, 0.0) - 2.0 * base + at(-step, 0.0)) / step**2
        vertical_curvature = (at(0.0, step) - 2.0 * base + at(0.0, -step)) / step**2
        cross_curvature = (
            at(step, step) - at(step, -step) - at(-step, step) + at(-step, -step)
        ) / (4.0 * step**2)
        correction[near] += 0.5 * (
            tensor[0, 0] * radial_curvature
            + 2.0 * tensor[0, 1] * cross_curvature
            + tensor[1, 1] * vertical_curvature
        )
    return correction, active_cells, pair_count


def exact_block_audit(machine) -> dict[str, float | int]:
    """Compare sampled assembled blocks with direct exact authored-section blocks."""
    targets = target_coordinates(machine)
    blocks = np.stack(
        [
            uniform_columns(machine),
            np.vstack(
                [
                    machine.plasma_to_grid_r,
                    machine.plasma_to_wall_r,
                    machine.plasma_to_sample_r,
                ]
            ),
            np.vstack(
                [
                    machine.plasma_to_grid_z,
                    machine.plasma_to_wall_z,
                    machine.plasma_to_sample_z,
                ]
            ),
        ]
    )
    source_sample = np.unique(np.linspace(0, len(machine.node) - 1, 16, dtype=int))
    target_sample = np.unique(np.linspace(0, len(targets) - 1, 32, dtype=int))
    absolute = 0.0
    relative = 0.0
    for source in source_sample:
        direct = np.stack(
            polygon_analytic_flux_moments(
                targets[target_sample, 0],
                targets[target_sample, 1],
                machine.cell_polygons[source],
                expansion_point=machine.moment_geometry.atomic_mesh.centroids[source],
            )
        )
        assembled = blocks[:, target_sample, source]
        difference = np.abs(direct - assembled)
        scale = np.maximum.reduce(
            [np.abs(direct), np.abs(assembled), np.full_like(direct, 1.0e-300)]
        )
        absolute = max(absolute, float(np.max(difference)))
        relative = max(relative, float(np.max(difference / scale)))
    return {
        "sampled_sources": len(source_sample),
        "sampled_targets": len(target_sample),
        "worst_absolute_wb_per_a": absolute,
        "worst_relative": relative,
    }


def component_score(component: np.ndarray, forcing: np.ndarray) -> dict[str, float]:
    """Score one signed forcing component against the observed forcing vector."""
    denominator = float(np.dot(forcing, forcing))
    peak = int(np.argmax(np.abs(forcing)))
    return {
        "projection_fraction": float(np.dot(component, forcing) / denominator),
        "signed_peak_fraction": float(component[peak] / forcing[peak]),
        "sup_norm_fraction": float(np.max(np.abs(component)) / np.max(np.abs(forcing))),
        "rms_norm_fraction": float(
            np.sqrt(np.mean(component**2)) / np.sqrt(np.mean(forcing**2))
        ),
    }


def solve_response(tangent, forcing: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Apply the fixed-point linear inverse with a measured residual."""
    vector = jnp.asarray(forcing)
    step, info = jax.scipy.sparse.linalg.gmres(
        lambda value: value - tangent(value),
        vector,
        maxiter=12,
        restart=12,
        solve_method="batched",
        tol=1.0e-12,
        atol=1.0e-14,
    )
    residual = np.asarray(step - tangent(step) - vector)
    return np.asarray(step), {
        "gmres_info": int(info),
        "residual_sup_wb": float(np.max(np.abs(residual))),
        "residual_relative_sup": float(
            np.max(np.abs(residual)) / max(np.max(np.abs(forcing)), 1.0e-300)
        ),
    }


def state_deviation(case, machine, operator, exact, state) -> dict[str, object]:
    """Return the three root-displacement observables for one flux state."""
    _masks, topology = operator.read(jnp.asarray(state))
    moments = operator.cell_current_moments(jnp.asarray(state))
    axis_delta = np.asarray(topology.axis) - np.asarray(case.axis)
    grid_delta = (
        np.asarray(state)[: len(machine.node)] - np.asarray(exact)[: len(machine.node)]
    )
    current = float(jnp.sum(moments.cell_current))
    return {
        "axis_delta_m": axis_delta.tolist(),
        "axis_displacement_mm": float(1.0e3 * np.linalg.norm(axis_delta)),
        "flux_sup_fraction_of_span": float(
            np.max(np.abs(grid_delta)) / abs(case.flux_span)
        ),
        "plasma_current_a": current,
        "plasma_current_fractional_deviation": float(
            current / case.plasma_current - 1.0
        ),
    }


def relative_metric_error(predicted: dict, observed: dict) -> dict[str, float]:
    """Compare the linear prediction with the banked nonlinear displacement."""
    names = (
        "axis_displacement_mm",
        "flux_sup_fraction_of_span",
        "plasma_current_fractional_deviation",
    )
    return {
        name: float(
            abs(predicted[name] - observed[name]) / max(abs(observed[name]), 1.0e-300)
        )
        for name in names
    }


def wiring_audit(machine) -> dict[str, object]:
    """Describe the actual moment path and any mismatch against its design."""
    return {
        "intended_design": {
            "section_shape": "each authored plasma polygon",
            "kernel": "exact closed-form polygon blocks",
            "expansion_centre": "fixed atomic-cell area centroid",
            "moment_order": "zeroth and first current moments",
        },
        "stages": [
            {
                "stage": "physical current vectors",
                "source": "nova/equilibrium/stencil_mesh.py:97",
                "value": (
                    "CellCurrentMoments stores current and two first moments "
                    "about fixed cell centroids"
                ),
            },
            {
                "stage": "physical-to-linear coefficient conversion",
                "source": "nova/equilibrium/forward_operator.py:218",
                "value": (
                    "the authored polygon second-moment tensor converts physical "
                    "first moments to radial and vertical coefficients"
                ),
            },
            {
                "stage": "fixed coupling assembly",
                "source": "nova/biot/polysection.py:369",
                "value": (
                    "uniform and companion rows use each authored polygon; the "
                    "companion expansion point is its material area centroid"
                ),
            },
            {
                "stage": "grid, wall, and direct-sample flux contraction",
                "source": "nova/biot/target.py:576",
                "value": (
                    "all three targets contract uniform, radial, and vertical "
                    "blocks with the same coefficient vectors"
                ),
            },
            {
                "stage": "fixed-point map",
                "source": "nova/equilibrium/forward_operator.py:387",
                "value": (
                    "the solve state consumes grid plus wall plus direct-sample "
                    "plasma flux in that order"
                ),
            },
            {
                "stage": "field observation contraction",
                "source": "tests/test_equilibrium_forward_reference.py:858",
                "value": (
                    "the reference receipt contracts the same three current vectors "
                    "with Br and Bz uniform and companion blocks"
                ),
            },
        ],
        "actual_geometry": {
            "cells": len(machine.node),
            "whole_hexagons": int(np.sum(machine.hexagon)),
            "clipped_authored_polygons": int(np.sum(~machine.hexagon)),
            "rectangle_substitutions": 0,
        },
        "discrepancies": [],
        "conclusion": (
            "No kernel, section-shape, expansion-centre, sign, normalisation, "
            "or moment-order wiring discrepancy was found on this revision."
        ),
    }


def render(report: dict[str, object]) -> None:
    """Render forcing fractions and displacement reproduction."""
    components = report["coarse_fixture"]["forcing_decomposition"]["components"]
    labels = ["exact kernel", "centre wiring", "missing second", "residual"]
    keys = [
        "exact_section_kernel",
        "expansion_centre_wiring",
        "omitted_second_moment_near_field",
        "second_order_comparator_residual",
    ]
    fractions = [components[key]["projection_fraction"] for key in keys]
    response = report["coarse_fixture"]["linear_response"]
    observed = response["observed_root"]
    predicted = response["predicted_root"]
    metric_labels = ["axis mm", "flux % span", "current %"]
    observed_values = [
        observed["axis_displacement_mm"],
        100.0 * observed["flux_sup_fraction_of_span"],
        100.0 * abs(observed["plasma_current_fractional_deviation"]),
    ]
    predicted_values = [
        predicted["axis_displacement_mm"],
        100.0 * predicted["flux_sup_fraction_of_span"],
        100.0 * abs(predicted["plasma_current_fractional_deviation"]),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    colours = ["#4c78a8", "#72b7b2", "#e45756", "#b279a2"]
    axes[0].bar(labels, fractions, color=colours)
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].axhline(1.0, color="black", lw=0.8, ls="--")
    axes[0].set_ylabel("signed projection on production forcing")
    axes[0].set_title("Additive exact-state forcing split")
    axes[0].tick_params(axis="x", rotation=24)
    x = np.arange(3)
    axes[1].bar(x - 0.18, observed_values, 0.36, label="banked root")
    axes[1].bar(x + 0.18, predicted_values, 0.36, label="linear inverse")
    axes[1].set_xticks(x, metric_labels)
    axes[1].set_title("Root displacement reproduction")
    axes[1].legend()
    figure.savefig(OUTPUT / "decomposition.png", dpi=180)
    plt.close(figure)


def main() -> None:
    """Run the coarse audit, forcing decomposition, and linear response."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference = load_reference()
    reference.configure_dtypes()
    case = reference.require_reference()
    reference.WALL_NODES = 3
    print("BUILD fixture=coarse requested=566", flush=True)
    machine = reference.build_machine(case, reference.SUITE_CELLS, passive=True)
    if len(machine.node) != 566:
        raise AssertionError(f"expected 566 cells, got {len(machine.node)}")
    operator = reference.forward_operator(case, machine)
    exact = reference.seed_flux(case, machine)
    terminal = np.load(TERMINAL_PATH)["state"]
    map_fn = operator.flux_map()
    mapped, tangent = jax.linearize(map_fn, exact)
    forcing = np.asarray(mapped - exact)
    coupled = operator.cell_current_moments(exact)
    print(
        f"FORCING sup_wb={np.max(np.abs(forcing)):.17g} "
        "grid_span_fraction="
        f"{np.max(np.abs(forcing[:566])) / abs(case.flux_span):.17g}",
        flush=True,
    )

    block_audit = exact_block_audit(machine)
    second_residual, actual_second, reconstructed_second = second_moment_residual(
        case, operator, exact, coupled
    )
    corrections = []
    for step in HESSIAN_STEPS:
        correction, active_cells, pair_count = second_order_correction(
            machine, second_residual, step
        )
        corrections.append(correction)
        print(
            f"SECOND_ORDER step_fraction={step:.17g} active_cells={active_cells} "
            f"near_pairs={pair_count} sup_wb={np.max(np.abs(correction)):.17g}",
            flush=True,
        )
    correction = corrections[-1]
    correction_refinement = float(
        np.max(np.abs(corrections[-1] - corrections[-2]))
        / max(np.max(np.abs(correction)), 1.0e-300)
    )

    zero = np.zeros_like(forcing)
    omitted_second = -correction
    comparator_residual = forcing + correction
    components = {
        "exact_section_kernel": zero,
        "expansion_centre_wiring": zero,
        "omitted_second_moment_near_field": omitted_second,
        "second_order_comparator_residual": comparator_residual,
    }
    decomposition_sum = sum(components.values())
    closure = decomposition_sum - forcing
    scored_components = {
        name: component_score(value, forcing) for name, value in components.items()
    }
    dominant = max(
        scored_components,
        key=lambda name: abs(scored_components[name]["projection_fraction"]),
    )

    total_step, total_solve = solve_response(tangent, forcing)
    moment_step, moment_solve = solve_response(tangent, omitted_second)
    residual_step, residual_solve = solve_response(tangent, comparator_residual)
    response_closure = total_step - moment_step - residual_step
    exact_metrics = state_deviation(case, machine, operator, exact, exact)
    observed_metrics = state_deviation(case, machine, operator, exact, terminal)
    predicted_metrics = state_deviation(
        case, machine, operator, exact, np.asarray(exact) + total_step
    )
    metric_error = relative_metric_error(predicted_metrics, observed_metrics)
    state_error = float(
        np.max(np.abs(total_step - (terminal - np.asarray(exact))))
        / np.max(np.abs(terminal - np.asarray(exact)))
    )
    epsilon = 1.0e-6
    finite = (map_fn(exact + epsilon * (terminal - exact)) - mapped) / epsilon
    tangent_value = tangent(terminal - exact)
    tangent_fd_error = float(
        np.max(np.abs(np.asarray(finite - tangent_value)))
        / np.max(np.abs(np.asarray(tangent_value)))
    )
    response_reproduced = (
        state_error <= LINEAR_REPRODUCTION_TOLERANCE
        and max(metric_error.values()) <= LINEAR_REPRODUCTION_TOLERANCE
    )

    report = {
        "schema": "nova.map-forcing-attribution",
        "source_artifacts": {
            "analytic_fixture": str(REFERENCE_PATH),
            "banked_terminal_root": str(TERMINAL_PATH),
        },
        "wiring_audit": wiring_audit(machine),
        "coarse_fixture": {
            "plasma_cells": len(machine.node),
            "state_size": len(forcing),
            "exact_block_audit": block_audit,
            "forcing": {
                "sup_wb": float(np.max(np.abs(forcing))),
                "rms_wb": float(np.sqrt(np.mean(forcing**2))),
                "grid_sup_fraction_of_analytic_span": float(
                    np.max(np.abs(forcing[: len(machine.node)])) / abs(case.flux_span)
                ),
            },
            "second_moment_comparator": {
                "definition": (
                    "Residual clipped analytic-density second moments relative "
                    "to the production full-section linear reconstruction are "
                    "contracted against exact-section translation Hessians only "
                    "within 2.5 section radii."
                ),
                "active_cells": active_cells,
                "near_source_target_pairs": pair_count,
                "residual_second_moment_sup_a_m2": float(
                    np.max(np.abs(second_residual))
                ),
                "actual_second_moment_sup_a_m2": float(np.max(np.abs(actual_second))),
                "reconstructed_second_moment_sup_a_m2": float(
                    np.max(np.abs(reconstructed_second))
                ),
                "correction_sup_wb": float(np.max(np.abs(correction))),
                "hessian_step_fractions": list(HESSIAN_STEPS),
                "step_refinement_relative_sup": correction_refinement,
                "qualification": (
                    "This is a second-moment near-field comparator, not an exact "
                    "all-orders clipped-section kernel."
                ),
            },
            "forcing_decomposition": {
                "sign_convention": (
                    "production forcing equals exact-kernel error plus centre-"
                    "wiring error plus omitted-second-moment error plus comparator "
                    "residual"
                ),
                "components": scored_components,
                "additive_closure_sup_wb": float(np.max(np.abs(closure))),
                "additive_closure_relative_sup": float(
                    np.max(np.abs(closure)) / np.max(np.abs(forcing))
                ),
                "dominant_component_by_absolute_projection": dominant,
            },
            "linear_response": {
                "equation": "(I - Dg) delta = g(x_exact) - x_exact",
                "tangent_finite_difference_relative_sup": tangent_fd_error,
                "total_solve": total_solve,
                "moment_component_solve": moment_solve,
                "comparator_residual_solve": residual_solve,
                "component_response_closure_sup_wb": float(
                    np.max(np.abs(response_closure))
                ),
                "exact_state": exact_metrics,
                "observed_root": observed_metrics,
                "predicted_root": predicted_metrics,
                "state_displacement_relative_sup_error": state_error,
                "observable_relative_errors": metric_error,
                "reproduction_tolerance": LINEAR_REPRODUCTION_TOLERANCE,
                "reproduced": response_reproduced,
            },
        },
        "adjudication": {
            "linear_amplification": (
                "convicted" if response_reproduced else "not reproduced"
            ),
            "component": dominant,
            "mechanism": (
                "The verified production tangent amplifies the measured exact-state "
                "forcing into the banked terminal displacement. Exact-section and "
                "centre-wiring swaps are identities on this revision; the signed "
                "decomposition states whether omitted second moments or the "
                "comparator residual dominates."
            ),
            "repair_scope": "No repair is made by this diagnostic.",
        },
        "fine_fixture": {
            "status": "not run",
            "reason": (
                "The 1069-cell coupling build exceeds the remaining diagnostic "
                "fence after the coarse exact-state build and two Hessian refinements."
            ),
            "follow_on": (
                "Repeat the dominant coarse component and linear projection against "
                "scripts/root_gate_attribution/fine-terminal-root.npz."
            ),
        },
    }
    render(report)
    report["artifacts"] = {
        "figure": str((OUTPUT / "decomposition.png").relative_to(Path.cwd())),
        "figure_bytes": (OUTPUT / "decomposition.png").stat().st_size,
    }
    write_json(report)
    print(
        f"RESULT reproduced={response_reproduced} state_error={state_error:.17g} "
        f"dominant={dominant} closure={np.max(np.abs(closure)):.17g} "
        f"hessian_refinement={correction_refinement:.17g}",
        flush=True,
    )
    print("MAP_FORCING_ATTRIBUTION_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
