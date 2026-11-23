"""Adjudicate fitted and faithful coupling moments on one closed-form carrier."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import statistics
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import mu_0

from nova.equilibrium.observation import clipped_support_quadrature
from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures import measure as fixture


OUTPUT = Path(__file__).resolve().parent
TRUTH_ORDERS = (12, 18, 27, 40, 60)
TRUTH_RELATIVE_TOLERANCE = 2.0e-12
TRUTH_ABSOLUTE_TOLERANCE_A = 2.0e-8
TIMING_REPETITIONS = 15
STEEP_EDGE_FRACTION = 0.05
STEEP_PEDESTAL_CENTRE = 0.90
STEEP_PEDESTAL_WIDTH = 0.025
GPU_MAP_SECONDS = 1.1198455467820168e-3


@dataclass(frozen=True)
class PedestalProfile:
    """Finite-edge current density with a narrow normalised-flux pedestal."""

    amplitude: float

    def current_density(self, radius, psi_norm):
        del radius
        flux = jnp.asarray(psi_norm)
        taper = 0.5 * (
            1.0 - jnp.tanh((flux - STEEP_PEDESTAL_CENTRE) / STEEP_PEDESTAL_WIDTH)
        )
        return self.amplitude * (
            STEEP_EDGE_FRACTION + (1.0 - STEEP_EDGE_FRACTION) * taper
        )


def _smooth_density(case, radius: np.ndarray, psi_norm: np.ndarray) -> np.ndarray:
    """Evaluate the analytic rotating profile as a NumPy functional."""
    del psi_norm
    pressure_gradient = -case.pressure_flux_gradient / (2.0 * np.pi)
    field_gradient = -case.f_f_prime / (2.0 * np.pi)
    centrifugal = np.exp(case.rotation_parameter * (radius**2 - case.major_radius**2))
    return (
        -2.0
        * np.pi
        * (radius * pressure_gradient * centrifugal + field_gradient / (mu_0 * radius))
    )


def _pedestal_density(
    amplitude: float, radius: np.ndarray, psi_norm: np.ndarray
) -> np.ndarray:
    """Evaluate the steep profile with the same formula as ``PedestalProfile``."""
    del radius
    taper = 0.5 * (
        1.0 - np.tanh((psi_norm - STEEP_PEDESTAL_CENTRE) / STEEP_PEDESTAL_WIDTH)
    )
    return amplitude * (STEEP_EDGE_FRACTION + (1.0 - STEEP_EDGE_FRACTION) * taper)


def _quadratic_design(local: np.ndarray) -> np.ndarray:
    radial, vertical = local[..., 0], local[..., 1]
    return np.stack(
        [
            np.ones_like(radial),
            radial,
            vertical,
            radial**2,
            radial * vertical,
            vertical**2,
        ],
        axis=-1,
    )


def _flux_coefficients(stencil, centroid_flux, sample_flux) -> np.ndarray:
    pool = np.r_[np.asarray(centroid_flux), np.asarray(sample_flux)]
    gathered = pool[np.asarray(stencil.ring_gather_index)]
    return np.einsum("nps,ns->np", np.asarray(stencil.ring_flux_weight), gathered)


def _polygon_rule(vertices: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return tensor-Duffy points and area weights over a convex polygon."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    unit = 0.5 * (nodes + 1.0)
    unit_weight = 0.5 * weights
    u, v = np.meshgrid(unit, unit, indexing="ij")
    wu, wv = np.meshgrid(unit_weight, unit_weight, indexing="ij")
    u = u.ravel()
    v = v.ravel()
    rule_weight = (wu * wv).ravel()
    points = []
    area_weights = []
    first = vertices[0]
    for index in range(1, len(vertices) - 1):
        edge_first = vertices[index] - first
        edge_second = vertices[index + 1] - first
        cross = abs(edge_first[0] * edge_second[1] - edge_first[1] * edge_second[0])
        points.append(
            first
            + u[:, None] * edge_first
            + (1.0 - u)[:, None] * v[:, None] * edge_second
        )
        area_weights.append(cross * (1.0 - u) * rule_weight)
    return np.concatenate(points), np.concatenate(area_weights)


def _integrate_cell(
    vertices: np.ndarray,
    centre: np.ndarray,
    sampling_centre: np.ndarray,
    coordinate_scale: np.ndarray,
    flux_coefficient: np.ndarray,
    density,
    order: int,
) -> np.ndarray:
    points, weights = _polygon_rule(vertices, order)
    local = (points - sampling_centre) / coordinate_scale
    psi_norm = _quadratic_design(local) @ flux_coefficient
    current_density = density(points[:, 0], psi_norm)
    weighted = weights * current_density
    offset = points - centre
    return np.asarray(
        [
            np.sum(weighted),
            np.sum(weighted * offset[:, 0]),
            np.sum(weighted * offset[:, 1]),
        ]
    )


def adaptive_truth(
    support,
    selection: np.ndarray,
    stencil,
    coefficients: np.ndarray,
    centres: np.ndarray,
    cell_width: np.ndarray,
    density,
) -> tuple[np.ndarray, dict[str, object]]:
    """Integrate each support until its three scaled moments converge."""
    vertices = np.asarray(support.support_vertices)
    counts = np.asarray(support.vertex_count)
    values = np.zeros((3, len(counts)))
    selected_order = np.zeros(len(counts), dtype=np.int64)
    final_delta = np.zeros(len(counts))
    unconverged = []
    sampling_centre = np.asarray(stencil.ring_sampling_centre)
    coordinate_scale = np.asarray(stencil.ring_coordinate_scale)
    for cell in np.flatnonzero(selection & (counts >= 3)):
        previous = None
        converged = False
        for order in TRUTH_ORDERS:
            current = _integrate_cell(
                vertices[cell, : counts[cell]],
                centres[cell],
                sampling_centre[cell],
                coordinate_scale[cell],
                coefficients[cell],
                density,
                order,
            )
            if previous is not None:
                scaled_current = current.copy()
                scaled_previous = previous.copy()
                scaled_current[1:] /= cell_width[cell]
                scaled_previous[1:] /= cell_width[cell]
                delta = float(np.max(np.abs(scaled_current - scaled_previous)))
                scale = max(float(np.max(np.abs(scaled_current))), 1.0)
                final_delta[cell] = delta
                if delta <= (
                    TRUTH_ABSOLUTE_TOLERANCE_A + TRUTH_RELATIVE_TOLERANCE * scale
                ):
                    converged = True
                    values[:, cell] = current
                    selected_order[cell] = order
                    break
            previous = current
        if not converged:
            values[:, cell] = current
            selected_order[cell] = TRUTH_ORDERS[-1]
            unconverged.append(int(cell))
    active = selection & (counts >= 3)
    return values, {
        "method": "per-cell tensor-Duffy order refinement",
        "orders": list(TRUTH_ORDERS),
        "relative_tolerance": TRUTH_RELATIVE_TOLERANCE,
        "absolute_scaled_moment_tolerance_a": TRUTH_ABSOLUTE_TOLERANCE_A,
        "active_cells": int(np.count_nonzero(active)),
        "converged_cells": int(np.count_nonzero(active) - len(unconverged)),
        "unconverged_cells": unconverged,
        "maximum_selected_order": int(np.max(selected_order)),
        "selected_order_counts": {
            str(order): int(np.count_nonzero(selected_order == order))
            for order in TRUTH_ORDERS
        },
        "maximum_final_scaled_delta_a": float(np.max(final_delta[active])),
    }


def production_fit(profile, operator, centroid_flux, sample_flux, support, selection):
    moments = operator.support_current_moments(
        profile, centroid_flux, sample_flux, support
    )
    return jnp.stack(
        [jnp.where(jnp.asarray(selection), value, 0.0) for value in moments]
    )


def faithful_quadrature(
    profile, operator, centroid_flux, sample_flux, support, selection
):
    """Integrate the quadratic flux image directly with degree-fifteen Duffy."""
    selected = jnp.asarray(selection)
    points, weights = clipped_support_quadrature(support, selected)
    psi_norm, _radial, _vertical = operator.sample_flux_field(
        centroid_flux, sample_flux, points
    )
    density = profile.current_density(points[..., 0], psi_norm)
    weighted = density * weights
    centre = jnp.asarray(operator.moment_geometry.atomic_mesh.centroids)
    offset = points - centre[:, None, :]
    values = jnp.stack(
        [
            jnp.sum(weighted, axis=1),
            jnp.sum(weighted * offset[..., 0], axis=1),
            jnp.sum(weighted * offset[..., 1], axis=1),
        ]
    )
    return jnp.where(selected[None, :], values, 0.0)


def _time_route(function, arguments) -> tuple[np.ndarray, dict[str, object]]:
    compiled = jax.jit(function)
    started = perf_counter()
    result = compiled(*arguments)
    jax.block_until_ready(result)
    compile_seconds = perf_counter() - started
    samples = []
    for _ in range(TIMING_REPETITIONS):
        started = perf_counter()
        result = compiled(*arguments)
        jax.block_until_ready(result)
        samples.append(perf_counter() - started)
    return np.asarray(result), {
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "compile_and_first_evaluation_seconds": compile_seconds,
        "repetitions": TIMING_REPETITIONS,
        "steady_median_seconds": statistics.median(samples),
        "steady_minimum_seconds": min(samples),
        "steady_maximum_seconds": max(samples),
        "host_transfer_included": False,
        "scope": "support moments only; separatrix tracing excluded equally",
    }


def _populations(machine, support, selection: np.ndarray) -> dict[str, np.ndarray]:
    fraction = np.divide(
        np.asarray(support.area),
        np.asarray(machine.area),
        out=np.zeros_like(np.asarray(machine.area)),
        where=np.asarray(machine.area) > 0.0,
    )
    tolerance = 2.0e-12
    boundary = selection & (fraction > tolerance) & (fraction < 1.0 - tolerance)
    interior = selection & (fraction >= 1.0 - tolerance)
    ring = boundary.copy()
    for neighbours in np.asarray(machine.stencil):
        if np.any(boundary[neighbours]):
            ring[neighbours] = True
    ring &= selection
    return {
        "interior": interior,
        "boundary_clipped": boundary,
        "ring": ring,
    }


def _route_errors(
    values: np.ndarray,
    truth: np.ndarray,
    population: np.ndarray,
    cell_width: np.ndarray,
) -> dict[str, object]:
    error = np.abs(values - truth)
    scale = np.vstack(
        [
            np.abs(truth[0]),
            np.abs(truth[0]) * cell_width,
            np.abs(truth[0]) * cell_width,
        ]
    )
    floor = max(float(np.sum(np.abs(truth[0]))) * 1.0e-15, 1.0e-14)
    relative = error / np.maximum(scale, floor)
    names = ("m0", "mR", "mZ")
    result = {"cell_count": int(np.count_nonzero(population))}
    for index, name in enumerate(names):
        selected_error = error[index, population]
        selected_relative = relative[index, population]
        denominator = float(np.sum(scale[index, population]))
        result[name] = {
            "absolute_l1": float(np.sum(selected_error)),
            "current_weighted_l1": float(
                np.sum(selected_error) / max(denominator, floor)
            ),
            "per_cell_relative_median": float(np.median(selected_relative)),
            "per_cell_relative_p95": float(np.quantile(selected_relative, 0.95)),
            "per_cell_relative_maximum": float(np.max(selected_relative)),
        }
    return result


def _render(report: dict[str, object]) -> list[str]:
    populations = ("interior", "boundary_clipped", "ring")
    labels = ("interior", "boundary", "ring")
    figures = []
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.6), constrained_layout=True)
    for row, functional in enumerate(("smooth", "steep_pedestal")):
        for column, population in enumerate(populations):
            axis = axes[row, column]
            values = [
                report["functionals"][functional]["errors"][route][population]["m0"][
                    "current_weighted_l1"
                ]
                for route in ("degree_nine_fit", "faithful_duffy")
            ]
            axis.bar(["fit", "faithful"], values, color=["#4c78a8", "#f58518"])
            axis.set_yscale("log")
            axis.set_title(f"{functional.replace('_', ' ')}: {labels[column]}")
            axis.set_ylabel("current-weighted m0 L1")
            axis.grid(axis="y", alpha=0.25)
    path = OUTPUT / "moment-fidelity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(path.name)

    timing = report["cost"]
    fig, axis = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    values = [
        1.0e3 * timing[route]["steady_median_seconds"]
        for route in ("degree_nine_fit", "faithful_duffy")
    ]
    axis.bar(["fit", "faithful"], values, color=["#4c78a8", "#f58518"])
    axis.set_ylabel("CPU moment time per iteration [ms]")
    axis.set_title("Warm coarse carrier, support tracing excluded equally")
    axis.grid(axis="y", alpha=0.25)
    path = OUTPUT / "moment-cost.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(path.name)
    return figures


def measure() -> dict[str, object]:
    configure_dtypes()
    case = fixture.analytic_case()
    machine = fixture.cached_machine(case, -500, wall_nodes=fixture.WALL_POINT_COUNT)
    operator = fixture.forward_operator(case, machine)
    coordinates = np.r_[machine.node, machine.wall_node, machine.sample_coordinates]
    state = fixture.exact_state(case, coordinates)
    masks, topology, sample_flux, support, _common = operator._support_partition(
        jnp.asarray(state)
    )
    centroid_flux = np.asarray(masks.psi_norm)
    sample_flux = np.asarray(sample_flux)
    selection = np.asarray(masks.core | masks.common_sol) & np.asarray(support.included)
    stencil = operator._support_moment_stencils[0]
    coefficients = _flux_coefficients(stencil, centroid_flux, sample_flux)
    centres = np.asarray(operator.moment_geometry.atomic_mesh.centroids)
    cell_width = np.sqrt(np.asarray(machine.area))
    populations = _populations(machine, support, selection)

    smooth_profile = fixture.analytic_profile(case)
    smooth_axis_density = float(
        smooth_profile.current_density(jnp.asarray(case.major_radius), jnp.asarray(0.0))
    )
    steep_profile = PedestalProfile(abs(smooth_axis_density))
    functionals = {
        "smooth": (
            smooth_profile,
            lambda radius, flux: _smooth_density(case, radius, flux),
        ),
        "steep_pedestal": (
            steep_profile,
            lambda radius, flux: _pedestal_density(
                abs(smooth_axis_density), radius, flux
            ),
        ),
    }
    functional_results = {}
    timed_values = {}
    timed_receipts = {}
    for name, (profile, host_density) in functionals.items():
        truth, truth_receipt = adaptive_truth(
            support,
            selection,
            stencil,
            coefficients,
            centres,
            cell_width,
            host_density,
        )

        def fit_function(centre, sample):
            return production_fit(profile, operator, centre, sample, support, selection)

        def faithful_function(centre, sample):
            return faithful_quadrature(
                profile, operator, centre, sample, support, selection
            )

        fit, fit_time = _time_route(
            fit_function, (jnp.asarray(centroid_flux), jnp.asarray(sample_flux))
        )
        faithful, faithful_time = _time_route(
            faithful_function, (jnp.asarray(centroid_flux), jnp.asarray(sample_flux))
        )
        timed_values[name] = {"degree_nine_fit": fit, "faithful_duffy": faithful}
        timed_receipts[name] = {
            "degree_nine_fit": fit_time,
            "faithful_duffy": faithful_time,
        }
        errors = {}
        for route, values in timed_values[name].items():
            errors[route] = {
                population: _route_errors(values, truth, mask, cell_width)
                for population, mask in populations.items()
            }
        functional_results[name] = {
            "definition": (
                "moderate-rotation closed-form source profile"
                if name == "smooth"
                else "finite-edge tanh pedestal in psi_norm"
            ),
            "truth": truth_receipt,
            "errors": errors,
        }

    cost = {}
    for route in ("degree_nine_fit", "faithful_duffy"):
        samples = [timed_receipts[name][route] for name in functionals]
        cost[route] = dict(samples[0])
        cost[route]["steady_median_seconds"] = statistics.median(
            item["steady_median_seconds"] for item in samples
        )
        cost[route]["functional_receipts"] = {
            name: timed_receipts[name][route] for name in functionals
        }
    fidelity_fit = functional_results["steep_pedestal"]["errors"]["degree_nine_fit"][
        "ring"
    ]["m0"]["current_weighted_l1"]
    fidelity_faithful = functional_results["steep_pedestal"]["errors"][
        "faithful_duffy"
    ]["ring"]["m0"]["current_weighted_l1"]
    fidelity_improvement = fidelity_fit / max(fidelity_faithful, 1.0e-300)
    cost_ratio = (
        cost["faithful_duffy"]["steady_median_seconds"]
        / cost["degree_nine_fit"]["steady_median_seconds"]
    )
    qualifies = fidelity_improvement >= 3.0 and cost_ratio <= 2.0
    decision = {
        "rule": (
            "faithful qualifies iff steep ring current-weighted m0 fidelity "
            "improves by at least 3x and CPU per-iteration moment cost is no "
            "more than 2x the fit"
        ),
        "fidelity_improvement_factor": fidelity_improvement,
        "cost_ratio": cost_ratio,
        "fidelity_threshold": 3.0,
        "cost_ratio_ceiling": 2.0,
        "faithful_qualifies": qualifies,
        "branch_disposition": (
            "merge held commits after a main merge-in and pin rebase"
            if qualifies
            else "discard held worktree and retain the degree-nine fit route"
        ),
        "mechanical_application": True,
    }
    report = {
        "schema": "moment-route-adjudication",
        "oracle": {
            "case": case.name,
            "fixture": "warm coarse closed-form carrier",
            "cell_count": len(machine.node),
            "active_support_count": int(np.count_nonzero(selection)),
            "cache": machine.cache,
            "boundary_flux_wb": float(topology.boundary_flux),
            "flux_interpolant": "production own-node quadratic",
            "support_geometry": "production traced atomic-edge clip",
            "held_worktree_dependency": False,
        },
        "functional_parameters": {
            "smooth": "closed-form moderate-rotation-conventional profile",
            "steep_pedestal": {
                "amplitude_a_per_m2": abs(smooth_axis_density),
                "edge_fraction": STEEP_EDGE_FRACTION,
                "centre_psi_norm": STEEP_PEDESTAL_CENTRE,
                "width_psi_norm": STEEP_PEDESTAL_WIDTH,
                "family": "finite-edge pedestal precedent with a narrow tanh shoulder",
            },
        },
        "population_definitions": {
            "interior": "active support with area equal to its authored full cell",
            "boundary_clipped": (
                "active support with fill fraction strictly between zero and one"
            ),
            "ring": "boundary-clipped cells plus their centre-first stencil neighbours",
            "counts": {
                name: int(np.count_nonzero(mask)) for name, mask in populations.items()
            },
        },
        "routes": {
            "degree_nine_fit": (
                "production weighted-QR pointwise density fit followed by exact "
                "polynomial support moments"
            ),
            "faithful_duffy": (
                "degree-fifteen fixed Duffy quadrature of j(psi_interp) on the "
                "same traced supports"
            ),
            "adaptive_truth": (
                "per-cell order-refined tensor-Duffy integral of the identical "
                "j(psi_interp) functional"
            ),
        },
        "functionals": functional_results,
        "cost": cost,
        "gpu_estimate": {
            "status": "estimate from banked H200 phase profile; no device run here",
            "banked_coarse_map_seconds": GPU_MAP_SECONDS,
            "method": "scale the banked map phase by each CPU moment-route share",
            "fit_seconds": GPU_MAP_SECONDS
            * cost["degree_nine_fit"]["steady_median_seconds"]
            / sum(item["steady_median_seconds"] for item in cost.values()),
            "faithful_seconds": GPU_MAP_SECONDS
            * cost["faithful_duffy"]["steady_median_seconds"]
            / sum(item["steady_median_seconds"] for item in cost.values()),
            "qualification": (
                "throughput projection only; the banked phase did not isolate "
                "moment integration from the complete map"
            ),
        },
        "decision": decision,
        "bounds_moved_or_applied": False,
    }
    report["artifacts"] = _render(report)
    return report


def main() -> None:
    report = measure()
    (OUTPUT / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["decision"], indent=2, sort_keys=True))
    print(json.dumps(report["population_definitions"]["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
