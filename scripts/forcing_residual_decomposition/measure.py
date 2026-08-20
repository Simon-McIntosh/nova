"""Decompose exact-state map forcing by density, geometry, and topology."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import triangulate

from nova.biot.greens import section_centroid
from nova.equilibrium.stencil_mesh import CellCurrentMoments


OUTPUT = Path(__file__).resolve().parent
REFERENCE_PATH = Path("tests/test_equilibrium_forward_reference.py")
TERMINAL_PATH = Path("scripts/root_gate_attribution/coarse-terminal-root.npz")
COVERAGE_RESULTS = Path("scripts/coverage_scaling/results.json")
FORCING_RESULTS = Path("scripts/map_forcing_attribution/results.json")
ARC_POINTS = 8192
EXPECTED_FORCING_FRACTION = 0.015976762660416564
EXPECTED_RESIDUAL_PROJECTION = 1.000233035506072


def load_module(path: Path, name: str):
    """Load a repository measurement module without collecting its tests."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Read a compressed bank into detached arrays."""
    with np.load(path) as stored:
        return {name: stored[name] for name in stored.files}


def polygon_parts(geometry) -> list[Polygon]:
    """Return every positive-area polygon carried by a Shapely geometry."""
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        return [
            polygon
            for item in geometry.geoms
            for polygon in polygon_parts(item)
            if polygon.area > 0.0
        ]
    return []


def integrate_vertices(case, vertices, centre, density_fn, depth):
    """Integrate current and first moments on one convex polygon."""
    origin = section_centroid(vertices)
    triangles = np.stack(
        [
            np.broadcast_to(origin, vertices.shape),
            vertices,
            np.roll(vertices, -1, axis=0),
        ],
        axis=1,
    )
    for _ in range(depth):
        triangles = density_fn.subdivide(triangles)
    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 0]
    area = 0.5 * np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    barycentric = np.asarray(
        [
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
        ]
    )
    weight = np.asarray([-27.0, 25.0, 25.0, 25.0]) / 48.0
    points = np.einsum("qa,tad->tqd", barycentric, triangles)
    density = density_fn.source_current_density(case, points)
    weighted = area[:, None] * weight[None, :] * density
    current = float(np.sum(weighted))
    moment = np.sum(weighted[..., None] * (points - centre), axis=(0, 1))
    return current, moment


def integrate_geometry(case, geometry, centre, density_fn, depth):
    """Integrate possibly non-convex clipped geometry without double counting."""
    current = 0.0
    moment = np.zeros(2)
    for polygon in polygon_parts(geometry):
        tolerance = 1.0e-12 * max(polygon.area, 1.0)
        pieces = [polygon]
        if (
            polygon.interiors
            or abs(polygon.convex_hull.area - polygon.area) > tolerance
        ):
            pieces = [
                piece
                for candidate in triangulate(polygon)
                for piece in polygon_parts(candidate.intersection(polygon))
                if piece.area > tolerance
            ]
            covered = math.fsum(piece.area for piece in pieces)
            if abs(covered - polygon.area) > tolerance:
                raise AssertionError("triangulation does not cover clipped geometry")
        for piece in pieces:
            vertices = np.asarray(piece.exterior.coords, dtype=float)[:-1, :2]
            value, first = integrate_vertices(case, vertices, centre, density_fn, depth)
            current += value
            moment += first
    return current, moment


def curved_moments(case, machine, core, production_selected, density_fn):
    """Build adaptive analytic-density moments on curved per-cell intersections."""
    centres = np.asarray(machine.moment_geometry.atomic_mesh.centroids)
    geometries = [
        Polygon(vertices).intersection(core) for vertices in machine.cell_polygons
    ]

    def evaluate(depth):
        current = np.zeros(len(geometries))
        first = np.zeros((len(geometries), 2))
        for cell in np.flatnonzero(production_selected):
            current[cell], first[cell] = integrate_geometry(
                case, geometries[cell], centres[cell], density_fn, depth
            )
        return current, first

    coarse_current, coarse_first = evaluate(3)
    current, first = evaluate(4)
    current_delta = float(np.sum(np.abs(current - coarse_current)))
    first_delta = float(np.sum(np.linalg.norm(first - coarse_first, axis=1)))
    nonempty = np.asarray(
        [not geometry.is_empty and geometry.area > 0.0 for geometry in geometries]
    )
    return (
        current,
        first,
        nonempty,
        {
            "coarse_depth": 3,
            "accepted_depth": 4,
            "current_l1_refinement_a": current_delta,
            "first_moment_l1_refinement_a_m": first_delta,
            "intersected_cells": int(np.count_nonzero(nonempty)),
        },
    )


def internal_flux(operator, physical_current, physical_first):
    """Contract physical zeroth and first moments through production blocks."""
    physical = CellCurrentMoments(
        jnp.asarray(physical_current),
        jnp.asarray(physical_first[:, 0]),
        jnp.asarray(physical_first[:, 1]),
    )
    coupled = operator.coupling_current_moments(physical)
    physical_flux = jnp.r_[
        operator.grid.internal(coupled), operator.wall.internal(coupled)
    ]
    return np.asarray(jnp.r_[physical_flux, operator.sample.internal(coupled)])


def score(component, total):
    """Measure one signed component against the comparator residual."""
    denominator = float(np.dot(total, total))
    peak = int(np.argmax(np.abs(total)))
    return {
        "projection_fraction": float(np.dot(component, total) / denominator),
        "signed_peak_fraction": float(component[peak] / total[peak]),
        "sup_norm_fraction": float(np.max(np.abs(component)) / np.max(np.abs(total))),
        "rms_norm_fraction": float(
            np.sqrt(np.mean(component**2)) / np.sqrt(np.mean(total**2))
        ),
        "sup_wb": float(np.max(np.abs(component))),
        "rms_wb": float(np.sqrt(np.mean(component**2))),
    }


def response_observables(case, machine, operator, exact, terminal, step):
    """Project one response through linearised root observables."""
    terminal_axis = np.asarray(operator.read(jnp.asarray(terminal))[1].axis)
    axis_direction = terminal_axis - np.asarray(case.axis)
    axis_direction /= np.linalg.norm(axis_direction)

    def axis_fn(state):
        return operator.read(state)[1].axis

    def current_fn(state):
        return jnp.sum(operator.cell_current_moments(state).cell_current)

    _axis, axis_delta = jax.jvp(axis_fn, (exact,), (jnp.asarray(step),))
    _current, current_delta = jax.jvp(current_fn, (exact,), (jnp.asarray(step),))
    observed_grid_delta = np.asarray(terminal - exact)[: len(machine.node)]
    flux_slot = int(np.argmax(np.abs(observed_grid_delta)))
    return {
        "axis_signed_projection_mm": float(
            1.0e3 * np.dot(np.asarray(axis_delta), axis_direction)
        ),
        "flux_signed_peak_percent_of_span": float(
            100.0 * step[flux_slot] / abs(case.flux_span)
        ),
        "plasma_current_signed_percent": float(
            100.0 * float(current_delta) / case.plasma_current
        ),
        "observed_flux_peak_grid_slot": flux_slot,
    }


def render(report):
    """Render forcing and drift shares for the three named swaps."""
    names = ["source density", "support geometry", "topology qualification"]
    keys = ["source_density", "support_geometry", "topology_qualification"]
    forcing = [
        report["components"][key]["forcing"]["projection_fraction"] for key in keys
    ]
    response = [report["components"][key]["root_response"] for key in keys]
    metrics = (
        "axis_signed_projection_mm",
        "flux_signed_peak_percent_of_span",
        "plasma_current_signed_percent",
    )
    labels = ["axis [mm]", "flux [% span]", "current [%]"]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    axes[0].bar(names, forcing, color=["#4c78a8", "#f2cf5b", "#e45756"])
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_ylabel("signed projection on comparator residual")
    axes[0].set_title("Exact-state forcing decomposition")
    axes[0].tick_params(axis="x", rotation=18)
    x = np.arange(3)
    width = 0.24
    for index, (name, values) in enumerate(zip(names, response, strict=True)):
        axes[1].bar(
            x + (index - 1) * width,
            [values[metric] for metric in metrics],
            width,
            label=name,
        )
    axes[1].set_xticks(x, labels)
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_title("Linear-inverse root-drift shares")
    axes[1].legend(fontsize=8)
    figure.savefig(OUTPUT / "decomposition.png", dpi=180)
    plt.close(figure)


def main():
    """Run the banked coarse decomposition and write durable evidence."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference = load_module(REFERENCE_PATH, "residual_decomposition_reference")
    reference.configure_dtypes()
    case = reference.require_reference()
    forcing_module = load_module(
        Path("scripts/map_forcing_attribution/measure.py"), "forcing_measurement"
    )
    gate_module = load_module(
        Path("scripts/root_gate_attribution/measure_root_attribution.py"),
        "root_gate_measurement",
    )
    coverage_module = load_module(
        Path("scripts/coverage_scaling/measure_coverage.py"), "coverage_measurement"
    )
    coverage_bank = json.loads(COVERAGE_RESULTS.read_text(encoding="utf-8"))
    forcing_bank = json.loads(FORCING_RESULTS.read_text(encoding="utf-8"))
    terminal = load_npz(TERMINAL_PATH)["state"]

    reference.WALL_NODES = 3
    print("BUILD coarse analytic fixture", flush=True)
    machine = reference.build_machine(case, reference.SUITE_CELLS, passive=True)
    operator = reference.forward_operator(case, machine)
    exact = reference.seed_flux(case, machine)
    if len(terminal) != len(exact):
        raise AssertionError(
            "banked terminal root and reconstructed map differ in shape"
        )
    map_fn = operator.flux_map()
    mapped, tangent = jax.linearize(map_fn, exact)
    forcing = np.asarray(mapped - exact)
    forcing_fraction = float(
        np.max(np.abs(forcing[: len(machine.node)])) / abs(case.flux_span)
    )
    if abs(forcing_fraction - EXPECTED_FORCING_FRACTION) > 1.0e-14:
        raise AssertionError("coarse forcing control did not reproduce")

    coupled = operator.cell_current_moments(exact)
    second_residual, _actual, _reconstructed = forcing_module.second_moment_residual(
        case, operator, exact, coupled
    )
    correction, active_cells, pair_count = forcing_module.second_order_correction(
        machine, second_residual, forcing_module.HESSIAN_STEPS[-1]
    )
    comparator = forcing + correction
    residual_projection = float(np.dot(comparator, forcing) / np.dot(forcing, forcing))
    if abs(residual_projection - EXPECTED_RESIDUAL_PROJECTION) > 1.0e-12:
        raise AssertionError("second-order comparator control did not reproduce")

    masks, topology = operator.read(exact)
    support, straight_current, straight_first, _coupled = gate_module.support_reference(
        case, operator, exact, masks, topology
    )
    production_selected = np.asarray(masks.core | masks.common_sol)
    straight_image = np.asarray(operator.external()) + internal_flux(
        operator, straight_current, straight_first
    )

    saddle, saddle_receipt = coverage_module.solve_flux_saddle(case)
    core, core_receipt = coverage_module.analytic_core(case, ARC_POINTS, saddle)
    curved_current, curved_first, curved_nonempty, curved_refinement = curved_moments(
        case, machine, core, production_selected, coverage_module
    )
    curved_image = np.asarray(operator.external()) + internal_flux(
        operator, curved_current, curved_first
    )

    lower_leg = curved_nonempty & (np.asarray(machine.node)[:, 1] < float(saddle[1]))
    analytic_selected = curved_nonempty & ~lower_leg
    topology_current, topology_first, _unused, topology_refinement = curved_moments(
        case, machine, core, analytic_selected, coverage_module
    )
    topology_image = np.asarray(operator.external()) + internal_flux(
        operator, topology_current, topology_first
    )

    components = {
        "source_density": np.asarray(mapped) + correction - straight_image,
        "support_geometry": straight_image - curved_image,
        "topology_qualification": curved_image - np.asarray(exact),
    }
    topology_subcomponents = {
        "participation_swap": curved_image - topology_image,
        "qualified_reference_remainder": topology_image - np.asarray(exact),
    }
    closure = sum(components.values()) - comparator

    response = {}
    steps = {}
    for name, component in components.items():
        step, solve = forcing_module.solve_response(tangent, component)
        steps[name] = step
        response[name] = {
            "solve": solve,
            **response_observables(case, machine, operator, exact, terminal, step),
        }
    total_step = sum(steps.values())
    total_response = response_observables(
        case, machine, operator, exact, terminal, total_step
    )
    observed = forcing_module.state_deviation(case, machine, operator, exact, terminal)
    component_report = {
        name: {"forcing": score(component, comparator), "root_response": response[name]}
        for name, component in components.items()
    }
    topology_report = {
        name: score(component, comparator)
        for name, component in topology_subcomponents.items()
    }
    dominant = max(
        components,
        key=lambda name: abs(component_report[name]["forcing"]["projection_fraction"]),
    )

    refinement_bank = coverage_bank["analytic_core_oracle"]["refinement_pairs"][-1]
    report = {
        "schema": "nova.forcing-residual-decomposition",
        "controls": {
            "plasma_cells": len(machine.node),
            "state_size": len(exact),
            "forcing_grid_sup_fraction_of_span": forcing_fraction,
            "expected_forcing_grid_sup_fraction_of_span": EXPECTED_FORCING_FRACTION,
            "comparator_residual_projection_on_total_forcing": residual_projection,
            "expected_comparator_residual_projection": EXPECTED_RESIDUAL_PROJECTION,
            "banked_observed_root_drift": {
                "axis_displacement_mm": observed["axis_displacement_mm"],
                "flux_sup_percent_of_span": 100.0
                * observed["flux_sup_fraction_of_span"],
                "plasma_current_percent": 100.0
                * observed["plasma_current_fractional_deviation"],
            },
        },
        "method": {
            "swap_path": [
                "second-order comparator image",
                (
                    "exact analytic density on the same straight supports and "
                    "production participation"
                ),
                (
                    "exact analytic density on saddle-aware curved per-cell "
                    "intersections and production participation"
                ),
                "topology-qualified analytic reference state",
            ],
            "sign_convention": (
                "Each component is the earlier image minus the later image; the "
                "three named components sum to the comparator residual."
            ),
            "topology_arm_qualification": (
                "The topology arm is separately split into the measured lower-leg "
                "participation swap and the remaining qualified-reference residual."
            ),
            "second_moment_comparator": {
                "active_cells": active_cells,
                "near_source_target_pairs": pair_count,
                "correction_sup_wb": float(np.max(np.abs(correction))),
            },
        },
        "oracle": {
            "arc_points": ARC_POINTS,
            "saddle_coordinates_m": np.asarray(saddle).tolist(),
            "saddle": saddle_receipt,
            "core": core_receipt,
            "banked_arc_refinement_current_delta_a": refinement_bank["current_delta_a"],
            "banked_arc_refinement_fraction_of_representation_term": refinement_bank[
                "delta_fraction_of_representation_term"
            ],
            "curved_production_participation_refinement": curved_refinement,
            "curved_analytic_participation_refinement": topology_refinement,
            "straight_support_cells": int(
                np.count_nonzero(np.asarray(support.vertex_count) >= 3)
            ),
            "production_selected_cells": int(np.count_nonzero(production_selected)),
            "analytic_selected_cells": int(np.count_nonzero(analytic_selected)),
            "curved_topology_excluded_lower_leg_cells": int(
                np.count_nonzero(lower_leg)
            ),
            "banked_straight_topology_excluded_lower_leg_cells": coverage_bank[
                "topology_decomposition"
            ]["topology_zero_lower_leg_cells"],
        },
        "components": component_report,
        "topology_arm_subcomponents": topology_report,
        "additive_closure": {
            "sup_wb": float(np.max(np.abs(closure))),
            "relative_sup": float(np.max(np.abs(closure)) / np.max(np.abs(comparator))),
            "projection_fraction_sum": float(
                sum(
                    value["forcing"]["projection_fraction"]
                    for value in component_report.values()
                )
            ),
        },
        "linear_response": {
            "equation": "(I - Dg) delta = forcing component",
            "component_sum": total_response,
            "component_response_closure_sup_wb": float(
                np.max(np.abs(total_step - sum(steps.values())))
            ),
            "observable_signs": (
                "Axis is projected along the observed displacement; flux uses the "
                "observed root's peak grid slot; current is a JVP at the exact state."
            ),
        },
        "verdict": {
            "dominant_component": dominant,
            "dominant_projection_fraction": component_report[dominant]["forcing"][
                "projection_fraction"
            ],
            "measurement_only": True,
        },
        "fine_fixture": {
            "status": "not run",
            "reason": (
                "The fine 1069-cell coupling build is not banked as reusable map "
                "blocks; repeat the dominant coarse swap when that build is available."
            ),
        },
        "source_artifacts": {
            "prior_forcing": str(FORCING_RESULTS),
            "coverage_oracle": str(COVERAGE_RESULTS),
            "terminal_root": str(TERMINAL_PATH),
            "prior_forcing_control": forcing_bank["coarse_fixture"]["forcing"],
        },
    }
    render(report)
    report["artifacts"] = {
        "figure": str(OUTPUT / "decomposition.png"),
        "figure_bytes": (OUTPUT / "decomposition.png").stat().st_size,
    }
    (OUTPUT / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"RESULT dominant={dominant} "
        f"projection={report['verdict']['dominant_projection_fraction']:.17g} "
        f"closure={report['additive_closure']['relative_sup']:.17g}",
        flush=True,
    )
    print("FORCING_RESIDUAL_DECOMPOSITION_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
