"""Re-score serialized equilibrium roots with clipped-support observations."""

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

from nova.equilibrium.observation import (
    clipped_support_quadrature,
    observe_moments,
)


OUTPUT = Path(__file__).resolve().parent
REFERENCE_PATH = Path("tests/test_equilibrium_forward_reference.py")
BASELINE_SCORE_PATH = Path("scripts/analytic_seed_adjudication/results.json")
BASELINE_SPLIT_PATH = Path("scripts/root_gate_attribution/results.json")
ROOT_DIRECTORY = Path("scripts/root_gate_attribution")
FIGURE_PATH = OUTPUT / "observation-rescore.png"
RESULT_PATH = OUTPUT / "results.json"
RESIDUAL_TOLERANCE = 1.0e-14
CHORD_CRESCENT_CONTEXT_FRACTION = 0.002
FIXTURES = (
    ("coarse", 1, 566, 1.625101639852996e-15),
    ("fine", 2, 1069, 1.0264068138122435e-15),
)


def load_module(path: Path, name: str):
    """Load one fixture module without collecting its tests."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(report: dict[str, object]) -> None:
    """Checkpoint the scorecard after each fixture."""
    RESULT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def relative_residual(map_fn, state) -> float:
    """Return map displacement relative to the map image scale."""
    mapped = map_fn(state)
    jax.block_until_ready(mapped)
    return float(
        jnp.max(jnp.abs(mapped - state))
        / jnp.maximum(jnp.max(jnp.abs(mapped)), 1.0e-30)
    )


def triangle_fan(vertices: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Triangulate an ordered polygon about an interior reference point."""
    return np.stack(
        [
            np.broadcast_to(centre, vertices.shape),
            vertices,
            np.roll(vertices, -1, axis=0),
        ],
        axis=1,
    )


def subdivide(triangles: np.ndarray) -> np.ndarray:
    """Split every triangle into four congruent children."""
    first, second, third = np.moveaxis(triangles, 1, 0)
    first_second = 0.5 * (first + second)
    second_third = 0.5 * (second + third)
    third_first = 0.5 * (third + first)
    return np.concatenate(
        [
            np.stack([first, first_second, third_first], axis=1),
            np.stack([first_second, second, second_third], axis=1),
            np.stack([third_first, second_third, third], axis=1),
            np.stack([first_second, second_third, third_first], axis=1),
        ]
    )


def current_density(case, coordinates: np.ndarray) -> np.ndarray:
    """Evaluate the stored analytic-profile density at poloidal points."""
    radius, height = coordinates[..., 0], coordinates[..., 1]
    flux = case.flux(radius, height)
    normalised = (flux - case.flux_axis) / (case.flux_boundary - case.flux_axis)
    pressure_gradient = np.interp(normalised, case.psi_norm, case.p_prime)
    diamagnetic_gradient = np.interp(normalised, case.psi_norm, case.ff_prime)
    return (
        -2.0
        * np.pi
        * (radius * pressure_gradient + diamagnetic_gradient / (mu_0 * radius))
    )


def polygon_current(case, vertices: np.ndarray, centre: np.ndarray) -> float:
    """Integrate analytic current density on one traced support."""
    triangles = triangle_fan(vertices, centre)
    for _ in range(4):
        triangles = subdivide(triangles)
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
    return float(
        np.sum(area[:, None] * weight[None, :] * current_density(case, points))
    )


def support_reference(case, operator, state, masks, topology) -> dict[str, float]:
    """Integrate the stored analytic map on the production traced supports."""
    shared_flux = operator.shared_node_flux(state)
    signed_flux = operator.polarity * (shared_flux - topology.boundary_flux)
    support = operator.moment_geometry.atomic_mesh.traced_clip(signed_flux)
    selected = np.asarray(masks.core | masks.common_sol)
    counts = np.asarray(support.vertex_count)
    vertices = np.asarray(support.support_vertices)
    centres = np.asarray(support.centroids)
    currents = np.zeros(len(counts))
    for cell in np.flatnonzero(selected & (counts >= 3)):
        currents[cell] = polygon_current(
            case,
            vertices[cell, : counts[cell]],
            centres[cell],
        )

    points, weights = clipped_support_quadrature(support, jnp.asarray(selected))
    points = np.asarray(points)
    weights = np.asarray(weights)
    radius = points[..., 0]
    height = points[..., 1]
    radial_gradient = case.spline.ev(radius, height, dx=1)
    vertical_gradient = case.spline.ev(radius, height, dy=1)
    field_squared = (radial_gradient**2 + vertical_gradient**2) / (
        2.0 * np.pi * radius
    ) ** 2
    field_integral = float(np.sum(field_squared * 2.0 * np.pi * radius * weights))

    area = np.where(selected, np.asarray(support.area), 0.0)
    radial_first = np.where(selected, np.asarray(support.first_area_moment)[:, 0], 0.0)
    radial_second = np.where(
        selected, np.asarray(support.second_area_moment)[:, 0, 0], 0.0
    )
    centre_radius = centres[:, 0]
    volume = 2.0 * np.pi * np.sum(centre_radius * area + radial_first)
    radial_volume = (
        2.0
        * np.pi
        * np.sum(
            centre_radius**2 * area + 2.0 * centre_radius * radial_first + radial_second
        )
    )
    major_radius = radial_volume / volume
    plasma_current = float(np.sum(currents))
    internal_inductance = (
        2.0 * field_integral / (mu_0**2 * major_radius * plasma_current**2)
    )
    return {
        "volume_m3": volume,
        "major_radius_m": major_radius,
        "plasma_current_a": plasma_current,
        "field_integral_t2_m3": field_integral,
        "internal_inductance_raw": internal_inductance,
        "internal_inductance_referred": internal_inductance
        * major_radius
        / case.reference_radius,
        "selected_cells": int(np.count_nonzero(selected & (counts >= 3))),
    }


def signed_decomposition(
    attributed: float, support: float, analytic: float
) -> dict[str, float]:
    """Split one deviation on a common analytic denominator."""
    representation = (attributed - support) / analytic
    coverage = (support - analytic) / analytic
    total = (attributed - analytic) / analytic
    return {
        "attributed_vs_support_fraction": attributed / support - 1.0,
        "support_vs_analytic_fraction": support / analytic - 1.0,
        "attributed_vs_analytic_fraction": total,
        "representation_component_on_analytic_denominator": representation,
        "support_component_on_analytic_denominator": coverage,
        "additive_closure_error": total - representation - coverage,
        "chord_crescent_context_fraction": CHORD_CRESCENT_CONTEXT_FRACTION,
        "support_to_context_scale_ratio": abs(coverage)
        / CHORD_CRESCENT_CONTEXT_FRACTION,
    }


def less_than_gate(
    baseline_value: float,
    repaired_value: float,
    baseline_bound: float,
    repaired_bound: float | None = None,
) -> dict[str, object]:
    """Return one immutable less-than comparison."""
    if repaired_bound is None:
        repaired_bound = baseline_bound
    return {
        "comparison": "less_than",
        "baseline": {
            "value": baseline_value,
            "bound": baseline_bound,
            "passed": baseline_value < baseline_bound,
            "margin": baseline_bound - baseline_value,
        },
        "repaired": {
            "value": repaired_value,
            "bound": repaired_bound,
            "passed": repaired_value < repaired_bound,
            "margin": repaired_bound - repaired_value,
        },
    }


def score_gates(reference, case, machine, operator, state, residual, baseline):
    """Score the unchanged physical contract at one loaded root."""
    current_moments, measure, masks, topology = (
        operator.current_moments_and_observation(state)
    )
    moments = observe_moments(measure, topology.flux_span)
    profile = reference.ForwardProfile(
        operator=operator,
        lattice=reference.receipt_mesh(machine),
        newton_steps=reference.NEWTON_STEPS,
    )
    receipt = profile.observe(state)
    analytic = case.map_moments()
    core = np.asarray(masks.core)
    scale = float(moments.major_radius) / case.reference_radius
    repaired = {
        "axis_max_deviation_m": float(
            np.max(np.abs(np.asarray(topology.axis) - case.axis))
        ),
        "flux_deviation_fraction": float(
            np.max(
                np.abs(
                    np.asarray(state)[: len(machine.node)]
                    - case.flux(machine.radius, machine.node[:, 1])
                )[core]
            )
            / abs(case.flux_span)
        ),
        "plasma_current_deviation_fraction": abs(
            float(moments.plasma_current) / case.plasma_current - 1.0
        ),
        "poloidal_beta_deviation_fraction": abs(
            float(moments.poloidal_beta) * scale / analytic["poloidal_beta"] - 1.0
        ),
        "internal_inductance_deviation_fraction": abs(
            float(moments.internal_inductance) * scale / analytic["internal_inductance"]
            - 1.0
        ),
        "analytic_field_quadrature_deviation_fraction": abs(
            float(moments.poloidal_field_integral) / analytic["field_integral"] - 1.0
        ),
        "receipt_field_deviation_fraction": abs(
            float(receipt.moments.poloidal_field_integral)
            / float(moments.poloidal_field_integral)
            - 1.0
        ),
        "grad_shafranov_residual": float(receipt.conservation.relative_grad_shafranov),
        "relative_divergence_b": float(receipt.conservation.relative_divergence_b),
        "relative_divergence_j": float(receipt.conservation.relative_divergence_j),
        "core_cells": int(np.count_nonzero(core)),
        "diverted": bool(topology.diverted),
        "plasma_current_a": float(moments.plasma_current),
        "poloidal_field_integral_t2_m3": float(moments.poloidal_field_integral),
        "internal_inductance_raw": float(moments.internal_inductance),
        "internal_inductance_referred": float(moments.internal_inductance) * scale,
        "major_radius_m": float(moments.major_radius),
        "volume_m3": float(moments.volume),
    }
    gates = {
        "axis": less_than_gate(
            baseline["axis_max_deviation_m"],
            repaired["axis_max_deviation_m"],
            reference.AXIS_TOLERANCE,
        ),
        "flux": less_than_gate(
            baseline["flux_deviation_fraction"],
            repaired["flux_deviation_fraction"],
            reference.FLUX_TOLERANCE,
        ),
        "plasma_current": less_than_gate(
            baseline["plasma_current_deviation_fraction"],
            repaired["plasma_current_deviation_fraction"],
            reference.PLASMA_CURRENT_TOLERANCE,
        ),
        "poloidal_beta": less_than_gate(
            baseline["poloidal_beta_deviation_fraction"],
            repaired["poloidal_beta_deviation_fraction"],
            reference.MOMENT_TOLERANCE,
        ),
        "internal_inductance": less_than_gate(
            baseline["internal_inductance_deviation_fraction"],
            repaired["internal_inductance_deviation_fraction"],
            reference.MOMENT_TOLERANCE,
        ),
        "analytic_field": less_than_gate(
            baseline["analytic_field_quadrature_deviation_fraction"],
            repaired["analytic_field_quadrature_deviation_fraction"],
            reference.QUADRATURE_TOLERANCE,
        ),
        "receipt_field": less_than_gate(
            baseline["receipt_field_deviation_fraction"],
            repaired["receipt_field_deviation_fraction"],
            reference.FIELD_INTEGRAL_TOLERANCE,
        ),
        "grad_shafranov": less_than_gate(
            baseline["grad_shafranov_residual"],
            repaired["grad_shafranov_residual"],
            reference.GRAD_SHAFRANOV_TOLERANCE,
        ),
        "divergence_b": less_than_gate(
            baseline["relative_divergence_b"],
            repaired["relative_divergence_b"],
            reference.DIVERGENCE_MARGIN * baseline["grad_shafranov_residual"],
            reference.DIVERGENCE_MARGIN * repaired["grad_shafranov_residual"],
        ),
        "divergence_j": less_than_gate(
            baseline["relative_divergence_j"],
            repaired["relative_divergence_j"],
            reference.DIVERGENCE_MARGIN * baseline["grad_shafranov_residual"],
            reference.DIVERGENCE_MARGIN * repaired["grad_shafranov_residual"],
        ),
        "physical": {
            "comparison": "nonempty_diverted_topology",
            "baseline": {
                "value": {
                    "core_cells": int(baseline["core_cells"]),
                    "diverted": bool(baseline["diverted"]),
                },
                "passed": int(baseline["core_cells"]) > 0
                and bool(baseline["diverted"]),
            },
            "repaired": {
                "value": {
                    "core_cells": repaired["core_cells"],
                    "diverted": repaired["diverted"],
                },
                "passed": repaired["core_cells"] > 0 and repaired["diverted"],
            },
        },
    }
    return repaired, gates, masks, topology


def score_decomposition(case, operator, state, repaired, masks, topology):
    """Build field and inductance splits on identical clipped domains."""
    support = support_reference(case, operator, state, masks, topology)
    analytic = case.map_moments()
    return {
        "definition": (
            "The support reference evaluates the stored analytic flux gradient "
            "and analytic profile current over the exact production traced "
            "straight-chord polygons. The representation component compares the "
            "repaired observation with that same-domain reference; the support "
            "component compares the same-domain reference with the full stored "
            "analytic-boundary raster."
        ),
        "support_reference": support,
        "analytic_reference": {
            "field_integral_t2_m3": float(analytic["field_integral"]),
            "internal_inductance": float(analytic["internal_inductance"]),
            "plasma_current_a": float(case.plasma_current),
            "reference_radius_m": float(case.reference_radius),
        },
        "analytic_field": signed_decomposition(
            repaired["poloidal_field_integral_t2_m3"],
            support["field_integral_t2_m3"],
            float(analytic["field_integral"]),
        ),
        "internal_inductance": signed_decomposition(
            repaired["internal_inductance_referred"],
            support["internal_inductance_referred"],
            float(analytic["internal_inductance"]),
        ),
    }


def measure_fixture(reference, case, baseline_score, name, multiplier, cells, banked):
    """Rebuild one operator, qualify its loaded root, and score observations."""
    reference.WALL_NODES = 3 * multiplier
    requested = reference.SUITE_CELLS * multiplier
    print(
        f"BUILD fixture={name} requested={requested} wall_nodes={reference.WALL_NODES}",
        flush=True,
    )
    machine = reference.build_machine(case, requested, passive=True)
    if len(machine.node) != cells:
        raise AssertionError(f"expected {cells} {name} cells, got {len(machine.node)}")
    operator = reference.forward_operator(case, machine)
    root_path = ROOT_DIRECTORY / f"{name}-terminal-root.npz"
    with np.load(root_path) as bank:
        state = jnp.asarray(bank["state"])
    if (
        state.size
        != operator.grid.node_number
        + operator.wall.node_number
        + operator.sample.node_number
    ):
        raise AssertionError(f"{name} root shape does not match the rebuilt operator")
    residual = relative_residual(operator.flux_map(), state)
    residual_difference = residual - banked
    residual_pass = abs(residual_difference) <= RESIDUAL_TOLERANCE
    print(
        f"RESIDUAL fixture={name} rebuilt={residual:.17g} banked={banked:.17g} "
        f"difference={residual_difference:.17g} pass={residual_pass}",
        flush=True,
    )
    if not residual_pass:
        raise AssertionError(
            f"{name} loaded root moved off its banked machine-precision residual"
        )

    baseline = baseline_score["fixtures"][name]["routes"]["undamped"]["physics_score"]
    repaired, gates, masks, topology = score_gates(
        reference, case, machine, operator, state, residual, baseline
    )
    decomposition = score_decomposition(
        case, operator, state, repaired, masks, topology
    )
    passed = sum(gate["repaired"]["passed"] for gate in gates.values())
    print(
        f"GATES fixture={name} passed={passed}/{len(gates)} "
        f"field={repaired['analytic_field_quadrature_deviation_fraction']:.12g} "
        f"li={repaired['internal_inductance_deviation_fraction']:.12g} "
        f"receipt={repaired['receipt_field_deviation_fraction']:.12g} "
        f"ip={repaired['plasma_current_deviation_fraction']:.12g}",
        flush=True,
    )
    return {
        "plasma_cells": cells,
        "operator_state_size": int(state.size),
        "root_artifact": str(root_path),
        "fixed_point_residual": {
            "banked": banked,
            "rebuilt": residual,
            "signed_difference": residual_difference,
            "absolute_tolerance": RESIDUAL_TOLERANCE,
            "passed": residual_pass,
        },
        "physics_gate_count": len(gates),
        "physics_gates_passed": passed,
        "all_physics_gates": passed == len(gates),
        "physics_gates": gates,
        "repaired_score": repaired,
        "decomposition": decomposition,
    }


def render(report: dict[str, object]) -> None:
    """Render before/after gates and the repaired two-reference splits."""
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), layout="constrained")
    metric_rows = (
        ("analytic_field", "analytic field"),
        ("internal_inductance", "internal inductance"),
        ("receipt_field", "receipt field"),
        ("plasma_current", "plasma current"),
    )
    for row, name in enumerate(("coarse", "fine")):
        gate_axis = axes[row, 0]
        split_axis = axes[row, 1]
        if name not in report["fixtures"]:
            for axis in (gate_axis, split_axis):
                axis.text(
                    0.5,
                    0.5,
                    "not scored — operator build stopped at time fence",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                axis.set_axis_off()
            continue
        fixture = report["fixtures"][name]
        x = np.arange(len(metric_rows))
        baseline_ratio = [
            fixture["physics_gates"][key]["baseline"]["value"]
            / fixture["physics_gates"][key]["baseline"]["bound"]
            for key, _label in metric_rows
        ]
        repaired_ratio = [
            fixture["physics_gates"][key]["repaired"]["value"]
            / fixture["physics_gates"][key]["repaired"]["bound"]
            for key, _label in metric_rows
        ]
        gate_axis.bar(x - 0.18, baseline_ratio, 0.36, color="0.65", label="banked")
        gate_axis.bar(
            x + 0.18, repaired_ratio, 0.36, color="seagreen", label="repaired"
        )
        gate_axis.axhline(1.0, color="black", linestyle=":", linewidth=1.0)
        gate_axis.set_xticks(x, [label for _key, label in metric_rows], rotation=18)
        gate_axis.set_ylabel("deviation / unchanged bound")
        gate_axis.set_title(f"{name}: observation gates")
        gate_axis.grid(axis="y", alpha=0.18)
        if row == 0:
            gate_axis.legend(frameon=False)

        baseline_split = report["baseline_two_reference"]["fixtures"][name]["score"]
        repaired_split = fixture["decomposition"]
        groups = np.arange(2)
        width = 0.18
        components = (
            (
                "banked representation",
                baseline_split,
                "representation_component_on_analytic_denominator",
                "0.55",
            ),
            (
                "banked support",
                baseline_split,
                "support_coverage_component_on_analytic_denominator",
                "0.78",
            ),
            (
                "repaired representation",
                repaired_split,
                "representation_component_on_analytic_denominator",
                "seagreen",
            ),
            (
                "repaired support",
                repaired_split,
                "support_component_on_analytic_denominator",
                "yellowgreen",
            ),
        )
        for index, (label, source, key, colour) in enumerate(components):
            values = [
                source[quantity][key]
                for quantity in ("analytic_field", "internal_inductance")
            ]
            split_axis.bar(
                groups + (index - 1.5) * width,
                values,
                width,
                color=colour,
                label=label,
            )
        split_axis.axhline(0.0, color="black", linewidth=0.8)
        split_axis.axhline(
            CHORD_CRESCENT_CONTEXT_FRACTION,
            color="darkorange",
            linestyle=":",
            linewidth=1.0,
        )
        split_axis.axhline(
            -CHORD_CRESCENT_CONTEXT_FRACTION,
            color="darkorange",
            linestyle=":",
            linewidth=1.0,
        )
        split_axis.set_xticks(groups, ["analytic field", "internal inductance"])
        split_axis.set_ylabel("signed fraction of analytic reference")
        split_axis.set_title(f"{name}: same-domain decomposition")
        split_axis.grid(axis="y", alpha=0.18)
        if row == 0:
            split_axis.legend(frameon=False, fontsize=8, ncols=2)
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def validate(report: dict[str, object]) -> dict[str, object]:
    """Validate roots, gate coverage, splits, and figure integrity."""
    closures = []
    for name, _multiplier, cells, _residual in FIXTURES:
        fixture = report["fixtures"][name]
        if fixture["plasma_cells"] != cells:
            raise AssertionError(f"{name} cell count changed")
        if not fixture["fixed_point_residual"]["passed"]:
            raise AssertionError(f"{name} root residual did not reproduce")
        if fixture["physics_gate_count"] != 11:
            raise AssertionError(f"{name} did not score eleven physics gates")
        if len(fixture["physics_gates"]) != 11:
            raise AssertionError(f"{name} gate scorecard is incomplete")
        for quantity in ("analytic_field", "internal_inductance"):
            closure = fixture["decomposition"][quantity]["additive_closure_error"]
            closures.append(abs(closure))
            if abs(closure) > 4.0 * np.finfo(float).eps:
                raise AssertionError(f"{name} {quantity} split does not close")
    if FIGURE_PATH.stat().st_size <= 100_000:
        raise AssertionError("before/after figure is absent or truncated")
    return {
        "fixtures_checked": len(FIXTURES),
        "physics_gates_checked": 11 * len(FIXTURES),
        "maximum_additive_closure_error": max(closures),
        "figure_bytes": FIGURE_PATH.stat().st_size,
        "passed": True,
    }


def add_adjudication(report: dict[str, object]) -> None:
    """Attach the support-term interpretation used for merge adjudication."""
    coarse = report["fixtures"]["coarse"]["decomposition"]
    field = coarse["analytic_field"]
    inductance = coarse["internal_inductance"]
    report["adjudication"] = {
        "coarse_residual_support_term": {
            "analytic_field_absolute_percentage_points": 100.0
            * abs(field["support_component_on_analytic_denominator"]),
            "internal_inductance_absolute_percentage_points": 100.0
            * abs(inductance["support_component_on_analytic_denominator"]),
            "analytic_field_to_chord_crescent_context_ratio": field[
                "support_to_context_scale_ratio"
            ],
            "internal_inductance_to_chord_crescent_context_ratio": inductance[
                "support_to_context_scale_ratio"
            ],
            "interpretation": (
                "The coarse residual support term is about 2.1-2.3 percentage "
                "points, 10-11 times the 0.2-point chord-crescent context. It "
                "therefore includes displacement of the solved support boundary "
                "relative to the stored analytic contour and must not be "
                "attributed to straight-chord curvature alone."
            ),
        }
    }


def main() -> None:
    """Score both loaded roots or validate the banked outputs."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == ["--render-existing"]:
        report = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        render(report)
        print("OBSERVATION_RESCORE_RENDER_EXIT=0", flush=True)
        return
    if sys.argv[1:] == ["--validate-existing"]:
        report = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        print(json.dumps(validate(report), sort_keys=True), flush=True)
        print("OBSERVATION_RESCORE_VALIDATION_EXIT=0", flush=True)
        return
    fine_only = sys.argv[1:] == ["--fixture", "fine"]
    if sys.argv[1:] and not fine_only:
        raise SystemExit(
            "only --fixture fine, --render-existing, or --validate-existing is accepted"
        )

    reference = load_module(REFERENCE_PATH, "observation_rescore_reference")
    reference.configure_dtypes()
    case = reference.require_reference()
    baseline_score = json.loads(BASELINE_SCORE_PATH.read_text(encoding="utf-8"))
    baseline_split = json.loads(BASELINE_SPLIT_PATH.read_text(encoding="utf-8"))
    if fine_only:
        report = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if set(report["fixtures"]) != {"coarse"}:
            raise AssertionError(
                "fine-only continuation requires exactly one coarse result"
            )
        selected_fixtures = tuple(item for item in FIXTURES if item[0] == "fine")
    else:
        report = {
            "schema": "nova.observation-clip-rescore",
            "source_artifacts": {
                "fixture_definition": str(REFERENCE_PATH),
                "banked_gate_scorecard": str(BASELINE_SCORE_PATH),
                "banked_two_reference_scorecard": str(BASELINE_SPLIT_PATH),
                "production_observation_commit": (
                    "9d4d3f4fd9eefee936f7849bc2ea158541a5e30d"
                ),
                "terminal_roots": [
                    str(ROOT_DIRECTORY / "coarse-terminal-root.npz"),
                    str(ROOT_DIRECTORY / "fine-terminal-root.npz"),
                ],
            },
            "policy": {
                "root_handling": "load and score without solving",
                "fixed_point_residual_absolute_tolerance": RESIDUAL_TOLERANCE,
                "physics_gate_count_excluding_root_residual": 11,
                "bounds_unchanged": True,
                "chord_crescent_context_fraction": CHORD_CRESCENT_CONTEXT_FRACTION,
            },
            "baseline_two_reference": {
                "source": str(BASELINE_SPLIT_PATH),
                "fixtures": {
                    name: {"score": baseline_split["fixtures"][name]["score"]}
                    for name, *_rest in FIXTURES
                },
            },
            "fixtures": {},
        }
        write_json(report)
        selected_fixtures = FIXTURES
    for name, multiplier, cells, residual in selected_fixtures:
        report["fixtures"][name] = measure_fixture(
            reference,
            case,
            baseline_score,
            name,
            multiplier,
            cells,
            residual,
        )
        write_json(report)
    report.pop("execution", None)
    add_adjudication(report)
    render(report)
    report["verdict"] = {
        "roots_reproduced": all(
            fixture["fixed_point_residual"]["passed"]
            for fixture in report["fixtures"].values()
        ),
        "physics_gates_passed": {
            name: f"{fixture['physics_gates_passed']}/11"
            for name, fixture in report["fixtures"].items()
        },
        "surviving_misses": {
            name: [
                gate
                for gate, score in fixture["physics_gates"].items()
                if not score["repaired"]["passed"]
            ]
            for name, fixture in report["fixtures"].items()
        },
    }
    report["validation"] = validate(report)
    write_json(report)
    print(f"VERDICT {report['verdict']!r}", flush=True)
    print("OBSERVATION_RESCORE_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
