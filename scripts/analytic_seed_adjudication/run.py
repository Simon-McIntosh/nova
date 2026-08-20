"""Adjudicate the production moment map from the analytic reference seed."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nova.equilibrium import fixed_point
from nova.equilibrium.observation import current_ledger, observe_moments


OUTPUT = Path(__file__).resolve().parent
REFERENCE_PATH = Path("tests/test_equilibrium_forward_reference.py")
RESIDUAL_GATE = 1.0e-6
ROUTES = (
    ("undamped", None),
    ("eigen_weight", 1.0 / 1.766),
    ("weight_0p55", 0.55),
)
CONTROLS = {
    "coarse": {
        "legacy_seed_defect": 9.54e-3,
        "consistent_seed_defect": 1.29e-1,
        "best_direct_residual": 1.67e-2,
    },
    "fine": {
        "legacy_seed_defect": 1.16e-2,
        "consistent_seed_defect": 2.35e-2,
        "best_direct_residual": 1.20e-2,
    },
}


def load_reference_module():
    """Load the banked fixture definitions without running its test suite."""
    spec = importlib.util.spec_from_file_location("reference_fixture", REFERENCE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {REFERENCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def relative_residual(map_fn, state) -> float:
    """Return the production map displacement relative to its image scale."""
    mapped = map_fn(state)
    return float(
        jnp.max(jnp.abs(mapped - state))
        / jnp.maximum(jnp.max(jnp.abs(mapped)), 1.0e-30)
    )


def topology_receipt(operator, state) -> dict[str, object]:
    """Report the physical branch carried by a terminal state."""
    masks, topology = operator.read(state)
    current = operator.cell_current_moments(state).cell_current
    core_cells = int(np.count_nonzero(np.asarray(masks.core)))
    if core_cells == 0:
        classification = "vacuum"
    elif bool(topology.diverted):
        classification = "diverted"
    else:
        classification = "limited"
    return {
        "axis_m": np.asarray(topology.axis).tolist(),
        "plasma_current_a": float(jnp.sum(current)),
        "core_cells": core_cells,
        "topology": classification,
    }


def score_physics(reference, case, machine, operator, history):
    """Score every unchanged physical gate at a residual-qualified root."""
    flux = history.state
    masks, topology = operator.read(flux)
    current_masks = operator.current_domain_masks(flux)
    current_moments = operator.cell_current_moments(flux)
    cell_current = current_moments.cell_current
    moments = observe_moments(
        operator.source,
        current_masks,
        jnp.asarray(machine.radius),
        operator.area,
        cell_current,
        machine.poloidal_field_squared(operator.external_current, current_moments),
        topology.flux_span,
    )
    solved = reference.SolvedEquilibrium(
        case=case,
        machine=machine,
        flux=flux,
        cell_current=cell_current,
        masks=masks,
        topology=topology,
        moments=moments,
        ledger=current_ledger(cell_current, current_masks),
        fixed_point=history,
    )
    profile = reference.ForwardProfile(
        operator=operator,
        lattice=reference.receipt_mesh(machine),
        newton_steps=reference.NEWTON_STEPS,
    )
    receipt = profile.observe(flux)
    analytic_moments = case.map_moments()
    scale = solved.reference_scale
    core = np.asarray(masks.core)
    flux_deviation = (
        float(
            np.max(np.abs(solved.grid_flux - solved.reference_flux)[core])
            / abs(case.flux_span)
        )
        if np.any(core)
        else float("inf")
    )
    score = {
        "residual": float(history.residual),
        "axis_max_deviation_m": float(
            np.max(np.abs(np.asarray(topology.axis) - case.axis))
        ),
        "flux_deviation_fraction": flux_deviation,
        "plasma_current_deviation_fraction": abs(
            float(moments.plasma_current) / case.plasma_current - 1.0
        ),
        "poloidal_beta_deviation_fraction": abs(
            float(moments.poloidal_beta) * scale / analytic_moments["poloidal_beta"]
            - 1.0
        ),
        "internal_inductance_deviation_fraction": abs(
            float(moments.internal_inductance)
            * scale
            / analytic_moments["internal_inductance"]
            - 1.0
        ),
        "analytic_field_quadrature_deviation_fraction": abs(
            float(moments.poloidal_field_integral) / analytic_moments["field_integral"]
            - 1.0
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
    }
    gates = {
        "residual": score["residual"] < reference.RESIDUAL_TOLERANCE,
        "axis": score["axis_max_deviation_m"] < reference.AXIS_TOLERANCE,
        "flux": score["flux_deviation_fraction"] < reference.FLUX_TOLERANCE,
        "plasma_current": score["plasma_current_deviation_fraction"]
        < reference.PLASMA_CURRENT_TOLERANCE,
        "poloidal_beta": score["poloidal_beta_deviation_fraction"]
        < reference.MOMENT_TOLERANCE,
        "internal_inductance": score["internal_inductance_deviation_fraction"]
        < reference.MOMENT_TOLERANCE,
        "analytic_field": score["analytic_field_quadrature_deviation_fraction"]
        < reference.QUADRATURE_TOLERANCE,
        "receipt_field": score["receipt_field_deviation_fraction"]
        < reference.FIELD_INTEGRAL_TOLERANCE,
        "grad_shafranov": score["grad_shafranov_residual"]
        < reference.GRAD_SHAFRANOV_TOLERANCE,
        "divergence_b": score["relative_divergence_b"]
        < reference.DIVERGENCE_MARGIN * score["grad_shafranov_residual"],
        "divergence_j": score["relative_divergence_j"]
        < reference.DIVERGENCE_MARGIN * score["grad_shafranov_residual"],
        "physical": score["core_cells"] > 0 and score["diverted"],
    }
    return score, gates


def save(report: dict[str, object]) -> None:
    """Checkpoint the scorecard after every expensive result."""
    (OUTPUT / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_route(reference, map_fn, seed, weight):
    """Run one fixed-budget route from the unmodified analytic seed."""
    if weight is None:
        return fixed_point.newton_krylov(
            map_fn,
            seed,
            newton_steps=reference.NEWTON_STEPS,
            gmres_iterations=reference.KRYLOV_ITERATIONS,
            warmup=0,
        )
    return fixed_point.kink_aware_newton_krylov(
        map_fn,
        seed,
        strategy="damped_hybrid",
        newton_steps=reference.NEWTON_STEPS,
        gmres_iterations=reference.KRYLOV_ITERATIONS,
        warmup=0,
        hybrid_weight=weight,
        hybrid_schedule="fixed",
    )


def measure_fixture(reference, case, machine, name, report) -> None:
    """Measure the seed defect and three direct routes on one fixture."""
    operator = reference.forward_operator(case, machine)
    map_fn = operator.flux_map()
    seed = reference.seed_flux(case, machine)
    seed_defect = relative_residual(map_fn, seed)
    controls = CONTROLS[name]
    seed_beats_legacy = seed_defect < controls["legacy_seed_defect"]
    fixture = {
        "plasma_cells": int(len(machine.node)),
        "wall_targets": int(len(machine.wall_node)),
        "state_size": int(seed.size),
        "seed_map_defect": {
            "repaired_unified": seed_defect,
            "legacy_control": controls["legacy_seed_defect"],
            "pre_repair_consistent_control": controls["consistent_seed_defect"],
            "beats_legacy": seed_beats_legacy,
            "margin_to_legacy": controls["legacy_seed_defect"] - seed_defect,
            "ratio_to_legacy": seed_defect / controls["legacy_seed_defect"],
        },
        "banked_best_direct_residual": controls["best_direct_residual"],
        "routes": {},
    }
    report["fixtures"][name] = fixture
    save(report)
    print(
        f"DEFECT fixture={name} cells={len(machine.node)} repaired={seed_defect:.17g} "
        f"legacy={controls['legacy_seed_defect']:.17g} "
        f"margin={controls['legacy_seed_defect'] - seed_defect:.17g}",
        flush=True,
    )

    for route_name, weight in ROUTES:
        history = run_route(reference, map_fn, seed, weight)
        jax.block_until_ready(history.state)
        residual = float(history.residual)
        converged = bool(np.isfinite(residual) and residual < RESIDUAL_GATE)
        trace = np.asarray(history.trace)
        route = {
            **topology_receipt(operator, history.state),
            "weight": weight,
            "residual": residual,
            "residual_history": trace[np.isfinite(trace)].tolist(),
            "converged": converged,
        }
        if converged:
            score, gates = score_physics(reference, case, machine, operator, history)
            route["physics_score"] = score
            route["physics_gates"] = gates
            route["all_physics_gates"] = bool(all(gates.values()))
        fixture["routes"][route_name] = route
        save(report)
        print(
            f"ROUTE fixture={name} route={route_name} weight={weight} "
            f"residual={residual:.17g} converged={converged} "
            f"topology={route['topology']} core={route['core_cells']} "
            f"current_a={route['plasma_current_a']:.17g}",
            flush=True,
        )

    best_name, best = min(
        fixture["routes"].items(), key=lambda item: item[1]["residual"]
    )
    fixture["best_route"] = best_name
    fixture["best_terminal_residual"] = best["residual"]
    fixture["best_vs_banked_control"] = (
        best["residual"] - controls["best_direct_residual"]
    )
    fixture["any_converged"] = any(
        route["converged"] for route in fixture["routes"].values()
    )
    save(report)
    print(
        f"BEST fixture={name} route={best_name} residual={best['residual']:.17g} "
        f"delta_to_control={fixture['best_vs_banked_control']:.17g}",
        flush=True,
    )


def render_figures(report: dict[str, object]) -> None:
    """Render the seed comparison and fixed-budget residual trajectories."""
    names = ("coarse", "fine")
    schemes = (
        ("legacy control", "legacy_control", "#777777"),
        ("pre-repair consistent", "pre_repair_consistent_control", "#d95f02"),
        ("repaired unified", "repaired_unified", "#1b9e77"),
    )
    x = np.arange(len(names))
    width = 0.24
    figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for offset, (label, key, color) in enumerate(schemes):
        values = [report["fixtures"][name]["seed_map_defect"][key] for name in names]
        axis.bar(x + (offset - 1) * width, values, width, label=label, color=color)
    axis.axhline(RESIDUAL_GATE, color="black", linestyle=":", linewidth=1)
    axis.set_yscale("log")
    axis.set_xticks(x, ["coarse: 566 cells", "fine: 1069 cells"])
    axis.set_ylabel("analytic-seed relative map defect")
    axis.set_title("Unified-map seed defect against banked controls")
    axis.legend(frameon=False)
    figure.savefig(OUTPUT / "seed-defect-comparison.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    colors = {
        "undamped": "#7570b3",
        "eigen_weight": "#1b9e77",
        "weight_0p55": "#d95f02",
    }
    for axis, name in zip(axes, names, strict=True):
        fixture = report["fixtures"][name]
        for route_name, route in fixture["routes"].items():
            history = np.asarray(route["residual_history"])
            axis.semilogy(
                np.arange(1, len(history) + 1),
                history,
                marker="o",
                markersize=3,
                label=route_name.replace("_", " "),
                color=colors[route_name],
            )
        axis.axhline(RESIDUAL_GATE, color="black", linestyle=":", linewidth=1)
        axis.set_title(f"{name}: {fixture['plasma_cells']} cells")
        axis.set_xlabel("reported residual evaluation")
        axis.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel("relative fixed-point residual")
    axes[1].legend(frameon=False)
    figure.suptitle("Direct 10-step, 30-GMRES analytic-seed routes")
    figure.savefig(OUTPUT / "direct-residual-trends.png", dpi=180)
    plt.close(figure)


def main() -> None:
    """Build both banked fixtures once and write the adjudication scorecard."""
    reference = load_reference_module()
    reference.configure_dtypes()
    case = reference.require_reference()
    report = {
        "schema": "nova.analytic-seed-adjudication",
        "lineage_head": "03a0a774",
        "policy": {
            "newton_steps": int(reference.NEWTON_STEPS),
            "gmres_iterations": int(reference.KRYLOV_ITERATIONS),
            "warmup": 0,
            "residual_gate": RESIDUAL_GATE,
            "damping_weights": [weight for _, weight in ROUTES],
            "full_physics_score_trigger": "residual below residual_gate",
        },
        "controls": CONTROLS,
        "fixtures": {},
    }
    save(report)

    print(
        f"CACHE_REQUEST fixture=coarse requested={reference.SUITE_CELLS} "
        f"wall_nodes={reference.WALL_NODES}",
        flush=True,
    )
    coarse = reference.cached_machine(case, reference.SUITE_CELLS, passive=True)
    print(reference.machine_cache_summary("coarse", coarse), flush=True)
    if len(coarse.node) != 566:
        raise AssertionError(f"expected 566 coarse cells, got {len(coarse.node)}")
    measure_fixture(reference, case, coarse, "coarse", report)

    reference.WALL_NODES *= 2
    fine_request = 2 * reference.SUITE_CELLS
    print(
        f"CACHE_REQUEST fixture=fine requested={fine_request} "
        f"wall_nodes={reference.WALL_NODES}",
        flush=True,
    )
    fine = reference.cached_machine(case, fine_request, passive=True)
    print(reference.machine_cache_summary("fine", fine), flush=True)
    if len(fine.node) != 1069:
        raise AssertionError(f"expected 1069 fine cells, got {len(fine.node)}")
    measure_fixture(reference, case, fine, "fine", report)

    report["verdict"] = {
        "repaired_beats_legacy_both": all(
            fixture["seed_map_defect"]["beats_legacy"]
            for fixture in report["fixtures"].values()
        ),
        "any_direct_root": any(
            fixture["any_converged"] for fixture in report["fixtures"].values()
        ),
        "all_fixtures_have_direct_root": all(
            fixture["any_converged"] for fixture in report["fixtures"].values()
        ),
        "no_root_obstruction": (
            None
            if any(fixture["any_converged"] for fixture in report["fixtures"].values())
            else (
                "The repaired map remains outside the analytic seed's direct Newton "
                "basin: fixed hybrid damping changes the terminal cycle but leaves a "
                "finite residual floor on the diverted branch."
            )
        ),
    }
    save(report)
    render_figures(report)
    print(f"VERDICT {report['verdict']!r}", flush=True)
    print("ANALYTIC_SEED_ADJUDICATION_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
