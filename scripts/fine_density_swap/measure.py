"""Confirm the analytic-density forcing component on the fine fixture."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import jax
import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path(__file__).resolve().parent
REFERENCE_PATH = Path("tests/test_equilibrium_forward_reference.py")
FORCING_PATH = Path("scripts/map_forcing_attribution/measure.py")
DECOMPOSITION_PATH = Path("scripts/forcing_residual_decomposition/measure.py")
GATE_PATH = Path("scripts/root_gate_attribution/measure_root_attribution.py")
COARSE_RESULTS = Path("scripts/forcing_residual_decomposition/results.json")
CACHE_RESULTS = Path("scripts/build_cache_audit/results.json")
FINE_TERMINAL = Path("scripts/root_gate_attribution/fine-terminal-root.npz")
EXPECTED_FINE_CELLS = 1069
EXPECTED_CACHE_KEY = "f0f96aa214aa9459"


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
    """Read one compressed result bank into detached arrays."""
    with np.load(path) as stored:
        return {name: stored[name] for name in stored.files}


def cache_children(store: str, key: str) -> list[str]:
    """List the array groups persisted under one semantic machine key."""
    group = Path(store) / key
    if not group.is_dir():
        raise AssertionError(f"semantic cache group is absent: {group}")
    return sorted(path.name for path in group.iterdir() if path.name != "zarr.json")


def grid_score(component: np.ndarray, comparator: np.ndarray, cells: int, span: float):
    """Score the plasma-grid part of a forcing component."""
    component_grid = component[:cells]
    comparator_grid = comparator[:cells]
    denominator = float(np.dot(comparator_grid, comparator_grid))
    return {
        "projection_fraction": float(
            np.dot(component_grid, comparator_grid) / denominator
        ),
        "sup_fraction_of_span": float(np.max(np.abs(component_grid)) / abs(span)),
        "rms_fraction_of_span": float(np.sqrt(np.mean(component_grid**2)) / abs(span)),
        "sup_wb": float(np.max(np.abs(component_grid))),
        "rms_wb": float(np.sqrt(np.mean(component_grid**2))),
    }


def refinement_estimate(coarse: float, fine: float, coarse_cells: int, fine_cells: int):
    """Fit a two-point power against nominal cell width."""
    width_ratio = math.sqrt(fine_cells / coarse_cells)
    order = math.log(coarse / fine) / math.log(width_ratio)
    return {
        "coarse_value": coarse,
        "fine_value": fine,
        "nominal_coarse_to_fine_width_ratio": width_ratio,
        "estimated_power_against_cell_width": order,
        "fine_to_coarse_ratio": fine / coarse,
        "qualification": (
            "Two grids determine this exponent exactly; it is a refinement "
            "estimate, not an asymptotic convergence order."
        ),
    }


def render(report: dict[str, object]) -> None:
    """Plot the density share, response, and two-point forcing trend."""
    coarse = report["coarse_control"]
    fine = report["fine_measurement"]
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), constrained_layout=True)

    axes[0].bar(
        ["coarse", "fine"],
        [
            100.0 * coarse["density_projection_fraction"],
            100.0 * fine["forcing"]["projection_fraction"],
        ],
        color=["#9ecae9", "#3182bd"],
    )
    axes[0].axhline(100.0, color="black", lw=0.8, ls="--")
    axes[0].set_ylabel("density projection [% comparator]")
    axes[0].set_title("Source-density share")

    metrics = (
        "axis_signed_projection_mm",
        "flux_signed_peak_percent_of_span",
        "plasma_current_signed_percent",
    )
    labels = ("axis [mm]", "flux [% span]", "current [%]")
    x = np.arange(len(metrics))
    width = 0.34
    axes[1].bar(
        x - width / 2,
        [coarse["root_response"][name] for name in metrics],
        width,
        label="coarse",
        color="#9ecae9",
    )
    axes[1].bar(
        x + width / 2,
        [fine["root_response"][name] for name in metrics],
        width,
        label="fine",
        color="#3182bd",
    )
    axes[1].set_xticks(x, labels)
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_title("Linear-inverse response")
    axes[1].legend()

    trend = report["refinement"]
    widths = np.asarray([trend["nominal_coarse_to_fine_width_ratio"], 1.0])
    values = np.asarray([trend["coarse_value"], trend["fine_value"]])
    axes[2].plot(widths, values, "o-", color="#e6550d")
    axes[2].set_xticks(widths, ["coarse", "fine"])
    axes[2].set_xlabel("nominal width / fine width")
    axes[2].set_ylabel("density forcing sup [Wb]")
    axes[2].set_title(
        f"two-point width power {trend['estimated_power_against_cell_width']:.3f}"
    )
    axes[2].grid(True, alpha=0.25)

    figure.savefig(OUTPUT / "fine-density-swap.png", dpi=180)
    plt.close(figure)


def main() -> None:
    """Run the warm-loaded fine density swap and bank its diagnostics."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference = load_module(REFERENCE_PATH, "fine_density_reference")
    forcing_module = load_module(FORCING_PATH, "fine_density_forcing")
    decomposition = load_module(DECOMPOSITION_PATH, "fine_density_decomposition")
    gate_module = load_module(GATE_PATH, "fine_density_gate")
    reference.configure_dtypes()
    case = reference.require_reference()
    coarse_bank = json.loads(COARSE_RESULTS.read_text(encoding="utf-8"))
    cache_bank = json.loads(CACHE_RESULTS.read_text(encoding="utf-8"))
    terminal = load_npz(FINE_TERMINAL)["state"]

    reference.WALL_NODES = 6
    requested = 2 * reference.SUITE_CELLS
    print(
        f"CACHE_REQUEST fixture=fine requested={requested} expectation=warm",
        flush=True,
    )
    machine = reference.cached_machine(case, requested, passive=True)
    receipt = machine.cache_receipt
    if receipt is None or not receipt.hit:
        raise AssertionError("fine carrier rebuilt instead of warm-loading")
    if receipt.key != EXPECTED_CACHE_KEY:
        raise AssertionError(f"fine cache key changed: {receipt.key}")
    if len(machine.node) != EXPECTED_FINE_CELLS:
        raise AssertionError(
            f"expected {EXPECTED_FINE_CELLS} fine cells, got {len(machine.node)}"
        )
    children = cache_children(receipt.store, receipt.key)
    expected_children = sorted(reference._packed_machine_arrays(machine))
    if children != expected_children:
        raise AssertionError("persisted cache children differ from the carrier arrays")
    print(reference.machine_cache_summary("fine", machine), flush=True)
    print(
        f"CACHE_CHILDREN key={receipt.key} count={len(children)} "
        f"coupling={sum('to_' in name or 'field_' in name for name in children)}",
        flush=True,
    )

    print("OPERATOR reconstructing from cached carrier", flush=True)
    operator = reference.forward_operator(case, machine)
    exact = reference.seed_flux(case, machine)
    if len(terminal) != len(exact):
        raise AssertionError("fine terminal root and exact map state differ in shape")
    print(f"LINEARIZE state_size={len(exact)}", flush=True)
    mapped, tangent = jax.linearize(operator.flux_map(), exact)
    jax.block_until_ready(mapped)
    forcing = np.asarray(mapped - exact)

    print("COMPARATOR evaluating omitted second moments", flush=True)
    coupled = operator.cell_current_moments(exact)
    second_residual, _actual, _reconstructed = forcing_module.second_moment_residual(
        case, operator, exact, coupled
    )
    correction, active_cells, pair_count = forcing_module.second_order_correction(
        machine, second_residual, forcing_module.HESSIAN_STEPS[-1]
    )
    comparator = forcing + correction

    print(
        "DENSITY_SWAP integrating analytic density over production supports",
        flush=True,
    )
    masks, topology = operator.read(exact)
    support, current, first, _coupled = gate_module.support_reference(
        case, operator, exact, masks, topology
    )
    straight_image = np.asarray(operator.external()) + decomposition.internal_flux(
        operator, current, first
    )
    density_component = np.asarray(mapped) + correction - straight_image
    forcing_score = decomposition.score(density_component, comparator)
    grid_forcing = grid_score(
        density_component, comparator, len(machine.node), case.flux_span
    )

    print("LINEAR_RESPONSE applying tangent inverse", flush=True)
    step, solve = forcing_module.solve_response(tangent, density_component)
    response = decomposition.response_observables(
        case, machine, operator, exact, terminal, step
    )

    coarse_component = coarse_bank["components"]["source_density"]["forcing"]
    coarse_response = coarse_bank["components"]["source_density"]["root_response"]
    refinement = refinement_estimate(
        coarse_component["sup_wb"],
        forcing_score["sup_wb"],
        coarse_bank["controls"]["plasma_cells"],
        len(machine.node),
    )
    nearly_constant = 0.5 <= refinement["fine_to_coarse_ratio"] <= 2.0

    prior_cold = cache_bank["timings"]["fine"]["cache_measurement"]["cold"]
    report = {
        "schema": "nova.fine-density-swap",
        "cache_carrier": {
            "requested_cells": requested,
            "realised_cells": len(machine.node),
            "store": receipt.store,
            "semantic_key": receipt.key,
            "warm_hit": receipt.hit,
            "warm_load_seconds": receipt.load_seconds,
            "arrays_verified": receipt.arrays_verified,
            "bytes_verified": receipt.bytes_verified,
            "persisted_child_count": len(children),
            "persisted_children": children,
            "persisted_coupling_child_count": sum(
                "to_" in name or "field_" in name for name in children
            ),
            "banked_cold_build": {
                "artifact": str(CACHE_RESULTS),
                "hit": prior_cold["hit"],
                "key": prior_cold["key"],
                "build_seconds": prior_cold["build_seconds"],
                "arrays_verified": prior_cold["arrays_verified"],
                "bytes_verified": prior_cold["bytes_verified"],
            },
        },
        "coarse_control": {
            "plasma_cells": coarse_bank["controls"]["plasma_cells"],
            "density_projection_fraction": coarse_component["projection_fraction"],
            "density_projection_percent": 100.0
            * coarse_component["projection_fraction"],
            "density_forcing_sup_wb": coarse_component["sup_wb"],
            "root_response": {
                name: coarse_response[name]
                for name in (
                    "axis_signed_projection_mm",
                    "flux_signed_peak_percent_of_span",
                    "plasma_current_signed_percent",
                )
            },
        },
        "fine_measurement": {
            "plasma_cells": len(machine.node),
            "state_size": len(exact),
            "forcing": forcing_score,
            "forcing_projection_percent": 100.0 * forcing_score["projection_fraction"],
            "grid_forcing": grid_forcing,
            "root_response": {**solve, **response},
            "support_cells": int(
                np.count_nonzero(np.asarray(support.vertex_count) >= 3)
            ),
            "second_moment_comparator": {
                "active_cells": active_cells,
                "near_source_target_pairs": pair_count,
                "correction_sup_wb": float(np.max(np.abs(correction))),
            },
        },
        "refinement": refinement,
        "verdict": {
            "density_is_h_independent_carrier_estimate": nearly_constant,
            "basis": (
                "The fine-to-coarse absolute sup ratio and its two-point width "
                "power distinguish persistence from decay; the two-grid result "
                "is explicitly an estimate."
            ),
            "measurement_only": True,
        },
        "source_artifacts": {
            "coarse_decomposition": str(COARSE_RESULTS),
            "fine_terminal_root": str(FINE_TERMINAL),
            "cache_audit": str(CACHE_RESULTS),
        },
    }
    render(report)
    report["artifacts"] = {
        "figure": str(OUTPUT / "fine-density-swap.png"),
        "figure_bytes": (OUTPUT / "fine-density-swap.png").stat().st_size,
    }
    (OUTPUT / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not report["cache_carrier"]["warm_hit"]:
        raise AssertionError("fine carrier was not a warm load")
    if abs(report["coarse_control"]["density_projection_percent"] - 99.176) > 0.001:
        raise AssertionError("coarse density-share control did not reproduce")
    if report["fine_measurement"]["root_response"]["residual_relative_sup"] > 1.0e-6:
        raise AssertionError("fine tangent-inverse residual is too large")
    fine_share = report["fine_measurement"]["forcing_projection_percent"]
    print(
        f"RESULT density_share_percent={fine_share:.9f} "
        f"axis_mm={response['axis_signed_projection_mm']:.9f} "
        f"flux_percent={response['flux_signed_peak_percent_of_span']:.9f} "
        f"current_percent={response['plasma_current_signed_percent']:.9f} "
        f"width_power={refinement['estimated_power_against_cell_width']:.9f}",
        flush=True,
    )
    print("FINE_DENSITY_SWAP_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
