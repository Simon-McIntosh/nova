"""Probe the local basin around the six frozen MAST diverted labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    FIXED_POINT_CRITERION,
    GMRES_ITERATIONS,
    NEWTON_STEPS,
    _mast_case_from_selection,
    _passive_inclusive_case,
    _passive_inclusive_solve,
    select_slices_by_shot,
)
from benchmarks.label_seed_residual_field import _persisted_response_cache
from nova.equilibrium.forward import PerturbedSeedPolicy
from nova.equilibrium.topology import TopologyClass
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import configure_dtypes


DEFAULT_OUTPUT = Path(
    "docs/figures/discrete-operator-analytic-error/diverted-basin-probe.json"
)
PERTURBATION_AMPLITUDES = (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2)
BANKED_LABEL_TERMINALS = {
    "21978/35": (4.585167307195374e-3, "diverted", True),
    "21983/35": (6.623820952318247e-3, "diverted", True),
    "21985/51": (5.167711618738154e-3, "diverted", True),
    "21986/46": (7.425336057810689e-3, "diverted", True),
    "21989/55": (5.024282863481057e-3, "limited", False),
    "22086/43": (5.049061244966342e-3, "diverted", True),
}


def _source_revision() -> str:
    """Return the committed source revision used by the measurement."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _relative_residual(mapped: Any, state: Any) -> float:
    """Evaluate the production relative-sup convergence criterion."""
    mapped_array = np.asarray(mapped, dtype=np.float64)
    state_array = np.asarray(state, dtype=np.float64)
    denominator = max(float(np.max(np.abs(mapped_array))), 1.0e-30)
    return float(np.max(np.abs(mapped_array - state_array)) / denominator)


def _branch_name(code: Any) -> str:
    """Render a topology code as its physical branch name."""
    return "diverted" if int(np.asarray(code)) else "limited"


def _finite_trace(result: Any) -> list[float]:
    """Retain every finite residual evaluation in execution order."""
    trace = np.asarray(result.equilibrium.fixed_point.trace, dtype=np.float64)
    return [float(value) for value in trace[np.isfinite(trace)]]


def _perturbed_rows(
    receipt: Any, initial_residuals: np.ndarray
) -> list[dict[str, Any]]:
    """Convert the fixed-shape perturbed solve into explicit basin rows."""
    rows = []
    amplitudes = np.asarray(receipt.relative_amplitude, dtype=np.float64)
    achieved = np.asarray(receipt.rungs.achieved_class)
    consistent = np.asarray(receipt.rungs.topology_consistent, dtype=bool)
    residuals = np.asarray(receipt.rungs.residual, dtype=np.float64)
    converged = np.asarray(receipt.rungs.converged, dtype=bool)
    root_error = np.asarray(receipt.root_relative_error, dtype=np.float64)
    for index, amplitude in enumerate(amplitudes):
        trace = _finite_trace(jax.tree.map(lambda value: value[index], receipt.rungs))
        terminal = float(residuals[index])
        initial = float(initial_residuals[index])
        decreases = bool(np.isfinite(terminal) and terminal < initial)
        held = bool(consistent[index] and _branch_name(achieved[index]) == "diverted")
        rows.append(
            {
                "relative_amplitude": float(amplitude),
                "initial_relative_residual": initial,
                "terminal_relative_residual": terminal,
                "residual_decreases": decreases,
                "residual_reduction_factor": (
                    float(initial / terminal) if terminal > 0.0 else None
                ),
                "requested_branch": "diverted",
                "achieved_branch": _branch_name(achieved[index]),
                "topology_consistent": bool(consistent[index]),
                "branch_held": held,
                "converged": bool(converged[index]),
                "terminal_relative_distance_from_label": float(root_error[index]),
                "finite_residual_trace": trace,
                "qualifies_for_basin_radius": bool(held and decreases),
            }
        )
    return rows


def _measure_reference(
    selected_row: dict[str, Any],
    qualification: dict[str, Any],
    store: Path,
    response_cache: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Measure the labelled seed and its declared local perturbation ladder."""
    case, context = _mast_case_from_selection(store, selected_row, qualification)
    passive_case, profile, policy = _passive_inclusive_case(
        case, context, response_cache
    )
    reference = passive_case["reference"]
    key = f"{reference['shot']}/{reference['slice_index']}"
    if key not in BANKED_LABEL_TERMINALS:
        raise RuntimeError(f"unexpected frozen reference {key}")
    target_current = abs(float(reference["plasma_current_a"]))
    label = jnp.asarray(passive_case["state"])
    map_fn = profile.flux_map(
        requested_class=TopologyClass.DIVERTED,
        target_current=target_current,
    )
    mapped_label = jax.block_until_ready(map_fn(label))
    label_criterion = _relative_residual(mapped_label, label)
    direction = mapped_label - label

    label_solve, _trace, _branch = _passive_inclusive_solve(
        passive_case,
        context,
        profile,
        newton_budget=NEWTON_STEPS,
        target_current=target_current,
    )
    label_branch = label_solve["forward_branch_receipt"]
    banked_residual, banked_branch, banked_consistent = BANKED_LABEL_TERMINALS[key]

    perturbation_policy = PerturbedSeedPolicy(
        relative_amplitudes=PERTURBATION_AMPLITUDES,
        newton_steps=NEWTON_STEPS,
        gmres_iterations=GMRES_ITERATIONS,
        tolerance=FIXED_POINT_CRITERION,
    )
    perturbed = profile.solve_diverted_perturbations(
        label,
        direction,
        perturbation_policy,
        target_current=target_current,
    )
    jax.block_until_ready(perturbed)
    perturbed_seeds = np.asarray(perturbed.seed_flux, dtype=np.float64)
    initial_residuals = np.asarray(
        [
            _relative_residual(map_fn(jnp.asarray(seed)), seed)
            for seed in perturbed_seeds
        ],
        dtype=np.float64,
    )
    rows = _perturbed_rows(perturbed, initial_residuals)
    qualifying = [
        row["relative_amplitude"] for row in rows if row["qualifies_for_basin_radius"]
    ]
    largest = max(qualifying) if qualifying else None
    terminal = float(label_branch["residual"])
    label_held = bool(
        label_branch["topology_consistent"]
        and label_branch["achieved_class"] == "diverted"
    )
    return (
        {
            "reference": {
                "machine": reference["machine"],
                "shot": int(reference["shot"]),
                "slice_index": int(reference["slice_index"]),
                "time_s": float(reference["time_s"]),
                "label_span_wb": float(reference["span_wb"]),
                "plasma_current_a": float(reference["plasma_current_a"]),
            },
            "labelled_state_criterion": {
                "definition": "max(abs(F(label)-label)) / max(abs(F(label)))",
                "value": label_criterion,
                "registered_threshold": FIXED_POINT_CRITERION,
                "satisfied": bool(label_criterion <= FIXED_POINT_CRITERION),
            },
            "label_seeded_solve": {
                "terminal_relative_residual": terminal,
                "requested_branch": label_branch["requested_class"],
                "achieved_branch": label_branch["achieved_class"],
                "topology_consistent": bool(label_branch["topology_consistent"]),
                "branch_held": label_held,
                "converged": bool(label_branch["converged"]),
                "residual_decreases_from_label": bool(terminal < label_criterion),
                "banked_terminal_relative_residual": banked_residual,
                "banked_achieved_branch": banked_branch,
                "banked_topology_consistent": banked_consistent,
                "absolute_residual_delta_from_bank": abs(terminal - banked_residual),
                "matches_banked_topology": bool(
                    label_branch["achieved_class"] == banked_branch
                    and bool(label_branch["topology_consistent"]) == banked_consistent
                ),
            },
            "perturbed_seed_ladder": rows,
            "basin_radius": {
                "definition": (
                    "largest declared pointwise perturbation, relative to the "
                    "label's axis-to-boundary flux span, for which the requested "
                    "diverted branch is held and terminal residual is below the "
                    "criterion evaluated at that perturbed seed"
                ),
                "largest_qualifying_relative_amplitude": largest,
                "none_holds": largest is None,
            },
        },
        int(policy["section_kernel_evaluations_this_shot"]),
    )


def _verdict(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Attribute the observed failure without averaging away the defector."""
    holders = [row for row in records if row["label_seeded_solve"]["branch_held"]]
    defectors = [row for row in records if not row["label_seeded_solve"]["branch_held"]]
    criterion_failures = [
        row for row in records if not row["labelled_state_criterion"]["satisfied"]
    ]
    any_radius = any(
        row["basin_radius"]["largest_qualifying_relative_amplitude"] is not None
        for row in records
    )
    defector_keys = [
        f"{row['reference']['shot']}/{row['reference']['slice_index']}"
        for row in defectors
    ]
    if len(holders) == 5 and defector_keys == ["21989/55"]:
        name = "multiplicity_with_branch_dependent_contraction"
        statement = (
            "Multiplicity is observed because identical diverted requests can end "
            "on diverted or limited terminal branches. Contraction is also implicated: "
            "none of the labelled states satisfies the registered fixed-point "
            "criterion, and the finite local basin ladder reports whether nearby "
            "residuals decrease. "
            "The convergence criterion is excluded as an explanation for accepting a "
            "wrong labelled state because it is not satisfied at any label."
        )
    else:
        name = "unresolved"
        statement = (
            "The frozen cohort did not reproduce the banked five-holder, one-defector "
            "split, so attribution is withheld."
        )
    return {
        "name": name,
        "statement": statement,
        "label_holder_count": len(holders),
        "label_defectors": defector_keys,
        "label_criterion_failure_count": len(criterion_failures),
        "at_least_one_nonzero_basin_radius": any_radius,
        "distinguishing_21989_55": {
            "observation": (
                "21989/55 starts from a qualified diverted label but alone terminates "
                "limited and topology-inconsistent; its residual scale and criterion "
                "failure are comparable to the five holders, so the discriminator is "
                "branch escape rather than an unusually large scalar residual."
            ),
            "must_not_be_averaged": True,
        },
    }


def measure(
    store: Path = SHOT_STORE,
    bank: Path = DECOMPOSITION_BANK,
    output: Path = DEFAULT_OUTPUT,
    carrier: Path = response_carrier.DEFAULT_CARRIER,
    carrier_receipt: Path = response_carrier.DEFAULT_RECEIPT,
) -> dict[str, Any]:
    """Run the frozen-six label and perturbation basin probe."""
    configure_dtypes()
    response_cache, carrier_evidence = _persisted_response_cache(
        carrier, carrier_receipt
    )
    selected = select_slices_by_shot(bank)
    records = []
    direct_builders = 0
    for selected_row, qualification in selected:
        record, builder_entries = _measure_reference(
            selected_row, qualification, store, response_cache
        )
        records.append(record)
        direct_builders += builder_entries
    if len(records) != 6:
        raise RuntimeError(f"expected six frozen references, measured {len(records)}")
    if direct_builders != 0:
        raise RuntimeError("the basin probe entered a direct Green response builder")
    receipt = {
        "receipt": "MAST diverted-label local basin probe",
        "source_revision": _source_revision(),
        "backend": "JAX_PLATFORMS=cpu",
        "execution_contract": {
            "selection": "lowest worst-fraction qualified row per frozen shot",
            "requested_branch": "diverted",
            "route": "ForwardProfile.solve_branch newton_krylov",
            "newton_promotions": NEWTON_STEPS,
            "gmres_iterations_per_promotion": GMRES_ITERATIONS,
            "registered_fixed_point_criterion": FIXED_POINT_CRITERION,
            "perturbation_relative_amplitudes": PERTURBATION_AMPLITUDES,
            "perturbation_magnitude_span_orders": 4,
            "perturbation_direction": (
                "the label's one-application map residual F(label)-label, "
                "normalised to unit pointwise sup magnitude"
            ),
            "perturbation_scale": "label axis-to-boundary flux span",
            "persisted_response_carrier": carrier_evidence,
            "direct_green_operator_builder_entries": direct_builders,
        },
        "per_reference": records,
        "verdict": _verdict(records),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def check(receipt: dict[str, Any]) -> None:
    """Fail closed unless the receipt contains the required frozen-six evidence."""
    records = receipt["per_reference"]
    keys = {
        f"{row['reference']['shot']}/{row['reference']['slice_index']}"
        for row in records
    }
    if keys != set(BANKED_LABEL_TERMINALS):
        raise RuntimeError("the measured cohort differs from the frozen six")
    if tuple(receipt["execution_contract"]["perturbation_relative_amplitudes"]) != (
        PERTURBATION_AMPLITUDES
    ):
        raise RuntimeError("the perturbation ladder changed")
    for row in records:
        label = row["label_seeded_solve"]
        if label["absolute_residual_delta_from_bank"] > 5.0e-12:
            raise RuntimeError("a label-seeded terminal residual differs from its bank")
        if not label["matches_banked_topology"]:
            raise RuntimeError("a label-seeded topology differs from its bank")
        if len(row["perturbed_seed_ladder"]) != len(PERTURBATION_AMPLITUDES):
            raise RuntimeError("a reference has an incomplete perturbation ladder")
    verdict = receipt["verdict"]
    if verdict["label_holder_count"] != 5 or verdict["label_defectors"] != ["21989/55"]:
        raise RuntimeError(
            "the banked five-holder, one-defector split did not reproduce"
        )
    if verdict["label_criterion_failure_count"] != 6:
        raise RuntimeError(
            "the criterion was unexpectedly satisfied at a labelled state"
        )


def main() -> None:
    """Measure or check the diverted-label basin receipt."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("measure", "check"), nargs="?", default="measure"
    )
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--bank", type=Path, default=DECOMPOSITION_BANK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--carrier", type=Path, default=response_carrier.DEFAULT_CARRIER
    )
    parser.add_argument(
        "--carrier-receipt", type=Path, default=response_carrier.DEFAULT_RECEIPT
    )
    arguments = parser.parse_args()
    if arguments.command == "measure":
        receipt = measure(
            arguments.store,
            arguments.bank,
            arguments.output,
            arguments.carrier,
            arguments.carrier_receipt,
        )
    else:
        receipt = json.loads(arguments.output.read_text())
    check(receipt)
    verdict = receipt["verdict"]
    radii = {
        f"{row['reference']['shot']}/{row['reference']['slice_index']}": row[
            "basin_radius"
        ]["largest_qualifying_relative_amplitude"]
        for row in receipt["per_reference"]
    }
    print(
        "DIVERTED_BASIN_PROBE "
        f"holders={verdict['label_holder_count']}/6 "
        f"defectors={verdict['label_defectors']} "
        f"criterion_failures={verdict['label_criterion_failure_count']}/6 "
        f"radii={radii} verdict={verdict['name']} PASS"
    )


if __name__ == "__main__":
    main()
