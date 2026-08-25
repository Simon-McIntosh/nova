"""Measure topology-root co-reachability on fixed MAST frames.

The sweep deliberately leaves production selection unchanged.  It applies the
same current-pinned branch maps used by the portfolio, but searches their
basins from deterministic limited- and diverted-neighbourhood seeds.  A root
qualifies only when its direct map residual passes the registered criterion,
its state is finite and nondegenerate, and connectivity realizes the class
requested by that map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.diiid_forward_gs_match import _margin_graded_newton_krylov
from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    FIXED_POINT_CRITERION,
    GMRES_ITERATIONS,
    NEWTON_STEPS,
    _mast_case_from_selection,
    _passive_inclusive_case,
    select_slices_by_shot,
)
from benchmarks.label_seed_residual_field import _persisted_response_cache
from benchmarks.portfolio_warm_start import _problem as diverted_control_problem
from benchmarks.two_branch_batch_cost import (
    _limited_fine_problem,
    _state_qualification,
)
from nova.equilibrium.topology import TopologyClass
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import configure_dtypes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/figures/dual-branch-selection/class-coreachability.json"
FRAME_KEYS = ((21989, 55), (22086, 43))
SEED_AMPLITUDES = (-0.50, -0.25, -0.10, -0.05, 0.0, 0.05, 0.10, 0.25, 0.50, 1.0)
CENSUS_SLICE_COUNT = 1_341_435
PINNED_SOLVE_MILLISECONDS = 42.7719
NONDEGENERATE_CURRENT_FRACTION = 0.01
ROOT_DISTINCT_RELATIVE_TOLERANCE = 1.0e-9
BANKED_DIVERTED_TERMINAL_RESIDUAL = 3.583837965085392e-16


def _digest(values: Any) -> str:
    """Return a stable identity for one binary64 state."""
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _source_revision() -> str:
    """Return the exact source revision executed by the benchmark."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write strict JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _class_name(value: Any) -> str:
    """Render the connectivity class carried by one topology read."""
    return "diverted" if bool(np.asarray(value)) else "limited"


def _connectivity_class(profile: Any, state: Any) -> str:
    """Classify one state solely from its flood-filled connectivity map."""
    _masks, topology = profile.operator.read(jnp.asarray(state))
    return _class_name(topology.diverted)


def _relative_residual(map_fn: Any, state: Any) -> float:
    """Evaluate the production relative-sup fixed-point criterion."""
    values = jnp.asarray(state)
    mapped = map_fn(values)
    return float(
        jnp.max(jnp.abs(mapped - values))
        / jnp.maximum(jnp.max(jnp.abs(mapped)), jnp.asarray(1.0e-30))
    )


def _seed_ladder(profile: Any, label: jax.Array, target_current: float) -> list[dict]:
    """Construct ten seeds in each requested-class neighbourhood."""
    rows = []
    for requested_class in (TopologyClass.LIMITED, TopologyClass.DIVERTED):
        map_fn = profile.flux_map(
            requested_class=requested_class,
            target_current=target_current,
        )
        mapped_label = map_fn(label)
        direction = mapped_label - label
        scale = float(jnp.max(jnp.abs(direction)))
        _masks, topology = profile.operator.read(label, requested_class)
        span = abs(float(topology.flux_span))
        if (
            not np.isfinite(scale)
            or scale <= 0.0
            or not np.isfinite(span)
            or span <= 0.0
        ):
            raise RuntimeError("the label-to-map direction or flux span is degenerate")
        normalized = direction / scale
        for amplitude in SEED_AMPLITUDES:
            seed = label + amplitude * span * normalized
            rows.append(
                {
                    "requested_class": _class_name(int(requested_class)),
                    "requested_class_code": int(requested_class),
                    "relative_amplitude": float(amplitude),
                    "seed": seed,
                    "seed_sha256": _digest(seed),
                    "initial_connectivity_class": _connectivity_class(profile, seed),
                    "initial_relative_residual": _relative_residual(map_fn, seed),
                }
            )
    if len(rows) != 20 or len({row["seed_sha256"] for row in rows}) != 19:
        # The zero-amplitude label is intentionally shared by both class maps.
        raise RuntimeError("the declared twenty-seed, nineteen-state ladder changed")
    return rows


def _solve_seed(
    profile: Any,
    seed: jax.Array,
    requested_class: TopologyClass,
    target_current: float,
) -> tuple[jax.Array, float]:
    """Run the residual-ranked fixed Newton ladder for one pinned map."""
    map_fn = profile.flux_map(
        requested_class=requested_class,
        target_current=target_current,
    )
    result = _margin_graded_newton_krylov(
        map_fn,
        lambda state: jnp.asarray(jnp.inf, dtype=state.dtype),
        seed,
        newton_steps=NEWTON_STEPS,
        gmres_iterations=GMRES_ITERATIONS,
    )
    state = jax.block_until_ready(result.state)
    return state, _relative_residual(map_fn, state)


def _root_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group qualified terminals by class and relative state proximity."""
    groups: list[dict[str, Any]] = []
    for row in rows:
        if not row["qualified_root"]:
            continue
        state = np.asarray(row.pop("_terminal_state"), dtype=np.float64)
        scale = max(float(np.max(np.abs(state))), np.finfo(float).tiny)
        match = None
        for group in groups:
            if group["class"] != row["achieved_class"]:
                continue
            distance = float(np.max(np.abs(state - group["_state"])) / scale)
            if distance <= ROOT_DISTINCT_RELATIVE_TOLERANCE:
                match = group
                break
        if match is None:
            groups.append(
                {
                    "root_index": len(groups),
                    "class": row["achieved_class"],
                    "state_sha256": row["terminal_state_sha256"],
                    "best_terminal_residual": row["terminal_residual"],
                    "seed_indices": [row["seed_index"]],
                    "_state": state,
                }
            )
        else:
            match["seed_indices"].append(row["seed_index"])
            match["best_terminal_residual"] = min(
                match["best_terminal_residual"], row["terminal_residual"]
            )
    for group in groups:
        group.pop("_state")
    for row in rows:
        row.pop("_terminal_state", None)
    return groups


def _measure_frame(
    selected_row: dict[str, Any],
    qualification: dict[str, Any],
    response_cache: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Sweep both seed neighbourhoods on one frozen MAST configuration."""
    case, context = _mast_case_from_selection(SHOT_STORE, selected_row, qualification)
    passive_case, profile, policy = _passive_inclusive_case(
        case, context, response_cache
    )
    reference = passive_case["reference"]
    target_current = abs(float(reference["plasma_current_a"]))
    label = jnp.asarray(passive_case["state"])
    seeds = _seed_ladder(profile, label, target_current)
    rows = []
    for index, seed_row in enumerate(seeds):
        requested = TopologyClass(seed_row["requested_class_code"])
        terminal, residual = _solve_seed(
            profile, seed_row.pop("seed"), requested, target_current
        )
        _masks, topology = profile.operator.read(terminal)
        achieved_class = _class_name(topology.diverted)
        connectivity_class = _connectivity_class(profile, terminal)
        moments = profile.integral_observation(terminal)
        plasma_current = float(moments.plasma_current)
        current_fraction = abs(plasma_current) / target_current
        finite = bool(np.all(np.isfinite(np.asarray(terminal))))
        converged = bool(finite and residual <= FIXED_POINT_CRITERION)
        topology_consistent = achieved_class == seed_row["requested_class"]
        nondegenerate = bool(
            np.isfinite(float(topology.flux_span))
            and abs(float(topology.flux_span)) > 0.0
            and current_fraction >= NONDEGENERATE_CURRENT_FRACTION
        )
        rows.append(
            {
                "seed_index": index,
                **seed_row,
                "achieved_class": achieved_class,
                "connectivity_class": connectivity_class,
                "connectivity_matches_achieved_class": (
                    connectivity_class == achieved_class
                ),
                "converged": converged,
                "finite": finite,
                "nondegenerate": nondegenerate,
                "topology_consistent": topology_consistent,
                "terminal_residual": residual,
                "terminal_plasma_current_a": plasma_current,
                "terminal_plasma_current_fraction_of_target": current_fraction,
                "terminal_flux_span_wb": float(topology.flux_span),
                "terminal_state_sha256": _digest(terminal),
                "qualified_root": bool(
                    converged and nondegenerate and topology_consistent
                ),
                "_terminal_state": terminal,
            }
        )
    roots = _root_groups(rows)
    classes = sorted({root["class"] for root in roots})
    return (
        {
            "reference": {
                "machine": reference["machine"],
                "shot": int(reference["shot"]),
                "slice_index": int(reference["slice_index"]),
                "time_s": float(reference["time_s"]),
                "label_flux_span_wb": float(reference["span_wb"]),
                "target_plasma_current_a": target_current,
            },
            "seed_count": len(rows),
            "initial_connectivity_class_counts": {
                name: sum(row["initial_connectivity_class"] == name for row in rows)
                for name in ("limited", "diverted")
            },
            "converged_seed_count": sum(row["converged"] for row in rows),
            "qualified_seed_count": sum(row["qualified_root"] for row in rows),
            "distinct_qualified_root_count": len(roots),
            "qualified_root_classes": classes,
            "different_class_roots_coexist": classes == ["diverted", "limited"],
            "banked_diverted_terminal_residual_matched_or_improved": bool(
                int(reference["shot"]) == 22086
                and any(
                    row["requested_class"] == "diverted"
                    and row["topology_consistent"]
                    and row["terminal_residual"]
                    <= BANKED_DIVERTED_TERMINAL_RESIDUAL
                    for row in rows
                )
            ),
            "distinct_qualified_roots": roots,
            "seeds": rows,
        },
        int(policy["section_kernel_evaluations_this_shot"]),
    )


def _connectivity_controls() -> list[dict[str, Any]]:
    """Classify independently qualified roots of both classes by connectivity."""
    limited_profile, _limited_cold, limited_root = _limited_fine_problem()
    diverted_profile, _diverted_cold, diverted_root = diverted_control_problem()
    controls = []
    for expected, profile, root in (
        (TopologyClass.LIMITED, limited_profile, limited_root),
        (TopologyClass.DIVERTED, diverted_profile, diverted_root),
    ):
        qualification = _state_qualification(profile, root, expected)
        connectivity_class = _connectivity_class(profile, root)
        expected_class = _class_name(int(expected))
        converged = bool(
            qualification["finite_state"]
            and qualification["relative_map_residual"] <= FIXED_POINT_CRITERION
            and qualification["topology_consistent"]
            and abs(qualification["flux_span_wb"]) > 0.0
            and abs(qualification["plasma_current_a"]) > 0.0
            and qualification["volume_m3"] > 0.0
        )
        controls.append(
            {
                "expected_class": expected_class,
                "connectivity_class": connectivity_class,
                "cross_classified": connectivity_class != expected_class,
                "converged_nondegenerate": converged,
                "relative_map_residual": qualification["relative_map_residual"],
                "plasma_current_a": qualification["plasma_current_a"],
                "flux_span_wb": qualification["flux_span_wb"],
                "volume_m3": qualification["volume_m3"],
                "state_sha256": qualification["state_sha256"],
            }
        )
    if not all(row["converged_nondegenerate"] for row in controls):
        raise RuntimeError("a connectivity classification control is not qualified")
    return controls


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Run the fixed-frame reachability measurement and bank its receipt."""
    configure_dtypes()
    response_cache, carrier_evidence = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER,
        response_carrier.DEFAULT_RECEIPT,
    )
    selected = {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in select_slices_by_shot(DECOMPOSITION_BANK)
    }
    missing = [key for key in FRAME_KEYS if key not in selected]
    if missing:
        raise RuntimeError(f"frozen frame selection is missing {missing}")
    frames = []
    direct_builder_entries = 0
    for key in FRAME_KEYS:
        frame, direct = _measure_frame(*selected[key], response_cache)
        frames.append(frame)
        direct_builder_entries += direct
    if direct_builder_entries != 0:
        raise RuntimeError("persisted-carrier run entered the direct response builder")

    converged_rows = [
        row for frame in frames for row in frame["seeds"] if row["converged"]
    ]
    sweep_connectivity_mismatches = sum(
        not row["connectivity_matches_achieved_class"] for row in converged_rows
    )
    controls = _connectivity_controls()
    control_cross_classifications = sum(row["cross_classified"] for row in controls)
    coexisting = [
        f"{frame['reference']['shot']}/{frame['reference']['slice_index']}"
        for frame in frames
        if frame["different_class_roots_coexist"]
    ]
    one_solve_hours = PINNED_SOLVE_MILLISECONDS * CENSUS_SLICE_COUNT / 1000.0 / 3600.0
    fork_payoffs = sum(len(frame["qualified_root_classes"]) > 1 for frame in frames)
    receipt = {
        "artifact": "fixed-frame topology-class root co-reachability sweep",
        "source_commit": _source_revision(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "measurement_contract": {
            "frames": [f"{shot}/{row}" for shot, row in FRAME_KEYS],
            "seeds_per_frame": 20,
            "seed_neighbourhoods": ["limited", "diverted"],
            "relative_amplitudes_per_neighbourhood": list(SEED_AMPLITUDES),
            "seed_displacement": (
                "label plus declared amplitude times the label flux span along "
                "the normalized label-to-requested-map direction"
            ),
            "solver": (
                "residual-ranked fixed Newton-Krylov ladder on the current-pinned "
                "requested-class map; no class penalty or production selector"
            ),
            "fixed_point_criterion": FIXED_POINT_CRITERION,
            "newton_promotions": NEWTON_STEPS,
            "gmres_iterations_per_promotion": GMRES_ITERATIONS,
            "qualified_root": (
                "finite direct residual at or below criterion, nondegenerate flux "
                "and plasma current, and connectivity class equal to requested map"
            ),
            "root_distinct_relative_sup_tolerance": ROOT_DISTINCT_RELATIVE_TOLERANCE,
        },
        "response_carrier": carrier_evidence,
        "direct_green_operator_builder_entries": direct_builder_entries,
        "frames": frames,
        "connectivity_classification": {
            "method": "production operator read of each state's flood-filled cell map",
            "mast_sweep_converged_state_count": len(converged_rows),
            "mast_sweep_limited_converged_state_count": sum(
                row["achieved_class"] == "limited" for row in converged_rows
            ),
            "mast_sweep_diverted_converged_state_count": sum(
                row["achieved_class"] == "diverted" for row in converged_rows
            ),
            "mast_sweep_cross_classification_count": (sweep_connectivity_mismatches),
            "independent_converged_class_controls": controls,
            "control_class_count": len(controls),
            "control_cross_classification_count": control_cross_classifications,
            "both_classes_tested": {row["expected_class"] for row in controls}
            == {"limited", "diverted"},
            "unambiguous": (
                sweep_connectivity_mismatches == 0
                and control_cross_classifications == 0
            ),
        },
        "fork_value": {
            "frames_with_different_class_qualified_roots": coexisting,
            "frame_count": len(frames),
            "different_basin_payoff_count": fork_payoffs,
            "different_basin_payoff_fraction": fork_payoffs / len(frames),
            "conclusion": (
                "the fork found a second qualified topology basin on at least one "
                "fixed frame"
                if coexisting
                else (
                    "no fixed frame admitted qualified roots of both topology "
                    "classes; the fork is solving for a root not observed at that "
                    "frame, so repeated degrade-path selection is expected"
                )
            ),
        },
        "catalog_cost": {
            "slice_count": CENSUS_SLICE_COUNT,
            "measured_pinned_solve_milliseconds_per_slice": (PINNED_SOLVE_MILLISECONDS),
            "one_solve_plus_connectivity_classification_hours": one_solve_hours,
            "two_independent_solves_hours": 2.0 * one_solve_hours,
            "avoidable_second_solve_hours": one_solve_hours,
            "post_hoc_classification_incremental_solve_cost_assumption": 0.0,
        },
    }
    _write_json(output, receipt)
    return receipt


def main() -> None:
    """Run the sweep from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output)
    print(
        json.dumps(
            {
                "frames": [
                    {
                        "reference": frame["reference"],
                        "distinct_qualified_root_count": frame[
                            "distinct_qualified_root_count"
                        ],
                        "qualified_root_classes": frame["qualified_root_classes"],
                    }
                    for frame in result["frames"]
                ],
                "connectivity_classification": result["connectivity_classification"],
                "fork_value": result["fork_value"],
                "catalog_cost": result["catalog_cost"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
