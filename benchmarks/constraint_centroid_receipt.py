"""Reproduce the two held vertical-centroid solves through constraint data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import settled_mask_stall as settled
from nova.equilibrium.constraint import (
    CircuitCurrentUnknown,
    ConstraintBinding,
    ConstraintMultiplier,
    ConstraintPair,
    CurrentCentroidConstraint,
    compensator_rule_name,
    derive_circuit_compensators,
)
from nova.equilibrium.observation import MomentIntegralSupport
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/constraint-augmented-newton-krylov/centroid/two-rows.json"
)
DEFAULT_PROTOTYPE = Path(
    "/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/"
    "s19-relaunch/scr-vertical-position-constraint-prototype/docs/figures/"
    "solver-convergence-regression/vertical-mode/constraint/four-rows.json"
)
SELECTION_OUTPUT = (
    ROOT
    / "docs/figures/constraint-augmented-newton-krylov"
    / "compensator-selection/two-rows.json"
)
CONVERGED_SELECTION_OUTPUT = (
    ROOT
    / "docs/figures/constraint-augmented-newton-krylov"
    / "centroid/converged-compensator.json"
)
ROWS = ((21986, 46), (21989, 55))


def _strict_float(value: Any) -> float | None:
    result = float(np.asarray(value))
    return result if np.isfinite(result) else None


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _centroid(profile, flux, target_current):
    return profile.current_moment_observation(
        flux,
        support=MomentIntegralSupport.ALL_DOMAIN,
        target_current=target_current,
    ).centroid_z


def _p6_pair(policy, profile, *, target, span):
    mapping = {
        str(item["family"]): int(item["stored_circuit"])
        for item in policy["active_mapping"]
    }
    if not {"p6_upper", "p6_lower"}.issubset(mapping):
        raise RuntimeError("active circuit mapping lacks the P6 upper/lower pair")
    prescribed = profile.operator.prescribed_current_field
    if prescribed is None or prescribed.circuit_count != 101:
        raise RuntimeError("the persisted 101-circuit field is unavailable")
    direction = np.zeros(prescribed.circuit_count, dtype=np.float64)
    direction[mapping["p6_upper"] - 1] = 1.0
    direction[mapping["p6_lower"] - 1] = -1.0
    response_span = float(np.ptp(np.asarray(prescribed.response) @ direction))
    current_scale = float(span / response_span)
    position_scale = float(np.ptp(np.asarray(profile.lattice.height)))
    pair = ConstraintPair(
        functional=CurrentCentroidConstraint(
            components=("centroid_z",),
            support=MomentIntegralSupport.ALL_DOMAIN,
        ),
        unknown=CircuitCurrentUnknown(
            direction=direction,
            ampere_scale=np.asarray([current_scale]),
        ),
        binding=ConstraintBinding(
            target=jnp.atleast_1d(target),
            tolerance=jnp.asarray([1.0e-6]),
            scale=jnp.asarray([position_scale]),
            initial_unknown=jnp.asarray([0.0]),
            payload=None,
            policy="imposed",
        ),
    )
    actuator = {
        "definition": "P6 upper current minus P6 lower current",
        "upper_stored_circuit": mapping["p6_upper"],
        "lower_stored_circuit": mapping["p6_lower"],
        "unit_direction_response_span_wb_per_a": response_span,
        "current_scale_a": current_scale,
        "position_scale_m": position_scale,
    }
    return pair, actuator


def _summary(branch, profile, target_current):
    equilibrium = branch.equilibrium
    record = equilibrium.constraints[0] if equilibrium.constraints else None
    centroid = _centroid(profile, equilibrium.flux, target_current)
    return {
        "qualified": bool(np.asarray(branch.converged)),
        "topology_consistent": bool(np.asarray(branch.topology_consistent)),
        "terminal_residual": _strict_float(branch.residual),
        "active_set_trips": int(
            np.asarray(equilibrium.fixed_point.active_set_iterations)
        ),
        "termination": settled._termination_name(
            equilibrium.fixed_point.termination_reason
        ),
        "vertical_centroid_m": _strict_float(centroid),
        "vertical_target_m": (
            None if record is None else _strict_float(record.target[0])
        ),
        "vertical_error_m": (
            None if record is None else _strict_float(record.physical_residual[0])
        ),
        "compensating_current_a": (
            None if record is None else _strict_float(record.physical_unknown[0])
        ),
        "scaled_constraint_residual": (
            None if record is None else _strict_float(record.scaled_residual[0])
        ),
        "soft_mode_projection": (
            None if record is None else _strict_float(record.soft_mode_projection[0])
        ),
    }


def _circuit_names(policy) -> dict[int, str]:
    """Return the zero-based circuit index of every named active family."""
    return {
        int(item["stored_circuit"]) - 1: str(item["family"])
        for item in policy["active_mapping"]
    }


def _pair_projection(delta, actuator) -> float:
    """Return the antisymmetric pair current one full circuit delta carries."""
    upper = int(actuator["upper_stored_circuit"]) - 1
    lower = int(actuator["lower_stored_circuit"]) - 1
    delta = np.asarray(delta)
    return 0.5 * float(delta[upper] - delta[lower])


def _authority_report(selection, names, *, count=8):
    """Rank the circuits by the row scale each moves per ampere."""
    authority = np.asarray(selection.authority)[0]
    drivable = set(int(index) for index in np.asarray(selection.drivable))
    order = np.argsort(np.abs(authority))[::-1][:count]
    return [
        {
            "circuit": int(index),
            "family": names.get(int(index)),
            "drivable": int(index) in drivable,
            "row_scales_per_ampere": float(authority[index]),
            "direction_component": float(np.asarray(selection.directions)[index, 0]),
        }
        for index in order
    ]


def _derived_pair(profile, fixed, seed, *, target_current, requested_class, circuits):
    """Return the same centroid row with a matrix-led compensating direction."""
    (derived,), selection = derive_circuit_compensators(
        profile,
        (fixed,),
        seed,
        requested_class=requested_class,
        target_current=target_current,
        circuits=circuits,
    )
    return derived, selection


def _selection_report(selection, names, *, count=3):
    """Describe the selection rule, spectrum and leading circuit weights."""
    leading = selection.leading_circuits(0, count=count)
    return {
        "rule": selection.rule.name.lower(),
        "competing_rows": bool(selection.competing),
        "singular_values_row_scales_per_ampere": [
            float(value) for value in np.asarray(selection.singular_values)
        ],
        "direction_authority_row_scales_per_ampere": [
            float(value) for value in np.asarray(selection.direction_authority)
        ],
        "leading_circuits": [
            {
                "circuit": int(index),
                "family": names.get(int(index), f"circuit_{int(index)}"),
                "weight": float(np.asarray(selection.directions)[index, 0]),
            }
            for index in leading
        ],
    }


def _current_delta(pair, branch):
    """Return the full circuit-current vector driven by one solved row."""
    record = branch.equilibrium.constraints[0]
    return np.asarray(pair.unknown.direction) @ np.asarray(record.physical_unknown)


def _direction_angle_degrees(first, second) -> float:
    """Return the angle between two single-row circuit directions."""
    first = np.ravel(np.asarray(first, dtype=np.float64))
    second = np.ravel(np.asarray(second, dtype=np.float64))
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0.0:
        raise ValueError("a compensating direction must have non-zero norm")
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _arm(branch, profile, target_current, pair, actuator):
    """Summarise one solved arm together with the circuits it actually drove."""
    summary = _summary(branch, profile, target_current)
    record = branch.equilibrium.constraints[0]
    direction = np.asarray(pair.unknown.direction)
    delta = direction @ np.asarray(record.physical_unknown)
    singular = (
        None
        if record.compensator_singular_values is None
        else [float(value) for value in np.asarray(record.compensator_singular_values)]
    )
    summary.update(
        {
            "compensator_rule": compensator_rule_name(record.compensator_rule),
            "singular_values_row_scales_per_ampere": singular,
            "pair_projected_current_a": _pair_projection(delta, actuator),
            "circuit_current_delta_norm_a": float(np.linalg.norm(delta)),
            "driven_circuits": [
                {"circuit": int(index), "current_a": float(delta[index])}
                for index in np.argsort(np.abs(delta))[::-1][:6]
                if abs(float(delta[index])) > 1.0e-9 * float(np.max(np.abs(delta)))
            ],
        }
    )
    return summary


def _converged_direction_row(
    *,
    profile,
    fixed,
    seed,
    target_current,
    requested_class,
    circuits,
    names,
    actuator,
):
    """Derive at the seed, solve, then re-derive and re-solve at convergence."""
    seed_pair, seed_selection = _derived_pair(
        profile,
        fixed,
        seed,
        target_current=target_current,
        requested_class=requested_class,
        circuits=circuits,
    )
    seed_branch = profile.solve_branch(
        seed,
        requested_class,
        target_current=target_current,
        constraint_pairs=(seed_pair,),
    )
    converged_flux = seed_branch.equilibrium.flux
    converged_flux.block_until_ready()
    converged_pair, converged_selection = _derived_pair(
        profile,
        fixed,
        converged_flux,
        target_current=target_current,
        requested_class=requested_class,
        circuits=circuits,
    )
    converged_branch = profile.solve_branch(
        converged_flux,
        requested_class,
        target_current=target_current,
        constraint_pairs=(converged_pair,),
    )
    converged_branch.equilibrium.flux.block_until_ready()

    seed_delta = _current_delta(seed_pair, seed_branch)
    converged_delta = _current_delta(converged_pair, converged_branch)
    seed_norm = float(np.linalg.norm(seed_delta))
    converged_norm = float(np.linalg.norm(converged_delta))
    current_shift = float(np.linalg.norm(converged_delta - seed_delta))
    return {
        "seed_derivation": {
            "selection": _selection_report(seed_selection, names),
            "solve": _arm(seed_branch, profile, target_current, seed_pair, actuator),
        },
        "converged_derivation": {
            "selection": _selection_report(converged_selection, names),
            "solve": _arm(
                converged_branch,
                profile,
                target_current,
                converged_pair,
                actuator,
            ),
        },
        "comparison": {
            "direction_angle_degrees": _direction_angle_degrees(
                seed_selection.directions[:, 0],
                converged_selection.directions[:, 0],
            ),
            "compensating_current_norm_change_a": converged_norm - seed_norm,
            "compensating_current_vector_change_norm_a": current_shift,
            "compensating_current_vector_relative_change": (
                current_shift / seed_norm if seed_norm else None
            ),
        },
    }


def _render_converged_selection(receipt, output):
    """Draw seed-derived and converged-derived weights for both bank rows."""
    rows = receipt["rows"]
    figure, axes = plt.subplots(len(rows), 1, figsize=(12.5, 7.0), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, row in zip(axes, rows, strict=True):
        seed = row["seed_derivation"]["selection"]["leading_circuits"]
        converged = row["converged_derivation"]["selection"]["leading_circuits"]
        families = list(
            dict.fromkeys(
                [item["family"] for item in seed]
                + [item["family"] for item in converged]
            )
        )
        seed_weights = {item["family"]: item["weight"] for item in seed}
        converged_weights = {item["family"]: item["weight"] for item in converged}
        x = np.arange(len(families))
        width = 0.38
        axis.bar(
            x - width / 2,
            [seed_weights.get(family, 0.0) for family in families],
            width,
            label="derived at seed",
        )
        axis.bar(
            x + width / 2,
            [converged_weights.get(family, 0.0) for family in families],
            width,
            label="derived at convergence",
        )
        axis.axhline(0.0, color="0.4", linewidth=0.8)
        axis.set_ylabel("direction weight")
        axis.set_title(
            f"{row['identity']}: {row['comparison']['direction_angle_degrees']:.4g}°"
        )
        axis.set_xticks(x, families, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    figure.suptitle("Vertical-centroid compensator stability under the solve", y=0.98)
    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.19, top=0.88, hspace=0.42)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def measure_converged_selection(
    *, operands: Path, output: Path, figure: Path, cache_root: Path | None = None
):
    """Measure seed-derived against converged-derived directions on bank rows."""
    configure_dtypes()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
        if cache_root is None
        else cache_root
    )
    response_cache, carrier_evidence = settled._persisted_response_cache(
        settled.response_carrier.DEFAULT_CARRIER,
        settled.response_carrier.DEFAULT_RECEIPT,
    )
    selected = {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in settled.select_slices_by_shot(
            settled.DECOMPOSITION_BANK
        )
    }
    receipt = {
        "receipt": "compensating direction stability under the constrained solve",
        "source": {
            "revision": _source_revision(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
        },
        "configuration": {
            "route": "ForwardProfile.solve_branch public defaults",
            "constraint_policy": "imposed",
            "support": MomentIntegralSupport.ALL_DOMAIN.value,
            "derivations": [
                "seed flux followed by a constrained solve",
                "converged flux followed by a warm-started constrained re-solve",
            ],
            "drivable_circuits": "the machine active mapping",
            "persistent_compilation_cache": {
                "directory": str(cache.directory),
                "version": cache.version_key,
            },
        },
        "inputs": {"operands": str(operands), "carrier_evidence": carrier_evidence},
        "rows": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    for key in ROWS:
        identity = f"{key[0]}/{key[1]}"
        print(f"CONVERGED-COMPENSATOR {identity}", flush=True)
        selected_row, qualification = selected[key]
        case, context = settled._mast_case_from_selection(
            settled.SHOT_STORE, selected_row, qualification
        )
        passive_case, profile, policy = settled._passive_inclusive_case(
            case, context, response_cache
        )
        if int(policy["section_kernel_evaluations_this_shot"]) != 0:
            raise RuntimeError("profile rebuild entered the direct response builder")
        target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
        seed = jnp.asarray(passive_case["state"])
        seed_centroid = _centroid(profile, seed, target_current)
        fixed, actuator = _p6_pair(
            policy,
            profile,
            target=seed_centroid,
            span=float(passive_case["span_wb"]),
        )
        names = _circuit_names(policy)
        row = {
            "identity": identity,
            "seed_vertical_centroid_m": _strict_float(seed_centroid),
            "actuator": actuator,
            **_converged_direction_row(
                profile=profile,
                fixed=fixed,
                seed=seed,
                target_current=target_current,
                requested_class=requested,
                circuits=sorted(names),
                names=names,
                actuator=actuator,
            ),
        }
        receipt["rows"].append(row)
        output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(
            "CONVERGED-COMPENSATOR-DONE " + json.dumps(row, sort_keys=True),
            flush=True,
        )
    receipt["verdict"] = {
        "row_count": len(receipt["rows"]),
        "direction_angle_max_degrees": max(
            row["comparison"]["direction_angle_degrees"] for row in receipt["rows"]
        ),
        "compensating_current_vector_change_norm_max_a": max(
            row["comparison"]["compensating_current_vector_change_norm_a"]
            for row in receipt["rows"]
        ),
        "vertical_error_max_abs_m": max(
            abs(row[derivation]["solve"]["vertical_error_m"])
            for row in receipt["rows"]
            for derivation in ("seed_derivation", "converged_derivation")
        ),
        "rules": sorted(
            {
                row[derivation]["selection"]["rule"]
                for row in receipt["rows"]
                for derivation in ("seed_derivation", "converged_derivation")
            }
        ),
    }
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    _render_converged_selection(receipt, figure)
    return receipt


def smoke_converged_selection():
    """Exercise both derivations on the playable Solov'ev fixture."""
    from apps.playable.solovev import build_machine

    configure_dtypes()
    configure_persistent_compilation_cache(default_persistent_compilation_cache_root())
    machine = build_machine()
    profile = machine.profile
    seed = jnp.asarray(machine.seed)
    target = _centroid(profile, seed, None)
    scale = float(np.ptp(np.asarray(profile.lattice.height)))
    fixed = ConstraintPair(
        functional=CurrentCentroidConstraint(
            components=("centroid_z",),
            support=MomentIntegralSupport.ALL_DOMAIN,
        ),
        unknown=ConstraintMultiplier(multiplier_scale=jnp.asarray([1.0])),
        binding=ConstraintBinding(
            target=jnp.atleast_1d(target),
            tolerance=jnp.asarray([1.0e-6]),
            scale=jnp.asarray([scale]),
            initial_unknown=jnp.asarray([0.0]),
            payload=None,
            policy="imposed",
        ),
    )
    names = {index: f"conductor_{index}" for index in range(machine.circuit_count)}
    actuator = {
        "definition": "first antisymmetric conductor pair",
        "upper_stored_circuit": 1,
        "lower_stored_circuit": 2,
    }
    row = _converged_direction_row(
        profile=profile,
        fixed=fixed,
        seed=seed,
        target_current=None,
        requested_class=jnp.asarray(int(TopologyClass.LIMITED), dtype=jnp.int8),
        circuits=range(machine.circuit_count),
        names=names,
        actuator=actuator,
    )
    result = {
        "fixture": machine.identity,
        "direction_angle_degrees": row["comparison"]["direction_angle_degrees"],
        "seed_vertical_error_m": row["seed_derivation"]["solve"]["vertical_error_m"],
        "converged_vertical_error_m": row["converged_derivation"]["solve"][
            "vertical_error_m"
        ],
        "seed_trips": row["seed_derivation"]["solve"]["active_set_trips"],
        "converged_trips": row["converged_derivation"]["solve"]["active_set_trips"],
    }
    print("CONVERGED-COMPENSATOR-SMOKE " + json.dumps(result, sort_keys=True))
    return result


def _render_selection(receipt, output):
    """Draw the derived and fixed arms side by side on the two bank rows."""
    rows = receipt["rows"]
    labels = [row["identity"] for row in rows]
    x = np.arange(len(rows))
    width = 0.35
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.5))
    for offset, arm, label in (
        (-width / 2, "fixed", "named pair"),
        (width / 2, "derived", "matrix-led"),
    ):
        axes[0].bar(
            x + offset,
            [row[arm]["terminal_residual"] for row in rows],
            width,
            label=label,
        )
        axes[1].bar(
            x + offset,
            [row[arm]["pair_projected_current_a"] / 1.0e3 for row in rows],
            width,
            label=label,
        )
        axes[2].bar(
            x + offset,
            [row[arm]["active_set_trips"] for row in rows],
            width,
            label=label,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("terminal residual")
    axes[1].set_ylabel("pair-projected compensating current [kA]")
    axes[1].axhline(0.0, color="0.4", linewidth=0.8)
    axes[2].set_ylabel("active-set trips")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    figure.suptitle(
        "Compensating direction: named pair against the constraint-response matrix",
        y=0.96,
    )
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.80, wspace=0.30)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def measure_selection(
    *, operands: Path, output: Path, figure: Path, cache_root: Path | None = None
):
    """Compare the named P6 pair with the derived direction on the bank rows."""
    configure_dtypes()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
        if cache_root is None
        else cache_root
    )
    response_cache, carrier_evidence = settled._persisted_response_cache(
        settled.response_carrier.DEFAULT_CARRIER,
        settled.response_carrier.DEFAULT_RECEIPT,
    )
    selected = {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in settled.select_slices_by_shot(
            settled.DECOMPOSITION_BANK
        )
    }
    receipt = {
        "receipt": "compensating circuit direction from the constraint response",
        "source": {
            "revision": _source_revision(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
        },
        "configuration": {
            "route": "ForwardProfile.solve_branch public defaults",
            "constraint_policy": "imposed",
            "support": MomentIntegralSupport.ALL_DOMAIN.value,
            "authority": "row scale moved per ampere; the direction is normalised "
            "so the largest participating circuit carries unity",
            "drivable_circuits": "the machine active mapping; the response "
            "carrier also holds passive structure, which no compensator drives",
            "response_state": "the derivation reads the matrix at the seed flux, "
            "the state the solve starts from",
            "persistent_compilation_cache": {
                "directory": str(cache.directory),
                "version": cache.version_key,
            },
        },
        "inputs": {"operands": str(operands), "carrier_evidence": carrier_evidence},
        "rows": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    for key in ROWS:
        identity = f"{key[0]}/{key[1]}"
        print(f"COMPENSATOR-SELECTION {identity}", flush=True)
        selected_row, qualification = selected[key]
        case, context = settled._mast_case_from_selection(
            settled.SHOT_STORE, selected_row, qualification
        )
        passive_case, profile, policy = settled._passive_inclusive_case(
            case, context, response_cache
        )
        if int(policy["section_kernel_evaluations_this_shot"]) != 0:
            raise RuntimeError("profile rebuild entered the direct response builder")
        target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
        seed = jnp.asarray(passive_case["state"])
        seed_centroid = _centroid(profile, seed, target_current)
        fixed, actuator = _p6_pair(
            policy,
            profile,
            target=seed_centroid,
            span=float(passive_case["span_wb"]),
        )
        names = _circuit_names(policy)
        derived, selection = _derived_pair(
            profile,
            fixed,
            seed,
            target_current=target_current,
            requested_class=requested,
            circuits=sorted(names),
        )
        fixed_branch = profile.solve_branch(
            seed,
            requested,
            target_current=target_current,
            constraint_pairs=(fixed,),
        )
        fixed_branch.equilibrium.flux.block_until_ready()
        derived_branch = profile.solve_branch(
            seed,
            requested,
            target_current=target_current,
            constraint_pairs=(derived,),
        )
        derived_branch.equilibrium.flux.block_until_ready()
        chosen = selection.leading_circuits(0, count=6)
        row = {
            "identity": identity,
            "seed_vertical_centroid_m": _strict_float(seed_centroid),
            "actuator": actuator,
            "selection": {
                "rule": selection.rule.name.lower(),
                "competing_rows": bool(selection.competing),
                "drivable_circuits": [
                    {"circuit": int(index), "family": names.get(int(index))}
                    for index in np.asarray(selection.drivable)
                ],
                "prescribed_circuit_count": int(
                    np.asarray(selection.authority).shape[1]
                ),
                "singular_values_row_scales_per_ampere": [
                    float(value) for value in np.asarray(selection.singular_values)
                ],
                "direction_authority_row_scales_per_ampere": [
                    float(value) for value in np.asarray(selection.direction_authority)
                ],
                "chosen_circuits": [
                    {
                        "circuit": int(index),
                        "family": names.get(int(index)),
                        "component": float(np.asarray(selection.directions)[index, 0]),
                    }
                    for index in chosen
                ],
                "authority_ranking": _authority_report(selection, names),
            },
            "fixed": _arm(fixed_branch, profile, target_current, fixed, actuator),
            "derived": _arm(derived_branch, profile, target_current, derived, actuator),
        }
        receipt["rows"].append(row)
        output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(
            "COMPENSATOR-SELECTION-DONE " + json.dumps(row, sort_keys=True), flush=True
        )
    receipt["verdict"] = {
        "row_count": len(receipt["rows"]),
        "derived_qualified_count": sum(
            row["derived"]["qualified"] for row in receipt["rows"]
        ),
        "pair_projected_current_max_abs_difference_a": max(
            abs(
                row["derived"]["pair_projected_current_a"]
                - row["fixed"]["pair_projected_current_a"]
            )
            for row in receipt["rows"]
        ),
        "vertical_error_max_abs_m": max(
            abs(row["derived"]["vertical_error_m"]) for row in receipt["rows"]
        ),
        "rules": sorted({row["selection"]["rule"] for row in receipt["rows"]}),
    }
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    _render_selection(receipt, figure)
    return receipt


def _prototype_rows(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["identity"]: row for row in payload["rows"]}


def _render(receipt, output):
    rows = receipt["rows"]
    labels = [row["identity"] for row in rows]
    x = np.arange(len(rows))
    width = 0.25
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.5))
    axes[0].bar(
        x - width,
        [row["free"]["terminal_residual"] for row in rows],
        width,
        label="free",
    )
    axes[0].bar(
        x,
        [row["protocol"]["terminal_residual"] for row in rows],
        width,
        label="protocol",
    )
    axes[0].bar(
        x + width,
        [row["prototype"]["terminal_residual"] for row in rows],
        width,
        label="held prototype",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("terminal residual")
    axes[0].legend(frameon=False)
    for offset, branch, label in (
        (-width, "free", "free"),
        (0.0, "protocol", "protocol"),
        (width, "prototype", "held prototype"),
    ):
        axes[1].bar(
            x + offset,
            [row[branch]["active_set_trips"] for row in rows],
            width,
            label=label,
        )
    axes[1].set_ylabel("active-set trips")
    axes[1].legend(frameon=False)
    axes[2].scatter(
        x - width,
        np.zeros(len(rows)),
        marker="x",
        color="C0",
        label="free: no pair",
    )
    axes[2].bar(
        x,
        [row["protocol"]["compensating_current_a"] / 1.0e3 for row in rows],
        width,
        color="C1",
        label="protocol",
    )
    axes[2].bar(
        x + width,
        [row["prototype"]["compensating_current_a"] / 1.0e3 for row in rows],
        width,
        color="C2",
        label="held prototype",
    )
    axes[2].axhline(0.0, color="0.4", linewidth=0.8)
    axes[2].set_ylabel("P6 compensating current [kA]")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Vertical current centroid through the constraint protocol", y=0.96)
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.80, wspace=0.28)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def measure(*, operands: Path, prototype: Path, output: Path, figure: Path):
    configure_dtypes()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    response_cache, carrier_evidence = settled._persisted_response_cache(
        settled.response_carrier.DEFAULT_CARRIER,
        settled.response_carrier.DEFAULT_RECEIPT,
    )
    selected = {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in settled.select_slices_by_shot(
            settled.DECOMPOSITION_BANK
        )
    }
    prototype_rows = _prototype_rows(prototype)
    receipt = {
        "receipt": "vertical current centroid through typed constraint pairs",
        "source": {
            "revision": _source_revision(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
        },
        "configuration": {
            "route": "ForwardProfile.solve_branch public defaults",
            "constraint_policy": "imposed",
            "support": MomentIntegralSupport.ALL_DOMAIN.value,
            "persistent_compilation_cache": {
                "directory": str(cache.directory),
                "version": cache.version_key,
            },
        },
        "inputs": {
            "operands": str(operands),
            "held_prototype": str(prototype),
            "carrier_evidence": carrier_evidence,
        },
        "rows": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    for key in ROWS:
        identity = f"{key[0]}/{key[1]}"
        print(f"CONSTRAINT-CENTROID {identity}", flush=True)
        selected_row, qualification = selected[key]
        case, context = settled._mast_case_from_selection(
            settled.SHOT_STORE, selected_row, qualification
        )
        passive_case, profile, policy = settled._passive_inclusive_case(
            case, context, response_cache
        )
        if int(policy["section_kernel_evaluations_this_shot"]) != 0:
            raise RuntimeError("profile rebuild entered the direct response builder")
        target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
        seed = jnp.asarray(passive_case["state"])
        seed_centroid = _centroid(profile, seed, target_current)
        pair, actuator = _p6_pair(
            policy,
            profile,
            target=seed_centroid,
            span=float(passive_case["span_wb"]),
        )
        free = profile.solve_branch(
            seed,
            jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8),
            target_current=target_current,
        )
        constrained = profile.solve_branch(
            seed,
            jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8),
            target_current=target_current,
            constraint_pairs=(pair,),
        )
        constrained.equilibrium.flux.block_until_ready()
        held = prototype_rows[identity]["constrained"]
        row = {
            "identity": identity,
            "seed_vertical_centroid_m": _strict_float(seed_centroid),
            "actuator": actuator,
            "free": _summary(free, profile, target_current),
            "protocol": _summary(constrained, profile, target_current),
            "prototype": held,
        }
        receipt["rows"].append(row)
        output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print("CONSTRAINT-CENTROID-DONE " + json.dumps(row, sort_keys=True), flush=True)
    receipt["verdict"] = {
        "row_count": len(receipt["rows"]),
        "qualified_count": sum(row["protocol"]["qualified"] for row in receipt["rows"]),
        "prototype_current_max_abs_difference_a": max(
            abs(
                row["protocol"]["compensating_current_a"]
                - row["prototype"]["compensating_current_a"]
            )
            for row in receipt["rows"]
        ),
        "prototype_residual_max_abs_difference": max(
            abs(
                row["protocol"]["terminal_residual"]
                - row["prototype"]["terminal_residual"]
            )
            for row in receipt["rows"]
        ),
    }
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    _render(receipt, figure)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operands", type=Path, default=settled.DEFAULT_OPERANDS)
    parser.add_argument("--prototype", type=Path, default=DEFAULT_PROTOTYPE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--figure", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument(
        "--selection",
        action="store_true",
        help="compare the named pair with the matrix-led compensating direction",
    )
    parser.add_argument(
        "--converged-selection",
        action="store_true",
        help="re-derive the matrix-led direction after the constrained solve",
    )
    parser.add_argument(
        "--smoke-converged-selection",
        action="store_true",
        help="exercise both direction derivations on the Solov'ev fixture",
    )
    args = parser.parse_args()
    modes = sum(
        (args.selection, args.converged_selection, args.smoke_converged_selection)
    )
    if modes > 1:
        parser.error("select at most one measurement mode")
    if args.smoke_converged_selection:
        smoke_converged_selection()
        return
    default = (
        CONVERGED_SELECTION_OUTPUT
        if args.converged_selection
        else SELECTION_OUTPUT
        if args.selection
        else DEFAULT_OUTPUT
    )
    output = default if args.output is None else args.output
    figure = output.with_suffix(".png") if args.figure is None else args.figure
    if args.converged_selection:
        receipt = measure_converged_selection(
            operands=args.operands,
            output=output,
            figure=figure,
            cache_root=args.cache_root,
        )
        print(
            "CONVERGED-COMPENSATOR-RESULT "
            + json.dumps(receipt["verdict"], sort_keys=True)
        )
        return
    if args.selection:
        receipt = measure_selection(
            operands=args.operands,
            output=output,
            figure=figure,
            cache_root=args.cache_root,
        )
        print(
            "COMPENSATOR-SELECTION-RESULT "
            + json.dumps(receipt["verdict"], sort_keys=True)
        )
        return
    receipt = measure(
        operands=args.operands,
        prototype=args.prototype,
        output=output,
        figure=figure,
    )
    print(
        "CONSTRAINT-CENTROID-RESULT " + json.dumps(receipt["verdict"], sort_keys=True)
    )


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
