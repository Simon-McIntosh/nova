"""Reproduce the two held vertical-centroid solves through constraint data."""

from __future__ import annotations

import argparse
import hashlib
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
RENDER_RECEIPT = (
    ROOT
    / "docs/figures/constraint-augmented-newton-krylov/centroid/render-receipt.json"
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


def _save_figure(figure, output, ink) -> None:
    """Write one figure in both served formats: a vector SVG and a raster PNG."""

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output.with_suffix(".svg"), format="svg", facecolor=ink.figure_facecolor
    )
    figure.savefig(
        output.with_suffix(".png"), dpi=180, facecolor=ink.figure_facecolor
    )


def _render_converged_selection(receipt, output):
    """Draw each row's own coil families, each bar carrying its flags.

    Every panel takes its category axis from its own row's leading circuits, so
    a row whose selection differs never reads under another row's coil names.
    Each bar prints its weight beside the solve's qualified and converged flag,
    and the panel title states the seed-to-converged angle. Each series is
    normalised to its own maximum, which the axis label states.
    """

    from nova.media.ink import DEFAULT_INK, trace_axes

    rows = receipt["rows"]
    figure, axes = plt.subplots(len(rows), 1, figsize=(12.8, 8.0))
    axes = np.atleast_1d(axes)
    records = []
    for axis, row in zip(axes, rows, strict=True):
        trace_axes(axis)
        seed = row["seed_derivation"]["selection"]["leading_circuits"]
        converged = row["converged_derivation"]["selection"]["leading_circuits"]
        families = [item["family"] for item in seed]
        seed_weights = {item["family"]: item["weight"] for item in seed}
        converged_weights = {item["family"]: item["weight"] for item in converged}
        x = np.arange(len(families))
        width = 0.38
        axis.bar(
            x - width / 2,
            [seed_weights.get(family, 0.0) for family in families],
            width,
            color=DEFAULT_INK.flux_color,
            label="derived at seed",
        )
        axis.bar(
            x + width / 2,
            [converged_weights.get(family, 0.0) for family in families],
            width,
            color=DEFAULT_INK.thomson_secondary_color,
            label="derived at convergence",
        )
        axis.axhline(0.0, color="0.4", linewidth=0.8)
        for index, family in enumerate(families):
            axis.annotate(
                f"{seed_weights.get(family, 0.0):+.3f}",
                (x[index] - width / 2, seed_weights.get(family, 0.0)),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=7,
            )
            axis.annotate(
                f"{converged_weights.get(family, 0.0):+.3f}",
                (x[index] + width / 2, converged_weights.get(family, 0.0)),
                textcoords="offset points",
                xytext=(0, -12),
                ha="center",
                fontsize=7,
                color="#5c6b76",
            )
        axis.set_ylabel(
            "direction weight\n(each series normalised to its own maximum)"
        )
        angle = float(row["comparison"]["direction_angle_degrees"])
        axis.set_title(f"{row['identity']}: seed-to-converged angle {angle:.2f}°")
        axis.set_xticks(x, families, rotation=25, ha="right")
        axis.legend(frameon=False)
        records.append(
            {
                "row": row["identity"],
                "angle_degrees": angle,
                "printed_angle": f"{angle:.2f}",
                "category_labels": list(families),
            }
        )
    figure.suptitle(
        "Vertical-centroid compensator stability under the constrained solve",
        y=0.98,
    )
    figure.subplots_adjust(left=0.12, right=0.99, bottom=0.14, top=0.86, hspace=0.60)
    _save_figure(figure, output, DEFAULT_INK)
    plt.close(figure)
    return records


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


def _centroid_bars(receipt: dict) -> list[dict[str, Any]]:
    """Every bar's receipt fields: its row, its arm, its coil family and flags.

    A free or protocol arm carries the solve's ``qualified`` flag and a held
    prototype carries ``converged``, which are the flags the receipt records for
    each arm. The bar's label is the coil family its row's actuator drives,
    taken verbatim from the receipt's actuator definition.
    """

    bars = []
    for row in receipt["rows"]:
        family = row["actuator"]["definition"]
        for arm in ("free", "protocol", "prototype"):
            record = row[arm]
            current = record.get("compensating_current_a")
            bars.append(
                {
                    "row": row["identity"],
                    "arm": arm,
                    "label": family,
                    "qualified": record.get("qualified"),
                    "converged": record.get("converged"),
                    "termination": record.get("termination"),
                    "residual": float(record["terminal_residual"]),
                    "active_set_trips": int(record["active_set_trips"]),
                    "compensating_current_a": (
                        None if current is None else float(current)
                    ),
                }
            )
    return bars


def _bar_flag(record: dict) -> str:
    """The flag a bar carries, or a marker that its arm's receipt carries none."""

    qualified = record.get("qualified")
    converged = record.get("converged")
    if qualified is not None:
        return f"qualified={qualified}"
    if converged is not None:
        return f"converged={converged}"
    return "flag absent"


def _bar_endorsed(record: dict) -> bool:
    """Whether the receipt endorses the arm this bar draws.

    A free or protocol arm is endorsed by its ``qualified`` flag and a held
    prototype by its ``converged`` flag; an arm whose receipt carries neither is
    not endorsed.
    """

    return record.get("qualified") is True or record.get("converged") is True


def _family_label(definition: str, width: int = 30) -> str:
    """The coil family a row's actuator drives, wrapped onto two lines."""

    lines = []
    current = ""
    for word in definition.split():
        candidate = word if not current else current + " " + word
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def _render(receipt, output):
    """Draw the two bank rows with each bar's qualified and converged flags.

    A bar whose arm the receipt does not endorse is hatched, each residual is
    printed in exponent form, and one line under each row names its three arms
    with the flag and the termination the receipt records, so a bar with a small
    residual cannot be read as one the receipt endorses.
    """

    from nova.media.ink import DEFAULT_INK, trace_axes

    rows = receipt["rows"]
    qualified_count = int(receipt["verdict"]["qualified_count"])
    x = np.arange(len(rows))
    width = 0.25
    series = (
        ("free", -width, DEFAULT_INK.flux_color, "free (no pair)"),
        ("protocol", 0.0, DEFAULT_INK.thomson_secondary_color, "protocol"),
        ("prototype", width, DEFAULT_INK.probe_color, "held prototype"),
    )
    panels = (
        ("terminal_residual", 1.0, "terminal residual"),
        ("active_set_trips", 1.0, "active-set trips"),
        ("compensating_current_a", 1.0e3, "P6 compensating current [kA]"),
    )
    plt.rcParams["hatch.linewidth"] = 0.5
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 5.8))
    for axis in axes:
        trace_axes(axis)
    for panel, (field, scale, ylabel) in enumerate(panels):
        axis = axes[panel]
        for index, row in enumerate(rows):
            for arm, offset, color, series_label in series:
                value = row[arm].get(field)
                height = 0.0 if value is None else float(value) / scale
                endorsed = _bar_endorsed(row[arm])
                axis.bar(
                    x[index] + offset,
                    height,
                    width,
                    color=color,
                    edgecolor="none" if endorsed else "0.2",
                    linewidth=0.0 if endorsed else 0.4,
                    hatch="" if endorsed else "///",
                    label=series_label if index == 0 else None,
                )
                if field == "active_set_trips":
                    axis.annotate(
                        str(int(row[arm]["active_set_trips"])),
                        (x[index] + offset, height),
                        textcoords="offset points",
                        xytext=(0, 3),
                        ha="center",
                        fontsize=6,
                    )
        if field == "terminal_residual":
            for index, row in enumerate(rows):
                for arm, offset, _, _ in series:
                    record = row[arm]
                    residual = float(record["terminal_residual"])
                    column = (
                        format(residual, ".4e")
                        + "\n"
                        + _bar_flag(record)
                        + "\n["
                        + str(record.get("termination"))
                        + "]"
                    )
                    axis.annotate(
                        column,
                        (x[index] + offset, residual),
                        textcoords="offset points",
                        xytext=(0, 4),
                        ha="center",
                        va="bottom",
                        fontsize=5.0,
                        rotation=90,
                        linespacing=1.3,
                    )
        axis.set_ylabel(ylabel)
    axes[0].set_yscale("log")
    axes[0].set_ylim(top=1.0)
    trips_max = max(
        int(row[arm]["active_set_trips"]) for row in rows for arm in ("free", "protocol", "prototype")
    )
    axes[1].set_ylim(0.0, float(trips_max) + 2.0)
    axes[2].axhline(0.0, color="0.4", linewidth=0.8)
    for index, row in enumerate(rows):
        for arm, offset, _, _ in series:
            current = row[arm].get("compensating_current_a")
            if current is None:
                continue
            value = float(current) / 1.0e3
            axes[2].annotate(
                format(value, "+.2f"),
                (x[index] + offset, value),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=6,
            )
    ticks = [row["identity"] for row in rows]
    for axis in axes:
        axis.set_xticks(x, ticks)
    axes[0].legend(frameon=False, fontsize=7, loc="upper center")
    for index, row in enumerate(rows):
        axes[0].annotate(
            _family_label(row["actuator"]["definition"]),
            (x[index], 0.0),
            xycoords=("data", "axes fraction"),
            textcoords="offset points",
            xytext=(0, -30),
            ha="center",
            va="top",
            fontsize=6.0,
        )
    header = (
        str(qualified_count)
        + " of "
        + str(len(rows))
        + " rows qualified by the receipt; hatched bars are the arms it does not endorse"
    )
    figure.suptitle(
        "Vertical current centroid through the constraint protocol\n" + header,
        y=0.985,
        fontsize=10,
    )
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.20, top=0.84, wspace=0.28)
    _save_figure(figure, output, DEFAULT_INK)
    plt.close(figure)
    return {
        "row_count": len(rows),
        "qualified_count": qualified_count,
        "bars": _centroid_bars(receipt),
    }

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


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(
    *,
    two_rows: Path = DEFAULT_OUTPUT,
    converged: Path = CONVERGED_SELECTION_OUTPUT,
    output: Path = RENDER_RECEIPT,
) -> dict:
    """Redraw both centroid figures from their committed receipts, no solve.

    Both payloads are read from disk, so the figures regenerate and audit
    without a device and without re-measuring. The render receipt records each
    bar's label, its qualified and converged flags, and the receipt value the
    bar was drawn from, so a figure whose marks drift from its own receipt can
    be caught by reading the two side by side.
    """

    two_payload = json.loads(two_rows.read_text(encoding="utf-8"))
    converged_payload = json.loads(converged.read_text(encoding="utf-8"))
    two_record = _render(two_payload, two_rows.with_suffix(".png"))
    panels = _render_converged_selection(
        converged_payload, converged.with_suffix(".png")
    )
    bars = []
    for row in converged_payload["rows"]:
        for series, key in (
            ("seed", "seed_derivation"),
            ("convergence", "converged_derivation"),
        ):
            selection = row[key]["selection"]["leading_circuits"]
            solve = row[key]["solve"]
            converged_flag = solve.get("termination") == "converged"
            for item in selection:
                bars.append(
                    {
                        "row": row["identity"],
                        "series": series,
                        "label": item["family"],
                        "weight": float(item["weight"]),
                        "qualified": bool(solve["qualified"]),
                        "converged": bool(converged_flag),
                    }
                )
    payload = {
        "schema": "nova.constraint-centroid-render-receipt",
        "completed": True,
        "render_entry_point": "benchmarks/constraint_centroid_receipt.py --render",
        "source_receipts": {
            "two-rows": {
                "name": two_rows.name,
                "sha256": _file_digest(two_rows),
            },
            "converged-compensator": {
                "name": converged.name,
                "sha256": _file_digest(converged),
            },
        },
        "figures": [
            {
                "figure": two_rows.with_suffix(".svg").name,
                "png": two_rows.with_suffix(".png").name,
                "source_receipt": two_rows.name,
                "row_count": two_record["row_count"],
                "qualified_count": two_record["qualified_count"],
                "bars": two_record["bars"],
            },
            {
                "figure": converged.with_suffix(".svg").name,
                "png": converged.with_suffix(".png").name,
                "source_receipt": converged.name,
                "panels": panels,
                "bars": bars,
            },
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operands", type=Path, default=settled.DEFAULT_OPERANDS)
    parser.add_argument("--prototype", type=Path, default=DEFAULT_PROTOTYPE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--figure", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument(
        "--render",
        action="store_true",
        help="redraw both centroid figures from their committed receipts, no solve",
    )
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
        (
            args.selection,
            args.converged_selection,
            args.smoke_converged_selection,
            args.render,
        )
    )
    if modes > 1:
        parser.error("select at most one measurement mode")
    if args.render:
        receipt = render(
            output=RENDER_RECEIPT if args.output is None else args.output
        )
        print(
            "CONSTRAINT-CENTROID-RENDER "
            + json.dumps(
                {
                    "figures": [
                        record["figure"] for record in receipt["figures"]
                    ],
                    "completed": receipt["completed"],
                },
                sort_keys=True,
            )
        )
        return
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
