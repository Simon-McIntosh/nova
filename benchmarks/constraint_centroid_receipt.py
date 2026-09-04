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
    ConstraintPair,
    CurrentCentroidConstraint,
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


def _prototype_rows(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["identity"]: row for row in payload["rows"]}


def _render(receipt, output):
    rows = receipt["rows"]
    labels = [row["identity"] for row in rows]
    x = np.arange(len(rows))
    width = 0.25
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), constrained_layout=True)
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
    axes[1].bar(
        x,
        [row["protocol"]["compensating_current_a"] / 1.0e3 for row in rows],
    )
    axes[1].axhline(0.0, color="0.4", linewidth=0.8)
    axes[1].set_ylabel("P6 compensating current [kA]")
    axes[2].bar(
        x,
        [row["protocol"]["vertical_error_m"] * 1.0e6 for row in rows],
    )
    axes[2].axhline(0.0, color="0.4", linewidth=0.8)
    axes[2].set_ylabel("centroid target error [micrometre]")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Vertical current centroid through the constraint protocol")
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--figure", type=Path, default=DEFAULT_OUTPUT.with_suffix(".png")
    )
    args = parser.parse_args()
    receipt = measure(
        operands=args.operands,
        prototype=args.prototype,
        output=args.output,
        figure=args.figure,
    )
    print(
        "CONSTRAINT-CENTROID-RESULT " + json.dumps(receipt["verdict"], sort_keys=True)
    )


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
