#!/usr/bin/env python3
"""Receipt the centroid-row fixed-point response on selected MAST rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import jax.numpy as jnp
import numpy as np
import zarr

from benchmarks import compensator_jacobian_response as comparison
from benchmarks.efit_forward_parity_slice import FIXED_POINT_CRITERION
from benchmarks.forward_labeller_throughput import (
    NEWTON_STEPS,
    _centroid_pair,
    _circuit_names,
)
from nova.equilibrium import reduced_newton
from scripts.labeller_batch import shard


ROOT = Path(__file__).resolve().parents[1]
SHOT = 27079
ROWS = (16, 96)
BANKED_RECEIPT = (
    ROOT / "docs/figures/playable-forward-solve/compensator-jacobian/"
    "compensator-jacobian-response.json"
)


def _revision() -> str:
    """Return the source revision used for one receipt."""
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _step_count(result) -> int | None:
    """Return the inner Newton work when a constrained solve returned."""
    return None if result is None else sum(result.newton_steps_per_trip)


def _constrained_steps(profile, state, *, pair, requested, target_current, current):
    """Return the centroid-row solve work for one supplied compensator."""
    try:
        result = reduced_newton.solve_constrained_reduced_newton(
            profile,
            state,
            constraint_pairs=(pair,),
            requested_class=requested,
            target_current=target_current,
            prescribed_current=current,
            tolerance=FIXED_POINT_CRITERION,
            newton_steps=NEWTON_STEPS,
        )
    except Exception as error:
        return {"newton_steps": None, "exception": f"{type(error).__name__}: {error}"}
    return {
        "newton_steps": _step_count(result),
        "converged": bool(result.converged),
        "terminal_residual": float(result.terminal_residual),
        "exception": None,
    }


def _banked_rows() -> dict[tuple[int, str], dict[str, Any]]:
    """Read the converged central pairs that define physical acceptance."""
    payload = json.loads(BANKED_RECEIPT.read_text(encoding="utf-8"))
    return {
        (int(item["row"]), str(item["circuit"])): item
        for item in payload["rows_detail"]
    }


def _physical_comparison(
    row: int, derivative: float, reference: float
) -> dict[str, Any]:
    """Apply the predeclared flat-top and early-frame acceptance bounds."""
    ratio = derivative / reference
    if row == 96:
        relative_error = abs(derivative - reference) / abs(reference)
        passed = relative_error <= 0.2
        criterion = "relative error at most 20 percent"
    else:
        relative_error = None
        passed = 0.5 <= ratio <= 2.0
        criterion = "signed ratio between 0.5 and 2.0"
    return {
        "ratio": ratio,
        "relative_error": relative_error,
        "criterion": criterion,
        "passed": passed,
    }


def _measure_row(
    prepared: shard.PreparedLabeller,
    group: zarr.Group,
    *,
    row: int,
    active_names: dict[int, str],
    banked: dict[tuple[int, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare both requested directions at one converged equilibrium."""
    full_r = np.asarray(group["gridr"], dtype=np.float64)
    full_z = np.asarray(group["gridz"], dtype=np.float64)
    seed = np.asarray(shard._slices_seed(group, row, full_r, full_z), dtype=float)
    inputs = shard._slice_inputs(group, row)
    if inputs is None or not np.all(np.isfinite(seed)):
        raise ValueError(f"row {row} has no finite labeller seed and inputs")
    requested_value = shard._requested_class(group, row)
    requested = jnp.asarray(requested_value, dtype=jnp.int8)
    target_current = abs(float(inputs["reference_plasma_current"]))
    current = np.asarray(inputs["current"], dtype=np.float64)
    free = reduced_newton.solve_reduced_newton(
        prepared.profile.operator,
        jnp.asarray(seed),
        requested_class=requested,
        target_current=target_current,
        prescribed_current=jnp.asarray(current),
        tolerance=FIXED_POINT_CRITERION,
        newton_steps=NEWTON_STEPS,
    )
    if not free.converged:
        raise ValueError(f"row {row} free equilibrium did not converge")
    seeded, direct_selection = _centroid_pair(
        prepared.profile,
        free.state,
        target=float(inputs["target_centroid_z"]),
        unknown=None,
        target_current=target_current,
        requested=requested,
        names=active_names,
    )
    (pair,), selection = reduced_newton.derive_reduced_constraint_pairs(
        prepared.profile,
        (seeded,),
        free.state,
        requested_class=requested,
        target_current=target_current,
        prescribed_current=jnp.asarray(current),
        circuits=sorted(active_names),
        program=free.program,
    )
    response = np.ravel(np.asarray(selection.response, dtype=float))
    direct_response = np.ravel(np.asarray(direct_selection.response, dtype=float))
    step_comparison = (
        {
            "direct_image": _constrained_steps(
                prepared.profile,
                free.state,
                pair=seeded,
                requested=requested,
                target_current=target_current,
                current=current,
            ),
            "linearized": _constrained_steps(
                prepared.profile,
                free.state,
                pair=pair,
                requested=requested,
                target_current=target_current,
                current=current,
            ),
        }
        if row == 96
        else None
    )
    rows = []
    for circuit in ("p6_upper", "p6_upper_minus_p6_lower"):
        direction = comparison._direction(active_names, circuit, current.size)
        reference = banked[(row, circuit)]
        finite_difference = float(reference["finite_difference_m_per_a"])
        derivative = float(response @ direction)
        rows.append(
            {
                "row": row,
                "circuit": circuit,
                "requested_class": ("diverted" if requested_value else "limited"),
                "linearized_jacobian_m_per_a": derivative,
                "direct_image_jacobian_m_per_a": float(direct_response @ direction),
                "banked_direct_image_jacobian_m_per_a": float(
                    reference["constraint_jacobian_m_per_a"]
                ),
                "banked_finite_difference_m_per_a": finite_difference,
                "physical_acceptance": _physical_comparison(
                    row, derivative, finite_difference
                ),
                "constrained_newton_steps": step_comparison,
                "banked_perturbed_solves": reference["perturbed_solves"],
            }
        )
    return rows


def measure(output: Path, report: Path) -> dict[str, Any]:
    """Write the before-and-after response receipt without reading an image."""
    output.mkdir(parents=True, exist_ok=True)
    prepared = shard.prepare_labeller()
    prepared = shard._prepared_with_polarity(prepared, shard._shot_polarity(SHOT))
    group = zarr.open_group(str(shard.SHOT_STORE / f"{SHOT}.zarr"), mode="r")["efm"]
    active_names = _circuit_names(prepared.policy_evidence)
    banked = _banked_rows()
    rows = [
        item
        for row in ROWS
        for item in _measure_row(
            prepared,
            group,
            row=row,
            active_names=active_names,
            banked=banked,
        )
    ]
    payload = {
        "schema": "constraint-fixed-point-response",
        "source_revision": _revision(),
        "shot": SHOT,
        "rows": list(ROWS),
        "derivative_unit": "metres per ampere",
        "banked_receipt": str(BANKED_RECEIPT.relative_to(ROOT)),
        "passed": all(item["physical_acceptance"]["passed"] for item in rows),
        "rows_detail": rows,
    }
    receipt = output / "constraint-jacobian-response.json"
    receipt.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Centroid constraint fixed-point response\n\n"
        "The centroid compensator is selected from the implicit derivative of "
        "the converged reduced fixed point: the reduced residual Jacobian "
        "maps a prescribed-current perturbation into plasma-amplitude motion, "
        "which is then carried through reconstruction and the centroid read. "
        "The receipt records the prior direct-image entry, the linearized "
        "entry, the banked converged central difference, and the constrained "
        "Newton work on the flat-top row. The unknown remains a circuit current; "
        "this is diagnostic placement, not shape control.\n\n"
        f"Receipt: `{receipt}`.\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    """Run the receipt with explicitly chosen destinations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    payload = measure(args.output.resolve(), args.report.resolve())
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
