"""Re-measure score-blind DIII-D frames with qualified trial admission.

This driver reuses the established DIII-D profile, current-completion, target-
current, and cold-seed machinery without invoking its score gate.  It changes
only nonlinear trial admission: every promoted candidate must remain on the
requested diverted branch and every Krylov action must be qualified.

The two residual bounds reported here have no traced derivation from reference
accuracy or discretisation error.  They are therefore comparisons, not a gate
verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any

import jax.numpy as jnp
import numpy as np

from benchmarks import diiid_forward_gs_match as forward_case
from nova.equilibrium.fixed_point import (
    KrylovActionQualification,
    kink_aware_newton_krylov,
)
from nova.equilibrium.forward import SaddleSeedGeometry
from nova.equilibrium.topology import TopologyClass
from nova.imas.diiid_current import (
    complete_profile_current_adapter,
    shipped_current_at,
)
from nova.imas.diiid_description import POLOIDAL_CONDUCTORS, dataset_machine_description
from nova.jax.config import configure_dtypes


HERE = Path(__file__).resolve().parents[1]
DEFAULT_DATA = forward_case.DEFAULT_DATA
DEFAULT_OUTPUT = (
    HERE / "docs/figures/diiid-forward-onboarding/"
    "repaired-solve-five-frame-remeasure.json"
)
BASELINE_RECEIPT = (
    HERE / "docs/figures/diiid-forward-onboarding/forward-gs/forward_gs_receipt.json"
)
REQUESTED_CLASS = TopologyClass.DIVERTED
NEWTON_STEPS = 89
GMRES_ITERATIONS = 24
NONMONOTONE_FACTORS = (1.0, 0.5, 0.25, 0.125)
SHIPPED_RESIDUAL_BOUND = forward_case.GATE_RESIDUAL_TOLERANCE
REGISTERED_RESIDUAL_BOUND = forward_case.REGISTERED_RESIDUAL_TOLERANCE


@dataclass(frozen=True)
class FrameCase:
    """One score-blind frame fixed by the earlier circuit-current run."""

    shot: str
    frame: int
    previous_relative_residual: float


COHORT = (
    FrameCase("d3d_shot_00000c4a7b.parquet", 179, 0.03951699657560261),
    FrameCase("d3d_shot_0003ff34e7.parquet", 44, 0.08472159556298314),
    FrameCase("d3d_shot_001554e054.parquet", 144, 0.02590839051236691),
    FrameCase("d3d_shot_002495e835.parquet", 146, 0.029674835000125376),
    FrameCase("d3d_shot_0040ca9bdc.parquet", 137, 0.030823235945666667),
)


def _source_commit() -> str:
    """Return the checked-out source identity used for this measurement."""

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
    ).strip()


def _sha256(path: Path) -> str:
    """Return one file's SHA-256 identity."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_case(path: Path) -> dict[str, Any]:
    """Read exactly the columns consumed by the established profile machinery."""

    columns = tuple(
        dict.fromkeys(
            (
                *forward_case._GEOMETRY_COLUMNS,
                *forward_case._LABEL_COLUMNS,
                *forward_case._CURRENT_COLUMNS,
                *forward_case._PLASMA_CURRENT_COLUMNS,
            )
        )
    )
    return forward_case._read(path, columns)


def _validate_baseline() -> None:
    """Require the fixed cohort and comparators to match the banked receipt."""

    receipt = json.loads(BASELINE_RECEIPT.read_text(encoding="utf-8"))
    records = receipt["result"]["frame_records"]
    observed = {
        (record["shot"], int(record["frame"])): float(
            record["fixed_point_relative_residual"]
        )
        for record in records
    }
    expected = {
        (case.shot, case.frame): case.previous_relative_residual for case in COHORT
    }
    if observed != expected:
        raise RuntimeError("the banked score-blind cohort or residuals changed")


def _selected_candidates_were_admitted(
    candidate_admissibility: np.ndarray, accepted_factors: np.ndarray
) -> bool:
    """Verify every promoted state was selected from an admitted trial."""

    factor_to_column = {value: index for index, value in enumerate(NONMONOTONE_FACTORS)}
    for iteration, factor in enumerate(accepted_factors):
        if factor == 0.0:
            continue
        column = factor_to_column.get(float(factor))
        if column is None or not bool(candidate_admissibility[iteration, column]):
            return False
    return True


def _prepare_frame(row: dict[str, Any], frame: int):
    """Build the established 24-current constrained diverted branch input."""

    profile, _label_seed, _label, _wall, _reliable, _statement = (
        forward_case.build_profile(
            row,
            frame,
            forward_case.REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION,
        )
    )
    time_ms = float(row["efit_times"][frame])
    shipped_current = shipped_current_at(
        row,
        dataset_machine_description(
            row, source_row=str(row.get("_source_path", "corpus row"))
        ).physical,
        POLOIDAL_CONDUCTORS,
        time_ms,
    )
    adapter = complete_profile_current_adapter(
        profile,
        shipped_names=POLOIDAL_CONDUCTORS,
        shipped_current_a=shipped_current,
        use_circuit=True,
    )
    profile = adapter.profile
    current = np.asarray(adapter.resolution.current(()), dtype=float)
    if len(current) != 24 or adapter.resolution.unknown_names:
        raise RuntimeError("fixed wiring did not prescribe all 24 conductor currents")
    target_current_a = forward_case._target_current(row, time_ms)
    count = int(row["efit_lcfs_n"][frame])
    contour = np.c_[
        np.asarray(row["efit_lcfs_r"][frame][:count], dtype=float),
        np.asarray(row["efit_lcfs_z"][frame][:count], dtype=float),
    ]
    axis = np.asarray(
        (row["efit_r_axis"][frame], row["efit_z_axis"][frame]), dtype=float
    )
    saddle = contour[int(np.argmin(contour[:, 1]))]
    cold = profile.cold_seed_portfolio(
        target_current_a,
        axis,
        current=jnp.asarray(current),
        diverted_geometry=SaddleSeedGeometry(tuple(axis), tuple(saddle)),
    )
    seed = cold.branches.flux[int(REQUESTED_CLASS)]
    return profile, current, target_current_a, time_ms, seed


def _solve_frame(data: Path, case: FrameCase) -> dict[str, Any]:
    """Run one topology-qualified constrained solve and report its terminal state."""

    started = perf_counter()
    row = _read_case(data / case.shot)
    profile, current, target_current_a, time_ms, seed = _prepare_frame(row, case.frame)
    mapped = profile.flux_map(
        jnp.asarray(current),
        REQUESTED_CLASS,
        target_current_a,
    )

    def remains_diverted(candidate):
        _masks, topology = profile.operator.read(candidate)
        return jnp.all(jnp.isfinite(candidate)) & topology.diverted

    result = kink_aware_newton_krylov(
        mapped,
        seed,
        strategy="nonmonotone",
        newton_steps=NEWTON_STEPS,
        gmres_iterations=GMRES_ITERATIONS,
        warmup=0,
        admissibility_fn=remains_diverted,
    )
    terminal_state = np.asarray(result.state, dtype=float)
    terminal_image = np.asarray(mapped(result.state), dtype=float)
    terminal_relative_residual = float(
        np.max(np.abs(terminal_image - terminal_state))
        / max(np.max(np.abs(terminal_image)), 1.0e-30)
    )
    _masks, topology = profile.operator.read(result.state)
    x_point = np.asarray(topology.x_point, dtype=float)
    accepted_factors = np.asarray(result.accepted_factors, dtype=float)
    candidate_admissibility = np.asarray(result.candidate_admissibility, dtype=bool)
    promoted = accepted_factors > 0.0
    selected_admitted = _selected_candidates_were_admitted(
        candidate_admissibility, accepted_factors
    )
    qualification = KrylovActionQualification(
        int(result.krylov_action_qualification)
    ).name
    return {
        "shot": case.shot,
        "frame": case.frame,
        "time_ms": time_ms,
        "target_current_a": float(target_current_a),
        "target_current_ma": float(abs(target_current_a) / 1.0e6),
        "conductor_count": int(current.size),
        "previous_unqualified_relative_residual": case.previous_relative_residual,
        "terminal_relative_residual": terminal_relative_residual,
        "terminal_topology_class": (
            "diverted" if bool(topology.diverted) else "limited"
        ),
        "finite_terminal_x_point": bool(np.all(np.isfinite(x_point))),
        "terminal_x_point_rz_m": (
            x_point.tolist() if np.all(np.isfinite(x_point)) else None
        ),
        "promoted_iteration_count": int(np.count_nonzero(promoted)),
        "unpromoted_iteration_count": int(np.count_nonzero(~promoted)),
        "all_promoted_iterations_retained_requested_class": bool(
            selected_admitted and bool(topology.diverted)
        ),
        "selected_candidates_were_admitted": selected_admitted,
        "krylov_action_qualification": qualification,
        "meets_shipped_hard_coded_1e_6": bool(
            terminal_relative_residual <= SHIPPED_RESIDUAL_BOUND
        ),
        "meets_benchmark_registered_1e_5": bool(
            terminal_relative_residual <= REGISTERED_RESIDUAL_BOUND
        ),
        "accepted_factor_counts": {
            str(factor): int(np.count_nonzero(accepted_factors == factor))
            for factor in (*NONMONOTONE_FACTORS, 0.0)
        },
        "runtime_seconds": perf_counter() - started,
    }


def run(data: Path, output: Path) -> dict[str, Any]:
    """Run the fixed cohort once and write the no-verdict receipt."""

    configure_dtypes()
    _validate_baseline()
    records = [_solve_frame(data, case) for case in COHORT]
    shipped_count = sum(record["meets_shipped_hard_coded_1e_6"] for record in records)
    registered_count = sum(
        record["meets_benchmark_registered_1e_5"] for record in records
    )
    receipt = {
        "artifact": "repaired_solve_five_frame_remeasure",
        "source_commit": _source_commit(),
        "measurement_scope": {
            "source_split": "development",
            "selection": "the five banked score-blind circuit-current frames",
            "frame_count": len(records),
            "target_current_ma_range": [
                min(record["target_current_ma"] for record in records),
                max(record["target_current_ma"] for record in records),
            ],
            "baseline_receipt": str(BASELINE_RECEIPT.relative_to(HERE)),
            "baseline_receipt_sha256": _sha256(BASELINE_RECEIPT),
            "coefficients_fitted": 0,
            "current_adjustments": 0,
        },
        "solver": {
            "profile_machinery": "benchmarks.diiid_forward_gs_match",
            "current_authority": (
                "24 inference-admissible conductors from shipped_current_at and "
                "fixed circuit wiring"
            ),
            "target_current_seam": "ForwardProfile.flux_map target_current",
            "basin_entry": "ForwardProfile.cold_seed_portfolio diverted seed",
            "route": "topology-qualified nonmonotone Newton-Krylov",
            "requested_topology_class": "diverted",
            "newton_steps": NEWTON_STEPS,
            "gmres_iterations": GMRES_ITERATIONS,
            "warmup": 0,
            "candidate_factors": list(NONMONOTONE_FACTORS),
            "admission": (
                "finite trial, emergent diverted topology, and accepted Krylov action"
            ),
        },
        "residual_bounds": {
            "shipped_hard_coded": SHIPPED_RESIDUAL_BOUND,
            "benchmark_registered": REGISTERED_RESIDUAL_BOUND,
            "shipped_source": (
                "benchmarks.diiid_forward_gs_match.GATE_RESIDUAL_TOLERANCE"
            ),
            "registered_source": (
                "benchmarks.diiid_forward_gs_match.REGISTERED_RESIDUAL_TOLERANCE"
            ),
            "derivation_status": (
                "both bounds are untraced; criterion derivation is in flight on "
                "the operator-refinement lane"
            ),
        },
        "frame_records": records,
        "counts": {
            "frames": len(records),
            "at_or_below_shipped_hard_coded_1e_6": int(shipped_count),
            "above_shipped_hard_coded_1e_6": len(records) - int(shipped_count),
            "at_or_below_benchmark_registered_1e_5": int(registered_count),
            "above_benchmark_registered_1e_5": len(records) - int(registered_count),
            "terminal_diverted": sum(
                record["terminal_topology_class"] == "diverted" for record in records
            ),
            "finite_terminal_x_point": sum(
                record["finite_terminal_x_point"] for record in records
            ),
            "all_promotions_retained_requested_class": sum(
                record["all_promoted_iterations_retained_requested_class"]
                for record in records
            ),
            "krylov_action_accepted": sum(
                record["krylov_action_qualification"] == "ACCEPTED"
                for record in records
            ),
        },
        "interpretation": {
            "gate_verdict": "DECLINED",
            "statement": (
                "No gate verdict is declared because criterion derivation is in "
                "flight on the operator-refinement lane and both residual bounds "
                "reported here are untraced."
            ),
            "shipped_gate_constant_changed": False,
            "score_gate_driver_edited": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    receipt = run(arguments.data, arguments.output)
    print(json.dumps(receipt["counts"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
