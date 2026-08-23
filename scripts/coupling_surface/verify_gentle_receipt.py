"""Compare the live gentle window against its banked gating receipt."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from nova.jax.config import configure_dtypes
from scripts.window_demonstration import run_window as demonstration


def _banked_receipt(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream, delimiter="\t")
            if row["regime"] == "gentle" and row["candidate"] == "1"
        ]
    convergence = {
        row["field"]: row["value"] for row in rows if row["kind"] == "convergence"
    }
    trace = tuple(
        float(row["value"])
        for row in rows
        if row["kind"] == "norm_trace" and row["field"] == "gating_norm"
    )
    conservation = {
        row["field"]: row["value"] for row in rows if row["kind"] == "conservation"
    }
    return {
        "iterations": int(convergence["iterations_used"]),
        "contraction": float(convergence["contraction_estimate"]),
        "gating_trace": trace,
        "flux_closure": float(conservation["flux_closure_residual"]),
    }


def _live_receipt() -> dict[str, object]:
    configure_dtypes()
    profile, seed, _vacuum = demonstration._fixture_machine()
    extraction_lattice = demonstration._extraction_lattice(profile)
    fixture_sources = demonstration._fixture_sources(profile)
    baseline_equilibrium = profile.solve(
        seed,
        route="anderson",
        evaluations=demonstration.EVALUATIONS,
    )
    baseline_geometry, baseline_extraction = demonstration._geometry_from_equilibrium(
        baseline_equilibrium,
        profile.source,
        extraction_lattice,
        fixture_sources,
    )
    baseline_extraction.update(iteration=0, sample=0)
    result = demonstration._run_regime(
        demonstration.GENTLE_CANDIDATES[0],
        profile=profile,
        baseline_equilibrium=baseline_equilibrium,
        baseline_geometry=baseline_geometry,
        baseline_extraction=baseline_extraction,
        extraction_lattice=extraction_lattice,
        fixture_sources=fixture_sources,
    )
    if not result.converged:
        raise RuntimeError(
            f"gentle receipt returned {result.outcome_type}: {result.outcome}"
        )
    return {
        "iterations": result.convergence.iterations_used,
        "contraction": result.convergence.contraction_estimate,
        "gating_trace": tuple(result.convergence.gating_norm_trace),
        "flux_closure": result.conservation_receipt.flux_closure_residual,
        "contract_version": result.convergence.contract_version,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    scripts = Path(__file__).resolve().parents[1]
    banked = _banked_receipt(scripts / "window_demonstration" / "receipts.tsv")
    live = _live_receipt()
    comparisons = {
        field: live[field] == banked[field]
        for field in ("iterations", "contraction", "gating_trace", "flux_closure")
    }
    record = {
        "contract_version": live["contract_version"],
        "banked": banked,
        "live": live,
        "bitwise_equal": comparisons,
        "all_equal": all(comparisons.values()),
    }
    arguments.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not record["all_equal"]:
        raise RuntimeError(f"gentle receipt differs from bank: {comparisons}")
    print(f"iterations={live['iterations']}")
    print(f"contraction={live['contraction']:.17g}")
    print(f"gating_trace={live['gating_trace']}")
    print(f"flux_closure={live['flux_closure']:.17g}")
    print(f"contract_version={live['contract_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
