"""Rescore solved equilibrium arms against an independent analytic field.

The emitted ``arms`` keys retain the source matrix's output-data glossary:
``A`` is exact flux with first-order current support and ``B`` is exact flux
with quadratic current support.  Only arms whose implementation and banked
measurement are present at the measured source revision are emitted.

Every regional pair uses the analytic normalized-flux partition and the full
analytic grid-field span for both references.  This makes the analytic and
banked-root errors directly comparable without changing supports, masks, or
normalisation between the two columns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/coefficient-space-newton/analytic-truth-rescore.json"
)
SOURCE_RECEIPT = ROOT / "docs/figures/coefficient-space-newton/support-order-arms.json"
ROOT_BANK = ROOT / "scripts/oracle_rebaseline"
REGIONS = ("closed_flux_region", "separatrix_band", "scrape_off_layer")
SEPARATRIX_HALF_WIDTH = 0.05


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _region_masks(psi_norm: np.ndarray) -> dict[str, np.ndarray]:
    lower = 1.0 - SEPARATRIX_HALF_WIDTH
    upper = 1.0 + SEPARATRIX_HALF_WIDTH
    return {
        "closed_flux_region": psi_norm < lower,
        "separatrix_band": (psi_norm >= lower) & (psi_norm <= upper),
        "scrape_off_layer": psi_norm > upper,
    }


def _regional_error(
    measured: np.ndarray,
    reference: np.ndarray,
    psi_norm: np.ndarray,
    normalisation: float,
) -> dict[str, dict[str, float | int]]:
    absolute = np.abs(measured - reference)
    result: dict[str, dict[str, float | int]] = {}
    for name, mask in _region_masks(psi_norm).items():
        count = int(np.count_nonzero(mask))
        if count == 0:
            raise RuntimeError(f"the {name} partition has no cells")
        result[name] = {
            "cell_count": count,
            "relative_sup_error": float(np.max(absolute[mask]) / normalisation),
            "relative_rms_error": float(
                np.sqrt(np.mean(absolute[mask] ** 2)) / normalisation
            ),
        }
    return result


def _measure_terminal(carrier: str, label: str) -> tuple[dict[str, Any], np.ndarray]:
    from benchmarks import support_order_arms

    captured: list[np.ndarray] = []
    original = support_order_arms._regional_errors

    def capture(state, reference, psi_norm, cell_count):
        captured.append(np.asarray(state, dtype=np.float64).copy())
        return original(state, reference, psi_norm, cell_count)

    support_order_arms._regional_errors = capture
    try:
        record = support_order_arms._measure(carrier, label)
    finally:
        support_order_arms._regional_errors = original
    if len(captured) != 1:
        raise RuntimeError(
            f"expected one terminal accuracy evaluation, captured {len(captured)}"
        )
    return record, captured[0]


def _with_regional_errors(carrier: str, label: str) -> dict[str, Any]:
    record, terminal = _measure_terminal(carrier, label)
    bank_path = ROOT_BANK / f"root-{carrier}.npz"
    with np.load(bank_path, allow_pickle=False) as bank:
        oracle_psi_norm = np.asarray(bank["oracle_grid_psi_norm"], dtype=np.float64)
        cell_count = len(oracle_psi_norm)
        analytic = np.asarray(bank["oracle_state"][:cell_count], dtype=np.float64)
        banked_root = np.asarray(bank["root_state"][:cell_count], dtype=np.float64)
    measured = terminal[:cell_count]
    normalisation = float(np.ptp(analytic))
    if normalisation <= 0.0:
        raise RuntimeError("the analytic grid field has zero span")
    analytic_errors = _regional_error(
        measured, analytic, oracle_psi_norm, normalisation
    )
    banked_errors = _regional_error(
        measured, banked_root, oracle_psi_norm, normalisation
    )
    return {
        "label": label,
        "carrier": {
            **record["carrier"],
            "analytic_field_array": "oracle_state[:cell_count]",
            "analytic_partition_array": "oracle_grid_psi_norm",
        },
        "common_comparison": {
            "supports": "the carrier plasma-cell centroids",
            "normalisation": "full analytic grid-field flux span",
            "normalisation_wb": normalisation,
            "region_partition": {
                "closed_flux_region": "analytic psi_N < 0.95",
                "separatrix_band": "0.95 <= analytic psi_N <= 1.05",
                "scrape_off_layer": "analytic psi_N > 1.05",
            },
            "regions": {
                region: {
                    "against_analytic_closed_form": analytic_errors[region],
                    "against_banked_converged_root": banked_errors[region],
                }
                for region in REGIONS
            },
        },
        "terminal_reproduction": {
            "terminal_exact_field_relative_residual": record["convergence"][
                "terminal_exact_field_relative_residual"
            ],
            "banked_receipt_regional_error": record[
                "accuracy_against_banked_converged_root"
            ],
        },
    }


def _regional_winners(
    rows: list[dict[str, Any]], carriers: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_carrier: dict[str, Any] = {}
    for carrier in carriers:
        carrier_rows = [row for row in rows if row["carrier"]["name"] == carrier]
        by_carrier[carrier] = {}
        for region in REGIONS:
            metric_records = {}
            for metric in ("relative_sup_error", "relative_rms_error"):
                scores = {
                    row["label"]: row["common_comparison"]["regions"][region][
                        "against_analytic_closed_form"
                    ][metric]
                    for row in carrier_rows
                }
                ordered = sorted(scores.items(), key=lambda item: (item[1], item[0]))
                winner, best = ordered[0]
                runner_up = ordered[1][1] if len(ordered) > 1 else None
                metric_records[metric] = {
                    "closer_arm": winner,
                    "relative_error": best,
                    "margin_to_next_arm": (
                        float(runner_up - best) if runner_up is not None else None
                    ),
                }
            sup_winner = metric_records["relative_sup_error"]["closer_arm"]
            rms_winner = metric_records["relative_rms_error"]["closer_arm"]
            if sup_winner == rms_winner:
                classification = "metrics_agree"
                statement = (
                    f"Arm {sup_winner} is closer to analytic truth by both "
                    f"relative-sup and relative-RMS error in {region} on the "
                    f"{carrier} carrier."
                )
            else:
                classification = "metric_dependent"
                statement = (
                    f"The analytic-truth ranking in {region} on the {carrier} "
                    f"carrier is metric-dependent: arm {sup_winner} is closer by "
                    f"relative-sup error, while arm {rms_winner} is closer by "
                    "relative-RMS error."
                )
            by_carrier[carrier][region] = {
                "classification": classification,
                **metric_records,
                "statement": statement,
            }

    across_carriers: dict[str, Any] = {}
    for region in REGIONS:
        summaries = {}
        for metric in ("relative_sup_error", "relative_rms_error"):
            winners = [
                by_carrier[carrier][region][metric]["closer_arm"]
                for carrier in carriers
            ]
            summaries[metric] = {
                "classification": (
                    "consistent" if len(set(winners)) == 1 else "carrier_dependent"
                ),
                "closer_arms_by_carrier": dict(zip(carriers, winners, strict=True)),
            }
        sup = summaries["relative_sup_error"]["closer_arms_by_carrier"]
        rms = summaries["relative_rms_error"]["closer_arms_by_carrier"]
        across_carriers[region] = {
            **summaries,
            "statement": (
                "Relative-sup closer arms: "
                + ", ".join(f"{carrier}={sup[carrier]}" for carrier in carriers)
                + ". Relative-RMS closer arms: "
                + ", ".join(f"{carrier}={rms[carrier]}" for carrier in carriers)
                + "."
            ),
        }
    return by_carrier, across_carriers


def _validate(receipt: dict[str, Any]) -> None:
    labels = receipt["availability"]["measured_arms"]
    carriers = receipt["comparison_contract"]["carriers"]
    if not labels or not carriers:
        raise RuntimeError("no matrix arms or carriers were measured")
    if "cannot rank arms on accuracy" not in receipt["reference_warning"]:
        raise RuntimeError("the baseline-root authority warning is missing")
    for label in labels:
        rows = receipt["arms"][label]
        if [row["carrier"]["name"] for row in rows] != carriers:
            raise RuntimeError(f"arm {label} does not cover the declared carriers")
        for row in rows:
            regions = row["common_comparison"]["regions"]
            if set(regions) != set(REGIONS):
                raise RuntimeError("a regional comparison is incomplete")
            for region in REGIONS:
                pair = regions[region]
                for reference in (
                    "against_analytic_closed_form",
                    "against_banked_converged_root",
                ):
                    for metric in ("relative_sup_error", "relative_rms_error"):
                        if not np.isfinite(pair[reference][metric]):
                            raise RuntimeError(
                                f"non-finite {label} {region} {reference} {metric}"
                            )
    for carrier in carriers:
        for region in REGIONS:
            if not receipt["closer_to_analytic_truth"]["by_carrier"][carrier][region][
                "statement"
            ]:
                raise RuntimeError("an analytic-truth regional verdict is missing")


def _aggregate(parts: list[Path], output: Path) -> dict[str, Any]:
    source = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in parts]
    labels = list(source["arms"])
    carriers = list(source["comparison_contract"]["common_carrier_set"])
    arms = {
        label: [
            next(
                row
                for row in rows
                if row["label"] == label and row["carrier"]["name"] == carrier
            )
            for carrier in carriers
        ]
        for label in labels
    }
    by_carrier, across_carriers = _regional_winners(rows, carriers)
    receipt = {
        "schema": "nova.analytic-truth-rescore",
        "source_revision": _source_revision(),
        "availability": {
            "measured_arms": labels,
            "basis": (
                "arms present in the integrated support-order matrix receipt at "
                "this source revision"
            ),
            "not_measured": {
                "coefficient_carrier_arms": (
                    "not integrated at this source revision; concurrent in-flight "
                    "work is not evidence input"
                )
            },
        },
        "comparison_contract": {
            "carriers": carriers,
            "analytic_authority": (
                "moderate-rotation-conventional closed-form total flux evaluated "
                "on each carrier's plasma-cell centroids"
            ),
            "common_supports": True,
            "common_region_partition": "analytic normalized flux",
            "common_normalisation": "full analytic grid-field flux span",
            "terminal_state_provenance": (
                "fresh reproduction of each source arm with the source benchmark's "
                "solver, seed, exterior, support and iteration budget"
            ),
        },
        "reference_warning": (
            "The banked converged-root reference is the baseline arm's own fixed "
            "point. It therefore cannot rank arms on accuracy; only the independent "
            "analytic closed form can support that ranking."
        ),
        "evidence_inputs": {
            "matrix_receipt": str(SOURCE_RECEIPT.relative_to(ROOT)),
            "matrix_receipt_sha256": _sha256(SOURCE_RECEIPT),
            "matrix_receipt_source_revision": source["source_revision"],
            "root_banks": {
                carrier: {
                    "path": str((ROOT_BANK / f"root-{carrier}.npz").relative_to(ROOT)),
                    "sha256": _sha256(ROOT_BANK / f"root-{carrier}.npz"),
                }
                for carrier in carriers
            },
        },
        "arms": arms,
        "closer_to_analytic_truth": {
            "ranking_metrics": ["relative_sup_error", "relative_rms_error"],
            "by_carrier": by_carrier,
            "across_carriers": across_carriers,
        },
    }
    _validate(receipt)
    _write_json(output, receipt)
    return receipt


def run(output: Path) -> dict[str, Any]:
    source = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    labels = list(source["arms"])
    carriers = list(source["comparison_contract"]["common_carrier_set"])
    with tempfile.TemporaryDirectory(prefix="nova-analytic-rescore-") as directory:
        work = Path(directory)
        parts: list[Path] = []
        for carrier in carriers:
            for label in labels:
                part = work / f"{carrier}-{label}.json"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "measure",
                    "--carrier",
                    carrier,
                    "--label",
                    label,
                    "--output",
                    str(part),
                ]
                environment = dict(os.environ)
                environment["PYTHONPATH"] = str(ROOT)
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"{carrier} {label} rescore failed:\n{completed.stdout}"
                    )
                parts.append(part)
        return _aggregate(parts, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--carrier", required=True)
    measure_parser.add_argument("--label", required=True)
    measure_parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "measure":
        source = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
        if arguments.label not in source["arms"]:
            raise RuntimeError(f"arm {arguments.label} is not in the source receipt")
        if arguments.carrier not in source["comparison_contract"]["common_carrier_set"]:
            raise RuntimeError(
                f"carrier {arguments.carrier} is not in the source receipt"
            )
        payload = _with_regional_errors(arguments.carrier, arguments.label)
        _write_json(arguments.output, payload)
        print(
            json.dumps(
                {"carrier": arguments.carrier, "label": arguments.label},
                sort_keys=True,
            ),
            flush=True,
        )
        return
    receipt = run(arguments.output)
    if arguments.check:
        _validate(receipt)
    print(json.dumps(receipt["closer_to_analytic_truth"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
