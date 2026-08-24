"""Attribute analytic-field error to flux normalisation or the discrete map.

The emitted ``arms`` keys retain the source benchmark's output-data glossary:
``A`` is exact flux with first-order current support and ``B`` is exact flux
with quadratic current support.  The comparison first reproduces the banked
raw-field score, then expresses each field in the common physical normalized
flux gauge using anchors read from that same field.  Applying either field's
anchor pair to both raw fields is retained as a cross-check.

No solver or operator implementation is changed by this benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/coefficient-space-newton/analytic-truth-gauge.json"
)
SOURCE_RECEIPT = ROOT / "docs/figures/coefficient-space-newton/support-order-arms.json"
RESCORE_RECEIPT = (
    ROOT / "docs/figures/coefficient-space-newton/analytic-truth-rescore.json"
)
ROOT_BANK = ROOT / "scripts/oracle_rebaseline"
REGIONS = ("closed_flux_region", "separatrix_band", "scrape_off_layer")
SEPARATRIX_HALF_WIDTH = 0.05
GAUGE_ONLY_ROUNDOFF_TOLERANCE = 4096.0 * np.finfo(np.float64).eps


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _measurement_lane() -> dict[str, Any]:
    memory_mb = os.environ.get("SLURM_MEM_PER_NODE")
    return {
        "execution": "slurm" if os.environ.get("SLURM_JOB_ID") else "local",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "hostname": socket.gethostname(),
        "requested_memory_mb": int(memory_mb) if memory_mb else None,
        "cpus_per_task": (
            int(os.environ["SLURM_CPUS_PER_TASK"])
            if os.environ.get("SLURM_CPUS_PER_TASK")
            else None
        ),
        "measurement_children_backend": "cpu",
        "precision": "float64",
        "tmpdir": os.environ.get("TMPDIR"),
    }


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


def _anchors(axis_flux: float, boundary_flux: float) -> dict[str, float]:
    span = boundary_flux - axis_flux
    if not np.isfinite(span) or span == 0.0:
        raise RuntimeError("the flux anchors do not define a finite nonzero span")
    return {
        "axis_flux_wb": float(axis_flux),
        "boundary_flux_wb": float(boundary_flux),
        "boundary_minus_axis_span_wb": float(span),
    }


def _normalise(field: np.ndarray, anchors: dict[str, Any]) -> np.ndarray:
    return (field - float(anchors["axis_flux_wb"])) / float(
        anchors["boundary_minus_axis_span_wb"]
    )


def _region_masks(psi_norm: np.ndarray) -> dict[str, np.ndarray]:
    lower = 1.0 - SEPARATRIX_HALF_WIDTH
    upper = 1.0 + SEPARATRIX_HALF_WIDTH
    return {
        "all_carrier_cells": np.ones_like(psi_norm, dtype=bool),
        "closed_flux_region": psi_norm < lower,
        "separatrix_band": (psi_norm >= lower) & (psi_norm <= upper),
        "scrape_off_layer": psi_norm > upper,
    }


def _regional_metrics(
    discrete: np.ndarray,
    analytic: np.ndarray,
    partition: np.ndarray,
    relative_difference: np.ndarray,
    normalisation_wb: float | None,
) -> dict[str, dict[str, float | int | None]]:
    absolute = np.abs(discrete - analytic)
    relative = np.abs(relative_difference)
    result: dict[str, dict[str, float | int | None]] = {}
    for name, mask in _region_masks(partition).items():
        count = int(np.count_nonzero(mask))
        if count == 0:
            raise RuntimeError(f"the {name} partition has no carrier cells")
        result[name] = {
            "cell_count": count,
            "absolute_sup_error_wb": float(np.max(absolute[mask])),
            "absolute_rms_error_wb": float(np.sqrt(np.mean(absolute[mask] ** 2))),
            "relative_sup_error": float(np.max(relative[mask])),
            "relative_rms_error": float(np.sqrt(np.mean(relative[mask] ** 2))),
            "relative_normalisation_wb": normalisation_wb,
        }
    return result


def _terminal_field(carrier: str, label: str) -> tuple[dict[str, Any], np.ndarray]:
    import jax.numpy as jnp

    from benchmarks import analytic_truth_rescore, support_order_arms

    captured: list[dict[str, Any]] = []
    original = support_order_arms._topology_record

    def capture(operator, state):
        record = original(operator, state)
        _masks, topology = operator.read(jnp.asarray(state))
        captured.append(
            {
                **record,
                "axis_rz_m": np.asarray(topology.axis, dtype=np.float64).tolist(),
                "axis_flux_wb": float(topology.axis_flux),
                "boundary_flux_wb": float(topology.boundary_flux),
                "boundary_minus_axis_span_wb": float(topology.flux_span),
            }
        )
        return record

    support_order_arms._topology_record = capture
    try:
        record, terminal = analytic_truth_rescore._measure_terminal(carrier, label)
    finally:
        support_order_arms._topology_record = original
    if len(captured) != 2:
        raise RuntimeError(
            f"expected reference and terminal topology reads, captured {len(captured)}"
        )
    return {
        "source_measurement": record,
        "reference_topology_read": captured[0],
        "terminal_topology_read": captured[1],
    }, terminal


def _prior_row(receipt: dict[str, Any], carrier: str, label: str) -> dict[str, Any]:
    return next(
        row for row in receipt["arms"][label] if row["carrier"]["name"] == carrier
    )


def _measure(carrier: str, label: str) -> dict[str, Any]:
    import jax

    bank_path = ROOT_BANK / f"root-{carrier}.npz"
    bank_receipt_path = ROOT_BANK / f"receipt-{carrier}.json"
    bank_receipt = json.loads(bank_receipt_path.read_text(encoding="utf-8"))
    prior = json.loads(RESCORE_RECEIPT.read_text(encoding="utf-8"))
    measurement, terminal = _terminal_field(carrier, label)
    with np.load(bank_path, allow_pickle=False) as bank:
        cell_count = len(bank["oracle_grid_psi_norm"])
        analytic = np.asarray(bank["oracle_state"][:cell_count], dtype=np.float64)
        analytic_psi_norm = np.asarray(bank["oracle_grid_psi_norm"], dtype=np.float64)
    discrete = np.asarray(terminal[:cell_count], dtype=np.float64)

    analytic_topology = bank_receipt["oracle_topology"]
    analytic_anchors = _anchors(
        analytic_topology["axis_flux_wb"], analytic_topology["boundary_flux_wb"]
    )
    analytic_anchors.update(
        {
            "axis_rz_m": analytic_topology["axis_m"],
            "provenance": (
                "operator.read of the independently evaluated closed-form field "
                "when the root bank was authored"
            ),
            "field": "analytic_closed_form",
            "receipt_path": str(bank_receipt_path.relative_to(ROOT)),
            "receipt_sha256": _sha256(bank_receipt_path),
        }
    )
    terminal_topology = measurement["terminal_topology_read"]
    discrete_anchors = _anchors(
        terminal_topology["axis_flux_wb"], terminal_topology["boundary_flux_wb"]
    )
    discrete_anchors.update(
        {
            "axis_rz_m": terminal_topology["axis_rz_m"],
            "provenance": ("fresh operator.read of this arm's terminal discrete field"),
            "field": "terminal_discrete_fixed_point",
            "terminal_state_sha256": _array_sha256(discrete),
        }
    )

    analytic_owned = _normalise(analytic, analytic_anchors)
    discrete_owned = _normalise(discrete, discrete_anchors)
    field_owned_difference = discrete_owned - analytic_owned

    analytic_common_discrete = _normalise(discrete, analytic_anchors)
    analytic_common_analytic = _normalise(analytic, analytic_anchors)
    discrete_common_discrete = _normalise(discrete, discrete_anchors)
    discrete_common_analytic = _normalise(analytic, discrete_anchors)

    analytic_span = float(np.ptp(analytic))
    if analytic_span <= 0.0:
        raise RuntimeError("the analytic carrier field has zero full-field span")
    raw_difference = discrete - analytic
    original = _regional_metrics(
        discrete,
        analytic,
        analytic_psi_norm,
        raw_difference / analytic_span,
        analytic_span,
    )
    field_owned = _regional_metrics(
        discrete,
        analytic,
        analytic_psi_norm,
        field_owned_difference,
        None,
    )
    analytic_pair_common = _regional_metrics(
        discrete,
        analytic,
        analytic_owned,
        analytic_common_discrete - analytic_common_analytic,
        abs(float(analytic_anchors["boundary_minus_axis_span_wb"])),
    )
    discrete_pair_common = _regional_metrics(
        discrete,
        analytic,
        discrete_owned,
        discrete_common_discrete - discrete_common_analytic,
        abs(float(discrete_anchors["boundary_minus_axis_span_wb"])),
    )

    previous = _prior_row(prior, carrier, label)["common_comparison"]["regions"]
    reproduction_delta = max(
        abs(
            original[region]["relative_sup_error"]
            - previous[region]["against_analytic_closed_form"]["relative_sup_error"]
        )
        for region in REGIONS
    )
    return {
        "label": label,
        "carrier": {
            "name": carrier,
            "cell_count": cell_count,
            "bank_path": str(bank_path.relative_to(ROOT)),
            "bank_sha256": _sha256(bank_path),
            "terminal_state_sha256": _array_sha256(discrete),
        },
        "anchor_pairs": {
            "analytic_closed_form": analytic_anchors,
            "terminal_discrete_fixed_point": discrete_anchors,
        },
        "raw_flux_gauge_provenance": bank_receipt["gauge_receipt"],
        "banked_rescore_reproduction": {
            "contract": (
                "raw terminal minus analytic field, divided by the full analytic "
                "carrier-field span, partitioned by analytic psi_N"
            ),
            "regions": original,
            "maximum_absolute_delta_from_banked_relative_sup": reproduction_delta,
        },
        "field_owned_common_psi_norm_gauge": {
            "contract": (
                "each field is normalized by axis and boundary anchors read from "
                "that same field; differences are dimensionless psi_N errors and "
                "the regional partition remains the analytic field's psi_N"
            ),
            "regions": field_owned,
        },
        "single_anchor_pair_cross_checks": {
            "analytic_pair_applied_to_both_fields": {
                "contract": (
                    "one analytic-field axis/boundary pair is applied to both raw "
                    "fields; regions are partitioned by the analytic field"
                ),
                "regions": analytic_pair_common,
            },
            "discrete_pair_applied_to_both_fields": {
                "contract": (
                    "one terminal-discrete axis/boundary pair is applied to both "
                    "raw fields; regions are partitioned by the discrete field"
                ),
                "regions": discrete_pair_common,
            },
        },
        "terminal_reproduction": {
            "terminal_exact_field_relative_residual": measurement["source_measurement"][
                "convergence"
            ]["terminal_exact_field_relative_residual"],
            "terminal_topology_class": measurement["terminal_topology_read"]["class"],
            "measurement_backend": jax.default_backend(),
        },
    }


def _range(rows: list[dict[str, Any]], comparison: str, metric: str) -> dict[str, Any]:
    values = [
        {
            "value": row[comparison]["regions"][region][metric],
            "arm": row["label"],
            "carrier": row["carrier"]["name"],
            "region": region,
        }
        for row in rows
        for region in REGIONS
    ]
    return {
        "minimum": min(values, key=lambda item: item["value"]),
        "maximum": max(values, key=lambda item: item["value"]),
    }


def _validate(receipt: dict[str, Any]) -> None:
    rows = [
        row
        for label in receipt["availability"]["measured_arms"]
        for row in receipt["arms"][label]
    ]
    if not rows:
        raise RuntimeError("no gauge-attribution rows were measured")
    for row in rows:
        if set(row["anchor_pairs"]) != {
            "analytic_closed_form",
            "terminal_discrete_fixed_point",
        }:
            raise RuntimeError("a field-owned anchor pair is missing")
        if (
            row["banked_rescore_reproduction"][
                "maximum_absolute_delta_from_banked_relative_sup"
            ]
            > 5.0e-12
        ):
            raise RuntimeError("the fresh run does not reproduce the banked rescore")
        for comparison in (
            "banked_rescore_reproduction",
            "field_owned_common_psi_norm_gauge",
        ):
            regions = row[comparison]["regions"]
            if not {"all_carrier_cells", *REGIONS}.issubset(regions):
                raise RuntimeError(f"{comparison} lacks a regional split")
            for region in ("all_carrier_cells", *REGIONS):
                for metric in (
                    "absolute_sup_error_wb",
                    "absolute_rms_error_wb",
                    "relative_sup_error",
                    "relative_rms_error",
                ):
                    if not np.isfinite(regions[region][metric]):
                        raise RuntimeError(f"non-finite {comparison} {region} {metric}")
    banked = receipt["headline_metrics"]["banked_relative_sup_range"]
    if abs(banked["minimum"]["value"] - 0.2579180329327117) > 5.0e-12:
        raise RuntimeError(
            "the lower banked analytic-error endpoint was not reproduced"
        )
    if abs(banked["maximum"]["value"] - 0.45067410659678714) > 5.0e-12:
        raise RuntimeError(
            "the upper banked analytic-error endpoint was not reproduced"
        )
    surviving = receipt["headline_metrics"][
        "field_owned_common_gauge_relative_sup_range"
    ]["maximum"]["value"]
    expected = (
        "comparison" if surviving <= GAUGE_ONLY_ROUNDOFF_TOLERANCE else "operator"
    )
    if receipt["verdict"]["cause"] != expected:
        raise RuntimeError("the verdict contradicts the gauge-corrected measurement")
    if not receipt["lane_validation"]["passed"]:
        raise RuntimeError(
            "the selected execution lane did not reproduce the banked scores"
        )
    if receipt["measurement_lane"]["measurement_children_backend"] != "cpu":
        raise RuntimeError("the gauge-attribution rows did not share the CPU backend")


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
    ordered_rows = [row for label in labels for row in arms[label]]
    banked_range = _range(
        ordered_rows, "banked_rescore_reproduction", "relative_sup_error"
    )
    common_range = _range(
        ordered_rows, "field_owned_common_psi_norm_gauge", "relative_sup_error"
    )
    absolute_range = _range(
        ordered_rows, "banked_rescore_reproduction", "absolute_sup_error_wb"
    )
    maximum_reproduction_delta = max(
        row["banked_rescore_reproduction"][
            "maximum_absolute_delta_from_banked_relative_sup"
        ]
        for row in ordered_rows
    )
    surviving = common_range["maximum"]["value"]
    cause = "comparison" if surviving <= GAUGE_ONLY_ROUNDOFF_TOLERANCE else "operator"
    if cause == "comparison":
        statement = (
            "The analytic discrepancy collapses to binary64 roundoff when each "
            "field supplies its own normalized-flux anchors; the comparison "
            "normalisation caused the banked error."
        )
    else:
        statement = (
            "The comparison gauge is exonerated.  Field-owned axis/boundary "
            "normalisation leaves a large regional relative-sup error, while the "
            "raw fields already share the exact exterior gauge and retain a large "
            "absolute flux difference.  The discrepancy is therefore a genuine "
            "error in the discrete operator's fixed point, not a normalisation "
            "artefact."
        )
    receipt = {
        "schema": "nova.analytic-truth-gauge-attribution",
        "source_revision": _source_revision(),
        "availability": {
            "measured_arms": labels,
            "carriers": carriers,
            "rows": len(ordered_rows),
        },
        "measurement_lane": _measurement_lane(),
        "lane_validation": {
            "role": (
                "validates that this CPU code-generation lane measures the same "
                "four arm/carrier objects as the banked analytic rescore"
            ),
            "required_rows_reproduced": len(ordered_rows),
            "required_rows": len(labels) * len(carriers),
            "maximum_absolute_delta_from_banked_relative_sup": (
                maximum_reproduction_delta
            ),
            "tolerance": 5.0e-12,
            "passed": maximum_reproduction_delta <= 5.0e-12,
            "banked_relative_sup_range": banked_range,
        },
        "comparison_contract": {
            "common_supports": "each carrier's plasma-cell centroids",
            "regional_partition": {
                "closed_flux_region": "psi_N < 0.95",
                "separatrix_band": "0.95 <= psi_N <= 1.05",
                "scrape_off_layer": "psi_N > 1.05",
            },
            "gauge_only_roundoff_tolerance": GAUGE_ONLY_ROUNDOFF_TOLERANCE,
            "gauge_only_decision_rule": (
                "a pure affine flux-gauge mismatch must collapse to binary64 "
                "roundoff after each field is normalized by its own anchors"
            ),
            "repair_authored": False,
        },
        "evidence_inputs": {
            "support_order_receipt": {
                "path": str(SOURCE_RECEIPT.relative_to(ROOT)),
                "sha256": _sha256(SOURCE_RECEIPT),
                "source_revision": source["source_revision"],
            },
            "analytic_rescore_receipt": {
                "path": str(RESCORE_RECEIPT.relative_to(ROOT)),
                "sha256": _sha256(RESCORE_RECEIPT),
            },
            "root_banks": {
                carrier: {
                    "path": str((ROOT_BANK / f"root-{carrier}.npz").relative_to(ROOT)),
                    "sha256": _sha256(ROOT_BANK / f"root-{carrier}.npz"),
                    "receipt": str(
                        (ROOT_BANK / f"receipt-{carrier}.json").relative_to(ROOT)
                    ),
                    "receipt_sha256": _sha256(ROOT_BANK / f"receipt-{carrier}.json"),
                }
                for carrier in carriers
            },
        },
        "arms": arms,
        "headline_metrics": {
            "banked_relative_sup_range": banked_range,
            "field_owned_common_gauge_relative_sup_range": common_range,
            "raw_absolute_sup_range_wb": absolute_range,
        },
        "verdict": {
            "cause": cause,
            "classification": (
                "comparison_normalisation_artefact"
                if cause == "comparison"
                else "genuine_discrete_operator_error"
            ),
            "statement": statement,
            "surviving_relative_sup_error": common_range,
            "surviving_absolute_sup_error_wb": absolute_range,
            "repair_authored": False,
        },
    }
    _validate(receipt)
    _write_json(output, receipt)
    return receipt


def run(output: Path) -> dict[str, Any]:
    source = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    labels = list(source["arms"])
    carriers = list(source["comparison_contract"]["common_carrier_set"])
    with tempfile.TemporaryDirectory(prefix="nova-analytic-gauge-") as directory:
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
                environment["JAX_PLATFORMS"] = "cpu"
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
                        f"{carrier} {label} gauge measurement failed:\n"
                        f"{completed.stdout}"
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
        payload = _measure(arguments.carrier, arguments.label)
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
    print(json.dumps(receipt["verdict"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
