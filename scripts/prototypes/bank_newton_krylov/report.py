"""Verify paired receipts and render one table of the per-lane evidence."""

import argparse
from html import escape
import json
from pathlib import Path
import re

import numpy as np

from scripts.prototypes.bank_solve_cycle.report import pair_gaps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--fragment", type=Path, required=True)
    parser.add_argument("--allocation", required=True)
    args = parser.parse_args()
    baseline = json.loads((args.run.parent / "baseline-suite.json").read_text())
    head_suite = json.loads((args.run / "after-suite.json").read_text())
    assert baseline["completed"] and head_suite["completed"]
    for suite in (baseline, head_suite):
        assert suite["exit_status"] == int(suite["failure_count"] > 0)
    assert not set(head_suite["failure_ids"]) - set(baseline["failure_ids"])
    booking = [
        next(
            row
            for row in suite["file_results"]
            if row["file"] == "tests/test_bank_step_booking.py"
        )
        for suite in (baseline, head_suite)
    ]
    for row in booking:
        log = Path(row["log_path"]).read_text()
        for name in (
            "test_reuses_factor_only_while_residual_contracts",
            "test_reuse_requires_observed_contraction",
        ):
            assert f"::{name} PASSED" in log
    fingerprint = json.loads(
        (args.run.parent / "carried-factor/fingerprint.json").read_text()
    )
    assert len(fingerprint["base"]) == 6
    assert fingerprint["base"] == fingerprint["head"]
    negative = (args.run / "negative.log").read_text()
    physical = (args.run / "physical.log").read_text()
    assert physical.splitlines()[-1] == "EXIT=0"
    assert negative.splitlines()[-1] == "EXIT=1"
    failed = re.findall(r"^FAILED (\S+::\S+)", negative, re.MULTILINE)
    assert len(failed) == 2 and all("physical_update" in name for name in failed)
    errors = re.findall(r"PHYSICAL_UPDATE (\w+) error ([^ ]+)", physical)
    negative_errors = re.findall(r"PHYSICAL_UPDATE (\w+) error ([^ ]+)", negative)
    assert len(errors) == len(negative_errors) == 2
    receipts = {
        arm: json.loads((args.run / f"device-{arm}/receipt.json").read_text())
        for arm in ("base", "head")
    }
    precision = []
    for before, after in zip(
        receipts["base"]["rows"], receipts["head"]["rows"], strict=True
    ):
        assert before["lanes"] == after["lanes"]
        gaps, limits = pair_gaps(before, after)
        for key, values in gaps.items():
            assert np.all(np.isfinite(values)) and max(values) <= limits[key], (
                key,
                values,
            )
        for row in (before, after):
            assert all(row["converged"]) and max(row["residual"]) <= 1e-10
        for state in ("trial", "converged"):
            diagnostics = after[state]
            bound = np.asarray(diagnostics["krylov_condition"]) * np.asarray(
                diagnostics["krylov_residual"]
            )
            assert np.all(np.isfinite(bound)) and np.all(bound <= 1e-8)
        precision.append(dict(lanes=after["lanes"], gaps=gaps, limits=limits))
    (args.run / "precision.json").write_text(json.dumps(precision, indent=2))
    rows = []
    for width_index, width in enumerate((1, 16)):
        for arm in ("base", "head"):
            row = receipts[arm]["rows"][width_index]
            for lane in range(width):
                timing = ""
                if lane == 0:
                    timing = "".join(
                        f'<td rowspan="{width}">{v}</td>'
                        for v in (
                            arm,
                            width,
                            f"{row['linear_device_ms']:.3f}",
                            f"{row['wall_ms']:.3f}",
                            f"{row['per_slice_ms']:.3f}",
                        )
                    )
                linear = (
                    ["—", "—"]
                    if arm == "base"
                    else [
                        f"{row[state]['krylov_condition'][lane]:.3f} / "
                        f"{row[state]['krylov_iterations'][lane]}"
                        for state in ("trial", "converged")
                    ]
                )
                rows.append(
                    "<tr>" + timing + f"<td>{lane}</td><td>{row['steps'][lane]}</td>"
                    f"<td>{row['reads'][lane]}</td><td>{row['residual'][lane]:.3e}</td>"
                    f"<td>{linear[0]}</td><td>{linear[1]}</td></tr>"
                )
    maxima = {
        key: max(max(p["gaps"][key]) for p in precision) for key in precision[0]["gaps"]
    }
    emit = {key: float(value) for key, value in maxima.items()}
    print("PRECISION", json.dumps(emit), flush=True)
    provenance = "; ".join(
        f"{arm} {record['revision']}" for arm, record in receipts.items()
    )
    text = """<section id="bank-newton-krylov-landing">
<h2>Equilibrated block GMRES in the bank Newton step</h2>
<p>The step uses restarted Arnoldi with twice-applied Gram–Schmidt, QR on the
small Hessenberg system, and explicit cell-local 3 × 3 inverses. No dense LU
factor or LU solve remains on the step path. A contracting step can reuse the
matrix, scales and block inverses; every acceptance still uses a fresh read.</p>
<p>The estimated condition number is that of the equilibrated,
block-preconditioned operator the Krylov solve actually iterates on. The solver
stops when that estimate times the true relative residual of that system is at
most 1e-8. This estimate is distinct from the balanced-matrix estimate near 39
in the earlier linear-route measurement. The physical-update test independently
compares the trial and converged 550-cell updates with a float64 reference at
1e-8 relative, and every converged lane must have fresh residual at most 1e-10.
Interpreting the condition estimate in unbalanced coordinates (about 1.8e8)
would demand a residual below machine epsilon; the earlier strict GMRES rows
show that interpretation cannot be met.</p>
<p>The table records one complete paired H200 allocation. Timing is the median
of five synchronized unprofiled calls after compile and warm-up. Linear device
time is attributed inside the whole solve using the existing source-stack
classifier: its two exclusive equilibration/factor and solve/update groups
include the block setup, condition estimate and Krylov loop at head. Kernel
source attribution is not an isolated linear-kernel benchmark. The profiler's
positive control saw the expected 2,097,152-byte host-to-device copy and device
kernels. Each lane is listed, with its trial/converged condition estimate and
Arnoldi count; lane numbers start at zero.</p>
<table><thead><tr><th>Arm</th><th>Slices</th><th>Linear device [ms/solve]</th>
<th>Whole solve [ms]</th><th>Per slice [ms]</th><th>Lane</th><th>Updates</th>
<th>Fresh reads</th><th>Fresh residual</th><th>Trial κ / iterations</th>
<th>Converged κ / iterations</th></tr></thead><tbody>"""
    text += "\n".join(rows) + "</tbody></table>\n"
    base_batch = receipts["base"]["rows"][-1]
    head_batch = receipts["head"]["rows"][-1]
    text += (
        "<p>At sixteen slices, linear device time falls from %.3f to %.3f ms "
        "(%.2f times faster), while whole-solve wall falls from %.3f to %.3f ms "
        "(%.1f percent). The head remains %.3f ms per slice, above the "
        "sub-millisecond target. The base/head update ceilings are %d/%d, "
        "and whole-batch fresh reads are %d/%d. The individual lane "
        "trajectories differ, so the whole-solve change includes that "
        "trajectory difference. Only the paired "
        "measurements in this table are used for the speed ratio.</p>\n"
        % (
            base_batch["linear_device_ms"],
            head_batch["linear_device_ms"],
            base_batch["linear_device_ms"] / head_batch["linear_device_ms"],
            base_batch["wall_ms"],
            head_batch["wall_ms"],
            100 * (1 - head_batch["wall_ms"] / base_batch["wall_ms"]),
            head_batch["per_slice_ms"],
            max(base_batch["steps"]),
            max(head_batch["steps"]),
            max(base_batch["reads"]),
            max(head_batch["reads"]),
        )
    )
    text += (
        "<p>Paired flux/span maximum: %.3e; paired axis/minor-radius: %.3e; "
        "paired X-point/minor-radius: %.3e; paired boundary-flux/span: %.3e. "
        "The limits are 1e-8, 5e-4, 5e-4 and 1e-4, respectively. Across both "
        "arms the independent analytic oracle has axis/minor-radius at most "
        "%.3e, X-point/minor-radius at most %.3e, and boundary-flux/span at most "
        "%.3e. The checks use <code>pair_gaps</code> and retain every lane in "
        "<code>precision.json</code>.</p>\n"
        % (
            maxima["flux"],
            maxima["axis"],
            maxima["saddle"],
            maxima["boundary_flux"],
            max(maxima["base_oracle_axis"], maxima["head_oracle_axis"]),
            max(maxima["base_oracle_saddle"], maxima["head_oracle_saddle"]),
            max(maxima["base_oracle_boundary"], maxima["head_oracle_boundary"]),
        )
    )
    text += (
        "<p>The %d-file Sirius comparison has %d passed, %d skipped and %d "
        "failed at base, and %d passed, %d skipped and %d failed at head, "
        "with no added failure. Baseline's two stale synthetic "
        "booking/step doubles are corrected in the assigned test file. Both "
        "arms retain five skips: two limited-state cases without an X-point, "
        "two optional archived-matrix comparisons and one GPU timing case. "
        "The added <code>test_bank_step_booking.py</code> ran on Sirius in "
        "one fresh pytest process per arm with separate basetemps: base "
        "%d passed/%d failed, head %d passed/%d failed. The real carried-factor "
        "reuse/rebuild and missing-contraction checks pass in both arms. The "
        "shared failure is <code>test_book_primal_keeps_exact_diverted_moments</code>: "
        "base and head produce identical observed hashes, differing from the "
        "stored expectations. No product code or tests changed for this "
        "additional gate. The paired logs and fingerprint comparison are under "
        "<code>carried-factor/</code> in the parent run directory. "
        "The unchanged cycle test accepts every lane below 1e-10. The H200 "
        "physical-update checks pass with trial/converged relative errors "
        "%s. Removing the cell-block preconditioner while capping GMRES at "
        "the preconditioned iteration count fails both physical-update tests "
        "with errors %s; <code>negative.log</code> ends with EXIT=1.</p>\n"
        % (
            baseline["file_count"],
            baseline["passed_count"],
            baseline["skipped_count"],
            baseline["failure_count"],
            head_suite["passed_count"],
            head_suite["skipped_count"],
            head_suite["failure_count"],
            booking[0]["passed_count"],
            booking[0]["failure_count"],
            booking[1]["passed_count"],
            booking[1]["failure_count"],
            escape(str(errors)),
            escape(str(negative_errors)),
        )
    )
    text += (
        "<p>H200 allocation "
        + escape(args.allocation)
        + " on betelgeuse under gpu_0003_grpA: two CPU cores, one GPU, 48 GB. "
        "Revisions: "
        + escape(provenance)
        + ". Receipts, traces and complete logs: <code>"
        + escape(str(args.run))
        + "</code>.</p>\n</section>\n"
    )
    args.fragment.write_text(text)


if __name__ == "__main__":
    main()
