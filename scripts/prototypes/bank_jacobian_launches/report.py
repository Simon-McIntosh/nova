"""Table loop launches, paired process walls, memory and precision"
" receipts."""

from html import escape
import itertools
import json
from pathlib import Path
import re
import sys

import numpy as np

from scripts.prototypes.bank_booking_primal_buckets.report import table
from scripts.prototypes.bank_solve_cycle.report import pair_gaps


def load(path):
    return json.loads(path.read_text())


def main():
    run, output = map(Path, sys.argv[1:3])
    device = run / "device"
    captures = {
        (arm, process): load(device / f"{arm}-{process}/capture.json")
        for arm in ("base", "head")
        for process in range(3)
    }
    rows, loop_rows, precision, trace_rows = [], [], [], []
    metrics = {}
    for lanes in (1, 16):
        for arm in ("base", "head"):
            for process in range(3):
                directory = device / f"{arm}-{process}"
                row = captures[arm, process][str(lanes)]
                assert all(row["converged"]) and max(row["residual"]) <= 1e-10
                wall = float(np.median(row["unprofiled_ms"]))
                memory = load(directory / "memory.json")[str(lanes)]
                assert 0 < memory["peak_bytes_in_use"] < memory["bytes_limit"]
                calls = max(row["steps"])
                if process:
                    attribution = load(directory / "attribution.json")
                    assert attribution["control"]["h2d_bytes"] == 2097152
                    assert attribution["control"]["kernels"] > 0
                    solve_trace = next(
                        x for x in attribution["solves"] if x["lanes"] == lanes
                    )
                    jacobian = solve_trace["groups"]["step_jacobian"]
                    trace_rows.append(
                        [
                            arm,
                            process,
                            lanes,
                            jacobian["launches"],
                            solve_trace["groups"]["step_residual"]["launches"],
                            solve_trace["kernel_launches"],
                        ]
                    )
                    device_ms = jacobian["device_ms"] / calls
                    launches = jacobian["launches"] / calls
                    log = (device / f"capture-{arm}-{process}.log").read_text()
                    assert (
                        log.count("Persistent compilation cache hit for 'jit_solve'")
                        >= 2
                    )
                    loops = load(directory / "loops.json")[str(lanes)]
                    assert (
                        sum(x["launches"] for x in loops["groups"].values())
                        == jacobian["launches"]
                    )
                    for label, counts in loops["groups"].items():
                        loop_rows.append(
                            [
                                arm,
                                process,
                                lanes,
                                label,
                                f"{counts['launches'] / calls:.1f}",
                                f"{counts['device_ms'] / calls:.6f}",
                            ]
                        )
                    metrics[arm, process, lanes] = dict(
                        device_ms=device_ms,
                        launches=launches,
                        wall=wall,
                        wall_per_read=wall / max(row["reads"]),
                    )
                    device_label, launch_label = f"{device_ms:.6f}", f"{launches:.1f}"
                else:
                    device_label = launch_label = "cache warm-up"
                rows.append(
                    [
                        arm,
                        process,
                        lanes,
                        str(row["reads"]),
                        str(row["steps"]),
                        calls,
                        f"{wall:.6f}",
                        f"{wall / max(row['reads']):.6f}",
                        device_label,
                        launch_label,
                        f"{memory['peak_bytes_in_use'] / 2**30:.3f}",
                    ]
                )
        for before, after in itertools.product((1, 2), repeat=2):
            base = captures["base", before][str(lanes)]
            head = captures["head", after][str(lanes)]
            gaps, limits = pair_gaps(base, head)
            assert all(max(values) <= limits[key] for key, values in gaps.items()), gaps
            assert (
                metrics["head", after, lanes]["launches"]
                < metrics["base", before, lanes]["launches"]
            )
            precision.append(
                dict(
                    lanes=lanes,
                    base_process=before,
                    head_process=after,
                    gaps=gaps,
                    limits=limits,
                )
            )
    (run / "precision-pairs.json").write_text(json.dumps(precision, indent=2))
    maxima = {
        key: max(max(row["gaps"][key]) for row in precision)
        for key in precision[0]["gaps"]
    }
    residual = max(
        max(row[str(lanes)]["residual"])
        for row in captures.values()
        for lanes in (1, 16)
    )
    tests = (device / "certificate-tests.log").read_text()
    parity = re.findall(r"JACOBIAN_PARITY ([^ ]+) relative ([0-9.e+-]+)", tests)
    assert len(parity) == 5 and tests.rstrip().endswith("EXIT=0")
    negative = run / "suites/negative-test_bank_step_booking_batch.log"
    negative_text = negative.read_text()
    failures = re.findall(r"^FAILED (\S+)", negative_text, re.M)
    assert failures and re.search(r"EXIT=[1-9][0-9]*\s*$", negative_text)
    suites, suite_rows = {}, []
    for arm in ("base", "head"):
        suites[arm] = load(run / f"suites/{arm}-suite.json")
        assert suites[arm]["completed"]
        for path in suites[arm]["log_paths"]:
            content = Path(path).read_text()
            summary = re.findall(r"^=+ (.* in .*?) =+$", content, re.M)[-1]
            suite_rows.append(
                [
                    arm,
                    Path(path).stem.removeprefix(arm + "-"),
                    summary,
                    content.strip().splitlines()[-1],
                ]
            )
    added = set(suites["head"]["failure_ids"]) - set(suites["base"]["failure_ids"])
    assert not added and set(failures) - set(suites["head"]["failure_ids"])
    base_revision = captures["base", 1]["1"]["revision"]
    head_revision = captures["head", 1]["1"]["revision"]
    text = (
        '<section id="bank-jacobian-launches-landing">\n<h2>Wider tangent '
        "tiles reduce repeated conic propagation</h2>\n"
    )
    text += (
        "<p>The default conic tangent tile is 32 cells, twice the base "
        "width. Tangent ring integration uses groups of two; scalar "
        "booking retains groups of four. The derivative formula and "
        "channel vmap are unchanged. The explicit quadrature width "
        "remains available to bound memory. No product profiling "
        "annotations were added.</p>\n"
    )
    base_wall = np.mean([metrics["base", p, 16]["wall"] for p in (1, 2)])
    head_wall = np.mean([metrics["head", p, 16]["wall"] for p in (1, 2)])
    base_reads = max(captures["base", 1]["16"]["reads"])
    head_reads = max(captures["head", 1]["16"]["reads"])
    text += (
        f"<p>At sixteen slices the measured programs take {base_reads} reads "
        f"at base and {head_reads} at head. Their mean whole-solve walls are "
        f"{base_wall:.3f} and {head_wall:.3f} ms. The head's longer trajectory "
        "raises whole-solve wall despite lower cost per Jacobian call and "
        "per read. These results establish a per-call improvement; they do "
        "not establish a whole-solve latency improvement at sixteen slices.</p>\n"
    )
    text += (
        f"<p>Base <code>{base_revision}</code>; measured head "
        f"<code>{head_revision}</code>. The initial base trace reproduced "
        "2,056 and 2,787 exclusive Jacobian launches per call at one and "
        "sixteen slices, respectively, with 5.134596 and 26.526395 ms of "
        "device work. The launch count differs between these two compiled "
        "batch shapes.</p>\n"
    )
    text += (
        "<p>Conic strip propagation is the dominant launch population: 896"
        " launches per call at one slice and 1,344 at sixteen in the "
        "reproduction. Wider tiles reduce the number of times that fixed "
        "strip loop runs. Ring groups and vectorized channel arithmetic "
        "consume more device time than the propagation kernels; all are "
        "retained in the breakdown below.</p>\n"
    )
    text += (
        "<h3>Paired process measurements</h3><p>Timings share one reserved "
        "H200 allocation, two CPUs and 96 GB host memory. Each revision has its"
        " own compilation cache; process 0 warms it, and processes 1 and 2"
        " are fresh processes with confirmed solve cache hits and "
        "XLA_FLAGS unset. Each wall is the median of five synchronized "
        "unprofiled solves. Per-call device measures divide the exclusive "
        "step_jacobian trace group by the maximum per-lane update count. "
        "Wall/read divides by the maximum lane read count. Memory is the "
        "JAX allocator cumulative process peak after each shape, including"
        " fixture construction and compilation; the sixteen-slice reading "
        "also retains the one-slice program.</p>\n"
    )
    text += table(
        [
            "Arm",
            "Process",
            "Slices",
            "Per-lane reads",
            "Per-lane updates",
            "Jacobian calls",
            "Solve wall [ms]",
            "Wall/read [ms]",
            "Jacobian device/call [ms]",
            "Launches/call",
            "Peak device [GiB]",
        ],
        rows,
    )
    comparison = []
    for lanes in (1, 16):
        for quantity in ("launches", "device_ms", "wall_per_read"):
            before = np.mean([metrics["base", p, lanes][quantity] for p in (1, 2)])
            after = np.mean([metrics["head", p, lanes][quantity] for p in (1, 2)])
            comparison.append(
                [
                    lanes,
                    quantity,
                    f"{before:.6f}",
                    f"{after:.6f}",
                    f"{100 * (1 - after / before):.2f}%",
                ]
            )
    text += table(
        ["Slices", "Measure", "Base mean", "Head mean", "Reduction"], comparison
    )
    text += (
        "<p>The preceding wider-tile-only candidate reduced launch count "
        "but slowed sixteen-slice ring integration. Its first measured "
        "pair took 26.533513 versus 28.825903 Jacobian ms/call and "
        "73.128484 versus 74.265157 ms/read. This motivated groups of two "
        "rings within the wider tangent tile, retaining the base number "
        "of cell-ring pairs per integration group. The preceding complete "
        "receipts remain in the parent run's device and suites directories.</p>\n"
    )
    text += (
        "<h3>Trace-wide accounting</h3><p>Compiler deduplication can bind "
        "a shared conic kernel to the residual source. At one slice the "
        "wider-tile-only trace attributed 1,344 extra launches per solve "
        "to step_residual, so the exclusive Jacobian reduction alone "
        "overstates the launch savings. Whole-trace counts below retain "
        "that cost. Compare totals only beside the read/update counts "
        "above; the trajectory controls how many calls run.</p>\n"
    )
    text += table(
        [
            "Arm",
            "Process",
            "Slices",
            "Jacobian launches/solve",
            "Residual launches/solve",
            "All launches/solve",
        ],
        trace_rows,
    )
    text += (
        "<h3>Exclusive loop attribution</h3><p>Compiler source frames "
        "classify each Jacobian kernel once, from the innermost "
        "propagation loop outward to channel ring tangents, live-ring "
        "integration and cell tiles. A kernel with multiple possible "
        "owners is reported as shared. Runtime copy kernels without their "
        "own source use the enclosing HLO loop. The channel vmap is a "
        "vector dimension, not a sequential loop over seven channels; its "
        "row records the kernels computing the channel-dependent ring "
        "velocities. The residual Jacobian work includes contraction with "
        "the bank and normal-form geometry derivatives. Every measured "
        "trace positive control detected 2,097,152 transfer bytes and a "
        "matrix kernel; every row reconciles exactly with the exclusive "
        "Jacobian group.</p>\n"
    )
    text += table(
        ["Arm", "Process", "Slices", "Loop owner", "Launches/call", "Device/call [ms]"],
        loop_rows,
    )
    text += (
        "<h3>Precision</h3><p>Every lane of every measured base/head "
        "process pairing is checked using pair_gaps. Flux and boundary "
        "differences are normalized by the base flux span; point positions"
        " by its minor radius. The analytic-oracle comparisons are also "
        "retained.</p>\n"
    )
    text += table(
        ["Quantity", "Maximum gap", "Limit"],
        [
            [key, f"{value:.6e}", f"{precision[0]['limits'][key]:.6e}"]
            for key, value in maxima.items()
        ]
        + [["Fresh residual", f"{residual:.6e}", "1e-10"]],
    )
    text += (
        "<p>The four certificate states use the existing 550-cell machine "
        "and certificate state convention: trial moments booked from the "
        "sampled field, converged moments booked from the analytic field. "
        "These labels identify the certificate inputs, not a claim that "
        "the sampled bank equation has converged. Each assembled "
        "fresh-geometry Jacobian is compared against the archived base "
        "channel function at its own width of 16, while all operands are "
        "held identical. The small CPU test independently compares the "
        "batched conic derivatives with the scalar derivative "
        "implementation.</p>\n"
    )
    text += table(
        ["Case", "Relative Jacobian difference", "Limit"],
        [[label, value, "1e-12"] for label, value in parity],
    )
    text += (
        "<h3>CPU suites and negative control</h3><p>The six named files "
        "ran in fresh processes with distinct basetemps on Sirius, one "
        "serial sequence per revision and the two arms concurrent inside "
        "that allocation. Added failures: 0.</p>\n"
    )
    text += table(["Arm", "File", "Pytest summary", "Exit"], suite_rows)
    certificate_revision = tests.splitlines()[0].split(" tree:", 1)[0]
    text += (
        "<p>The certificate-only H200 gate ran separately after the timing "
        "captures: <code>" + escape(certificate_revision) + "</code>. "
        "The earlier timing drivers had already loaded a marker selector "
        "that excluded slow tests; their one-case logs are retained as "
        "filtered receipts. The Jacobian table requires the corrected gate's "
        "five passing comparisons. CPU suite revision: <code>"
        + escape(suites["head"]["revision"])
        + "</code>. Product and test "
        "source are identical across these verification revisions; later "
        "commits only change measurement drivers and this report.</p>\n"
    )
    text += (
        (
            "<p>The scratch mutation sets the first boundary channel "
            "conic-ring velocity to zero immediately before integration. It "
            "fails the small Jacobian comparison: <code>"
        )
        + escape(", ".join(failures))
        + (
            "</code>. The same test passes at the head. Mutation diff and the "
            "nonzero terminal EXIT line are retained beside the suite "
            "logs.</p>\n"
        )
    )
    text += (
        "<p>Receipts: <code>"
        + escape(str(run))
        + (
            "</code>. The device directory carries per-process capture, "
            "memory, compiler provenance, kernel and loop ledgers; "
            "precision-pairs.json carries every lane; suites carries complete "
            "base, head and negative logs. No figure is required for these "
            "tables.</p>\n</section>\n"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    print(
        "REPORT",
        json.dumps(
            dict(
                precision=maxima,
                maximum_residual=residual,
                parity=parity,
                comparison=comparison,
                added_failures=len(added),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
