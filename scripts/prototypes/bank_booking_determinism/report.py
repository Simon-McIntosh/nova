"""Table compiler-environment repeats and their precision against the unset arm."""

import argparse
from collections import Counter
from copy import deepcopy
from html import escape
import json
from pathlib import Path
import re

import numpy as np

from scripts.prototypes.bank_booking_determinism.compare import compare
from scripts.prototypes.bank_booking_determinism.environments import SETTINGS
from scripts.prototypes.bank_solve_cycle.report import pair_gaps


def within_budget(gaps, limits):
    """Require every supplied ratio to be finite and below its own limit."""
    return all(
        np.all(np.isfinite(values)) and np.all(np.asarray(values) <= limits[key])
        for key, values in gaps.items()
    )


def table(identifier, headings, rows):
    return (
        f'<table id="bank-booking-determinism-{identifier}"><thead><tr>'
        + "".join(f"<th>{escape(str(value))}</th>" for value in headings)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            + "".join(f"<td>{escape(str(value))}</td>" for value in row)
            + "</tr>"
            for row in rows
        )
        + "</tbody></table>"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--fragment", type=Path, required=True)
    args = parser.parse_args()
    outcomes = json.loads((args.run / "processes.json").read_text())
    assert len(outcomes) == 15
    loaded = {}
    settings = {}
    for setting in SETTINGS:
        calls = [row for row in outcomes if row["setting"] == setting]
        assert [row["process"] for row in calls] == list(range(5))
        records = []
        for row in calls:
            path = args.run / f"{setting}-{row['process']}.json"
            if row["exit_status"] == 0:
                record = json.loads(path.read_text())
                assert "precision" in record and len(record["calls"]) == 3
                assert record["environment"]["xla_flags"] == SETTINGS[setting]
                assert record["environment"]["cache_root"] == str(
                    args.run / "compilation-cache"
                )
                records.append(record)
        loaded[setting] = records
        comparison = compare(args.run, setting) if records else None
        hashes = sorted({row["initial"]["sha256"] for row in records})
        axes = sorted({row["axis_flux"] for row in records})
        read_patterns = {
            tuple(call["reads"]) for row in records for call in row["calls"]
        }
        complete = len(records) == 5
        settings[setting] = dict(
            complete=complete,
            distinct_hashes=hashes,
            distinct_axis_flux=axes,
            max_moment_difference=comparison["max_initial_difference"]
            if comparison
            else None,
            max_axis_flux_difference=max(axes) - min(axes) if axes else None,
            read_patterns=[list(value) for value in sorted(read_patterns)],
            byte_identical=complete and len(hashes) == 1 and len(axes) == 1,
            identical_reads=complete and len(read_patterns) == 1,
            median_wall_ms=float(np.median([row["median_wall_ms"] for row in records]))
            if records
            else None,
            largest_residual=max(
                (max(call["residual"]) for row in records for call in row["calls"]),
                default=None,
            ),
            comparison=comparison,
        )
    assert len(loaded["unset"]) == 5
    control_reference = loaded["unset"][0]["precision"]
    control_changed = deepcopy(control_reference)
    control_changed["flux"][0][0] += control_reference["span"] * 1e-6
    control_changed["axis"][0][0] += control_reference["minor_radius"] * 1e-3
    control_changed["saddle"][0][0] += control_reference["minor_radius"] * 1e-3
    control_changed["boundary_flux"][0] += control_reference["span"] * 2e-4
    control_gaps, control_limits = pair_gaps(control_reference, control_changed)
    assert not within_budget(control_gaps, control_limits)
    control = {
        key: float(np.max(control_gaps[key]))
        for key in ("flux", "axis", "saddle", "boundary_flux")
    }
    assert all(control[key] > control_limits[key] for key in control)
    baseline_wall = settings["unset"]["median_wall_ms"]
    precision = {}
    for setting, records in loaded.items():
        maxima = {}
        limits = {}
        per_lane = []
        for before_index, before in enumerate(loaded["unset"]):
            for after_index, after in enumerate(records):
                gaps, limits = pair_gaps(before["precision"], after["precision"])
                for key, values in gaps.items():
                    values = np.asarray(values)
                    assert np.all(np.isfinite(values)), (setting, key, values)
                    maxima[key] = max(maxima.get(key, 0), float(np.max(values)))
                per_lane.append(
                    dict(unset_process=before_index, process=after_index, gaps=gaps)
                )
        passed = within_budget(maxima, limits)
        residual_passed = all(
            all(call["converged"]) and max(call["residual"]) <= 1e-10
            for row in records
            for call in row["calls"]
        )
        precision[setting] = dict(
            maxima=maxima,
            limits=limits,
            per_lane=per_lane,
            passed=bool(records) and passed and residual_passed,
        )
        settings[setting]["wall_change_percent"] = (
            100 * (settings[setting]["median_wall_ms"] / baseline_wall - 1)
            if records
            else None
        )
        settings[setting]["precision_passed"] = precision[setting]["passed"]
    result = dict(
        settings=settings,
        precision=precision,
        processes=outcomes,
        precision_control=dict(ratios=control, accepted=False),
    )
    (args.run / "environment-comparison.json").write_text(json.dumps(result, indent=2))
    summary_rows = []
    process_rows = []
    fingerprint_rows = []
    cache_rows = []
    for setting, summary in settings.items():
        summary_rows.append(
            [
                setting,
                SETTINGS[setting] or "unset",
                len(loaded[setting]),
                len(summary["distinct_hashes"]),
                len(summary["distinct_axis_flux"]),
                summary["max_moment_difference"],
                summary["max_axis_flux_difference"],
                summary["byte_identical"],
                summary["identical_reads"],
                summary["median_wall_ms"],
                summary["wall_change_percent"],
                summary["largest_residual"],
            ]
        )
        for index, record in enumerate(loaded[setting]):
            process_rows.append(
                [
                    setting,
                    index,
                    format(record["axis_flux"], ".17g"),
                    ", ".join(map(str, record["calls"][0]["reads"])),
                    f"{record['median_wall_ms']:.6f}",
                    f"{max(max(call['residual']) for call in record['calls']):.9e}",
                ]
            )
            fingerprint_rows.append([setting, index, record["initial"]["sha256"]])
            counts = Counter(row["outcome"] for row in record["cache_lookups"])
            solve = [
                row
                for row in record["solve_cache_lookups"]
                if row["module"] == "jit_solve"
            ]
            assert len(solve) == 1
            cache_rows.append(
                [
                    setting,
                    index,
                    counts["hit"],
                    counts["miss"],
                    solve[0]["outcome"],
                    solve[0]["key"],
                    f"{record['solve_compile_seconds']:.6f}",
                ]
            )
    good = [
        name
        for name, row in settings.items()
        if row["byte_identical"] and row["identical_reads"] and row["precision_passed"]
    ]
    recommendation = (
        "Settings meeting byte identity, identical reads and the precision budget: "
        + ", ".join(good)
        if good
        else "No setting establishes byte identity with identical reads "
        "and the precision budget."
    )
    text = '<section id="bank-booking-determinism-environments">\n'
    text += "<h2>Compiler environment comparison in one H200 allocation</h2>\n"
    text += "<p>" + escape(recommendation) + "</p>\n"
    text += "<p>Product source and tests are unchanged at base "
    text += "<code>cb42f656f76a7d7e56da8a53d4754c22d131bdc0</code>. "
    text += "Each setting runs five fresh processes, each with one warm-up and three "
    text += (
        "timed converged solves. A process wall is the median of its three repeats; "
    )
    text += "a setting wall is the median of its five process medians. Costs below are "
    text += "relative to the unset setting within this allocation.</p>\n"
    text += table(
        "environment-summary",
        [
            "Setting",
            "Effective XLA_FLAGS",
            "Complete processes",
            "Distinct moment hashes",
            "Distinct axis flux values",
            "Largest moment difference",
            "Largest axis flux difference",
            "Byte-identical",
            "Identical reads",
            "Median solve [ms / 16 slices]",
            "Wall change [%]",
            "Largest fresh residual",
        ],
        summary_rows,
    )
    text += table(
        "environment-processes",
        [
            "Setting",
            "Process",
            "read.axis_flux",
            "Reads, lanes 0–15",
            "Median solve [ms / 16 slices]",
            "Largest fresh residual",
        ],
        process_rows,
    )
    text += table(
        "environment-fingerprints",
        ["Setting", "Process", "Initial-moment SHA-256"],
        fingerprint_rows,
    )
    text += "<p>All fifteen processes use the same persistent cache root: <code>"
    text += escape(str(args.run / "compilation-cache")) + "</code>. "
    text += "Caching is enabled with the normal one-second minimum compilation time "
    text += "and zero-byte default size threshold. Small programs can miss in every "
    text += "process because they do not qualify for persistence. Counts below come "
    text += "from actual JAX compiler cache lookup events, not from elapsed time. "
    text += (
        "The first miss and later hits are the cache instrument’s positive control. "
    )
    text += "Flags are set before each Python process starts; "
    text += "no lane launcher was edited.</p>\n"
    text += table(
        "environment-cache",
        [
            "Setting",
            "Process",
            "Cache hits",
            "Cache misses",
            "Solve cache",
            "Solve cache key",
            "Solve compile/load [s]",
        ],
        cache_rows,
    )
    precision_rows = []
    for setting, row in precision.items():
        for key, value in row["maxima"].items():
            precision_rows.append(
                [
                    setting,
                    key,
                    f"{value:.9e}",
                    row["limits"][key],
                    value <= row["limits"][key],
                ]
            )
    text += "<p>The existing pair_gaps function compares every completed process "
    text += "against every unset process, retaining all sixteen lane values in "
    text += "environment-comparison.json. The table reports the maximum over those "
    text += "pairs and lanes; the fresh residual limit remains 1e-10.</p>\n"
    text += table(
        "environment-precision",
        ["Setting", "Quantity / reference scale", "Largest ratio", "Budget", "Pass"],
        precision_rows,
    )
    text += "<p>A controlled copy exceeds each paired precision budget: "
    text += escape(json.dumps(control))
    text += ". The same budget predicate rejects it. No product or test source "
    text += "is mutated; this checks the comparison instrument.</p>\n"
    refused = [row for row in outcomes if row["exit_status"]]
    if refused:
        text += table(
            "environment-refusals",
            ["Setting", "Process", "Exit", "Log"],
            [
                [r["setting"], r["process"], r["exit_status"], r["log_path"]]
                for r in refused
            ],
        )
    suite = json.loads((args.run / "base-suite.json").read_text())
    assert suite["completed"]
    suite_rows = []
    for row in suite["files"]:
        log = Path(row["log_path"]).read_text()
        summary = re.findall(r"^=+ (.* in .*?) =+$", log, re.MULTILINE)[-1]
        suite_rows.append([Path(row["log_path"]).name, summary, row["exit_status"]])
    text += "<p>Head is base for the CPU suite because neither product source nor "
    text += "tests changed. Four completed baseline files are reused, and the "
    text += "interrupted driver file is completed in a fresh sirius process. "
    text += "The known exact-moment byte pin remains unchanged.</p>\n"
    text += table(
        "environment-suite", ["File log", "Complete result", "Exit"], suite_rows
    )
    text += "<p>Raw process logs, arrays, cache keys and per-lane "
    text += "precision receipts: <code>"
    text += escape(str(args.run)) + "</code>.</p>\n</section>\n"
    prior = args.fragment.read_text()
    marker = '<section id="bank-booking-determinism-landing">'
    prior = prior[prior.index(marker) :]
    args.fragment.write_text(text + "\n" + prior)
    print(
        "SUMMARY",
        json.dumps(
            {
                key: {k: v for k, v in row.items() if k != "comparison"}
                for key, row in settings.items()
            }
        ),
        flush=True,
    )
    print("QUALIFYING_SETTINGS", json.dumps(good), flush=True)
    print("PRECISION_CONTROL_REJECTED", json.dumps(control), flush=True)
    print("EXIT=0", flush=True)


if __name__ == "__main__":
    main()
