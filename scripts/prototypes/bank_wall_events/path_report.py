"""Compare clean repetitions and wall outputs captured along diagnostic solves."""

import argparse
from html import escape
import json
from pathlib import Path
import re

import numpy as np

from scripts.prototypes.bank_solve_cycle.report import pair_gaps
from scripts.prototypes.bank_wall_events.trajectory import difference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    records = {
        f"{arm}-{repeat}": json.loads((args.run / f"{arm}-{repeat}.json").read_text())
        for repeat in range(2)
        for arm in ("base", "head")
    }
    physical = (args.run / "physical.log").read_text()
    assert physical.rstrip().endswith("EXIT=0")
    measured = re.findall(
        r"WALL_RELATIVE (limited|diverted) (trial|converged) (\[[^\]]+\])", physical
    )
    assert {(k, s) for k, s, _ in measured} == {
        (k, s) for k in ("limited", "diverted") for s in ("trial", "converged")
    }
    errors = {k + "-" + s: json.loads(v) for k, s, v in measured}
    assert all(np.all(np.asarray(v) <= 1e-13) for v in errors.values())
    comparison = dict(
        physical_errors=errors, inputs={}, precision={}, wall={}, repeats={}
    )
    reference = records["base-0"]
    for key, row in records.items():
        assert (args.run / (key + ".log")).read_text().rstrip().endswith("EXIT=0")
        comparison["inputs"][key] = [
            name
            for name, value in row["inputs"].items()
            if value != reference["inputs"][name]
        ]
        gaps, limits = pair_gaps(reference["precision"], row["precision"])
        comparison["precision"][key] = {
            name: max(value) for name, value in gaps.items()
        }
        assert all(max(value) <= limits[name] for name, value in gaps.items())
        for call in row["calls"]:
            assert all(call["converged"]) and max(call["residual"]) <= 1e-10
        comparison["repeats"][key] = dict(
            within_process_counters_equal=all(
                row["calls"][0][name] == row["calls"][1][name]
                for name in ("steps", "reads")
            ),
            within_process_state_equal=row["calls"][0]["state"]
            == row["calls"][1]["state"],
            observed_matches_unaltered=row["observed"]["matches_unaltered_counters"],
            unaltered_reads=[call["reads"] for call in row["calls"]],
            observed_reads=row["observed"]["reads"],
        )
    for repeat in range(2):
        before = np.load(args.run / f"base-{repeat}.npz")
        after = np.load(args.run / f"head-{repeat}.npz")
        count = min(len(before["points"]), len(after["points"]))
        metrics = {
            name: difference(
                np.moveaxis(after[name][:count], 1, 0),
                np.moveaxis(before[name][:count], 1, 0),
            )
            for name in ("points", "values", "coefficients")
        }
        first = {name: difference(after[name][0], before[name][0]) for name in metrics}
        comparison["wall"][str(repeat)] = dict(
            common_read_ordinals=count, maxima=metrics, first_read=first
        )
        before.close()
        after.close()
    same_input = {}
    replay = {}
    for output, prefix in ((same_input, "same_input"), (replay, "own_replay")):
        for kind in ("points", "values"):
            columns = [
                entry[prefix + "_" + kind]
                for row in records.values()
                for entry in row["wall_comparisons"]
            ]
            output[kind] = {
                metric: np.max([col[metric] for col in columns], axis=0).tolist()
                for metric in (
                    "absolute",
                    "relative",
                    "nonfinite_mismatch",
                    "bit_differences",
                )
            }
    comparison["same_input"] = same_input
    comparison["own_replay"] = replay
    (args.run / "path-comparison.json").write_text(json.dumps(comparison, indent=2))
    for key, row in records.items():
        print("COUNTERS", key, json.dumps(comparison["repeats"][key]), flush=True)
        print("INPUT_DIFFERENCES", key, comparison["inputs"][key], flush=True)
    print("PHYSICAL", json.dumps(errors), flush=True)
    print(
        "SAME_INPUT_MAX",
        json.dumps(
            {
                kind: {metric: max(values) for metric, values in row.items()}
                for kind, row in same_input.items()
            }
        ),
        flush=True,
    )
    print(
        "OWN_REPLAY_MAX",
        json.dumps(
            {
                kind: {metric: max(values) for metric, values in row.items()}
                for kind, row in replay.items()
            }
        ),
        flush=True,
    )
    text = '<h3 id="bank-wall-events-repeats">Repeated sixteen-slice solves</h3>\n'
    text += (
        "<p>Each arm runs in two fresh Python processes. Each process "
        "compiles once, warms once, then makes two timed calls on identical"
        " inputs. Timing excludes diagnostic output capture. Lane vectors "
        "below are ordered 0 through 15. Per-read wall is whole-solve wall "
        "divided by each lane’s reported reads; it includes factor and "
        "update work, whose counts are shown alongside it.</p>\n"
    )
    text += (
        '<table id="bank-wall-events-repeat-table"><thead><tr><th>Arm / '
        "process / call</th><th>Updates by lane</th><th>Reads by "
        "lane</th><th>Whole solve [ms]</th><th>Whole solve / reads "
        "[ms]</th></tr></thead><tbody>\n"
    )
    for key, row in records.items():
        for call in row["calls"]:
            readings = np.asarray(call["per_read_ms"])
            normalized = (
                f"{readings.min():.3f}"
                if np.ptp(readings) == 0
                else f"{readings.min():.3f}–{readings.max():.3f}"
            )
            text += (
                f"<tr><td>{key} / {call['call']}</td>"
                f"<td>{', '.join(map(str, call['steps']))}</td>"
                f"<td>{', '.join(map(str, call['reads']))}</td>"
                f"<td>{call['wall_ms']:.3f}</td><td>{normalized}</td></tr>\n"
            )
    text += "</tbody></table>\n"
    text += (
        '<h3 id="bank-wall-events-output-pairs">Wall outputs inside '
        "diagnostic solves</h3>\n"
    )
    text += (
        "<p>Separate diagnostic solves return the original wall outputs and"
        " capture those outputs plus their Bell coefficients through "
        "ordered callbacks. A positive control checks lane order 0–15, and "
        "the callback count must equal the solve’s read counter in every "
        "lane. Raw captures are retained. The cross-arm columns compare "
        "common read ordinals; after paths diverge, the fields at a shared "
        "ordinal need not represent the same Newton state. The identical-"
        "input columns separately evaluate both wall implementations, "
        "compiled independently, on every captured field. Each cell lists "
        "the largest absolute point-coordinate / finite-value difference "
        "for that lane, in metres / native flux units. Nonfinite-mask "
        "differences and bit differences are preserved in the JSON "
        "receipt.</p>\n"
    )
    text += (
        '<table id="bank-wall-events-output-'
        'table"><thead><tr><th>Lane</th><th>Captured base/head, process '
        "0</th><th>Captured base/head, process 1</th><th>Identical-input "
        "implementations</th></tr></thead><tbody>\n"
    )
    for lane in range(16):
        cells = []
        for repeat in range(2):
            row = comparison["wall"][str(repeat)]["maxima"]
            cells.append(
                f"{row['points']['absolute'][lane]:.3g} / "
                f"{row['values']['absolute'][lane]:.3g}"
            )
        cells.append(
            f"{same_input['points']['absolute'][lane]:.3g} / "
            f"{same_input['values']['absolute'][lane]:.3g}"
        )
        text += (
            "<tr><td>"
            + str(lane)
            + "</td>"
            + "".join("<td>" + value + "</td>" for value in cells)
            + "</tr>\n"
        )
    text += "</tbody></table>\n"
    text += (
        "<p>Input-leaf fingerprints, both timed residual histories, diagnostic "
        "counters, per-read comparisons and full captures are under <code>"
        + escape(str(args.run))
        + "</code>.</p>\n"
    )
    (args.run / "tables.html").write_text(text)


if __name__ == "__main__":
    main()
