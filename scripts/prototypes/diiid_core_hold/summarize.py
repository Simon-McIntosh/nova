"""Reduce completed paired receipts without advancing a forward solve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CASES = (
    "frame-137-diverted",
    "frame-146-diverted",
    "frame-146-limited",
    "frame-179-limited",
    "mast-21978-35",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, pairs = [], {}
    for case in CASES:
        entries = {}
        for arm in ("base", "head"):
            folder = args.input / arm / case
            receipt = json.loads((folder / "receipt.json").read_text())
            analysis = json.loads((folder / "core-analysis.json").read_text())
            assert (folder / "row.log").read_text().rstrip().endswith("EXIT=0")
            topology = receipt["terminal_topology"]
            row = dict(
                arm=arm,
                case=case,
                revision=receipt["base_revision"],
                residual=receipt["residual"],
                converged=receipt["converged"],
                termination_reason=receipt["terminal_reason"],
                support=receipt["support"],
                maxima=analysis["maxima_count"],
                minima=analysis["minima_count"],
                source_centre_to_shoulder=analysis["source_centre_to_shoulder"],
                source_axis_to_shoulder=analysis["closure_shape"]["ratio"],
                map_centre_to_shoulder=analysis["map_core_shape"][
                    "central_to_shoulder_density"
                ],
                lower_xpoint_distance_m=topology["lower_saddle_distance_m"],
                lower_xpoint_qualified=topology["lower_saddle_qualified"],
                negative_terminal_nodes=analysis["terminal_negative_nodes"],
                source_profiles=analysis["profiles"],
                seed=receipt["seed_admission"],
                source_path=str(folder),
            )
            rows.append(row)
            entries[arm] = (row, analysis)
        b, h = [entries[arm][0] for arm in ("base", "head")]
        seed_equal = b["seed"][0]["sha256"] == h["seed"][0]["sha256"]
        distances = (b["lower_xpoint_distance_m"], h["lower_xpoint_distance_m"])
        pairs[case] = dict(
            paired_seed=seed_equal,
            one_maximum=h["maxima"] == 1,
            lower_xpoint_improved=all(value is not None for value in distances)
            and distances[1] < distances[0],
        )
        if case == "mast-21978-35":
            with (
                np.load(Path(b["source_path"]) / "terminal.npz") as base,
                np.load(Path(h["source_path"]) / "terminal.npz") as head,
            ):
                difference = np.asarray(head["state"] - base["state"])
                maximum = float(np.max(np.abs(difference)))
                tolerance = (
                    8
                    * np.finfo(float).eps
                    * max(1.0, float(np.max(np.abs(base["state"]))))
                )
            observed = {}
            for arm in ("base", "head"):
                records = entries[arm][1]["negative_source_evaluations"]
                positions = sorted(
                    {tuple(point) for record in records for point in record["rz_m"]}
                )
                observed[arm] = dict(
                    negative_calls=len(records),
                    unique_negative_nodes=len(positions),
                    rz_m=positions,
                )
            pairs[case].update(
                max_abs_state_difference_wb=maximum,
                roundoff_tolerance_wb=tolerance,
                unchanged_to_roundoff=maximum <= tolerance,
                converged=h["converged"],
                negative_source_nodes=observed,
                changed_source_domain=(
                    "psi_norm < 0 only; first knot is zero in both p_prime and ff_prime"
                ),
            )
            pairs[case]["passed"] = bool(
                h["maxima"] == 1
                and h["converged"]
                and (
                    maximum <= tolerance
                    or sum(
                        value["unique_negative_nodes"] for value in observed.values()
                    )
                    > 0
                )
            )
    verdict = (
        all(
            pairs[case]["one_maximum"]
            and pairs[case]["lower_xpoint_improved"]
            and pairs[case]["paired_seed"]
            for case in CASES[:2]
        )
        and pairs["mast-21978-35"]["passed"]
    )
    result = dict(rows=rows, pairs=pairs, passed=bool(verdict), forward_solves=0)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("PAIRED_VERDICT " + json.dumps(pairs), flush=True)
    print("MEASURE_PASSED=" + str(bool(verdict)), flush=True)


if __name__ == "__main__":
    main()
