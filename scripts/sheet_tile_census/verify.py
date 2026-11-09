"""Validate completed sheet census receipts without trusting the batch exit alone."""

import argparse
import json
from pathlib import Path
import sys


def verify(directory):
    failures, summaries = [], []

    def require(condition, message):
        if not condition:
            failures.append(message)

    for size, count in (("0.08", 1696), ("0.04", 6493)):
        pair = []
        for outcome in ("miss", "hit"):
            name = f"{size}-{outcome}"
            path = directory / f"{name}.json"
            if not path.is_file():
                failures.append(f"{name}: receipt absent")
                continue
            row = json.loads(path.read_text())
            pair.append(row)
            require(row.get("phase") == "complete", f"{name}: receipt incomplete")
            require(row.get("partition") == "titan", f"{name}: wrong partition")
            require("P100" in row.get("device", ""), f"{name}: wrong device")
            require(row.get("platform") == "gpu", f"{name}: wrong platform")
            require(
                row.get("host", "").split(".")[0] == "98dci4-gpu-0002",
                f"{name}: wrong host",
            )
            require(row.get("cells") == count, f"{name}: wrong cell count")
            require(
                row.get("tile_cache_result") == outcome, f"{name}: wrong cache result"
            )
            entries = [
                r
                for r in row.get("cache_rows", [])
                if r.get("cache_key") == row.get("tile_cache_key")
            ]
            require(len(entries) == 1, f"{name}: tile cache entry absent or ambiguous")
            if len(entries) == 1:
                require(
                    entries[0].get("misses") == int(outcome == "miss")
                    and entries[0].get("hits") == int(outcome == "hit"),
                    f"{name}: wrong cache event counts",
                )
            log = directory / f"{name}.log"
            require(
                log.is_file() and log.read_text().rstrip().endswith("EXIT=0"),
                f"{name}: successful process exit absent",
            )
            if outcome == "miss":
                require(
                    row.get("matrix_finite") is True
                    and row.get("matrix_max_abs", 0) > 0,
                    f"{name}: finite nonzero matrix absent",
                )
                require(
                    row.get("matrix_shape") == [2 * count, 2 * count],
                    f"{name}: wrong matrix shape",
                )
                require(
                    row.get("compile_wall_seconds", float("inf")) <= 60,
                    f"{name}: compile exceeds 60 s",
                )
                require(
                    0 < row.get("executable_bytes", 0) <= 50_000_000,
                    f"{name}: executable outside byte bound",
                )
                require(
                    0 < row.get("peak_host_bytes", 0) <= 64_000_000_000,
                    f"{name}: host memory outside byte bound",
                )
                summaries.append(
                    {
                        k: row[k]
                        for k in (
                            "cells",
                            "compile_wall_seconds",
                            "executable_bytes",
                            "peak_host_bytes",
                            "warm_build_seconds",
                            "stablehlo_instructions",
                        )
                        if k in row
                    }
                )
        if len(pair) == 2:
            cold, warm = pair
            for key in ("tile_cache_key", "cache_root", "revision", "host", "job"):
                require(cold.get(key) == warm.get(key), f"{size}: {key} differs")
            require(cold.get("pid") != warm.get("pid"), f"{size}: process not fresh")
    path = directory / "control.json"
    if not path.is_file():
        failures.append("control: receipt absent")
    else:
        row = json.loads(path.read_text())
        require(
            row.get("phase") == "failed"
            and row.get("tile_cache_result") == "miss"
            and row.get("error", "").startswith("AssertionError:"),
            "control: cache-hit refusal absent",
        )
        log = directory / "control.log"
        ending = log.read_text().rstrip().splitlines()[-1] if log.is_file() else ""
        require(
            ending.startswith("EXIT=") and ending != "EXIT=0",
            "control: nonzero process exit absent",
        )
        require(
            all(
                row.get("cache_root") != json.loads(p.read_text()).get("cache_root")
                for p in directory.glob("*-miss.json")
            ),
            "control: root not distinct",
        )
    return {"complete": not failures, "cold_builds": summaries, "failures": failures}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.directory)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("SUMMARY " + json.dumps(result), flush=True)
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
