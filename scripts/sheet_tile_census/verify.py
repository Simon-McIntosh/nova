"""Validate completed sheet census receipts without trusting the batch exit alone."""

import argparse
import json
from pathlib import Path
import sys


def verify(directory, recovery_directory=None):
    failures, summaries = [], []

    def require(condition, message):
        if not condition:
            failures.append(message)

    for size, count in (("0.08", 1696), ("0.04", 6493)):
        pair = []
        cache_directory = (
            recovery_directory if size == "0.04" and recovery_directory else directory
        )
        arms = [(directory, "miss", True)]
        if cache_directory != directory:
            arms.append((cache_directory, "miss", False))
        arms.append((cache_directory, "hit", False))
        measured = None
        for source, outcome, full_build in arms:
            name = f"{size}-{outcome}"
            path = source / f"{name}.json"
            if not path.is_file():
                failures.append(f"{name}: receipt absent")
                continue
            row = json.loads(path.read_text())
            if full_build:
                measured = row
            if source == cache_directory:
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
            require(
                row.get("warm_tile_seconds", 0) > 0, f"{name}: tile execution absent"
            )
            log = source / f"{name}.log"
            require(
                log.is_file() and log.read_text().rstrip().endswith("EXIT=0"),
                f"{name}: successful process exit absent",
            )
            if full_build:
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
            if measured is not None:
                for key in (
                    "tile_cache_key",
                    "stablehlo_instructions",
                    "stablehlo_bytes",
                    "cells",
                    "platform",
                    "device",
                    "host",
                ):
                    require(
                        cold.get(key) == measured.get(key),
                        f"{size}: recovery {key} differs from full build",
                    )
    control_directory = recovery_directory or directory
    path = control_directory / "control.json"
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
        require(
            row.get("partition") == "titan"
            and row.get("platform") == "gpu"
            and "P100" in row.get("device", "")
            and row.get("host", "").split(".")[0] == "98dci4-gpu-0002",
            "control: wrong platform or host",
        )
        if recovery_directory:
            require(row.get("cells") == 6493, "control: wrong recovery cell count")
        log = control_directory / "control.log"
        ending = log.read_text().rstrip().splitlines()[-1] if log.is_file() else ""
        require(
            ending.startswith("EXIT=") and ending != "EXIT=0",
            "control: nonzero process exit absent",
        )
        require(
            all(
                row.get("cache_root") != json.loads(p.read_text()).get("cache_root")
                for source in {directory, control_directory}
                for p in source.glob("*-miss.json")
            ),
            "control: root not distinct",
        )
    return {"complete": not failures, "cold_builds": summaries, "failures": failures}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--recovery-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.directory, args.recovery_directory)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("SUMMARY " + json.dumps(result), flush=True)
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
