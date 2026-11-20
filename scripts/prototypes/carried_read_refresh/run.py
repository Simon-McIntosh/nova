"""Pair full and selective fragment refresh in one H200 allocation."""

import argparse
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile

from scripts.prototypes.h200_levers.run import PYTHON, refuse_inert


def compare(left, right, program):
    def count(record):
        return {
            "rows": [
                {
                    "lowered_sha256": None,
                    "execution_counts": record["programs"][program][
                        "optimized_operations"
                    ],
                }
            ]
        }

    refuse_inert(count(left), count(right))
    try:
        refuse_inert(count(left), count(left))
    except ValueError as error:
        return str(error)
    raise AssertionError("identical program escaped the inert-lever guard")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("revision")
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    summary = dict(revision=args.revision, complete=False, arms=[], comparisons=[])

    def save():
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))

    save()
    env = dict(
        os.environ,
        JAX_PLATFORMS="cuda,cpu",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        JAX_COMPILATION_CACHE_DIR=str(args.output / "compilation-cache"),
        JAX_ENABLE_COMPILATION_CACHE="true",
        JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="1",
        JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="0",
        NOVA_COMPILATION_CACHE_ROOT="/work/projects/imas_gpu/nova/jax-cache/nova-prewarm",
    )
    for key in ("UV_NO_SYNC", "UV_RUN_RECURSION_DEPTH", "XLA_FLAGS", "PYTEST_ADDOPTS"):
        env.pop(key, None)
    with tempfile.TemporaryDirectory(
        prefix="carried-refresh-", dir=os.environ["TMPDIR"]
    ) as scratch:
        tree = Path(scratch) / "source"
        tree.mkdir()
        data = subprocess.check_output(
            [
                "git",
                "-C",
                str(args.root),
                "archive",
                args.revision,
                "nova",
                "scripts",
                "tests",
                "benchmarks",
                "conftest.py",
                "pyproject.toml",
            ]
        )
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            archive.extractall(tree, filter="data")
        env.update(PYTHONPATH=str(tree), TMPDIR=scratch)
        for width in (1, 16, 64):
            for kind in ("read", "slice"):
                measured = {}
                for process in (0, 1, 2):
                    for arm in ("off", "on"):
                        target = args.output / f"{kind}-{width}-{arm}-{process}"
                        target.mkdir(exist_ok=True)
                        command = [
                            PYTHON,
                            "-m",
                            "scripts.prototypes.carried_read_refresh.measure",
                            "--reuse",
                            arm,
                            "--output",
                            str(target),
                            "--inputs",
                            str(args.inputs),
                            "--width",
                            str(width),
                            "--revision",
                            args.revision,
                            "--instrument",
                            args.revision,
                        ]
                        if kind == "read":
                            command.append("--read-only")
                        log = target / "gate.log"
                        with log.open("w") as stream:
                            stream.write(
                                f"# revision: {args.revision} tree: {tree} "
                                f"command: PYTHONPATH={tree} {shlex.join(command)}\n"
                            )
                            stream.flush()
                            result = subprocess.run(
                                command,
                                cwd=tree,
                                env=env,
                                stdout=stream,
                                stderr=subprocess.STDOUT,
                            )
                            stream.write(f"EXIT={result.returncode}\n")
                        summary["arms"].append(
                            dict(
                                arm=arm,
                                width=width,
                                kind=kind,
                                process=process,
                                log=str(log),
                                exit_status=result.returncode,
                            )
                        )
                        save()
                        print("ARM", target.name, "EXIT", result.returncode, flush=True)
                        if result.returncode:
                            raise RuntimeError(f"measurement failed: {log}")
                        receipt = json.loads((target / "receipt.json").read_text())
                        assert receipt["complete"]
                        measured[arm, process] = receipt
                key = "carried" if kind == "read" else "solve"
                refusal = compare(measured["off", 1], measured["on", 1], key)
                means, spreads = {}, {}
                for arm in ("off", "on"):

                    def wall(record):
                        return (
                            record["programs"]["carried"]["median_ms"]
                            if kind == "read"
                            else record["end_to_end_ms"]
                        )

                    a, b = (wall(measured[arm, p]) for p in (1, 2))
                    spreads[arm] = abs(a - b) / min(a, b)
                    assert spreads[arm] <= 0.05, (
                        kind,
                        width,
                        arm,
                        "fresh processes disagree",
                        a,
                        b,
                    )
                    means[arm] = (a + b) / 2
                margin = max(0.03, *spreads.values())
                gain = 1 - means["on"] / means["off"]
                summary["comparisons"].append(
                    dict(
                        width=width,
                        kind=kind,
                        batch_ms=means,
                        spread=spreads,
                        gain=gain,
                        margin=margin,
                        faster=gain > margin,
                        inert_refusal=refusal,
                    )
                )
                save()
                print("COMPARISON", json.dumps(summary["comparisons"][-1]), flush=True)
    summary["complete"] = True
    save()
    print("SUMMARY all paired widths complete", flush=True)


if __name__ == "__main__":
    main()
