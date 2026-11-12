"""Run one measurement family with archived trees and durable arm receipts."""

import argparse
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile

PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
BASES = {
    "decisions": "8ce11904123bfebd516bc418416a6ec5eeb10117",
    "booking": "a766ce176848d12fe88771788216007a28df0ad6",
    "candidates": "597c44f4e9a568c007b1f01e6a8c79545c333925",
    "wall": "519d378f9e6f8d2786f69af64569625971651a73",
}
LANE_BASE = "eb78afe5355009db83dfbc5750cc5081f37859d7"


def refuse_inert(left, right):
    """Require a program or executed-work difference before timing comparison."""
    a = [
        (r.get("lowered_sha256"), r.get("execution_counts"))
        for r in left["rows"]
        if "lowered_sha256" in r
    ]
    b = [
        (r.get("lowered_sha256"), r.get("execution_counts"))
        for r in right["rows"]
        if "lowered_sha256" in r
    ]
    if a == b:
        raise ValueError("inert lever: identical lowered programs and execution counts")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--measurement",
        choices=[
            "all",
            "decisions",
            "booking",
            "factor",
            "scalar",
            "lane",
            "candidates",
            "wall",
            "headline",
        ],
        default="all",
    )
    parser.add_argument("--revision")
    parser.add_argument("--reuse-run", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    head = (
        args.revision
        or subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    )
    output.mkdir(exist_ok=True)
    print("REVISION", head, "JOB", os.environ.get("SLURM_JOB_ID"), flush=True)
    env = dict(
        os.environ,
        TMPDIR="/tmp",
        JAX_PLATFORMS="cpu" if args.smoke else "cuda,cpu",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        JAX_MEMORY_FITTING_LEVEL="O3",
        JAX_MEMORY_FITTING_EFFORT="1.0",
        JAX_COMPILATION_CACHE_DIR=str(output / "compilation-cache"),
        JAX_ENABLE_COMPILATION_CACHE="true",
        JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="1",
        JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="0",
    )
    for key in ("UV_NO_SYNC", "UV_RUN_RECURSION_DEPTH", "XLA_FLAGS"):
        env.pop(key, None)
    summary = dict(
        revision=head,
        measurement=args.measurement,
        smoke=args.smoke,
        completed=False,
        arms=[],
        comparisons=[],
    )

    def save():
        (output / "summary.json").write_text(json.dumps(summary, indent=2))

    save()
    with tempfile.TemporaryDirectory(
        prefix="h200-measure-", dir=env["TMPDIR"]
    ) as temporary:
        scratch = Path(temporary)
        env["TMPDIR"] = str(scratch)

        def extract(revision, tree, paths):
            tree.mkdir(exist_ok=True)
            data = subprocess.check_output(
                ["git", "-C", str(root), "archive", revision, *paths]
            )
            with tarfile.open(fileobj=io.BytesIO(data)) as archive:
                archive.extractall(tree, filter="data")

        main_tree = scratch / "main"
        extract(
            head,
            main_tree,
            ["nova", "scripts", "tests", "benchmarks", "conftest.py", "pyproject.toml"],
        )
        trees = {"main": (main_tree, head)}
        for name, revision in BASES.items():
            tree = scratch / name
            extract(revision, tree, ["nova"])
            for path in ["scripts", "tests", "benchmarks"]:
                (tree / path).symlink_to(main_tree / path, target_is_directory=True)
            trees[name] = (tree, revision)
        old_driver = scratch / "shared_drive.py"
        old_driver.write_bytes(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(root),
                    "show",
                    LANE_BASE + ":nova/equilibrium/drive_continuation.py",
                ]
            )
        )
        instrument = main_tree / "scripts/prototypes/h200_levers/measure.py"
        common = (
            (args.reuse_run / "common-streams.npz").resolve()
            if args.reuse_run
            else output / "common-streams.npz"
        )
        assert not args.reuse_run or common.is_file(), common

        def execute(
            label,
            tree,
            revision,
            kind,
            arm,
            width,
            step=1e-5,
            driver=None,
            population="diverted",
        ):
            receipt = output / (label + ".json")
            if args.reuse_run and not args.smoke:
                prior = args.reuse_run / (label + ".json")
                prior_log = args.reuse_run / (label + ".log")
                if prior.is_file() and prior_log.is_file():
                    record = json.loads(prior.read_text())
                    if (
                        record.get("complete")
                        and prior_log.read_text().splitlines()[-1] == "EXIT=0"
                    ):
                        summary["arms"].append(
                            dict(
                                label=label,
                                receipt=str(prior),
                                log=str(prior_log),
                                exit_status=0,
                                retained=True,
                            )
                        )
                        save()
                        print("RETAINED", label, flush=True)
                        return record
            command = [
                PYTHON,
                str(instrument),
                str(tree),
                "--revision",
                revision,
                "--kind",
                kind,
                "--arm",
                arm,
                "--width",
                str(width),
                "--output",
                str(receipt),
                "--inputs",
                str(common),
                "--step",
                str(step),
            ]
            command += [
                "--backend",
                "cpu" if args.smoke else "gpu",
                "--population",
                population,
            ]
            if args.smoke:
                command += ["--smoke"]
            if driver:
                command += ["--driver", str(driver)]
            log = output / (label + ".log")
            with log.open("w") as stream:
                stream.write(
                    f"# revision: {revision} tree: {tree} "
                    f"command: PYTHONPATH={tree} {shlex.join(command)}\n"
                )
                stream.flush()
                result = subprocess.run(
                    command,
                    cwd=tree,
                    env=dict(env, PYTHONPATH=str(tree)),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                stream.write(f"EXIT={result.returncode}\n")
            print("ARM", label, "EXIT", result.returncode, flush=True)
            summary["arms"].append(
                dict(
                    label=label,
                    receipt=str(receipt),
                    log=str(log),
                    exit_status=result.returncode,
                )
            )
            save()
            if result.returncode:
                raise RuntimeError(f"arm failed: {label}; inspect {log}")
            record = json.loads(receipt.read_text())
            assert record["complete"]
            return record

        if not args.reuse_run:
            execute("common-inputs", main_tree, head, "inputs", "common", 16)

        def metrics(record):
            return {
                r["name"]: r["median_ms"] for r in record["rows"] if "median_ms" in r
            }

        def arm(
            name,
            tree_key,
            kind,
            flag,
            width,
            step=1e-5,
            driver=None,
            population="diverted",
        ):
            tree, revision = trees[tree_key]
            stem = f"{name}-{flag}-{width}-{step:g}"
            if kind == "wall":
                stem += "-" + population
            if args.smoke:
                record = execute(
                    stem + "-smoke",
                    tree,
                    revision,
                    kind,
                    flag,
                    width,
                    step,
                    driver,
                    population,
                )
                return dict(
                    name=name,
                    arm=flag,
                    width=width,
                    step=step,
                    metrics=metrics(record),
                    second_metrics=metrics(record),
                    spreads={key: 0.0 for key in metrics(record)},
                    records=[record],
                )
            execute(
                stem + "-warmup",
                tree,
                revision,
                kind,
                flag,
                width,
                step,
                driver,
                population,
            )
            # A disagreeing pair is discarded in full, never averaged into another pair.
            for attempt in range(1, 4):
                records = [
                    execute(
                        stem + f"-pair{attempt}-{i}",
                        tree,
                        revision,
                        kind,
                        flag,
                        width,
                        step,
                        driver,
                        population,
                    )
                    for i in (1, 2)
                ]
                a, b = map(metrics, records)
                spreads = {key: abs(a[key] - b[key]) / min(a[key], b[key]) for key in a}
                if max(spreads.values()) <= 0.05:
                    return dict(
                        name=name,
                        arm=flag,
                        width=width,
                        step=step,
                        metrics=a,
                        second_metrics=b,
                        spreads=spreads,
                        records=records,
                    )
                summary.setdefault("disagreements", []).append(
                    dict(arm=stem, attempt=attempt, spreads=spreads)
                )
                save()
            raise RuntimeError(f"fresh processes disagree after three pairs: {stem}")

        def paired(
            name, kind, off_tree="main", step=1e-5, drivers=None, population="diverted"
        ):
            for width in (
                (16,)
                if args.smoke and kind in ("wall", "lane")
                else (1,)
                if args.smoke
                else (1, 16)
            ):
                off = arm(
                    name,
                    off_tree,
                    kind,
                    "off",
                    width,
                    step,
                    drivers[0] if drivers else None,
                    population,
                )
                on = arm(
                    name,
                    "main",
                    kind,
                    "on",
                    width,
                    step,
                    drivers[1] if drivers else None,
                    population,
                )
                entry = dict(
                    name=name,
                    width=width,
                    step=step,
                    population=population,
                    off=off,
                    on=on,
                )
                try:
                    refuse_inert(off["records"][0], on["records"][0])
                except ValueError as error:
                    entry["refusal"] = str(error)
                summary["comparisons"].append(entry)
                save()

        chosen = args.measurement
        if chosen in ("all", "decisions"):
            paired("carried-decisions", "read", "decisions")
        if chosen in ("all", "booking"):
            paired("booking", "booking", "booking")
        if chosen in ("all", "factor"):
            for step in (1e-5, 1e-3):
                paired("factor-carry", "stream", step=step)
        if chosen in ("all", "scalar"):
            for width in (1,) if args.smoke else (1, 16):
                tiles = [
                    arm("scalar-tile", "main", "scalar", str(tile), width)
                    for tile in (32, 64, 128)
                ]
                for tile in tiles[1:]:
                    try:
                        refuse_inert(tiles[0]["records"][0], tile["records"][0])
                    except ValueError as error:
                        tile["refusal"] = str(error)
                summary["comparisons"].append(
                    dict(name="scalar-tile", width=width, tiles=tiles)
                )
                save()
        if chosen in ("all", "lane"):
            paired(
                "lane-drive",
                "lane",
                drivers=(
                    old_driver,
                    main_tree / "nova/equilibrium/drive_continuation.py",
                ),
            )
        if chosen in ("all", "candidates"):
            paired("live-candidates", "read", "candidates")
        if chosen in ("all", "wall"):
            for population in ("diverted", "mixed"):
                paired("wall-skip", "wall", "wall", population=population)
        if chosen in ("all", "headline"):
            summary["headline"] = arm(
                "secant-flattop", "main", "stream", "off", 1 if args.smoke else 16
            )
        sample = dict(rows=[dict(lowered_sha256="known-present-program")])
        try:
            refuse_inert(sample, sample)
        except ValueError as error:
            (output / "negative-control.log").write_text(
                f"# revision: {head} command: refuse_inert(receipt, receipt)\n"
                f"{error}\nEXIT=1\n"
            )
        else:
            raise AssertionError("identical-program negative control escaped refusal")
        summary["completed"] = True
        save()
    print("SUMMARY all arms completed with two agreeing processes", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        print("EXIT=1", flush=True)
        raise
    print("EXIT=0", flush=True)
