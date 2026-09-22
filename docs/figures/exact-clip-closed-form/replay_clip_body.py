"""Validate the frozen clip source in four lanes of the held allocation."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import measure_clip_body as gates
import measure_reference as census
import measure_bernstein as prior

HELD = gates.OUTPUT
OUTPUT = HELD.parent / "clip-body-audit"
gates.OUTPUT = census.OUTPUT = prior.OUTPUT = OUTPUT
CASES = (
    "weak-rotation-reactor-static",
    "moderate-rotation-conventional-static",
    "strong-rotation-compact-static",
    "diverted-single-null",
)


def child(args, name, cpus):
    import subprocess

    command = [sys.executable, str(Path(__file__).resolve()), *args]
    with (OUTPUT / (name + ".log")).open("w") as log:
        if name.startswith("negative"):
            log.write(gates.MUTATION + "\n")
        log.write(
            f"revision={prior.revision()} tree={gates.ROOT} command={command!r}\n"
        )
        log.flush()
        return subprocess.Popen(
            command,
            cwd=gates.ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            preexec_fn=lambda: os.sched_setaffinity(0, cpus),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program")
    parser.add_argument("--programs", action="store_true")
    parser.add_argument("--suite")
    parser.add_argument("--suites", action="store_true")
    parser.add_argument("--row", nargs=2)
    parser.add_argument("--rows", type=int)
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    cpus = sorted(os.sched_getaffinity(0))
    if args.program:
        census.SOURCE = prior.revision()
        census.program(args.program)
        path = OUTPUT / (args.program + "-program.json")
        value = json.loads(path.read_text())
        value["shared_cell_body_present"] = any(
            ":_clip_cell" in name for name, _ in value["function_ownership"]
        )
        prior.persist(path.name, value)
        return 0
    if args.programs:
        return max(
            gates.finish(
                child(["--program", arm], arm + "-program", cpus), arm + "-program"
            )
            for arm in ("candidate", "negative")
        )
    if args.suite:
        prior.configure()
        import pytest

        return pytest.main(
            [str(gates.ROOT / "tests" / args.suite), "-q", "-p", "no:cacheprovider"]
        )
    if args.suites:
        return max(
            gates.finish(
                child(["--suite", name], "candidate-" + Path(name).stem, cpus),
                "candidate-" + Path(name).stem,
            )
            for name in gates.TESTS
        )
    if args.row:
        prior.configure()
        gates.row(*args.row)
        return 0
    if args.rows is not None:
        rows = [(case, cells) for case in CASES for cells in (110, 300)]
        return max(
            gates.finish(
                child(["--row", case, str(cells)], f"{case}-{cells}", cpus),
                f"{case}-{cells}",
            )
            for case, cells in rows[args.rows :: 2]
        )
    assert len(cpus) >= 16
    for name in (
        "baseline-program.json",
        "baseline-program.log",
        "baseline-optimized-hlo.txt.gz",
        "baseline-refusal-corrected.log",
        "baseline-suite-receipt.log",
    ):
        shutil.copy2(HELD / name, OUTPUT / name)
    for path in HELD.glob("baseline-test_*.log"):
        shutil.copy2(path, OUTPUT / path.name)
    jobs = [
        (child(["--programs"], "programs", cpus[:4]), "programs"),
        (child(["--suites"], "suites", cpus[4:8]), "suites"),
        (child(["--rows", "0"], "rows-even", cpus[8:12]), "rows-even"),
        (child(["--rows", "1"], "rows-odd", cpus[12:16]), "rows-odd"),
    ]
    codes = {name: gates.finish(process, name) for process, name in jobs}
    prior.persist("allocation-result.json", codes)
    if all(code == 0 for code in codes.values()):
        import render_clip_body

        render_clip_body.OUTPUT = OUTPUT
        render_clip_body.render()
    (HELD / "final-validation.done").write_text(json.dumps(codes) + "\n")
    return max(codes.values())


if __name__ == "__main__":
    raise SystemExit(main())
