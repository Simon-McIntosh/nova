"""Run a complete source-checkpoint audit in four lanes of one allocation."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import measure_reference as gates
import measure_bernstein as prior

ROOT = gates.ROOT
OUTPUT = Path(__file__).resolve().parent / "bernstein-reference-audit"
gates.OUTPUT = OUTPUT
prior.OUTPUT = OUTPUT


def child(args, name, cpus):
    command = [sys.executable, str(Path(__file__).resolve()), *args]
    with (OUTPUT / f"{name}.log").open("w") as log:
        log.write(
            f"revision={gates.revision()} source_checkpoint={gates.SOURCE} "
            f"tree={ROOT} command={command!r}\n"
        )
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            preexec_fn=lambda: os.sched_setaffinity(0, cpus),
        )
    return process


def finish(process, name):
    code = process.wait()
    (OUTPUT / f"{name}.exit").write_text(str(code) + "\n")
    print(name, code, flush=True)
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program")
    parser.add_argument("--suite")
    parser.add_argument("--tests", action="store_true")
    parser.add_argument("--row", nargs=2)
    parser.add_argument("--rows", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    for path in (
        "nova/linalg/interpolant.py",
        "nova/equilibrium/flux_surface_extraction.py",
    ):
        assert (
            subprocess.check_output(["git", "show", f"{gates.SOURCE}:{path}"], cwd=ROOT)
            == (ROOT / path).read_bytes()
        )
    if args.program:
        gates.program(args.program)
        return 0
    if args.suite:
        return prior.suite("candidate", args.suite)
    cpus = sorted(os.sched_getaffinity(0))
    if args.tests:
        return max(
            finish(
                child(["--suite", name], "candidate-" + Path(name).stem, cpus),
                "candidate-" + Path(name).stem,
            )
            for name in prior.TESTS
        )
    if args.row:
        persist = prior.persist

        def publish(name, value):
            value["previous_route_equivalence_1e14"] = value.pop("passed")
            value["previous_route_equivalence_is_acceptance_gate"] = False
            persist(name, value)

        prior.persist = publish
        prior.row(*args.row)
        return 0
    if args.rows:
        codes = []
        for case in (
            "weak-rotation-reactor-static",
            "moderate-rotation-conventional-static",
            "strong-rotation-compact-static",
            "diverted-single-null",
        ):
            for cells in (110, 300):
                name = f"{case}-{cells}"
                codes.append(
                    finish(child(["--row", case, str(cells)], name, cpus), name)
                )
        from reference_clip import measure

        measure(OUTPUT)
        return max(codes)
    assert len(cpus) >= 16
    jobs = [
        (
            child(["--program", "baseline"], "baseline-program", cpus[:4]),
            "baseline-program",
        ),
        (
            child(["--program", "candidate"], "candidate-program", cpus[4:8]),
            "candidate-program",
        ),
        (child(["--tests"], "suites", cpus[8:12]), "suites"),
        (child(["--rows"], "rows", cpus[12:16]), "rows"),
    ]
    results = {name: finish(process, name) for process, name in jobs}
    (OUTPUT / "allocation-result.json").write_text(json.dumps(results, indent=2) + "\n")
    return max(results.values())


if __name__ == "__main__":
    raise SystemExit(main())
