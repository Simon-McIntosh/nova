"""Measure a frozen coefficient evaluator against independent extended precision."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import measure_bernstein as prior

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "bernstein-reference"
BASE = prior.BASE
SOURCE = "067ea754ce138a8b93a704f42877ab501957c9ad"
prior.OUTPUT = OUTPUT
revision = prior.revision
configure = prior.configure
original_evaluator = prior.original_evaluator
persist = prior.persist
TESTS = prior.TESTS


def program(arm):
    configure()
    import jax
    import jax.numpy as jnp
    from nova.linalg.interpolant import Bernstein

    if arm in ("baseline", "negative"):
        original_evaluator()
    # The small positive control must expose the traced special function before
    # an absence in the much larger certificate can be interpreted.
    probe = (
        jax.jit(lambda term: Bernstein(order=3).binom(term))
        .lower(jnp.arange(4))
        .compile()
    )
    probe_text = probe.as_text()
    print(
        "BINOMIAL_PROBE",
        {
            "gamma_present": "lgamma" in probe_text or "gamma" in probe_text,
            "bytes": len(probe.runtime_executable().serialize()),
        },
        flush=True,
    )
    from benchmarks.program_scope_census import _build_profile, _parse_instruction
    from nova.equilibrium.forward_operator import set_support_clip_mode

    launch_revision = revision()
    set_support_clip_mode("exact")
    profile, operator, request, seed, target = _build_profile(
        "weak-rotation-reactor-static", 110
    )
    kernel = profile._accelerated_history_program(
        "newton_krylov",
        requested_class=None,
        target_current=target,
        **request.policy.kernel_options(),
    )
    result = {
        "revision": launch_revision,
        "base_revision": BASE,
        "arm": arm,
        "tree": str(ROOT),
        "job": os.environ["SLURM_JOB_ID"],
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "requested_cells": 110,
        "realised_cells": int(operator.grid.node_number),
        "completed": False,
    }
    started = time.perf_counter()
    lowered = kernel.lower(seed, operator.external(None, None))
    result["lower_seconds"] = time.perf_counter() - started
    persist(f"{arm}-program.json", result)
    started = time.perf_counter()
    compiled = lowered.compile()
    result["compile_seconds"] = time.perf_counter() - started
    serialized = compiled.runtime_executable().serialize()
    result["serialized_bytes"] = len(serialized)
    result["serialized_sha256"] = hashlib.sha256(serialized).hexdigest()
    text = compiled.as_text()
    counts = Counter()
    for line in text.splitlines():
        parsed = _parse_instruction(line)
        if parsed is not None:
            counts[parsed["opcode"]] += 1
    assert counts["multiply"] > 0
    result["top_five_hlo_ops"] = counts.most_common(5)
    result["hlo_instructions"] = sum(counts.values())
    result["bernstein_binom_present"] = "Bernstein.binom" in text
    result["static_basis_present"] = "bernstein_basis" in text
    from benchmarks.program_scope_census import _parse_tables, _nova_source
    import gzip

    tables = _parse_tables(text)
    owners = Counter()
    functions = Counter()
    unowned = 0
    examples = {}
    for line in text.splitlines():
        parsed = _parse_instruction(line)
        if parsed is None:
            continue
        source = _nova_source(tables, parsed["meta"])
        if source:
            path = source["file"]
            path = "nova/" + path.rsplit("/nova/", 1)[-1] if "/nova/" in path else path
            owners[path] += 1
            name = path + ":" + str(source.get("function"))
            functions[name] += 1
            examples.setdefault(name, source)
        else:
            unowned += 1
    assert sum(owners.values()) + unowned == sum(counts.values())
    result["instruction_ownership"] = owners.most_common()
    result["function_ownership"] = functions.most_common(30)
    result["unowned_instructions"] = unowned
    result["ownership_examples"] = {
        name: examples[name] for name, _ in functions.most_common(15)
    }
    with gzip.open(OUTPUT / f"{arm}-optimized-hlo.txt.gz", "wt") as handle:
        handle.write(text)
    result["source_checkpoint"] = SOURCE
    result["completed"] = True
    result["below_ceiling"] = len(serialized) < 100_000_000
    persist(f"{arm}-program.json", result)


def child(arguments, name, cpus, wait=True):
    command = [sys.executable, str(Path(__file__).resolve()), *arguments]
    with (OUTPUT / f"{name}.log").open("w") as log:
        if name.startswith("negative"):
            log.write(prior.MUTATION + "\n")
        log.write(
            f"revision={revision()} source_checkpoint={SOURCE} "
            f"tree={ROOT} command={command!r}\n"
        )
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
            preexec_fn=lambda: os.sched_setaffinity(0, cpus),
        )
    if not wait:
        return process
    code = process.wait()
    (OUTPUT / f"{name}.exit").write_text(str(code) + "\n")
    print(name, "exit", code, flush=True)
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", choices=("baseline", "candidate", "negative"))
    parser.add_argument("--suite", nargs=2)
    parser.add_argument("--suites", action="store_true")
    parser.add_argument("--row", nargs=2)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--rows", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path in (
        "nova/linalg/interpolant.py",
        "nova/equilibrium/flux_surface_extraction.py",
    ):
        expected = subprocess.check_output(
            ["git", "show", f"{SOURCE}:{path}"], cwd=ROOT
        )
        assert expected == (ROOT / path).read_bytes(), (
            f"source checkpoint changed: {path}"
        )
    if args.program:
        program(args.program)
        return 0
    if args.suite:
        return prior.suite(*args.suite)
    cpus = sorted(os.sched_getaffinity(0))
    if args.rows:
        for case in (
            "weak-rotation-reactor-static",
            "moderate-rotation-conventional-static",
            "strong-rotation-compact-static",
            "diverted-single-null",
        ):
            for cells in (110, 300):
                child(["--row", case, str(cells)], f"{case}-{cells}-measurement", cpus)
        return 0
    if args.suites:
        return max(
            child(["--suite", arm, name], f"{arm}-{Path(name).stem}", cpus)
            for arm in ("baseline", "candidate")
            for name in TESTS
        )
    if args.row:
        original_persist = prior.persist

        def publish(name, value):
            value["previous_route_equivalence_1e14"] = value.pop("passed")
            value["previous_route_equivalence_is_acceptance_gate"] = False
            original_persist(name, value)

        prior.persist = publish
        prior.row(*args.row)
        return 0
    if args.reference:
        deadline = time.monotonic() + 600
        while (OUTPUT / "row-repair-pending").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("single-null measurement repair is pending")
            time.sleep(1)
        from reference_clip import measure

        measure(OUTPUT)
        return 0
    assert len(cpus) >= 8 and os.environ.get("SLURM_JOB_ID")
    tests = child(["--suites"], "suites", cpus[:4], wait=False)
    for arm in ("baseline", "candidate"):
        child(["--program", arm], f"{arm}-program", cpus[4:8])
    (OUTPUT / "suites.exit").write_text(str(tests.wait()) + "\n")
    for case in (
        "weak-rotation-reactor-static",
        "moderate-rotation-conventional-static",
        "strong-rotation-compact-static",
        "diverted-single-null",
    ):
        for cells in (110, 300):
            child(["--row", case, str(cells)], f"{case}-{cells}", cpus[:4])
    deadline = time.monotonic() + 600
    while not (OUTPUT / "reference-ready.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("independent reference is not ready")
        time.sleep(5)
    child(["--reference"], "reference", cpus[:4])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
