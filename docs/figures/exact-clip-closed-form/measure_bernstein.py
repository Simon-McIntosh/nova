"""Measure static Bernstein evaluation against the traced gamma implementation."""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "bernstein"
BASE = "b3abce70bd92fa143dbfbfd9d449f7de06de7dc2"
TESTS = (
    "test_interpolant.py",
    "test_exact_clip_moments.py",
    "test_flux_surface_extraction.py",
)
MUTATION = "restore the traced binomial evaluation"


def revision():
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def baseline_source(path):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{BASE}:{path}"], text=True
    )


def baseline_module(path, name):
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / path)
    sys.modules[name] = module
    exec(compile(baseline_source(path), module.__file__, "exec"), module.__dict__)
    return module


def original_evaluator():
    import nova.linalg.interpolant as interpolant
    import nova.equilibrium.flux_surface_extraction as extraction

    reference = baseline_module("nova/linalg/interpolant.py", "bernstein_reference")
    surface = baseline_module(
        "nova/equilibrium/flux_surface_extraction.py", "surface_reference"
    )
    surface.Bernstein = reference.Bernstein
    saved = {
        name: getattr(interpolant.Bernstein, name)
        for name in ("binom", "basis", "coefficent_matrix")
    }
    matrix = extraction._bernstein_matrix
    arc = extraction._bicubic_arc_moment_correction
    for name in saved:
        setattr(interpolant.Bernstein, name, getattr(reference.Bernstein, name))
    extraction._bernstein_matrix = surface._bernstein_matrix
    extraction._bicubic_arc_moment_correction = surface._bicubic_arc_moment_correction

    def restore():
        for name, value in saved.items():
            setattr(interpolant.Bernstein, name, value)
        extraction._bernstein_matrix = matrix
        extraction._bicubic_arc_moment_correction = arc

    return restore


def persist(name, value):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps(value, allow_nan=False), flush=True)


def configure():
    import jax
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    assert jax.config.jax_enable_x64 and jax.default_backend() == "cpu"


def suite(arm, filename):
    configure()
    import pytest

    if arm == "baseline":
        original_evaluator()
        path = "tests/" + filename
        baseline_module(path, Path(filename).stem)
    return pytest.main([str(ROOT / "tests" / filename), "-q", "-p", "no:cacheprovider"])


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
    result["completed"] = True
    result["below_ceiling"] = len(serialized) < 100_000_000
    persist(f"{arm}-program.json", result)


def row(case, cells):
    with (OUTPUT / f"{case}-{cells}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _row(case, cells)


def _row(case, cells):
    fingerprint = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in (
            "nova/linalg/interpolant.py",
            "nova/equilibrium/flux_surface_extraction.py",
        )
    }
    receipt = OUTPUT / f"{case}-{cells}.json"
    arrays = OUTPUT / f"{case}-{cells}.npz"
    if receipt.exists() and arrays.exists():
        previous = json.loads(receipt.read_text())
        if (
            previous.get("source_sha256") == fingerprint
            and previous.get("job") == os.environ["SLURM_JOB_ID"]
        ):
            print("REUSE_COMPLETED_ROW", str(receipt), flush=True)
            return
    configure()
    import jax
    import numpy as np
    from benchmarks.exact_clip_moment_floor import _build
    from nova.equilibrium.clip_quadrature import clipped_support_current_moments

    def snapshot():
        operator, support, field, capacity, _span = _build(case, -int(cells))
        moments = jax.jit(
            lambda s, f: clipped_support_current_moments(
                s,
                s.included,
                f,
                operator.source.core,
                cut_cell_capacity=capacity,
                boundary_reduction=True,
            )
        )(support, field)
        jax.block_until_ready(moments)
        vertices = np.asarray(support.support_vertices)
        live = np.asarray(support.included) & np.asarray(support.boundary)
        counts = np.asarray(support.vertex_count)
        assert np.any(live) and np.all(counts[live] >= 129)
        return {
            "vertices": vertices,
            "crossings": vertices[live][:, (0, 128)],
            "current": np.asarray(moments[0]),
            "radial_moment": np.asarray(moments[1]),
            "vertical_moment": np.asarray(moments[2]),
            "area": np.asarray(support.area),
            "first_area_moment": np.asarray(support.first_area_moment),
            "second_area_moment": np.asarray(support.second_area_moment),
            "counts": counts,
            "live": live,
            "refused": np.asarray(support.refused_cell_count),
        }

    restore = original_evaluator()
    before = snapshot()
    restore()
    jax.clear_caches()
    after = snapshot()
    result = {
        "source_sha256": fingerprint,
        "case": case,
        "requested_cells": int(cells),
        "revision": revision(),
        "base_revision": BASE,
        "job": os.environ["SLURM_JOB_ID"],
        "comparisons": {},
    }
    for name in before:
        a, b = before[name], after[name]
        assert a.shape == b.shape
        if a.dtype.kind in "biu":
            relative = 0.0 if np.array_equal(a, b) else 1.0
        else:
            assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
            assert np.linalg.norm(a) > 0
            relative = float(np.linalg.norm(b - a) / np.linalg.norm(a))
        result["comparisons"][name] = {
            "array_equal": bool(np.array_equal(a, b)),
            "relative_l2": relative,
            "max_absolute": float(np.max(np.abs(b.astype(float) - a.astype(float)))),
        }
    result["passed"] = all(
        item["relative_l2"] <= 1e-14 for item in result["comparisons"].values()
    )
    np.savez(
        OUTPUT / f"{case}-{cells}.npz",
        **{f"before_{k}": v for k, v in before.items()},
        **{f"after_{k}": v for k, v in after.items()},
    )
    persist(f"{case}-{cells}.json", result)


def child(arguments, name, cpus, *, wait=True):
    command = [sys.executable, str(Path(__file__).resolve()), *arguments]
    log = (OUTPUT / f"{name}.log").open("w")
    if name.startswith("negative"):
        log.write(MUTATION + "\n")
    log.write(f"revision={revision()} tree={ROOT} command={command!r}\n")
    log.flush()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        preexec_fn=lambda: os.sched_setaffinity(0, cpus),
    )
    log.close()
    if not wait:
        return process
    code = process.wait()
    (OUTPUT / f"{name}.exit").write_text(str(code) + "\n")
    print(name, "exit", code, flush=True)
    return code


def summarize():
    programs = {
        arm: json.loads((OUTPUT / f"{arm}-program.json").read_text())
        for arm in ("baseline", "candidate", "negative")
    }
    rows = [
        json.loads(path.read_text())
        for path in sorted(OUTPUT.glob("*-*.json"))
        if path.stem.endswith(("-110", "-300"))
    ]
    suites = {
        arm: {
            name: int((OUTPUT / f"{arm}-{Path(name).stem}.exit").read_text())
            for name in TESTS
        }
        for arm in ("baseline", "candidate")
    }
    baseline, candidate, negative = (
        programs[arm] for arm in ("baseline", "candidate", "negative")
    )
    gates = {
        "programs_completed": all(p["completed"] for p in programs.values()),
        "below_byte_ceiling": candidate["serialized_bytes"] < 100_000_000,
        "baseline_marker_present": baseline["bernstein_binom_present"],
        "candidate_marker_absent": not candidate["bernstein_binom_present"],
        "candidate_static_basis_present": candidate["static_basis_present"],
        "negative_marker_returns": negative["bernstein_binom_present"],
        "negative_bytes_rise": negative["serialized_bytes"]
        > candidate["serialized_bytes"],
        "eight_rows_within_tolerance": len(rows) == 8
        and all(r["passed"] for r in rows),
        "both_suites_pass": all(
            code == 0 for suite in suites.values() for code in suite.values()
        ),
    }
    result = {
        "programs": programs,
        "rows": rows,
        "suite_exit_status": suites,
        "gates": gates,
        "passed": all(gates.values()),
        "serialized_byte_reduction_fraction": 1
        - candidate["serialized_bytes"] / baseline["serialized_bytes"],
    }
    persist("summary.json", result)
    return 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", choices=("baseline", "candidate", "negative"))
    parser.add_argument("--suite", nargs=2)
    parser.add_argument("--suite-set", choices=("baseline", "candidate"))
    parser.add_argument("--row", nargs=2)
    parser.add_argument("--negative-tests", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--rows", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.rows:
        cpus = sorted(os.sched_getaffinity(0))
        for case in (
            "weak-rotation-reactor-static",
            "moderate-rotation-conventional-static",
            "strong-rotation-compact-static",
            "diverted-single-null",
        ):
            for cells in (110, 300):
                child(["--row", case, str(cells)], f"{case}-{cells}-measurement", cpus)
        return 0
    if args.summarize:
        return summarize()
    if args.negative_tests:
        print(MUTATION, flush=True)
        configure()
        original_evaluator()
        import pytest

        return pytest.main(
            [
                str(ROOT / "tests/test_flux_surface_extraction.py")
                + "::test_bicubic_derivatives_lower_to_static_polynomials",
                "-q",
                "-p",
                "no:cacheprovider",
            ]
        )
    if args.program:
        program(args.program)
        return 0
    if args.suite:
        return suite(*args.suite)
    if args.row:
        row(*args.row)
        return 0
    cpus = sorted(os.sched_getaffinity(0))
    if args.suite_set:
        return max(
            child(
                ["--suite", args.suite_set, name],
                f"{args.suite_set}-{Path(name).stem}",
                cpus,
            )
            for name in TESTS
        )
    assert len(cpus) >= 8 and os.environ.get("SLURM_JOB_ID")
    test_cpus, compile_cpus = cpus[:4], cpus[4:8]
    tests = child(["--suite-set", "baseline"], "baseline-suite", test_cpus, wait=False)
    if child(["--program", "baseline"], "baseline-program", compile_cpus):
        tests.wait()
        return 1
    deadline = time.monotonic() + 600
    while not (OUTPUT / "candidate-ready.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("candidate not ready")
        time.sleep(5)
    (OUTPUT / "baseline-suite.exit").write_text(str(tests.wait()) + "\n")
    tests = child(
        ["--suite-set", "candidate"], "candidate-suite", test_cpus, wait=False
    )
    child(["--program", "candidate"], "candidate-program", compile_cpus)
    child(["--program", "negative"], "negative-program", compile_cpus)
    for case in (
        "weak-rotation-reactor-static",
        "moderate-rotation-conventional-static",
        "strong-rotation-compact-static",
        "diverted-single-null",
    ):
        for cells in (110, 300):
            child(["--row", case, str(cells)], f"{case}-{cells}", compile_cpus)
    (OUTPUT / "candidate-suite.exit").write_text(str(tests.wait()) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
