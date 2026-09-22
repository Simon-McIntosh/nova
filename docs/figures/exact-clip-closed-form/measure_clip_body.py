"""Compare fixed-capacity clip bodies in one CPU allocation."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types
import measure_reference as census
import measure_bernstein as prior

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "clip-body"
BASE = "abeccb9e4c29a94594ec3e2238f5cf5df8a7448f"
MUTATION = "restore the unrolled per-cell body"
TESTS = (
    "test_equilibrium_separatrix_clip.py",
    "test_exact_clip_moments.py",
    "test_null_generator_admission.py",
    "test_flux_surface_extraction.py",
)
SOURCE_PATHS = ("nova/equilibrium/separatrix_clip.py",)
census.OUTPUT = prior.OUTPUT = OUTPUT
census.BASE = BASE
census.SOURCE = BASE


def baseline():
    """Replace source in memory, retaining the candidate files on disk."""
    import nova.equilibrium.separatrix_clip as clip
    import nova.equilibrium.forward_operator as forward

    text = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", BASE + ":nova/equilibrium/separatrix_clip.py"],
        text=True,
    )
    reference = types.ModuleType("clip_body_reference")
    reference.__file__ = clip.__file__
    sys.modules[reference.__name__] = reference
    exec(compile(text, clip.__file__, "exec"), reference.__dict__)
    reference.TracedCapacityRefusalError = clip.TracedCapacityRefusalError
    reference.TracedClippedSupports = clip.TracedClippedSupports
    reference.SaddleCellWedges = clip.SaddleCellWedges
    saved = dict(clip.__dict__)
    functions = {
        key: value
        for key, value in reference.__dict__.items()
        if isinstance(value, types.FunctionType)
        and value.__module__ == reference.__name__
    }
    for key, value in functions.items():
        setattr(clip, key, value)
    mesh = clip.AtomicCellMesh
    methods = {
        key: getattr(mesh, key) for key in ("traced_clip", "traced_saddle_wedges")
    }
    for key in methods:
        setattr(mesh, key, getattr(reference.AtomicCellMesh, key))
    forward_function = forward._traced_clip
    forward._traced_clip = reference._traced_clip

    def restore():
        for key in functions:
            setattr(clip, key, saved[key])
        for key, value in methods.items():
            setattr(mesh, key, value)
        forward._traced_clip = forward_function

    return restore


census.original_evaluator = baseline


def run_child(args, name, cpus):
    command = [sys.executable, str(Path(__file__).resolve()), *args]
    with (OUTPUT / (name + ".log")).open("w") as log:
        if name.startswith("negative"):
            log.write(MUTATION + "\n")
        log.write(
            f"revision={prior.revision()} base={BASE} tree={ROOT} command={command!r}\n"
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
    (OUTPUT / (name + ".exit")).write_text(str(code) + "\n")
    print(name, "EXIT", code, flush=True)
    return code


def row(case, cells):
    import jax
    import numpy as np
    from benchmarks.exact_clip_moment_floor import _build
    from nova.equilibrium.clip_quadrature import clipped_support_current_moments

    def snapshot():
        import nova.equilibrium.forward_operator as forward
        from benchmarks import solovev_certificate as certificate

        captured = {}
        clip = forward._traced_clip

        def record_clip(*args, **kwargs):
            captured.update(args=args, kwargs=kwargs)
            return clip(*args, **kwargs)

        forward._traced_clip = record_clip
        try:
            operator, support, field, capacity, span = _build(case, -int(cells))
        finally:
            forward._traced_clip = clip
        values = jax.jit(
            lambda s, f: clipped_support_current_moments(
                s,
                s.included,
                f,
                operator.source.core,
                cut_cell_capacity=capacity,
                boundary_reduction=True,
            )
        )(support, field)
        jax.block_until_ready(values)
        result = {
            name: np.asarray(value) for name, value in zip(support._fields, support)
        }
        result.update(
            {
                name: np.asarray(value)
                for name, value in zip(("current", "radial", "vertical"), values)
            }
        )
        mesh = operator.moment_geometry.atomic_mesh
        live = np.asarray(support.included) & np.asarray(support.boundary)
        assert np.any(live)
        atomic = np.asarray(mesh.node_coordinates)[np.asarray(mesh.cell_nodes)]
        mask = prior._edge_crossing_mask(
            result["support_vertices"],
            result["vertex_count"],
            live,
            atomic,
            np.asarray(mesh.cell_vertex_count),
        )
        assert np.any(mask)
        result["crossing_mask"] = mask
        result["crossings"] = np.where(mask[..., None], result["support_vertices"], 0.0)
        args, kwargs = captured["args"], captured["kwargs"]
        exact = certificate._case(case)[2]
        core = np.asarray(exact.magnetic_axis)
        saddle = np.asarray(kwargs["saddle_vertex"])
        # Closed rows have no saddle; a finite dummy locates no four-root cell.
        saddle = np.where(np.isfinite(saddle), saddle, core)
        wedges = mesh.traced_saddle_wedges(
            args[5],
            saddle_vertex=saddle,
            core_reference=core,
            participating_cell=kwargs["participating_cell"],
            curve_evaluator=kwargs["curve_evaluator"],
            arc_tracer=kwargs["arc_tracer"],
        )
        jax.block_until_ready(wedges)
        result.update(
            {
                "wedge_" + name: np.asarray(value)
                for name, value in zip(wedges._fields, wedges)
            }
        )
        print(
            "SADDLE_WEDGE_POSITIVE_COUNT",
            int(np.asarray(wedges.saddle).sum()),
            flush=True,
        )
        return result

    restore = baseline()
    before = snapshot()
    restore()
    jax.clear_caches()
    after = snapshot()
    comparison = {}
    for name, a in before.items():
        b = after[name]
        assert (
            a.shape == b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b))
        ), name
        equal = np.array_equal(a, b)
        norm = float(np.linalg.norm(a.astype(float)))
        error = float(np.linalg.norm(b.astype(float) - a.astype(float)))
        relative = 0.0 if equal else (error / norm if norm else error)
        comparison[name] = dict(
            array_equal=bool(equal),
            relative_l2=relative,
            baseline_norm=norm,
            max_absolute=float(np.max(np.abs(b.astype(float) - a.astype(float)))),
        )
    name = f"{case}-{cells}"
    np.savez(
        OUTPUT / (name + ".npz"),
        **{"before_" + k: v for k, v in before.items()},
        **{"after_" + k: v for k, v in after.items()},
    )
    prior.persist(
        name + ".json",
        dict(
            case=case,
            cells=int(cells),
            job=os.environ["SLURM_JOB_ID"],
            revision=prior.revision(),
            base=BASE,
            comparisons=comparison,
            passed=all(v["relative_l2"] <= 1e-14 for v in comparison.values()),
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program")
    parser.add_argument("--guard")
    parser.add_argument("--suite", nargs=2)
    parser.add_argument("--suites")
    parser.add_argument("--rows", action="store_true")
    parser.add_argument("--row", nargs=2)
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    if args.guard == "negative":
        print(MUTATION, flush=True)
    print(f"revision={prior.revision()} tree={ROOT} command={sys.argv!r}", flush=True)
    if args.guard:
        prior.configure()
        if args.guard == "negative":
            baseline()
        import pytest

        return pytest.main(
            [
                str(ROOT / "tests/test_equilibrium_separatrix_clip.py"),
                "-q",
                "-p",
                "no:cacheprovider",
                "-k",
                "polish_body",
            ]
        )
    if args.program:
        census.SOURCE = prior.revision()
        census.program(args.program)
        path = OUTPUT / (args.program + "-program.json")
        receipt = json.loads(path.read_text())
        receipt["source_sha256"] = {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (*SOURCE_PATHS, "nova/equilibrium/forward_operator.py")
        }
        receipt["shared_cell_body_present"] = any(
            ":_clip_cell" in name for name, _ in receipt["function_ownership"]
        )
        prior.persist(path.name, receipt)
        return 0
    if args.suite:
        prior.configure()
        arm, filename = args.suite
        if arm == "baseline":
            baseline()
        import pytest

        return pytest.main(
            [str(ROOT / "tests" / filename), "-q", "-p", "no:cacheprovider"]
        )
    cpus = sorted(os.sched_getaffinity(0))
    if args.suites:
        return max(
            finish(
                run_child(
                    ["--suite", args.suites, name],
                    args.suites + "-" + Path(name).stem,
                    cpus,
                ),
                args.suites + "-" + Path(name).stem,
            )
            for name in TESTS
        )
    if args.row:
        prior.configure()
        row(*args.row)
        return 0
    if args.rows:
        code = max(
            finish(
                run_child(["--row", case, str(cells)], f"{case}-{cells}", cpus),
                f"{case}-{cells}",
            )
            for case in (
                "weak-rotation-reactor-static",
                "moderate-rotation-conventional-static",
                "strong-rotation-compact-static",
                "diverted-single-null",
            )
            for cells in (110, 300)
        )
        if code == 0:
            from render_clip_body import render

            render()
        return code
    assert len(cpus) >= 16
    jobs = [
        (
            run_child(["--program", "baseline"], "baseline-program", cpus[:4]),
            "baseline-program",
        ),
        (
            run_child(["--suites", "baseline"], "baseline-suites", cpus[4:8]),
            "baseline-suites",
        ),
    ]
    codes = {name: finish(p, name) for p, name in jobs}
    prior.persist("baseline-complete.json", codes)
    deadline = time.monotonic() + 2400
    while not (OUTPUT / "candidate-ready").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("candidate-ready not supplied within allocation budget")
        time.sleep(2)
    jobs = [
        (
            run_child(["--program", arm], arm + "-program", cpus[index : index + 4]),
            arm + "-program",
        )
        for arm, index in (("candidate", 0), ("negative", 4))
    ]
    jobs += [
        (
            run_child(["--suites", "candidate"], "candidate-suites", cpus[8:12]),
            "candidate-suites",
        ),
        (run_child(["--rows"], "rows", cpus[12:16]), "rows"),
    ]
    codes.update({name: finish(p, name) for p, name in jobs})
    prior.persist("allocation-result.json", codes)
    return max(codes.values())


if __name__ == "__main__":
    raise SystemExit(main())
