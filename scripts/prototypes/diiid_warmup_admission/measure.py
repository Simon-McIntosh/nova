"""Compare warm-up admission through the governed-wall operand capture."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

from timing import CompilationTiming

PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
CASES = (
    (179, "limited"),
    (137, "diverted"),
    (146, "limited"),
    (146, "diverted"),
    (35, "mast"),
)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def extract(tree, revision, repository):
    tree.mkdir()
    archive = subprocess.Popen(
        ["git", "-C", str(repository), "archive", revision], stdout=subprocess.PIPE
    )
    unpack = subprocess.run(
        ["tar", "-x", "-C", str(tree)], stdin=archive.stdout, check=False
    )
    archive.stdout.close()
    assert archive.wait() == unpack.returncode == 0


def row(args):
    import jax
    import numpy as np

    from nova.equilibrium import fixed_point
    from nova.equilibrium.flux_surface_connectivity import polish_stationary_points
    from nova.jax.config import configure_dtypes
    from nova.linalg.tensor_spline import fit_tensor_spline

    configure_dtypes()
    assert jax.config.jax_enable_x64
    if os.environ.get("NOVA_LANE_RUNG") in ("titan", "h200"):
        assert jax.default_backend() == "gpu", "GPU rung did not serve a GPU"
    timing = CompilationTiming()
    timing.control()
    timing.install_solve()
    tree = args.tree.resolve()
    assert Path(fixed_point.__file__).resolve().is_relative_to(tree)
    print(
        f"# revision: {args.revision}; tree: {tree}; command: {sys.argv!r}", flush=True
    )
    print(f"# module: {fixed_point.__file__}; cwd: {Path.cwd().resolve()}", flush=True)
    capture = load(
        "operand_capture", tree / "scripts/prototypes/diiid_residual_plateau/measure.py"
    )
    material = load(
        "material_measure", tree / "scripts/prototypes/diiid_material_mask/measure.py"
    )
    prepare = material.probe._prepare_frame
    capture.measure._prepare_frame = lambda frame, expansion: prepare(frame, None)
    capture.subprocess = SimpleNamespace(check_output=lambda *a, **k: args.revision)
    history = []
    instances = []
    original_capture = capture.Capture

    class HistoryCapture(original_capture):
        def __init__(self, output):
            super().__init__(output)
            instances.append(self)
            self.last_warmup = 0

        def install(self):
            super().install()
            inner = fixed_point._newton_krylov_inner

            def measured(*a, **kw):
                result = inner(*a, **kw)
                terminal = result[0] if not hasattr(result, "state") else result
                if hasattr(terminal, "warmup_refused_proposals"):
                    jax.debug.callback(
                        self.save_history,
                        terminal.warmup_refused_proposals,
                        terminal.warmup_first_nonfinite_index,
                        ordered=True,
                    )
                return result

            fixed_point._newton_krylov_inner = measured

        def save_inner(self, *values):
            super().save_inner(*values)
            if not hasattr(fixed_point.FixedPointResult, "warmup_refused_proposals"):
                stages = [r for r in self.stages if r["stage"] == "warmup"][
                    self.last_warmup :
                ]
                self.last_warmup += len(stages)
                bad = [i for i, r in enumerate(stages) if not r["finite_image"]]
                history.append(
                    dict(
                        refused=0,
                        nonfinite_images=len(bad),
                        first_nonfinite_index=bad[0] if bad else -1,
                    )
                )

        def save_history(self, refused, first):
            history.append(
                dict(
                    refused=int(np.sum(refused)),
                    first_nonfinite_index=int(np.asarray(first)[0]),
                )
            )

    capture.Capture = HistoryCapture
    capture.run(
        args.frame,
        "diverted" if args.branch == "mast" else args.branch,
        args.output,
        "mast" if args.branch == "mast" else "diiid",
    )
    path = args.output / "receipt.json"
    record = json.loads(path.read_text())
    with np.load(args.output / "terminal.npz") as saved:
        assert record["storage_order"] == "radius_height"
        assert saved["flux"].shape == (saved["radius"].size, saved["height"].size)
        import jax.numpy as jnp

        field = fit_tensor_spline(
            jnp.asarray(saved["radius"]),
            jnp.asarray(saved["height"]),
            jnp.asarray(saved["label"].T),
        )
        starts = np.vstack((saved["reference_axis"], saved["reference_saddle"]))
        roots = polish_stationary_points(field, jnp.asarray(starts), jnp.ones(2, bool))
        assert np.asarray(roots["converged"]).all(), "EFIT null polishing refused"
        reference_saddle = np.asarray(roots["position_rz"])[1]
        record.update(
            axis_rz_m=saved["axis"][:2].tolist(),
            x_point_rz_m=saved["saddle"][:2].tolist(),
            efit_x_point_rz_m=reference_saddle.tolist(),
            x_point_distance_m=float(
                np.linalg.norm(saved["saddle"][:2] - reference_saddle)
            ),
            warmup=history,
            warmup_refused_proposals=sum(r["refused"] for r in history),
            module=fixed_point.__file__,
            cwd=str(Path.cwd().resolve()),
            timing=timing.solves,
            backend_compile_programs=timing.rows,
        )
    instance = instances[0]
    assert instance.counts["warmup"] > 0 and instance.counts["inner"] > 0
    # The actual material mask is a positive control for the governed-wall input.
    record["material_cells"] = int(
        np.asarray(instance.profile.operator.inside_material).sum()
    )
    if args.branch != "mast":
        assert record["material_cells"] == 527
    path.write_text(json.dumps(record, indent=2, allow_nan=True) + "\n")
    print("ADMISSION_SUMMARY " + json.dumps(record), flush=True)


def batch(args):
    repository = Path(__file__).resolve().parents[3]
    head = (
        args.head_revision
        or subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="warmup-admission-", dir=os.environ["TMPDIR"]
    ) as temporary:
        base = Path(temporary) / "base"
        extract(base, args.base, repository)
        head_tree = Path(temporary) / "head"
        extract(head_tree, head, repository)
        records = []
        for arm, tree, revision in (
            ("base", base, args.base),
            ("head", head_tree, head),
        ):
            for frame, branch in CASES:
                name = (
                    "mast-21978-35" if branch == "mast" else f"frame-{frame}-{branch}"
                )
                output = args.output / arm / name
                previous = args.resume_from / arm / name if args.resume_from else None
                if previous is not None and (previous / "receipt.json").is_file():
                    receipt = json.loads((previous / "receipt.json").read_text())
                    assert receipt["base_revision"] == revision
                    assert (
                        (previous / "row.log").read_text().rstrip().endswith("EXIT=0")
                    )
                    assert (previous / "terminal.npz").is_file()
                    records.append(
                        dict(arm=arm, case=name, source_path=str(previous), **receipt)
                    )
                    (args.output / "comparison.json").write_text(
                        json.dumps(records, indent=2) + "\n"
                    )
                    print(f"RETAINED {arm} {name} SOURCE={previous}", flush=True)
                    continue
                output.mkdir(parents=True)
                command = [
                    PYTHON,
                    str(Path(__file__).resolve()),
                    "--row",
                    "--tree",
                    str(tree),
                    "--revision",
                    revision,
                    "--frame",
                    str(frame),
                    "--branch",
                    branch,
                    "--output",
                    str(output),
                ]
                env = dict(os.environ, PYTHONPATH=str(tree))
                log = output / "row.log"
                with log.open("w") as stream:
                    stream.write(
                        f"# revision: {revision}; tree: {tree}; "
                        f"command: {' '.join(command)}\n"
                    )
                    stream.flush()
                    result = subprocess.run(
                        command,
                        cwd=tree,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                    stream.write(f"EXIT={result.returncode}\n")
                print(
                    f"ROW {arm} {name} EXIT={result.returncode} LOG={log}", flush=True
                )
                if result.returncode:
                    raise SystemExit(result.returncode)
                records.append(
                    dict(
                        arm=arm,
                        case=name,
                        source_path=str(output),
                        **json.loads((output / "receipt.json").read_text()),
                    )
                )
                (args.output / "comparison.json").write_text(
                    json.dumps(records, indent=2) + "\n"
                )
        if args.forward_regressions:
            from regression import gate

            arms = {}
            for arm, tree, revision in (
                ("base", base, args.base),
                ("head", head_tree, head),
            ):
                arms[arm] = gate(
                    tree,
                    revision,
                    "tests/test_equilibrium_forward_solve.py",
                    args.output / f"forward-{arm}.log",
                    platforms=os.environ["JAX_PLATFORMS"],
                )
                (args.output / "forward-regressions.json").write_text(
                    json.dumps(arms, indent=2) + "\n"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base")
    parser.add_argument("--head-revision")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--row", action="store_true")
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--frame", type=int)
    parser.add_argument("--branch")
    parser.add_argument("--forward-regressions", action="store_true")
    args = parser.parse_args()
    row(args) if args.row else batch(args)


if __name__ == "__main__":
    main()
