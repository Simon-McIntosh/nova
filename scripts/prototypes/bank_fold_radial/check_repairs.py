"""Check the figure and harness, including isolated inverses of their repairs."""

import argparse
import io
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCOPE = (
    "scripts/prototypes/bank_fold_radial",
    "docs/evidence/fragments/sub-millisecond-forward-path/bank-fold-radial.html",
    "docs/figures/sub-millisecond-forward-path/bank-fold-radial",
)


def execute(command, log, revision, cwd, expected):
    environment = dict(os.environ, PYTHONPATH=str(ROOT), JAX_PLATFORMS="cpu")
    with log.open("w") as stream:
        stream.write(
            f"# revision={revision} tree={cwd} command={shlex.join(command)}\n"
        )
        stream.flush()
        result = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
        stream.write(f"EXIT={result.returncode}\n")
    assert result.returncode == expected, (
        f"{log}: expected {expected}, got {result.returncode}"
    )
    print(f"CHECK {log.name} EXIT={result.returncode}", flush=True)


def inverse_tree(answer, target):
    target.mkdir()
    archive = subprocess.check_output(
        ["git", "-C", str(ROOT), "archive", answer, *SCOPE]
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
        tree.extractall(target, filter="data")
    patch = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", "--format=", "--binary", answer, "--", *SCOPE]
    )
    result = subprocess.run(
        ["git", "apply", "--reverse"],
        input=patch,
        cwd=target,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", answer + "^"], text=True
    ).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--plot-answer", required=True)
    parser.add_argument("--reuse-answer", required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    receipt = args.run_dir / "first.json"
    py = sys.executable
    with tempfile.TemporaryDirectory(prefix="radial-repair-") as folder:
        scratch = Path(folder)
        plot = [
            py,
            "-m",
            "scripts.prototypes.bank_fold_radial.check_plot",
            "--receipt",
            str(receipt),
            "--output",
            str(scratch / "spectrum.svg"),
            "--png",
            str(args.run_dir / "repair-checked-spectrum.png"),
        ]
        reuse = [
            py,
            "-m",
            "scripts.prototypes.bank_fold_radial.check_reuse",
            "--receipt",
            str(receipt),
        ]
        execute(
            plot + ["--source", str(ROOT / SCOPE[0] / "plot.py")],
            args.run_dir / "repair-plot-check.log",
            revision,
            ROOT,
            0,
        )
        execute(
            reuse + ["--source", str(ROOT / SCOPE[0] / "measure.py")],
            args.run_dir / "repair-reuse-check.log",
            revision,
            ROOT,
            0,
        )
        execute(
            [py, "-m", "scripts.prototypes.bank_fold_radial.check_contract"],
            args.run_dir / "repair-contract-check.log",
            revision,
            ROOT,
            0,
        )
        for answer, name, command, source in (
            (args.plot_answer, "plot", plot, "plot.py"),
            (args.reuse_answer, "reuse", reuse, "measure.py"),
        ):
            reverted = scratch / name
            parent = inverse_tree(answer, reverted)
            candidate = reverted / SCOPE[0] / source
            execute(
                command + ["--source", str(candidate)],
                args.run_dir / f"repair-{name}-inverse.log",
                parent,
                ROOT,
                1,
            )
            print(f"INVERSE answer={answer} source={candidate}", flush=True)
        print(f"CLEANUP isolated scratch trees under {scratch}", flush=True)
    print(
        "REPAIRS plot=pass reuse=pass contract=pass isolated_inverses_refused=2",
        flush=True,
    )


if __name__ == "__main__":
    main()
