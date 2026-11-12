"""Require the pre-symmetrisation asymmetry test to reject a non-reciprocal pair."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

ANCHOR = """                mutual[rows[chosen], columns[chosen]] = value
                mutual[columns[chosen], rows[chosen]] = value
"""

PERTURBATION = """                mutual[rows[chosen], columns[chosen]] = value
                mutual[columns[chosen], rows[chosen]] = value
                if rows[chosen[0]] != columns[chosen[0]]:
                    # one edge-pair quadrature, mirrored only, so reciprocity breaks
                    mutual[columns[chosen[0]], rows[chosen[0]]] = 1.001 * value[0]
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True
    ).strip()
    target_rel = "nova/biot/tiledassembly.py"
    original = subprocess.check_output(
        ["git", "-C", str(args.repo), "show", revision + ":" + target_rel]
    )
    scratch = Path(
        tempfile.mkdtemp(prefix="asymmetry-mutant-", dir=os.environ["TMPDIR"])
    )
    python = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
    try:
        archive = subprocess.Popen(
            [
                "git",
                "-C",
                str(args.repo),
                "archive",
                revision,
                "nova",
                "tests",
                "scripts",
                "pyproject.toml",
                "conftest.py",
            ],
            stdout=subprocess.PIPE,
        )
        with tarfile.open(fileobj=archive.stdout, mode="r|") as tar:
            tar.extractall(scratch, filter="data")
        assert archive.wait() == 0
        target = scratch / target_rel
        source = target.read_text()
        assert source.count(ANCHOR) == 1, source.count(ANCHOR)
        mutated = source.replace(ANCHOR, PERTURBATION)
        assert mutated != source
        target.write_text(mutated)
        env = os.environ.copy()
        env.update(
            PYTHONPATH=str(scratch),
            JAX_PLATFORMS="cpu",
            OPENBLAS_NUM_THREADS="4",
            OMP_NUM_THREADS="4",
        )
        command = [
            python,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            "tests/test_magnetisation.py",
            "-k",
            "pre_symmetrisation_energy_asymmetry",
        ]
        with args.log.open("w") as stream:
            stream.write(
                f"# revision: {revision} tree: {scratch} command: {' '.join(command)}\n"
            )
            stream.flush()
            identity = (
                "from pathlib import Path; import inspect; "
                "from nova.biot import tiledassembly as t; "
                + f"assert Path(t.__file__).resolve() == "
                f"Path({str(target)!r}).resolve(); " + 'assert "1.001 * value[0]" '
                "in inspect.getsource(t.galerkin_cell_energy); "
                'print("MUTANT_MODULE="+t.__file__+" CWD="'
                "+str(Path.cwd().resolve()), flush=True)"
            )
            subprocess.run(
                [python, "-c", identity],
                cwd=scratch,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
            result = subprocess.run(
                command, cwd=scratch, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
            stream.write(f"\nEXIT={result.returncode}\n")
        content = args.log.read_text()
        assert result.returncode != 0
        rejected = [
            line
            for line in content.splitlines()
            if line.startswith("FAILED tests/test_magnetisation.py::test_")
            and "pre_symmetrisation_energy_asymmetry" in line
        ]
        assert rejected, content[-2000:]
        print(f"ASYMMETRY_MUTANT_REJECTED_BY={len(rejected)} test ids", flush=True)
        assert (args.repo / target_rel).read_bytes() == original
        print("ASYMMETRY_MUTANT_REJECTED; product source blob matches HEAD", flush=True)
    finally:
        shutil.rmtree(scratch)
        print("Removed own scratch", scratch, flush=True)


if __name__ == "__main__":
    main()
