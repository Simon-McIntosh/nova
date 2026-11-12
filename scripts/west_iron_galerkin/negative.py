"""Require the reciprocal-energy test to reject a centroid-sampling extraction."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True
    ).strip()
    original = subprocess.check_output(
        ["git", "-C", str(args.repo), "show", revision + ":nova/biot/magnetisation.py"]
    )
    scratch = Path(
        tempfile.mkdtemp(prefix="centroid-mutant-", dir=os.environ["TMPDIR"])
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
        target = scratch / "nova/biot/magnetisation.py"
        source = target.read_text()
        begin, end = (
            source.index("    if points is None:", source.index("def _interaction")),
            source.index("    # B/M is invariant", source.index("def _interaction")),
        )
        mutated = (
            source[:begin]
            + "    if points is None:\n        points = cells.centroids\n"
            + source[end:]
        )
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
            "cell_average_interaction_is_reciprocal_and_energy_bounded",
        ]
        log = args.run / "negative-control.log"
        with log.open("w") as stream:
            stream.write(
                f"# revision: {revision} tree: {scratch} command: {' '.join(command)}\n"
            )
            stream.flush()
            identity = (
                "from pathlib import Path; import inspect; "
                "from nova.biot import magnetisation as m; "
                + f"assert Path(m.__file__).resolve() == "
                f"Path({str(target)!r}).resolve(); "
                + 'assert "points = cells.centroids" in inspect.getsource(m._interaction); '
                'print("MUTANT_MODULE="+m.__file__+" CWD="'
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
        content = log.read_text()
        assert (
            result.returncode != 0
            and (
                "FAILED tests/test_magnetisation.py::"
                "test_cell_average_interaction_is_reciprocal_and_energy_bounded"
            )
            in content
        )
        assert (args.repo / "nova/biot/magnetisation.py").read_bytes() == original
        print("CENTROID_MUTANT_REJECTED; product source blob matches HEAD", flush=True)
    finally:
        shutil.rmtree(scratch)
        print("Removed own scratch", scratch, flush=True)


if __name__ == "__main__":
    main()
