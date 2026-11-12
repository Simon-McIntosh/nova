"""Extract a revision into a scratch tree, mutate one source file, require a failure.

Every mutant this directory runs does the same seven things: resolve the
revision, keep the tracked blob of the file it will edit, extract the sources
with ``git archive``, apply one in-place mutation, prove the imported module is
that extraction, run a selected test subset, and prove the product source was
left untouched.  The owner is here so a change to the extraction set or to the
identity guard is made once.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from typing import Callable

ARCHIVE_PATHS = ("nova", "tests", "scripts", "pyproject.toml", "conftest.py")
PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
ENVIRONMENT = {
    "JAX_PLATFORMS": "cpu",
    "OPENBLAS_NUM_THREADS": "4",
    "OMP_NUM_THREADS": "4",
}


def revision(repo: Path) -> str:
    """Return the revision resolved in ``repo``."""
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def tracked_blob(repo: Path, revision: str, relative: str) -> bytes:
    """Return the tracked bytes of ``relative`` at ``revision``."""
    return subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{revision}:{relative}"]
    )


def extract(repo: Path, revision: str, *, prefix: str) -> Path:
    """Extract the sources needed to import nova and run its tests."""
    scratch = Path(tempfile.mkdtemp(prefix=prefix, dir=os.environ["TMPDIR"]))
    archive = subprocess.Popen(
        ["git", "-C", str(repo), "archive", revision, *ARCHIVE_PATHS],
        stdout=subprocess.PIPE,
    )
    try:
        with tarfile.open(fileobj=archive.stdout, mode="r|") as tar:
            tar.extractall(scratch, filter="data")
        assert archive.wait() == 0
    except BaseException:
        archive.kill()
        archive.wait()
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    return scratch


def identity_check(
    *, scratch: Path, module: str, function: str, marker: str, target: Path
) -> str:
    """Return a program that fails unless the mutation is what gets imported."""
    return (
        "from pathlib import Path; import inspect, sys; "
        f"import {module} as m; "
        f"assert Path(m.__file__).resolve() == Path({str(target)!r}).resolve(); "
        f"assert {marker!r} in inspect.getsource(m.{function}); "
        'print("MUTANT_MODULE="+m.__file__+" CWD="'
        "+str(Path.cwd().resolve()), flush=True)"
    )


def require_rejection(
    *,
    repo: Path,
    prefix: str,
    relative: str,
    mutate: Callable[[str], str],
    module: str,
    function: str,
    marker: str,
    selection: str,
    log: Path,
) -> int:
    """Run a mutant and return how many selected test ids it turned red.

    ``mutate`` receives the source of ``relative`` and must return a source
    that differs.  ``marker`` is a fragment the mutation introduces, asserted
    present in the imported function so a stale import cannot pass for the
    mutant.  The log ends with a bare ``EXIT=<code>`` line and the product
    file is proved byte-identical to its tracked blob afterwards.
    """
    head = revision(repo)
    original = tracked_blob(repo, head, relative)
    scratch = extract(repo, head, prefix=prefix)
    try:
        target = scratch / relative
        source = target.read_text()
        mutated = mutate(source)
        assert mutated != source
        target.write_text(mutated)
        env = os.environ | {"PYTHONPATH": str(scratch), **ENVIRONMENT}
        command = [
            PYTHON,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            "tests/test_magnetisation.py",
            "-k",
            selection,
        ]
        with log.open("w") as stream:
            stream.write(
                f"# revision: {head} tree: {scratch} command: {' '.join(command)}\n"
            )
            stream.flush()
            subprocess.run(
                [
                    PYTHON,
                    "-c",
                    identity_check(
                        scratch=scratch,
                        module=module,
                        function=function,
                        marker=marker,
                        target=target,
                    ),
                ],
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
        assert result.returncode != 0
        rejected = [
            line
            for line in content.splitlines()
            if line.startswith("FAILED tests/test_magnetisation.py::test_")
            and selection in line
        ]
        assert rejected, content[-2000:]
        assert (repo / relative).read_bytes() == original
        return len(rejected)
    finally:
        shutil.rmtree(scratch)
        print("Removed own scratch", scratch, flush=True)
